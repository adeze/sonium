//! Measure actual BEM sweeps with accepted-error CPU and GPU comparisons.

// Rust guideline compliant 2026-02-21
use serde_json::json;
use sonium_sim_speaker::mesh_gen::generate_loudspeaker_mesh;
use sonium_sim_speaker::mlg_loader::load_model;
use sonium_sim_speaker::simulation::try_run_simulation_with_config;
use sonium_sim_speaker::{SimulationResult, SolverConfig};
use std::error::Error;
use std::time::Instant;

fn relative_field_error(result: &SimulationResult, reference: &SimulationResult) -> f64 {
    let scale = reference
        .complex_pressure
        .as_slice()
        .iter()
        .map(|p| p.norm())
        .fold(0.0_f64, f64::max);
    result
        .complex_pressure
        .as_slice()
        .iter()
        .zip(reference.complex_pressure.as_slice())
        .map(|(p, r)| (p - r).norm() / r.norm().max(scale * 1e-6).max(1e-30))
        .fold(0.0, f64::max)
}

fn process_peak_rss_kib() -> Option<usize> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let option = |name: &str| {
        args.windows(2)
            .find(|a| a[0] == name)
            .map(|a| a[1].as_str())
    };
    let thread_counts = option("--threads")
        .unwrap_or("1,2,4")
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let count: usize = option("--frequencies").unwrap_or("12").parse()?;
    let maximum: f64 = option("--max-frequency").unwrap_or("800").parse()?;
    let repeats: usize = option("--repeats").unwrap_or("3").parse()?;
    if count < 2
        || repeats == 0
        || thread_counts.is_empty()
        || thread_counts.contains(&0)
        || !maximum.is_finite()
        || maximum < 200.0
    {
        return Err(
            "positive threads/repeats, >=2 frequencies and max-frequency >=200 required".into(),
        );
    }
    let mut model = load_model(include_str!("two_way_speaker.mlg"), "two_way_speaker.mlg")?;
    model.simulation.freq_start = 200.0;
    model.simulation.freq_end = maximum;
    model.simulation.freq_points = count;
    model.simulation.max_mesh_frequency = Some(maximum);
    let mesh_start = Instant::now();
    let mesh = generate_loudspeaker_mesh(&model);
    let mesh_seconds = mesh_start.elapsed().as_secs_f64();
    let front = model.baffle_center()[2];
    let points: Vec<_> = (-9..=9)
        .map(|angle| {
            let angle = (angle as f64 * 10.0).to_radians();
            [angle.sin(), 0.0, front + angle.cos()]
        })
        .collect();
    let reference_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let reference = reference_pool.install(|| {
        try_run_simulation_with_config(
            &model,
            &points,
            &SolverConfig {
                parallel_frequency_sweep: false,
                ..Default::default()
            },
        )
    })?;
    let mut rows = Vec::new();
    #[cfg(feature = "cuda")]
    let mut accepted = true;
    #[cfg(not(feature = "cuda"))]
    let accepted = true;
    for threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        for repeat in 0..repeats {
            let start = Instant::now();
            let result = pool.install(|| {
                try_run_simulation_with_config(
                    &model,
                    &points,
                    &SolverConfig {
                        max_frequency_lanes: Some(threads),
                        ..Default::default()
                    },
                )
            })?;
            let seconds = start.elapsed().as_secs_f64();
            let error = relative_field_error(&result, &reference);
            if !error.is_finite() || error > 1e-6 {
                return Err("CPU sweep parity gate failed".into());
            }
            rows.push(json!({ "backend":"cpu", "threads":threads, "repeat":repeat,
                "solve_and_field_seconds":seconds, "max_relative_field_error":error,
                "process_peak_rss_kib":process_peak_rss_kib(), "provenance":result.provenance }));
        }
    }
    if args.iter().any(|a| a == "--cuda") {
        #[cfg(feature = "cuda")]
        {
            use sonium_sim_speaker::{GpuValidationPolicy, run_simulation_cuda_with_report};
            // Qualification must pass before timing the periodic production policy.
            for policy in [
                GpuValidationPolicy::Qualification,
                GpuValidationPolicy::production(),
            ] {
                let start = Instant::now();
                let outcome = run_simulation_cuda_with_report(
                    &model,
                    &points,
                    &SolverConfig {
                        gpu_validation: policy,
                        ..Default::default()
                    },
                );
                let (result, report) = match outcome {
                    Ok(value) => value,
                    Err(error) => {
                        rows.push(json!({"backend":"cuda", "policy":policy, "accepted":false, "error":error.to_string()}));
                        accepted = false;
                        break;
                    }
                };
                let seconds = start.elapsed().as_secs_f64();
                let error = relative_field_error(&result, &reference);
                let passed = error.is_finite() && error <= 1e-3;
                rows.push(json!({ "backend":"cuda", "policy":policy,
                    "solve_and_field_seconds":seconds, "max_relative_field_error":error,
                    "process_peak_rss_kib":process_peak_rss_kib(), "gpu_report":report,
                    "accepted":passed, "cpu_pressure":reference.complex_pressure.as_slice(),
                    "gpu_pressure":result.complex_pressure.as_slice() }));
                if !passed {
                    accepted = false;
                    break;
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err("--cuda requires --features cuda".into());
    }
    let artifact = json!({ "schema_version":1, "kind":"bem_sweep_scaling", "accepted":accepted,
        "toolchain":std::process::Command::new("rustc").arg("--version").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
        "available_threads":std::thread::available_parallelism().map(usize::from).unwrap_or(1),
        "mesh_triangles":mesh.triangles.len(), "mesh_seconds":mesh_seconds,
        "frequencies_hz":reference.frequencies, "receivers":points.len(),
        "memory_note":"VmHWM is the cumulative process high-water mark, not per-run allocation", "rows":rows });
    let output = option("--output").unwrap_or("speaker-sweep-scaling.json");
    std::fs::write(output, serde_json::to_vec_pretty(&artifact)?)?;
    println!("Wrote {output}");
    if !accepted {
        return Err(
            "GPU qualification failed; failed evidence saved and production timing skipped".into(),
        );
    }
    Ok(())
}
