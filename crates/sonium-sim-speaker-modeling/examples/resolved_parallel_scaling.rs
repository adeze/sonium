//! Compare real resolved FEM sweeps across Rayon thread budgets.

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use serde_json::json;
use sonium_sim_speaker_modeling::{ResolvedAcousticConfig, ResolvedAcousticModel};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let cells = std::env::var("BENCH_CELLS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    if repeats == 0 {
        return Err("BENCH_REPEATS must be positive".into());
    }
    let start = Instant::now();
    let model = ResolvedAcousticModel::build(ResolvedAcousticConfig {
        cells_per_physical_interval: cells,
        ..Default::default()
    })?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let frequencies: Vec<_> = (0..12)
        .map(|i| 40.0 * 25.0_f64.powf(i as f64 / 11.0))
        .collect();
    let drive = Complex64::new(1e-3, 0.0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let reference = pool.install(|| model.solve_sweep(&frequencies, drive))?;
    let mut rows = Vec::new();
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        for repeat in 0..repeats {
            let start = Instant::now();
            let result = pool.install(|| {
                model.solve_driven_sweep_parallel(&frequencies, &vec![drive; frequencies.len()])
            })?;
            let seconds = start.elapsed().as_secs_f64();
            let scale = reference
                .iter()
                .map(|r| r.far_field_pressure_rms_pa.norm())
                .fold(0.0_f64, f64::max);
            let error = result
                .iter()
                .zip(&reference)
                .map(|(r, p)| {
                    (r.far_field_pressure_rms_pa - p.far_field_pressure_rms_pa).norm()
                        / p.far_field_pressure_rms_pa
                            .norm()
                            .max(scale * 1e-6)
                            .max(1e-30)
                })
                .fold(0.0_f64, f64::max);
            if !error.is_finite() || error > 1e-6 {
                return Err("resolved FEM sweep parity gate failed".into());
            }
            rows.push(json!({ "threads":threads, "repeat":repeat, "seconds":seconds,
                "max_relative_field_error":error,
                "max_relative_residual":result.iter().map(|p| p.solve_diagnostics.relative_residual).fold(0.0_f64,f64::max),
                "iterations":result.iter().map(|p| p.solve_diagnostics.iterations).collect::<Vec<_>>() }));
        }
    }
    let artifact = json!({"schema_version":1, "kind":"accepted_resolved_fem_sweep_scaling",
        "cells_per_interval":cells, "nodes":model.mesh.num_nodes(), "setup_seconds":setup_seconds,
        "frequencies_hz":frequencies, "rows":rows});
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "resolved-parallel-scaling.json".into());
    std::fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
    println!("Wrote {output}");
    Ok(())
}
