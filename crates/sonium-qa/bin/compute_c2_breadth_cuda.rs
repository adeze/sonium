//! Bounded C2 breadth evidence for the resident Complex32 regular-pair ABI.
//!
//! The existing specialised C2 matrix exercises the Complex64 BEM/FMM
//! entries.  This producer covers the production Complex32 regular-pair
//! kernel with the same resident-buffer contract: an odd three-record batch,
//! an independent f64 CPU oracle, checked shape admission, and a one-shot
//! launch failure whose output remains publishable after recovery.

use num_complex::{Complex32, Complex64};
use serde_json::{Value, json};
use sonium_compute::pool::Buffer;
use sonium_compute::{
    Backend, BackendError, BemRegularPairIn, BemRegularPairOut, CudaBackend,
    CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "compute.cuda-c2-breadth.c32-resident-bem.v1";
const ARTIFACT: &str = "c2-c32-resident.json";
const TOLERANCE: f64 = 2.0e-4;

fn transfer_json(stats: &CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "h2d_operations": stats.transfer.h2d_operations,
        "h2d_bytes": stats.transfer.h2d_bytes,
        "d2h_operations": stats.transfer.d2h_operations,
        "d2h_bytes": stats.transfer.d2h_bytes,
        "d2d_operations": stats.transfer.d2d_operations,
        "d2d_bytes": stats.transfer.d2d_bytes,
        "staging_allocation_operations": stats.staging_allocation_operations,
        "live_staging_bytes": stats.live_staging_bytes,
    })
}

fn runtime_json(stats: &CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "allocations": stats.allocation_operations,
        "transfers": transfer_json(stats),
        "live_allocation_bytes": stats.live_allocation_bytes,
        "allocation_high_water_bytes": stats.allocation_high_water_bytes,
    })
}

fn fixture_pairs() -> Vec<BemRegularPairIn> {
    // Three records are intentional: this exercises the non-full final CUDA
    // block while remaining a small, reviewable production pair batch.
    vec![
        BemRegularPairIn::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        BemRegularPairIn::from_triangle(
            [-3.0, 4.0, 5.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
        ),
        BemRegularPairIn::from_triangle(
            [2.5, -4.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.5],
            [1.5, 0.25, 0.5],
            [0.25, 0.75, 1.75],
        ),
    ]
}

/// Independent f64 reference for the four-point regular triangle rule.
/// Geometry is read from the public ABI and evaluated without calling any
/// sonium-compute BEM helper, so parity cannot be self-validation.
fn cpu_reference(
    pair: &BemRegularPairIn,
    wave_number: f32,
    harmonic_factor: f32,
) -> [Complex64; 4] {
    let f = |v: [f32; 3]| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])];
    let src = f(pair.source_point);
    let nx = f(pair.source_normal);
    let v0 = f(pair.v0);
    let v1 = f(pair.v1);
    let v2 = f(pair.v2);
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let raw_n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let jac = raw_n.iter().map(|x| x * x).sum::<f64>().sqrt();
    let inv_jac = if jac > 1.0e-15 { 1.0 / jac } else { 0.0 };
    let ny = [raw_n[0] * inv_jac, raw_n[1] * inv_jac, raw_n[2] * inv_jac];
    let xis = [0.333333333333333, 0.6, 0.2, 0.2];
    let etas = [0.333333333333333, 0.2, 0.6, 0.2];
    let weights = [
        -0.281250,
        0.260416666666667,
        0.260416666666667,
        0.260416666666667,
    ];
    let k = f64::from(wave_number);
    let k2 = f64::from(wave_number * wave_number);
    let wavruim = f64::from(harmonic_factor * wave_number);
    let mut result = [Complex64::new(0.0, 0.0); 4];
    for q in 0..4 {
        let n0 = 1.0 - xis[q] - etas[q];
        let crd = [
            n0 * v0[0] + xis[q] * v1[0] + etas[q] * v2[0],
            n0 * v0[1] + xis[q] * v1[1] + etas[q] * v2[1],
            n0 * v0[2] + xis[q] * v1[2] + etas[q] * v2[2],
        ];
        let wga = weights[q] * jac;
        let diff = [crd[0] - src[0], crd[1] - src[1], crd[2] - src[2]];
        let r = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r < 1.0e-15 {
            continue;
        }
        let inv_r = 1.0 / r;
        let ur = [diff[0] * inv_r, diff[1] * inv_r, diff[2] * inv_r];
        let g = Complex64::from_polar(wga / (12.56637061435917295385 * r), wavruim * r);
        let hb = g * Complex64::new(-inv_r, wavruim);
        let r1h = ur[0] * ny[0] + ur[1] * ny[1] + ur[2] * ny[2];
        let r2h = -(ur[0] * nx[0] + ur[1] * nx[1] + ur[2] * nx[2]);
        let h = hb * r1h;
        let ht = hb * r2h;
        let rq = r1h * r2h;
        let nx_ny = nx[0] * ny[0] + nx[1] * ny[1] + nx[2] * ny[2];
        let e_fac = Complex64::new(
            (3.0 / (r * r) - k2) * rq + nx_ny / (r * r),
            -wavruim / r * (3.0 * rq + nx_ny),
        );
        result[0] += g;
        result[1] += h;
        result[2] += ht;
        result[3] += g * e_fac;
    }
    // Keep the local k binding meaningful in the independent formulation;
    // k2 above mirrors the f32 kernel's rounded k*k value.
    let _ = k;
    result
}

fn channels(out: &BemRegularPairOut) -> [Complex32; 4] {
    [out.g, out.h, out.ht, out.e]
}

fn max_relative_error(actual: &[BemRegularPairOut], expected: &[[Complex64; 4]]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .flat_map(|(out, row)| channels(out).into_iter().zip(row))
        .map(|(got, want)| {
            let got = Complex64::new(f64::from(got.re), f64::from(got.im));
            (got - *want).norm() / want.norm().max(1.0)
        })
        .fold(0.0, f64::max)
}

fn exactly_equal(lhs: &[BemRegularPairOut], rhs: &[BemRegularPairOut]) -> bool {
    lhs.iter().zip(rhs).all(|(left, right)| {
        channels(left)
            .into_iter()
            .zip(channels(right))
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits())
    })
}

fn readback(
    backend: &CudaBackend,
    output: &Buffer<BemRegularPairOut, CudaBackend>,
) -> Result<Vec<BemRegularPairOut>, String> {
    let mut host = vec![BemRegularPairOut::default(); output.len()];
    backend
        .try_copy_d2h(output.raw(), &mut host)
        .map_err(|error| format!("resident Complex32 output readback: {error}"))?;
    Ok(host)
}

fn resident_action(
    backend: &CudaBackend,
    pairs: &Buffer<BemRegularPairIn, CudaBackend>,
    output: &mut Buffer<BemRegularPairOut, CudaBackend>,
    wave_number: f32,
    harmonic_factor: f32,
    phase: &'static str,
) -> Result<Value, String> {
    let before = backend.begin_runtime_stats();
    backend
        .try_bem_regular_pair_c32_device(
            pairs.raw(),
            output.raw_mut(),
            wave_number,
            harmonic_factor,
        )
        .map_err(|error| format!("{phase}: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("{phase} synchronization: {error}"))?;
    let stats = backend.runtime_stats_since(before);
    if stats.kernel_launches != 1 {
        return Err(format!(
            "{phase}: expected one kernel launch, got {stats:?}"
        ));
    }
    if stats.allocation_operations != 0
        || stats.transfer.h2d_operations != 0
        || stats.transfer.d2h_operations != 0
        || stats.transfer.d2d_operations != 0
        || stats.staging_allocation_operations != 0
    {
        return Err(format!(
            "{phase}: non-resident runtime counters: {}",
            runtime_json(&stats)
        ));
    }
    Ok(json!({
        "phase": phase,
        "scalar": "Complex32",
        "layout": "contiguous-resident",
        "status": "native",
        "residency": {
            "host_transfers": false,
            "device_staging": false,
            "runtime": runtime_json(&stats),
        },
    }))
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
        .map_err(|error| format!("C2 breadth CUDA pool: {error}"))?;
    let backend = pool.backend();
    let pairs_host = fixture_pairs();
    let expected = pairs_host
        .iter()
        .map(|pair| cpu_reference(pair, 1.7, -0.75))
        .collect::<Vec<_>>();
    let mut pairs = pool
        .try_alloc::<BemRegularPairIn>(pairs_host.len())
        .map_err(|error| format!("C2 breadth pair allocation: {error}"))?;
    backend
        .try_copy_h2d(&pairs_host, pairs.raw_mut())
        .map_err(|error| format!("C2 breadth pair upload: {error}"))?;
    let mut output = pool
        .try_alloc::<BemRegularPairOut>(pairs_host.len())
        .map_err(|error| format!("C2 breadth output allocation: {error}"))?;
    let setup_stats = backend.runtime_stats();
    backend.reset_runtime_stats();

    let initial_action = resident_action(backend, &pairs, &mut output, 1.7, -0.75, "initial")?;
    let initial = readback(backend, &output)?;
    let initial_error = max_relative_error(&initial, &expected);
    if initial_error > TOLERANCE {
        return Err(format!(
            "initial Complex32 CPU parity error {initial_error} > {TOLERANCE}"
        ));
    }
    let warm_action = resident_action(backend, &pairs, &mut output, 1.7, -0.75, "warm")?;
    let warm = readback(backend, &output)?;
    let warm_error = max_relative_error(&warm, &expected);
    if warm_error > TOLERANCE {
        return Err(format!(
            "warm Complex32 CPU parity error {warm_error} > {TOLERANCE}"
        ));
    }

    let mut malformed_output = pool
        .try_alloc::<BemRegularPairOut>(pairs_host.len() - 1)
        .map_err(|error| format!("C2 malformed output allocation: {error}"))?;
    let malformed = backend
        .try_bem_regular_pair_c32_device(pairs.raw(), malformed_output.raw_mut(), 1.7, -0.75)
        .expect_err("shape mismatch must be rejected before launch");
    let malformed_typed = matches!(
        malformed,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c32_device",
            ..
        }
    );
    if !malformed_typed {
        return Err(format!(
            "malformed shape returned an untyped error: {malformed}"
        ));
    }
    let empty_pairs = pool
        .try_alloc::<BemRegularPairIn>(0)
        .map_err(|error| format!("C2 empty pair allocation: {error}"))?;
    let mut empty_output = pool
        .try_alloc::<BemRegularPairOut>(0)
        .map_err(|error| format!("C2 empty output allocation: {error}"))?;
    let empty_noop = backend
        .try_bem_regular_pair_c32_device(empty_pairs.raw(), empty_output.raw_mut(), 1.7, -0.75)
        .is_ok();
    if !empty_noop {
        return Err("empty resident batch was not accepted as a no-op".into());
    }

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let launch_error = backend
        .try_bem_regular_pair_c32_device(pairs.raw(), output.raw_mut(), 1.7, -0.75)
        .expect_err("injected launch failure must be returned");
    let launch_typed = matches!(
        launch_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    if !launch_typed {
        return Err(format!("launch fault was not typed: {launch_error}"));
    }
    backend
        .recover_after_error()
        .map_err(|error| format!("launch-failure recovery: {error}"))?;
    let preserved = readback(backend, &output)?;
    let output_preserved = exactly_equal(&preserved, &warm);
    if !output_preserved {
        return Err("launch failure changed the previously published output".into());
    }
    let retry_action = resident_action(backend, &pairs, &mut output, 1.7, -0.75, "retry")?;
    let retry = readback(backend, &output)?;
    let retry_error = max_relative_error(&retry, &expected);
    if retry_error > TOLERANCE {
        return Err(format!(
            "retry Complex32 CPU parity error {retry_error} > {TOLERANCE}"
        ));
    }

    let final_stats = backend.runtime_stats();
    if final_stats.allocation_high_water_bytes == 0 || final_stats.live_allocation_bytes == 0 {
        return Err("resident evidence did not record a live allocation/high-water ledger".into());
    }
    let artifact_value = json!({
        "schema": "sonium-compute-cuda-c2-breadth-evidence/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "hardware": cuda_hardware_provenance(),
        "backend": {
            "requested": "cuda",
            "actual": "cuda",
            "fallback_status": "none",
            "operation": "bem_regular_pair_c32_device",
            "scalar": "Complex32",
            "layout": "contiguous-resident",
        },
        "workload": {
            "pair_count": pairs_host.len(),
            "tail": "odd-three-record CUDA block tail",
            "wave_number": 1.7,
            "harmonic_factor": -0.75,
            "oracle": "independent f64 four-point regular triangle rule",
        },
        "parity": {
            "initial_max_relative_error": initial_error,
            "warm_max_relative_error": warm_error,
            "retry_max_relative_error": retry_error,
            "tolerance": TOLERANCE,
        },
        "admission": {
            "malformed_shape_typed": malformed_typed,
            "empty_noop": empty_noop,
            "operation": "bem_regular_pair_c32_device",
        },
        "failure_recovery": {
            "typed_launch": launch_typed,
            "output_preserved": output_preserved,
            "retry": true,
            "context_recovery": "reusable resident context",
            "publication": "failed launch publishes no replacement output",
        },
        "residency": {
            "setup": runtime_json(&setup_stats),
            "initial": initial_action,
            "warm": warm_action,
            "retry": retry_action,
            "final": runtime_json(&final_stats),
        },
        "counts": {"executed": 3, "passed": 3, "unsupported": 0},
    });
    let bytes = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|error| format!("serialize C2 breadth artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("independent f64 CPU four-point regular-pair oracle")
        .reference_checksum("analytic/deterministic-fixture:c2-c32-resident-v1")
        .input_checksum("analytic/deterministic-fixture:c2-c32-resident-v1")
        .dimension("odd three-record Complex32 regular BEM pair batch")
        .formulation("resident contiguous CUDA ABI with checked shape admission and retry")
        .tolerance("max relative error <= 2e-4; resident actions have zero transfers/allocations")
        .metric("pair_count", pairs_host.len() as f64)
        .metric("initial_max_relative_error", initial_error)
        .metric("warm_max_relative_error", warm_error)
        .metric("retry_max_relative_error", retry_error)
        .metric("malformed_shape_typed", f64::from(malformed_typed))
        .metric(
            "output_preserved_after_launch_failure",
            f64::from(output_preserved),
        )
        .metric(
            "allocation_high_water_bytes",
            final_stats.allocation_high_water_bytes as f64,
        )
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/compute_c2_breadth_cuda.rs",
            "native SM120 Complex32 resident BEM pair parity and failure evidence",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            final_stats,
            QaExecutionCounts {
                executed: 3,
                passed: 3,
                ..QaExecutionCounts::default()
            },
        ))
        .hardware_provenance(cuda_hardware_provenance()))
}

fn failed(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(backend) {
            Ok(result) => result,
            Err(error) => failed(error, "cuda", QaFailureStage::Execution),
        },
        Err(error) => failed(
            format!("required CUDA backend unavailable: {error}"),
            "unavailable",
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => {
            eprintln!("C2 breadth CUDA QA result serialization failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
