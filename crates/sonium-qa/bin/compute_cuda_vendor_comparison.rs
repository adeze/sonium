//! Measured CUDA-vendor comparison for the production Krylov reduction.
//!
//! The native path is the fused, device-resident multi_dot kernel used by
//! iterative solvers. The matched vendor path evaluates the same real-f64
//! dot products as one cuBLAS GEMV per basis vector. Inputs and outputs stay
//! in device allocations during both timed sections; only the pre-timing
//! correctness gate and post-timing publication read results back.

use serde_json::json;
use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig, ReductionMode};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use std::mem::size_of;
use std::time::Instant;

const CASE_ID: &str = "compute.fem-krylov-multi-dot-cuda-vendor-comparison.v1";
const SCHEMA: &str = "sonium-compute-cuda-vendor-comparison/v1";
const DEFAULT_VECTOR_LEN: usize = 16_384;
const DEFAULT_BASIS_COUNT: usize = 16;
const DEFAULT_REPEATS: usize = 20;
const MAX_VECTOR_LEN: usize = 1 << 20;
const MAX_BASIS_COUNT: usize = 128;
const MAX_REPEATS: usize = 1_000;
const CORRECTNESS_TOLERANCE: f64 = 2.0e-10;
const REAL_DOT_FLOPS_PER_ELEMENT: u64 = 2;

fn setting(name: &str, default: usize, minimum: usize, maximum: usize) -> Result<usize, String> {
    let value = std::env::var(name)
        .ok()
        .map(|raw| {
            raw.parse::<usize>()
                .map_err(|error| format!("{name} must be an integer: {error}"))
        })
        .transpose()?
        .unwrap_or(default);
    if !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "{name}={value} is outside the supported range {minimum}..={maximum}"
        ));
    }
    Ok(value)
}

fn inputs(vector_len: usize, basis_count: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let y: Vec<f64> = (0..vector_len)
        .map(|index| {
            let x = index as f64;
            0.25 + (x * 0.00037).sin() * 0.5 + (x * 0.00113).cos() * 0.125
        })
        .collect();
    let basis: Vec<Vec<f64>> = (0..basis_count)
        .map(|column| {
            let phase = (column + 1) as f64 * 0.017;
            (0..vector_len)
                .map(|index| {
                    let x = index as f64;
                    (x * (0.00019 + column as f64 * 0.000003) + phase).sin() * 0.4
                        + (x * 0.00071 - phase * 0.5).cos() * 0.2
                })
                .collect()
        })
        .collect();
    let expected: Vec<f64> = basis
        .iter()
        .map(|column| column.iter().zip(&y).map(|(lhs, rhs)| lhs * rhs).sum())
        .collect();
    (basis, y, expected)
}

fn maximum_relative_error(actual: &[f64], expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "result length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut maximum = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("reduction result is non-finite".into());
        }
        maximum = maximum.max((actual - expected).abs() / expected.abs().max(1.0e-30));
    }
    Ok(maximum)
}

fn hardware_value(value: Option<String>, field: &str) -> Result<String, String> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("CUDA hardware provenance field {field} is missing"))
}

fn run(backend: CudaBackend) -> Result<serde_json::Value, String> {
    let vector_len = setting(
        "SONIUM_C6_VENDOR_VECTOR_LEN",
        DEFAULT_VECTOR_LEN,
        1_024,
        MAX_VECTOR_LEN,
    )?;
    let basis_count = setting(
        "SONIUM_C6_VENDOR_BASIS_COUNT",
        DEFAULT_BASIS_COUNT,
        2,
        MAX_BASIS_COUNT,
    )?;
    let repeats = setting("SONIUM_C6_VENDOR_REPEATS", DEFAULT_REPEATS, 3, MAX_REPEATS)?;
    let (basis_host, y_host, expected) = inputs(vector_len, basis_count);
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(256 * 1024 * 1024))
        .map_err(|error| format!("construct CUDA memory pool: {error}"))?;
    let backend = pool.backend();

    let mut basis = Vec::with_capacity(basis_count);
    for values in &basis_host {
        let mut device = pool
            .try_alloc::<f64>(vector_len)
            .map_err(|error| format!("allocate basis vector: {error}"))?;
        backend
            .try_copy_h2d(values, device.raw_mut())
            .map_err(|error| format!("upload basis vector: {error}"))?;
        basis.push(device);
    }
    let mut y = pool
        .try_alloc::<f64>(vector_len)
        .map_err(|error| format!("allocate reduction vector: {error}"))?;
    backend
        .try_copy_h2d(&y_host, y.raw_mut())
        .map_err(|error| format!("upload reduction vector: {error}"))?;
    let mut native_output = pool
        .try_alloc::<f64>(basis_count)
        .map_err(|error| format!("allocate fused reduction output: {error}"))?;
    let mut vendor_output = Vec::with_capacity(basis_count);
    for _ in 0..basis_count {
        vendor_output.push(
            pool.try_alloc::<f64>(1)
                .map_err(|error| format!("allocate cuBLAS output: {error}"))?,
        );
    }
    let basis_refs: Vec<_> = basis.iter().map(|vector| vector.raw()).collect();

    // Warm both dispatches before the correctness gate and reset the event
    // stream counters only after all setup/compilation work is complete.
    backend
        .try_multi_dot_with_mode_device_into(
            ReductionMode::Fast,
            &basis_refs,
            y.raw(),
            native_output.raw_mut(),
        )
        .map_err(|error| format!("warm native fused reduction: {error}"))?;
    for (vector, output) in basis.iter().zip(vendor_output.iter_mut()) {
        backend
            .try_gemv(
                sonium_compute::Transpose::NoTrans,
                1,
                vector_len,
                1.0,
                vector.raw(),
                vector_len,
                y.raw(),
                0.0,
                output.raw_mut(),
            )
            .map_err(|error| format!("warm cuBLAS GEMV: {error}"))?;
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("warm dispatch synchronization: {error}"))?;

    let mut native_actual = vec![0.0; basis_count];
    backend
        .try_copy_d2h(native_output.raw(), &mut native_actual)
        .map_err(|error| format!("native correctness publication: {error}"))?;
    let native_error = maximum_relative_error(&native_actual, &expected)?;
    let mut vendor_actual = vec![0.0; basis_count];
    for (actual, output) in vendor_actual.iter_mut().zip(&vendor_output) {
        backend
            .try_copy_d2h(output.raw(), std::slice::from_mut(actual))
            .map_err(|error| format!("cuBLAS correctness publication: {error}"))?;
    }
    let vendor_error = maximum_relative_error(&vendor_actual, &expected)?;
    if native_error > CORRECTNESS_TOLERANCE || vendor_error > CORRECTNESS_TOLERANCE {
        return Err(format!(
            "correctness gate failed before timing: native={native_error:.3e}, vendor={vendor_error:.3e}, tolerance={CORRECTNESS_TOLERANCE:.3e}"
        ));
    }

    backend.reset_runtime_stats();
    let native_baseline = backend.begin_runtime_stats();
    let native_start = Instant::now();
    for _ in 0..repeats {
        backend
            .try_multi_dot_with_mode_device_into(
                ReductionMode::Fast,
                &basis_refs,
                y.raw(),
                native_output.raw_mut(),
            )
            .map_err(|error| format!("timed native fused reduction: {error}"))?;
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("native timing synchronization: {error}"))?;
    let native_elapsed_ms = native_start.elapsed().as_secs_f64() * 1_000.0;
    let native_stats = backend.runtime_stats_since(native_baseline);

    backend.reset_runtime_stats();
    let vendor_baseline = backend.begin_runtime_stats();
    let vendor_start = Instant::now();
    for _ in 0..repeats {
        for (vector, output) in basis.iter().zip(vendor_output.iter_mut()) {
            backend
                .try_gemv(
                    sonium_compute::Transpose::NoTrans,
                    1,
                    vector_len,
                    1.0,
                    vector.raw(),
                    vector_len,
                    y.raw(),
                    0.0,
                    output.raw_mut(),
                )
                .map_err(|error| format!("timed cuBLAS GEMV: {error}"))?;
        }
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("cuBLAS timing synchronization: {error}"))?;
    let vendor_elapsed_ms = vendor_start.elapsed().as_secs_f64() * 1_000.0;
    let vendor_stats = backend.runtime_stats_since(vendor_baseline);

    // Publish once after both timings. This is outside both timed regions and
    // is retained separately so the checker can reject hidden action staging.
    backend
        .try_copy_d2h(native_output.raw(), &mut native_actual)
        .map_err(|error| format!("native result publication: {error}"))?;
    for (actual, output) in vendor_actual.iter_mut().zip(&vendor_output) {
        backend
            .try_copy_d2h(output.raw(), std::slice::from_mut(actual))
            .map_err(|error| format!("vendor result publication: {error}"))?;
    }
    let final_native_error = maximum_relative_error(&native_actual, &expected)?;
    let final_vendor_error = maximum_relative_error(&vendor_actual, &expected)?;
    if final_native_error > CORRECTNESS_TOLERANCE || final_vendor_error > CORRECTNESS_TOLERANCE {
        return Err("post-timing result publication failed the correctness gate".into());
    }

    let flops_per_repeat = (basis_count as u64)
        .saturating_mul(vector_len as u64)
        .saturating_mul(REAL_DOT_FLOPS_PER_ELEMENT);
    let total_flops = flops_per_repeat.saturating_mul(repeats as u64);
    let bytes_per_repeat = (basis_count as u64)
        .saturating_mul((vector_len as u64).saturating_mul(2).saturating_add(1))
        .saturating_mul(size_of::<f64>() as u64);
    let total_bytes = bytes_per_repeat.saturating_mul(repeats as u64);
    let provenance = cuda_hardware_provenance();
    let peak_flops = hardware_value(provenance.peak_flops.clone(), "peak_flops")?
        .parse::<f64>()
        .map_err(|error| format!("hardware.peak_flops is not numeric: {error}"))?;
    let memory_bandwidth = hardware_value(provenance.memory_bandwidth.clone(), "memory_bandwidth")?
        .parse::<f64>()
        .map_err(|error| format!("hardware.memory_bandwidth is not numeric: {error}"))?;
    let measured = |elapsed_ms: f64| {
        let seconds = elapsed_ms / 1_000.0;
        json!({
            "elapsed_ms": elapsed_ms,
            "throughput_flops_per_second": total_flops as f64 / seconds,
            "effective_memory_bandwidth_bytes_per_second": total_bytes as f64 / seconds,
            "roofline_compute_fraction": total_flops as f64 / seconds / peak_flops,
            "roofline_memory_fraction": total_bytes as f64 / seconds / memory_bandwidth,
        })
    };
    let speedup = vendor_elapsed_ms / native_elapsed_ms;
    let runtime = |stats: sonium_compute::cuda::CudaRuntimeStatsSnapshot| {
        json!({
            "kernel_launches": stats.kernel_launches,
            "library_operations": stats.library_operations,
            "synchronization_calls": stats.synchronization_calls,
            "h2d_operations": stats.transfer.h2d_operations,
            "d2h_operations": stats.transfer.d2h_operations,
            "d2d_operations": stats.transfer.d2d_operations,
            "allocation_operations": stats.allocation_operations,
            "allocation_high_water_bytes": stats.allocation_high_water_bytes,
            "live_allocation_bytes": stats.live_allocation_bytes,
        })
    };
    Ok(json!({
        "schema": SCHEMA,
        "version": 1,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "single NVIDIA SM120; production Krylov reduction; resident action timing",
        "claims": {
            "vendor_comparison": "measured_shape_matched_cublas_gemv_vs_native_fused_multi_dot",
            "nsight_counters": "not_claimed",
        },
        "hardware": provenance,
        "workload": {
            "scalar": "f64",
            "operation": "multi_dot",
            "native_dispatch": "sonium fused CUDA multi_dot kernel",
            "vendor_dispatch": "cuBLAS GEMV",
            "vector_len": vector_len,
            "basis_count": basis_count,
            "repeats": repeats,
            "shape_match": "same basis_count x vector_len real dot products; vendor uses m=1,n=vector_len GEMV per basis",
            "correctness_before_timing": true,
            "native_relative_error": native_error,
            "vendor_relative_error": vendor_error,
            "tolerance": CORRECTNESS_TOLERANCE,
            "flops_per_repeat": flops_per_repeat,
            "total_flops": total_flops,
            "bytes_per_repeat": bytes_per_repeat,
            "total_bytes": total_bytes,
            "arithmetic_intensity_flops_per_byte": flops_per_repeat as f64 / bytes_per_repeat as f64,
        },
        "measurements": {
            "native": measured(native_elapsed_ms),
            "vendor": measured(vendor_elapsed_ms),
            "vendor_over_native_speedup": speedup,
        },
        "runtime": {
            "native": runtime(native_stats),
            "vendor": runtime(vendor_stats),
            "publication_d2h_operations": basis_count + 1,
        },
        "provenance": {
            "producer": "crates/sonium-qa/bin/compute_cuda_vendor_comparison.rs",
            "artifact": "cuda_vendor_comparison.json",
            "measurement": "wall-clock around resident CUDA submissions plus one explicit synchronization; setup and publication excluded",
            "hardware": provenance,
        },
    }))
}

fn main() {
    let result = required_cuda_backend()
        .map_err(|error| format!("required CUDA backend: {error}"))
        .and_then(run)
        .and_then(|artifact| {
            let bytes = serde_json::to_vec_pretty(&artifact)
                .map_err(|error| format!("serialize comparison artifact: {error}"))?;
            write_qa_artifact("cuda_vendor_comparison.json", &bytes)
                .map(|(_, checksum)| (artifact, checksum))
        });
    match result {
        Ok((artifact, checksum)) => {
            println!(
                "{}",
                serde_json::json!({"status":"passed", "artifact": artifact, "artifact_checksum": checksum})
            );
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: CUDA vendor comparison: {error}");
            std::process::exit(1);
        }
    }
}
