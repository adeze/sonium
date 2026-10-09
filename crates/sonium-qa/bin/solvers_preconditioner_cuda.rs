//! Native CUDA evidence for backend-resident solver preconditioners.
//!
//! This case is intentionally accelerator-required: a skip-capable unit test
//! is useful on ordinary developer machines, but release evidence must prove
//! that the promoted preconditioner cells actually ran on CUDA.  Every action
//! interval is bracketed by CUDA runtime counters; host/device copies used to
//! seed inputs and collect references are outside the measured interval.

use num_complex::Complex32;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendCsr, ChebyshevBackend, CudaBackend, CudaRuntimeStatsSnapshot,
    DeviceAdjointCapability, DevicePreconditioner, DevicePreconditionerDiagnosticsSnapshot,
    HostCsrRef, PointJacobiBackend, PolynomialJacobiBackend,
};
use sonium_qa::{QaResult, cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use sonium_solvers::preconditioners::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-cuda-native.v1";
const TOLERANCE: f64 = 2.0e-10;

fn tridiag_f64(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in 0..n {
        if row > 0 {
            col_idx.push((row - 1) as i32);
            values.push(-1.0);
        }
        col_idx.push(row as i32);
        values.push(2.0);
        if row + 1 < n {
            col_idx.push((row + 1) as i32);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as i32);
    }
    (row_ptr, col_idx, values)
}

fn max_abs_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn max_abs_c32(lhs: &[Complex32], rhs: &[Complex32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| f64::from((*left - *right).norm()))
        .fold(0.0, f64::max)
}

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

fn diagnostics_json(stats: DevicePreconditionerDiagnosticsSnapshot) -> Value {
    json!({
        "setup_count": stats.setup_count,
        "setup_failures": stats.setup_failures,
        "apply_count": stats.apply_count,
        "apply_failures": stats.apply_failures,
        "transpose_count": stats.transpose_count,
        "transpose_failures": stats.transpose_failures,
        "hermitian_count": stats.hermitian_count,
        "hermitian_failures": stats.hermitian_failures,
        "fallback_count": stats.fallback_count,
        "first_failure_stage": stats.first_failure_stage.map(|stage| format!("{stage:?}")),
    })
}

fn require_native_action(name: &str, stats: &CudaRuntimeStatsSnapshot) -> Result<(), String> {
    if stats.transfer.h2d_operations != 0
        || stats.transfer.d2h_operations != 0
        || stats.staging_allocation_operations != 0
        || stats.live_staging_bytes != 0
    {
        return Err(format!(
            "{name} action used undocumented host staging: {:?}",
            transfer_json(stats)
        ));
    }
    if stats.kernel_launches == 0 && stats.library_operations == 0 {
        return Err(format!("{name} action dispatched no CUDA work"));
    }
    Ok(())
}

fn require_unsupported_before_device_work<P>(
    name: &str,
    cuda: &CudaBackend,
    preconditioner: &P,
    rhs: &sonium_compute::CudaBuffer<f64>,
    output: &mut sonium_compute::CudaBuffer<f64>,
    hermitian: bool,
) -> Result<Value, String>
where
    P: DevicePreconditioner<CudaBackend, f64>,
{
    let before = cuda.runtime_stats();
    let error = if hermitian {
        preconditioner
            .try_apply_hermitian(cuda, rhs, output)
            .expect_err("unsupported Hermitian action unexpectedly succeeded")
    } else {
        preconditioner
            .try_apply_transpose(cuda, rhs, output)
            .expect_err("unsupported transpose action unexpectedly succeeded")
    };
    if !matches!(error, sonium_compute::BackendError::Unsupported { .. }) {
        return Err(format!(
            "{name} returned the wrong unsupported error: {error}"
        ));
    }
    let stats = cuda.runtime_stats_since(before);
    if stats.kernel_launches != 0
        || stats.library_operations != 0
        || stats.synchronization_calls != 0
        || stats.transfer.h2d_operations != 0
        || stats.transfer.d2h_operations != 0
        || stats.staging_allocation_operations != 0
    {
        return Err(format!(
            "{name} unsupported action touched device state: {:?}",
            transfer_json(&stats)
        ));
    }
    Ok(json!({
        "action": if hermitian { "hermitian" } else { "transpose" },
        "status": "unsupported",
        "error": error.to_string(),
        "device_work": false,
    }))
}

fn exercise_f64<Pc, Pg>(
    name: &str,
    cpu: &sonium_compute::CpuBackend,
    cuda: &CudaBackend,
    cpu_preconditioner: &Pc,
    cuda_preconditioner: &Pg,
    rhs_host: &[f64],
    block_rhs_host: &[f64],
) -> Result<Value, String>
where
    Pc: DevicePreconditioner<sonium_compute::CpuBackend, f64>,
    Pg: DevicePreconditioner<CudaBackend, f64>,
{
    let mut rhs_cpu = cpu
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut rhs_cuda = cuda
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(rhs_host, &mut rhs_cuda);
    let mut scalar_cpu = cpu
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut scalar_cuda = cuda
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply(cpu, &rhs_cpu, &mut scalar_cpu)
        .map_err(|e| format!("{name} CPU scalar action: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply(cuda, &rhs_cuda, &mut scalar_cuda)
        .map_err(|e| format!("{name} CUDA scalar action: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA scalar synchronization: {e}"))?;
    let scalar_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/scalar"), &scalar_stats)?;
    let mut scalar_cpu_host = vec![0.0; rhs_host.len()];
    let mut scalar_cuda_host = vec![0.0; rhs_host.len()];
    cpu.copy_d2h(&scalar_cpu, &mut scalar_cpu_host);
    cuda.copy_d2h(&scalar_cuda, &mut scalar_cuda_host);
    let scalar_error = max_abs_f64(&scalar_cpu_host, &scalar_cuda_host);
    if scalar_error > TOLERANCE {
        return Err(format!("{name} scalar parity error {scalar_error:e}"));
    }

    let rhs = 2;
    if block_rhs_host.len() != rhs_host.len() * rhs {
        return Err(format!("{name} block fixture has the wrong length"));
    }
    let mut block_cpu = cpu
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_cuda = cuda
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(block_rhs_host, &mut block_cpu);
    cuda.copy_h2d(block_rhs_host, &mut block_cuda);
    let mut block_out_cpu = cpu
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_out_cuda = cuda
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_colmajor(cpu, rhs, &block_cpu, &mut block_out_cpu)
        .map_err(|e| format!("{name} CPU block action: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_colmajor(cuda, rhs, &block_cuda, &mut block_out_cuda)
        .map_err(|e| format!("{name} CUDA block action: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA block synchronization: {e}"))?;
    let block_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/block"), &block_stats)?;
    let mut block_cpu_host = vec![0.0; block_rhs_host.len()];
    let mut block_cuda_host = vec![0.0; block_rhs_host.len()];
    cpu.copy_d2h(&block_out_cpu, &mut block_cpu_host);
    cuda.copy_d2h(&block_out_cuda, &mut block_cuda_host);
    let block_error = max_abs_f64(&block_cpu_host, &block_cuda_host);
    if block_error > TOLERANCE {
        return Err(format!("{name} block parity error {block_error:e}"));
    }

    Ok(json!({
        "scalar": {"status": "passed", "max_abs_error": scalar_error, "transfers": transfer_json(&scalar_stats)},
        "block_rhs": {"status": "passed", "rhs": rhs, "max_abs_error": block_error, "transfers": transfer_json(&block_stats)},
        "capabilities": {
            "forward": format!("{:?}", cuda_preconditioner.forward_capability()),
            "transpose": format!("{:?}", cuda_preconditioner.transpose_capability()),
            "hermitian": format!("{:?}", cuda_preconditioner.hermitian_capability()),
        },
        "diagnostics": diagnostics_json(cuda_preconditioner.diagnostics()),
    }))
}

fn exercise_c32<Pc, Pg>(
    name: &str,
    cpu: &sonium_compute::CpuBackend,
    cuda: &CudaBackend,
    cpu_preconditioner: &Pc,
    cuda_preconditioner: &Pg,
    rhs_host: &[Complex32],
    block_rhs_host: &[Complex32],
) -> Result<Value, String>
where
    Pc: DevicePreconditioner<sonium_compute::CpuBackend, Complex32>,
    Pg: DevicePreconditioner<CudaBackend, Complex32>,
{
    let mut rhs_cpu = cpu
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut rhs_cuda = cuda
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(rhs_host, &mut rhs_cuda);
    let mut out_cpu = cpu
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut out_cuda = cuda
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply(cpu, &rhs_cpu, &mut out_cpu)
        .map_err(|e| format!("{name} CPU scalar action: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply(cuda, &rhs_cuda, &mut out_cuda)
        .map_err(|e| format!("{name} CUDA scalar action: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA synchronization: {e}"))?;
    let scalar_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/complex32"), &scalar_stats)?;
    let mut cpu_values = vec![Complex32::default(); rhs_host.len()];
    let mut cuda_values = vec![Complex32::default(); rhs_host.len()];
    cpu.copy_d2h(&out_cpu, &mut cpu_values);
    cuda.copy_d2h(&out_cuda, &mut cuda_values);
    let scalar_error = max_abs_c32(&cpu_values, &cuda_values);
    if scalar_error > 3.0e-4 {
        return Err(format!(
            "{name} complex32 scalar parity error {scalar_error:e}"
        ));
    }

    let rhs = 2;
    let mut block_cpu = cpu
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_cuda = cuda
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(block_rhs_host, &mut block_cpu);
    cuda.copy_h2d(block_rhs_host, &mut block_cuda);
    let mut block_out_cpu = cpu
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_out_cuda = cuda
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_colmajor(cpu, rhs, &block_cpu, &mut block_out_cpu)
        .map_err(|e| format!("{name} CPU complex32 block action: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_colmajor(cuda, rhs, &block_cuda, &mut block_out_cuda)
        .map_err(|e| format!("{name} CUDA complex32 block action: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA block synchronization: {e}"))?;
    let block_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/complex32-block"), &block_stats)?;
    let mut cpu_block_values = vec![Complex32::default(); block_rhs_host.len()];
    let mut cuda_block_values = vec![Complex32::default(); block_rhs_host.len()];
    cpu.copy_d2h(&block_out_cpu, &mut cpu_block_values);
    cuda.copy_d2h(&block_out_cuda, &mut cuda_block_values);
    let block_error = max_abs_c32(&cpu_block_values, &cuda_block_values);
    if block_error > 3.0e-4 {
        return Err(format!(
            "{name} complex32 block parity error {block_error:e}"
        ));
    }

    Ok(json!({
        "complex32_scalar": {"status": "passed", "max_abs_error": scalar_error, "transfers": transfer_json(&scalar_stats)},
        "complex32_block_rhs": {"status": "passed", "rhs": rhs, "max_abs_error": block_error, "transfers": transfer_json(&block_stats)},
        "capabilities": {
            "forward": format!("{:?}", cuda_preconditioner.forward_capability()),
            "transpose": format!("{:?}", cuda_preconditioner.transpose_capability()),
            "hermitian": format!("{:?}", cuda_preconditioner.hermitian_capability()),
        },
        "diagnostics": diagnostics_json(cuda_preconditioner.diagnostics()),
    }))
}

fn exercise_exact_transpose_f64<Pc, Pg>(
    name: &str,
    cpu: &sonium_compute::CpuBackend,
    cuda: &CudaBackend,
    cpu_preconditioner: &Pc,
    cuda_preconditioner: &Pg,
    rhs_host: &[f64],
    block_rhs_host: &[f64],
) -> Result<Value, String>
where
    Pc: DevicePreconditioner<sonium_compute::CpuBackend, f64>,
    Pg: DevicePreconditioner<CudaBackend, f64>,
{
    let mut rhs_cpu = cpu
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut rhs_cuda = cuda
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(rhs_host, &mut rhs_cuda);
    let mut cpu_out = cpu
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut cuda_out = cuda
        .raw_alloc::<f64>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_transpose(cpu, &rhs_cpu, &mut cpu_out)
        .map_err(|e| format!("{name} CPU transpose: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_transpose(cuda, &rhs_cuda, &mut cuda_out)
        .map_err(|e| format!("{name} CUDA transpose: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA transpose synchronization: {e}"))?;
    let scalar_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/transpose-scalar"), &scalar_stats)?;
    let mut cpu_host = vec![0.0; rhs_host.len()];
    let mut cuda_host = vec![0.0; rhs_host.len()];
    cpu.copy_d2h(&cpu_out, &mut cpu_host);
    cuda.copy_d2h(&cuda_out, &mut cuda_host);
    let scalar_error = max_abs_f64(&cpu_host, &cuda_host);
    if scalar_error > TOLERANCE {
        return Err(format!(
            "{name} transpose scalar parity error {scalar_error:e}"
        ));
    }

    let rhs = 2;
    if block_rhs_host.len() != rhs_host.len() * rhs {
        return Err(format!(
            "{name} transpose block fixture has the wrong length"
        ));
    }
    let mut block_cpu = cpu
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_cuda = cuda
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(block_rhs_host, &mut block_cpu);
    cuda.copy_h2d(block_rhs_host, &mut block_cuda);
    let mut block_out_cpu = cpu
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_out_cuda = cuda
        .raw_alloc::<f64>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_transpose_colmajor(cpu, rhs, &block_cpu, &mut block_out_cpu)
        .map_err(|e| format!("{name} CPU transpose block: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_transpose_colmajor(cuda, rhs, &block_cuda, &mut block_out_cuda)
        .map_err(|e| format!("{name} CUDA transpose block: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA transpose block synchronization: {e}"))?;
    let block_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/transpose-block"), &block_stats)?;
    let mut block_cpu_host = vec![0.0; block_rhs_host.len()];
    let mut block_cuda_host = vec![0.0; block_rhs_host.len()];
    cpu.copy_d2h(&block_out_cpu, &mut block_cpu_host);
    cuda.copy_d2h(&block_out_cuda, &mut block_cuda_host);
    let block_error = max_abs_f64(&block_cpu_host, &block_cuda_host);
    if block_error > TOLERANCE {
        return Err(format!(
            "{name} transpose block parity error {block_error:e}"
        ));
    }
    Ok(json!({
        "scalar": {"status": "passed", "max_abs_error": scalar_error, "transfers": transfer_json(&scalar_stats)},
        "block_rhs": {"status": "passed", "rhs": rhs, "max_abs_error": block_error, "transfers": transfer_json(&block_stats)},
        "capability": "Exact",
    }))
}

fn exercise_exact_transpose_c32<Pc, Pg>(
    name: &str,
    cpu: &sonium_compute::CpuBackend,
    cuda: &CudaBackend,
    cpu_preconditioner: &Pc,
    cuda_preconditioner: &Pg,
    rhs_host: &[Complex32],
    block_rhs_host: &[Complex32],
) -> Result<Value, String>
where
    Pc: DevicePreconditioner<sonium_compute::CpuBackend, Complex32>,
    Pg: DevicePreconditioner<CudaBackend, Complex32>,
{
    let mut rhs_cpu = cpu
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut rhs_cuda = cuda
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(rhs_host, &mut rhs_cpu);
    cuda.copy_h2d(rhs_host, &mut rhs_cuda);
    let mut cpu_out = cpu
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut cuda_out = cuda
        .raw_alloc::<Complex32>(rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_transpose(cpu, &rhs_cpu, &mut cpu_out)
        .map_err(|e| format!("{name} CPU transpose: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_transpose(cuda, &rhs_cuda, &mut cuda_out)
        .map_err(|e| format!("{name} CUDA transpose: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA transpose synchronization: {e}"))?;
    let scalar_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/transpose-scalar"), &scalar_stats)?;
    let mut cpu_host = vec![Complex32::default(); rhs_host.len()];
    let mut cuda_host = vec![Complex32::default(); rhs_host.len()];
    cpu.copy_d2h(&cpu_out, &mut cpu_host);
    cuda.copy_d2h(&cuda_out, &mut cuda_host);
    let scalar_error = max_abs_c32(&cpu_host, &cuda_host);
    if scalar_error > 3.0e-4 {
        return Err(format!(
            "{name} transpose scalar parity error {scalar_error:e}"
        ));
    }

    let rhs = 2;
    if block_rhs_host.len() != rhs_host.len() * rhs {
        return Err(format!(
            "{name} transpose block fixture has the wrong length"
        ));
    }
    let mut block_cpu = cpu
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_cuda = cuda
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu.copy_h2d(block_rhs_host, &mut block_cpu);
    cuda.copy_h2d(block_rhs_host, &mut block_cuda);
    let mut block_out_cpu = cpu
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    let mut block_out_cuda = cuda
        .raw_alloc::<Complex32>(block_rhs_host.len())
        .map_err(|e| e.to_string())?;
    cpu_preconditioner
        .try_apply_transpose_colmajor(cpu, rhs, &block_cpu, &mut block_out_cpu)
        .map_err(|e| format!("{name} CPU transpose block: {e}"))?;
    let before = cuda.runtime_stats();
    cuda_preconditioner
        .try_apply_transpose_colmajor(cuda, rhs, &block_cuda, &mut block_out_cuda)
        .map_err(|e| format!("{name} CUDA transpose block: {e}"))?;
    cuda.try_synchronize()
        .map_err(|e| format!("{name} CUDA transpose block synchronization: {e}"))?;
    let block_stats = cuda.runtime_stats_since(before);
    require_native_action(&format!("{name}/transpose-block"), &block_stats)?;
    let mut block_cpu_host = vec![Complex32::default(); block_rhs_host.len()];
    let mut block_cuda_host = vec![Complex32::default(); block_rhs_host.len()];
    cpu.copy_d2h(&block_out_cpu, &mut block_cpu_host);
    cuda.copy_d2h(&block_out_cuda, &mut block_cuda_host);
    let block_error = max_abs_c32(&block_cpu_host, &block_cuda_host);
    if block_error > 3.0e-4 {
        return Err(format!(
            "{name} transpose block parity error {block_error:e}"
        ));
    }
    Ok(json!({
        "complex32_scalar": {"status": "passed", "max_abs_error": scalar_error, "transfers": transfer_json(&scalar_stats)},
        "complex32_block_rhs": {"status": "passed", "rhs": rhs, "max_abs_error": block_error, "transfers": transfer_json(&block_stats)},
        "capability": "Exact",
    }))
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let cuda = required_cuda_backend()?;
    let cpu = sonium_compute::CpuBackend::new();
    let (row_ptr, col_idx, values) = tridiag_f64(9);
    let a_cpu = BackendCsr::try_from_host(&cpu, 9, 9, &row_ptr, &col_idx, &values)
        .map_err(|e| format!("fixture upload failed: {e}"))?;
    let a_cuda = BackendCsr::try_from_host(&cuda, 9, 9, &row_ptr, &col_idx, &values)
        .map_err(|e| format!("fixture upload failed: {e}"))?;
    let rhs_f64 = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
    let block_f64 = [
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -9.0, 8.0, -7.0, 6.0, -5.0, 4.0, -3.0,
        2.0, -1.0,
    ];
    let point_cpu = PointJacobiBackend::from_csr_host(&cpu, 9, &row_ptr, &col_idx, &values);
    let point_cuda = PointJacobiBackend::from_csr_host(&cuda, 9, &row_ptr, &col_idx, &values);
    let polynomial_cpu =
        PolynomialJacobiBackend::from_csr_host(&cpu, &a_cpu, &row_ptr, &col_idx, &values, 3, 0.75);
    let polynomial_cuda = PolynomialJacobiBackend::from_csr_host(
        &cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 0.75,
    );
    let chebyshev_cpu =
        ChebyshevBackend::from_csr_host(&cpu, &a_cpu, &row_ptr, &col_idx, &values, 3, 30.0);
    let chebyshev_cuda =
        ChebyshevBackend::from_csr_host(&cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 30.0);

    let mut records = serde_json::Map::new();
    records.insert(
        "point_jacobi_f64".into(),
        exercise_f64(
            "point_jacobi_f64",
            &cpu,
            &cuda,
            &point_cpu,
            &point_cuda,
            &rhs_f64,
            &block_f64,
        )?,
    );
    records.insert(
        "point_jacobi_f64_transpose".into(),
        exercise_exact_transpose_f64(
            "point_jacobi_f64",
            &cpu,
            &cuda,
            &point_cpu,
            &point_cuda,
            &rhs_f64,
            &block_f64,
        )?,
    );
    records.insert(
        "polynomial_jacobi_f64".into(),
        exercise_f64(
            "polynomial_jacobi_f64",
            &cpu,
            &cuda,
            &polynomial_cpu,
            &polynomial_cuda,
            &rhs_f64,
            &block_f64,
        )?,
    );
    records.insert(
        "chebyshev_f64".into(),
        exercise_f64(
            "chebyshev_f64",
            &cpu,
            &cuda,
            &chebyshev_cpu,
            &chebyshev_cuda,
            &rhs_f64,
            &block_f64,
        )?,
    );

    let complex_values = values
        .iter()
        .enumerate()
        .map(|(index, value)| Complex32::new(*value as f32, (index as f32 - 3.0) * 0.01))
        .collect::<Vec<_>>();
    let a_complex_cpu = BackendCsr::try_from_host(&cpu, 9, 9, &row_ptr, &col_idx, &complex_values)
        .map_err(|e| format!("fixture upload failed: {e}"))?;
    let a_complex_cuda =
        BackendCsr::try_from_host(&cuda, 9, 9, &row_ptr, &col_idx, &complex_values)
            .map_err(|e| format!("fixture upload failed: {e}"))?;
    let complex_rhs = rhs_f64
        .iter()
        .enumerate()
        .map(|(index, value)| Complex32::new(*value as f32, (index as f32 + 1.0) * 0.125))
        .collect::<Vec<_>>();
    let complex_block = complex_rhs
        .iter()
        .copied()
        .chain(complex_rhs.iter().rev().map(|value| value.conj()))
        .collect::<Vec<_>>();
    let point_complex_cpu =
        PointJacobiBackend::from_csr_host(&cpu, 9, &row_ptr, &col_idx, &complex_values);
    let point_complex_cuda =
        PointJacobiBackend::from_csr_host(&cuda, 9, &row_ptr, &col_idx, &complex_values);
    let polynomial_complex_cpu = PolynomialJacobiBackend::from_csr_host(
        &cpu,
        &a_complex_cpu,
        &row_ptr,
        &col_idx,
        &complex_values,
        2,
        Complex32::new(0.75, 0.0),
    );
    let polynomial_complex_cuda = PolynomialJacobiBackend::from_csr_host(
        &cuda,
        &a_complex_cuda,
        &row_ptr,
        &col_idx,
        &complex_values,
        2,
        Complex32::new(0.75, 0.0),
    );
    let chebyshev_complex_cpu = ChebyshevBackend::from_csr_host(
        &cpu,
        &a_complex_cpu,
        &row_ptr,
        &col_idx,
        &complex_values,
        2,
        30.0,
    );
    let chebyshev_complex_cuda = ChebyshevBackend::from_csr_host(
        &cuda,
        &a_complex_cuda,
        &row_ptr,
        &col_idx,
        &complex_values,
        2,
        30.0,
    );
    records.insert(
        "point_jacobi_complex32".into(),
        exercise_c32(
            "point_jacobi_complex32",
            &cpu,
            &cuda,
            &point_complex_cpu,
            &point_complex_cuda,
            &complex_rhs,
            &complex_block,
        )?,
    );
    records.insert(
        "point_jacobi_complex32_transpose".into(),
        exercise_exact_transpose_c32(
            "point_jacobi_complex32",
            &cpu,
            &cuda,
            &point_complex_cpu,
            &point_complex_cuda,
            &complex_rhs,
            &complex_block,
        )?,
    );
    records.insert(
        "polynomial_jacobi_complex32".into(),
        exercise_c32(
            "polynomial_jacobi_complex32",
            &cpu,
            &cuda,
            &polynomial_complex_cpu,
            &polynomial_complex_cuda,
            &complex_rhs,
            &complex_block,
        )?,
    );
    records.insert(
        "chebyshev_complex32".into(),
        exercise_c32(
            "chebyshev_complex32",
            &cpu,
            &cuda,
            &chebyshev_complex_cpu,
            &chebyshev_complex_cuda,
            &complex_rhs,
            &complex_block,
        )?,
    );

    let amg_matrix = CsrMatrix::from_raw_parts(
        9,
        9,
        row_ptr.iter().map(|index| *index as usize).collect(),
        col_idx.iter().map(|index| *index as usize).collect(),
        values,
    );
    let amg_config = AmgConfig {
        coarse_size: 2,
        max_levels: 8,
        num_pre_smooth: 1,
        num_post_smooth: 1,
        smoother: AmgSmoother::Jacobi,
        ..AmgConfig::default()
    };
    let amg_cpu = BackendAmgPreconditioner::<sonium_compute::CpuBackend, f64>::try_from_csr(
        &cpu,
        &amg_matrix,
        amg_config.clone(),
    )
    .map_err(|e| format!("AMG CPU setup: {e}"))?;
    let amg_cuda =
        BackendAmgPreconditioner::<CudaBackend, f64>::try_from_csr(&cuda, &amg_matrix, amg_config)
            .map_err(|e| format!("AMG CUDA setup: {e}"))?;
    records.insert(
        "amg_f64".into(),
        exercise_f64(
            "amg_f64", &cpu, &cuda, &amg_cpu, &amg_cuda, &rhs_f64, &block_f64,
        )?,
    );

    let rhs = vec![1.0, -2.0, 0.5, 3.0];
    let block_rhs = vec![1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
    let saddle_row_ptr = vec![0, 3, 6, 7, 8];
    let saddle_col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
    let saddle_values = vec![4.0, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
    let schur_row_ptr = vec![0_usize, 2, 4];
    let schur_col_idx = vec![0_usize, 1, 0, 1];
    let schur_values = vec![2.0, 0.5, 0.5, 2.0];
    let saddle_a_cpu =
        BackendCsr::try_from_host(&cpu, 4, 4, &saddle_row_ptr, &saddle_col_idx, &saddle_values)
            .map_err(|e| format!("fixture upload failed: {e}"))?;
    let saddle_a_cuda = BackendCsr::try_from_host(
        &cuda,
        4,
        4,
        &saddle_row_ptr,
        &saddle_col_idx,
        &saddle_values,
    )
    .map_err(|e| format!("fixture upload failed: {e}"))?;
    let saddle_config = sonium_compute::preconditioner::SaddlePointJacobiSchurBackendConfig {
        velocity_block_size: 2,
        velocity_degree: 2,
        velocity_damping: 0.75,
        pressure_degree: 2,
        pressure_damping: 0.75,
        pressure_kind: sonium_compute::preconditioner::PressureSchurBackendKind::Polynomial,
        pressure_chebyshev_degree: 3,
        pressure_chebyshev_eigenvalue_ratio: 30.0,
    };
    let saddle_cpu =
        sonium_compute::preconditioner::SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
            &cpu,
            &saddle_a_cpu,
            &saddle_row_ptr,
            &saddle_col_idx,
            &saddle_values,
            HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
            saddle_config,
        )
        .map_err(|e| format!("saddle-point setup failed: {e}"))?;
    let saddle_cuda =
        sonium_compute::preconditioner::SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
            &cuda,
            &saddle_a_cuda,
            &saddle_row_ptr,
            &saddle_col_idx,
            &saddle_values,
            HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
            saddle_config,
        )
        .map_err(|e| format!("saddle-point setup failed: {e}"))?;
    records.insert(
        "saddle_composed_f64".into(),
        exercise_f64(
            "saddle_composed_f64",
            &cpu,
            &cuda,
            &saddle_cpu,
            &saddle_cuda,
            &rhs,
            &block_rhs,
        )?,
    );

    let mut unsupported = Vec::new();
    let mut unsupported_rhs = cuda.raw_alloc::<f64>(9).map_err(|e| e.to_string())?;
    let mut unsupported_output = cuda.raw_alloc::<f64>(9).map_err(|e| e.to_string())?;
    cuda.copy_h2d(&rhs_f64, &mut unsupported_rhs);
    if point_cuda.transpose_capability() != DeviceAdjointCapability::Exact
        || point_cuda.hermitian_capability() != DeviceAdjointCapability::Unsupported
        || polynomial_cuda.transpose_capability() != DeviceAdjointCapability::Unsupported
        || polynomial_cuda.hermitian_capability() != DeviceAdjointCapability::Unsupported
        || chebyshev_cuda.transpose_capability() != DeviceAdjointCapability::Unsupported
        || chebyshev_cuda.hermitian_capability() != DeviceAdjointCapability::Unsupported
        || amg_cuda.transpose_capability() != DeviceAdjointCapability::Unsupported
        || amg_cuda.hermitian_capability() != DeviceAdjointCapability::Unsupported
    {
        return Err("promoted preconditioner capability contract changed unexpectedly".into());
    }
    unsupported.push(require_unsupported_before_device_work(
        "point_jacobi_f64/hermitian",
        &cuda,
        &point_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        true,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "polynomial_jacobi_f64/transpose",
        &cuda,
        &polynomial_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        false,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "polynomial_jacobi_f64/hermitian",
        &cuda,
        &polynomial_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        true,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "chebyshev_f64/transpose",
        &cuda,
        &chebyshev_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        false,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "chebyshev_f64/hermitian",
        &cuda,
        &chebyshev_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        true,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "amg_f64/transpose",
        &cuda,
        &amg_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        false,
    )?);
    unsupported.push(require_unsupported_before_device_work(
        "amg_f64/hermitian",
        &cuda,
        &amg_cuda,
        &unsupported_rhs,
        &mut unsupported_output,
        true,
    )?);

    let artifact_value = json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "hardware": cuda_hardware_provenance(),
        "device_ordinal": std::env::var("SONIUM_CUDA_DEVICE_ORDINAL").unwrap_or_else(|_| "0".into()),
        "resident_state": {
            "setup_before_actions": true,
            "hierarchy_and_workspaces_reused": true,
            "per_application_host_staging": false,
            "note": "setup uploads and output readbacks are outside action intervals"
        },
        "preconditioners": records,
        "unsupported_adjoint_actions": unsupported,
        "acceptance": {
            "scalar_and_block_host_reference_parity": true,
            "real_and_complex_promoted_cells": true,
            "native_action_intervals_have_zero_h2d_d2h_and_staging": true,
            "unsupported_adjoint_actions_fail_before_device_work": true,
            "tolerance_f64": TOLERANCE,
        },
    });
    let artifact = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|e| format!("serialize CUDA preconditioner artifact: {e}"))?;
    let (path, checksum) = write_qa_artifact("solvers_preconditioner_cuda.json", &artifact)?;

    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("CPU backend-resident preconditioner actions")
        .reference_checksum("analytic:tridiag-preconditioner-cuda-v1")
        .dimension("f64 and Complex32 scalar/block RHS across Jacobi, polynomial, Chebyshev, AMG, and saddle composition")
        .formulation("native CUDA device-resident preconditioner action and capability matrix")
        .tolerance("f64 <= 2e-10; Complex32 <= 3e-4; no action-interval H2D/D2H/staging")
        .metric("promoted_preconditioner_families", 5.0)
        .metric("native_scalar_block_action_cells", 16.0)
        .metric("native_adjoint_action_cells", 4.0)
        .metric("unsupported_adjoint_cells", unsupported.len() as f64)
        .minimum("promoted_preconditioner_families", 5.0)
        .minimum("native_scalar_block_action_cells", 16.0)
        .minimum("native_adjoint_action_cells", 4.0)
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/solvers_preconditioner_cuda.rs",
            "native CUDA preconditioner parity, residency, transfer, and capability evidence",
            "v1",
            checksum,
        );
    let mut hardware = cuda_hardware_provenance();
    hardware.numa_policy = Some("CUDA ordinal 0; native device action intervals".into());
    result = result.hardware_provenance(hardware);
    Ok(result.metric("elapsed_seconds", started.elapsed().as_secs_f64()))
}

fn main() {
    match run() {
        Ok(result) => {
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
