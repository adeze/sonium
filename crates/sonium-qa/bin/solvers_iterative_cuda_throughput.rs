//! Correctness-gated throughput evidence for the backend-resident iterative
//! solver cells.  The CUDA side is required to be a real device backend; a
//! host fallback is recorded as unsupported and never reported as throughput.

#![cfg(feature = "cuda")]

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::ReductionMode;
use sonium_compute::{
    Backend, BackendContextDisposition, BackendCsr, BackendOutputDisposition, CapabilityState,
    CpuBackend, CudaBackend, CudaRuntimeStatsSnapshot, IdentityDevicePreconditioner, MemoryPool,
    MemoryPoolConfig, TransferStatsSnapshot,
};
use sonium_qa::{QaResult, cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use sonium_solvers::{
    BackendKrylovCapability, BackendKrylovConfig, BackendKrylovError,
    BackendKrylovFailureDiagnostic, BackendKrylovMethod, BackendKrylovScalar, BackendKrylovStatus,
    BackendKrylovWorkspace, BackendStoppingReason, backend_krylov_capability_matrix,
    try_block_fgmres_backend, try_block_gmres_backend, try_fgmres_backend_preconditioned,
    try_gmres_backend,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-cuda-throughput.v2";
const ARTIFACT: &str = "solvers_iterative_cuda_throughput.json";
const REPEATS: usize = 3;
const N: usize = 64;

#[derive(Debug, Clone, Copy)]
struct Probe {
    transfers: TransferStatsSnapshot,
    synchronizations: usize,
    launches: usize,
    peak_bytes: usize,
    live_bytes: usize,
}

impl Probe {
    fn delta(self, before: Self) -> Self {
        Self {
            transfers: self.transfers.saturating_sub(before.transfers),
            synchronizations: self
                .synchronizations
                .saturating_sub(before.synchronizations),
            launches: self.launches.saturating_sub(before.launches),
            peak_bytes: self.peak_bytes,
            live_bytes: self.live_bytes,
        }
    }
}

trait RuntimeProbe: Backend {
    fn probe(&self) -> Probe;
    fn reset_probe(&self);
}

impl RuntimeProbe for CpuBackend {
    fn probe(&self) -> Probe {
        let memory = self.memory_stats();
        Probe {
            transfers: self.transfer_stats(),
            synchronizations: 0,
            launches: 0,
            peak_bytes: memory.allocation_high_water_bytes,
            live_bytes: memory.live_allocation_bytes,
        }
    }

    fn reset_probe(&self) {
        self.reset_transfer_stats();
    }
}

impl RuntimeProbe for CudaBackend {
    fn probe(&self) -> Probe {
        let stats: CudaRuntimeStatsSnapshot = self.runtime_stats();
        Probe {
            transfers: stats.transfer,
            synchronizations: stats.synchronization_calls,
            launches: stats.kernel_launches + stats.library_operations,
            peak_bytes: stats.allocation_high_water_bytes + stats.staging_high_water_bytes,
            live_bytes: stats.live_allocation_bytes + stats.live_staging_bytes,
        }
    }

    fn reset_probe(&self) {
        self.reset_runtime_stats();
    }
}

#[derive(Debug)]
struct RealWorkload {
    id: &'static str,
    kind: &'static str,
    method: BackendKrylovMethod,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    values: Vec<f64>,
    rhs: Vec<f64>,
    expected: Vec<f64>,
    rhs_count: usize,
}

#[derive(Debug)]
struct ComplexWorkload {
    id: &'static str,
    kind: &'static str,
    method: BackendKrylovMethod,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    values: Vec<Complex64>,
    rhs: Vec<Complex64>,
    expected: Vec<Complex64>,
}

#[derive(Debug)]
struct RunReport<T> {
    backend: String,
    status: BackendKrylovStatus<f64>,
    solution: Vec<T>,
    setup_ms: f64,
    warmup_ms: f64,
    iteration_ms_per_solve: f64,
    synchronization_ms: f64,
    steady_probe: Probe,
    sync_probe: Probe,
    final_probe: Probe,
    final_residual: f64,
}

/// Direct vendor-library reference for the sparse operation represented by a
/// workload.  This is intentionally separate from the Krylov timing: setup,
/// warm-up, steady submissions, synchronization, cuBLAS validation, and final
/// readback are all reported independently.
#[derive(Debug)]
struct VendorBaseline {
    backend: String,
    operation: &'static str,
    primitive: &'static str,
    setup_ms: f64,
    warmup_ms: f64,
    steady_ms_per_operation: f64,
    synchronization_ms: f64,
    validation_ms: f64,
    setup_probe: Probe,
    steady_probe: Probe,
    sync_probe: Probe,
    validation_probe: Probe,
    final_probe: Probe,
    independent_final_relative_residual: f64,
    expected_output_max_abs_error: f64,
    vendor_output_nrm2: f64,
    expected_output_nrm2: f64,
}

fn real_stencil(diagonal: f64, lower: f64, upper: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(N + 1);
    let mut col_idx = Vec::with_capacity(3 * N);
    let mut values = Vec::with_capacity(3 * N);
    row_ptr.push(0);
    for row in 0..N {
        if row > 0 {
            col_idx.push((row - 1) as i32);
            values.push(lower);
        }
        col_idx.push(row as i32);
        values.push(diagonal);
        if row + 1 < N {
            col_idx.push((row + 1) as i32);
            values.push(upper);
        }
        row_ptr.push(col_idx.len() as i32);
    }
    (row_ptr, col_idx, values)
}

fn complex_helmholtz() -> (Vec<i32>, Vec<i32>, Vec<Complex64>) {
    let mut row_ptr = Vec::with_capacity(N + 1);
    let mut col_idx = Vec::with_capacity(3 * N);
    let mut values = Vec::with_capacity(3 * N);
    row_ptr.push(0);
    for row in 0..N {
        if row > 0 {
            col_idx.push((row - 1) as i32);
            values.push(Complex64::new(-1.0, -0.03));
        }
        col_idx.push(row as i32);
        values.push(Complex64::new(1.55, 0.11));
        if row + 1 < N {
            col_idx.push((row + 1) as i32);
            values.push(Complex64::new(-1.0, 0.02));
        }
        row_ptr.push(col_idx.len() as i32);
    }
    (row_ptr, col_idx, values)
}

fn real_matvec(row_ptr: &[i32], col_idx: &[i32], values: &[f64], x: &[f64]) -> Vec<f64> {
    row_ptr
        .windows(2)
        .map(|window| {
            let start = window[0] as usize;
            let end = window[1] as usize;
            (start..end)
                .map(|index| values[index] * x[col_idx[index] as usize])
                .sum()
        })
        .collect()
}

fn complex_matvec(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[Complex64],
    x: &[Complex64],
) -> Vec<Complex64> {
    row_ptr
        .windows(2)
        .map(|window| {
            let start = window[0] as usize;
            let end = window[1] as usize;
            (start..end)
                .map(|index| values[index] * x[col_idx[index] as usize])
                .sum()
        })
        .collect()
}

fn real_case(
    id: &'static str,
    kind: &'static str,
    method: BackendKrylovMethod,
    diagonal: f64,
    lower: f64,
    upper: f64,
    rhs_count: usize,
) -> RealWorkload {
    let (row_ptr, col_idx, values) = real_stencil(diagonal, lower, upper);
    let expected_one: Vec<f64> = (0..N).map(|index| 1.0 + index as f64 * 0.002).collect();
    let rhs_one = real_matvec(&row_ptr, &col_idx, &values, &expected_one);
    let (rhs, expected) = if rhs_count == 1 {
        (rhs_one, expected_one)
    } else {
        let expected_two: Vec<f64> = (0..N).map(|index| -0.5 + index as f64 * 0.001).collect();
        let rhs_two = real_matvec(&row_ptr, &col_idx, &values, &expected_two);
        let mut rhs = rhs_one;
        rhs.extend_from_slice(&rhs_two);
        let mut expected = expected_one;
        expected.extend_from_slice(&expected_two);
        (rhs, expected)
    };
    RealWorkload {
        id,
        kind,
        method,
        row_ptr,
        col_idx,
        values,
        rhs,
        expected,
        rhs_count,
    }
}

fn complex_case() -> ComplexWorkload {
    let (row_ptr, col_idx, values) = complex_helmholtz();
    let expected: Vec<_> = (0..N)
        .map(|index| Complex64::new(0.8 + index as f64 * 0.001, -0.2 + index as f64 * 0.0005))
        .collect();
    let rhs = complex_matvec(&row_ptr, &col_idx, &values, &expected);
    ComplexWorkload {
        id: "helmholtz_complex64",
        kind: "helmholtz",
        method: BackendKrylovMethod::Fgmres,
        row_ptr,
        col_idx,
        values,
        rhs,
        expected,
    }
}

fn config() -> BackendKrylovConfig<f64> {
    BackendKrylovConfig {
        max_iterations: 320,
        restart: 32,
        tolerance: 1.0e-8,
        require_native: true,
        exact_residual_check_interval: 1,
    }
}

fn invoke_real<B: Backend>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut BackendKrylovWorkspace<B, f64>,
    matrix: &BackendCsr<B, f64>,
    rhs: &B::Buffer<f64>,
    output: &mut B::Buffer<f64>,
    method: BackendKrylovMethod,
    rhs_count: usize,
) -> Result<BackendKrylovStatus<f64>, String> {
    let cfg = config();
    let result = match method {
        BackendKrylovMethod::Gmres => try_gmres_backend(
            backend,
            pool,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            &cfg,
            None,
        ),
        BackendKrylovMethod::Fgmres => try_fgmres_backend_preconditioned(
            backend,
            pool,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            &cfg,
            None,
        ),
        BackendKrylovMethod::BlockGmres => try_block_gmres_backend(
            backend,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            rhs_count,
            &cfg,
            None,
        ),
        BackendKrylovMethod::BlockFgmres => try_block_fgmres_backend(
            backend,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            rhs_count,
            &cfg,
            None,
        ),
    };
    result.map_err(|error| error.to_string())
}

fn invoke_complex<B: Backend>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut BackendKrylovWorkspace<B, Complex64>,
    matrix: &BackendCsr<B, Complex64>,
    rhs: &B::Buffer<Complex64>,
    output: &mut B::Buffer<Complex64>,
    method: BackendKrylovMethod,
) -> Result<BackendKrylovStatus<f64>, String> {
    let cfg = config();
    match method {
        BackendKrylovMethod::Gmres => try_gmres_backend(
            backend,
            pool,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            &cfg,
            None,
        ),
        BackendKrylovMethod::Fgmres => try_fgmres_backend_preconditioned(
            backend,
            pool,
            workspace,
            matrix,
            &IdentityDevicePreconditioner,
            rhs,
            None,
            output,
            &cfg,
            None,
        ),
        BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres => {
            Err(BackendKrylovError::InvalidArgument {
                method,
                reason: "complex SI3 fixture is scalar-only".into(),
                diagnostic: BackendKrylovFailureDiagnostic {
                    method,
                    actual_backend: backend.capabilities().backend_name,
                    iterations: 0,
                    stopping_reason: BackendStoppingReason::Failed,
                    projected_residual: None,
                    true_residual: None,
                    backward_error: None,
                    partial_iterate: BackendOutputDisposition::Preserved,
                    context_disposition: BackendContextDisposition::Reusable,
                },
            })
        }
    }
    .map_err(|error| error.to_string())
}

fn invoke_vendor_real<B: Backend>(
    backend: &B,
    prebaked: &B::CsrPrebaked<'_, f64>,
    rhs: &B::Buffer<f64>,
    output: &mut B::Buffer<f64>,
    rhs_count: usize,
) -> Result<(), String> {
    let result = if rhs_count == 1 {
        backend.try_spmv_prebaked(prebaked, rhs, output)
    } else {
        backend.try_spmm_prebaked_colmajor(prebaked, rhs_count, rhs, output)
    };
    result.map_err(|error| format!("vendor sparse operation: {error}"))
}

fn invoke_vendor_complex<B: Backend>(
    backend: &B,
    prebaked: &B::CsrPrebaked<'_, Complex64>,
    rhs: &B::Buffer<Complex64>,
    output: &mut B::Buffer<Complex64>,
) -> Result<(), String> {
    backend
        .try_spmv_prebaked(prebaked, rhs, output)
        .map_err(|error| format!("vendor sparse operation: {error}"))
}

fn real_residual(work: &RealWorkload, solution: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut rhs_sum = 0.0;
    for column in 0..work.rhs_count {
        let start = column * N;
        let end = start + N;
        let actual = real_matvec(
            &work.row_ptr,
            &work.col_idx,
            &work.values,
            &solution[start..end],
        );
        for (computed, requested) in actual.iter().zip(&work.rhs[start..end]) {
            let delta = computed - requested;
            sum += delta * delta;
            rhs_sum += requested * requested;
        }
    }
    (sum / rhs_sum.max(f64::MIN_POSITIVE)).sqrt()
}

fn complex_residual(work: &ComplexWorkload, solution: &[Complex64]) -> f64 {
    let actual = complex_matvec(&work.row_ptr, &work.col_idx, &work.values, solution);
    let numerator: f64 = actual
        .iter()
        .zip(&work.rhs)
        .map(|(computed, requested)| (*computed - *requested).norm_sqr())
        .sum();
    let denominator: f64 = work.rhs.iter().map(Complex64::norm_sqr).sum();
    (numerator / denominator.max(f64::MIN_POSITIVE)).sqrt()
}

fn relative_output_error(actual: &[f64], expected: &[f64]) -> f64 {
    let numerator: f64 = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum();
    let denominator: f64 = expected.iter().map(|value| value * value).sum();
    (numerator / denominator.max(f64::MIN_POSITIVE)).sqrt()
}

fn relative_complex_output_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let numerator: f64 = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum();
    let denominator: f64 = expected.iter().map(Complex64::norm_sqr).sum();
    (numerator / denominator.max(f64::MIN_POSITIVE)).sqrt()
}

fn run_vendor_real<B: RuntimeProbe>(
    pool: MemoryPool<B>,
    work: &RealWorkload,
) -> Result<VendorBaseline, String> {
    let backend = pool.backend();
    let setup_before = backend.probe();
    let setup_start = Instant::now();
    let matrix =
        BackendCsr::try_from_host(backend, N, N, &work.row_ptr, &work.col_idx, &work.values)
            .map_err(|error| format!("{} vendor CSR setup: {error}", work.id))?;
    let prebaked = matrix
        .try_prebake(backend)
        .map_err(|error| format!("{} vendor CSR descriptor: {error}", work.id))?;
    let mut input = pool
        .try_alloc::<f64>(work.rhs.len())
        .map_err(|error| format!("{} vendor input allocation: {error}", work.id))?;
    let mut output = pool
        .try_alloc::<f64>(work.rhs.len())
        .map_err(|error| format!("{} vendor output allocation: {error}", work.id))?;
    backend
        .try_copy_h2d(&work.expected, input.raw_mut())
        .map_err(|error| format!("{} vendor input upload: {error}", work.id))?;
    backend
        .try_copy_h2d(&vec![0.0; work.rhs.len()], output.raw_mut())
        .map_err(|error| format!("{} vendor output initialization: {error}", work.id))?;
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let after_setup = backend.probe();

    let warm_start = Instant::now();
    invoke_vendor_real(
        backend,
        &prebaked,
        input.raw(),
        output.raw_mut(),
        work.rhs_count,
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("{} vendor warm-up synchronization: {error}", work.id))?;
    let warmup_ms = warm_start.elapsed().as_secs_f64() * 1_000.0;

    backend.reset_probe();
    let before = backend.probe();
    let steady_start = Instant::now();
    for _ in 0..REPEATS {
        invoke_vendor_real(
            backend,
            &prebaked,
            input.raw(),
            output.raw_mut(),
            work.rhs_count,
        )?;
    }
    let steady_ms_per_operation = steady_start.elapsed().as_secs_f64() * 1_000.0 / REPEATS as f64;
    let after_steady = backend.probe();

    let synchronization_start = Instant::now();
    backend
        .try_synchronize()
        .map_err(|error| format!("{} vendor synchronization: {error}", work.id))?;
    let synchronization_ms = synchronization_start.elapsed().as_secs_f64() * 1_000.0;
    let after_synchronization = backend.probe();

    let validation_start = Instant::now();
    let vendor_output_nrm2 = backend
        .try_nrm2_with_mode(ReductionMode::Fast, output.raw())
        .map_err(|error| format!("{} vendor cuBLAS nrm2 validation: {error}", work.id))?;
    let validation_ms = validation_start.elapsed().as_secs_f64() * 1_000.0;
    let after_validation = backend.probe();

    let mut output_host = vec![0.0; work.rhs.len()];
    backend
        .try_copy_d2h(output.raw(), &mut output_host)
        .map_err(|error| format!("{} vendor final readback: {error}", work.id))?;
    let after_readback = backend.probe();
    let independent_final_relative_residual = relative_output_error(&output_host, &work.rhs);
    let expected_output_max_abs_error = real_error(&output_host, &work.rhs);
    let expected_output_nrm2 = work
        .rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !independent_final_relative_residual.is_finite()
        || independent_final_relative_residual > 2.0e-7
        || !expected_output_max_abs_error.is_finite()
        || expected_output_max_abs_error > 2.0e-6
        || !vendor_output_nrm2.is_finite()
        || !expected_output_nrm2.is_finite()
        || (vendor_output_nrm2 - expected_output_nrm2).abs() > 2.0e-6 * expected_output_nrm2
    {
        return Err(format!(
            "{} vendor numerical gate failed: residual={independent_final_relative_residual:e}, output_error={expected_output_max_abs_error:e}, output_nrm2={vendor_output_nrm2:e}, expected_nrm2={expected_output_nrm2:e}",
            work.id
        ));
    }
    Ok(VendorBaseline {
        backend: backend.capabilities().backend_name.to_owned(),
        operation: if work.rhs_count == 1 {
            "cusparseSpMV"
        } else {
            "cusparseSpMM"
        },
        primitive: if work.rhs_count == 1 {
            "cuSPARSE generic CSR SpMV + cuBLAS NRM2 validation"
        } else {
            "cuSPARSE generic CSR SpMM + cuBLAS NRM2 validation"
        },
        setup_ms,
        warmup_ms,
        steady_ms_per_operation,
        synchronization_ms,
        validation_ms,
        setup_probe: after_setup.delta(setup_before),
        steady_probe: after_steady.delta(before),
        sync_probe: after_synchronization.delta(after_steady),
        validation_probe: after_validation.delta(after_synchronization),
        final_probe: after_readback.delta(after_validation),
        independent_final_relative_residual,
        expected_output_max_abs_error,
        vendor_output_nrm2,
        expected_output_nrm2,
    })
}

fn run_vendor_complex<B: RuntimeProbe>(
    pool: MemoryPool<B>,
    work: &ComplexWorkload,
) -> Result<VendorBaseline, String> {
    let backend = pool.backend();
    let setup_before = backend.probe();
    let setup_start = Instant::now();
    let matrix =
        BackendCsr::try_from_host(backend, N, N, &work.row_ptr, &work.col_idx, &work.values)
            .map_err(|error| format!("{} vendor CSR setup: {error}", work.id))?;
    let prebaked = matrix
        .try_prebake(backend)
        .map_err(|error| format!("{} vendor CSR descriptor: {error}", work.id))?;
    let mut input = pool
        .try_alloc::<Complex64>(work.rhs.len())
        .map_err(|error| format!("{} vendor input allocation: {error}", work.id))?;
    let mut output = pool
        .try_alloc::<Complex64>(work.rhs.len())
        .map_err(|error| format!("{} vendor output allocation: {error}", work.id))?;
    backend
        .try_copy_h2d(&work.expected, input.raw_mut())
        .map_err(|error| format!("{} vendor input upload: {error}", work.id))?;
    backend
        .try_copy_h2d(
            &vec![Complex64::default(); work.rhs.len()],
            output.raw_mut(),
        )
        .map_err(|error| format!("{} vendor output initialization: {error}", work.id))?;
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let after_setup = backend.probe();

    let warm_start = Instant::now();
    invoke_vendor_complex(backend, &prebaked, input.raw(), output.raw_mut())?;
    backend
        .try_synchronize()
        .map_err(|error| format!("{} vendor warm-up synchronization: {error}", work.id))?;
    let warmup_ms = warm_start.elapsed().as_secs_f64() * 1_000.0;

    backend.reset_probe();
    let before = backend.probe();
    let steady_start = Instant::now();
    for _ in 0..REPEATS {
        invoke_vendor_complex(backend, &prebaked, input.raw(), output.raw_mut())?;
    }
    let steady_ms_per_operation = steady_start.elapsed().as_secs_f64() * 1_000.0 / REPEATS as f64;
    let after_steady = backend.probe();

    let synchronization_start = Instant::now();
    backend
        .try_synchronize()
        .map_err(|error| format!("{} vendor synchronization: {error}", work.id))?;
    let synchronization_ms = synchronization_start.elapsed().as_secs_f64() * 1_000.0;
    let after_synchronization = backend.probe();

    let validation_start = Instant::now();
    let vendor_output_nrm2 = backend
        .try_nrm2_with_mode(ReductionMode::Fast, output.raw())
        .map_err(|error| format!("{} vendor cuBLAS nrm2 validation: {error}", work.id))?;
    let validation_ms = validation_start.elapsed().as_secs_f64() * 1_000.0;
    let after_validation = backend.probe();

    let mut output_host = vec![Complex64::default(); work.rhs.len()];
    backend
        .try_copy_d2h(output.raw(), &mut output_host)
        .map_err(|error| format!("{} vendor final readback: {error}", work.id))?;
    let after_readback = backend.probe();
    let independent_final_relative_residual =
        relative_complex_output_error(&output_host, &work.rhs);
    let expected_output_max_abs_error = complex_error(&output_host, &work.rhs);
    let expected_output_nrm2 = work.rhs.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if !independent_final_relative_residual.is_finite()
        || independent_final_relative_residual > 2.0e-7
        || !expected_output_max_abs_error.is_finite()
        || expected_output_max_abs_error > 2.0e-5
        || !vendor_output_nrm2.is_finite()
        || !expected_output_nrm2.is_finite()
        || (vendor_output_nrm2 - expected_output_nrm2).abs() > 2.0e-5 * expected_output_nrm2
    {
        return Err(format!(
            "{} vendor numerical gate failed: residual={independent_final_relative_residual:e}, output_error={expected_output_max_abs_error:e}, output_nrm2={vendor_output_nrm2:e}, expected_nrm2={expected_output_nrm2:e}",
            work.id
        ));
    }
    Ok(VendorBaseline {
        backend: backend.capabilities().backend_name.to_owned(),
        operation: "cusparseSpMV",
        primitive: "cuSPARSE generic complex CSR SpMV + cuBLAS NRM2 validation",
        setup_ms,
        warmup_ms,
        steady_ms_per_operation,
        synchronization_ms,
        validation_ms,
        setup_probe: after_setup.delta(setup_before),
        steady_probe: after_steady.delta(before),
        sync_probe: after_synchronization.delta(after_steady),
        validation_probe: after_validation.delta(after_synchronization),
        final_probe: after_readback.delta(after_validation),
        independent_final_relative_residual,
        expected_output_max_abs_error,
        vendor_output_nrm2,
        expected_output_nrm2,
    })
}

fn probe_json(probe: Probe) -> Value {
    json!({
        "h2d_operations": probe.transfers.h2d_operations,
        "h2d_bytes": probe.transfers.h2d_bytes,
        "d2h_operations": probe.transfers.d2h_operations,
        "d2h_bytes": probe.transfers.d2h_bytes,
        "d2d_operations": probe.transfers.d2d_operations,
        "d2d_bytes": probe.transfers.d2d_bytes,
        "synchronizations": probe.synchronizations,
        "launches_or_vendor_operations": probe.launches,
        "peak_bytes": probe.peak_bytes,
        "live_bytes": probe.live_bytes,
    })
}

fn vendor_baseline_json(run: &VendorBaseline, dofs: usize, rhs_count: usize) -> Value {
    let steady_throughput =
        dofs as f64 * rhs_count as f64 / (run.steady_ms_per_operation.max(1.0e-9) / 1_000.0);
    json!({
        "status": "passed",
        "backend": run.backend,
        "operation": run.operation,
        "primitive": run.primitive,
        "setup_ms": run.setup_ms,
        "warmup_ms": run.warmup_ms,
        "steady_ms_per_operation": run.steady_ms_per_operation,
        "synchronization_ms": run.synchronization_ms,
        "validation_ms": run.validation_ms,
        "steady_throughput_dofs_per_second": steady_throughput,
        "setup_runtime": probe_json(run.setup_probe),
        "steady_runtime": probe_json(run.steady_probe),
        "synchronization_runtime": probe_json(run.sync_probe),
        "validation_runtime": probe_json(run.validation_probe),
        "final_readback_runtime": probe_json(run.final_probe),
        "numerical_gate": {
            "status": "passed",
            "independent_final_relative_residual": run.independent_final_relative_residual,
            "expected_output_max_abs_error": run.expected_output_max_abs_error,
            "vendor_output_nrm2": run.vendor_output_nrm2,
            "expected_output_nrm2": run.expected_output_nrm2,
        },
        "comparison": {
            "status": "not_claimed",
            "reason": "vendor primitive timing is recorded without a solver speedup claim",
        },
    })
}

fn status_json<T>(run: &RunReport<T>, expected: &[T], error: f64) -> Value {
    let _ = expected;
    let dofs = run.solution.len();
    let throughput =
        dofs as f64 * REPEATS as f64 / (run.iteration_ms_per_solve.max(1.0e-9) / 1_000.0);
    json!({
        "backend": run.backend,
        "status": "passed",
        "iterations": run.status.iterations,
        "restarts": run.status.restarts,
        "projected_residual": run.status.projected_residual,
        "true_residual": run.status.true_residual,
        "backward_error": run.status.backward_error,
        "converged": run.status.converged,
        "stopping_reason": format!("{:?}", run.status.stopping_reason),
        "partial_iterate": format!("{:?}", run.status.partial_iterate),
        "setup_ms": run.setup_ms,
        "warmup_ms": run.warmup_ms,
        "iteration_ms_per_solve": run.iteration_ms_per_solve,
        "synchronization_ms": run.synchronization_ms,
        "steady_throughput_dofs_per_second": throughput,
        "steady_runtime": probe_json(run.steady_probe),
        "synchronization_runtime": probe_json(run.sync_probe),
        "final_readback_runtime": probe_json(run.final_probe),
        "independent_final_relative_residual": run.final_residual,
        "expected_solution_max_abs_error": error,
    })
}

fn run_real<B: RuntimeProbe>(
    pool: MemoryPool<B>,
    work: &RealWorkload,
) -> Result<RunReport<f64>, String> {
    let backend = pool.backend();
    let setup_start = Instant::now();
    let matrix =
        BackendCsr::try_from_host(backend, N, N, &work.row_ptr, &work.col_idx, &work.values)
            .map_err(|error| format!("{} CSR setup: {error}", work.id))?;
    let mut rhs = pool
        .try_alloc::<f64>(work.rhs.len())
        .map_err(|error| format!("{} RHS allocation: {error}", work.id))?;
    let mut output = pool
        .try_alloc::<f64>(work.rhs.len())
        .map_err(|error| format!("{} output allocation: {error}", work.id))?;
    backend
        .try_copy_h2d(&work.rhs, rhs.raw_mut())
        .map_err(|error| format!("{} RHS upload: {error}", work.id))?;
    backend
        .try_copy_h2d(&vec![0.0; work.rhs.len()], output.raw_mut())
        .map_err(|error| format!("{} output initialization: {error}", work.id))?;
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let mut workspace = BackendKrylovWorkspace::new(N);
    let warm_start = Instant::now();
    let warm = invoke_real(
        backend,
        &pool,
        &mut workspace,
        &matrix,
        rhs.raw(),
        output.raw_mut(),
        work.method,
        work.rhs_count,
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("{} warmup synchronization: {error}", work.id))?;
    if !warm.converged || !warm.true_residual.is_finite() {
        return Err(format!(
            "{} CPU/CUDA warmup did not converge: {warm:?}",
            work.id
        ));
    }
    let warmup_ms = warm_start.elapsed().as_secs_f64() * 1_000.0;
    backend.reset_probe();
    let before = backend.probe();
    let iteration_start = Instant::now();
    let mut status = warm;
    for _ in 0..REPEATS {
        status = invoke_real(
            backend,
            &pool,
            &mut workspace,
            &matrix,
            rhs.raw(),
            output.raw_mut(),
            work.method,
            work.rhs_count,
        )?;
        if !status.converged || !status.true_residual.is_finite() {
            return Err(format!(
                "{} timed solve did not converge: {status:?}",
                work.id
            ));
        }
    }
    let iteration_ms_per_solve = iteration_start.elapsed().as_secs_f64() * 1_000.0 / REPEATS as f64;
    let after_iteration = backend.probe();
    let sync_start = Instant::now();
    backend
        .try_synchronize()
        .map_err(|error| format!("{} final synchronization: {error}", work.id))?;
    let synchronization_ms = sync_start.elapsed().as_secs_f64() * 1_000.0;
    let after_sync = backend.probe();
    let mut solution = vec![0.0; work.rhs.len()];
    backend
        .try_copy_d2h(output.raw(), &mut solution)
        .map_err(|error| format!("{} final readback: {error}", work.id))?;
    let after_readback = backend.probe();
    let final_residual = real_residual(work, &solution);
    if !final_residual.is_finite() || final_residual > 2.0e-7 {
        return Err(format!(
            "{} independent residual {final_residual:e}",
            work.id
        ));
    }
    Ok(RunReport {
        backend: backend.capabilities().backend_name.to_owned(),
        status,
        solution,
        setup_ms,
        warmup_ms,
        iteration_ms_per_solve,
        synchronization_ms,
        steady_probe: after_iteration.delta(before),
        sync_probe: after_sync.delta(after_iteration),
        final_probe: after_readback.delta(after_sync),
        final_residual,
    })
}

fn run_complex<B: RuntimeProbe>(
    pool: MemoryPool<B>,
    work: &ComplexWorkload,
) -> Result<RunReport<Complex64>, String> {
    let backend = pool.backend();
    let setup_start = Instant::now();
    let matrix =
        BackendCsr::try_from_host(backend, N, N, &work.row_ptr, &work.col_idx, &work.values)
            .map_err(|error| format!("{} CSR setup: {error}", work.id))?;
    let mut rhs = pool
        .try_alloc::<Complex64>(work.rhs.len())
        .map_err(|error| format!("{} RHS allocation: {error}", work.id))?;
    let mut output = pool
        .try_alloc::<Complex64>(work.rhs.len())
        .map_err(|error| format!("{} output allocation: {error}", work.id))?;
    backend
        .try_copy_h2d(&work.rhs, rhs.raw_mut())
        .map_err(|error| format!("{} RHS upload: {error}", work.id))?;
    backend
        .try_copy_h2d(
            &vec![Complex64::default(); work.rhs.len()],
            output.raw_mut(),
        )
        .map_err(|error| format!("{} output initialization: {error}", work.id))?;
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let mut workspace = BackendKrylovWorkspace::new(N);
    let warm_start = Instant::now();
    let warm = invoke_complex(
        backend,
        &pool,
        &mut workspace,
        &matrix,
        rhs.raw(),
        output.raw_mut(),
        work.method,
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("{} warmup synchronization: {error}", work.id))?;
    if !warm.converged || !warm.true_residual.is_finite() {
        return Err(format!("{} warmup did not converge: {warm:?}", work.id));
    }
    let warmup_ms = warm_start.elapsed().as_secs_f64() * 1_000.0;
    backend.reset_probe();
    let before = backend.probe();
    let iteration_start = Instant::now();
    let mut status = warm;
    for _ in 0..REPEATS {
        status = invoke_complex(
            backend,
            &pool,
            &mut workspace,
            &matrix,
            rhs.raw(),
            output.raw_mut(),
            work.method,
        )?;
        if !status.converged || !status.true_residual.is_finite() {
            return Err(format!(
                "{} timed solve did not converge: {status:?}",
                work.id
            ));
        }
    }
    let iteration_ms_per_solve = iteration_start.elapsed().as_secs_f64() * 1_000.0 / REPEATS as f64;
    let after_iteration = backend.probe();
    let sync_start = Instant::now();
    backend
        .try_synchronize()
        .map_err(|error| format!("{} final synchronization: {error}", work.id))?;
    let synchronization_ms = sync_start.elapsed().as_secs_f64() * 1_000.0;
    let after_sync = backend.probe();
    let mut solution = vec![Complex64::default(); work.rhs.len()];
    backend
        .try_copy_d2h(output.raw(), &mut solution)
        .map_err(|error| format!("{} final readback: {error}", work.id))?;
    let after_readback = backend.probe();
    let final_residual = complex_residual(work, &solution);
    if !final_residual.is_finite() || final_residual > 2.0e-7 {
        return Err(format!(
            "{} independent residual {final_residual:e}",
            work.id
        ));
    }
    Ok(RunReport {
        backend: backend.capabilities().backend_name.to_owned(),
        status,
        solution,
        setup_ms,
        warmup_ms,
        iteration_ms_per_solve,
        synchronization_ms,
        steady_probe: after_iteration.delta(before),
        sync_probe: after_sync.delta(after_iteration),
        final_probe: after_readback.delta(after_sync),
        final_residual,
    })
}

fn real_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn complex_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn capability_for(
    capabilities: &[BackendKrylovCapability],
    method: BackendKrylovMethod,
    scalar: BackendKrylovScalar,
) -> Option<BackendKrylovCapability> {
    capabilities
        .iter()
        .find(|cell| cell.method == method && cell.scalar == scalar)
        .copied()
}

fn state_json(state: CapabilityState) -> &'static str {
    match state {
        CapabilityState::Native => "native",
        CapabilityState::VendorNative => "vendor_native",
        CapabilityState::Unsupported => "unsupported",
        CapabilityState::HostSharedFallback => "host_shared_fallback",
        CapabilityState::HostStagedFallback => "host_staged_fallback",
    }
}

fn capability_json(cell: BackendKrylovCapability) -> Value {
    json!({
        "backend": cell.backend,
        "method": format!("{:?}", cell.method),
        "scalar": format!("{:?}", cell.scalar),
        "scalar_state": state_json(cell.scalar_state),
        "blas1": state_json(cell.blas1),
        "axpy_many": state_json(cell.axpy_many),
        "multi_dot": state_json(cell.multi_dot),
        "strided_projection": state_json(cell.strided_projection),
        "reduction": state_json(cell.reduction),
        "csr_spmm": state_json(cell.csr_spmm),
        "native_execution": cell.native_execution,
    })
}

fn real_workload_json(
    work: &RealWorkload,
    cpu: &RunReport<f64>,
    cuda: Option<&RunReport<f64>>,
    vendor: Option<&VendorBaseline>,
    capability: BackendKrylovCapability,
) -> Value {
    let cpu_error = real_error(&cpu.solution, &work.expected);
    let cuda_error = cuda.map(|run| real_error(&run.solution, &work.expected));
    let parity = cuda.map(|run| {
        cpu.solution
            .iter()
            .zip(&run.solution)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max)
    });
    json!({
        "id": work.id,
        "kind": work.kind,
        "method": format!("{:?}", work.method),
        "scalar": "f64",
        "dofs": N,
        "rhs_count": work.rhs_count,
        "status": if cuda.is_some() { "passed" } else { "unsupported" },
        "cpu_reference": status_json(cpu, &work.expected, cpu_error),
        "cuda": cuda.map(|run| status_json(run, &work.expected, cuda_error.unwrap_or(f64::INFINITY))),
        "vendor_baseline": vendor.map(|run| vendor_baseline_json(run, N, work.rhs_count)),
        "cpu_cuda_max_abs_parity": parity,
        "cuda_capability": capability_json(capability),
        "vendor_primitives": "cuBLAS vector/reduction/update and cuSPARSE CSR SpMV/SpMM through CudaBackend",
        "unsupported_reason": if cuda.is_none() { Some("CUDA method/scalar capability is not native on this device") } else { None::<&str> },
    })
}

fn complex_workload_json(
    work: &ComplexWorkload,
    cpu: &RunReport<Complex64>,
    cuda: Option<&RunReport<Complex64>>,
    vendor: Option<&VendorBaseline>,
    capability: BackendKrylovCapability,
) -> Value {
    let cpu_error = complex_error(&cpu.solution, &work.expected);
    let cuda_error = cuda.map(|run| complex_error(&run.solution, &work.expected));
    let parity = cuda.map(|run| {
        cpu.solution
            .iter()
            .zip(&run.solution)
            .map(|(left, right)| (*left - *right).norm())
            .fold(0.0, f64::max)
    });
    json!({
        "id": work.id,
        "kind": work.kind,
        "method": format!("{:?}", work.method),
        "scalar": "complex64",
        "dofs": N,
        "rhs_count": 1,
        "status": if cuda.is_some() { "passed" } else { "unsupported" },
        "cpu_reference": status_json(cpu, &work.expected, cpu_error),
        "cuda": cuda.map(|run| status_json(run, &work.expected, cuda_error.unwrap_or(f64::INFINITY))),
        "vendor_baseline": vendor.map(|run| vendor_baseline_json(run, N, 1)),
        "cpu_cuda_max_abs_parity": parity,
        "cuda_capability": capability_json(capability),
        "vendor_primitives": "cuBLAS complex vector/reduction/update and cuSPARSE complex CSR SpMV through CudaBackend",
        "unsupported_reason": if cuda.is_none() { Some("CUDA method/scalar capability is not native on this device") } else { None::<&str> },
    })
}

fn main() {
    let started = Instant::now();
    let result = (|| -> Result<QaResult, String> {
        let probe = required_cuda_backend()?;
        let capabilities = backend_krylov_capability_matrix(&probe);
        let capability_values: Vec<_> = capabilities.iter().copied().map(capability_json).collect();
        let workloads_real = [
            real_case(
                "spd_f64_gmres",
                "spd",
                BackendKrylovMethod::Gmres,
                3.5,
                -1.0,
                -1.0,
                1,
            ),
            real_case(
                "nonsymmetric_f64_fgmres",
                "nonsymmetric",
                BackendKrylovMethod::Fgmres,
                3.5,
                -1.2,
                0.7,
                1,
            ),
            real_case(
                "spd_block_f64_gmres",
                "block_spd",
                BackendKrylovMethod::BlockGmres,
                3.5,
                -1.0,
                -1.0,
                2,
            ),
        ];
        let workloads_complex = [complex_case()];
        let mut records = Vec::new();
        let mut native_cuda_workloads = 0_u64;

        for work in &workloads_real {
            let cpu_pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(64 << 20))
                .map_err(|error| format!("{} CPU pool: {error}", work.id))?;
            let cpu = run_real(cpu_pool, work)?;
            let capability = capability_for(&capabilities, work.method, BackendKrylovScalar::F64)
                .ok_or_else(|| format!("{} has no capability cell", work.id))?;
            let cuda = if capability.native_execution {
                native_cuda_workloads += 1;
                let cuda_pool = MemoryPool::new(
                    required_cuda_backend()?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| format!("{} CUDA pool: {error}", work.id))?;
                Some(run_real(cuda_pool, work)?)
            } else {
                None
            };
            let vendor = if capability.native_execution {
                let vendor_pool = MemoryPool::new(
                    required_cuda_backend()?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| format!("{} vendor CUDA pool: {error}", work.id))?;
                Some(run_vendor_real(vendor_pool, work)?)
            } else {
                None
            };
            if let Some(run) = &cuda {
                if run.backend != "cuda" {
                    return Err(format!(
                        "{} reported non-CUDA backend {}",
                        work.id, run.backend
                    ));
                }
                let parity = real_error(&cpu.solution, &run.solution);
                if parity > 2.0e-6 {
                    return Err(format!("{} CPU/CUDA parity {parity:e}", work.id));
                }
            }
            records.push(real_workload_json(
                work,
                &cpu,
                cuda.as_ref(),
                vendor.as_ref(),
                capability,
            ));
        }

        for work in &workloads_complex {
            let cpu_pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(64 << 20))
                .map_err(|error| format!("{} CPU pool: {error}", work.id))?;
            let cpu = run_complex(cpu_pool, work)?;
            let capability =
                capability_for(&capabilities, work.method, BackendKrylovScalar::Complex64)
                    .ok_or_else(|| format!("{} has no capability cell", work.id))?;
            let cuda = if capability.native_execution {
                native_cuda_workloads += 1;
                let cuda_pool = MemoryPool::new(
                    required_cuda_backend()?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| format!("{} CUDA pool: {error}", work.id))?;
                Some(run_complex(cuda_pool, work)?)
            } else {
                None
            };
            let vendor = if capability.native_execution {
                let vendor_pool = MemoryPool::new(
                    required_cuda_backend()?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| format!("{} vendor CUDA pool: {error}", work.id))?;
                Some(run_vendor_complex(vendor_pool, work)?)
            } else {
                None
            };
            if let Some(run) = &cuda {
                if run.backend != "cuda" {
                    return Err(format!(
                        "{} reported non-CUDA backend {}",
                        work.id, run.backend
                    ));
                }
                let parity = complex_error(&cpu.solution, &run.solution);
                if parity > 2.0e-5 {
                    return Err(format!("{} CPU/CUDA parity {parity:e}", work.id));
                }
            }
            records.push(complex_workload_json(
                work,
                &cpu,
                cuda.as_ref(),
                vendor.as_ref(),
                capability,
            ));
        }

        let artifact_value = json!({
            "schema_version": 2,
            "case_id": CASE_ID,
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "hardware": cuda_hardware_provenance(),
            "repetitions_per_steady_measurement": REPEATS,
            "capability_matrix": capability_values,
            "workloads": records,
            "measurement_contract": {
                "setup_ms": "CSR/device buffers and first warm solve are measured separately from steady solves",
                "iteration_ms_per_solve": "mean of repeated resident solves after warm-up",
                "synchronization_ms": "final checked synchronization measured after timed solver calls",
                "transfers": "steady solver, synchronization, and final readback counters are separate",
                "final_true_residual": "independently recomputed from read-back solution and host CSR reference",
                "vendor_baseline_setup_ms": "direct cuSPARSE descriptor/device-buffer setup and uploads, excluded from steady primitive time",
                "vendor_baseline_steady_ms_per_operation": "mean of repeated direct cuSPARSE operations after warm-up and before explicit synchronization",
                "vendor_baseline_numerical_gate": "direct cuBLAS norm plus independent host output residual/error gate must pass before any comparison is publishable",
            },
            "vendor_primitives": {
                "cuda": "cuBLAS and cuSPARSE selected by CudaBackend capability admission; direct vendor baseline uses cuSPARSE CSR SpMV/SpMM and cuBLAS NRM2",
                "cpu_reference": "CpuBackend accepted reference path",
            },
            "unsupported_policy": "A non-native method/scalar cell is recorded as unsupported and excluded from throughput claims.",
        });
        let bytes = serde_json::to_vec_pretty(&artifact_value)
            .map_err(|error| format!("serialize SI3 artifact: {error}"))?;
        let (path, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
        let mut result = if native_cuda_workloads == 4 {
            QaResult::passed(CASE_ID)
        } else {
            QaResult::failed(
                CASE_ID,
                format!("only {native_cuda_workloads}/4 SI3 workloads have native CUDA capability"),
            )
        };
        result = result
            .requested_backend("cuda")
            .actual_backend("cuda")
            .backend_passed("cuda")
            .reference("accepted CpuBackend reference with independent host residual")
            .reference_checksum("analytic:iterative-cuda-throughput-si3-v1")
            .dimension("64x64 SPD, nonsymmetric, complex Helmholtz, and 2-RHS block CSR workloads")
            .formulation(
                "resident Krylov steady-state throughput with separated lifecycle counters",
            )
            .metric("representative_workloads", 4.0)
            .metric("native_cuda_workloads", native_cuda_workloads as f64)
            .metric(
                "unsupported_cuda_workloads",
                (4 - native_cuda_workloads) as f64,
            )
            .metric("steady_repetitions", REPEATS as f64)
            .minimum("representative_workloads", 4.0)
            .minimum("native_cuda_workloads", 4.0)
            .artifact_with_provenance(
                path,
                "crates/sonium-qa/bin/solvers_iterative_cuda_throughput.rs",
                "native SM120 correctness-gated CUDA throughput and CPU parity evidence",
                "v1",
                checksum,
            )
            .hardware_provenance(cuda_hardware_provenance());
        Ok(result)
    })();
    match result {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("SI3 QA JSON")
            );
            println!("SUCCESS elapsed_ms={}", started.elapsed().as_millis());
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cuda")
                .actual_backend("cuda");
            println!(
                "QA_RESULT: {}",
                serde_json::to_string(&result).expect("SI3 QA failure JSON")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
