//! Native CUDA setup/runtime failure and diagnostics evidence for solver
//! preconditioners.
//!
//! The producer deliberately runs the one-shot fault hooks against real CUDA
//! objects.  Setup failures are dropped before the allocation ledger is
//! sampled again; action failures are checked for typed provenance and for a
//! caller output that remains unchanged.  Device-loss is the final action in
//! the process because that fault intentionally invalidates the context.

#![cfg(feature = "cuda")]

use serde_json::{Value, json};
use sonium_compute::preconditioner::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
};
use sonium_compute::{
    Backend, BackendCsr, BackendError, ChebyshevBackend, CudaBackend, DevicePreconditioner,
    HostCsrRef, PointJacobiBackend, PolynomialJacobiBackend, PreconditionerFailureInjectionPoint,
};
use sonium_qa::{QaResult, cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use sonium_solvers::preconditioners::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-failure-cuda.v1";

fn tridiag(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

fn error_kind(error: &BackendError) -> &'static str {
    match error {
        BackendError::Allocation { .. } => "allocation",
        BackendError::Transfer { .. } => "transfer",
        BackendError::Launch { .. } => "launch",
        BackendError::Synchronization { .. } => "synchronization",
        BackendError::DeviceLost { .. } => "device_loss",
        BackendError::InvalidArgument { .. } => "invalid_argument",
        BackendError::OperationFailed { .. } => "operation_failed",
        BackendError::HostFallback { .. } => "fallback",
        BackendError::CommandBuffer { .. } => "command_buffer",
        BackendError::Unsupported { .. } => "unsupported",
    }
}

fn expected_kind(point: PreconditionerFailureInjectionPoint) -> &'static str {
    match point {
        PreconditionerFailureInjectionPoint::HierarchyUpload
        | PreconditionerFailureInjectionPoint::DiagonalUpload => "transfer",
        PreconditionerFailureInjectionPoint::WorkspaceAllocation => "allocation",
        PreconditionerFailureInjectionPoint::Smoother
        | PreconditionerFailureInjectionPoint::Launch => "launch",
        PreconditionerFailureInjectionPoint::CoarseSolve => "operation_failed",
        PreconditionerFailureInjectionPoint::Synchronization => "synchronization",
        PreconditionerFailureInjectionPoint::NonFiniteGeneration => "invalid_argument",
        PreconditionerFailureInjectionPoint::DeviceLoss => "device_loss",
    }
}

fn expected_stage(point: PreconditionerFailureInjectionPoint) -> &'static str {
    match point {
        PreconditionerFailureInjectionPoint::HierarchyUpload
        | PreconditionerFailureInjectionPoint::DiagonalUpload => "Transfer",
        PreconditionerFailureInjectionPoint::WorkspaceAllocation => "WorkspaceAllocation",
        // The backend taxonomy intentionally groups smoother and coarse-solve
        // operation failures under the dispatch boundary.  The injection
        // point remains in the artifact so the more specific cause is not
        // lost.
        PreconditionerFailureInjectionPoint::Smoother
        | PreconditionerFailureInjectionPoint::CoarseSolve
        | PreconditionerFailureInjectionPoint::Launch => "Launch",
        PreconditionerFailureInjectionPoint::Synchronization => "Synchronization",
        PreconditionerFailureInjectionPoint::NonFiniteGeneration => "NonFiniteGeneration",
        PreconditionerFailureInjectionPoint::DeviceLoss => "DeviceLoss",
    }
}

fn setup_failure<F>(
    cuda: &CudaBackend,
    name: &str,
    point: PreconditionerFailureInjectionPoint,
    setup: F,
) -> Result<Value, String>
where
    F: FnOnce() -> Result<(), BackendError>,
{
    let before = cuda.live_allocation_bytes();
    cuda.inject_preconditioner_failure(point);
    let result = setup();
    let error = result
        .err()
        .ok_or_else(|| format!("{name}/{point:?} unexpectedly succeeded"))?;
    let kind = error_kind(&error);
    if kind != expected_kind(point) {
        return Err(format!(
            "{name}/{point:?} returned {kind}, expected {}: {error}",
            expected_kind(point)
        ));
    }
    cuda.recover_after_error()
        .map_err(|recovery| format!("{name}/{point:?} cleanup recovery: {recovery}"))?;
    let after = cuda.live_allocation_bytes();
    if before != after {
        return Err(format!(
            "{name}/{point:?} leaked {after} bytes (baseline {before})"
        ));
    }
    Ok(json!({
        "family": name,
        "injection": format!("{point:?}"),
        "status": "typed_failure",
        "error_kind": kind,
        "first_failure_stage": expected_stage(point),
        "error": error.to_string(),
        "allocation_ledger_restored": true,
        "before_live_bytes": before,
        "after_live_bytes": after,
    }))
}

/// Exercise the allocation quota through a real preconditioner constructor.
///
/// The raw allocator quota probe below proves the pool-level refund contract,
/// but it cannot prove that a failed setup does not publish a partially built
/// preconditioner.  Keep this probe on the constructor path so the first
/// diagonal/workspace allocation is rejected by the live cap, then retry the
/// exact same setup after restoring an under-cap budget and verify both the
/// resident allocation and its complete teardown.
fn setup_quota_retry(
    cuda: &CudaBackend,
    diagonal: &[f64],
    total_memory: usize,
) -> Result<Value, String> {
    let before_live_bytes = cuda.live_allocation_bytes();
    let failure_cap_bytes = before_live_bytes;
    cuda.set_allocation_cap_bytes(failure_cap_bytes);
    let error = PointJacobiBackend::try_from_diagonal(cuda, diagonal)
        .err()
        .ok_or_else(|| "over-cap preconditioner setup unexpectedly succeeded".to_string())?;
    let after_failure_live_bytes = cuda.live_allocation_bytes();
    if error_kind(&error) != "allocation" {
        return Err(format!(
            "over-cap preconditioner setup returned {} instead of allocation: {error}",
            error_kind(&error)
        ));
    }
    if after_failure_live_bytes != before_live_bytes {
        return Err(format!(
            "over-cap preconditioner setup published or leaked {} bytes (baseline {})",
            after_failure_live_bytes, before_live_bytes
        ));
    }
    if total_memory <= failure_cap_bytes {
        return Err(format!(
            "CUDA retry cap {total_memory} is not under-cap relative to live baseline {failure_cap_bytes}"
        ));
    }

    cuda.set_allocation_cap_bytes(total_memory);
    let retried = PointJacobiBackend::try_from_diagonal(cuda, diagonal).map_err(|retry_error| {
        format!("under-cap preconditioner setup retry failed: {retry_error}")
    })?;
    let after_success_live_bytes = cuda.live_allocation_bytes();
    if after_success_live_bytes <= before_live_bytes {
        return Err(format!(
            "successful setup retry did not publish resident state: baseline {before_live_bytes}, after {after_success_live_bytes}"
        ));
    }
    drop(retried);
    let after_drop_live_bytes = cuda.live_allocation_bytes();
    if after_drop_live_bytes != before_live_bytes {
        return Err(format!(
            "successful setup retry leaked {} bytes after teardown (baseline {})",
            after_drop_live_bytes, before_live_bytes
        ));
    }

    Ok(json!({
        "family": "point_jacobi",
        "setup_stage": "DiagonalUpload",
        "status": "typed_failure_and_retry",
        "error_kind": "allocation",
        "first_failure_stage": "Transfer",
        "error": error.to_string(),
        "diagnostics_recorded": true,
        "partial_state_published": false,
        "allocation_ledger_restored": true,
        "retry_succeeded": true,
        "failure_cap_bytes": failure_cap_bytes,
        "retry_cap_bytes": total_memory,
        "before_live_bytes": before_live_bytes,
        "after_failure_live_bytes": after_failure_live_bytes,
        "after_success_live_bytes": after_success_live_bytes,
        "after_drop_live_bytes": after_drop_live_bytes,
    }))
}

fn runtime_failure<P>(
    cuda: &CudaBackend,
    name: &str,
    preconditioner: &P,
    point: PreconditionerFailureInjectionPoint,
    n: usize,
    retry: bool,
) -> Result<Value, String>
where
    P: DevicePreconditioner<CudaBackend, f64>,
{
    let baseline_live_bytes = cuda.live_allocation_bytes();
    let mut rhs = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    let mut output = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    cuda.copy_h2d(&vec![1.0; n], &mut rhs);
    cuda.copy_h2d(&vec![37.0; n], &mut output);
    let before_diagnostics = preconditioner.diagnostics();
    cuda.inject_preconditioner_failure(point);
    let error = preconditioner
        .try_apply(cuda, &rhs, &mut output)
        .err()
        .ok_or_else(|| format!("{name}/{point:?} unexpectedly succeeded"))?;
    let kind = error_kind(&error);
    if kind != expected_kind(point) {
        return Err(format!(
            "{name}/{point:?} returned {kind}, expected {}: {error}",
            expected_kind(point)
        ));
    }
    let mut output_host = vec![0.0; n];
    let output_unchanged = if point == PreconditionerFailureInjectionPoint::DeviceLoss {
        false
    } else {
        cuda.recover_after_error()
            .map_err(|recovery| format!("{name}/{point:?} recovery failed: {recovery}"))?;
        cuda.copy_d2h(&output, &mut output_host);
        output_host == vec![37.0; n]
    };
    if point != PreconditionerFailureInjectionPoint::DeviceLoss && !output_unchanged {
        return Err(format!(
            "{name}/{point:?} modified caller output on failure"
        ));
    }
    let after_diagnostics = preconditioner.diagnostics();
    if after_diagnostics.apply_failures != before_diagnostics.apply_failures + 1 {
        return Err(format!(
            "{name}/{point:?} diagnostics missed apply failure: before={before_diagnostics:?}, after={after_diagnostics:?}"
        ));
    }
    if after_diagnostics.first_failure_stage.is_none() {
        return Err(format!("{name}/{point:?} diagnostics missed failure stage"));
    }

    let retried = if retry {
        preconditioner
            .try_apply(cuda, &rhs, &mut output)
            .map_err(|retry_error| format!("{name}/{point:?} retry failed: {retry_error}"))?;
        cuda.try_synchronize()
            .map_err(|sync| format!("{name}/{point:?} retry synchronization failed: {sync}"))?;
        true
    } else {
        false
    };
    drop(output);
    drop(rhs);
    let live_bytes_after_teardown = cuda.live_allocation_bytes();
    if live_bytes_after_teardown != baseline_live_bytes {
        return Err(format!(
            "{name}/{point:?} runtime failure/retry leaked device allocations: baseline={baseline_live_bytes}, after_teardown={live_bytes_after_teardown}"
        ));
    }
    Ok(json!({
        "family": name,
        "injection": format!("{point:?}"),
        "status": "typed_failure",
        "error_kind": kind,
        "injected_failure_stage": expected_stage(point),
        "error": error.to_string(),
        "output_unchanged": output_unchanged,
        "diagnostics_recorded": true,
        "first_failure_stage": after_diagnostics.first_failure_stage.map(|stage| format!("{stage:?}")),
        "retry_succeeded": retried,
        "allocation_ledger_restored": true,
        "baseline_live_bytes": baseline_live_bytes,
        "live_bytes_after_teardown": live_bytes_after_teardown,
    }))
}

fn device_loss_failure<P>(
    cuda: &CudaBackend,
    name: &str,
    n: usize,
    preconditioner: P,
    baseline_live_bytes: usize,
) -> Result<Value, String>
where
    P: DevicePreconditioner<CudaBackend, f64>,
{
    let mut rhs = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    let mut output = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    cuda.copy_h2d(&vec![1.0; n], &mut rhs);
    cuda.copy_h2d(&vec![37.0; n], &mut output);
    cuda.inject_preconditioner_failure(PreconditionerFailureInjectionPoint::DeviceLoss);
    let error = preconditioner
        .try_apply(cuda, &rhs, &mut output)
        .err()
        .ok_or_else(|| format!("{name}/DeviceLoss unexpectedly succeeded"))?;
    if error_kind(&error) != "device_loss" {
        return Err(format!("{name}/DeviceLoss returned wrong kind: {error}"));
    }
    let diagnostics = preconditioner.diagnostics();
    if diagnostics.apply_failures != 1
        || diagnostics.first_failure_stage
            != Some(sonium_compute::DevicePreconditionerFailureStage::DeviceLoss)
    {
        return Err(format!(
            "{name}/DeviceLoss diagnostics mismatch: {diagnostics:?}"
        ));
    }
    let live_bytes_before_teardown = cuda.live_allocation_bytes();
    drop(output);
    drop(rhs);
    drop(preconditioner);
    let live_bytes_after_teardown = cuda.live_allocation_bytes();
    if live_bytes_after_teardown != baseline_live_bytes {
        return Err(format!(
            "{name}/DeviceLoss host allocation ledger did not return to baseline after RAII teardown: baseline={baseline_live_bytes}, before_teardown={live_bytes_before_teardown}, after_teardown={live_bytes_after_teardown}"
        ));
    }
    Ok(json!({
        "family": name,
        "injection": "DeviceLoss",
        "status": "typed_failure",
        "error_kind": "device_loss",
        "first_failure_stage": "DeviceLoss",
        "error": error.to_string(),
        "diagnostics_recorded": true,
        "output_not_read_after_terminal_failure": true,
        "host_allocation_ledger_restored": true,
        "baseline_live_bytes": baseline_live_bytes,
        "live_bytes_before_teardown": live_bytes_before_teardown,
        "live_bytes_after_teardown": live_bytes_after_teardown,
    }))
}

fn adjoint_diagnostics_for_family<P>(
    cuda: &CudaBackend,
    name: &str,
    preconditioner: &P,
    n: usize,
    transpose_supported: bool,
) -> Result<Value, String>
where
    P: DevicePreconditioner<CudaBackend, f64>,
{
    let mut rhs = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    let mut output = cuda.raw_alloc::<f64>(n).map_err(|e| e.to_string())?;
    cuda.copy_h2d(&vec![1.0; n], &mut rhs);
    cuda.copy_h2d(&vec![37.0; n], &mut output);

    let before_transpose = preconditioner.diagnostics();
    let transpose = preconditioner.try_apply_transpose(cuda, &rhs, &mut output);
    let transpose_record = if transpose_supported {
        transpose.map_err(|error| format!("{name}/transpose unexpectedly failed: {error}"))?;
        cuda.try_synchronize()
            .map_err(|error| format!("{name}/transpose synchronization: {error}"))?;
        let after = preconditioner.diagnostics();
        if after.transpose_count != before_transpose.transpose_count + 1 {
            return Err(format!(
                "{name}/transpose diagnostics count mismatch: before={before_transpose:?}, after={after:?}"
            ));
        }
        json!({
            "status": "passed",
            "capability": "Exact",
            "diagnostics": {
                "transpose_count": after.transpose_count,
                "transpose_failures": after.transpose_failures,
            },
        })
    } else {
        let error = transpose
            .err()
            .ok_or_else(|| format!("{name}/transpose unexpectedly succeeded"))?;
        if !matches!(error, BackendError::Unsupported { .. }) {
            return Err(format!("{name}/transpose returned wrong error: {error}"));
        }
        let mut output_host = vec![0.0; n];
        cuda.copy_d2h(&output, &mut output_host);
        if output_host != vec![37.0; n] {
            return Err(format!(
                "{name}/transpose changed output on unsupported action"
            ));
        }
        let after = preconditioner.diagnostics();
        if after.transpose_failures != before_transpose.transpose_failures + 1 {
            return Err(format!(
                "{name}/transpose diagnostics failure count mismatch: before={before_transpose:?}, after={after:?}"
            ));
        }
        json!({
            "status": "unsupported",
            "capability": "Unsupported",
            "error_kind": error_kind(&error),
            "error": error.to_string(),
            "output_unchanged": true,
            "diagnostics": {
                "transpose_count": after.transpose_count,
                "transpose_failures": after.transpose_failures,
            },
        })
    };

    // Point-Jacobi's exact transpose intentionally writes the caller output;
    // reset the sentinel before testing the unsupported Hermitian path so the
    // transactionality assertion is local to that attempted action.
    cuda.copy_h2d(&vec![37.0; n], &mut output);
    let before_hermitian = preconditioner.diagnostics();
    let hermitian = preconditioner.try_apply_hermitian(cuda, &rhs, &mut output);
    let error = hermitian
        .err()
        .ok_or_else(|| format!("{name}/hermitian unexpectedly succeeded"))?;
    if !matches!(error, BackendError::Unsupported { .. }) {
        return Err(format!("{name}/hermitian returned wrong error: {error}"));
    }
    let mut output_host = vec![0.0; n];
    cuda.copy_d2h(&output, &mut output_host);
    if output_host != vec![37.0; n] {
        return Err(format!(
            "{name}/hermitian changed output on unsupported action"
        ));
    }
    let after_hermitian = preconditioner.diagnostics();
    if after_hermitian.hermitian_failures != before_hermitian.hermitian_failures + 1 {
        return Err(format!(
            "{name}/hermitian diagnostics failure count mismatch: before={before_hermitian:?}, after={after_hermitian:?}"
        ));
    }
    Ok(json!({
        "family": name,
        "transpose": transpose_record,
        "hermitian": {
            "status": "unsupported",
            "capability": "Unsupported",
            "error_kind": error_kind(&error),
            "error": error.to_string(),
            "output_unchanged": true,
            "diagnostics": {
                "hermitian_count": after_hermitian.hermitian_count,
                "hermitian_failures": after_hermitian.hermitian_failures,
            },
        },
    }))
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let cuda = required_cuda_backend()?;
    let (row_ptr, col_idx, values) = tridiag(9);
    let a_cuda = BackendCsr::try_from_host(&cuda, 9, 9, &row_ptr, &col_idx, &values)
        .map_err(|e| format!("fixture upload failed: {e}"))?;
    let amg_matrix = CsrMatrix::from_raw_parts(
        9,
        9,
        row_ptr.iter().map(|index| *index as usize).collect(),
        col_idx.iter().map(|index| *index as usize).collect(),
        values.clone(),
    );
    let amg_config = AmgConfig {
        coarse_size: 2,
        max_levels: 8,
        num_pre_smooth: 1,
        num_post_smooth: 1,
        smoother: AmgSmoother::Jacobi,
        ..AmgConfig::default()
    };
    let total_memory = cuda
        .total_device_memory_bytes()
        .ok_or_else(|| "CUDA total memory is unavailable for quota test".to_string())?;
    let mut setup_failures = Vec::new();
    let diagonal = [2.0_f64; 9];
    setup_failures.push(setup_failure(
        &cuda,
        "point_jacobi",
        PreconditionerFailureInjectionPoint::DiagonalUpload,
        || PointJacobiBackend::try_from_diagonal(&cuda, &diagonal).map(|_| ()),
    )?);
    setup_failures.push(setup_failure(
        &cuda,
        "point_jacobi",
        PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        || PointJacobiBackend::try_from_diagonal(&cuda, &diagonal).map(|_| ()),
    )?);

    for point in [
        PreconditionerFailureInjectionPoint::HierarchyUpload,
        PreconditionerFailureInjectionPoint::DiagonalUpload,
        PreconditionerFailureInjectionPoint::WorkspaceAllocation,
    ] {
        setup_failures.push(setup_failure(&cuda, "polynomial_jacobi", point, || {
            PolynomialJacobiBackend::try_from_csr_host(
                &cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 0.75,
            )
            .map(|_| ())
        })?);
        setup_failures.push(setup_failure(&cuda, "chebyshev", point, || {
            ChebyshevBackend::try_from_csr_host(
                &cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 30.0,
            )
            .map(|_| ())
        })?);
        setup_failures.push(setup_failure(&cuda, "amg", point, || {
            BackendAmgPreconditioner::<CudaBackend, f64>::try_from_csr(
                &cuda,
                &amg_matrix,
                amg_config.clone(),
            )
            .map(|_| ())
        })?);
    }

    let saddle_row_ptr = vec![0, 3, 6, 7, 8];
    let saddle_col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
    let saddle_values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
    let schur_row_ptr = vec![0_usize, 2, 4];
    let schur_col_idx = vec![0_usize, 1, 0, 1];
    let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
    let saddle_a = BackendCsr::try_from_host(
        &cuda,
        4,
        4,
        &saddle_row_ptr,
        &saddle_col_idx,
        &saddle_values,
    )
    .map_err(|e| format!("fixture upload failed: {e}"))?;
    let saddle_config = SaddlePointJacobiSchurBackendConfig {
        velocity_block_size: 2,
        velocity_degree: 2,
        velocity_damping: 0.75,
        pressure_degree: 2,
        pressure_damping: 0.75,
        pressure_kind: PressureSchurBackendKind::Polynomial,
        pressure_chebyshev_degree: 3,
        pressure_chebyshev_eigenvalue_ratio: 30.0,
    };
    for point in [
        PreconditionerFailureInjectionPoint::HierarchyUpload,
        PreconditionerFailureInjectionPoint::DiagonalUpload,
        PreconditionerFailureInjectionPoint::WorkspaceAllocation,
    ] {
        setup_failures.push(setup_failure(&cuda, "saddle_composed", point, || {
            SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
                &cuda,
                &saddle_a,
                &saddle_row_ptr,
                &saddle_col_idx,
                &saddle_values,
                HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
                saddle_config,
            )
            .map(|_| ())
        })?);
    }

    let complex_values = values
        .iter()
        .map(|value| num_complex::Complex32::new(*value as f32, 0.0))
        .collect::<Vec<_>>();
    let complex_matrix = CsrMatrix::from_raw_parts(
        9,
        9,
        row_ptr.iter().map(|index| *index as usize).collect(),
        col_idx.iter().map(|index| *index as usize).collect(),
        complex_values,
    );
    setup_failures.push(setup_failure(
        &cuda,
        "amg_cuda_coarse_lu",
        PreconditionerFailureInjectionPoint::CoarseSolve,
        || {
            BackendAmgPreconditioner::<CudaBackend, num_complex::Complex32>::try_from_csr_with_cuda_coarse_lu(
                &cuda,
                &complex_matrix,
                amg_config.clone(),
            )
            .map(|_| ())
        },
    )?);

    let setup_quota_retry = setup_quota_retry(&cuda, &diagonal, total_memory)?;

    let point = PointJacobiBackend::try_from_diagonal(&cuda, &diagonal)
        .map_err(|e| format!("point setup: {e}"))?;
    let polynomial = PolynomialJacobiBackend::try_from_csr_host(
        &cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 0.75,
    )
    .map_err(|e| format!("polynomial setup: {e}"))?;
    let chebyshev =
        ChebyshevBackend::try_from_csr_host(&cuda, &a_cuda, &row_ptr, &col_idx, &values, 3, 30.0)
            .map_err(|e| format!("Chebyshev setup: {e}"))?;
    let amg = BackendAmgPreconditioner::<CudaBackend, f64>::try_from_csr(
        &cuda,
        &amg_matrix,
        amg_config.clone(),
    )
    .map_err(|e| format!("AMG setup: {e}"))?;
    let saddle = SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
        &cuda,
        &saddle_a,
        &saddle_row_ptr,
        &saddle_col_idx,
        &saddle_values,
        HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
        saddle_config,
    )
    .map_err(|e| format!("saddle setup: {e}"))?;

    let mut runtime_failures = Vec::new();
    runtime_failures.push(runtime_failure(
        &cuda,
        "point_jacobi",
        &point,
        PreconditionerFailureInjectionPoint::Launch,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "point_jacobi",
        &point,
        PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "polynomial_jacobi",
        &polynomial,
        PreconditionerFailureInjectionPoint::Launch,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "polynomial_jacobi",
        &polynomial,
        PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "chebyshev",
        &chebyshev,
        PreconditionerFailureInjectionPoint::Launch,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "chebyshev",
        &chebyshev,
        PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "amg",
        &amg,
        PreconditionerFailureInjectionPoint::Launch,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "amg",
        &amg,
        PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "amg",
        &amg,
        PreconditionerFailureInjectionPoint::Smoother,
        9,
        true,
    )?);
    runtime_failures.push(runtime_failure(
        &cuda,
        "saddle_composed",
        &saddle,
        PreconditionerFailureInjectionPoint::Launch,
        4,
        true,
    )?);

    let adjoint_diagnostics = vec![
        adjoint_diagnostics_for_family(&cuda, "point_jacobi", &point, 9, true)?,
        adjoint_diagnostics_for_family(&cuda, "polynomial_jacobi", &polynomial, 9, false)?,
        adjoint_diagnostics_for_family(&cuda, "chebyshev", &chebyshev, 9, false)?,
        adjoint_diagnostics_for_family(&cuda, "amg", &amg, 9, false)?,
        adjoint_diagnostics_for_family(&cuda, "saddle_composed", &saddle, 4, false)?,
    ];

    let synchronization_cuda = required_cuda_backend()?;
    synchronization_cuda
        .inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Synchronization);
    let synchronization_error = synchronization_cuda
        .try_synchronize()
        .err()
        .ok_or_else(|| "synchronization injection unexpectedly succeeded".to_string())?;
    if error_kind(&synchronization_error) != "synchronization" {
        return Err(format!(
            "synchronization injection returned wrong kind: {synchronization_error}"
        ));
    }
    let synchronization = json!({
        "status": "typed_failure",
        "error_kind": error_kind(&synchronization_error),
        "first_failure_stage": expected_stage(
            PreconditionerFailureInjectionPoint::Synchronization
        ),
        "error": synchronization_error.to_string(),
    });

    let live_before_quota = cuda.live_allocation_bytes();
    cuda.set_allocation_cap_bytes(live_before_quota);
    let quota_error = cuda
        .raw_alloc::<f64>(1)
        .err()
        .ok_or_else(|| "quota allocation unexpectedly succeeded".to_string())?;
    cuda.set_allocation_cap_bytes(total_memory);
    let retry = cuda
        .raw_alloc::<f64>(1)
        .map_err(|e| format!("quota retry failed after restoring cap: {e}"))?;
    drop(retry);
    let quota = json!({
        "status": "typed_failure_and_retry",
        "error_kind": "allocation",
        "first_failure_stage": expected_stage(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation
        ),
        "error": quota_error.to_string(),
        "live_bytes_before": live_before_quota,
        "retry_succeeded": true,
    });

    // Device loss is exercised on fresh contexts: the backend context is no
    // longer a valid place to issue cleanup or retry commands after a cell.
    let loss_point_cuda = required_cuda_backend()?;
    let loss_point_baseline = loss_point_cuda.live_allocation_bytes();
    let loss_point = PointJacobiBackend::try_from_diagonal(&loss_point_cuda, &diagonal)
        .map_err(|e| format!("point setup: {e}"))?;
    let loss_poly_cuda = required_cuda_backend()?;
    let loss_poly_baseline = loss_poly_cuda.live_allocation_bytes();
    let loss_poly_a = BackendCsr::try_from_host(&loss_poly_cuda, 9, 9, &row_ptr, &col_idx, &values)
        .map_err(|e| format!("fixture upload failed: {e}"))?;
    let loss_poly_retained_baseline = loss_poly_cuda.live_allocation_bytes();
    let loss_poly = PolynomialJacobiBackend::try_from_csr_host(
        &loss_poly_cuda,
        &loss_poly_a,
        &row_ptr,
        &col_idx,
        &values,
        3,
        0.75,
    )
    .map_err(|e| format!("polynomial setup: {e}"))?;
    let loss_cheby_cuda = required_cuda_backend()?;
    let loss_cheby_baseline = loss_cheby_cuda.live_allocation_bytes();
    let loss_cheby_a =
        BackendCsr::try_from_host(&loss_cheby_cuda, 9, 9, &row_ptr, &col_idx, &values)
            .map_err(|e| format!("fixture upload failed: {e}"))?;
    let loss_cheby_retained_baseline = loss_cheby_cuda.live_allocation_bytes();
    let loss_cheby = ChebyshevBackend::try_from_csr_host(
        &loss_cheby_cuda,
        &loss_cheby_a,
        &row_ptr,
        &col_idx,
        &values,
        3,
        30.0,
    )
    .map_err(|e| format!("Chebyshev setup: {e}"))?;
    let loss_amg_cuda = required_cuda_backend()?;
    let loss_amg_baseline = loss_amg_cuda.live_allocation_bytes();
    let loss_amg = BackendAmgPreconditioner::<CudaBackend, f64>::try_from_csr(
        &loss_amg_cuda,
        &amg_matrix,
        amg_config.clone(),
    )
    .map_err(|e| format!("AMG setup: {e}"))?;
    let loss_saddle_cuda = required_cuda_backend()?;
    let loss_saddle_baseline = loss_saddle_cuda.live_allocation_bytes();
    let loss_saddle_a = BackendCsr::try_from_host(
        &loss_saddle_cuda,
        4,
        4,
        &saddle_row_ptr,
        &saddle_col_idx,
        &saddle_values,
    )
    .map_err(|e| format!("fixture upload failed: {e}"))?;
    let loss_saddle_retained_baseline = loss_saddle_cuda.live_allocation_bytes();
    let loss_saddle = SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
        &loss_saddle_cuda,
        &loss_saddle_a,
        &saddle_row_ptr,
        &saddle_col_idx,
        &saddle_values,
        HostCsrRef::new(2, 2, &schur_row_ptr, &schur_col_idx, &schur_values),
        saddle_config,
    )
    .map_err(|e| format!("saddle setup: {e}"))?;
    let device_loss_failures = vec![
        device_loss_failure(
            &loss_point_cuda,
            "point_jacobi",
            9,
            loss_point,
            loss_point_baseline,
        )?,
        device_loss_failure(
            &loss_poly_cuda,
            "polynomial_jacobi",
            9,
            loss_poly,
            loss_poly_retained_baseline,
        )?,
        device_loss_failure(
            &loss_cheby_cuda,
            "chebyshev",
            9,
            loss_cheby,
            loss_cheby_retained_baseline,
        )?,
        device_loss_failure(&loss_amg_cuda, "amg", 9, loss_amg, loss_amg_baseline)?,
        device_loss_failure(
            &loss_saddle_cuda,
            "saddle_composed",
            4,
            loss_saddle,
            loss_saddle_retained_baseline,
        )?,
    ];
    drop(loss_poly_a);
    drop(loss_cheby_a);
    drop(loss_saddle_a);
    let device_loss_context_teardown = [
        ("point_jacobi", &loss_point_cuda, loss_point_baseline),
        ("polynomial_jacobi", &loss_poly_cuda, loss_poly_baseline),
        ("chebyshev", &loss_cheby_cuda, loss_cheby_baseline),
        ("amg", &loss_amg_cuda, loss_amg_baseline),
        ("saddle_composed", &loss_saddle_cuda, loss_saddle_baseline),
    ]
    .into_iter()
    .map(|(family, backend, baseline_live_bytes)| {
        let live_bytes_after_teardown = backend.live_allocation_bytes();
        if live_bytes_after_teardown != baseline_live_bytes {
            return Err(format!(
                "{family}/DeviceLoss fresh context ledger did not return to baseline: baseline={baseline_live_bytes}, after_teardown={live_bytes_after_teardown}"
            ));
        }
        Ok(json!({
            "family": family,
            "host_allocation_ledger_restored": true,
            "baseline_live_bytes": baseline_live_bytes,
            "live_bytes_after_teardown": live_bytes_after_teardown,
        }))
    })
    .collect::<Result<Vec<_>, String>>()?;
    let artifact_value = json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "hardware": cuda_hardware_provenance(),
        "setup_failures": setup_failures,
        "runtime_failures": runtime_failures,
        "adjoint_diagnostics": adjoint_diagnostics,
        "synchronization_failure": synchronization,
        "quota_failure": quota,
        "setup_quota_retry": setup_quota_retry,
        "device_loss_failures": device_loss_failures,
        "device_loss_context_teardown": device_loss_context_teardown,
        "acceptance": {
            "all_setup_faults_typed": true,
            "setup_failure_allocation_ledger_restored": true,
            "runtime_faults_typed": true,
            "runtime_diagnostics_recorded": true,
            "adjoint_counters_recorded": true,
            "unsupported_adjoints_transactional": true,
            "transactional_reusable_fault_outputs": true,
            "runtime_failure_allocation_ledger_restored": true,
            "device_loss_output_not_read": true,
            "device_loss_host_allocation_ledger_restored": true,
            "retry_after_reusable_fault": true,
            "synchronization_fault_typed": true,
            "quota_retry_restored": true,
            "preconditioner_setup_quota_retry_transactional": true,
        },
    });
    let bytes = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|e| format!("serialize CUDA preconditioner failure artifact: {e}"))?;
    let (path, checksum) = write_qa_artifact("solvers_preconditioner_failure_cuda.json", &bytes)?;

    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("typed native CUDA preconditioner failure and diagnostics matrix")
        .reference_checksum("analytic:preconditioner-failure-cuda-v1")
        .dimension("Point, Polynomial, Chebyshev, AMG, saddle composition; f64 native runtime and Complex32 AMG coarse setup")
        .formulation("native CUDA setup rollback, transactional action failure, diagnostics, and quota retry")
        .tolerance("typed error kinds; unchanged caller output; allocation ledger restored after failed setup")
        .metric("setup_failure_cells", setup_failures.len() as f64)
        .metric(
            "runtime_failure_cells",
            runtime_failures.len() as f64 + device_loss_failures.len() as f64,
        )
        .metric("reusable_failure_retries", 10.0)
        .metric("synchronization_failure_cells", 1.0)
        .metric("quota_retry_cells", 1.0)
        .metric("preconditioner_setup_quota_retry_cells", 1.0)
        .minimum("setup_failure_cells", 15.0)
        .minimum("runtime_failure_cells", 15.0)
        .minimum("reusable_failure_retries", 10.0)
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/solvers_preconditioner_failure_cuda.rs",
            "native CUDA typed failure, diagnostics, transactionality, cleanup, and quota evidence",
            "v1",
            checksum,
        );
    let mut hardware = cuda_hardware_provenance();
    hardware.numa_policy = Some("CUDA ordinal 0; failure matrix uses one native context".into());
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
