//! Required CUDA method/capability and checked-failure evidence for SI2.
//!
//! This producer deliberately keeps the matrix small and deterministic.  Each
//! supported method/scalar cell uses a two-by-two diagonal CSR problem with an
//! identity preconditioner.  Failure probes use a fresh CUDA context and arm a
//! single checked seam, so a failure cannot be mistaken for max-iterations,
//! convergence, a panic, or an implicit CPU retry.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendCsr, BackendError, CudaBackend, CudaFailureInjectionPoint,
    CudaRuntimeStatsSnapshot, DeviceOperator, DeviceScalar, IdentityDevicePreconditioner,
    MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, QaStatus, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_solvers::{
    BackendCancellationToken, BackendKrylovCapability, BackendKrylovConfig, BackendKrylovError,
    BackendKrylovFailureDiagnostic, BackendKrylovMethod, BackendKrylovScalar, BackendKrylovStatus,
    BackendKrylovWorkspace, BackendStoppingReason, backend_krylov_capability_matrix,
    try_block_fgmres_backend, try_block_gmres_backend, try_fgmres_backend_preconditioned,
    try_gmres_backend,
};
use std::collections::BTreeMap;
use std::ops::AddAssign;

const CASE_ID: &str = "solvers.iterative-cuda-failure-matrix.v1";
const ARTIFACT: &str = "solvers_iterative_cuda_failure_matrix.json";
const DIMENSION: usize = 2;
const FAILURE_POINTS: [CudaFailureInjectionPoint; 11] = [
    CudaFailureInjectionPoint::Allocation,
    CudaFailureInjectionPoint::DescriptorCreation,
    CudaFailureInjectionPoint::Launch,
    CudaFailureInjectionPoint::TransferH2D,
    CudaFailureInjectionPoint::TransferD2H,
    CudaFailureInjectionPoint::TransferD2D,
    CudaFailureInjectionPoint::Synchronization,
    CudaFailureInjectionPoint::EventRecord,
    CudaFailureInjectionPoint::EventWait,
    CudaFailureInjectionPoint::DeviceLoss,
    CudaFailureInjectionPoint::Teardown,
];

fn method_name(method: BackendKrylovMethod) -> &'static str {
    match method {
        BackendKrylovMethod::Gmres => "gmres",
        BackendKrylovMethod::Fgmres => "fgmres",
        BackendKrylovMethod::BlockGmres => "block_gmres",
        BackendKrylovMethod::BlockFgmres => "block_fgmres",
    }
}

fn scalar_name(scalar: BackendKrylovScalar) -> &'static str {
    match scalar {
        BackendKrylovScalar::F32 => "f32",
        BackendKrylovScalar::F64 => "f64",
        BackendKrylovScalar::Complex32 => "complex32",
        BackendKrylovScalar::Complex64 => "complex64",
    }
}

fn state_name(state: sonium_compute::CapabilityState) -> &'static str {
    match state {
        sonium_compute::CapabilityState::Native => "native",
        sonium_compute::CapabilityState::VendorNative => "vendor_native",
        sonium_compute::CapabilityState::Unsupported => "unsupported",
        sonium_compute::CapabilityState::HostSharedFallback => "host_shared_fallback",
        sonium_compute::CapabilityState::HostStagedFallback => "host_staged_fallback",
    }
}

fn method_is_block(method: BackendKrylovMethod) -> bool {
    matches!(
        method,
        BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres
    )
}

fn capability_json(cell: BackendKrylovCapability) -> Value {
    json!({
        "method": method_name(cell.method),
        "scalar": scalar_name(cell.scalar),
        "operator": cell.operator,
        "operator_state": state_name(cell.operator_state),
        "preconditioner": cell.preconditioner,
        "preconditioner_state": state_name(cell.preconditioner_state),
        "scalar_state": state_name(cell.scalar_state),
        "blas1": state_name(cell.blas1),
        "axpy_many": state_name(cell.axpy_many),
        "multi_dot": state_name(cell.multi_dot),
        "strided_projection": state_name(cell.strided_projection),
        "reduction": state_name(cell.reduction),
        "csr_spmm": state_name(cell.csr_spmm),
        "native_execution": cell.native_execution,
    })
}

fn unavailable_capability(method: BackendKrylovMethod, scalar: BackendKrylovScalar) -> Value {
    json!({
        "backend": "cuda",
        "method": method_name(method),
        "scalar": scalar_name(scalar),
        "operator": if method_is_block(method) { "csr_spmm" } else { "csr_spmv" },
        "operator_state": "unsupported",
        "preconditioner": "identity",
        "preconditioner_state": "unsupported",
        "scalar_state": "unsupported",
        "blas1": "unsupported",
        "axpy_many": "unsupported",
        "multi_dot": "unsupported",
        "strided_projection": "unsupported",
        "reduction": "unsupported",
        "csr_spmm": "unsupported",
        "native_execution": false,
        "unsupported_reason": "required CUDA backend is unavailable",
    })
}

fn real_f64<R: ToPrimitive>(value: R) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn status_json<T: DeviceScalar>(status: &BackendKrylovStatus<T::Real>, output_finite: bool) -> Value
where
    T::Real: ToPrimitive,
{
    json!({
        "status": if status.converged && output_finite { "passed" } else { "failed" },
        "iterations": status.iterations,
        "restarts": status.restarts,
        "projected_residual": real_f64(status.projected_residual),
        "true_residual": real_f64(status.true_residual),
        "backward_error": real_f64(status.backward_error),
        "converged": status.converged,
        "stopping_reason": format!("{:?}", status.stopping_reason),
        "partial_iterate": format!("{:?}", status.partial_iterate),
        "workspace_device_buffers": status.workspace_device_buffers,
        "workspace_device_capacity": status.workspace_device_capacity,
        "transfers": {
            "h2d_operations": status.transfers.h2d_operations,
            "h2d_bytes": status.transfers.h2d_bytes,
            "d2h_operations": status.transfers.d2h_operations,
            "d2h_bytes": status.transfers.d2h_bytes,
            "d2d_operations": status.transfers.d2d_operations,
            "d2d_bytes": status.transfers.d2d_bytes,
        },
    })
}

fn config<R: Float + FromPrimitive>() -> BackendKrylovConfig<R> {
    BackendKrylovConfig {
        max_iterations: 4,
        restart: 4,
        tolerance: R::from_f64(1.0e-5).expect("built-in real type accepts tolerance"),
        require_native: true,
        exact_residual_check_interval: 1,
    }
}

fn run_cell<T>(pool: MemoryPool<CudaBackend>, method: BackendKrylovMethod) -> Result<Value, String>
where
    T: DeviceScalar + AddAssign,
    T::Real: ToPrimitive,
{
    let backend = pool.backend();
    let matrix = BackendCsr::try_from_host(
        backend,
        DIMENSION,
        DIMENSION,
        &[0, 1, 2],
        &[0, 1],
        &[T::one(), T::one()],
    )
    .map_err(|error| format!("CSR setup: {error}"))?;
    let rhs_count = if method_is_block(method) { 2 } else { 1 };
    let rhs_host: Vec<T> = (0..DIMENSION * rhs_count)
        .map(|index| {
            T::from_real(
                T::Real::from_f64(if rhs_count == 1 || index < DIMENSION {
                    1.0
                } else {
                    2.0
                })
                .expect("fixture value"),
            )
        })
        .collect();
    let mut rhs = pool
        .try_alloc::<T>(rhs_host.len())
        .map_err(|error| format!("RHS allocation: {error}"))?;
    let mut output = pool
        .try_alloc::<T>(rhs_host.len())
        .map_err(|error| format!("output allocation: {error}"))?;
    backend
        .try_copy_h2d(&rhs_host, rhs.raw_mut())
        .map_err(|error| format!("RHS upload: {error}"))?;
    backend
        .try_copy_h2d(&vec![T::zero(); rhs_host.len()], output.raw_mut())
        .map_err(|error| format!("output initialization: {error}"))?;
    let mut workspace = BackendKrylovWorkspace::new(DIMENSION);
    let status = match method {
        BackendKrylovMethod::Gmres => try_gmres_backend(
            backend,
            &pool,
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            &config::<T::Real>(),
            None,
        ),
        BackendKrylovMethod::Fgmres => try_fgmres_backend_preconditioned(
            backend,
            &pool,
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            &config::<T::Real>(),
            None,
        ),
        BackendKrylovMethod::BlockGmres => try_block_gmres_backend(
            backend,
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            rhs_count,
            &config::<T::Real>(),
            None,
        ),
        BackendKrylovMethod::BlockFgmres => try_block_fgmres_backend(
            backend,
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            rhs_count,
            &config::<T::Real>(),
            None,
        ),
    }
    .map_err(|error| format!("{error:?}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("final synchronization: {error}"))?;
    let mut observed = vec![T::zero(); rhs_host.len()];
    backend
        .try_copy_d2h(output.raw(), &mut observed)
        .map_err(|error| format!("final readback: {error}"))?;
    let output_finite = observed.iter().all(DeviceScalar::is_finite);
    Ok(status_json::<T>(&status, output_finite))
}

/// An operator which deliberately writes a non-finite value through the
/// checked device boundary.  This is used only by the SI2 fault producer.
struct NonFiniteOperator;

impl DeviceOperator<CudaBackend, f64> for NonFiniteOperator {
    fn n(&self) -> usize {
        DIMENSION
    }

    fn apply(
        &self,
        backend: &CudaBackend,
        _x: &sonium_compute::cuda::CudaBuffer<f64>,
        y: &mut sonium_compute::cuda::CudaBuffer<f64>,
    ) {
        backend.copy_h2d(&[f64::NAN, f64::NAN], y);
    }
}

fn diagnostic_json(diagnostic: &BackendKrylovFailureDiagnostic) -> Value {
    json!({
        "method": method_name(diagnostic.method),
        "actual_backend": diagnostic.actual_backend,
        "iterations": diagnostic.iterations,
        "stopping_reason": format!("{:?}", diagnostic.stopping_reason),
        "projected_residual": diagnostic.projected_residual,
        "true_residual": diagnostic.true_residual,
        "backward_error": diagnostic.backward_error,
        "partial_iterate": format!("{:?}", diagnostic.partial_iterate),
        "context_disposition": format!("{:?}", diagnostic.context_disposition),
    })
}

fn backend_error_json(error: &BackendError) -> Value {
    json!({
        "error_kind": match error {
            BackendError::Allocation { .. } => "allocation",
            BackendError::Transfer { .. } => "transfer",
            BackendError::Launch { .. } => "launch",
            BackendError::Synchronization { .. } => "synchronization",
            BackendError::DeviceLost { .. } => "device_loss",
            BackendError::OperationFailed { .. } => "operation",
            BackendError::InvalidArgument { .. } => "invalid_argument",
            BackendError::Unsupported { .. } => "unsupported",
            BackendError::HostFallback { .. } => "host_fallback",
            BackendError::CommandBuffer { .. } => "command_buffer",
        },
        "error": error.to_string(),
        "context_disposition": format!("{:?}", error.context_disposition()),
        "partial_iterate": format!("{:?}", error.output_disposition()),
    })
}

fn direct_failure_json(point: CudaFailureInjectionPoint, error: Option<BackendError>) -> Value {
    let stage = match point {
        CudaFailureInjectionPoint::Allocation => "allocation",
        CudaFailureInjectionPoint::DescriptorCreation => "descriptor_creation",
        CudaFailureInjectionPoint::Launch => "launch",
        CudaFailureInjectionPoint::TransferH2D => "transfer_h2d",
        CudaFailureInjectionPoint::TransferD2H => "transfer_d2h",
        CudaFailureInjectionPoint::TransferD2D => "transfer_d2d",
        CudaFailureInjectionPoint::Synchronization => "synchronization",
        CudaFailureInjectionPoint::EventRecord => "event_record",
        CudaFailureInjectionPoint::EventWait => "event_wait",
        CudaFailureInjectionPoint::DeviceLoss => "device_loss",
        CudaFailureInjectionPoint::Teardown => "teardown",
    };
    match error {
        Some(error) => {
            let expected = error.context_disposition();
            let mut record = backend_error_json(&error);
            if let Value::Object(object) = &mut record {
                object.insert("stage".into(), Value::String(stage.into()));
                object.insert("expected_failure".into(), Value::Bool(true));
                object.insert("status".into(), Value::String("passed".into()));
                object.insert(
                    "diagnostic_stopping_reason".into(),
                    Value::String("Failed".into()),
                );
                object.insert("diagnostic_iterations".into(), Value::Number(0.into()));
                object.insert("diagnostic_projected_residual".into(), Value::Null);
                object.insert("diagnostic_true_residual".into(), Value::Null);
                object.insert("diagnostic_backward_error".into(), Value::Null);
                object.insert(
                    "diagnostic_actual_backend".into(),
                    Value::String("cuda".into()),
                );
                object.insert(
                    "diagnostic_context_disposition".into(),
                    Value::String(format!("{:?}", expected)),
                );
            }
            record
        }
        None => json!({
            "stage": stage,
            "status": "failed",
            "expected_failure": true,
            "error": "injection was not consumed by a checked boundary",
            "diagnostic_stopping_reason": "Failed",
            "diagnostic_iterations": 0,
            "diagnostic_projected_residual": null,
            "diagnostic_true_residual": null,
            "diagnostic_backward_error": null,
            "diagnostic_actual_backend": "cuda",
        }),
    }
}

fn setup_f64() -> Result<(MemoryPool<CudaBackend>, BackendCsr<CudaBackend, f64>), String> {
    let pool = MemoryPool::new(
        CudaBackend::new_default().map_err(|error| error.to_string())?,
        MemoryPoolConfig::with_cap(64 << 20),
    )
    .map_err(|error| error.to_string())?;
    let matrix = BackendCsr::try_from_host(
        pool.backend(),
        DIMENSION,
        DIMENSION,
        &[0, 1, 2],
        &[0, 1],
        &[1.0, 1.0],
    )
    .map_err(|error| error.to_string())?;
    Ok((pool, matrix))
}

fn runtime_resources_json(snapshot: CudaRuntimeStatsSnapshot) -> Value {
    let driver = snapshot.driver_resources;
    json!({
        "live_allocation_bytes": snapshot.live_allocation_bytes,
        "live_staging_bytes": snapshot.live_staging_bytes,
        "driver_resources": {
            "live_contexts": driver.live_contexts,
            "live_streams": driver.live_streams,
            "live_modules": driver.live_modules,
            "live_dynamic_libraries": driver.live_dynamic_libraries,
            "live_library_handles": driver.live_library_handles,
            "live_fft_plans": driver.live_fft_plans,
            "live_completion_events": driver.live_completion_events,
            "initialization_baseline_free_bytes": driver.initialization_baseline_free_bytes,
            "initialization_free_bytes": driver.initialization_free_bytes,
            "measured_driver_bytes": driver.measured_driver_bytes,
        },
    })
}

fn runtime_resources_stable(
    before: CudaRuntimeStatsSnapshot,
    after: CudaRuntimeStatsSnapshot,
) -> bool {
    before.live_allocation_bytes == after.live_allocation_bytes
        && before.live_staging_bytes == after.live_staging_bytes
        && before.driver_resources == after.driver_resources
}

fn run_injected(point: CudaFailureInjectionPoint) -> Result<Value, String> {
    let (pool, matrix) = setup_f64()?;
    let backend = pool.backend();
    let mut rhs = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    let mut output = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[1.0, 1.0], rhs.raw_mut())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[7.0, 7.0], output.raw_mut())
        .map_err(|error| error.to_string())?;

    let mut solver_error = None;
    let direct_error = match point {
        CudaFailureInjectionPoint::Synchronization => {
            backend.inject_failure(point);
            backend.try_synchronize().err()
        }
        CudaFailureInjectionPoint::EventRecord => {
            backend.inject_failure(point);
            backend.try_submit_completion().err()
        }
        CudaFailureInjectionPoint::EventWait => {
            let mut completion = backend
                .try_submit_completion()
                .map_err(|error| error.to_string())?;
            backend.inject_failure(point);
            completion.wait().err()
        }
        CudaFailureInjectionPoint::Teardown => {
            {
                let prebaked = matrix
                    .try_prebake(backend)
                    .map_err(|error| error.to_string())?;
                backend.inject_failure(point);
                drop(prebaked);
            }
            backend.pending_error().map_err(|error| error.to_string())?
        }
        CudaFailureInjectionPoint::TransferH2D
        | CudaFailureInjectionPoint::TransferD2H
        | CudaFailureInjectionPoint::TransferD2D
        | CudaFailureInjectionPoint::Allocation
        | CudaFailureInjectionPoint::DescriptorCreation
        | CudaFailureInjectionPoint::Launch
        | CudaFailureInjectionPoint::DeviceLoss => {
            backend.inject_failure(point);
            let mut workspace = BackendKrylovWorkspace::new(DIMENSION);
            match try_gmres_backend(
                backend,
                &pool,
                &mut workspace,
                &matrix,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                None,
                output.raw_mut(),
                &config::<f64>(),
                None,
            ) {
                Ok(_) => None,
                Err(error) => {
                    solver_error = Some(error);
                    None
                }
            }
        }
    };

    if let Some(error) = solver_error {
        let mut record = json!({
            "stage": format!("{:?}", point),
            "status": "passed",
            "expected_failure": true,
            "error": error.to_string(),
            "error_kind": "iterative_backend",
            "diagnostic": diagnostic_json(error.diagnostic()),
            "output_disposition": format!("{:?}", error.output_disposition()),
            "context_disposition": format!("{:?}", error.context_disposition()),
        });
        if error.diagnostic().stopping_reason != BackendStoppingReason::Failed {
            record["status"] = Value::String("failed".into());
        }
        Ok(record)
    } else {
        Ok(direct_failure_json(point, direct_error))
    }
}

fn run_capture_abort_teardown() -> Result<Value, String> {
    let (pool, _matrix) = setup_f64()?;
    let backend = pool.backend();
    let mut output = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[7.0, 7.0], output.raw_mut())
        .map_err(|error| error.to_string())?;
    // Warm cuBLAS before entering capture.  This keeps any handle/lazy
    // workspace creation outside the graph and makes the retry a pure native
    // stream operation.
    backend
        .try_scale(1.0_f64, output.raw_mut())
        .map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    let baseline = backend.runtime_stats();

    backend.inject_failure(CudaFailureInjectionPoint::Teardown);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: sonium_compute::BackendResult<_> =
            backend.try_capture_graph_checked(|_| panic!("capture abort teardown probe"));
    }));
    let panic_caught = panic_result.is_err();
    let pending = backend.pending_error().map_err(|error| error.to_string())?;
    let Some(pending) = pending else {
        return Err("capture abort teardown did not retain a typed error".into());
    };
    let typed_teardown = matches!(
        pending,
        BackendError::OperationFailed {
            operation: "cuda_teardown",
            ..
        }
    );
    let after_failure = backend.runtime_stats();
    let resources_stable_after_failure = runtime_resources_stable(baseline, after_failure);

    backend
        .recover_after_error()
        .map_err(|error| error.to_string())?;
    let pending_after_recovery = backend.pending_error().map_err(|error| error.to_string())?;
    let mut after_failure_output = [0.0; DIMENSION];
    backend
        .try_copy_d2h(output.raw(), &mut after_failure_output)
        .map_err(|error| error.to_string())?;
    let output_preserved = after_failure_output == [7.0, 7.0];

    let replay = backend
        .try_capture_graph_checked(|capture_backend| {
            capture_backend.try_scale(2.0_f64, output.raw_mut())
        })
        .map_err(|error| error.to_string())?;
    replay.try_launch().map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    drop(replay);
    let mut retry_output = [0.0; DIMENSION];
    backend
        .try_copy_d2h(output.raw(), &mut retry_output)
        .map_err(|error| error.to_string())?;
    let retry_succeeded = retry_output == [14.0, 14.0];
    let after_retry = backend.runtime_stats();
    let resources_stable_after_retry = runtime_resources_stable(baseline, after_retry);
    let recovered = pending_after_recovery.is_none();
    let status = if panic_caught
        && typed_teardown
        && output_preserved
        && resources_stable_after_failure
        && recovered
        && retry_succeeded
        && resources_stable_after_retry
    {
        "passed"
    } else {
        "failed"
    };
    Ok(json!({
        "stage": "capture_abort_teardown",
        "status": status,
        "expected_failure": true,
        "error": pending.to_string(),
        "error_kind": "operation",
        "typed_teardown": typed_teardown,
        "panic_caught": panic_caught,
        "output_preserved": output_preserved,
        "output_before_recovery": [7.0, 7.0],
        "output_after_failure": after_failure_output,
        "context_disposition": format!("{:?}", pending.context_disposition()),
        "output_disposition": format!("{:?}", pending.output_disposition()),
        "recovered": recovered,
        "retry_succeeded": retry_succeeded,
        "retry_output": retry_output,
        "resource_baseline": {
            "before": runtime_resources_json(baseline),
            "after_failure": runtime_resources_json(after_failure),
            "after_retry": runtime_resources_json(after_retry),
            "stable_after_failure": resources_stable_after_failure,
            "stable_after_retry": resources_stable_after_retry,
        },
    }))
}

fn run_cancelled() -> Result<Value, String> {
    let (pool, matrix) = setup_f64()?;
    let backend = pool.backend();
    let mut rhs = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    let mut output = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[1.0, 1.0], rhs.raw_mut())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[7.0, 7.0], output.raw_mut())
        .map_err(|error| error.to_string())?;
    let token = BackendCancellationToken::new();
    token.cancel();
    let result = try_gmres_backend(
        backend,
        &pool,
        &mut BackendKrylovWorkspace::new(DIMENSION),
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config::<f64>(),
        Some(&token),
    );
    match result {
        Err(error @ BackendKrylovError::Cancelled { .. }) => Ok(json!({
            "stage": "cancellation",
            "status": "passed",
            "expected_failure": true,
            "error": error.to_string(),
            "error_kind": "cancelled",
            "diagnostic": diagnostic_json(error.diagnostic()),
            "output_disposition": format!("{:?}", error.output_disposition()),
            "context_disposition": format!("{:?}", error.context_disposition()),
        })),
        Ok(status) => Ok(json!({
            "stage": "cancellation",
            "status": "failed",
            "expected_failure": true,
            "error": "pre-cancelled token was ignored",
            "iterations": status.iterations,
            "stopping_reason": format!("{:?}", status.stopping_reason),
        })),
        Err(error) => Ok(json!({
            "stage": "cancellation",
            "status": "failed",
            "expected_failure": true,
            "error": error.to_string(),
            "error_kind": "wrong_failure_class",
            "diagnostic": diagnostic_json(error.diagnostic()),
        })),
    }
}

fn run_nonfinite() -> Result<Value, String> {
    let (pool, _matrix) = setup_f64()?;
    let backend = pool.backend();
    let mut rhs = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    let mut output = pool
        .try_alloc::<f64>(DIMENSION)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[1.0, 1.0], rhs.raw_mut())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&[7.0, 7.0], output.raw_mut())
        .map_err(|error| error.to_string())?;
    let result = try_gmres_backend(
        backend,
        &pool,
        &mut BackendKrylovWorkspace::new(DIMENSION),
        &NonFiniteOperator,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config::<f64>(),
        None,
    );
    match result {
        Err(error) => Ok(json!({
            "stage": "nonfinite_operator_output",
            "status": "passed",
            "expected_failure": true,
            "error": error.to_string(),
            "error_kind": "nonfinite",
            "diagnostic": diagnostic_json(error.diagnostic()),
            "output_disposition": format!("{:?}", error.output_disposition()),
            "context_disposition": format!("{:?}", error.context_disposition()),
        })),
        Ok(status) => Ok(json!({
            "stage": "nonfinite_operator_output",
            "status": "failed",
            "expected_failure": true,
            "error": "non-finite operator output was accepted",
            "iterations": status.iterations,
            "stopping_reason": format!("{:?}", status.stopping_reason),
        })),
    }
}

fn run() -> Result<(QaResult, Value), String> {
    let started = std::time::Instant::now();
    let cuda = match CudaBackend::new_default() {
        Ok(cuda) => cuda,
        Err(error) => {
            let cells: Vec<Value> = [
                BackendKrylovMethod::Gmres,
                BackendKrylovMethod::Fgmres,
                BackendKrylovMethod::BlockGmres,
                BackendKrylovMethod::BlockFgmres,
            ]
            .into_iter()
            .flat_map(|method| {
                [
                    BackendKrylovScalar::F32,
                    BackendKrylovScalar::F64,
                    BackendKrylovScalar::Complex32,
                    BackendKrylovScalar::Complex64,
                ]
                .into_iter()
                .map(move |scalar| unavailable_capability(method, scalar))
            })
            .collect();
            let method_counts: BTreeMap<String, QaExecutionCounts> = [
                BackendKrylovMethod::Gmres,
                BackendKrylovMethod::Fgmres,
                BackendKrylovMethod::BlockGmres,
                BackendKrylovMethod::BlockFgmres,
            ]
            .into_iter()
            .map(|method| {
                (
                    method_name(method).to_owned(),
                    QaExecutionCounts {
                        executed: 4,
                        unsupported: 4,
                        ..QaExecutionCounts::default()
                    },
                )
            })
            .collect();
            let mut failures: Vec<Value> = FAILURE_POINTS
                .iter()
                .map(|point| {
                    json!({
                        "stage": format!("{:?}", point),
                        "status": "unsupported",
                        "expected_failure": true,
                        "reason": format!("CUDA backend unavailable: {error}"),
                    })
                })
                .collect();
            failures.push(json!({
                "stage": "cancellation",
                "status": "unsupported",
                "expected_failure": true,
                "reason": format!("CUDA backend unavailable: {error}"),
            }));
            failures.push(json!({
                "stage": "nonfinite_operator_output",
                "status": "unsupported",
                "expected_failure": true,
                "reason": format!("CUDA backend unavailable: {error}"),
            }));
            failures.push(json!({
                "stage": "capture_abort_teardown",
                "status": "unsupported",
                "expected_failure": true,
                "reason": format!("CUDA backend unavailable: {error}"),
            }));
            let artifact = json!({
                "schema_version": 1,
                "case_id": CASE_ID,
                "requested_backend": "cuda",
                "actual_backend": "unavailable",
                "hardware": cuda_hardware_provenance(),
                "capability_matrix": cells,
                "method_counts": method_counts,
                "failure_injection": failures,
                "counts": {"executed": 30, "passed": 0, "failed": 0, "unsupported": 30},
                "unsupported_reason": error.to_string(),
            });
            let bytes = serde_json::to_vec_pretty(&artifact).map_err(|error| error.to_string())?;
            let (path, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
            let result = QaResult::failed(
                CASE_ID,
                format!("required CUDA backend unavailable: {error}"),
            )
            .requested_backend("cuda")
            .actual_backend("unavailable")
            .reference("deterministic SI2 matrix and checked-failure contract")
            .reference_checksum("analytic/deterministic-fixture:iterative-si2-v1")
            .dimension(
                "4 methods × 4 scalar families; 11 checked failure seams + capture-abort teardown + cancellation/nonfinite",
            )
            .formulation("backend-resident iterative capability and failure matrix")
            .metric("matrix_cells", 16.0)
            .metric("failure_points", 14.0)
            .metric("unsupported_cells", 30.0)
            .artifact_with_provenance(
                path,
                "crates/sonium-qa/bin/solvers_iterative_cuda_failure_matrix.rs",
                "required CUDA SI2 matrix/failure artifact",
                "v1",
                checksum,
            )
            .runtime_evidence(QaRuntimeEvidence {
                failure_stage: Some(QaFailureStage::Environment),
                context_reuse: QaContextReuse::NotApplicable,
                counts: QaExecutionCounts {
                    executed: 30,
                    unsupported: 30,
                    ..QaExecutionCounts::default()
                },
                hardware: cuda_hardware_provenance(),
                ..QaRuntimeEvidence::default()
            })
            .elapsed_ms(started.elapsed().as_millis() as u64);
            return Ok((result, artifact));
        }
    };

    let capabilities = backend_krylov_capability_matrix(&cuda);
    let mut cells = Vec::new();
    let mut counts = QaExecutionCounts::default();
    let mut method_counts: BTreeMap<String, QaExecutionCounts> = BTreeMap::new();
    for cell in capabilities.iter().copied() {
        counts.executed += 1;
        let method_counter = method_counts
            .entry(method_name(cell.method).to_owned())
            .or_default();
        method_counter.executed += 1;
        let mut record = capability_json(cell);
        if !cell.native_execution
            || !cell.operator_state.is_native()
            || !cell.preconditioner_state.is_native()
        {
            record["status"] = Value::String("unsupported".into());
            record["unsupported_reason"] = Value::String("non-native capability cell".into());
            counts.unsupported += 1;
            method_counter.unsupported += 1;
            cells.push(record);
            continue;
        }
        let result = match cell.scalar {
            BackendKrylovScalar::F32 => run_cell::<f32>(
                MemoryPool::new(
                    CudaBackend::new_default().map_err(|error| error.to_string())?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| error.to_string())?,
                cell.method,
            ),
            BackendKrylovScalar::F64 => run_cell::<f64>(
                MemoryPool::new(
                    CudaBackend::new_default().map_err(|error| error.to_string())?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| error.to_string())?,
                cell.method,
            ),
            BackendKrylovScalar::Complex32 => run_cell::<Complex32>(
                MemoryPool::new(
                    CudaBackend::new_default().map_err(|error| error.to_string())?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| error.to_string())?,
                cell.method,
            ),
            BackendKrylovScalar::Complex64 => run_cell::<Complex64>(
                MemoryPool::new(
                    CudaBackend::new_default().map_err(|error| error.to_string())?,
                    MemoryPoolConfig::with_cap(64 << 20),
                )
                .map_err(|error| error.to_string())?,
                cell.method,
            ),
        };
        match result {
            Ok(status) => {
                if status["status"] == "passed" {
                    counts.passed += 1;
                    method_counter.passed += 1;
                } else {
                    counts.failed += 1;
                    method_counter.failed += 1;
                }
                record["execution"] = status;
            }
            Err(error) => {
                counts.failed += 1;
                method_counter.failed += 1;
                record["status"] = Value::String("failed".into());
                record["error"] = Value::String(error);
            }
        }
        cells.push(record);
    }

    // Fault probes use fresh contexts, which isolates terminal device-loss
    // state and records context recreation explicitly.
    let mut failures = Vec::new();
    for point in FAILURE_POINTS {
        counts.executed += 1;
        match run_injected(point) {
            Ok(record) => {
                if record["status"] == "passed" {
                    counts.passed += 1;
                } else {
                    counts.failed += 1;
                }
                failures.push(record);
            }
            Err(error) => {
                counts.failed += 1;
                failures.push(json!({
                    "stage": format!("{:?}", point),
                    "status": "failed",
                    "error": error,
                }));
            }
        }
    }
    for (stage, probe) in [
        (
            "capture_abort_teardown",
            run_capture_abort_teardown as fn() -> Result<Value, String>,
        ),
        (
            "cancellation",
            run_cancelled as fn() -> Result<Value, String>,
        ),
        (
            "nonfinite_operator_output",
            run_nonfinite as fn() -> Result<Value, String>,
        ),
    ] {
        counts.executed += 1;
        match probe() {
            Ok(record) => {
                if record["status"] == "passed" {
                    counts.passed += 1;
                } else {
                    counts.failed += 1;
                }
                failures.push(record);
            }
            Err(error) => {
                counts.failed += 1;
                failures.push(json!({
                    "stage": stage,
                    "status": "failed",
                    "expected_failure": true,
                    "error": error,
                }));
            }
        }
    }
    let artifact = json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "hardware": cuda_hardware_provenance(),
        "capability_matrix": cells,
        "method_counts": method_counts,
        "failure_injection": failures,
        "counts": {
            "executed": counts.executed,
            "passed": counts.passed,
            "failed": counts.failed,
            "unsupported": counts.unsupported,
        },
        "failure_diagnostic_contract": [
            "method", "actual_backend", "iterations", "stopping_reason",
            "projected_residual", "true_residual", "backward_error",
            "partial_iterate", "context_disposition",
        ],
        "unsupported_policy": "non-native method/scalar/operator/preconditioner cells are explicit and excluded from pass counts",
    });
    let bytes = serde_json::to_vec_pretty(&artifact).map_err(|error| error.to_string())?;
    let (path, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
    let status = if counts.failed == 0 && counts.executed > 0 {
        QaStatus::Passed
    } else {
        QaStatus::Failed
    };
    let result = match status {
        QaStatus::Passed => QaResult::passed(CASE_ID),
        _ => QaResult::failed(CASE_ID, "one or more SI2 matrix/failure probes failed"),
    }
    .requested_backend("cuda")
    .actual_backend("cuda")
    .backend_passed("cuda")
    .reference("deterministic SI2 matrix and checked-failure contract")
    .reference_checksum("analytic/deterministic-fixture:iterative-si2-v1")
    .dimension("4 methods × 4 scalar families; 11 checked failure seams + capture-abort teardown + cancellation/nonfinite")
    .formulation("backend-resident iterative capability and failure matrix")
    .metric("executed", counts.executed as f64)
    .metric("passed", counts.passed as f64)
    .metric("failed", counts.failed as f64)
    .metric("unsupported", counts.unsupported as f64)
    .minimum("executed", 1.0)
    .artifact_with_provenance(
        path,
        "crates/sonium-qa/bin/solvers_iterative_cuda_failure_matrix.rs",
        "required CUDA SI2 matrix/failure artifact",
        "v1",
        checksum,
    )
    .runtime_evidence(QaRuntimeEvidence {
        failure_stage: (counts.failed > 0).then_some(QaFailureStage::Execution),
        context_reuse: QaContextReuse::Recreated,
        counts,
        hardware: cuda_hardware_provenance(),
        memory: QaMemoryEvidence::default(),
        counters: QaRuntimeCounters::default(),
    })
    .elapsed_ms(started.elapsed().as_millis() as u64);
    Ok((result, artifact))
}

fn main() {
    match run() {
        Ok((result, _artifact)) => {
            println!("QA_RESULT: {}", result.to_json_line().expect("SI2 QA JSON"));
            if result.status != QaStatus::Passed {
                println!("FAILED");
                std::process::exit(1);
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("SI2 failure JSON")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
