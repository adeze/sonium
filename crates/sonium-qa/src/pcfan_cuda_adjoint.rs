//! Bounded native CUDA paired-CSR adjoint action used by the PF2 contract.
//!
//! This is deliberately a sparse algebra seam: it proves that the production
//! CUDA CSR operator can execute forward and Hermitian/transpose actions while
//! resident, but does not represent a PC-fan FEM/CFD design derivative.

use serde_json::{Value, json};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendContextDisposition, BackendError, CudaBackend,
    CudaFailureInjectionPoint, DeviceAdjointOperator, DeviceOperator, HostCsrRef, MemoryPool,
    MemoryPoolConfig,
};
use sonium_sim_pcfan::{
    PcFanResolvedFlowNewtonLinearization, PcFanResolvedFlowPicardLinearization,
};

const ROW_PTR: [i32; 4] = [0, 2, 5, 7];
const COL_IDX: [i32; 7] = [0, 1, 0, 1, 2, 1, 2];
const HOST_ROW_PTR: [usize; 4] = [0, 2, 5, 7];
const HOST_COL_IDX: [usize; 7] = [0, 1, 0, 1, 2, 1, 2];
const VALUES: [f64; 7] = [2.0, 0.3, -0.2, 1.7, 0.4, 0.5, 1.2];
const X: [f64; 3] = [0.7, -0.4, 1.1];
const Y: [f64; 3] = [-0.2, 0.6, -0.8];
const MAX_ERROR: f64 = 2.0e-12;

fn forward(values: &[f64; 7], input: &[f64; 3]) -> [f64; 3] {
    let mut output = [0.0; 3];
    for row in 0..3 {
        let start = ROW_PTR[row] as usize;
        let end = ROW_PTR[row + 1] as usize;
        output[row] = (start..end)
            .map(|entry| values[entry] * input[COL_IDX[entry] as usize])
            .sum();
    }
    output
}

fn transpose(values: &[f64; 7], input: &[f64; 3]) -> [f64; 3] {
    let mut output = [0.0; 3];
    for row in 0..3 {
        let start = ROW_PTR[row] as usize;
        let end = ROW_PTR[row + 1] as usize;
        for entry in start..end {
            output[COL_IDX[entry] as usize] += values[entry] * input[row];
        }
    }
    output
}

fn max_abs_diff(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn backend_operation(error: &BackendError) -> String {
    match error {
        BackendError::Unsupported { operation, .. }
        | BackendError::HostFallback { operation, .. }
        | BackendError::InvalidArgument { operation, .. }
        | BackendError::OperationFailed { operation, .. }
        | BackendError::DeviceLost { operation, .. }
        | BackendError::CommandBuffer { operation, .. }
        | BackendError::Launch { operation, .. }
        | BackendError::Synchronization { operation, .. }
        | BackendError::Transfer { operation, .. } => (*operation).to_owned(),
        BackendError::Allocation { .. } => "cuda_allocation".to_owned(),
    }
}

fn recovery_label(disposition: BackendContextDisposition) -> &'static str {
    match disposition {
        BackendContextDisposition::Reusable => "reusable",
        BackendContextDisposition::SynchronizeBeforeReuse => "synchronize_before_reuse",
        BackendContextDisposition::MustRecreate => "recreate",
    }
}

fn sparse_forward(
    dimension: usize,
    row_ptrs: &[usize],
    column_indices: &[usize],
    values: &[f64],
    input: &[f64],
) -> Vec<f64> {
    let mut output = vec![0.0; dimension];
    for row in 0..dimension {
        for entry in row_ptrs[row]..row_ptrs[row + 1] {
            output[row] += values[entry] * input[column_indices[entry]];
        }
    }
    output
}

fn sparse_transpose(
    dimension: usize,
    row_ptrs: &[usize],
    column_indices: &[usize],
    values: &[f64],
    input: &[f64],
) -> Vec<f64> {
    let mut output = vec![0.0; dimension];
    for row in 0..dimension {
        for entry in row_ptrs[row]..row_ptrs[row + 1] {
            output[column_indices[entry]] += values[entry] * input[row];
        }
    }
    output
}

fn max_abs_diff_vec(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn dot_vec(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// Execute the last production Picard/Oseen flow matrix on CUDA and its
/// transpose action.  This deliberately proves a production flow algebra
/// boundary only: the matrix is not promoted to the exact Newton/design
/// residual needed by the full resolved-flow adjoint.
pub fn production_picard_flow_transpose_probe(
    picard: &PcFanResolvedFlowPicardLinearization,
) -> Result<Value, String> {
    picard
        .validate()
        .map_err(|error| format!("production Picard linearization is invalid: {error}"))?;
    let matrix = &picard.state_jacobian;
    let dimension = matrix.dimension;
    if dimension == 0 {
        return Err("production Picard flow matrix is empty".into());
    }
    let input: Vec<f64> = (0..dimension)
        .map(|index| 0.25 + 0.001 * (index % 17) as f64)
        .collect();
    let adjoint_input: Vec<f64> = (0..dimension)
        .map(|index| -0.15 + 0.002 * (index % 23) as f64)
        .collect();
    let cpu_forward = sparse_forward(
        dimension,
        &matrix.row_ptrs,
        &matrix.column_indices,
        &matrix.values,
        &input,
    );
    let cpu_transpose = sparse_transpose(
        dimension,
        &matrix.row_ptrs,
        &matrix.column_indices,
        &matrix.values,
        &adjoint_input,
    );
    let cpu_identity_residual =
        (dot_vec(&cpu_forward, &adjoint_input) - dot_vec(&input, &cpu_transpose)).abs();

    let backend = CudaBackend::new_default()
        .map_err(|error| format!("production Picard CUDA backend unavailable: {error}"))?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(128 << 20))
        .map_err(|error| format!("production Picard CUDA pool construction failed: {error}"))?;
    let backend = pool.backend();
    let host = HostCsrRef::try_new(
        dimension,
        dimension,
        &matrix.row_ptrs,
        &matrix.column_indices,
        &matrix.values,
    )
    .map_err(|error| format!("production Picard CSR admission failed: {error}"))?;
    let operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("production Picard CSR upload failed: {error}"))?;
    let prebaked = operator
        .try_prebake(backend)
        .map_err(|error| format!("production Picard CSR prebake failed: {error}"))?;
    let mut input_device = pool
        .try_alloc::<f64>(dimension)
        .map_err(|error| format!("production Picard input allocation failed: {error}"))?;
    let mut adjoint_input_device = pool
        .try_alloc::<f64>(dimension)
        .map_err(|error| format!("production Picard adjoint-input allocation failed: {error}"))?;
    let mut forward_output = pool
        .try_alloc::<f64>(dimension)
        .map_err(|error| format!("production Picard forward-output allocation failed: {error}"))?;
    let mut transpose_output = pool.try_alloc::<f64>(dimension).map_err(|error| {
        format!("production Picard transpose-output allocation failed: {error}")
    })?;
    backend
        .try_copy_h2d(&input, input_device.raw_mut())
        .map_err(|error| format!("production Picard input upload failed: {error}"))?;
    backend
        .try_copy_h2d(&adjoint_input, adjoint_input_device.raw_mut())
        .map_err(|error| format!("production Picard adjoint-input upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("production Picard setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();
    let warm_start = backend.begin_runtime_stats();
    for _ in 0..2 {
        prebaked
            .try_apply(backend, input_device.raw(), forward_output.raw_mut())
            .map_err(|error| format!("production Picard forward action failed: {error}"))?;
        prebaked
            .try_apply_hermitian(
                backend,
                adjoint_input_device.raw(),
                transpose_output.raw_mut(),
            )
            .map_err(|error| format!("production Picard transpose action failed: {error}"))?;
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("production Picard warm synchronization failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_start);
    let mut cuda_forward = vec![0.0; dimension];
    let mut cuda_transpose = vec![0.0; dimension];
    backend
        .try_copy_d2h(forward_output.raw(), &mut cuda_forward)
        .map_err(|error| format!("production Picard forward download failed: {error}"))?;
    backend
        .try_copy_d2h(transpose_output.raw(), &mut cuda_transpose)
        .map_err(|error| format!("production Picard transpose download failed: {error}"))?;
    let forward_residual = max_abs_diff_vec(&cuda_forward, &cpu_forward);
    let transpose_residual = max_abs_diff_vec(&cuda_transpose, &cpu_transpose);
    let cuda_identity_residual =
        (dot_vec(&cuda_forward, &adjoint_input) - dot_vec(&input, &cuda_transpose)).abs();
    if forward_residual > MAX_ERROR
        || transpose_residual > MAX_ERROR
        || cpu_identity_residual > MAX_ERROR
        || cuda_identity_residual > MAX_ERROR
    {
        return Err(format!(
            "production Picard transpose residual budget failed: forward={forward_residual:.3e}, transpose={transpose_residual:.3e}, cpu_identity={cpu_identity_residual:.3e}, cuda_identity={cuda_identity_residual:.3e}"
        ));
    }
    if warm_stats.allocation_operations != 0
        || warm_stats.staging_allocation_operations != 0
        || warm_stats.transfer.h2d_operations != 0
    {
        return Err(format!(
            "production Picard warm path allocated/uploaded: device={} staging={} h2d={}",
            warm_stats.allocation_operations,
            warm_stats.staging_allocation_operations,
            warm_stats.transfer.h2d_operations
        ));
    }

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = prebaked.try_apply(backend, input_device.raw(), forward_output.raw_mut());
    let failure_error = failure.err();
    let recovery = if failure_error.is_some() {
        backend.clear_failure();
        backend.recover_after_error()
    } else {
        Ok(())
    };
    let recovery_succeeded = recovery.is_ok();
    let mut preserved = vec![0.0; dimension];
    backend
        .try_copy_d2h(forward_output.raw(), &mut preserved)
        .map_err(|error| format!("production Picard preserved-output download failed: {error}"))?;
    let preserved_residual = max_abs_diff_vec(&preserved, &cuda_forward);
    let retry = prebaked
        .try_apply(backend, input_device.raw(), forward_output.raw_mut())
        .and_then(|()| backend.try_synchronize());
    let retry_succeeded = retry.is_ok();
    let mut retried = vec![0.0; dimension];
    if retry_succeeded {
        backend
            .try_copy_d2h(forward_output.raw(), &mut retried)
            .map_err(|error| format!("production Picard retry download failed: {error}"))?;
    }
    let retry_residual = max_abs_diff_vec(&retried, &cuda_forward);
    let typed_launch_failure = matches!(failure_error, Some(BackendError::Launch { .. }));
    if !typed_launch_failure
        || !recovery_succeeded
        || preserved_residual > MAX_ERROR
        || !retry_succeeded
        || retry_residual > MAX_ERROR
    {
        return Err(format!(
            "production Picard failure contract failed: typed={typed_launch_failure}, recovery={recovery_succeeded}, preserved={preserved_residual:.3e}, retry={retry_succeeded}, retry_residual={retry_residual:.3e}"
        ));
    }
    let failure_operation = failure_error
        .as_ref()
        .map(backend_operation)
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let failure_typed_error = failure_error
        .as_ref()
        .map(|error| format!("{error:?}"))
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let context_recovery = failure_error
        .as_ref()
        .map(|error| recovery_label(error.context_disposition()))
        .unwrap_or("unknown");
    let warm_kernel_and_library = warm_stats.kernel_launches + warm_stats.library_operations;

    drop(prebaked);
    drop(operator);
    drop(input_device);
    drop(adjoint_input_device);
    drop(forward_output);
    drop(transpose_output);
    let cleanup_stats = backend.runtime_stats();
    let resources_released = cleanup_stats.live_allocation_bytes == 0
        && cleanup_stats.live_staging_bytes == 0
        && pool.bytes_in_use() == 0;

    Ok(json!({
        "status": "passed",
        "backend": "Cuda",
        "execution_mode": "DeviceResident",
        "dispatch": "native-cuda-cusparse",
        "operator": "production PcFanResolvedFlowPicardLinearization.state_jacobian",
        "flow_model": format!("{:?}", picard.flow_model),
        "provenance": picard.provenance,
        "dimension": dimension,
        "nnz": matrix.values.len(),
        "repeated_actions": 2,
        "cpu_reference": {
            "forward_l2": dot_vec(&cpu_forward, &cpu_forward).sqrt(),
            "transpose_l2": dot_vec(&cpu_transpose, &cpu_transpose).sqrt(),
            "inner_product_residual": cpu_identity_residual,
        },
        "cuda_observation": {
            "forward_residual_max_abs": forward_residual,
            "transpose_residual_max_abs": transpose_residual,
            "inner_product_residual": cuda_identity_residual,
        },
        "warm_counters": {
            "device_allocations": warm_stats.allocation_operations,
            "staging_allocations": warm_stats.staging_allocation_operations,
            "h2d_operations": warm_stats.transfer.h2d_operations,
            "d2h_operations": warm_stats.transfer.d2h_operations,
            "synchronizations": warm_stats.synchronization_calls,
            "kernel_and_library_operations": warm_kernel_and_library,
            "live_device_bytes": warm_stats.live_allocation_bytes,
            "live_staging_bytes": warm_stats.live_staging_bytes,
        },
        "launch_failure": {
            "status": "typed",
            "operation": failure_operation,
            "typed_error": failure_typed_error,
            "context_recovery": context_recovery,
            "recovery_succeeded": recovery_succeeded,
            "output_preserved": preserved_residual <= MAX_ERROR,
            "output_preservation_residual_max_abs": preserved_residual,
            "retry_succeeded": retry_succeeded,
            "retry_residual_max_abs": retry_residual,
            "partial_result_published": false,
        },
        "resources_released_on_teardown": resources_released,
        "live_device_bytes_after_teardown": cleanup_stats.live_allocation_bytes,
        "live_staging_bytes_after_teardown": cleanup_stats.live_staging_bytes,
        "adjoint_scope": "Picard/Oseen state transpose action; not exact Newton/design-residual adjoint",
    }))
}

/// Execute the production exact-Newton state and design-residual transpose
/// actions on a resident CUDA CSR representation.
///
/// The FEM assembly remains on the host.  This bounded cell proves that the
/// published analytic Newton tangent and frozen-mesh residual-design block
/// can be admitted as checked CSR data and that both transpose actions run on
/// CUDA without warm host traffic.  The finite-difference gates are scalar
/// directional derivatives of independent CPU CSR actions; they do not
/// substitute a black-box or Picard derivative for the production bundle.
pub fn production_newton_design_transpose_probe(
    newton: &PcFanResolvedFlowNewtonLinearization,
) -> Result<Value, String> {
    newton
        .validate()
        .map_err(|error| format!("production Newton linearization is invalid: {error}"))?;
    let state_matrix = &newton.state_jacobian;
    let design_matrix = newton
        .residual_design_jacobian
        .as_ref()
        .ok_or_else(|| "production Newton design residual Jacobian is missing".to_owned())?;
    let state_dimension = state_matrix.dimension;
    let design_dimension = design_matrix.first().map_or(0, Vec::len);
    if state_dimension == 0 || design_dimension == 0 {
        return Err("production Newton transpose fixture is empty".into());
    }
    if design_matrix.len() != state_dimension
        || design_matrix
            .iter()
            .any(|row| row.len() != design_dimension)
    {
        return Err("production Newton design residual dimensions are stale".into());
    }

    // Exercise the production metadata boundary before allocating a device
    // operator.  These rows intentionally mutate only local clones.
    let mut stale_metadata = newton.clone();
    stale_metadata.state_jacobian.dimension =
        stale_metadata.state_jacobian.dimension.saturating_add(1);
    let stale_metadata_rejected = stale_metadata.validate().is_err();
    let mut malformed_metadata = newton.clone();
    malformed_metadata
        .residual_design_jacobian
        .as_mut()
        .expect("checked design residual Jacobian")
        .pop();
    let malformed_metadata_rejected = malformed_metadata.validate().is_err();
    if !stale_metadata_rejected || !malformed_metadata_rejected {
        return Err("production Newton stale/malformed metadata was accepted".into());
    }

    let input: Vec<f64> = (0..state_dimension)
        .map(|index| 0.125 + 0.0007 * (index % 29) as f64)
        .collect();
    let lambda: Vec<f64> = (0..state_dimension)
        .map(|index| -0.09 + 0.0011 * (index % 31) as f64)
        .collect();
    let direction: Vec<f64> = (0..state_dimension)
        .map(|index| 0.07 - 0.0009 * (index % 19) as f64)
        .collect();
    let design_direction: Vec<f64> = (0..design_dimension)
        .map(|index| -0.11 + 0.003 * (index % 13) as f64)
        .collect();
    let cpu_forward = sparse_forward(
        state_dimension,
        &state_matrix.row_ptrs,
        &state_matrix.column_indices,
        &state_matrix.values,
        &input,
    );
    let cpu_transpose = sparse_transpose(
        state_dimension,
        &state_matrix.row_ptrs,
        &state_matrix.column_indices,
        &state_matrix.values,
        &lambda,
    );
    let mut cpu_design_transpose = vec![0.0; design_dimension];
    let mut design_direction_forward = vec![0.0; state_dimension];
    for row in 0..state_dimension {
        for (column, &value) in design_matrix[row].iter().enumerate() {
            cpu_design_transpose[column] += value * lambda[row];
            design_direction_forward[row] += value * design_direction[column];
        }
    }
    let cpu_state_identity_residual =
        (dot_vec(&cpu_forward, &lambda) - dot_vec(&input, &cpu_transpose)).abs();
    let cpu_design_identity_residual = (dot_vec(&design_direction_forward, &lambda)
        - dot_vec(&design_direction, &cpu_design_transpose))
    .abs();

    // Independent centered directional checks for the two CPU transpose
    // oracles.  The scalar functions below deliberately use only the host
    // CSR data and do not inspect CUDA output.
    let state_fd_step = 1.0e-6;
    let mut state_plus = input.clone();
    let mut state_minus = input.clone();
    for ((plus, minus), &value) in state_plus
        .iter_mut()
        .zip(state_minus.iter_mut())
        .zip(&direction)
    {
        *plus += state_fd_step * value;
        *minus -= state_fd_step * value;
    }
    let state_plus_value = dot_vec(
        &sparse_forward(
            state_dimension,
            &state_matrix.row_ptrs,
            &state_matrix.column_indices,
            &state_matrix.values,
            &state_plus,
        ),
        &lambda,
    );
    let state_minus_value = dot_vec(
        &sparse_forward(
            state_dimension,
            &state_matrix.row_ptrs,
            &state_matrix.column_indices,
            &state_matrix.values,
            &state_minus,
        ),
        &lambda,
    );
    let state_fd_slope = (state_plus_value - state_minus_value) / (2.0 * state_fd_step);
    let state_adjoint_slope = dot_vec(&direction, &cpu_transpose);
    let state_fd_relative_error = (state_fd_slope - state_adjoint_slope).abs()
        / state_fd_slope.abs().max(state_adjoint_slope.abs()).max(1.0);

    let design_fd_step = 1.0e-6;
    let mut design_plus = design_direction.clone();
    let mut design_minus = design_direction.clone();
    for ((plus, minus), &value) in design_plus
        .iter_mut()
        .zip(design_minus.iter_mut())
        .zip(&design_direction)
    {
        *plus += design_fd_step * value;
        *minus -= design_fd_step * value;
    }
    let design_scalar = |trial: &[f64]| {
        let mut output = vec![0.0; state_dimension];
        for row in 0..state_dimension {
            output[row] = design_matrix[row]
                .iter()
                .zip(trial)
                .map(|(&value, &component)| value * component)
                .sum();
        }
        dot_vec(&output, &lambda)
    };
    let design_fd_slope =
        (design_scalar(&design_plus) - design_scalar(&design_minus)) / (2.0 * design_fd_step);
    let design_adjoint_slope = dot_vec(&design_direction, &cpu_design_transpose);
    let design_fd_relative_error = (design_fd_slope - design_adjoint_slope).abs()
        / design_fd_slope
            .abs()
            .max(design_adjoint_slope.abs())
            .max(1.0);

    let backend = CudaBackend::new_default()
        .map_err(|error| format!("production Newton CUDA backend unavailable: {error}"))?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(256 << 20))
        .map_err(|error| format!("production Newton CUDA pool construction failed: {error}"))?;
    let backend = pool.backend();
    let state_host = HostCsrRef::try_new(
        state_dimension,
        state_dimension,
        &state_matrix.row_ptrs,
        &state_matrix.column_indices,
        &state_matrix.values,
    )
    .map_err(|error| format!("production Newton state CSR admission failed: {error}"))?;

    // AdjointDeviceCsr intentionally requires square metadata.  Pad the
    // rectangular production residual-design block with zero rows/columns;
    // the published state-to-design contraction is unchanged in its first
    // `design_dimension` output entries, while admission stays fail-closed.
    let padded_dimension = state_dimension.max(design_dimension);
    let mut design_row_ptrs = Vec::with_capacity(padded_dimension + 1);
    let mut design_column_indices = Vec::new();
    let mut design_values = Vec::new();
    design_row_ptrs.push(0);
    for row in 0..padded_dimension {
        if row < state_dimension {
            for (column, &value) in design_matrix[row].iter().enumerate() {
                if value != 0.0 {
                    design_column_indices.push(column);
                    design_values.push(value);
                }
            }
        }
        design_row_ptrs.push(design_values.len());
    }
    let design_host = HostCsrRef::try_new(
        padded_dimension,
        padded_dimension,
        &design_row_ptrs,
        &design_column_indices,
        &design_values,
    )
    .map_err(|error| format!("production Newton design CSR admission failed: {error}"))?;
    let state_operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, state_host)
        .map_err(|error| format!("production Newton state CSR upload failed: {error}"))?;
    let design_operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, design_host)
        .map_err(|error| format!("production Newton design CSR upload failed: {error}"))?;
    let state_prebaked = state_operator
        .try_prebake(backend)
        .map_err(|error| format!("production Newton state CSR prebake failed: {error}"))?;
    let design_prebaked = design_operator
        .try_prebake(backend)
        .map_err(|error| format!("production Newton design CSR prebake failed: {error}"))?;
    let mut input_device = pool
        .try_alloc::<f64>(state_dimension)
        .map_err(|error| format!("production Newton input allocation failed: {error}"))?;
    let mut lambda_device = pool
        .try_alloc::<f64>(padded_dimension)
        .map_err(|error| format!("production Newton lambda allocation failed: {error}"))?;
    let mut state_forward_output = pool
        .try_alloc::<f64>(state_dimension)
        .map_err(|error| format!("production Newton state forward allocation failed: {error}"))?;
    let mut state_transpose_output = pool
        .try_alloc::<f64>(state_dimension)
        .map_err(|error| format!("production Newton state transpose allocation failed: {error}"))?;
    let mut design_transpose_output = pool.try_alloc::<f64>(padded_dimension).map_err(|error| {
        format!("production Newton design transpose allocation failed: {error}")
    })?;
    let mut padded_lambda = vec![0.0; padded_dimension];
    padded_lambda[..state_dimension].copy_from_slice(&lambda);
    backend
        .try_copy_h2d(&input, input_device.raw_mut())
        .map_err(|error| format!("production Newton input upload failed: {error}"))?;
    backend
        .try_copy_h2d(&padded_lambda, lambda_device.raw_mut())
        .map_err(|error| format!("production Newton lambda upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("production Newton setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();
    let warm_start = backend.begin_runtime_stats();
    for _ in 0..2 {
        state_prebaked
            .try_apply(backend, input_device.raw(), state_forward_output.raw_mut())
            .map_err(|error| format!("production Newton state forward action failed: {error}"))?;
        state_prebaked
            .try_apply_hermitian(
                backend,
                lambda_device.raw(),
                state_transpose_output.raw_mut(),
            )
            .map_err(|error| format!("production Newton state transpose action failed: {error}"))?;
        design_prebaked
            .try_apply_hermitian(
                backend,
                lambda_device.raw(),
                design_transpose_output.raw_mut(),
            )
            .map_err(|error| {
                format!("production Newton design transpose action failed: {error}")
            })?;
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("production Newton warm synchronization failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_start);
    let mut cuda_forward = vec![0.0; state_dimension];
    let mut cuda_transpose = vec![0.0; state_dimension];
    let mut cuda_design_transpose = vec![0.0; padded_dimension];
    backend
        .try_copy_d2h(state_forward_output.raw(), &mut cuda_forward)
        .map_err(|error| format!("production Newton state forward download failed: {error}"))?;
    backend
        .try_copy_d2h(state_transpose_output.raw(), &mut cuda_transpose)
        .map_err(|error| format!("production Newton state transpose download failed: {error}"))?;
    backend
        .try_copy_d2h(design_transpose_output.raw(), &mut cuda_design_transpose)
        .map_err(|error| format!("production Newton design transpose download failed: {error}"))?;
    let state_forward_residual = max_abs_diff_vec(&cuda_forward, &cpu_forward);
    let state_transpose_residual = max_abs_diff_vec(&cuda_transpose, &cpu_transpose);
    let design_transpose_residual = max_abs_diff_vec(
        &cuda_design_transpose[..design_dimension],
        &cpu_design_transpose,
    );
    let cuda_state_identity_residual =
        (dot_vec(&cuda_forward, &lambda) - dot_vec(&input, &cuda_transpose)).abs();
    let cuda_design_identity_residual = (dot_vec(&design_direction_forward, &lambda)
        - dot_vec(
            &design_direction,
            &cuda_design_transpose[..design_dimension],
        ))
    .abs();
    if state_forward_residual > MAX_ERROR
        || state_transpose_residual > MAX_ERROR
        || design_transpose_residual > MAX_ERROR
        || cpu_state_identity_residual > MAX_ERROR
        || cuda_state_identity_residual > MAX_ERROR
        || cpu_design_identity_residual > MAX_ERROR
        || cuda_design_identity_residual > MAX_ERROR
        || state_fd_relative_error > 5.0e-8
        || design_fd_relative_error > 5.0e-8
    {
        return Err(format!(
            "production Newton transpose budget failed: state_forward={state_forward_residual:.3e}, state_transpose={state_transpose_residual:.3e}, design_transpose={design_transpose_residual:.3e}, state_fd={state_fd_relative_error:.3e}, design_fd={design_fd_relative_error:.3e}"
        ));
    }
    if warm_stats.allocation_operations != 0
        || warm_stats.staging_allocation_operations != 0
        || warm_stats.transfer.h2d_operations != 0
    {
        return Err(format!(
            "production Newton warm path allocated/uploaded: device={} staging={} h2d={}",
            warm_stats.allocation_operations,
            warm_stats.staging_allocation_operations,
            warm_stats.transfer.h2d_operations
        ));
    }

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = design_prebaked.try_apply_hermitian(
        backend,
        lambda_device.raw(),
        design_transpose_output.raw_mut(),
    );
    let failure_error = failure.err();
    let recovery = if failure_error.is_some() {
        backend.clear_failure();
        backend.recover_after_error()
    } else {
        Ok(())
    };
    let recovery_succeeded = recovery.is_ok();
    let mut preserved = vec![0.0; padded_dimension];
    backend
        .try_copy_d2h(design_transpose_output.raw(), &mut preserved)
        .map_err(|error| format!("production Newton preserved-output download failed: {error}"))?;
    let preserved_residual = max_abs_diff_vec(
        &preserved[..design_dimension],
        &cuda_design_transpose[..design_dimension],
    );
    let retry = design_prebaked
        .try_apply_hermitian(
            backend,
            lambda_device.raw(),
            design_transpose_output.raw_mut(),
        )
        .and_then(|()| backend.try_synchronize());
    let retry_succeeded = retry.is_ok();
    let mut retried = vec![0.0; padded_dimension];
    if retry_succeeded {
        backend
            .try_copy_d2h(design_transpose_output.raw(), &mut retried)
            .map_err(|error| format!("production Newton retry download failed: {error}"))?;
    }
    let retry_residual = max_abs_diff_vec(
        &retried[..design_dimension],
        &cuda_design_transpose[..design_dimension],
    );
    let typed_launch_failure = matches!(failure_error, Some(BackendError::Launch { .. }));
    if !typed_launch_failure
        || !recovery_succeeded
        || preserved_residual > MAX_ERROR
        || !retry_succeeded
        || retry_residual > MAX_ERROR
    {
        return Err(format!(
            "production Newton failure contract failed: typed={typed_launch_failure}, recovery={recovery_succeeded}, preserved={preserved_residual:.3e}, retry={retry_succeeded}, retry_residual={retry_residual:.3e}"
        ));
    }
    let failure_operation = failure_error
        .as_ref()
        .map(backend_operation)
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let failure_typed_error = failure_error
        .as_ref()
        .map(|error| format!("{error:?}"))
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let context_recovery = failure_error
        .as_ref()
        .map(|error| recovery_label(error.context_disposition()))
        .unwrap_or("unknown");
    let warm_kernel_and_library = warm_stats.kernel_launches + warm_stats.library_operations;

    drop(state_prebaked);
    drop(design_prebaked);
    drop(state_operator);
    drop(design_operator);
    drop(input_device);
    drop(lambda_device);
    drop(state_forward_output);
    drop(state_transpose_output);
    drop(design_transpose_output);
    let cleanup_stats = backend.runtime_stats();
    let resources_released = cleanup_stats.live_allocation_bytes == 0
        && cleanup_stats.live_staging_bytes == 0
        && pool.bytes_in_use() == 0;

    Ok(json!({
        "status": "passed",
        "backend": "Cuda",
        "execution_mode": "DeviceResident",
        "dispatch": "native-cuda-cusparse",
        "operator": "production PcFanResolvedFlowNewtonLinearization.state_jacobian + residual_design_jacobian",
        "flow_model": format!("{:?}", newton.flow_model),
        "provenance": newton.provenance,
        "dimension": state_dimension,
        "nnz": state_matrix.values.len(),
        "design_dimension": design_dimension,
        "design_nnz": design_values.len(),
        "padded_operator_dimension": padded_dimension,
        "repeated_actions": 2,
        "cpu_reference": {
            "state_forward_l2": dot_vec(&cpu_forward, &cpu_forward).sqrt(),
            "state_transpose_l2": dot_vec(&cpu_transpose, &cpu_transpose).sqrt(),
            "design_transpose_l2": dot_vec(&cpu_design_transpose, &cpu_design_transpose).sqrt(),
            "state_inner_product_residual": cpu_state_identity_residual,
            "design_inner_product_residual": cpu_design_identity_residual,
            "state_finite_difference_relative_error": state_fd_relative_error,
            "design_finite_difference_relative_error": design_fd_relative_error,
        },
        "cuda_observation": {
            "state_forward_residual_max_abs": state_forward_residual,
            "state_transpose_residual_max_abs": state_transpose_residual,
            "design_transpose_residual_max_abs": design_transpose_residual,
            "state_inner_product_residual": cuda_state_identity_residual,
            "design_inner_product_residual": cuda_design_identity_residual,
        },
        "warm_counters": {
            "device_allocations": warm_stats.allocation_operations,
            "staging_allocations": warm_stats.staging_allocation_operations,
            "h2d_operations": warm_stats.transfer.h2d_operations,
            "d2h_operations": warm_stats.transfer.d2h_operations,
            "synchronizations": warm_stats.synchronization_calls,
            "kernel_and_library_operations": warm_kernel_and_library,
            "live_device_bytes": warm_stats.live_allocation_bytes,
            "live_staging_bytes": warm_stats.live_staging_bytes,
        },
        "metadata": {
            "stale_rejected": stale_metadata_rejected,
            "malformed_rejected": malformed_metadata_rejected,
        },
        "launch_failure": {
            "status": "typed",
            "operation": failure_operation,
            "typed_error": failure_typed_error,
            "context_recovery": context_recovery,
            "recovery_succeeded": recovery_succeeded,
            "output_preserved": preserved_residual <= MAX_ERROR,
            "output_preservation_residual_max_abs": preserved_residual,
            "retry_succeeded": retry_succeeded,
            "retry_residual_max_abs": retry_residual,
            "partial_result_published": false,
        },
        "resources_released_on_teardown": resources_released,
        "live_device_bytes_after_teardown": cleanup_stats.live_allocation_bytes,
        "live_staging_bytes_after_teardown": cleanup_stats.live_staging_bytes,
        "adjoint_scope": "exact analytic Newton state transpose plus frozen-mesh design-residual transpose; not coupled acoustic/transient adjoint",
    }))
}

/// Execute the bounded production paired-CSR CUDA action and its failure gate.
pub fn paired_csr_probe() -> Result<Value, String> {
    let malformed_rejected =
        HostCsrRef::try_new(3, 3, &[0, 2, 1, 7], &HOST_COL_IDX, &VALUES).is_err();
    if !malformed_rejected {
        return Err("malformed paired-CSR input was accepted".into());
    }

    let backend = CudaBackend::new_default()
        .map_err(|error| format!("PF2 paired-CSR CUDA backend unavailable: {error}"))?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
        .map_err(|error| format!("PF2 paired-CSR memory pool construction failed: {error}"))?;
    let backend = pool.backend();
    let cpu_forward = forward(&VALUES, &X);
    let cpu_transpose = transpose(&VALUES, &Y);
    let cpu_lhs = cpu_forward
        .iter()
        .zip(Y)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let cpu_rhs = X
        .iter()
        .zip(cpu_transpose)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let cpu_identity_residual = (cpu_lhs - cpu_rhs).abs();

    let host = HostCsrRef::try_new(3, 3, &HOST_ROW_PTR, &HOST_COL_IDX, &VALUES)
        .map_err(|error| format!("PF2 paired-CSR fixture is invalid: {error}"))?;
    let operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("PF2 paired-CSR upload failed: {error}"))?;
    let prebaked = operator
        .try_prebake(backend)
        .map_err(|error| format!("PF2 paired-CSR prebake failed: {error}"))?;
    let mut x = pool
        .try_alloc::<f64>(3)
        .map_err(|error| format!("PF2 paired-CSR x allocation failed: {error}"))?;
    let mut y = pool
        .try_alloc::<f64>(3)
        .map_err(|error| format!("PF2 paired-CSR y allocation failed: {error}"))?;
    let mut forward_out = pool
        .try_alloc::<f64>(3)
        .map_err(|error| format!("PF2 paired-CSR forward output allocation failed: {error}"))?;
    let mut transpose_out = pool
        .try_alloc::<f64>(3)
        .map_err(|error| format!("PF2 paired-CSR transpose output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&X, x.raw_mut())
        .map_err(|error| format!("PF2 paired-CSR x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&Y, y.raw_mut())
        .map_err(|error| format!("PF2 paired-CSR y upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("PF2 paired-CSR setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();
    let warm_start = backend.begin_runtime_stats();
    for _ in 0..2 {
        prebaked
            .try_apply(backend, x.raw(), forward_out.raw_mut())
            .map_err(|error| format!("PF2 paired-CSR forward action failed: {error}"))?;
        prebaked
            .try_apply_hermitian(backend, y.raw(), transpose_out.raw_mut())
            .map_err(|error| format!("PF2 paired-CSR Hermitian action failed: {error}"))?;
    }
    backend
        .try_synchronize()
        .map_err(|error| format!("PF2 paired-CSR warm synchronization failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_start);
    let mut cuda_forward = [0.0; 3];
    let mut cuda_transpose = [0.0; 3];
    backend
        .try_copy_d2h(forward_out.raw(), &mut cuda_forward)
        .map_err(|error| format!("PF2 paired-CSR forward download failed: {error}"))?;
    backend
        .try_copy_d2h(transpose_out.raw(), &mut cuda_transpose)
        .map_err(|error| format!("PF2 paired-CSR transpose download failed: {error}"))?;
    let forward_residual = max_abs_diff(&cuda_forward, &cpu_forward);
    let transpose_residual = max_abs_diff(&cuda_transpose, &cpu_transpose);
    let cuda_lhs = cuda_forward
        .iter()
        .zip(Y)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let cuda_rhs = X
        .iter()
        .zip(cuda_transpose)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let cuda_identity_residual = (cuda_lhs - cuda_rhs).abs();
    if forward_residual > MAX_ERROR
        || transpose_residual > MAX_ERROR
        || cpu_identity_residual > MAX_ERROR
        || cuda_identity_residual > MAX_ERROR
    {
        return Err(format!(
            "PF2 paired-CSR residual budget failed: forward={forward_residual:.3e}, transpose={transpose_residual:.3e}, cpu_identity={cpu_identity_residual:.3e}, cuda_identity={cuda_identity_residual:.3e}"
        ));
    }
    if warm_stats.allocation_operations != 0
        || warm_stats.staging_allocation_operations != 0
        || warm_stats.transfer.h2d_operations != 0
    {
        return Err(format!(
            "PF2 paired-CSR warm path allocated/uploaded: device={} staging={} h2d={}",
            warm_stats.allocation_operations,
            warm_stats.staging_allocation_operations,
            warm_stats.transfer.h2d_operations
        ));
    }

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = prebaked.try_apply(backend, x.raw(), forward_out.raw_mut());
    let failure_error = failure.err();
    let recovery = if failure_error.is_some() {
        backend.clear_failure();
        backend.recover_after_error()
    } else {
        Ok(())
    };
    let recovery_succeeded = recovery.is_ok();
    let recovery_error = recovery.err().map(|error| error.to_string());
    let mut preserved = [0.0; 3];
    backend
        .try_copy_d2h(forward_out.raw(), &mut preserved)
        .map_err(|error| format!("PF2 paired-CSR preserved-output download failed: {error}"))?;
    let preserved_residual = max_abs_diff(&preserved, &cuda_forward);
    let retry = prebaked
        .try_apply(backend, x.raw(), forward_out.raw_mut())
        .and_then(|()| backend.try_synchronize());
    let retry_succeeded = retry.is_ok();
    let retry_error = retry.err().map(|error| error.to_string());
    let mut retried = [0.0; 3];
    if retry_succeeded {
        backend
            .try_copy_d2h(forward_out.raw(), &mut retried)
            .map_err(|error| format!("PF2 paired-CSR retry download failed: {error}"))?;
    }
    let retry_residual = max_abs_diff(&retried, &cuda_forward);
    let typed_launch_failure = matches!(failure_error, Some(BackendError::Launch { .. }));
    if !typed_launch_failure
        || !recovery_succeeded
        || preserved_residual > MAX_ERROR
        || !retry_succeeded
        || retry_residual > MAX_ERROR
    {
        return Err(format!(
            "PF2 paired-CSR failure contract failed: typed={typed_launch_failure}, recovery={recovery_succeeded}, preserved={preserved_residual:.3e}, retry={retry_succeeded}, retry_residual={retry_residual:.3e}"
        ));
    }
    let failure_operation = failure_error
        .as_ref()
        .map(backend_operation)
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let failure_typed_error = failure_error
        .as_ref()
        .map(|error| format!("{error:?}"))
        .unwrap_or_else(|| "missing_launch_error".to_owned());
    let context_recovery = failure_error
        .as_ref()
        .map(|error| recovery_label(error.context_disposition()))
        .unwrap_or("unknown");
    let warm_kernel_and_library = warm_stats.kernel_launches + warm_stats.library_operations;

    drop(prebaked);
    drop(operator);
    drop(x);
    drop(y);
    drop(forward_out);
    drop(transpose_out);
    let cleanup_stats = backend.runtime_stats();
    let resources_released = cleanup_stats.live_allocation_bytes == 0
        && cleanup_stats.live_staging_bytes == 0
        && pool.bytes_in_use() == 0;

    Ok(json!({
        "status": "passed",
        "backend": "Cuda",
        "execution_mode": "DeviceResident",
        "dispatch": "native-cuda-cusparse",
        "operator": "AdjointDeviceCsr::try_prebake + DeviceAdjointOperator::try_apply_hermitian",
        "scalar": "f64 (Hermitian action is transpose for this real fixture)",
        "dimension": 3,
        "nnz": VALUES.len(),
        "repeated_actions": 2,
        "malformed_csr_rejected": malformed_rejected,
        "cpu_reference": {
            "forward": cpu_forward,
            "transpose": cpu_transpose,
            "inner_product_residual": cpu_identity_residual,
        },
        "cuda_observation": {
            "forward": cuda_forward,
            "transpose": cuda_transpose,
            "forward_residual_max_abs": forward_residual,
            "transpose_residual_max_abs": transpose_residual,
            "inner_product_residual": cuda_identity_residual,
        },
        "warm_counters": {
            "device_allocations": warm_stats.allocation_operations,
            "staging_allocations": warm_stats.staging_allocation_operations,
            "h2d_operations": warm_stats.transfer.h2d_operations,
            "d2h_operations": warm_stats.transfer.d2h_operations,
            "synchronizations": warm_stats.synchronization_calls,
            "kernel_and_library_operations": warm_kernel_and_library,
            "live_device_bytes": warm_stats.live_allocation_bytes,
            "live_staging_bytes": warm_stats.live_staging_bytes,
        },
        "launch_failure": {
            "status": "typed",
            "operation": failure_operation,
            "typed_error": failure_typed_error,
            "context_recovery": context_recovery,
            "recovery_succeeded": recovery_succeeded,
            "recovery_error": recovery_error,
            "output_preserved": preserved_residual <= MAX_ERROR,
            "output_preservation_residual_max_abs": preserved_residual,
            "retry_succeeded": retry_succeeded,
            "retry_error": retry_error,
            "retry_residual_max_abs": retry_residual,
            "partial_result_published": false,
        },
        "resources_released_on_teardown": resources_released,
        "live_device_bytes_after_teardown": cleanup_stats.live_allocation_bytes,
        "live_staging_bytes_after_teardown": cleanup_stats.live_staging_bytes,
        "high_level_design_residual_coupling": {
            "status": "unsupported",
            "reason": "bounded paired-CSR algebra does not traverse the PC-fan FEM/CFD design-residual bundle",
            "partial_result_published": false,
        },
    }))
}
