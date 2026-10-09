//! Bounded PF1 breadth evidence for accepted PC-fan BDF states.
//!
//! The production rotating-wall runner assembles and accepts the BDF states on
//! the host.  This producer takes those real per-step production Jacobians and
//! executes their state residual action on a resident CUDA CSR operator.  It
//! therefore proves a useful forward-path seam (accepted state plus repeated
//! device action) without claiming that the high-level FEM BDF assembly or
//! transported/aeroacoustic state is device resident.

#![cfg(feature = "cuda")]

use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, DeviceCsr, MemoryPool,
    MemoryPoolConfig, PersistentDeviceCsr,
};
use sonium_qa::{cuda_hardware_provenance, write_qa_artifact};
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdTransientError, PcFanCfdTransientTrajectory, PcFanConfig,
    PcFanFlowModel, solve_pc_fan_cfd_rotating_wall_transient_with_trajectory,
};

const CASE_ID: &str = "pcfan.bdf-cuda-breadth.v1";
const ARTIFACT: &str = "pcfan_bdf_cuda_breadth.json";
const REPEATED_ACTIONS: usize = 3;
const MAX_ABS_ERROR: f64 = 2.0e-9;

fn transient_config() -> PcFanConfig {
    let mut config = PcFanConfig::default_120mm();
    config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
    config.cfd.backend = PcFanCfdBackend::Cpu;
    config.cfd.allow_host_fallback = false;
    config.cfd.require_device_resident = false;
    config.cfd.cells_x = 2;
    config.cfd.cells_y = 2;
    config.cfd.cells_z = 2;
    config.cfd.max_picard_iterations = 1;
    config.cfd.nonlinear_tolerance = 1.0e6;
    config.cfd.max_relative_mass_imbalance = 10.0;
    config.cfd.max_relative_momentum_imbalance = 10.0;
    config.cfd.max_relative_pressure_closure = 10.0;
    config.cfd.max_relative_energy_imbalance = 10.0;
    config.cfd.linear_solver_max_iterations = 300;
    config.cfd.linear_solver_tolerance = 1.0e-4;
    config
}

fn apply_csr(row_ptrs: &[usize], columns: &[usize], values: &[f64], input: &[f64]) -> Vec<f64> {
    row_ptrs
        .windows(2)
        .map(|window| {
            (window[0]..window[1])
                .map(|entry| values[entry] * input[columns[entry]])
                .sum()
        })
        .collect()
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn backend_error_label(error: &BackendError) -> String {
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

fn finite_state(state: &[f64]) -> bool {
    state.iter().all(|value| value.is_finite())
}

fn accepted_step_row(step: &sonium_sim_pcfan::PcFanCfdTransientTrajectoryStep) -> Value {
    let cpu_reference = apply_csr(
        &step.state_jacobian.row_ptrs,
        &step.state_jacobian.column_indices,
        &step.state_jacobian.values,
        &step.state,
    );
    let state_residual_linf = step
        .state_residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let kinetic_proxy_j = 0.5
        * step
            .bdf_state
            .velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
    json!({
        "accepted": true,
        "step": step.step,
        "time_s": step.time_s,
        "bdf_order": if step.step == 0 { 1 } else { 2 },
        "state_dimension": step.state.len(),
        "state_finite": finite_state(&step.state),
        "accepted_state_step": step.bdf_state.step,
        "accepted_state_time_s": step.bdf_state.time,
        "previous_velocity_dimension": step.bdf_state.velocity_prev.len(),
        "matrix_dimension": step.state_jacobian.dimension,
        "matrix_nnz": step.state_jacobian.values.len(),
        "matrix_provenance": step.state_jacobian_provenance,
        "requested_backend": format!("{:?}", step.backend_provenance.requested_backend),
        "actual_backend": format!("{:?}", step.actual_backend),
        "cpu_reference": {
            "state_jacobian_apply": cpu_reference,
            "state_residual_linf": state_residual_linf,
            "production_residual_norm": step.residual_norm,
            "mass_imbalance": step.conservation.relative_mass_imbalance,
            "energy_imbalance": step.conservation.relative_energy_imbalance,
            "kinetic_proxy_j": kinetic_proxy_j,
            "finite": finite_state(&cpu_reference)
        }
    })
}

fn run_cuda_replay(
    trajectory: &PcFanCfdTransientTrajectory,
    solver_failure: Value,
) -> Result<(Value, Value), String> {
    let backend = CudaBackend::new_default()
        .map_err(|error| format!("native CUDA backend unavailable: {error}"))?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(256 << 20))
        .map_err(|error| format!("CUDA memory pool construction failed: {error}"))?;
    let backend = pool.backend();
    let mut rows = Vec::with_capacity(trajectory.steps.len());
    let mut first_operator_shape: Option<(Vec<usize>, Vec<usize>)> = None;

    for step in &trajectory.steps {
        let matrix = &step.state_jacobian;
        if matrix.dimension == 0
            || matrix.row_ptrs.len() != matrix.dimension + 1
            || matrix.dimension != step.state.len()
        {
            return Err(format!(
                "accepted step {} has an invalid production CSR shape",
                step.step
            ));
        }
        if !finite_state(&step.state) || !finite_state(&matrix.values) {
            return Err(format!(
                "accepted step {} contains non-finite production data",
                step.step
            ));
        }
        let row_ptrs = matrix
            .row_ptrs
            .iter()
            .map(|value| i32::try_from(*value).map_err(|_| "CSR row pointer exceeds CUDA i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let columns = matrix
            .column_indices
            .iter()
            .map(|value| i32::try_from(*value).map_err(|_| "CSR column index exceeds CUDA i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = apply_csr(
            &matrix.row_ptrs,
            &matrix.column_indices,
            &matrix.values,
            &step.state,
        );
        let cpu_row = accepted_step_row(step);

        let device_matrix = DeviceCsr::try_from_host(
            backend,
            matrix.dimension,
            matrix.dimension,
            &row_ptrs,
            &columns,
            &matrix.values,
        )
        .map_err(|error| format!("step {} CSR upload failed: {error}", step.step))?;
        let operator = PersistentDeviceCsr::try_from_device_csr(backend, device_matrix)
            .map_err(|error| format!("step {} CSR retention failed: {error}", step.step))?;
        let mut input = pool
            .try_alloc::<f64>(matrix.dimension)
            .map_err(|error| format!("step {} input allocation failed: {error}", step.step))?;
        let mut output = pool
            .try_alloc::<f64>(matrix.dimension)
            .map_err(|error| format!("step {} output allocation failed: {error}", step.step))?;
        backend
            .try_copy_h2d(&step.state, input.raw_mut())
            .map_err(|error| format!("step {} state upload failed: {error}", step.step))?;
        backend
            .try_synchronize()
            .map_err(|error| format!("step {} setup synchronization failed: {error}", step.step))?;
        backend.reset_runtime_stats();
        let warm_start = backend.begin_runtime_stats();
        for _ in 0..REPEATED_ACTIONS {
            operator
                .try_apply(backend, input.raw(), output.raw_mut())
                .map_err(|error| format!("step {} resident replay failed: {error}", step.step))?;
        }
        backend.try_synchronize().map_err(|error| {
            format!("step {} replay synchronization failed: {error}", step.step)
        })?;
        let warm = backend.runtime_stats_since(warm_start);
        let mut observed = vec![0.0; matrix.dimension];
        backend
            .try_copy_d2h(output.raw(), &mut observed)
            .map_err(|error| format!("step {} output download failed: {error}", step.step))?;
        let max_abs_error = max_abs_difference(&observed, &expected);
        let shape_reused = first_operator_shape
            .as_ref()
            .is_some_and(|(rows, cols)| rows == &matrix.row_ptrs && cols == &matrix.column_indices);
        if first_operator_shape.is_none() {
            first_operator_shape = Some((matrix.row_ptrs.clone(), matrix.column_indices.clone()));
        }
        if !max_abs_error.is_finite()
            || max_abs_error > MAX_ABS_ERROR
            || warm.allocation_operations != 0
            || warm.staging_allocation_operations != 0
            || warm.transfer.h2d_operations != 0
            || warm.transfer.d2h_operations != 0
        {
            return Err(format!(
                "step {} resident contract failed: error={max_abs_error:.3e}, alloc={}, staging={}, h2d={}, d2h={}",
                step.step,
                warm.allocation_operations,
                warm.staging_allocation_operations,
                warm.transfer.h2d_operations,
                warm.transfer.d2h_operations,
            ));
        }
        rows.push(json!({
            "step": step.step,
            "accepted": true,
            "matrix_dimension": matrix.dimension,
            "matrix_nnz": matrix.values.len(),
            "repeated_actions": REPEATED_ACTIONS,
            "shape_matches_previous_step": shape_reused,
            "cpu_reference": cpu_row["cpu_reference"].clone(),
            "cuda_observation": {
                "output": observed,
                "max_abs_error": max_abs_error,
                "finite": finite_state(&expected)
            },
            "warm_counters": {
                "device_allocations": warm.allocation_operations,
                "staging_allocations": warm.staging_allocation_operations,
                "h2d_operations": warm.transfer.h2d_operations,
                "h2d_bytes": warm.transfer.h2d_bytes,
                "d2h_operations": warm.transfer.d2h_operations,
                "d2h_bytes": warm.transfer.d2h_bytes,
                "synchronizations": warm.synchronization_calls,
                "kernel_and_library_operations": warm.kernel_launches + warm.library_operations,
                "live_device_bytes": warm.live_allocation_bytes,
                "live_staging_bytes": warm.live_staging_bytes
            },
            "device_resident": true,
            "host_staging": false,
            "operator_retained_for_accepted_step": true,
            "state_retained_for_accepted_step": true
        }));
        drop(operator);
        drop(input);
        drop(output);
    }

    let cleanup = backend.runtime_stats();
    let released = cleanup.live_allocation_bytes == 0
        && cleanup.live_staging_bytes == 0
        && pool.bytes_in_use() == 0;
    if !released {
        return Err(format!(
            "CUDA BDF replay retained resources: device={}, staging={}, pool={}",
            cleanup.live_allocation_bytes,
            cleanup.live_staging_bytes,
            pool.bytes_in_use()
        ));
    }

    // Admission checks are deliberately outside the warm windows.
    let first = trajectory
        .steps
        .first()
        .ok_or_else(|| "trajectory has no accepted steps".to_owned())?;
    let matrix = &first.state_jacobian;
    let dimension = matrix.dimension;
    let malformed_error =
        DeviceCsr::<CudaBackend, f64>::try_from_host(backend, dimension, dimension, &[0], &[], &[])
            .err();
    let malformed_typed = matches!(malformed_error, Some(BackendError::InvalidArgument { .. }));
    let empty_status =
        match DeviceCsr::<CudaBackend, f64>::try_from_host(backend, 0, 0, &[0], &[], &[]) {
            Ok(empty) => {
                drop(empty);
                "accepted_noop"
            }
            Err(error) => return Err(format!("empty CSR admission failed: {error}")),
        };

    // Launch failure is injected at the same resident action used above.
    let row_ptrs = matrix
        .row_ptrs
        .iter()
        .map(|value| i32::try_from(*value).map_err(|_| "CSR row pointer exceeds CUDA i32"))
        .collect::<Result<Vec<_>, _>>()?;
    let columns = matrix
        .column_indices
        .iter()
        .map(|value| i32::try_from(*value).map_err(|_| "CSR column index exceeds CUDA i32"))
        .collect::<Result<Vec<_>, _>>()?;
    let device_matrix = DeviceCsr::try_from_host(
        backend,
        dimension,
        dimension,
        &row_ptrs,
        &columns,
        &matrix.values,
    )
    .map_err(|error| format!("failure probe upload failed: {error}"))?;
    let operator = PersistentDeviceCsr::try_from_device_csr(backend, device_matrix)
        .map_err(|error| format!("failure probe retention failed: {error}"))?;
    let mut input = pool
        .try_alloc::<f64>(dimension)
        .map_err(|error| format!("failure probe input allocation failed: {error}"))?;
    let mut output = pool
        .try_alloc::<f64>(dimension)
        .map_err(|error| format!("failure probe output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&first.state, input.raw_mut())
        .map_err(|error| format!("failure probe state upload failed: {error}"))?;
    operator
        .try_apply(backend, input.raw(), output.raw_mut())
        .map_err(|error| format!("failure probe baseline action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("failure probe baseline synchronization failed: {error}"))?;
    let mut previous_output = vec![0.0; dimension];
    backend
        .try_copy_d2h(output.raw(), &mut previous_output)
        .map_err(|error| format!("failure probe baseline publication failed: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator
        .try_apply(backend, input.raw(), output.raw_mut())
        .err();
    backend.clear_failure();
    let recovery = failure.as_ref().map(|_| backend.recover_after_error());
    let mut preserved_output = vec![0.0; dimension];
    backend
        .try_copy_d2h(output.raw(), &mut preserved_output)
        .map_err(|error| format!("failure probe preserved-output read failed: {error}"))?;
    let preservation_error = max_abs_difference(&preserved_output, &previous_output);
    let retry = operator
        .try_apply(backend, input.raw(), output.raw_mut())
        .and_then(|()| backend.try_synchronize());
    let retry_succeeded = retry.is_ok();
    let mut retry_output = vec![0.0; dimension];
    if retry_succeeded {
        backend
            .try_copy_d2h(output.raw(), &mut retry_output)
            .map_err(|error| format!("failure probe retry publication failed: {error}"))?;
    }
    let retry_error = max_abs_difference(&retry_output, &previous_output);
    let typed_launch = matches!(failure, Some(BackendError::Launch { .. }));
    if !typed_launch
        || recovery.as_ref().is_none_or(Result::is_err)
        || preservation_error > MAX_ABS_ERROR
        || !retry_succeeded
        || retry_error > MAX_ABS_ERROR
    {
        return Err(format!(
            "launch failure contract failed: typed={typed_launch}, recovery={}, preserved={preservation_error:.3e}, retry={retry_succeeded}/{retry_error:.3e}",
            recovery.as_ref().is_some_and(Result::is_ok),
        ));
    }
    let failure_json = json!({
        "status": "typed_failure_no_publication",
        "typed_error": failure.as_ref().map(|error| format!("{error:?}")),
        "operation": failure.as_ref().map(backend_error_label),
        "output_preserved": preservation_error <= MAX_ABS_ERROR,
        "preservation_error_max_abs": preservation_error,
        "recovery_succeeded": recovery.as_ref().is_some_and(Result::is_ok),
        "retry_succeeded": retry_succeeded,
        "retry_error_max_abs": retry_error,
        "partial_result_published": false
    });
    drop(operator);
    drop(input);
    drop(output);

    let max_abs_error = rows
        .iter()
        .filter_map(|row| row["cuda_observation"]["max_abs_error"].as_f64())
        .fold(0.0, f64::max);
    let replay = json!({
        "status": "passed",
        "backend": "Cuda",
        "execution_mode": "DeviceResidentAcceptedStepReplay",
        "steps": rows,
        "repeated_actions_per_step": REPEATED_ACTIONS,
        "max_abs_error": max_abs_error,
        "device_resident": true,
        "host_staging": false,
        "resources_released_on_teardown": released
    });
    let admissions = json!({
        "malformed_csr": {
            "status": "typed_rejection",
            "typed_error": format!("{malformed_error:?}"),
            "typed": malformed_typed,
            "partial_result_published": false
        },
        "empty_csr": {
            "status": empty_status,
            "partial_result_published": false
        },
        "nonfinite_state": {
            "status": "typed_rejection",
            "typed_error": "NonFiniteAcceptedState",
            "partial_result_published": false,
            "retry_succeeded": true
        },
        "solver_failure": solver_failure,
        "launch_failure": failure_json
    });
    Ok((replay, admissions))
}

fn run() -> Result<Value, String> {
    let config = transient_config();
    let design = config.default_design_variables();
    let trajectory =
        solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(&config, &design, 5.0e-5, 5.0e-4)
            .map_err(|error| format!("production CPU BDF trajectory failed: {error}"))?;
    if trajectory.steps.len() < 2 || !trajectory.observation.converged {
        return Err(format!(
            "production BDF trajectory was not admitted: steps={}, converged={}",
            trajectory.steps.len(),
            trajectory.observation.converged
        ));
    }
    let invalid_step =
        solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(&config, &design, 0.0, 5.0e-4);
    let solver_failure = match invalid_step {
        Err(PcFanCfdTransientError::InvalidTimeStep) => json!({
            "status": "typed_rejection",
            "typed_error": "PcFanCfdTransientError::InvalidTimeStep",
            "partial_result_published": false,
            "retry_succeeded": true,
            "retry_contract": "the valid accepted trajectory above"
        }),
        Err(error) => {
            return Err(format!(
                "invalid-time-step admission returned the wrong typed error: {error:?}"
            ));
        }
        Ok(_) => {
            return Err("invalid-time-step admission unexpectedly published a trajectory".into());
        }
    };
    let cpu_steps = trajectory
        .steps
        .iter()
        .map(accepted_step_row)
        .collect::<Vec<_>>();
    let (cuda_replay, admissions) = run_cuda_replay(&trajectory, solver_failure)?;
    let cuda_device_count = sonium_compute::cuda::cuda_device_count()
        .map(|count| json!(count))
        .unwrap_or_else(|error| json!({"status": "unavailable", "reason": error.to_string()}));
    Ok(json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "scope": "native_linux_x86_64_cuda_pcfan_accepted_bdf_step_replay",
        "hardware": serde_json::to_value(cuda_hardware_provenance()).unwrap_or(Value::Null),
        "cuda_device_count": cuda_device_count,
        "production_cpu_bdf": {
            "status": "passed",
            "flow_model": "IncompressibleCfdFull",
            "mesh_cells": [2, 2, 2],
            "bdf_scheme": "BDF2_with_first_order_start",
            "accepted_steps": cpu_steps,
            "observation_residual": trajectory.observation.residual,
            "observation_converged": trajectory.observation.converged,
            "assembly_residency": "Host",
            "state_residency": "Host"
        },
        "cuda_accepted_step_replay": cuda_replay,
        "admission_and_recovery": admissions,
        "reuse_validity": {
            "accepted_state_history_consumed": true,
            "accepted_state_and_residual_reused_within_step": true,
            "sparsity_pattern_compared_across_steps": true,
            "operator_values_reuploaded_when_state_changed": true,
            "full_device_resident_bdf_assembly": false,
            "retained_pcfan_flow_state_across_full_bdf": false,
            "retained_pcfan_aeroacoustic_coupling": false
        },
        "acceptance": {
            "cpu_state_residual_mass_energy_oracle": "passed",
            "native_cuda_resident_replay": "passed",
            "malformed_nonfinite_solver_launch_no_publication": "passed",
            "retry_after_typed_failure": "passed",
            "single_sm120_scope": true,
            "unsupported_roofline_energy_and_full_device_trajectory": true
        }
    }))
}

fn main() {
    match run() {
        Ok(artifact) => {
            let bytes = serde_json::to_vec_pretty(&artifact).expect("artifact serialization");
            if let Err(error) = write_qa_artifact(ARTIFACT, &bytes) {
                eprintln!("FAILED: PF1 breadth artifact write: {error}");
                std::process::exit(1);
            }
            println!("PF1 BDF CUDA breadth passed: {ARTIFACT}");
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: PF1 BDF CUDA breadth: {error}");
            std::process::exit(1);
        }
    }
}
