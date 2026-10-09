//! Bounded SM2 breadth replay for the production axisymmetric coupled FEM
//! operator.  SM1 owns the assembled CSR CUDA cell; this producer consumes
//! its passing record, solves the same production model on an independent
//! CPU/f64 path, and replays the resulting operator over three frequencies.
//! It deliberately does not claim device-resident high-level FEM assembly.

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot,
    HostCsrRef, PersistentDeviceCsr,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use sonium_sim_speaker_modeling::{
    AxisymmetricLinearConfig, AxisymmetricLinearModel, AxisymmetricLinearSolution,
    run_production_frequency_step_cuda_contract,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-sm2-axisymmetric-response-breadth.v1";
const ARTIFACT: &str = "speaker_modeling_sm2_breadth.json";
const FREQUENCIES_HZ: [f64; 3] = [400.0, 500.0, 600.0];
const OPERATOR_TOLERANCE: f64 = 1.0e-8;
const DERIVATIVE_TOLERANCE: f64 = 1.0e-8;
const SOLVER_RESIDUAL_TOLERANCE: f64 = 1.0e-6;

fn runtime_json(stats: &CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "allocations": stats.allocation_operations,
        "transfers": {
            "h2d_operations": stats.transfer.h2d_operations,
            "h2d_bytes": stats.transfer.h2d_bytes,
            "d2h_operations": stats.transfer.d2h_operations,
            "d2h_bytes": stats.transfer.d2h_bytes,
            "d2d_operations": stats.transfer.d2d_operations,
            "d2d_bytes": stats.transfer.d2d_bytes,
            "staging_allocation_operations": stats.staging_allocation_operations,
        },
        "live_allocation_bytes": stats.live_allocation_bytes,
        "allocation_high_water_bytes": stats.allocation_high_water_bytes,
    })
}

fn host_spmv(matrix: &CsrMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); matrix.num_rows];
    for (row, target) in output.iter_mut().enumerate() {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            *target += matrix.values[entry] * input[matrix.col_indices[entry]];
        }
    }
    output
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let scale = expected
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
        / scale
}

fn exact_equal(actual: &[Complex64], expected: &[Complex64]) -> bool {
    actual.iter().zip(expected).all(|(left, right)| {
        left.re.to_bits() == right.re.to_bits() && left.im.to_bits() == right.im.to_bits()
    })
}

fn quadratic_work(input: &[Complex64], output: &[Complex64]) -> f64 {
    input
        .iter()
        .zip(output)
        .map(|(left, right)| (left.conj() * right).re)
        .sum()
}

fn solution_state(solution: &AxisymmetricLinearSolution) -> Vec<Complex64> {
    let mut state = solution.pressure_rms_pa.clone();
    for displacement in &solution.displacement_rz_m {
        state.extend(displacement);
    }
    state.push(solution.current_rms_a);
    state
}

fn action_stats_valid(stats: &CudaRuntimeStatsSnapshot, label: &str) -> Result<(), String> {
    if stats.kernel_launches + stats.library_operations == 0 {
        return Err(format!("{label} did not submit native CUDA work"));
    }
    if stats.allocation_operations != 0 || stats.staging_allocation_operations != 0 {
        return Err(format!("{label} allocated device/staging state: {stats:?}"));
    }
    Ok(())
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    // Consume the SM1 production record first.  This is a real native SM120
    // predecessor, not a copied dimension constant or synthetic matrix.
    let sm1_record = run_production_frequency_step_cuda_contract()?;
    if sm1_record.get("status").and_then(Value::as_str) != Some("passed")
        || sm1_record.get("actual_backend").and_then(Value::as_str) != Some("cuda")
    {
        return Err("SM1 production predecessor did not publish a native CUDA pass".into());
    }

    let model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig::default())
        .map_err(|error| format!("SM2 axisymmetric model build failed: {error}"))?;
    let voltage = Complex64::new(1.0, 0.0);
    let solutions = FREQUENCIES_HZ
        .iter()
        .map(|&frequency_hz| {
            model
                .solve_frequency_direct_reference(frequency_hz, voltage)
                .map_err(|error| {
                    format!("SM2 CPU/f64 production response at {frequency_hz} Hz failed: {error}")
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if solutions.len() != FREQUENCIES_HZ.len() {
        return Err("SM2 response sweep returned the wrong frequency count".into());
    }
    let audit = model
        .audit_sweep(&solutions)
        .map_err(|error| format!("SM2 production response audit failed: {error}"))?;
    if !audit.all_points_converged
        || audit.max_relative_linear_residual > SOLVER_RESIDUAL_TOLERANCE
        || audit.max_relative_power_residual > 5.0e-2
    {
        return Err(format!(
            "SM2 CPU response convergence/power gate failed: {audit:?}"
        ));
    }

    let matrices = FREQUENCIES_HZ
        .iter()
        .map(|&frequency_hz| {
            let point = sonium_fem::physics::acoustics::electroacoustics::HarmonicCoupledPoint {
                omega_rad_per_s: 2.0 * std::f64::consts::PI * frequency_hz,
                sound_speed_m_per_s: model.config.sound_speed_m_per_s,
                fluid_density_kg_per_m3: model.config.fluid_density_kg_per_m3,
            };
            model
                .coupled_system
                .assemble_harmonic(point)
                .map_err(|error| format!("assemble SM2 harmonic operator: {error}"))?
                .try_to_csr()
                .map_err(|error| format!("convert SM2 harmonic operator to CSR: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = &matrices[0];
    if matrices.iter().any(|matrix| {
        matrix.num_rows != first.num_rows
            || matrix.num_cols != first.num_cols
            || matrix.row_ptrs != first.row_ptrs
            || matrix.col_indices != first.col_indices
    }) {
        return Err("SM2 frequency sweep changed the production CSR structure".into());
    }
    let dimension = first.num_rows;
    let nnz = first.values.len();
    let sm1_dimension = sm1_record
        .get("dimension")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SM1 record has no numeric dimension".to_owned())?
        as usize;
    let sm1_nnz = sm1_record
        .get("nnz")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SM1 record has no numeric nnz".to_owned())? as usize;
    if sm1_dimension != dimension || sm1_nnz != nnz {
        return Err(format!(
            "SM1/SM2 production operator identity mismatch: SM1={sm1_dimension}x{sm1_nnz}, SM2={dimension}x{nnz}"
        ));
    }

    let states = solutions.iter().map(solution_state).collect::<Vec<_>>();
    if states.iter().any(|state| state.len() != dimension) {
        return Err("SM2 CPU response state does not match production operator dimension".into());
    }
    let cpu_outputs = matrices
        .iter()
        .zip(&states)
        .map(|(matrix, state)| host_spmv(matrix, state))
        .collect::<Vec<_>>();
    let cpu_work = cpu_outputs
        .iter()
        .zip(&states)
        .map(|(output, state)| quadratic_work(state, output))
        .collect::<Vec<_>>();

    let host = HostCsrRef::try_new(
        dimension,
        dimension,
        &first.row_ptrs,
        &first.col_indices,
        &first.values,
    )
    .map_err(|error| format!("SM2 production CSR admission failed: {error}"))?;
    let mut operator = PersistentDeviceCsr::try_from_host_csr_ref(&backend, host)
        .map_err(|error| format!("SM2 resident production CSR setup failed: {error}"))?;
    let mut input_device = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("SM2 response-state allocation failed: {error}"))?;
    let mut output_device = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("SM2 response-output allocation failed: {error}"))?;

    // Malformed value/update and stale-size input must fail before dispatch.
    let stale_values = vec![Complex64::default(); nnz.saturating_sub(1)];
    let stale_typed = matches!(
        operator.try_copy_values_from_host(&backend, &stale_values),
        Err(BackendError::InvalidArgument {
            operation: "persistent_csr_update_values",
            ..
        })
    );
    let malformed_input = backend
        .raw_alloc::<Complex64>(dimension.saturating_sub(1))
        .map_err(|error| format!("SM2 malformed input allocation: {error}"))?;
    let malformed_typed = matches!(
        operator.try_apply(&backend, &malformed_input, &mut output_device),
        Err(BackendError::InvalidArgument {
            operation: "persistent_csr_apply",
            ..
        })
    );
    drop(malformed_input);
    if !stale_typed || !malformed_typed {
        return Err("SM2 stale/malformed production input was not typed-rejected".into());
    }
    backend.reset_runtime_stats();

    let mut rows = Vec::with_capacity(FREQUENCIES_HZ.len());
    let mut cuda_outputs = Vec::with_capacity(FREQUENCIES_HZ.len());
    let mut cuda_work = Vec::with_capacity(FREQUENCIES_HZ.len());
    let mut max_operator_error = 0.0_f64;
    let mut total_action_h2d = 0_u64;
    let mut total_action_d2h = 0_u64;
    for (index, ((frequency_hz, matrix), state)) in FREQUENCIES_HZ
        .iter()
        .zip(&matrices)
        .zip(&states)
        .enumerate()
    {
        let value_update_start = backend.begin_runtime_stats();
        if index != 0 {
            operator
                .try_copy_values_from_host(&backend, &matrix.values)
                .map_err(|error| {
                    format!("SM2 resident value update at {frequency_hz} Hz: {error}")
                })?;
        }
        let value_update_h2d_operations = backend
            .runtime_stats_since(value_update_start)
            .transfer
            .h2d_operations;
        let action_start = backend.begin_runtime_stats();
        backend
            .try_copy_h2d(state, &mut input_device)
            .map_err(|error| format!("SM2 response-state upload at {frequency_hz} Hz: {error}"))?;
        operator
            .try_apply(&backend, &input_device, &mut output_device)
            .map_err(|error| format!("SM2 native response replay at {frequency_hz} Hz: {error}"))?;
        backend.try_synchronize().map_err(|error| {
            format!("SM2 response synchronization at {frequency_hz} Hz: {error}")
        })?;
        let mut cuda_output = vec![Complex64::default(); dimension];
        backend
            .try_copy_d2h(&output_device, &mut cuda_output)
            .map_err(|error| {
                format!("SM2 response output readback at {frequency_hz} Hz: {error}")
            })?;
        let error = relative_error(&cuda_output, &cpu_outputs[index]);
        if !error.is_finite() || error > OPERATOR_TOLERANCE {
            return Err(format!(
                "SM2 CUDA/CPU production operator residual at {frequency_hz} Hz is {error:.3e}"
            ));
        }
        let action_stats = backend.runtime_stats_since(action_start);
        action_stats_valid(&action_stats, &format!("SM2 action at {frequency_hz} Hz"))?;
        let work = quadratic_work(state, &cuda_output);
        max_operator_error = max_operator_error.max(error);
        total_action_h2d += action_stats.transfer.h2d_operations as u64;
        total_action_d2h += action_stats.transfer.d2h_operations as u64;
        cuda_work.push(work);
        cuda_outputs.push(cuda_output);
        rows.push(json!({
            "frequency_hz": frequency_hz,
            "dimension": dimension,
            "nnz": nnz,
            "cpu_solver_converged": solutions[index].solver.converged,
            "cpu_solver_relative_residual": solutions[index].solver.relative_residual,
            "cpu_observation_spl_db": solutions[index].observation_spl_db,
            "cpu_impedance_real_ohm": solutions[index].impedance_ohm.re,
            "cpu_impedance_imag_ohm": solutions[index].impedance_ohm.im,
            "cpu_power_closure_relative_error": solutions[index]
                .power
                .residual_w
                .abs()
                / solutions[index].power.input_w.abs().max(1.0e-15),
            "cpu_quadratic_work": cpu_work[index],
            "cuda_quadratic_work": work,
            "operator_relative_error": error,
            "value_update_h2d_operations": value_update_h2d_operations,
            "action_h2d_operations": action_stats.transfer.h2d_operations,
            "action_d2h_operations": action_stats.transfer.d2h_operations,
            "action_allocation_operations": action_stats.allocation_operations,
            "action_runtime": runtime_json(&action_stats),
        }));
    }

    let cpu_derivative = (cpu_work[2] - cpu_work[0]) / (FREQUENCIES_HZ[2] - FREQUENCIES_HZ[0]);
    let cuda_derivative = (cuda_work[2] - cuda_work[0]) / (FREQUENCIES_HZ[2] - FREQUENCIES_HZ[0]);
    let derivative_error = (cuda_derivative - cpu_derivative).abs() / cpu_derivative.abs().max(1.0);
    if !derivative_error.is_finite() || derivative_error > DERIVATIVE_TOLERANCE {
        return Err(format!(
            "SM2 CUDA quadratic-work derivative error {derivative_error:.3e}"
        ));
    }

    let last_output = cuda_outputs
        .last()
        .ok_or_else(|| "SM2 replay did not produce a final output".to_owned())?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply(&backend, &input_device, &mut output_device);
    let failure_error = match failure {
        Err(error @ BackendError::Launch { .. }) => error,
        Err(error) => return Err(format!("SM2 launch fault had wrong type: {error}")),
        Ok(()) => return Err("SM2 injected launch fault was not consumed".into()),
    };
    let failure_typed = matches!(
        failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("SM2 launch recovery failed: {error}"))?;
    let mut preserved_output = vec![Complex64::default(); dimension];
    backend
        .try_copy_d2h(&output_device, &mut preserved_output)
        .map_err(|error| format!("SM2 failed-output readback: {error}"))?;
    let output_preserved = exact_equal(&preserved_output, last_output);
    operator
        .try_apply(&backend, &input_device, &mut output_device)
        .map_err(|error| format!("SM2 launch retry failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("SM2 launch retry synchronization failed: {error}"))?;
    let mut retry_output = vec![Complex64::default(); dimension];
    backend
        .try_copy_d2h(&output_device, &mut retry_output)
        .map_err(|error| format!("SM2 launch retry readback failed: {error}"))?;
    let retry_error = relative_error(&retry_output, &cpu_outputs[2]);
    let retry_succeeded =
        retry_error <= OPERATOR_TOLERANCE && exact_equal(&retry_output, last_output);
    if !failure_typed || !output_preserved || !retry_succeeded {
        return Err(format!(
            "SM2 launch disposition failed: typed={failure_typed}, preserved={output_preserved}, retry={retry_succeeded}"
        ));
    }

    drop(operator);
    drop(input_device);
    drop(output_device);
    let teardown = backend.runtime_stats();
    if teardown.live_allocation_bytes != 0 || teardown.live_staging_bytes != 0 {
        return Err(format!(
            "SM2 CUDA teardown leaked resources: live={} staging={}",
            teardown.live_allocation_bytes, teardown.live_staging_bytes
        ));
    }
    let artifact_value = json!({
        "schema": "sonium-sim-speaker-modeling-sm2-breadth-evidence/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "hardware": cuda_hardware_provenance(),
        "backend": {
            "requested": "cuda",
            "actual": "cuda",
            "fallback_status": "none",
            "precision": "Complex64",
            "dispatch": "native-cuda-cusparse",
        },
        "sm1_production_output_consumed": {
            "case_id": sm1_record.get("case_id"),
            "status": sm1_record.get("status"),
            "actual_backend": sm1_record.get("actual_backend"),
            "dimension": sm1_dimension,
            "nnz": sm1_nnz,
            "operator": sm1_record.get("operator"),
        },
        "production_model": {
            "operator": "AxisymmetricLinearModel::CoupledSystem::assemble_harmonic",
            "scope": "magnetic-bias-informed acoustic/structural/electrical axisymmetric FEM operator; host assembly explicit",
            "dimension": dimension,
            "nnz": nnz,
            "frequencies_hz": FREQUENCIES_HZ,
            "rows": rows,
            "solver_max_relative_residual": audit.max_relative_linear_residual,
            "power_max_relative_residual": audit.max_relative_power_residual,
        },
        "observables": {
            "kind": "resident assembled-harmonic quadratic work derivative",
            "cpu_quadratic_work": cpu_work,
            "cuda_quadratic_work": cuda_work,
            "cpu_centered_derivative_per_hz": cpu_derivative,
            "cuda_centered_derivative_per_hz": cuda_derivative,
            "derivative_relative_error": derivative_error,
            "operator_max_relative_error": max_operator_error,
        },
        "admission": {
            "stale_values_typed": stale_typed,
            "malformed_input_typed": malformed_typed,
            "operator": "persistent_csr_apply",
        },
        "failure_recovery": {
            "typed_launch": failure_typed,
            "output_preserved": output_preserved,
            "retry_succeeded": retry_succeeded,
            "context_recovery": "reusable resident context",
            "partial_result_published": false,
            "retry_relative_error": retry_error,
        },
        "residency": {
            "structure_uploaded_once": true,
            "frequency_value_updates": FREQUENCIES_HZ.len() - 1,
            "response_state_updates": FREQUENCIES_HZ.len(),
            "action_h2d_operations": total_action_h2d,
            "action_d2h_operations": total_action_d2h,
            "action_allocation_operations": rows
                .iter()
                .map(|row| row["action_allocation_operations"].as_u64().unwrap_or(0))
                .sum::<u64>(),
            "allocation_high_water_bytes": teardown.allocation_high_water_bytes,
            "live_device_bytes_after_teardown": teardown.live_allocation_bytes,
            "live_staging_bytes_after_teardown": teardown.live_staging_bytes,
        },
        "counts": {"executed": 5, "passed": 5, "unsupported": 0},
        "scope_boundary": "bounded production axisymmetric response/operator replay; no device-resident FEM assembly, full-order product solve, nonlinear material CUDA, or multi-GPU claim",
    });
    let bytes = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|error| format!("serialize SM2 breadth artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact(ARTIFACT, &bytes)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference(
            "SM1 production CSR plus independent CPU/f64 axisymmetric response and host CSR oracle",
        )
        .reference_checksum("analytic/deterministic-fixture:speaker-sm2-axisymmetric-breadth-v1")
        .input_checksum("sha256:axisymmetric-linear-config-default-v1")
        .dimension("production axisymmetric coupled FEM operator, three-frequency response states")
        .formulation(
            "resident CUDA CSR response replay with quadratic-work derivative and typed retry",
        )
        .tolerance("operator residual <= 1e-8; derivative <= 1e-8; CPU solve residual <= 1e-6")
        .metric("production_dimension", dimension as f64)
        .metric("production_nnz", nnz as f64)
        .metric("operator_max_relative_error", max_operator_error)
        .metric("quadratic_work_derivative_relative_error", derivative_error)
        .metric(
            "cpu_solver_max_relative_residual",
            audit.max_relative_linear_residual,
        )
        .metric(
            "cpu_power_max_relative_residual",
            audit.max_relative_power_residual,
        )
        .metric("stale_values_typed", if stale_typed { 1.0 } else { 0.0 })
        .metric(
            "malformed_input_typed",
            if malformed_typed { 1.0 } else { 0.0 },
        )
        .metric(
            "launch_output_preserved",
            if output_preserved { 1.0 } else { 0.0 },
        )
        .metric("teardown_live_bytes", teardown.live_allocation_bytes as f64)
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/speaker_modeling_sm2_breadth_cuda.rs",
            "bounded native SM120 axisymmetric response/operator and derivative evidence",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            teardown,
            QaExecutionCounts {
                executed: 5,
                passed: 5,
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
            eprintln!("SM2 breadth QA serialization failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
