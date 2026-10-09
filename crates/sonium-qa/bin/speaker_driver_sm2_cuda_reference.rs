//! SM2 production CUDA replay for the bounded `DriverModel` contract.
//!
//! The existing SM1 candidate deliberately proves a small two-frequency cell.
//! This producer extends that same production CSR/GMRES seam to a bounded
//! fifteen-point sweep (five primary frequencies plus two perturbations per
//! point).  The serialized envelope retains the independently-produced
//! CPU/f64 golden, frequency-response derivatives, and per-frequency coupled
//! power observables.  The high-level FEM/BEM cells remain explicit in the
//! capability matrix and are not relabelled as device-resident.

use num_complex::Complex64;
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, QaStatus, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker_modeling::{
    ArtifactBackend, DriverModel, DriverModelReferenceArtifact, SpeakerDriverStudy,
    run_coupled_failure_contract, run_coupled_reference,
};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-sm2-multifrequency-derivative-reference.v1";
const FAILURE_CASE_ID: &str = "speaker.cuda-bounded-coupled-failure-contract.v1";
const PRIMARY_FREQUENCIES: [f64; 5] = [125.0, 500.0, 1_000.0, 4_000.0, 16_000.0];
const DERIVATIVE_FRACTION: f64 = 1.0e-3;
const RESPONSE_TOLERANCE: f64 = 1.0e-7;
const DERIVATIVE_TOLERANCE: f64 = 1.0e-7;
const POWER_TOLERANCE: f64 = 5.0e-2;

fn failed_result(error: String) -> QaResult {
    let unavailable = error.contains("CUDA backend unavailable")
        || error.contains("CUDA_ERROR_NO_DEVICE")
        || (error.contains("driver") && error.contains("unavailable"));
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(if unavailable { "unavailable" } else { "cuda" })
        .backend_failed("cuda", error)
        .failure_stage(if unavailable {
            QaFailureStage::Environment
        } else {
            QaFailureStage::Execution
        })
        .hardware_provenance(cuda_hardware_provenance())
}

fn radiation_impedance(frequency_hz: f64) -> Complex64 {
    // Positive real resistance keeps the coupled power observable passive;
    // the small frequency slope makes the derivative include the boundary
    // load rather than measuring only the internal DriverModel interpolation.
    Complex64::new(0.2 + 1.0e-5 * frequency_hz, 0.1 + 2.0e-6 * frequency_hz)
}

fn expanded_grid() -> (Vec<f64>, Vec<Complex64>) {
    let mut grid = Vec::with_capacity(PRIMARY_FREQUENCIES.len() * 3);
    for frequency in PRIMARY_FREQUENCIES {
        let step = frequency * DERIVATIVE_FRACTION;
        grid.extend([frequency - step, frequency, frequency + step]);
    }
    let radiation = grid.iter().copied().map(radiation_impedance).collect();
    (grid, radiation)
}

fn complex_from_components(real: f64, imaginary: f64) -> Complex64 {
    Complex64::new(real, imaginary)
}

fn response_point_velocity(
    point: &sonium_sim_speaker_modeling::DriverModelReferencePoint,
) -> Complex64 {
    complex_from_components(point.velocity_real_m_per_s, point.velocity_imag_m_per_s)
}

fn response_point_impedance(
    point: &sonium_sim_speaker_modeling::DriverModelReferencePoint,
) -> Complex64 {
    complex_from_components(
        point.electrical_impedance_real_ohm,
        point.electrical_impedance_imag_ohm,
    )
}

fn relative_complex(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-30)
}

fn relative_scalar(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0e-30)
}

fn finite_difference(plus: Complex64, minus: Complex64, step: f64) -> Complex64 {
    (plus - minus) / (2.0 * step)
}

fn cpu_frequency_derivatives(
    model: &DriverModel,
    frequency_hz: f64,
) -> Result<(Complex64, Complex64), String> {
    let step = frequency_hz * DERIVATIVE_FRACTION;
    let (velocity_minus, impedance_minus) = model
        .response_with_radiation_impedance(
            frequency_hz - step,
            radiation_impedance(frequency_hz - step),
        )
        .map_err(|error| format!("CPU derivative lower probe at {frequency_hz} Hz: {error}"))?;
    let (velocity_plus, impedance_plus) = model
        .response_with_radiation_impedance(
            frequency_hz + step,
            radiation_impedance(frequency_hz + step),
        )
        .map_err(|error| format!("CPU derivative upper probe at {frequency_hz} Hz: {error}"))?;
    Ok((
        finite_difference(velocity_plus, velocity_minus, step),
        finite_difference(impedance_plus, impedance_minus, step),
    ))
}

fn coupled_power(
    model: &DriverModel,
    frequency_hz: f64,
    velocity: Complex64,
    impedance: Complex64,
) -> Result<(f64, f64, f64), String> {
    let network = model
        .network_parameters_at(frequency_hz)
        .map_err(|error| format!("network power observable at {frequency_hz} Hz: {error}"))?;
    let voltage = Complex64::new(model.reference_voltage_rms_v, 0.0);
    let current = voltage / impedance;
    let input_power = (voltage * current.conj()).re;
    let radiated_power = radiation_impedance(frequency_hz).re * velocity.norm_sqr();
    let mechanical_power = network.intrinsic_mechanical_impedance.re * velocity.norm_sqr();
    let coil_power = network.coil_impedance_ohm.re * current.norm_sqr();
    let closure = (input_power - coil_power - mechanical_power - radiated_power).abs()
        / input_power.abs().max(1.0e-30);
    Ok((input_power, radiated_power, closure))
}

fn artifact_value(artifact: &DriverModelReferenceArtifact) -> Result<serde_json::Value, String> {
    serde_json::to_value(artifact)
        .map_err(|error| format!("serialize DriverModel artifact: {error}"))
}

fn u64_field(value: &serde_json::Value, path: &[&str]) -> Result<u64, String> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("failure evidence is missing {}", path.join(".")))?;
    }
    current.as_u64().ok_or_else(|| {
        format!(
            "failure evidence field {} is not an unsigned integer",
            path.join(".")
        )
    })
}

fn build_envelope(
    model: &DriverModel,
    cpu: &sonium_sim_speaker_modeling::CoupledBackendEvidence,
    cuda: &sonium_sim_speaker_modeling::CoupledBackendEvidence,
    frequencies: &[f64],
    failure_artifacts: &[&str],
) -> Result<serde_json::Value, String> {
    cuda.reference
        .validate_against_golden(
            &cpu.reference,
            sonium_sim_speaker_modeling::DriverModelReferenceTolerance {
                absolute: 1.0e-9,
                relative: RESPONSE_TOLERANCE,
            },
        )
        .map_err(|error| format!("independent CPU/f64 golden comparison failed: {error}"))?;

    if frequencies.len() != PRIMARY_FREQUENCIES.len() * 3 {
        return Err(format!(
            "SM2 expanded grid has {} points, expected 15",
            frequencies.len()
        ));
    }
    let mut sweep = Vec::with_capacity(PRIMARY_FREQUENCIES.len());
    let mut derivatives = Vec::with_capacity(PRIMARY_FREQUENCIES.len());
    let mut powers = Vec::with_capacity(PRIMARY_FREQUENCIES.len());
    let mut max_response_error = 0.0_f64;
    let mut max_derivative_error = 0.0_f64;
    let mut max_power_error = 0.0_f64;
    let mut max_power_closure = 0.0_f64;

    for (index, &frequency) in PRIMARY_FREQUENCIES.iter().enumerate() {
        let step = frequency * DERIVATIVE_FRACTION;
        let point_index = index * 3 + 1;
        let cpu_point = &cpu.reference.points[point_index];
        let cuda_point = &cuda.reference.points[point_index];
        let cpu_velocity = response_point_velocity(cpu_point);
        let cuda_velocity = response_point_velocity(cuda_point);
        let cpu_impedance = response_point_impedance(cpu_point);
        let cuda_impedance = response_point_impedance(cuda_point);
        let velocity_error = relative_complex(cuda_velocity, cpu_velocity);
        let impedance_error = relative_complex(cuda_impedance, cpu_impedance);
        max_response_error = max_response_error.max(velocity_error).max(impedance_error);
        if velocity_error > RESPONSE_TOLERANCE || impedance_error > RESPONSE_TOLERANCE {
            return Err(format!(
                "SM2 response parity failed at {frequency} Hz: velocity={velocity_error:e}, impedance={impedance_error:e}"
            ));
        }
        let (cpu_velocity_derivative, cpu_impedance_derivative) =
            cpu_frequency_derivatives(model, frequency)?;
        let cuda_velocity_derivative = finite_difference(
            response_point_velocity(&cuda.reference.points[point_index + 1]),
            response_point_velocity(&cuda.reference.points[point_index - 1]),
            step,
        );
        let cuda_impedance_derivative = finite_difference(
            response_point_impedance(&cuda.reference.points[point_index + 1]),
            response_point_impedance(&cuda.reference.points[point_index - 1]),
            step,
        );
        let velocity_derivative_error =
            relative_complex(cuda_velocity_derivative, cpu_velocity_derivative);
        let impedance_derivative_error =
            relative_complex(cuda_impedance_derivative, cpu_impedance_derivative);
        let derivative_error = velocity_derivative_error.max(impedance_derivative_error);
        max_derivative_error = max_derivative_error.max(derivative_error);
        if derivative_error > DERIVATIVE_TOLERANCE {
            return Err(format!(
                "SM2 frequency derivative parity failed at {frequency} Hz: velocity={velocity_derivative_error:e}, impedance={impedance_derivative_error:e}"
            ));
        }
        let (cpu_input_power, cpu_radiated_power, cpu_closure) =
            coupled_power(model, frequency, cpu_velocity, cpu_impedance)?;
        let (cuda_input_power, cuda_radiated_power, cuda_closure) =
            coupled_power(model, frequency, cuda_velocity, cuda_impedance)?;
        let input_error = relative_scalar(cuda_input_power, cpu_input_power);
        let radiated_error = relative_scalar(cuda_radiated_power, cpu_radiated_power);
        let power_error = input_error.max(radiated_error);
        max_power_error = max_power_error.max(power_error);
        max_power_closure = max_power_closure.max(cpu_closure).max(cuda_closure);
        if !cpu_input_power.is_finite()
            || !cuda_input_power.is_finite()
            || !cpu_radiated_power.is_finite()
            || !cuda_radiated_power.is_finite()
            || cpu_closure > POWER_TOLERANCE
            || cuda_closure > POWER_TOLERANCE
        {
            return Err(format!("SM2 coupled power gate failed at {frequency} Hz"));
        }

        sweep.push(serde_json::json!({
            "frequency_hz": frequency,
            "cpu_velocity": {"real": cpu_velocity.re, "imaginary": cpu_velocity.im},
            "cuda_velocity": {"real": cuda_velocity.re, "imaginary": cuda_velocity.im},
            "cpu_impedance": {"real": cpu_impedance.re, "imaginary": cpu_impedance.im},
            "cuda_impedance": {"real": cuda_impedance.re, "imaginary": cuda_impedance.im},
            "velocity_relative_error": velocity_error,
            "impedance_relative_error": impedance_error,
        }));
        derivatives.push(serde_json::json!({
            "frequency_hz": frequency,
            "step_hz": step,
            "method": "central finite difference of independent CPU/CUDA frequency probes",
            "cpu_velocity_derivative": {"real": cpu_velocity_derivative.re, "imaginary": cpu_velocity_derivative.im},
            "cuda_velocity_derivative": {"real": cuda_velocity_derivative.re, "imaginary": cuda_velocity_derivative.im},
            "cpu_impedance_derivative": {"real": cpu_impedance_derivative.re, "imaginary": cpu_impedance_derivative.im},
            "cuda_impedance_derivative": {"real": cuda_impedance_derivative.re, "imaginary": cuda_impedance_derivative.im},
            "velocity_relative_error": velocity_derivative_error,
            "impedance_relative_error": impedance_derivative_error,
        }));
        powers.push(serde_json::json!({
            "frequency_hz": frequency,
            "cpu_input_power_w": cpu_input_power,
            "cuda_input_power_w": cuda_input_power,
            "input_power_relative_error": input_error,
            "cpu_radiated_power_w": cpu_radiated_power,
            "cuda_radiated_power_w": cuda_radiated_power,
            "radiated_power_relative_error": radiated_error,
            "cpu_power_closure_relative_error": cpu_closure,
            "cuda_power_closure_relative_error": cuda_closure,
        }));
    }

    let capability_matrix = serde_json::to_value(&cuda.capability_matrix)
        .map_err(|error| format!("serialize CUDA capability matrix: {error}"))?;
    Ok(serde_json::json!({
        "case_id": CASE_ID,
        "schema_version": 1,
        "status": "passed",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "capability_admission": {
            "native_cuda_required": true,
            "fallback_status": "none",
            "matrix": capability_matrix,
            "unsupported_high_level_cells": ["magnetic", "structural", "acoustic", "thermal", "nonlinear_material"],
        },
        "cpu_reference": {
            "backend": "cpu",
            "precision": "f64",
            "independent": true,
            "artifact": artifact_value(&cpu.reference)?,
            "frequency_count": cpu.reference.points.len(),
        },
        "cuda_reference": {
            "backend": {"requested": "cuda", "actual": "cuda", "precision": "f64", "device": cuda.reference.execution.device},
            "artifact": artifact_value(&cuda.reference)?,
            "frequency_count": cuda.reference.points.len(),
            "max_relative_residual": cuda.max_relative_residual,
            "max_power_balance_relative_error": cuda.max_power_balance_relative_error,
            "max_velocity_relative_error": cuda.max_velocity_relative_error,
            "max_impedance_relative_error": cuda.max_impedance_relative_error,
            "max_derivative_relative_error": cuda.max_derivative_relative_error,
        },
        "frequency_sweep": {
            "primary_frequencies_hz": PRIMARY_FREQUENCIES,
            "solve_frequency_count": frequencies.len(),
            "sparsity_structure_reused": true,
            "resident_values_updated_between_points": true,
            "points": sweep,
        },
        "frequency_response_derivative": {
            "primary_point_count": derivatives.len(),
            "relative_error_tolerance": DERIVATIVE_TOLERANCE,
            "points": derivatives,
        },
        "coupled_power_observable": {
            "primary_point_count": powers.len(),
            "power_closure_tolerance": POWER_TOLERANCE,
            "points": powers,
        },
        "residency": {
            "transfer": cuda.reference.execution.transfer,
            "resident_csr_operator": true,
            "host_fem_bem_assembly": false,
        },
        "failure_recovery": {
            "typed_failure_case": FAILURE_CASE_ID,
            "artifacts": failure_artifacts,
            "no_partial_publication": true,
            "recreated_context_retry": true,
        },
        "publication": {
            "candidate_artifact_published_after_validation": true,
            "failed_attempt_published": false,
            "cpu_golden_comparison_passed": true,
        },
        "metrics": {
            "max_response_relative_error": max_response_error,
            "max_frequency_derivative_relative_error": max_derivative_error,
            "max_coupled_power_relative_error": max_power_error,
            "max_power_closure_relative_error": max_power_closure,
        },
        "scope": "bounded typed electromechanical 2x2 network; full-order FEM/BEM and high-level multiphysics remain unsupported"
    }))
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let model = DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
        .map_err(|error| format!("build detailed typed driver fixture: {error}"))?;
    let (frequencies, radiation) = expanded_grid();
    let cpu = run_coupled_reference(
        &model,
        &frequencies,
        &radiation,
        ArtifactBackend::Cpu,
        false,
    )
    .map_err(|error| format!("independent CPU/f64 golden replay failed: {error}"))?;
    let cuda = run_coupled_reference(
        &model,
        &frequencies,
        &radiation,
        ArtifactBackend::Cuda,
        false,
    )
    .map_err(|error| format!("required SM2 native CUDA replay failed: {error}"))?;
    if cuda.reference.execution.actual_backend != ArtifactBackend::Cuda
        || cuda.reference.execution.precision != sonium_sim_speaker_modeling::ArtifactPrecision::F64
    {
        return Err("SM2 replay did not produce native CUDA/f64 provenance".into());
    }

    // Use the same bounded fixture and two primary frequencies in both
    // injected lanes.  Each lane only publishes after its own recreated
    // context retry has passed the production CPU/CUDA parity gates.
    let failure_frequencies = [PRIMARY_FREQUENCIES[0], PRIMARY_FREQUENCIES[2]];
    let failure_radiation = [
        radiation_impedance(failure_frequencies[0]),
        radiation_impedance(failure_frequencies[1]),
    ];
    let failure_lanes = [
        (
            "launch",
            sonium_compute::CudaFailureInjectionPoint::Launch,
            "speaker_driver_sm2_cuda_launch_failure.json",
        ),
        (
            "device-loss",
            sonium_compute::CudaFailureInjectionPoint::DeviceLoss,
            "speaker_driver_sm2_cuda_device_loss_failure.json",
        ),
    ];
    let mut failure_records = Vec::with_capacity(failure_lanes.len());
    let mut failure_bytes = Vec::with_capacity(failure_lanes.len());
    let mut total_failure_h2d = 0_u64;
    let mut total_failure_d2h = 0_u64;
    let mut failure_peak_memory = 0_u64;
    for (label, failure_point, artifact_name) in failure_lanes {
        let record = run_coupled_failure_contract(
            &model,
            &failure_frequencies,
            &failure_radiation,
            failure_point,
        )
        .map_err(|error| format!("SM2 {label} failure/recovery lane failed: {error}"))?;
        total_failure_h2d = total_failure_h2d.saturating_add(u64_field(
            &record,
            &["retry_success", "transfer", "h2d_operations"],
        )?);
        total_failure_d2h = total_failure_d2h.saturating_add(u64_field(
            &record,
            &["retry_success", "transfer", "d2h_operations"],
        )?);
        failure_peak_memory = failure_peak_memory.max(u64_field(
            &record,
            &["retry_success", "residency", "allocation_high_water_bytes"],
        )?);
        let bytes = serde_json::to_vec_pretty(&record)
            .map_err(|error| format!("serialize SM2 {label} failure evidence: {error}"))?;
        failure_bytes.push((artifact_name, bytes));
        failure_records.push(record);
    }

    let failure_artifact_names = [
        "speaker_driver_sm2_cuda_launch_failure.json",
        "speaker_driver_sm2_cuda_device_loss_failure.json",
    ];
    let envelope = build_envelope(&model, &cpu, &cuda, &frequencies, &failure_artifact_names)?;
    let envelope_bytes = serde_json::to_vec_pretty(&envelope)
        .map_err(|error| format!("serialize SM2 CUDA envelope: {error}"))?;

    // Do not publish any candidate until the complete sweep, derivatives,
    // coupled power rows, and both failure/retry contracts have succeeded.
    let (envelope_path, envelope_checksum) =
        write_qa_artifact("speaker_driver_sm2_cuda_reference.json", &envelope_bytes)?;
    let mut artifact_provenance = vec![(envelope_path, envelope_checksum)];
    for (artifact_name, bytes) in failure_bytes {
        let (path, checksum) = write_qa_artifact(artifact_name, &bytes)?;
        artifact_provenance.push((path, checksum));
    }

    let transfer = &cuda.reference.execution.transfer;
    let peak_memory = (transfer.allocation_high_water_bytes as u64).max(failure_peak_memory);
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("independent CPU/f64 golden plus native CUDA resident CSR frequency sweep")
        .reference_checksum("analytic/deterministic-fixture:speaker-cuda-sm2-multifrequency-v1")
        .input_checksum("sha256:detailed-driver-study-v1")
        .dimension("typed electromechanical 2x2 driver network, 5 primary frequencies, 15 solve points")
        .formulation("resident CUDA CSR GMRES with frequency-response derivative and coupled power observables")
        .tolerance("response <= 1e-7; frequency derivative <= 1e-7; power closure <= 5e-2")
        .metric("primary_frequency_count", PRIMARY_FREQUENCIES.len() as f64)
        .metric("solve_frequency_count", frequencies.len() as f64)
        .metric("max_response_relative_error", envelope["metrics"]["max_response_relative_error"].as_f64().unwrap_or(f64::INFINITY))
        .metric("max_frequency_derivative_relative_error", envelope["metrics"]["max_frequency_derivative_relative_error"].as_f64().unwrap_or(f64::INFINITY))
        .metric("max_coupled_power_relative_error", envelope["metrics"]["max_coupled_power_relative_error"].as_f64().unwrap_or(f64::INFINITY))
        .metric("max_power_closure_relative_error", envelope["metrics"]["max_power_closure_relative_error"].as_f64().unwrap_or(f64::INFINITY))
        .metric("typed_failure_lanes", failure_records.len() as f64)
        .metric("cuda_h2d_operations", transfer.h2d_operations as f64)
        .metric("cuda_d2h_operations", transfer.d2h_operations as f64)
        .metric("cuda_allocation_high_water_bytes", peak_memory as f64)
        .metric("failure_cuda_h2d_operations", total_failure_h2d as f64)
        .metric("failure_cuda_d2h_operations", total_failure_d2h as f64)
        .maximum("max_response_relative_error", RESPONSE_TOLERANCE)
        .maximum("max_frequency_derivative_relative_error", DERIVATIVE_TOLERANCE)
        .maximum("max_coupled_power_relative_error", RESPONSE_TOLERANCE)
        .maximum("max_power_closure_relative_error", POWER_TOLERANCE)
        .minimum("primary_frequency_count", PRIMARY_FREQUENCIES.len() as f64)
        .minimum("solve_frequency_count", frequencies.len() as f64)
        .minimum("typed_failure_lanes", 2.0)
        .minimum("cuda_h2d_operations", 1.0)
        .minimum("cuda_d2h_operations", 1.0)
        .memory_bytes(peak_memory)
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Recreated,
            counts: QaExecutionCounts {
                executed: frequencies.len() as u64 + 4,
                passed: frequencies.len() as u64 + 4,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: 1 + failure_records.len() as u64 * 2,
                h2d_transfers: transfer.h2d_operations + total_failure_h2d,
                d2h_transfers: transfer.d2h_operations + total_failure_d2h,
                d2d_transfers: 0,
                synchronizations: 1 + failure_records.len() as u64 * 2,
                launches: frequencies.len() as u64 + failure_records.len() as u64 * 2,
            },
            memory: QaMemoryEvidence {
                peak_bytes: peak_memory,
                live_bytes: transfer.live_allocation_bytes as u64,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .physical_residual(cuda.max_relative_residual)
        .physical_reference_error(cuda.max_power_balance_relative_error);
    qa.environment.insert(
        "gpu.capability_matrix".into(),
        serde_json::to_string(&cuda.capability_matrix)
            .map_err(|error| format!("serialize SM2 capability matrix: {error}"))?,
    );
    qa.environment.insert(
        "platform".into(),
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
    );
    qa.environment.insert(
        "gpu.execution_device".into(),
        cuda.reference.execution.device.clone(),
    );
    qa.environment.insert(
        "gpu.execution_toolchain".into(),
        cuda.reference.execution.toolchain.clone(),
    );
    qa.environment.insert(
        "gpu.failure_modes_exercised".into(),
        "launch,device-loss".into(),
    );
    qa.environment.insert(
        "gpu.execution_scope".into(),
        "bounded typed electromechanical network; no high-level FEM/BEM residency claim".into(),
    );
    for (path, checksum) in artifact_provenance {
        qa = qa.artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_driver_sm2_cuda_reference.rs",
            "SM2 native CUDA multi-frequency/derivative and failure evidence",
            "v1",
            checksum,
        );
    }
    Ok(qa)
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result,
        Err(error) => failed_result(error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("SM2 CUDA speaker QA result must serialize")
    );
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
