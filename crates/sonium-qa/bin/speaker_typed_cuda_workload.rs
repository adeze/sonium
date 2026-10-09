//! Required production-scale CUDA typed-coupling workload.
//!
//! This gate exercises the product speaker entry point with two live typed
//! DriverModels.  The BEM radiation probes, N-port network, and final field
//! solve must all use the CUDA backend; the CPU f64 path is retained only as
//! the independent residual, observable, power, and voltage-sensitivity
//! reference.

use sha2::{Digest, Sha256};
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker::{cea2034_output, coupling::RadiationMatrix, mlg_loader, simulation};
use sonium_sim_speaker_modeling::{DriverModel, SpeakerDriverStudy};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-bounded-typed-production.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;
const PHYSICAL_TOLERANCE: f64 = 1.0e-3;
const PARITY_TOLERANCE: f64 = 2.0e-3;
const DERIVATIVE_ASSEMBLY_TOLERANCE: f64 = 5.0e-3;
const RADIATION_RECIPROCITY_TOLERANCE: f64 = 1.0e-1;
const RADIATION_PASSIVITY_TOLERANCE: f64 = 5.0e-2;

fn failed_result(error: String) -> QaResult {
    let unavailable =
        error.contains("CUDA backend unavailable") || error.contains("CUDA_ERROR_NO_DEVICE");
    let fallback = error.contains("fallback");
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(if fallback {
            "host-fallback"
        } else if unavailable {
            "unavailable"
        } else {
            "cuda"
        })
        .backend_failed("cuda", error)
        .failure_stage(if unavailable || fallback {
            QaFailureStage::Environment
        } else {
            QaFailureStage::Execution
        })
        .hardware_provenance(cuda_hardware_provenance())
}

fn checksum(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn output_report(report: &simulation::GpuExecutionReport) -> Result<(String, String), String> {
    let relative = "speaker_typed_cuda_execution_report.json";
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize typed production CUDA report: {error}"))?;
    write_qa_artifact(relative, &bytes)
}

fn require_fail_closed_configuration_cells(
    report: &simulation::GpuExecutionReport,
) -> Result<(), String> {
    let configurations = &report.capability_matrix.speaker_configurations;
    for (name, cell) in [
        ("crossover_phase", &configurations.crossover_phase),
        ("coordinate_rotation", &configurations.coordinate_rotation),
        (
            "body_fitted_projected_reanalysis",
            &configurations.body_fitted_projected_reanalysis,
        ),
    ] {
        if cell.state != simulation::GpuCapabilityState::Unsupported
            || cell.reason.trim().is_empty()
        {
            return Err(format!(
                "typed CUDA configuration capability cell {name} is not explicitly fail-closed: state={:?}, reason={:?}",
                cell.state, cell.reason
            ));
        }
    }
    Ok(())
}

fn validate_radiation_gates(
    result: &simulation::SimulationResult,
) -> Result<(f64, f64, f64), String> {
    if result.radiation_matrices.len() != result.frequencies.len()
        || result.radiation_matrix_diagnostics.len() != result.frequencies.len()
        || result.driver_velocities_bem.len() != result.frequencies.len()
    {
        return Err(
            "typed production CUDA result has inconsistent radiation output lengths".into(),
        );
    }

    let mut max_reciprocity_defect: f64 = 0.0;
    let mut minimum_hermitian_pivot = f64::INFINITY;
    let mut minimum_radiated_power = f64::INFINITY;

    for (frequency_index, frequency) in result.frequencies.iter().copied().enumerate() {
        let values = result.radiation_matrices[frequency_index]
            .as_ref()
            .ok_or_else(|| {
                format!("typed production CUDA radiation matrix missing at {frequency:.3} Hz")
            })?;
        let matrix = RadiationMatrix::new(values.clone()).map_err(|error| {
            format!("typed production CUDA radiation matrix invalid at {frequency:.3} Hz: {error}")
        })?;
        let diagnostics = matrix.validate(
            RADIATION_RECIPROCITY_TOLERANCE,
            RADIATION_PASSIVITY_TOLERANCE,
        )
        .map_err(|error| {
            format!(
                "typed production CUDA radiation physical gate failed at {frequency:.3} Hz: {error}"
            )
        })?;
        let power = matrix
            .active_power_w(&result.driver_velocities_bem[frequency_index])
            .map_err(|error| {
                format!(
                    "typed production CUDA radiation power failed at {frequency:.3} Hz: {error}"
                )
            })?;
        let scale = result.driver_velocities_bem[frequency_index]
            .iter()
            .map(|velocity| velocity.norm_sqr())
            .sum::<f64>()
            .max(1.0e-12)
            * matrix
                .values()
                .iter()
                .flatten()
                .map(|value| value.norm())
                .fold(0.0, f64::max)
                .max(1.0e-12);
        if !power.is_finite() || power < -RADIATION_PASSIVITY_TOLERANCE * scale {
            return Err(format!(
                "typed production CUDA radiation load is not passive at {frequency:.3} Hz: power={power:.3e} W, scale={scale:.3e} W"
            ));
        }

        if let Some(reported) = result.radiation_matrix_diagnostics[frequency_index] {
            if (reported.reciprocity_defect - diagnostics.reciprocity_defect).abs() > 1.0e-12
                || (reported.minimum_hermitian_pivot - diagnostics.minimum_hermitian_pivot).abs()
                    > 1.0e-12
            {
                return Err(format!(
                    "typed production CUDA radiation diagnostics mismatch at {frequency:.3} Hz"
                ));
            }
        } else {
            return Err(format!(
                "typed production CUDA radiation diagnostics missing at {frequency:.3} Hz"
            ));
        }

        max_reciprocity_defect = max_reciprocity_defect.max(diagnostics.reciprocity_defect);
        minimum_hermitian_pivot = minimum_hermitian_pivot.min(diagnostics.minimum_hermitian_pivot);
        minimum_radiated_power = minimum_radiated_power.min(power);
    }

    Ok((
        max_reciprocity_defect,
        minimum_hermitian_pivot,
        minimum_radiated_power,
    ))
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let source = include_str!("../../sonium-sim-speaker/examples/two_way_speaker.mlg");
    let mut model = mlg_loader::load_model(source, "two_way_speaker.mlg")
        .map_err(|error| format!("load bounded production speaker fixture: {error}"))?;
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let driver = DriverModel::from_study(&study)
        .map_err(|error| format!("build typed production driver fixture: {error}"))?;
    model.woofer.radius = study.geometry.diaphragm_radius_m;
    model.tweeter.radius = study.geometry.diaphragm_radius_m;
    model.woofer.driver_model = Some(driver.clone());
    model.tweeter.driver_model = Some(driver);
    model.simulation.freq_start = 500.0;
    model.simulation.freq_end = 1000.0;
    model.simulation.freq_points = 2;
    model.simulation.max_mesh_frequency = Some(1000.0);

    let eval_points = cea2034_output::generate_eval_points([0.0, 0.0, 0.0]);
    let mut config = simulation::SolverConfig::default();
    config.tolerance = RESIDUAL_TOLERANCE;
    config.max_iterations = 400;
    let started = Instant::now();
    let (result, report) =
        simulation::run_simulation_cuda_with_report(&model, &eval_points, &config)
            .map_err(|error| format!("required typed production CUDA workload failed: {error}"))?;

    if report.requested_backend != simulation::SimulationBackend::Cuda
        || report.actual_backend != simulation::SimulationBackend::Cuda
        || !matches!(
            report.capability_matrix.coupled_driver_solve,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
        || !matches!(
            report.capability_matrix.field_evaluation,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
        || !report.cache.field_geometry_resident
        || !report.cache.mesh_resident_across_frequencies
        || !report.cache.operator_structure_resident
        || !report.cache.coupled_operator_structure_resident
        || !report.cache.coupled_rhs_buffer_reused
        || !report.cache.coupled_solution_buffer_reused
        || report.cache.matrix_value_updates + 1 != report.frequencies
        || report.cache.preconditioner_updates + 1 != report.frequencies
        || report.cache.coupled_matrix_value_updates + 1 != report.frequencies
        || report.cache.coupled_preconditioner_updates + 1 != report.frequencies
        || !report.cache.coupled_block_rhs_resident
        || !report.cache.coupled_block_solution_resident
        || report.cache.coupled_block_rhs_count != model.drivers().len()
        || report.cache.coupled_block_iterations == 0
        || !matches!(
            report
                .capability_matrix
                .speaker_configurations
                .typed_multi_driver
                .state,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
    {
        return Err(format!(
            "typed production CUDA provenance/capability mismatch: requested={:?}, actual={:?}, field={:?}, field_resident={}, mesh_resident={}, operator_resident={}, coupled_operator_resident={}, coupled_rhs_reused={}, coupled_solution_reused={}, matrix_updates={}, preconditioner_updates={}, coupled_matrix_updates={}, coupled_preconditioner_updates={}, block_rhs_resident={}, block_solution_resident={}, block_rhs_count={}, block_iterations={}, coupled={:?}, typed_config={:?}",
            report.requested_backend,
            report.actual_backend,
            report.capability_matrix.field_evaluation,
            report.cache.field_geometry_resident,
            report.cache.mesh_resident_across_frequencies,
            report.cache.operator_structure_resident,
            report.cache.coupled_operator_structure_resident,
            report.cache.coupled_rhs_buffer_reused,
            report.cache.coupled_solution_buffer_reused,
            report.cache.matrix_value_updates,
            report.cache.preconditioner_updates,
            report.cache.coupled_matrix_value_updates,
            report.cache.coupled_preconditioner_updates,
            report.cache.coupled_block_rhs_resident,
            report.cache.coupled_block_solution_resident,
            report.cache.coupled_block_rhs_count,
            report.cache.coupled_block_iterations,
            report.capability_matrix.coupled_driver_solve,
            report
                .capability_matrix
                .speaker_configurations
                .typed_multi_driver
                .state,
        ));
    }
    require_fail_closed_configuration_cells(&report)?;
    if report.residual_gate != simulation::ResidualGateStatus::Passed
        || report.max_relative_residual > RESIDUAL_TOLERANCE
        || report.max_physical_error > PHYSICAL_TOLERANCE
        || report.max_coupled_power_error > PARITY_TOLERANCE
        || report.max_coupled_observable_error > PARITY_TOLERANCE
        || report.max_coupled_derivative_error > PARITY_TOLERANCE
        || report.max_derivative_assembly_error > DERIVATIVE_ASSEMBLY_TOLERANCE
    {
        return Err(format!(
            "typed production CUDA acceptance gate mismatch: residual={:.3e}, physical={:.3e}, power={:.3e}, observable={:.3e}, derivative={:.3e}, derivative_assembly={:.3e}",
            report.max_relative_residual,
            report.max_physical_error,
            report.max_coupled_power_error,
            report.max_coupled_observable_error,
            report.max_coupled_derivative_error,
            report.max_derivative_assembly_error,
        ));
    }
    if result.radiation_matrices.iter().any(Option::is_none)
        || result.power_ledgers.iter().any(Option::is_none)
        || result
            .driver_velocities_bem
            .iter()
            .any(|velocities| velocities.len() != model.drivers().len())
        || result
            .driver_electrical_impedances_ohm
            .iter()
            .any(|impedances| impedances.iter().any(Option::is_none))
    {
        return Err("typed production CUDA result omitted coupled power, radiation, velocity, or impedance outputs".into());
    }
    let (max_radiation_reciprocity_defect, min_radiation_hermitian_pivot, min_radiated_power_w) =
        validate_radiation_gates(&result)?;
    if report.transfers.h2d_bytes == 0
        || report.transfers.d2h_bytes == 0
        || report.memory.allocation_high_water_bytes == 0
    {
        return Err(
            "typed production CUDA report omitted live device transfer or memory evidence".into(),
        );
    }
    let cea = cea2034_output::compute_spinorama_metrics(&result)
        .map_err(|error| format!("typed production CUDA CEA post-processing failed: {error}"))?;
    if !cea.preference_score.is_finite() {
        return Err("typed production CUDA CEA preference score is non-finite".into());
    }
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("serialize typed production CUDA evidence: {error}"))?;
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference(
            "two-frequency production BEM + typed N-port CUDA solve with independent CPU f64 gates",
        )
        .reference_checksum("analytic/deterministic-fixture:two-way-speaker-typed-cuda-v1")
        .input_checksum("sha256:embedded-two-way-speaker-mlg-v1+typed-driver-v1")
        .dimension("3D whole-speaker BEM, 2 typed drivers, 2 frequencies, 72 CEA-2034 points")
        .formulation(
            "CUDA resident Complex64 typed BEM CSR and analytic derivative channels + resident block-GMRES unit-load batch + CUDA typed N-port CSR/GMRES + CPU f64 parity",
        )
        .tolerance(
            "residual <= 1e-4; field <= 1e-3; derivative assembly <= 5e-3; coupled parity <= 2e-3; public radiation reciprocity/passivity contract",
        )
        .metric("max_relative_residual", report.max_relative_residual)
        .metric("max_physical_error", report.max_physical_error)
        .metric(
            "max_derivative_assembly_error",
            report.max_derivative_assembly_error,
        )
        .metric("max_coupled_power_error", report.max_coupled_power_error)
        .metric(
            "max_coupled_observable_error",
            report.max_coupled_observable_error,
        )
        .metric(
            "max_coupled_derivative_error",
            report.max_coupled_derivative_error,
        )
        .metric(
            "max_radiation_reciprocity_defect",
            max_radiation_reciprocity_defect,
        )
        .metric(
            "min_radiation_hermitian_pivot",
            min_radiation_hermitian_pivot,
        )
        .metric("min_radiated_power_w", min_radiated_power_w)
        .metric(
            "coupled_block_rhs_resident",
            report.cache.coupled_block_rhs_resident as u8 as f64,
        )
        .metric(
            "coupled_block_solution_resident",
            report.cache.coupled_block_solution_resident as u8 as f64,
        )
        .metric(
            "coupled_block_rhs_count",
            report.cache.coupled_block_rhs_count as f64,
        )
        .metric(
            "coupled_block_iterations",
            report.cache.coupled_block_iterations as f64,
        )
        .metric(
            "mesh_resident_across_frequencies",
            report.cache.mesh_resident_across_frequencies as u8 as f64,
        )
        .metric(
            "operator_structure_resident",
            report.cache.operator_structure_resident as u8 as f64,
        )
        .metric(
            "coupled_operator_structure_resident",
            report.cache.coupled_operator_structure_resident as u8 as f64,
        )
        .metric(
            "coupled_rhs_buffer_reused",
            report.cache.coupled_rhs_buffer_reused as u8 as f64,
        )
        .metric(
            "coupled_solution_buffer_reused",
            report.cache.coupled_solution_buffer_reused as u8 as f64,
        )
        .metric(
            "matrix_value_updates",
            report.cache.matrix_value_updates as f64,
        )
        .metric(
            "preconditioner_updates",
            report.cache.preconditioner_updates as f64,
        )
        .metric(
            "coupled_matrix_value_updates",
            report.cache.coupled_matrix_value_updates as f64,
        )
        .metric(
            "coupled_preconditioner_updates",
            report.cache.coupled_preconditioner_updates as f64,
        )
        .metric("cea_preference_score", cea.preference_score)
        .metric("active_dofs", report.active_dofs as f64)
        .metric("cuda_h2d_bytes", report.transfers.h2d_bytes as f64)
        .metric("cuda_d2h_bytes", report.transfers.d2h_bytes as f64)
        .maximum("max_relative_residual", RESIDUAL_TOLERANCE)
        .maximum("max_physical_error", PHYSICAL_TOLERANCE)
        .maximum(
            "max_derivative_assembly_error",
            DERIVATIVE_ASSEMBLY_TOLERANCE,
        )
        .maximum("max_coupled_power_error", PARITY_TOLERANCE)
        .maximum("max_coupled_observable_error", PARITY_TOLERANCE)
        .maximum("max_coupled_derivative_error", PARITY_TOLERANCE)
        .maximum(
            "max_radiation_reciprocity_defect",
            RADIATION_RECIPROCITY_TOLERANCE,
        )
        .minimum("cuda_h2d_bytes", 1.0)
        .minimum("cuda_d2h_bytes", 1.0)
        .timing_ms("total", report.timings.total_ms)
        .timing_ms("assembly", report.timings.assembly_ms)
        .timing_ms("solve", report.timings.solve_ms)
        .timing_ms("field", report.timings.field_ms)
        .timing_ms("reference", report.timings.reference_ms)
        .memory_bytes(report.memory.allocation_high_water_bytes as u64)
        .elapsed_ms(started.elapsed().as_millis() as u64);
    qa.environment.insert(
        "gpu.capability_matrix".into(),
        serde_json::to_string(&report.capability_matrix)
            .map_err(|error| format!("serialize typed capability matrix: {error}"))?,
    );
    qa.environment.insert(
        "gpu.execution_report_sha256".into(),
        checksum(&report_bytes),
    );
    let (path, digest) = output_report(&report)?;
    qa = qa
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_typed_cuda_workload.rs",
            "bounded typed production CUDA execution report",
            "v2",
            digest,
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Reused,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: u64::from(report.memory.allocation_high_water_bytes > 0),
                h2d_transfers: report.transfers.h2d_operations as u64,
                d2h_transfers: report.transfers.d2h_operations as u64,
                d2d_transfers: report.transfers.d2d_operations as u64,
                synchronizations: 0,
                launches: 1,
            },
            memory: QaMemoryEvidence {
                peak_bytes: report
                    .memory
                    .allocation_high_water_bytes
                    .saturating_add(report.memory.staging_high_water_bytes)
                    as u64,
                live_bytes: report
                    .memory
                    .live_allocation_bytes
                    .saturating_add(report.memory.live_staging_bytes)
                    as u64,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .physical_residual(report.max_relative_residual)
        .physical_reference_error(report.max_physical_error);
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
            .expect("typed production CUDA QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
