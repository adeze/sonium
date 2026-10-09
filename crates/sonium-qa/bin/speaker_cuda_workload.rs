//! Required bounded whole-speaker CUDA workload.
//!
//! This gate exercises the product entry point (mesh generation, CUDA
//! assembly, resident sparse solve, field evaluation, f64 residual/reference
//! gates, and CEA post-processing).  A missing device or any unsupported
//! phase is reported as failure/evidence; it is never converted to a CPU
//! result.

use num_complex::Complex64;
use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker::{cea2034_output, mlg_loader, simulation};
use sonium_sim_speaker_modeling::{
    CabinetDesign, CrossoverBranch, CrossoverKind, CrossoverNetwork, ProductDesign, ProductDriver,
    ProductWaveguideDesign,
};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-bounded-whole-workload.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-4;
const PHYSICAL_TOLERANCE: f64 = 1.0e-3;
const DERIVATIVE_TOLERANCE: f64 = 5.0e-3;

#[derive(Debug, Serialize)]
struct ReducedOrderPreviewEvidence {
    schema_version: u32,
    source_provenance: simulation::SimulationProvenance,
    preview: simulation::SpeakerPreviewArtifact,
}

fn checksum(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

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

fn output_report(report: &simulation::GpuExecutionReport) -> Result<(String, String), String> {
    let relative = "speaker_cuda_execution_report.json";
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize CUDA execution report: {error}"))?;
    write_qa_artifact(relative, &bytes)
}

fn output_adjoint_report(
    report: &simulation::GpuAdjointReport,
) -> Result<(String, String), String> {
    let relative = "speaker_cuda_adjoint_report.json";
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize CUDA adjoint report: {error}"))?;
    write_qa_artifact(relative, &bytes)
}

fn require_fail_closed_configuration_cells(
    report: &simulation::GpuExecutionReport,
) -> Result<(), String> {
    let configurations = &report.capability_matrix.speaker_configurations;
    for (name, cell) in [
        ("crossover_phase", &configurations.crossover_phase),
        ("coordinate_rotation", &configurations.coordinate_rotation),
        ("typed_multi_driver", &configurations.typed_multi_driver),
        (
            "body_fitted_projected_reanalysis",
            &configurations.body_fitted_projected_reanalysis,
        ),
    ] {
        if cell.state != simulation::GpuCapabilityState::Unsupported
            || cell.reason.trim().is_empty()
        {
            return Err(format!(
                "CUDA configuration capability cell {name} is not explicitly fail-closed: state={:?}, reason={:?}",
                cell.state, cell.reason
            ));
        }
    }
    Ok(())
}

fn product_driver(
    id: &str,
    position_m: [f64; 3],
    normal: [f64; 3],
    crossover_branch: usize,
) -> ProductDriver {
    ProductDriver {
        id: id.into(),
        radius_m: 0.025,
        position_m,
        normal,
        force_factor_tm: 5.0,
        coil_resistance_ohm: 5.0,
        coil_reactance_ohm: 1.0,
        mechanical_resistance_n_s_per_m: 0.5,
        mechanical_reactance_n_s_per_m: 2.0,
        crossover_branch,
    }
}

fn two_driver_product() -> ProductDesign {
    ProductDesign {
        cabinet: CabinetDesign::default(),
        drivers: vec![
            product_driver("woofer", [-0.07, -0.12, 0.0], [0.0, 0.0, 1.0], 0),
            product_driver("tweeter", [0.07, 0.12, 0.0], [0.0, 0.0, 1.0], 1),
        ],
        crossover: CrossoverNetwork {
            branches: vec![
                CrossoverBranch {
                    kind: CrossoverKind::LowPass,
                    corner_frequency_hz: 2_000.0,
                    ..CrossoverBranch::default()
                },
                CrossoverBranch {
                    kind: CrossoverKind::HighPass,
                    corner_frequency_hz: 2_000.0,
                    ..CrossoverBranch::default()
                },
            ],
        },
        waveguide: None,
        structural_acoustic: None,
    }
}

fn require_product_configuration_rejections(
    model: &sonium_sim_speaker::model::LoudspeakerModel,
) -> Result<usize, String> {
    let mut crossover = two_driver_product();
    crossover.crossover.branches[0].delay_s = 2.0e-6;

    let mut rotation = two_driver_product();
    rotation.drivers[0].normal = [0.1, 0.0, (1.0 - 0.1_f64.powi(2)).sqrt()];

    let mut typed_multi_driver = two_driver_product();
    typed_multi_driver.drivers.push(product_driver(
        "midrange",
        [0.0, -0.18, 0.0],
        [0.0, 0.0, 1.0],
        2,
    ));
    typed_multi_driver
        .crossover
        .branches
        .push(CrossoverBranch::default());

    let mut body_fitted = two_driver_product();
    body_fitted.waveguide = Some(ProductWaveguideDesign {
        driver_id: "tweeter".into(),
        horizontal_spline: vec![[0.0, 0.05], [0.5, 0.035], [1.0, 0.025]],
        vertical_spline: vec![[0.0, 0.04], [0.5, 0.03], [1.0, 0.025]],
        depth_m: 0.025,
        surface_specific_impedance_pa_s_per_m:
            sonium_sim_speaker_modeling::waveguide_design::DEFAULT_WAVEGUIDE_SPECIFIC_IMPEDANCE_PA_S_PER_M,
    });

    let configurations = [
        ("crossover_phase", crossover),
        ("coordinate_rotation", rotation),
        ("typed_multi_driver", typed_multi_driver),
        ("body_fitted_projected_reanalysis", body_fitted),
    ];
    let configuration_count = configurations.len();
    for (name, product) in configurations {
        product
            .validate()
            .map_err(|error| format!("representative {name} product is invalid: {error}"))?;
        let mut configured = model.clone();
        configured.product_design = Some(product);
        let error = match simulation::run_simulation_cuda_on_device(
            &configured,
            &[[0.0, 0.0, 1.0]],
            &simulation::SolverConfig::default(),
            0,
        ) {
            Ok(_) => {
                return Err(format!(
                    "CUDA {name} product configuration unexpectedly entered accelerator execution"
                ));
            }
            Err(error) => error,
        };
        if !error.to_string().contains("ProductDesign configurations") {
            return Err(format!(
                "CUDA {name} configuration failed without the published ProductDesign rejection: {error}"
            ));
        }
    }
    Ok(configuration_count)
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let source = include_str!("../../sonium-sim-speaker/examples/two_way_speaker.mlg");
    let mut model = mlg_loader::load_model(source, "two_way_speaker.mlg")
        .map_err(|error| format!("load bounded speaker fixture: {error}"))?;
    model.simulation.freq_start = 500.0;
    model.simulation.freq_end = 1000.0;
    model.simulation.freq_points = 2;
    model.simulation.max_mesh_frequency = Some(1000.0);

    let eval_points = cea2034_output::generate_eval_points([0.0, 0.0, 0.0]);
    let fail_closed_product_configurations = require_product_configuration_rejections(&model)?;
    let config = simulation::SolverConfig {
        tolerance: RESIDUAL_TOLERANCE,
        max_iterations: 400,
        ..Default::default()
    };
    let started = Instant::now();
    let (result, report) =
        simulation::run_simulation_cuda_with_report(&model, &eval_points, &config)
            .map_err(|error| format!("required CUDA whole-speaker workload failed: {error}"))?;
    let mut objective_weights = vec![Complex64::new(0.0, 0.0); eval_points.len()];
    objective_weights[0] = Complex64::new(1.0, 0.0);
    let (adjoints, adjoint_report) = simulation::run_simulation_cuda_adjoint_with_report(
        &model,
        &eval_points,
        &objective_weights,
        &config,
        0,
    )
    .map_err(|error| format!("required CUDA prescribed adjoint workload failed: {error}"))?;

    if report.requested_backend != simulation::SimulationBackend::Cuda
        || report.actual_backend != simulation::SimulationBackend::Cuda
        || !matches!(
            report.capability_matrix.field_evaluation,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
        || !report.cache.field_geometry_resident
        || !report.cache.mesh_resident_across_frequencies
        || !report.cache.operator_structure_resident
        || report.cache.assembly_workspace_builds == 0
        || report.cache.assembly_workspace_builds > report.frequencies
        || report.cache.assembly_workspace_builds
            != report.cache.assembly_quadrature_schedule_updates + 1
        || report.cache.assembly_geometry_reused
            != (report.cache.assembly_quadrature_schedule_updates == 0)
        || !report.cache.rhs_buffer_reused
        || !report.cache.solution_buffer_reused
        || report.cache.matrix_value_updates + 1 != report.frequencies
        || report.cache.preconditioner_updates + 1 != report.frequencies
        || !matches!(
            report
                .capability_matrix
                .speaker_configurations
                .prescribed_forward
                .state,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
    {
        return Err(format!(
            "CUDA workload provenance/capability mismatch: requested={:?}, actual={:?}, field={:?}, field_resident={}, mesh_resident={}, operator_resident={}, assembly_geometry_resident={}, assembly_builds={}, quadrature_schedule_updates={}, rhs_reused={}, solution_reused={}, matrix_updates={}, preconditioner_updates={}, prescribed={:?}",
            report.requested_backend,
            report.actual_backend,
            report.capability_matrix.field_evaluation,
            report.cache.field_geometry_resident,
            report.cache.mesh_resident_across_frequencies,
            report.cache.operator_structure_resident,
            report.cache.assembly_geometry_reused,
            report.cache.assembly_workspace_builds,
            report.cache.assembly_quadrature_schedule_updates,
            report.cache.rhs_buffer_reused,
            report.cache.solution_buffer_reused,
            report.cache.matrix_value_updates,
            report.cache.preconditioner_updates,
            report
                .capability_matrix
                .speaker_configurations
                .prescribed_forward
                .state,
        ));
    }
    require_fail_closed_configuration_cells(&report)?;
    if report.residual_gate != simulation::ResidualGateStatus::Passed
        || report.max_relative_residual > RESIDUAL_TOLERANCE
        || report.max_physical_error > PHYSICAL_TOLERANCE
        || report.max_derivative_assembly_error > DERIVATIVE_TOLERANCE
    {
        return Err(format!(
            "CUDA acceptance gate mismatch: residual={:.3e}, physical={:.3e}, derivative_assembly={:.3e}",
            report.max_relative_residual,
            report.max_physical_error,
            report.max_derivative_assembly_error,
        ));
    }
    if adjoints.len() != report.frequencies
        || adjoint_report.requested_backend != simulation::SimulationBackend::Cuda
        || adjoint_report.actual_backend != simulation::SimulationBackend::Cuda
        || adjoint_report.residual_gate != simulation::ResidualGateStatus::Passed
        || !adjoint_report.operator_structure_resident
        || !adjoint_report.rhs_buffer_reused
        || !adjoint_report.solution_buffer_reused
        || adjoint_report.max_relative_residual > RESIDUAL_TOLERANCE
        || adjoint_report.max_cpu_reference_error > 5.0e-3
        || adjoint_report.max_inner_product_error > 1.0e-10
        || !matches!(
            adjoint_report.capability_matrix.adjoint_solve,
            simulation::GpuCapabilityState::Native | simulation::GpuCapabilityState::VendorNative
        )
    {
        return Err(format!(
            "CUDA adjoint acceptance/provenance mismatch: frequencies={}, expected={}, residual={:.3e}, cpu_reference={:.3e}, inner_product={:.3e}, operator_resident={}, rhs_reused={}, solution_reused={}, capability={:?}",
            adjoints.len(),
            report.frequencies,
            adjoint_report.max_relative_residual,
            adjoint_report.max_cpu_reference_error,
            adjoint_report.max_inner_product_error,
            adjoint_report.operator_structure_resident,
            adjoint_report.rhs_buffer_reused,
            adjoint_report.solution_buffer_reused,
            adjoint_report.capability_matrix.adjoint_solve,
        ));
    }
    if adjoint_report.operator_structure_resident
        && adjoint_report.matrix_value_updates + 1 != adjoint_report.frequencies
    {
        return Err(format!(
            "resident adjoint CSR update count mismatch: updates={}, frequencies={}",
            adjoint_report.matrix_value_updates, adjoint_report.frequencies
        ));
    }
    if report.cache.operator_structure_resident
        && report.cache.matrix_value_updates + 1 != report.frequencies
    {
        return Err(format!(
            "resident CSR update count mismatch: updates={}, frequencies={}",
            report.cache.matrix_value_updates, report.frequencies
        ));
    }
    let cea = cea2034_output::compute_spinorama_metrics(&result)
        .map_err(|error| format!("bounded CUDA CEA post-processing failed: {error}"))?;
    if !cea.preference_score.is_finite() {
        return Err("bounded CUDA CEA preference score is non-finite".into());
    }
    let reduced_order_preview = result
        .to_wasm_preview(2, eval_points.len())
        .map_err(|error| format!("bounded CUDA reduced-order artifact failed: {error}"))?;
    if reduced_order_preview.frequencies.is_empty()
        || reduced_order_preview.frequencies.len() > result.frequencies.len()
        || reduced_order_preview.spl.len() != reduced_order_preview.frequencies.len()
        || reduced_order_preview.phase.len() != reduced_order_preview.frequencies.len()
        || reduced_order_preview
            .spl
            .iter()
            .chain(&reduced_order_preview.phase)
            .any(|row| row.len() != eval_points.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err("bounded CUDA reduced-order artifact has invalid shape or values".into());
    }
    let reduced_order_evidence = ReducedOrderPreviewEvidence {
        schema_version: 1,
        source_provenance: result.provenance.clone(),
        preview: reduced_order_preview,
    };
    if reduced_order_evidence.source_provenance.backend != simulation::SimulationBackend::Cuda {
        return Err("bounded CUDA reduced-order artifact lost CUDA source provenance".into());
    }

    let report_json = serde_json::to_string(&report)
        .map_err(|error| format!("serialize CUDA report for evidence: {error}"))?;
    let preview_json = serde_json::to_vec_pretty(&reduced_order_evidence)
        .map_err(|error| format!("serialize CUDA reduced-order artifact: {error}"))?;
    let unsupported_capability_cells = [
        report.capability_matrix.coupled_driver_solve,
        report.capability_matrix.derivative_assembly,
        report.capability_matrix.adjoint_solve,
        report.capability_matrix.reduced_order_model,
    ]
    .iter()
    .filter(|state| **state == simulation::GpuCapabilityState::Unsupported)
    .count();
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("bounded two-frequency two-way speaker CUDA solve and prescribed adjoint with CPU f64 reconstruction/reference gates")
        .reference_checksum("analytic/deterministic-fixture:two-way-speaker-cuda-workload-v1")
        .input_checksum("sha256:embedded-two-way-speaker-mlg-v1")
        .dimension("3D whole-speaker BEM, 2 frequencies, 72 CEA-2034 points")
        .formulation("CUDA dense regular-pair/analytic derivative assembly + resident Complex32 CSR GMRES + resident Complex64 Hermitian adjoint + host CEA + reduced-order preview sidecar")
        .tolerance("CPU f64 true residual <= 1e-4; physical field error <= 1e-3; derivative assembly <= 5e-3; reduced-order preview shape/finiteness contract")
        .metric("max_relative_residual", report.max_relative_residual)
        .metric("max_physical_error", report.max_physical_error)
        .metric(
            "max_derivative_assembly_error",
            report.max_derivative_assembly_error,
        )
        .metric("adjoint_max_relative_residual", adjoint_report.max_relative_residual)
        .metric("adjoint_max_cpu_reference_error", adjoint_report.max_cpu_reference_error)
        .metric("adjoint_max_inner_product_error", adjoint_report.max_inner_product_error)
        .metric(
            "reduced_order_preview_frequencies",
            reduced_order_evidence.preview.frequencies.len() as f64,
        )
        .metric(
            "reduced_order_preview_points",
            eval_points.len() as f64,
        )
        .metric(
            "adjoint_rhs_buffer_reused",
            adjoint_report.rhs_buffer_reused as u8 as f64,
        )
        .metric(
            "adjoint_solution_buffer_reused",
            adjoint_report.solution_buffer_reused as u8 as f64,
        )
        .metric("cea_preference_score", cea.preference_score)
        .metric("active_dofs", report.active_dofs as f64)
        .metric("cuda_h2d_bytes", report.transfers.h2d_bytes as f64)
        .metric("cuda_d2h_bytes", report.transfers.d2h_bytes as f64)
        .metric("cuda_d2d_bytes", report.transfers.d2d_bytes as f64)
        .metric("adjoint_cuda_h2d_bytes", adjoint_report.transfers.h2d_bytes as f64)
        .metric("adjoint_cuda_d2h_bytes", adjoint_report.transfers.d2h_bytes as f64)
        .metric("gpu_performance_eligible", report.performance_eligible as u8 as f64)
        .metric("assembly_workspace_builds", report.cache.assembly_workspace_builds as f64)
        .metric(
            "assembly_quadrature_schedule_updates",
            report.cache.assembly_quadrature_schedule_updates as f64,
        )
        .metric(
            "assembly_geometry_reused",
            report.cache.assembly_geometry_reused as u8 as f64,
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
            "rhs_buffer_reused",
            report.cache.rhs_buffer_reused as u8 as f64,
        )
        .metric(
            "solution_buffer_reused",
            report.cache.solution_buffer_reused as u8 as f64,
        )
        .metric(
            "matrix_value_updates",
            report.cache.matrix_value_updates as f64,
        )
        .metric(
            "preconditioner_updates",
            report.cache.preconditioner_updates as f64,
        )
        .metric("unsupported_capability_cells", unsupported_capability_cells as f64)
        .metric(
            "fail_closed_product_configurations",
            fail_closed_product_configurations as f64,
        )
        .maximum("max_relative_residual", RESIDUAL_TOLERANCE)
        .maximum("max_physical_error", PHYSICAL_TOLERANCE)
        .maximum("max_derivative_assembly_error", DERIVATIVE_TOLERANCE)
        .maximum("adjoint_max_relative_residual", RESIDUAL_TOLERANCE)
        .maximum("adjoint_max_cpu_reference_error", 5.0e-3)
        .maximum("adjoint_max_inner_product_error", 1.0e-10)
        .minimum("fail_closed_product_configurations", 4.0)
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
            .map_err(|error| format!("serialize capability matrix: {error}"))?,
    );
    qa.environment.insert(
        "gpu.failure_modes_fail_closed".into(),
        serde_json::to_string(&report.failure_modes_fail_closed)
            .map_err(|error| format!("serialize failure contract: {error}"))?,
    );
    qa.environment.insert(
        "gpu.timing_evidence_status".into(),
        if report.performance_eligible {
            "gated_performance"
        } else {
            "gated_diagnostic_only"
        }
        .into(),
    );
    qa.environment.insert(
        "gpu.execution_report_sha256".into(),
        checksum(report_json.as_bytes()),
    );
    let adjoint_report_json = serde_json::to_string(&adjoint_report)
        .map_err(|error| format!("serialize CUDA adjoint report for evidence: {error}"))?;
    qa.environment.insert(
        "gpu.adjoint_report_sha256".into(),
        checksum(adjoint_report_json.as_bytes()),
    );
    let (adjoint_path, adjoint_digest) = output_adjoint_report(&adjoint_report)?;
    qa.environment
        .insert("gpu.adjoint_report_path".into(), adjoint_path.clone());
    let (path, digest) = output_report(&report)?;
    let (preview_path, preview_digest) =
        write_qa_artifact("speaker_cuda_reduced_order_preview.json", &preview_json)?;
    qa = qa
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_cuda_workload.rs",
            "bounded whole-speaker CUDA execution report",
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
                allocations: u64::from(
                    report.memory.allocation_high_water_bytes > 0
                        || adjoint_report.memory.allocation_high_water_bytes > 0,
                ),
                h2d_transfers: (report.transfers.h2d_operations
                    + adjoint_report.transfers.h2d_operations)
                    as u64,
                d2h_transfers: (report.transfers.d2h_operations
                    + adjoint_report.transfers.d2h_operations)
                    as u64,
                d2d_transfers: (report.transfers.d2d_operations
                    + adjoint_report.transfers.d2d_operations)
                    as u64,
                synchronizations: 0,
                launches: 2,
            },
            memory: QaMemoryEvidence {
                peak_bytes: report
                    .memory
                    .allocation_high_water_bytes
                    .saturating_add(report.memory.staging_high_water_bytes)
                    .max(
                        adjoint_report
                            .memory
                            .allocation_high_water_bytes
                            .saturating_add(adjoint_report.memory.staging_high_water_bytes),
                    ) as u64,
                live_bytes: report
                    .memory
                    .live_allocation_bytes
                    .saturating_add(report.memory.live_staging_bytes)
                    .max(
                        adjoint_report
                            .memory
                            .live_allocation_bytes
                            .saturating_add(adjoint_report.memory.live_staging_bytes),
                    ) as u64,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .artifact_with_provenance(
            adjoint_path,
            "crates/sonium-qa/bin/speaker_cuda_workload.rs",
            "bounded prescribed CUDA adjoint report",
            "v1",
            adjoint_digest,
        )
        .artifact_with_provenance(
            preview_path,
            "crates/sonium-qa/bin/speaker_cuda_workload.rs",
            "bounded CUDA reduced-order preview sidecar",
            "v1",
            preview_digest,
        )
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
            .expect("CUDA speaker QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
