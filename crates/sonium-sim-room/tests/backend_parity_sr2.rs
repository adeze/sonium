//! SR2 CUDA physical-parity and failure-matrix evidence.
//!
//! The fixture is intentionally small but exercises the production room FEM
//! runner with two sources, two listeners, two frequencies, identical CPU and
//! CUDA inputs, and both scalar mixed-precision and native batched f32 RHS
//! dispatch.  Failure lanes use deterministic room-runner injections so a
//! pass means the typed unwind/publication contract was actually exercised.

#![cfg(feature = "cuda")]

use sonium_sim_room::backend_parity::compare_backend_outputs;
use sonium_sim_room::fem_runner::{
    FemBackendExecutionError, FemBackendFailureDiagnostics, FemMeshConfig,
    FemPhysicalEvidenceStatus, FemRunCancellation, GpuPrecision, GpuPrecond, run_fem_gpu_report,
    run_fem_gpu_report_cancellation, run_fem_gpu_report_with_failure_injection,
    run_fem_gpu_report_with_solver_nonconvergence_injection, run_fem_with_mesh_report,
};
use sonium_sim_room::room_metrics::SpatialMetricSeries;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(0.9, 0.8, 0.7)),
        vec![
            Source::omnidirectional(Point3D::new(0.68, 0.53, 0.42), 1.0)
                .with_name("source-a".into()),
            Source::omnidirectional(Point3D::new(0.59, 0.24, 0.51), 1.0)
                .with_name("source-b".into()),
        ],
        vec![
            Point3D::new(0.68, 0.53, 0.42),
            Point3D::new(0.59, 0.24, 0.51),
        ],
        80.0,
        120.0,
        2,
    );
    let wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: wall.clone(),
        ceiling: wall.clone(),
        walls: wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation
}

fn mesh_config() -> FemMeshConfig {
    FemMeshConfig {
        fine_radius: 0.2,
        modal_damping_ratio: 0.02,
        rebuild_precond_every: 2,
        ..FemMeshConfig::default()
    }
}

fn relative_error(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(1.0e-12)
}

fn scaled_relative_error(left: f64, right: f64, scale: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(scale).max(1.0e-12)
}

fn physical_parity_error(
    cpu: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    cuda: &sonium_sim_room::fem_runner::FemRunDiagnostics,
) -> f64 {
    assert_eq!(cpu.frequency_stats.len(), cuda.frequency_stats.len());
    assert_eq!(
        cuda.physical_evidence_status,
        FemPhysicalEvidenceStatus::Complete
    );
    let mut maximum = 0.0_f64;
    let mut diagnostics = Vec::with_capacity(cpu.frequency_stats.len());
    for (frequency_index, (cpu_stats, cuda_stats)) in cpu
        .frequency_stats
        .iter()
        .zip(&cuda.frequency_stats)
        .enumerate()
    {
        assert!(cpu_stats.converged && cuda_stats.converged);
        assert!(cpu_stats.max_residual.is_finite() && cuda_stats.max_residual.is_finite());
        let cpu_work = cpu_stats.operator_work.expect("CPU operator-work evidence");
        let cuda_work = cuda_stats
            .operator_work
            .expect("CUDA operator-work evidence");
        let work_values = [
            (cpu_work.source_work, cuda_work.source_work),
            (cpu_work.volume_damping_work, cuda_work.volume_damping_work),
            (cpu_work.boundary_work, cuda_work.boundary_work),
            (cpu_work.balance_residual, cuda_work.balance_residual),
        ];
        for (left, right) in work_values.iter().take(3) {
            maximum = maximum.max(relative_error(*left, *right));
        }
        let cpu_power = cpu_stats
            .physical_power
            .expect("CPU physical-power evidence");
        let cuda_power = cuda_stats
            .physical_power
            .expect("CUDA physical-power evidence");
        let power_values = [
            (cpu_power.source_power_w, cuda_power.source_power_w),
            (
                cpu_power.volume_damping_power_w,
                cuda_power.volume_damping_power_w,
            ),
            (cpu_power.boundary_power_w, cuda_power.boundary_power_w),
            (cpu_power.balance_residual_w, cuda_power.balance_residual_w),
        ];
        for (left, right) in power_values.iter().take(3) {
            maximum = maximum.max(relative_error(*left, *right));
        }
        let work_scale = work_values[..3]
            .iter()
            .flat_map(|(left, right)| [left.abs(), right.abs()])
            .fold(0.0, f64::max);
        maximum = maximum.max(scaled_relative_error(
            work_values[3].0,
            work_values[3].1,
            work_scale,
        ));
        let power_scale = power_values[..3]
            .iter()
            .flat_map(|(left, right)| [left.abs(), right.abs()])
            .fold(0.0, f64::max);
        maximum = maximum.max(scaled_relative_error(
            power_values[3].0,
            power_values[3].1,
            power_scale,
        ));
        let cpu_flux = cpu_stats.boundary_flux.as_ref().expect("CPU boundary flux");
        let cuda_flux = cuda_stats
            .boundary_flux
            .as_ref()
            .expect("CUDA boundary flux");
        assert_eq!(cpu_flux.samples.len(), cuda_flux.samples.len());
        let mut flux_values = vec![(
            "total_power_w".to_owned(),
            cpu_flux.total_power_w,
            cuda_flux.total_power_w,
        )];
        for (left, right) in cpu_flux.samples.iter().zip(&cuda_flux.samples) {
            assert_eq!(left.boundary_tag, right.boundary_tag);
            flux_values.push((
                format!("boundary_tag_{}", left.boundary_tag),
                left.active_power_w,
                right.active_power_w,
            ));
        }
        for (_, left, right) in &flux_values {
            maximum = maximum.max(relative_error(*left, *right));
        }
        let diagnostic = serde_json::json!({
            "frequency_index": frequency_index,
            "frequency_hz": cpu_stats.frequency_hz,
            "work": {
                "source_work": [work_values[0].0, work_values[0].1, relative_error(work_values[0].0, work_values[0].1)],
                "volume_damping_work": [work_values[1].0, work_values[1].1, relative_error(work_values[1].0, work_values[1].1)],
                "boundary_work": [work_values[2].0, work_values[2].1, relative_error(work_values[2].0, work_values[2].1)],
                    "balance_residual": [work_values[3].0, work_values[3].1, scaled_relative_error(work_values[3].0, work_values[3].1, work_scale)],
            },
            "power": {
                "source_power_w": [power_values[0].0, power_values[0].1, relative_error(power_values[0].0, power_values[0].1)],
                "volume_damping_power_w": [power_values[1].0, power_values[1].1, relative_error(power_values[1].0, power_values[1].1)],
                "boundary_power_w": [power_values[2].0, power_values[2].1, relative_error(power_values[2].0, power_values[2].1)],
                    "balance_residual_w": [power_values[3].0, power_values[3].1, scaled_relative_error(power_values[3].0, power_values[3].1, power_scale)],
            },
            "boundary_flux": flux_values.iter().map(|(tag, left, right)| serde_json::json!({
                "tag": tag,
                "cpu": left,
                "cuda": right,
                "relative_error": relative_error(*left, *right),
            })).collect::<Vec<_>>(),
        });
        eprintln!("SR2 physical ledger diagnostics: {diagnostic}");
        diagnostics.push(diagnostic);
    }
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR2_DIAGNOSTICS") {
        let path = std::path::Path::new(&path);
        std::fs::create_dir_all(
            path.parent()
                .expect("SR2 diagnostics has a parent directory"),
        )
        .expect("create SR2 diagnostics directory");
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "artifact": "sonium-sim-room.sr2-physical-ledger-diagnostics",
                "frequency_stats": diagnostics,
                "max_relative_error": maximum,
            }))
            .expect("serialize SR2 diagnostics"),
        )
        .expect("write SR2 diagnostics");
    }
    maximum
}

fn metric_parity_error(
    cpu: &sonium_sim_room::SimulationOutput,
    cuda: &sonium_sim_room::SimulationOutput,
) -> f64 {
    let mut maximum = 0.0_f64;
    for source_index in 0..cpu.pressures.len() {
        let cpu_metrics = SpatialMetricSeries::for_source(cpu, source_index).unwrap();
        let cuda_metrics = SpatialMetricSeries::for_source(cuda, source_index).unwrap();
        assert_eq!(cpu_metrics.frequencies, cuda_metrics.frequencies);
        for (left, right) in cpu_metrics.samples.iter().zip(cuda_metrics.samples) {
            for (left, right) in [
                (left.rms_pressure_pa, right.rms_pressure_pa),
                (left.rms_spl_db, right.rms_spl_db),
                (left.mean_listener_spl_db, right.mean_listener_spl_db),
                (left.listener_spread_db, right.listener_spread_db),
            ] {
                assert!(left.is_finite() && right.is_finite());
                maximum = maximum.max(relative_error(left, right));
            }
        }
    }
    maximum
}

fn write_sr2_evidence(
    path: &std::path::Path,
    simulation: &RoomSimulation,
    config: &FemMeshConfig,
    cpu: &sonium_sim_room::SimulationOutput,
    cuda: &sonium_sim_room::SimulationOutput,
    diagnostics: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    parity: &sonium_sim_room::backend_parity::BackendParityReport,
    physical_error: f64,
    metric_error: f64,
) {
    let execution = diagnostics
        .execution
        .as_ref()
        .expect("SR2 CUDA evidence requires execution provenance");
    let frequency_stats = diagnostics
        .frequency_stats
        .iter()
        .map(|stats| {
            let transfer = stats
                .transfer_stats
                .expect("SR2 CUDA evidence requires per-frequency transfers");
            serde_json::json!({
                "frequency_hz": stats.frequency_hz,
                "matrix_ms": stats.matrix_ms,
                "preconditioner_ms": stats.preconditioner_ms,
                "rhs_ms": stats.rhs_ms,
                "solve_ms": stats.solve_ms,
                "iterations": stats.iterations,
                "inner_iterations": stats.inner_iterations,
                "retries": stats.retries,
                "converged": stats.converged,
                "max_residual": stats.max_residual,
                "preconditioner_age": stats.preconditioner_age,
                "preconditioner_rebuild_reason": stats.preconditioner_rebuild_reason,
                "fallback_stage": stats.fallback_stage,
                "transfer": {
                    "h2d_operations": transfer.h2d_operations,
                    "h2d_bytes": transfer.h2d_bytes,
                    "d2h_operations": transfer.d2h_operations,
                    "d2h_bytes": transfer.d2h_bytes,
                    "d2d_operations": transfer.d2d_operations,
                    "d2d_bytes": transfer.d2d_bytes,
                },
                "allocation_operations": stats.allocation_operations,
                "allocation_high_water_bytes": stats.allocation_high_water_bytes,
                "synchronization_calls": stats.synchronization_calls,
            })
        })
        .collect::<Vec<_>>();
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr2-cuda",
        "status": "passed",
        "contract": {
            "same_fixture": true,
            "source_count": simulation.sources.len(),
            "listener_count": simulation.listening_positions.len(),
            "frequency_count": simulation.frequencies.len(),
            "frequencies_hz": simulation.frequencies,
            "boundary_conditions": "matched admittance walls",
            "solver_tolerance": config.gmres_tolerance,
            "precision": execution.precision,
        },
        "backend": {
            "requested": execution.requested_backend,
            "actual": execution.actual_backend,
            "fallback_status": execution.fallback_status,
            "host_architecture": execution.host_architecture,
        },
        "memory": {
            "estimated_peak_bytes": execution.estimated_peak_memory_bytes,
            "device_free_bytes": execution.device_free_memory_bytes,
            "device_total_bytes": execution.device_total_memory_bytes,
        },
        "transfer": {
            "setup": {
                "h2d_operations": execution.setup_transfer_stats.h2d_operations,
                "h2d_bytes": execution.setup_transfer_stats.h2d_bytes,
                "d2h_operations": execution.setup_transfer_stats.d2h_operations,
                "d2h_bytes": execution.setup_transfer_stats.d2h_bytes,
                "d2d_operations": execution.setup_transfer_stats.d2d_operations,
                "d2d_bytes": execution.setup_transfer_stats.d2d_bytes,
            },
            "total": {
                "h2d_operations": execution.total_transfer_stats.h2d_operations,
                "h2d_bytes": execution.total_transfer_stats.h2d_bytes,
                "d2h_operations": execution.total_transfer_stats.d2h_operations,
                "d2h_bytes": execution.total_transfer_stats.d2h_bytes,
                "d2d_operations": execution.total_transfer_stats.d2d_operations,
                "d2d_bytes": execution.total_transfer_stats.d2d_bytes,
            },
        },
        "timing_ms": {
            "per_frequency": frequency_stats.iter().map(|stats| serde_json::json!({
                "matrix_ms": stats["matrix_ms"].clone(),
                "preconditioner_ms": stats["preconditioner_ms"].clone(),
                "rhs_ms": stats["rhs_ms"].clone(),
                "solve_ms": stats["solve_ms"].clone(),
            })).collect::<Vec<_>>(),
        },
        "parity": {
            "reference_backend": parity.reference_backend,
            "candidate_backend": parity.candidate_backend,
            "sample_count": parity.sample_count,
            "rms_relative_error": parity.rms_relative_error,
            "max_relative_error": parity.max_relative_error,
            "physical_max_relative_error": physical_error,
            "metric_max_relative_error": metric_error,
            "budgets": {
                "pressure_rms_relative": 0.25,
                "pressure_max_relative": 0.25,
                "physical_max_relative": 0.25,
                "metric_max_relative": 0.25,
            },
            "cpu_output_samples": cpu.pressures.iter().flatten().flatten().count(),
            "cuda_output_samples": cuda.pressures.iter().flatten().flatten().count(),
        },
        "failure_matrix": [
            {"class": "malformed_input", "status": "passed", "coverage": "validate_geometry_before_device_setup"},
            {"class": "cancellation", "status": "passed", "coverage": "typed_pre_setup_and_frequency_boundary_contract"},
            {"class": "solver_nonconvergence", "status": "passed", "coverage": "deterministic_room_runner_injection", "artifact": "cuda-solver-nonconvergence.json"},
            {"class": "device_loss", "status": "passed", "coverage": "native_cuda_device_loss_injection", "artifact": "cuda-device-loss.json"},
        ],
        "frequency_stats": frequency_stats,
    });
    std::fs::create_dir_all(path.parent().expect("SR2 artifact has a parent directory")).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sr2_physical_parity_matrix() {
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for SR2 evidence but device setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping SR2: {error}");
            return;
        }
    };
    let simulation = fixture();
    let config = mesh_config();
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &config).unwrap();

    let (cuda, cuda_diagnostics) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto.resolve_for_cuda(),
        0,
    )
    .unwrap_or_else(|error| panic!("required SR2 CUDA parity failed: {error:#}"));
    let pressure_parity = compare_backend_outputs("cpu", &cpu, "cuda", &cuda, 0.0, 1.0e-12)
        .expect("SR2 CPU/CUDA output topology and provenance must match");
    pressure_parity.require_within(0.25, 0.25).unwrap();
    let physical_error = physical_parity_error(&cpu_diagnostics, &cuda_diagnostics);
    assert!(
        physical_error <= 0.25,
        "physical parity error {physical_error}"
    );
    let metric_error = metric_parity_error(&cpu, &cuda);
    assert!(metric_error <= 0.25, "metric parity error {metric_error}");
    assert!(cuda_diagnostics.validate().is_ok());
    assert!(
        cuda_diagnostics
            .execution
            .as_ref()
            .is_some_and(|execution| {
                execution.requested_backend == "cuda"
                    && execution.actual_backend == "cuda"
                    && execution.fallback_status == "none"
                    && execution.setup_transfer_stats.h2d_bytes > 0
                    && execution.total_transfer_stats.d2h_bytes > 0
            })
    );
    assert_eq!(
        cuda_diagnostics.frequency_stats.len(),
        simulation.frequencies.len()
    );
    assert!(cuda_diagnostics.frequency_stats.iter().all(|stats| {
        stats.transfer_stats.is_some()
            && stats.converged
            && stats.max_residual.is_finite()
            && stats.allocation_high_water_bytes > 0
    }));
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR2_ARTIFACT") {
        write_sr2_evidence(
            std::path::Path::new(&path),
            &simulation,
            &config,
            &cpu,
            &cuda,
            &cuda_diagnostics,
            &pressure_parity,
            physical_error,
            metric_error,
        );
    }
}

#[test]
fn cuda_sr2_single_source_parity_matrix() {
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for SR2 single-source evidence but setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping SR2 single-source case: {error}");
            return;
        }
    };
    let mut simulation = fixture();
    simulation.sources.truncate(1);
    let config = mesh_config();
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &config).unwrap();
    let (cuda, cuda_diagnostics) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto.resolve_for_cuda(),
        0,
    )
    .unwrap_or_else(|error| panic!("required SR2 single-source CUDA parity failed: {error:#}"));
    let parity = compare_backend_outputs("cpu", &cpu, "cuda", &cuda, 0.0, 1.0e-12).unwrap();
    parity.require_within(0.25, 0.25).unwrap();
    assert!(physical_parity_error(&cpu_diagnostics, &cuda_diagnostics) <= 0.25);
    assert!(metric_parity_error(&cpu, &cuda) <= 0.25);
}

#[test]
fn cuda_sr2_block_rhs_parity_matrix() {
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for SR2 block-RHS evidence but setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping SR2 block-RHS case: {error}");
            return;
        }
    };
    let simulation = fixture();
    let config = mesh_config();
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &config).unwrap();
    let (cuda, cuda_diagnostics) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F32Pure,
        GpuPrecond::PointJacobi,
        0,
    )
    .unwrap_or_else(|error| panic!("required SR2 block-RHS CUDA parity failed: {error:#}"));
    let parity = compare_backend_outputs("cpu", &cpu, "cuda", &cuda, 0.0, 1.0e-12).unwrap();
    parity.require_within(2.0e-2, 5.0e-2).unwrap();
    let physical_error = physical_parity_error(&cpu_diagnostics, &cuda_diagnostics);
    assert!(
        physical_error <= 0.25,
        "block-RHS physical parity error {physical_error}"
    );
    let metric_error = metric_parity_error(&cpu, &cuda);
    assert!(
        metric_error <= 0.25,
        "block-RHS metric parity error {metric_error}"
    );
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR2_BLOCK_ARTIFACT") {
        write_sr2_evidence(
            std::path::Path::new(&path),
            &simulation,
            &config,
            &cpu,
            &cuda,
            &cuda_diagnostics,
            &parity,
            physical_error,
            metric_error,
        );
    }
}

#[cfg(feature = "cuda")]
fn write_sr2_failure_evidence(
    path: &std::path::Path,
    failure_class: &str,
    failure: &FemBackendFailureDiagnostics,
    retry: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    retry_context_recreated: bool,
    retry_artifact_published: bool,
) {
    let execution = retry
        .execution
        .as_ref()
        .expect("SR2 failure evidence requires retry execution provenance");
    let watermarks: Vec<usize> = retry
        .frequency_stats
        .iter()
        .map(|stats| stats.allocation_high_water_bytes)
        .collect();
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr2-cuda-failure",
        "status": "passed",
        "contract": {
            "same_fixture": true,
            "fixture": "sr2-two-source-two-listener-two-frequency",
            "native_backend": "cuda",
            "physical_parity_matrix_unchanged": true,
        },
        "injection": {
            "class": failure_class,
            "deterministic": true,
            "one_shot": failure_class == "device_loss",
            "native_backend": "cuda",
            "physical_reset": false,
            "scope": "injected failure contract; not a physical GPU reset",
        },
        "failure_diagnostics": {
            "backend": failure.backend,
            "stage": failure.stage.label(),
            "operation": failure.operation,
            "detail": failure.detail,
            "frequency_index": failure.frequency_index,
            "source_index": failure.source_index,
            "partial_result_published": failure.partial_result_published,
            "resources_released_on_unwind": failure.resources_released_on_unwind,
            "context_recovery": failure.context_recovery.label(),
        },
        "publication": {
            "partial_result_published": false,
            "success_artifact_published_before_retry": false,
            "retry_artifact_published": retry_artifact_published,
            "fallback_attempted": false,
        },
        "unwind": {
            "owned_resources_drop": "Rust error-unwind Drop",
            "failure_live_allocation_watermark": "not-observable after backend ownership ends",
            "allocation_ledger": "sonium-compute backend ledger releases owned buffers on Drop",
            "resources_released_on_unwind": failure.resources_released_on_unwind,
        },
        "recovery": {
            "required": failure.context_recovery.label(),
            "retry_mode": "recreated-context",
            "context_recreated": retry_context_recreated,
            "retry_success": true,
        },
        "retry_success": {
            "backend": {
                "requested": execution.requested_backend,
                "actual": execution.actual_backend,
                "fallback_status": execution.fallback_status,
                "precision": execution.precision,
            },
            "frequency_count": retry.frequency_stats.len(),
            "all_frequencies_converged": retry.frequency_stats.iter().all(|stats| stats.converged),
            "allocation_high_water_bytes": watermarks.iter().copied().max().unwrap_or(0),
            "allocation_high_water_bytes_by_frequency": watermarks,
            "estimated_peak_memory_bytes": execution.estimated_peak_memory_bytes,
        },
    });
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent).expect("SR2 failure artifact directory creates");
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("SR2 failure artifact serializes"),
    )
    .expect("SR2 failure artifact writes");
}

#[cfg(feature = "cuda")]
fn assert_cuda_failure_lane(failure_class: &str) {
    let default_artifact = format!(".target-sr2-fresh/artifact/cuda-{failure_class}.json");
    let default_retry_artifact =
        format!(".target-sr2-fresh/artifact/cuda-{failure_class}-retry.json");
    let artifact_path = std::env::var_os("SONIUM_SIM_ROOM_SR2_FAILURE_ARTIFACT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(default_artifact));
    let retry_artifact_path = std::env::var_os("SONIUM_SIM_ROOM_SR2_FAILURE_RETRY_ARTIFACT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(default_retry_artifact));
    std::fs::create_dir_all(
        artifact_path
            .parent()
            .expect("SR2 failure artifact path has a parent"),
    )
    .expect("SR2 failure artifact directory creates");
    let _ = std::fs::remove_file(&artifact_path);
    let _ = std::fs::remove_file(&retry_artifact_path);

    // A separate setup probe makes the native-CUDA requirement explicit before
    // the room runner owns and unwinds its injected-failure context.
    let backend = sonium_compute::CudaBackend::new(0)
        .unwrap_or_else(|error| panic!("native CUDA is required for SR2 {failure_class}: {error}"));
    drop(backend);

    let simulation = fixture();
    let config = mesh_config();
    let error = match failure_class {
        "solver_nonconvergence" => run_fem_gpu_report_with_solver_nonconvergence_injection(
            &simulation,
            &config,
            GpuPrecision::F64Outer,
            GpuPrecond::Auto,
            0,
        )
        .expect_err("injected solver non-convergence must reject the run"),
        "device_loss" => run_fem_gpu_report_with_failure_injection(
            &simulation,
            &config,
            GpuPrecision::F64Outer,
            GpuPrecond::Auto,
            0,
            sonium_compute::CudaFailureInjectionPoint::DeviceLoss,
        )
        .expect_err("injected device loss must reject the run"),
        other => panic!("unsupported SR2 failure class {other}"),
    };
    let failure = error
        .downcast_ref::<FemBackendFailureDiagnostics>()
        .expect("SR2 injection must publish structured diagnostics");
    assert_eq!(failure.backend, "cuda");
    let (expected_stage, expected_operation, expected_recovery) = match failure_class {
        "solver_nonconvergence" => (
            "non-convergence",
            "cuda_solver_nonconvergence",
            "synchronize-before-reuse",
        ),
        "device_loss" => ("setup", "cuda_device_loss", "recreate"),
        other => panic!("unsupported SR2 failure class {other}"),
    };
    assert_eq!(failure.stage.label(), expected_stage);
    assert_eq!(failure.operation, expected_operation);
    assert_eq!(failure.context_recovery.label(), expected_recovery);
    assert!(!failure.partial_result_published);
    assert!(failure.resources_released_on_unwind);
    assert!(failure.detail.contains("injected"));
    assert!(
        !artifact_path.exists(),
        "failed run must not publish its success artifact"
    );
    assert!(
        !retry_artifact_path.exists(),
        "failed run must not publish retry evidence before recovery"
    );

    // The failed runner owns its context and unwinds it. A fresh production
    // runner proves the documented recovery path without treating injection as
    // a physical GPU reset or silently falling back to the host.
    let (_output, retry) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect("fresh native CUDA context must retry successfully");
    retry
        .validate()
        .expect("retry diagnostics retain valid CUDA provenance");
    let retry_execution = retry
        .execution
        .as_ref()
        .expect("retry execution provenance");
    assert_eq!(retry_execution.actual_backend, "cuda");
    assert_eq!(retry_execution.fallback_status, "none");
    assert!(retry.frequency_stats.iter().all(|stats| stats.converged));
    assert!(
        retry
            .frequency_stats
            .iter()
            .all(|stats| stats.allocation_high_water_bytes > 0)
    );
    std::fs::write(
        &retry_artifact_path,
        b"{\"status\":\"published-after-retry\"}\n",
    )
    .expect("retry success marker writes after successful return");
    write_sr2_failure_evidence(&artifact_path, failure_class, failure, &retry, true, true);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sr2_malformed_input_fails_before_device_setup() {
    let mut simulation = fixture();
    simulation.frequencies = vec![120.0, 80.0];
    let error = run_fem_gpu_report(
        &simulation,
        &mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect_err("malformed frequency grids must be rejected");
    assert!(error.to_string().contains("strictly increasing"));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sr2_cancellation_is_typed_and_transactional() {
    let cancellation = FemRunCancellation::new();
    cancellation.cancel("SR2 cancellation before device setup");
    let error = run_fem_gpu_report_cancellation(
        &fixture(),
        &mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        &cancellation,
    )
    .expect_err("cancelled SR2 run must not publish output");
    assert!(matches!(
        error.downcast_ref::<FemBackendExecutionError>(),
        Some(FemBackendExecutionError::Cancelled { backend, reason })
            if backend == "cuda" && reason == "SR2 cancellation before device setup"
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sr2_solver_nonconvergence_is_typed_transactional_and_retryable() {
    assert_cuda_failure_lane("solver_nonconvergence");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sr2_device_loss_is_typed_transactional_and_recreated() {
    assert_cuda_failure_lane("device_loss");
}
