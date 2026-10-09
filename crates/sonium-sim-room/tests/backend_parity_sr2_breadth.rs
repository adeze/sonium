//! SR2 native-CUDA breadth evidence for the production room FEM runner.
//!
//! This packet keeps the existing physical-parity matrix unchanged while
//! widening one real production fixture to three sources/listeners and four
//! frequencies.  It records resident per-frequency state, checks a
//! source-amplitude objective derivative against an independent CPU centered
//! difference, and exercises typed launch-failure unwind/recovery.  The
//! derivative is a linear-source sensitivity check, not a claim of a device
//! resident treatment optimizer or a Hermitian lossy-room operator.

#![cfg(feature = "cuda")]

use num_complex::Complex64;
use serde_json::json;
use sonium_sim_room::backend_parity::compare_backend_outputs;
use sonium_sim_room::fem_runner::{
    FemBackendFailureDiagnostics, FemMeshConfig, GpuPrecision, GpuPrecond, run_fem_gpu_report,
    run_fem_gpu_report_with_failure_injection, run_fem_with_mesh_report,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "sonium-sim-room.sr2-cuda-breadth";
const SOURCE_COUNT: usize = 3;
const LISTENER_COUNT: usize = 3;
const FREQUENCY_COUNT: usize = 4;
const DERIVATIVE_SOURCE: usize = 0;
const DERIVATIVE_STEP: f64 = 1.0e-3;

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(0.9, 0.8, 0.7)),
        vec![
            Source::omnidirectional(Point3D::new(0.68, 0.53, 0.42), 1.0)
                .with_name("source-a".into()),
            Source::omnidirectional(Point3D::new(0.59, 0.24, 0.51), 1.0)
                .with_name("source-b".into()),
            Source::omnidirectional(Point3D::new(0.31, 0.47, 0.28), 1.0)
                .with_name("source-c".into()),
        ],
        vec![
            Point3D::new(0.68, 0.53, 0.42),
            Point3D::new(0.59, 0.24, 0.51),
            Point3D::new(0.31, 0.47, 0.28),
        ],
        70.0,
        130.0,
        FREQUENCY_COUNT,
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
        rebuild_precond_every: FREQUENCY_COUNT,
        ..FemMeshConfig::default()
    }
}

fn objective(output: &sonium_sim_room::SimulationOutput, source_index: usize) -> f64 {
    output.pressures[source_index]
        .iter()
        .flatten()
        .map(Complex64::norm_sqr)
        .sum()
}

fn linear_source_derivative(
    output: &sonium_sim_room::SimulationOutput,
    source_index: usize,
    source_amplitude: f64,
) -> f64 {
    2.0 * objective(output, source_index) / source_amplitude
}

fn relative_error(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(1.0e-12)
}

fn transfer_json(stats: sonium_sim_room::fem_runner::FemTransferStats) -> serde_json::Value {
    json!({
        "h2d_operations": stats.h2d_operations,
        "h2d_bytes": stats.h2d_bytes,
        "d2h_operations": stats.d2h_operations,
        "d2h_bytes": stats.d2h_bytes,
        "d2d_operations": stats.d2d_operations,
        "d2d_bytes": stats.d2d_bytes,
    })
}

fn output_path() -> Option<PathBuf> {
    std::env::var_os("SONIUM_SIM_ROOM_SR2_BREADTH_ARTIFACT").map(PathBuf::from)
}

fn evidence(path: &Path) {
    assert_eq!(std::env::consts::OS, "linux", "SR2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "SR2 requires native x86_64"
    );
    let _ = fs::remove_file(path);
    let simulation = fixture();
    let config = mesh_config();
    assert_eq!(simulation.sources.len(), SOURCE_COUNT);
    assert_eq!(simulation.listening_positions.len(), LISTENER_COUNT);
    assert_eq!(simulation.frequencies.len(), FREQUENCY_COUNT);

    let (cpu, _cpu_diagnostics) =
        run_fem_with_mesh_report(&simulation, &config).expect("CPU breadth reference");
    cpu.validate_layout().expect("CPU breadth layout");
    let (cuda, cuda_diagnostics) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect("native CUDA breadth run");
    cuda.validate_layout().expect("CUDA breadth layout");
    let parity = compare_backend_outputs("cpu", &cpu, "cuda", &cuda, 0.0, 1.0e-12)
        .expect("breadth output topology and provenance");
    parity
        .require_within(0.25, 0.25)
        .expect("breadth pressure parity");
    let execution = cuda_diagnostics
        .execution
        .as_ref()
        .expect("breadth execution provenance");
    assert_eq!(execution.requested_backend, "cuda");
    assert_eq!(execution.actual_backend, "cuda");
    assert_eq!(execution.fallback_status, "none");
    assert_eq!(cuda_diagnostics.frequency_stats.len(), FREQUENCY_COUNT);
    assert!(cuda_diagnostics.frequency_stats.iter().all(|stats| {
        stats.converged
            && stats.max_residual.is_finite()
            && stats.transfer_stats.is_some()
            && stats.allocation_high_water_bytes > 0
    }));
    let watermarks = cuda_diagnostics
        .frequency_stats
        .iter()
        .map(|stats| stats.allocation_high_water_bytes)
        .collect::<Vec<_>>();
    assert!(watermarks.windows(2).all(|pair| pair[1] >= pair[0]));
    assert_eq!(watermarks[0], watermarks[watermarks.len() - 1]);
    let ages = cuda_diagnostics
        .frequency_stats
        .iter()
        .map(|stats| stats.preconditioner_age)
        .collect::<Vec<_>>();
    assert_eq!(ages, vec![0, 1, 2, 3]);
    let warm_transfers = cuda_diagnostics.frequency_stats[1..]
        .iter()
        .map(|stats| stats.transfer_stats.expect("warm transfer stats"))
        .collect::<Vec<_>>();
    assert!(warm_transfers.iter().all(|transfer| {
        transfer.h2d_operations > 0
            && transfer.d2h_operations > 0
            && transfer.h2d_bytes > 0
            && transfer.d2h_bytes > 0
    }));

    // The FEM solve is linear in a source amplitude.  The CPU centered
    // difference is an independent backend oracle for the CUDA sensitivity.
    let mut plus_simulation = simulation.clone();
    plus_simulation.sources[DERIVATIVE_SOURCE].amplitude += DERIVATIVE_STEP;
    let mut minus_simulation = simulation.clone();
    minus_simulation.sources[DERIVATIVE_SOURCE].amplitude -= DERIVATIVE_STEP;
    let (cpu_plus, _) =
        run_fem_with_mesh_report(&plus_simulation, &config).expect("CPU plus derivative run");
    let (cpu_minus, _) =
        run_fem_with_mesh_report(&minus_simulation, &config).expect("CPU minus derivative run");
    let cpu_fd_derivative = (objective(&cpu_plus, DERIVATIVE_SOURCE)
        - objective(&cpu_minus, DERIVATIVE_SOURCE))
        / (2.0 * DERIVATIVE_STEP);
    let cpu_linear_derivative = linear_source_derivative(&cpu, DERIVATIVE_SOURCE, 1.0);
    let cuda_linear_derivative = linear_source_derivative(&cuda, DERIVATIVE_SOURCE, 1.0);
    let cpu_fd_error = relative_error(cpu_fd_derivative, cpu_linear_derivative);
    let cuda_fd_error = relative_error(cuda_linear_derivative, cpu_fd_derivative);
    assert!(
        cpu_fd_error <= 5.0e-3,
        "CPU derivative oracle error {cpu_fd_error}"
    );
    assert!(
        cuda_fd_error <= 0.25,
        "CUDA derivative parity error {cuda_fd_error}"
    );

    let mut malformed = simulation.clone();
    malformed.frequencies = vec![130.0, 70.0, 90.0, 110.0];
    let malformed_error = run_fem_gpu_report(
        &malformed,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect_err("malformed breadth frequencies must reject before publication");
    let malformed_typed = malformed_error.to_string();
    assert!(malformed_typed.contains("strictly increasing"));

    let launch_error = run_fem_gpu_report_with_failure_injection(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        sonium_compute::CudaFailureInjectionPoint::Launch,
    )
    .expect_err("injected launch failure must reject breadth run");
    let failure = launch_error
        .downcast_ref::<FemBackendFailureDiagnostics>()
        .expect("launch injection must expose structured diagnostics");
    assert_eq!(failure.backend, "cuda");
    assert_eq!(failure.stage.label(), "launch");
    assert_eq!(failure.operation, "cuda_launch");
    assert_eq!(failure.context_recovery.label(), "synchronize-before-reuse");
    assert!(!failure.partial_result_published);
    assert!(failure.resources_released_on_unwind);
    assert!(failure.detail.contains("injected"));
    assert!(!path.exists(), "failure must not publish breadth artifact");

    let (retry_output, retry_diagnostics) = run_fem_gpu_report(
        &simulation,
        &config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect("fresh CUDA context must recover breadth run");
    let retry_execution = retry_diagnostics
        .execution
        .as_ref()
        .expect("retry execution provenance");
    assert_eq!(retry_execution.actual_backend, "cuda");
    assert_eq!(retry_execution.fallback_status, "none");
    assert!(
        retry_diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.converged)
    );
    let retry_parity = compare_backend_outputs("cuda", &cuda, "retry", &retry_output, 0.0, 1.0e-12)
        .expect("retry output topology");
    retry_parity
        .require_within(1.0e-8, 1.0e-6)
        .expect("fresh-context retry must reproduce breadth output");

    let frequency_stats = cuda_diagnostics
        .frequency_stats
        .iter()
        .map(|stats| {
            json!({
                "frequency_hz": stats.frequency_hz,
                "preconditioner_age": stats.preconditioner_age,
                "preconditioner_rebuild_reason": stats.preconditioner_rebuild_reason,
                "iterations": stats.iterations,
                "inner_iterations": stats.inner_iterations,
                "max_residual": stats.max_residual,
                "converged": stats.converged,
                "allocation_operations": stats.allocation_operations,
                "allocation_high_water_bytes": stats.allocation_high_water_bytes,
                "synchronization_calls": stats.synchronization_calls,
                "transfer": transfer_json(stats.transfer_stats.expect("frequency transfers")),
            })
        })
        .collect::<Vec<_>>();
    let warm_h2d_bytes: usize = warm_transfers
        .iter()
        .map(|transfer| transfer.h2d_bytes)
        .sum();
    let warm_d2h_bytes: usize = warm_transfers
        .iter()
        .map(|transfer| transfer.d2h_bytes)
        .sum();
    let artifact = json!({
        "schema_version": 1,
        "artifact": SCHEMA,
        "status": "passed",
        "contract": {
            "same_fixture": true,
            "source_count": SOURCE_COUNT,
            "listener_count": LISTENER_COUNT,
            "frequency_count": FREQUENCY_COUNT,
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
        "resident": {
            "state": "device CSR/RHS/solver state reused across accepted frequencies",
            "frequency_stats": frequency_stats,
            "preconditioner_ages": ages,
            "allocation_high_water_bytes_by_frequency": watermarks,
            "allocation_watermark_stable": true,
            "warm_frequency_count": warm_transfers.len(),
            "warm_h2d_bytes": warm_h2d_bytes,
            "warm_d2h_bytes": warm_d2h_bytes,
            "warm_transfers_are_explicit": true,
        },
        "parity": {
            "reference_backend": parity.reference_backend,
            "candidate_backend": parity.candidate_backend,
            "sample_count": parity.sample_count,
            "rms_relative_error": parity.rms_relative_error,
            "max_relative_error": parity.max_relative_error,
            "budgets": {"rms_relative": 0.25, "max_relative": 0.25},
        },
        "derivative": {
            "parameter": "source[0].amplitude",
            "objective": "sum listener-frequency |pressure|² for source 0",
            "step": DERIVATIVE_STEP,
            "cpu_finite_difference": cpu_fd_derivative,
            "cpu_linear_source_oracle": cpu_linear_derivative,
            "cuda_linear_source_derivative": cuda_linear_derivative,
            "cpu_oracle_relative_error": cpu_fd_error,
            "cuda_relative_error_to_cpu_oracle": cuda_fd_error,
            "oracle": "CPU centered difference against exact linear-source scaling; no device finite-difference claim",
        },
        "failure_recovery": {
            "class": "launch",
            "deterministic": true,
            "backend": failure.backend,
            "stage": failure.stage.label(),
            "operation": failure.operation,
            "context_recovery": failure.context_recovery.label(),
            "partial_result_published": failure.partial_result_published,
            "resources_released_on_unwind": failure.resources_released_on_unwind,
            "detail": failure.detail,
            "retry_mode": "fresh-context",
            "retry_backend": retry_execution.actual_backend,
            "retry_fallback_status": retry_execution.fallback_status,
            "retry_frequency_count": retry_diagnostics.frequency_stats.len(),
            "retry_converged": retry_diagnostics.frequency_stats.iter().all(|stats| stats.converged),
            "retry_parity_rms_relative_error": retry_parity.rms_relative_error,
            "retry_parity_max_relative_error": retry_parity.max_relative_error,
        },
        "malformed_input": {
            "status": "typed_rejected",
            "publication": "none",
            "error": malformed_typed,
        },
        "external_physical_validation": "unavailable: no authenticated external room dataset supplied",
        "hermitian_operator_claim": "not_claimed: matched lossy-room operator is not asserted Hermitian",
    });
    fs::create_dir_all(path.parent().expect("SR2 breadth artifact parent"))
        .expect("create SR2 breadth artifact directory");
    fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("serialize SR2 breadth artifact"),
    )
    .expect("write SR2 breadth artifact");
}

#[test]
fn cuda_sr2_breadth_resident_derivative_and_recovery() {
    let path = output_path().expect("runner must provide SR2 breadth artifact path");
    evidence(&path);
}
