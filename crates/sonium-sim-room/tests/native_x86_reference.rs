//! Deterministic native-x86 room reference evidence.
//!
//! This is intentionally a full-domain CPU run: it records the selected
//! dispatch path and checks topology, field-independent response, residual,
//! reciprocity/energy metadata, and repeatability. A dedicated x86 runner can
//! promote the same test with `SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86=1`; provide
//! `SONIUM_SIM_ROOM_REFERENCE_ARTIFACT` plus
//! `SONIUM_SIM_ROOM_REQUIRE_CROSS_ARCH_REFERENCE=1` to require comparison
//! against an accepted aarch64 artifact. This test never treats a
//! cross-compiled or Rosetta run as performance evidence.

use serde::{Deserialize, Serialize};
use sonium_fastmath::{dispatch_features, selected_backend};
use sonium_sim_room::SimulationOutput;
use sonium_sim_room::backend_parity::compare_backend_outputs;
use sonium_sim_room::fem_runner::{
    FemMeshConfig, FemMeshConvergenceSpec, FemNodalField, FemRunDiagnostics,
    run_fem_mesh_convergence_ladder, run_fem_with_mesh_field_report,
};
use sonium_sim_room::reciprocity_validation::compare_reciprocal_transfer_outputs;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![
            Source::omnidirectional(Point3D::new(0.35, 0.4, 0.35), 1.0)
                .with_name("source-a".into()),
            Source::omnidirectional(Point3D::new(0.55, 1.05, 0.45), 0.8)
                .with_name("source-b".into()),
        ],
        vec![Point3D::new(1.2, 1.05, 1.0), Point3D::new(0.95, 0.6, 1.05)],
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
    simulation.frequencies = vec![120.0, 160.0];
    simulation
}

fn reciprocity_fixture(reverse: bool) -> RoomSimulation {
    let source_positions = [
        Point3D::new(0.35, 0.4, 0.35),
        Point3D::new(0.55, 1.05, 0.45),
    ];
    let listener_positions = [Point3D::new(1.2, 1.05, 1.0), Point3D::new(0.95, 0.6, 1.05)];
    let (sources, listeners) = if reverse {
        (
            listener_positions
                .into_iter()
                .enumerate()
                .map(|(index, position)| {
                    Source::omnidirectional(position, 1.0)
                        .with_name(format!("listener-{}", (b'a' + index as u8) as char))
                })
                .collect(),
            source_positions.to_vec(),
        )
    } else {
        (
            source_positions
                .into_iter()
                .enumerate()
                .map(|(index, position)| {
                    Source::omnidirectional(position, 1.0)
                        .with_name(format!("source-{}", (b'a' + index as u8) as char))
                })
                .collect(),
            listener_positions.to_vec(),
        )
    };
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        sources,
        listeners,
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
    simulation.frequencies = vec![120.0, 160.0];
    simulation
}

fn convergence_specs() -> [FemMeshConvergenceSpec; 3] {
    let mut coarse = FemMeshConfig::default();
    coarse.grading_factor = 10.0;
    coarse.fine_radius = 0.0;
    coarse.recycle_subspace_size = 0;
    let mut medium = FemMeshConfig::default();
    medium.grading_factor = 2.0;
    medium.fine_radius = 0.3;
    medium.recycle_subspace_size = 0;
    let mut fine = FemMeshConfig::default();
    fine.recycle_subspace_size = 0;
    [
        FemMeshConvergenceSpec::new("coarse", coarse),
        FemMeshConvergenceSpec::new("medium", medium),
        FemMeshConvergenceSpec::new("fine", fine),
    ]
}

fn convergence_fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(2.4, 2.0, 1.8)),
        vec![Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)],
        vec![Point3D::new(1.8, 1.4, 1.2)],
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
    simulation.frequencies = vec![100.0];
    simulation
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ComplexValue {
    re: f64,
    im: f64,
}

impl From<num_complex::Complex64> for ComplexValue {
    fn from(value: num_complex::Complex64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

impl ComplexValue {
    fn complex(self) -> num_complex::Complex64 {
        num_complex::Complex64::new(self.re, self.im)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReferenceDiagnostic {
    frequency_hz: f64,
    max_residual: f64,
    converged: bool,
    operator_work: [f64; 4],
    physical_power: [f64; 4],
    boundary_flux: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConvergenceLevelEvidence {
    label: String,
    node_count: usize,
    element_count: usize,
    max_residual: f64,
    converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConvergenceEvidence {
    status: String,
    listener_position_tolerance_m: f64,
    frequency_tolerance_hz: f64,
    relative_pressure_floor_pa: f64,
    max_rms_relative_error: f64,
    max_max_relative_error: f64,
    sample_count: usize,
    coarse_to_medium_rms_relative_error: f64,
    coarse_to_medium_max_relative_error: f64,
    medium_to_fine_rms_relative_error: f64,
    medium_to_fine_max_relative_error: f64,
    monotonic: bool,
    levels: Vec<ConvergenceLevelEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComparisonEvidence {
    status: String,
    sample_count: usize,
    rms_relative_error: f64,
    max_relative_error: f64,
    max_rms_relative_error: f64,
    max_max_relative_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CrossArchitectureEvidence {
    status: String,
    target: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeRoomReferenceArtifact {
    schema_version: u32,
    artifact_id: String,
    host_architecture: String,
    cpu_vendor: Option<String>,
    cpu_simd_backend: String,
    dispatch_features: String,
    scalar_reference: String,
    vendor_reference: String,
    frequencies: Vec<f64>,
    source_names: Vec<String>,
    listener_positions: Vec<[f64; 3]>,
    node_coordinates: Vec<[f64; 3]>,
    element_connectivity: Vec<Vec<usize>>,
    element_count: usize,
    boundary_count: usize,
    output: Vec<Vec<Vec<ComplexValue>>>,
    field: Vec<Vec<Vec<ComplexValue>>>,
    diagnostics: Vec<ReferenceDiagnostic>,
    repeatability: ComparisonEvidence,
    reciprocity: ComparisonEvidence,
    convergence: ConvergenceEvidence,
    cross_architecture: CrossArchitectureEvidence,
}

fn encode_output(output: &SimulationOutput) -> Vec<Vec<Vec<ComplexValue>>> {
    output
        .pressures
        .iter()
        .map(|source| {
            source
                .iter()
                .map(|listener| listener.iter().copied().map(ComplexValue::from).collect())
                .collect()
        })
        .collect()
}

fn decode_output(artifact: &NativeRoomReferenceArtifact) -> SimulationOutput {
    SimulationOutput::solved(
        artifact.frequencies.clone(),
        artifact
            .output
            .iter()
            .map(|source| {
                source
                    .iter()
                    .map(|listener| {
                        listener
                            .iter()
                            .copied()
                            .map(ComplexValue::complex)
                            .collect()
                    })
                    .collect()
            })
            .collect(),
        artifact.source_names.clone(),
    )
}

fn cpu_vendor() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.strip_prefix("vendor_id\t:")
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                })
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn diagnostics_to_evidence(diagnostics: &FemRunDiagnostics) -> Vec<ReferenceDiagnostic> {
    diagnostics
        .frequency_stats
        .iter()
        .map(|stats| {
            let operator_work = stats.operator_work.unwrap();
            let physical_power = stats.physical_power.unwrap();
            let boundary_flux = stats
                .boundary_flux
                .as_ref()
                .unwrap()
                .samples
                .iter()
                .map(|sample| (sample.boundary_tag, sample.active_power_w))
                .collect();
            ReferenceDiagnostic {
                frequency_hz: stats.frequency_hz,
                max_residual: stats.max_residual,
                converged: stats.converged,
                operator_work: [
                    operator_work.source_work,
                    operator_work.volume_damping_work,
                    operator_work.boundary_work,
                    operator_work.balance_residual,
                ],
                physical_power: [
                    physical_power.source_power_w,
                    physical_power.volume_damping_power_w,
                    physical_power.boundary_power_w,
                    physical_power.balance_residual_w,
                ],
                boundary_flux,
            }
        })
        .collect()
}

fn build_artifact(
    output: &SimulationOutput,
    field: &FemNodalField,
    diagnostics: &FemRunDiagnostics,
    repeatability: ComparisonEvidence,
    reciprocity: ComparisonEvidence,
    convergence: ConvergenceEvidence,
) -> NativeRoomReferenceArtifact {
    let diagnostics = diagnostics_to_evidence(diagnostics);
    NativeRoomReferenceArtifact {
        schema_version: 2,
        artifact_id: "sonium-sim-room.native-x86-sr3.v2".into(),
        host_architecture: std::env::consts::ARCH.to_string(),
        cpu_vendor: cpu_vendor(),
        cpu_simd_backend: sonium_fastmath::selected_backend().as_str().to_string(),
        dispatch_features: dispatch_features(),
        scalar_reference: "available:SONIUM_FASTMATH_FORCE_SCALAR=1".into(),
        vendor_reference: "not-benchmarked:no-verified-room-FEM-vendor-FFI".into(),
        frequencies: field.frequencies.clone(),
        source_names: field.source_names.clone(),
        listener_positions: field
            .listener_positions
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect(),
        node_coordinates: field
            .mesh
            .nodes
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect(),
        element_connectivity: field
            .mesh
            .elements
            .iter()
            .map(|element| element.nodes.clone())
            .collect(),
        element_count: field.mesh.elements.len(),
        boundary_count: field.mesh.boundaries.len(),
        output: encode_output(output),
        field: field
            .pressures
            .iter()
            .map(|source| {
                source
                    .iter()
                    .map(|frequency| frequency.iter().copied().map(ComplexValue::from).collect())
                    .collect()
            })
            .collect(),
        diagnostics,
        repeatability,
        reciprocity,
        convergence,
        cross_architecture: CrossArchitectureEvidence {
            status: "waived".into(),
            target: "aarch64".into(),
            reason: "user-authorized Arm64 comparison waiver on native Linux/x86 lane".into(),
        },
    }
}

fn assert_close(label: &str, expected: f64, actual: f64, relative_budget: f64) {
    let scale = expected.abs().max(actual.abs()).max(1.0e-12);
    assert!(
        (expected - actual).abs() / scale <= relative_budget,
        "{label} differs: expected={expected:?}, actual={actual:?}"
    );
}

fn assert_matches_reference(
    artifact: &NativeRoomReferenceArtifact,
    output: &SimulationOutput,
    field: &FemNodalField,
    diagnostics: &FemRunDiagnostics,
) {
    assert_eq!(artifact.schema_version, 2);
    assert_eq!(artifact.artifact_id, "sonium-sim-room.native-x86-sr3.v2");
    assert_eq!(artifact.host_architecture, "aarch64");
    assert_eq!(artifact.frequencies, field.frequencies);
    assert_eq!(artifact.source_names, field.source_names);
    assert_eq!(
        artifact.listener_positions.len(),
        field.listener_positions.len()
    );
    for (expected, actual) in artifact
        .listener_positions
        .iter()
        .zip(&field.listener_positions)
    {
        for (axis, (&expected, actual)) in expected
            .iter()
            .zip([actual.x, actual.y, actual.z])
            .enumerate()
        {
            assert_close(
                &format!("listener coordinate axis {axis}"),
                expected,
                actual,
                1.0e-14,
            );
        }
    }
    assert_eq!(artifact.node_coordinates.len(), field.mesh.nodes.len());
    assert_eq!(
        artifact.element_connectivity.len(),
        field.mesh.elements.len()
    );
    assert_eq!(artifact.element_count, field.mesh.elements.len());
    assert_eq!(artifact.boundary_count, field.mesh.boundaries.len());
    for (expected, actual) in artifact
        .element_connectivity
        .iter()
        .zip(&field.mesh.elements)
    {
        assert_eq!(expected, &actual.nodes);
    }
    for (expected, actual) in artifact.node_coordinates.iter().zip(&field.mesh.nodes) {
        for (&expected, actual) in expected.iter().zip([actual.x, actual.y, actual.z]) {
            assert_close("mesh node coordinate", expected, actual, 1.0e-14);
        }
    }

    let reference_output = decode_output(artifact);
    let report = compare_backend_outputs(
        "accepted-aarch64",
        &reference_output,
        "native-x86_64",
        output,
        0.0,
        1.0e-12,
    )
    .unwrap();
    report.require_within(1.0e-8, 1.0e-7).unwrap();

    assert_eq!(artifact.field.len(), field.pressures.len());
    for (source_index, (expected_source, actual_source)) in
        artifact.field.iter().zip(&field.pressures).enumerate()
    {
        assert_eq!(expected_source.len(), actual_source.len());
        for (frequency_index, (expected_frequency, actual_frequency)) in
            expected_source.iter().zip(actual_source).enumerate()
        {
            assert_eq!(expected_frequency.len(), actual_frequency.len());
            for (node_index, (expected, actual)) in
                expected_frequency.iter().zip(actual_frequency).enumerate()
            {
                assert_close(
                    &format!("field {source_index}/{frequency_index}/{node_index} real"),
                    expected.re,
                    actual.re,
                    1.0e-8,
                );
                assert_close(
                    &format!("field {source_index}/{frequency_index}/{node_index} imag"),
                    expected.im,
                    actual.im,
                    1.0e-8,
                );
            }
        }
    }

    assert_eq!(
        artifact.diagnostics.len(),
        diagnostics.frequency_stats.len()
    );
    for (expected, actual) in artifact
        .diagnostics
        .iter()
        .zip(&diagnostics.frequency_stats)
    {
        assert_close(
            "diagnostic frequency",
            expected.frequency_hz,
            actual.frequency_hz,
            1.0e-14,
        );
        assert_eq!(expected.converged, actual.converged);
        assert_close(
            "diagnostic residual",
            expected.max_residual,
            actual.max_residual,
            1.0e-6,
        );
        let actual_work = actual.operator_work.unwrap();
        let actual_power = actual.physical_power.unwrap();
        for (&expected, actual) in expected.operator_work.iter().zip([
            actual_work.source_work,
            actual_work.volume_damping_work,
            actual_work.boundary_work,
            actual_work.balance_residual,
        ]) {
            assert_close("operator-work reference", expected, actual, 1.0e-7);
        }
        for (&expected, actual) in expected.physical_power.iter().zip([
            actual_power.source_power_w,
            actual_power.volume_damping_power_w,
            actual_power.boundary_power_w,
            actual_power.balance_residual_w,
        ]) {
            assert_close("physical-power reference", expected, actual, 1.0e-7);
        }
        let actual_flux = actual.boundary_flux.as_ref().unwrap();
        assert_eq!(expected.boundary_flux.len(), actual_flux.samples.len());
        for (&(expected_tag, expected_power), actual_sample) in
            expected.boundary_flux.iter().zip(&actual_flux.samples)
        {
            assert_eq!(expected_tag, actual_sample.boundary_tag);
            assert_close(
                "boundary-flux reference",
                expected_power,
                actual_sample.active_power_w,
                1.0e-7,
            );
        }
    }
}

#[test]
fn native_x86_reference_reports_dispatch_and_physical_gates() {
    let mut mesh_config = FemMeshConfig::default();
    mesh_config.fine_radius = 0.4;
    let (first, first_field, first_diagnostics) =
        run_fem_with_mesh_field_report(&fixture(), &mesh_config).unwrap();
    let (second, second_field, second_diagnostics) =
        run_fem_with_mesh_field_report(&fixture(), &mesh_config).unwrap();

    let execution = first_diagnostics
        .execution
        .as_ref()
        .expect("CPU reference must publish execution provenance");
    assert_eq!(execution.requested_backend, "cpu");
    assert_eq!(execution.actual_backend, "cpu");
    assert_eq!(execution.fallback_status, "none");
    assert_eq!(execution.host_architecture, "x86_64");
    assert_eq!(
        execution.cpu_simd_backend.as_deref(),
        Some(selected_backend().as_str())
    );
    assert!(first_diagnostics.require_converged().is_ok());
    assert_eq!(
        first_diagnostics.physical_evidence_status,
        sonium_sim_room::fem_runner::FemPhysicalEvidenceStatus::Complete
    );
    assert_eq!(first_diagnostics.frequency_stats.len(), 2);
    assert!(first_diagnostics.frequency_stats.iter().all(|stats| {
        stats.operator_work.is_some()
            && stats.physical_power.is_some()
            && stats.boundary_flux.is_some()
            && stats.transfer_stats.is_none()
    }));
    first_field.validate().unwrap();
    second_field.validate().unwrap();

    let report = compare_backend_outputs(
        "native-x86-reference-1",
        &first,
        "native-x86-reference-2",
        &second,
        0.0,
        1.0e-12,
    )
    .unwrap();
    assert_eq!(report.sample_count, 8);
    report.require_within(1.0e-12, 1.0e-11).unwrap();
    assert!(second_diagnostics.require_converged().is_ok());

    let convergence_run = run_fem_mesh_convergence_ladder(
        &convergence_fixture(),
        &convergence_specs(),
        1.0e-12,
        1.0e-9,
        1.0e-12,
        1.0,
        1.0,
    )
    .expect("bounded native physical convergence ladder should execute");
    for level in [
        &convergence_run.coarse,
        &convergence_run.medium,
        &convergence_run.fine,
    ] {
        level.field.validate().unwrap();
        level.diagnostics.require_converged().unwrap();
        assert_eq!(
            level.diagnostics.physical_evidence_status,
            sonium_sim_room::fem_runner::FemPhysicalEvidenceStatus::Complete
        );
    }
    let convergence_report = &convergence_run.report;
    let convergence = ConvergenceEvidence {
        status: "pass".into(),
        listener_position_tolerance_m: 1.0e-12,
        frequency_tolerance_hz: 1.0e-9,
        relative_pressure_floor_pa: 1.0e-12,
        max_rms_relative_error: 1.0,
        max_max_relative_error: 1.0,
        sample_count: convergence_report.sample_count,
        coarse_to_medium_rms_relative_error: convergence_report.coarse_to_medium_rms_relative_error,
        coarse_to_medium_max_relative_error: convergence_report.coarse_to_medium_max_relative_error,
        medium_to_fine_rms_relative_error: convergence_report.medium_to_fine_rms_relative_error,
        medium_to_fine_max_relative_error: convergence_report.medium_to_fine_max_relative_error,
        monotonic: convergence_report.is_monotonic(),
        levels: [
            &convergence_run.coarse,
            &convergence_run.medium,
            &convergence_run.fine,
        ]
        .into_iter()
        .map(|level| ConvergenceLevelEvidence {
            label: level.spec.label.clone(),
            node_count: level.field.mesh.nodes.len(),
            element_count: level.field.mesh.elements.len(),
            max_residual: level.diagnostics.max_residual,
            converged: level.diagnostics.nonconverged_count == 0,
        })
        .collect(),
    };

    let forward = reciprocity_fixture(false);
    let reverse = reciprocity_fixture(true);
    let mut reciprocity_config = FemMeshConfig::default();
    reciprocity_config.fine_radius = 0.4;
    reciprocity_config.recycle_subspace_size = 0;
    let forward_output =
        sonium_sim_room::fem_runner::run_fem_with_mesh(&forward, &reciprocity_config).unwrap();
    let reverse_output =
        sonium_sim_room::fem_runner::run_fem_with_mesh(&reverse, &reciprocity_config).unwrap();
    let reciprocity_report = compare_reciprocal_transfer_outputs(
        &forward_output,
        &["listener-a".into(), "listener-b".into()],
        &reverse_output,
        &["source-a".into(), "source-b".into()],
        0.0,
        1.0e-12,
    )
    .unwrap();
    reciprocity_report.require_within(2.0e-6, 2.0e-5).unwrap();
    let reciprocity = ComparisonEvidence {
        status: "pass".into(),
        sample_count: reciprocity_report.sample_count,
        rms_relative_error: reciprocity_report.rms_relative_error,
        max_relative_error: reciprocity_report.max_relative_error,
        max_rms_relative_error: 2.0e-6,
        max_max_relative_error: 2.0e-5,
    };
    let repeatability = ComparisonEvidence {
        status: "pass".into(),
        sample_count: report.sample_count,
        rms_relative_error: report.rms_relative_error,
        max_relative_error: report.max_relative_error,
        max_rms_relative_error: 1.0e-12,
        max_max_relative_error: 1.0e-11,
    };

    let artifact = build_artifact(
        &first,
        &first_field,
        &first_diagnostics,
        repeatability,
        reciprocity,
        convergence,
    );
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_WRITE_REFERENCE_ARTIFACT") {
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&artifact).expect("reference artifact serializes"),
        )
        .expect("reference artifact writes");
    }
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_REFERENCE_ARTIFACT") {
        let bytes = std::fs::read(&path).expect("reference artifact reads");
        let accepted: NativeRoomReferenceArtifact =
            serde_json::from_slice(&bytes).expect("reference artifact parses");
        assert!(!accepted.cpu_simd_backend.trim().is_empty());
        assert_matches_reference(&accepted, &first, &first_field, &first_diagnostics);
    } else if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CROSS_ARCH_REFERENCE").is_some() {
        panic!(
            "cross-architecture evidence requires SONIUM_SIM_ROOM_REFERENCE_ARTIFACT pointing to an accepted aarch64 artifact"
        );
    }
    if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86").is_some() {
        assert_eq!(std::env::consts::ARCH, "x86_64");
    }
}
