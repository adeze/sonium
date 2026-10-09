use super::fem_mesh_config::FemMeshConfig;
use super::fem_run_diagnostics::FemPhysicalEvidenceStatus;
use super::run::run_fem_symmetric_report;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
    SymmetryConfig, SymmetryPlane,
};

#[test]
fn symmetric_cpu_runtime_marks_reduced_domain_energy_as_unavailable() {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(2.4, 2.0, 1.8)),
        vec![
            Source::omnidirectional(Point3D::new(0.6, 0.8, 0.6), 1.0),
            Source::omnidirectional(Point3D::new(1.8, 0.8, 0.6), 1.0),
        ],
        vec![Point3D::new(1.2, 1.0, 1.0)],
    );
    let matched = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: matched.clone(),
        ceiling: matched.clone(),
        walls: matched,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![100.0];

    let symmetry = SymmetryConfig {
        planes: vec![SymmetryPlane::X],
    };
    let config = FemMeshConfig {
        recycle_subspace_size: 0,
        ..FemMeshConfig::default()
    };
    let (_, diagnostics) = run_fem_symmetric_report(&simulation, &symmetry, &config)
        .expect("symmetric CPU FEM solve should execute");

    assert_eq!(
        diagnostics.physical_evidence_status,
        FemPhysicalEvidenceStatus::UnavailableReducedDomain
    );
    assert!(diagnostics.validate().is_ok());
    assert!(diagnostics.frequency_stats.iter().all(|stats| {
        stats.operator_work.is_none()
            && stats.physical_power.is_none()
            && stats.boundary_flux.is_none()
    }));
}
