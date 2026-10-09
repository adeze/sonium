//! Focused regression tests for retained physical-domain FEM nodal fields.

use sonium_sim_room::{
    fem_runner, scenarios, source_calibration::SOURCE_VOLUME_VELOCITY_REFERENCE,
    spatial_field::SpatialFieldReference,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

#[test]
fn fem_nodal_field_matches_listener_interpolation() {
    let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
    let mut simulation = scenario.simulation.clone();
    simulation.sources.truncate(1);
    simulation.listening_positions.truncate(1);
    simulation.frequencies.truncate(1);

    let (output, field, diagnostics) = fem_runner::run_fem_with_mesh_field_report(
        &simulation,
        &fem_runner::FemMeshConfig::default(),
    )
    .unwrap();
    output.validate_layout().unwrap();
    field.validate().unwrap();
    assert_eq!(field.source_names.len(), 1);
    assert_eq!(field.frequencies, simulation.frequencies);
    assert!(field.mesh.nodes.len() >= 4);
    assert!(diagnostics.max_residual.is_finite());

    let sampled = field
        .sample_p1(0, 0, &simulation.listening_positions)
        .unwrap();
    let listener_pressure = output.pressures[0][0][0];
    assert!((sampled[0] - listener_pressure).norm() <= 1.0e-10);
}

#[test]
fn fem_nodal_field_matches_free_field_reference_at_off_node_probes() {
    let source_position = Point3D::new(0.4, 0.35, 0.35);
    let probes = vec![
        Point3D::new(1.2, 1.1, 1.0),
        Point3D::new(1.0, 0.95, 0.9),
        Point3D::new(1.25, 0.6, 0.85),
    ];
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(source_position, 1.0)],
        probes.clone(),
    );
    let matched_wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: matched_wall.clone(),
        ceiling: matched_wall.clone(),
        walls: matched_wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![400.0];

    let (_, field, diagnostics) = fem_runner::run_fem_with_mesh_field_report(
        &simulation,
        &fem_runner::FemMeshConfig {
            fine_radius: 0.4,
            modal_damping_ratio: 0.0,
            source_model: fem_runner::FemSourceModel::PointLoad,
            ..fem_runner::FemMeshConfig::default()
        },
    )
    .unwrap();
    field.validate().unwrap();
    assert!(diagnostics.max_residual.is_finite());

    let sampled = field.sample_p1(0, 0, &probes).unwrap();
    let reference = SpatialFieldReference::monopole(
        source_position,
        probes,
        simulation.frequencies.clone(),
        simulation.speed_of_sound,
        SOURCE_VOLUME_VELOCITY_REFERENCE,
    )
    .unwrap();
    reference.validate_layout().unwrap();

    for (probe_index, (actual, expected_row)) in
        sampled.iter().zip(&reference.pressures).enumerate()
    {
        let expected = expected_row[0].norm();
        let ratio = actual.norm() / expected;
        assert!(
            (0.6..=1.7).contains(&ratio),
            "off-node FEM field must track the free-field reference: probe={probe_index}, ratio={ratio:.3}, actual={:.4e}, reference={expected:.4e}",
            actual.norm()
        );
    }
}
