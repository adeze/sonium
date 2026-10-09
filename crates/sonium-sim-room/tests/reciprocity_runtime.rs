//! Focused runtime reciprocity evidence for the production CPU FEM path.
//!
//! The fixture solves the same two-point transfer problem in both source
//! directions.  It intentionally stays at one frequency and two sources so
//! this is a bounded dispatch/physics regression rather than a full room
//! simulation or a QA-binary invocation.

use sonium_sim_room::fem_runner::{FemMeshConfig, run_fem_with_mesh};
use sonium_sim_room::reciprocity_validation::compare_reciprocal_transfer_outputs;
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

const SOURCE_NAMES: [&str; 2] = ["source-a", "source-b"];
const LISTENER_NAMES: [&str; 2] = ["listener-a", "listener-b"];

const SOURCE_POSITIONS: [Point3D; 2] = [
    Point3D {
        x: 0.35,
        y: 0.4,
        z: 0.35,
    },
    Point3D {
        x: 0.55,
        y: 1.05,
        z: 0.45,
    },
];
const LISTENER_POSITIONS: [Point3D; 2] = [
    Point3D {
        x: 1.2,
        y: 1.05,
        z: 1.0,
    },
    Point3D {
        x: 0.95,
        y: 0.6,
        z: 1.05,
    },
];

fn fixture(
    source_positions: &[Point3D],
    listener_positions: &[Point3D],
    names: &[&str],
) -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        source_positions
            .iter()
            .zip(names)
            .map(|(position, name)| {
                Source::omnidirectional(*position, 1.0).with_name((*name).into())
            })
            .collect(),
        listener_positions.to_vec(),
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
    simulation.frequencies = vec![200.0];
    simulation
}

#[test]
fn production_fem_transfer_is_reciprocal_for_two_sources_and_listeners() {
    let forward = fixture(&SOURCE_POSITIONS, &LISTENER_POSITIONS, &SOURCE_NAMES);
    let reverse = fixture(&LISTENER_POSITIONS, &SOURCE_POSITIONS, &LISTENER_NAMES);
    let mut mesh_config = FemMeshConfig::default();
    mesh_config.fine_radius = 0.4;

    let forward_output = run_fem_with_mesh(&forward, &mesh_config).unwrap();
    let reverse_output = run_fem_with_mesh(&reverse, &mesh_config).unwrap();
    let forward_listener_names = LISTENER_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    let reverse_listener_names = SOURCE_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();

    let report = compare_reciprocal_transfer_outputs(
        &forward_output,
        &forward_listener_names,
        &reverse_output,
        &reverse_listener_names,
        0.0,
        1.0e-12,
    )
    .unwrap();

    assert_eq!(report.sample_count, 4);
    report.require_within(2.0e-6, 2.0e-5).unwrap();
}
