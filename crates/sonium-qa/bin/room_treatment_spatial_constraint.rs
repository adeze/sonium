//! Compile-only contract for RoomSim adjacency-based treatment patch guards.
//!
//! This source checks the public position-to-adjacency and MMA-constraint API
//! without launching a FEM solve, simulation, or QA workload.

use sonium_sim_room::treatment_optimization::{
    MultiFrequencyRoomTreatmentConfig, TreatmentSpatialConstraint,
};

fn main() {
    let spatial =
        TreatmentSpatialConstraint::from_positions(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 0.75, 0.2)
            .expect("compile-only QA adjacency fixture must be valid");
    let constraints = spatial
        .linearized_mma_constraints(&[0.3, 0.4])
        .expect("compile-only QA spatial fixture must linearize");
    assert_eq!(constraints.len(), 2);

    let config = MultiFrequencyRoomTreatmentConfig {
        spatial_constraint: Some(spatial),
        ..MultiFrequencyRoomTreatmentConfig::default()
    };
    let _ = config;
    println!("SUCCESS");
}
