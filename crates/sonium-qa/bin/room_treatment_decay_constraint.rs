//! Compile-only contract for the RoomSim reduced-order treatment decay API.
//!
//! This source verifies that the public Eyring prediction, analytic
//! gradient, and MMA constraint surface remains available to QA consumers.
//! It intentionally performs no FEM solve, simulation, or QA gate when this
//! source is compiled.

use sonium_sim_room::treatment_optimization::{
    MultiFrequencyRoomTreatmentConfig, TreatmentDecayConstraint,
};

fn main() {
    let decay =
        TreatmentDecayConstraint::new(100.0, 150.0, 5.0, vec![2.0, 4.0], 0.2, Some(0.5), Some(3.0))
            .expect("compile-only QA fixture must be physically admissible");
    let constraints = decay
        .linearized_mma_constraints(&[0.2, 0.3])
        .expect("compile-only QA fixture must linearize");
    assert_eq!(constraints.len(), 2);

    let config = MultiFrequencyRoomTreatmentConfig {
        decay_constraint: Some(decay),
        ..MultiFrequencyRoomTreatmentConfig::default()
    };
    let _ = config;
    println!("SUCCESS");
}
