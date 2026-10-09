//! Compile-only contract for the RoomSim continuous treatment-density filter.
//!
//! This source checks the public position-based filter, transpose pullback,
//! and optimizer configuration surface without launching a FEM solve,
//! simulation, or QA workload.

use sonium_sim_room::treatment_optimization::{
    MultiFrequencyRoomTreatmentConfig, TreatmentDensityFilter,
};

fn main() {
    let filter = TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], 0.5)
        .expect("compile-only QA density filter fixture must be valid");
    let filtered = filter
        .apply(&[0.2, 0.8])
        .expect("compile-only QA density filter must apply");
    assert_eq!(filtered.len(), 2);
    let gradient = filter
        .pullback_gradient(&[1.0, 1.0])
        .expect("compile-only QA density pullback must apply");
    assert_eq!(gradient.len(), 2);

    let config = MultiFrequencyRoomTreatmentConfig {
        density_filter: Some(filter),
        ..MultiFrequencyRoomTreatmentConfig::default()
    };
    let _ = config;
    println!("SUCCESS");
}
