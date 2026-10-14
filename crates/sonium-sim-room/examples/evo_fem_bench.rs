//! Evo benchmark harness for sonium-sim-room FEM solver
//!
//! Measures wall-clock time of a small FEM frequency sweep.
//! Run via: cargo run --release -p sonium-sim-room --example evo-fem-bench

use sonium_sim_room::fem_runner::{self, FemMeshConfig};
use sonium_sim_room::scenarios::{self, FrequencyConfig};
use std::time::Instant;

fn main() {
    let freq_config = FrequencyConfig {
        min_freq: 30.0,
        max_freq: 150.0,
        num_points: 15,
        spacing: sonium_xem::FrequencySpacing::default(),
    };
    let scenario = scenarios::scenario_by_name_with_config("small_stereo_2_0", &freq_config)
        .expect("scenario not found");

    let mesh_config = FemMeshConfig::default();

    let t0 = Instant::now();
    let output = fem_runner::run_fem_with_mesh(&scenario.simulation, &mesh_config)
        .expect("FEM solve failed");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Sanity check: ensure we got some non-trivial pressure values
    let max_pressure: f64 = output
        .pressures
        .iter()
        .flatten()
        .flatten()
        .map(|p| p.norm())
        .fold(0.0, f64::max);
    assert!(max_pressure > 0.0, "all pressures are zero");

    // Report result as JSON for evo
    println!(
        "{{\"score\": {:.6}, \"tasks\": {{\"fem_sweep_ms\": {:.6}}}}}",
        elapsed_ms, elapsed_ms
    );
}
