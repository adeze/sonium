//! Native rigid-sphere analytical accuracy; no external solver parity claim.

// Rust guideline compliant 2026-02-21
#[path = "bem_mesh2hrtf_parity.rs"]
mod analytical_accuracy;

fn main() {
    analytical_accuracy::main();
}
