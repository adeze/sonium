//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::aero_wavy_wall_pressure_amplitude;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p1 = aero_wavy_wall_pressure_amplitude(1.2, 40.0, 0.002, 8.0);
        let p2 = aero_wavy_wall_pressure_amplitude(1.2, 40.0, 0.004, 8.0);
        assert!(p1 > 0.0);
        assert!((p2 / p1 - 2.0).abs() < 1e-12);
        println!("Wavy-wall pressure amplitude = {:.6e} Pa", p1);
    });
    reference_result::emit("reference.fem-aeroacoustics-wavy-wall.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
