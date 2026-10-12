//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::{biharmonic_ss_rectangular_sinusoidal_center, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let d = plate_bending_stiffness(200e9, 0.3, 0.01);
        let w = biharmonic_ss_rectangular_sinusoidal_center(1000.0, 1.0, 0.8, d);
        assert!(w > 0.0);
        println!(
            "Biharmonic SS rectangle sinusoidal center deflection = {:.6e}",
            w
        );
    });
    reference_result::emit("reference.fem-biharmonic-ss-rectangular-sinusoidal.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
