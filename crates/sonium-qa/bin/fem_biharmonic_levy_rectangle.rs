//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::{biharmonic_levy_rectangle_first_term, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let d = plate_bending_stiffness(200e9, 0.3, 0.01);
        let w1 = biharmonic_levy_rectangle_first_term(1000.0, 1.0, 0.5, d);
        let w2 = biharmonic_levy_rectangle_first_term(1000.0, 1.0, 1.0, d);
        assert!(w1 < w2);
        println!(
            "Biharmonic Levy rectangle first-term deflection = {:.6e}",
            w1
        );
    });
    reference_result::emit("reference.fem-biharmonic-levy-rectangle.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
