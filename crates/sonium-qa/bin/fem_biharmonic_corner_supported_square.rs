//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::{biharmonic_levy_rectangle_first_term, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let d = plate_bending_stiffness(200e9, 0.3, 0.01);
        let w = biharmonic_levy_rectangle_first_term(1000.0, 1.0, 1.0, d);
        assert!(w > 0.0);
        println!(
            "Biharmonic corner-supported square reference scale = {:.6e}",
            w
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
