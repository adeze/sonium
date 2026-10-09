//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::{biharmonic_clamped_circular_center, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let d = plate_bending_stiffness(70e9, 0.33, 0.005);
        let w = biharmonic_clamped_circular_center(500.0, 0.25, d);
        assert!(w > 0.0);
        println!("Biharmonic clamped circular center deflection = {:.6e}", w);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
