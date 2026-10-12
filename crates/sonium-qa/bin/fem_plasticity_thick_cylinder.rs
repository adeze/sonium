//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::plasticity_thick_cylinder_limit_pressure;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p = plasticity_thick_cylinder_limit_pressure(0.05, 0.1, 250e6);
        assert!(p > 0.0);
        println!("Plastic thick-cylinder limit pressure = {:.6e}", p);
    });
    reference_result::emit("reference.fem-plasticity-thick-cylinder.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
