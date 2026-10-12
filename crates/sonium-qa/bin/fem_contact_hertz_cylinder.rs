//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::contact_hertz_cylinder_half_width;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let b = contact_hertz_cylinder_half_width(1000.0, 0.01, 1.0e9);
        assert!(b > 0.0 && b < 0.01);
        println!("Hertz cylinder half-width = {:.6e}", b);
    });
    reference_result::emit("reference.fem-contact-hertz-cylinder.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
