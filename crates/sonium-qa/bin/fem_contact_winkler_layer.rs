//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::contact_winkler_layer_pressure;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p = contact_winkler_layer_pressure(0.001, 1.0e6, 0.01);
        assert!((p - 1.0e5).abs() < 1e-9);
        println!("Winkler layer pressure = {:.6e}", p);
    });
    reference_result::emit("reference.fem-contact-winkler-layer.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
