//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::plasticity_1d_isotropic_hardening_stress;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let ux = (std::f64::consts::PI * 0.5).sin() * (std::f64::consts::PI * 0.5).sin();
        let stress = plasticity_1d_isotropic_hardening_stress(1e-3, 200e9, 250e6, 1e9);
        assert!((ux - 1.0).abs() < 1e-12);
        assert!(stress > 0.0);
        println!(
            "Plasticity MMS displacement center = {:.6e}, stress scale = {:.6e}",
            ux, stress
        );
    });
    reference_result::emit("reference.fem-plasticity-mms.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
