//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::aero_convected_duct_wavenumbers;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let omega = 2.0 * std::f64::consts::PI * 500.0;
        let c0 = 343.0;
        let u0 = 34.3;
        let (k_plus, k_minus) = aero_convected_duct_wavenumbers(omega, c0, u0);
        assert!(k_plus < omega / c0);
        assert!(k_minus > omega / c0);
        assert!((k_plus - omega / (c0 + u0)).abs() < 1e-12);
        println!(
            "Convected duct wavenumbers = {:.6e}, {:.6e}",
            k_plus, k_minus
        );
    });
    reference_result::emit(
        "reference.fem-aeroacoustics-convected-duct.v1",
        result.is_ok(),
    );
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
