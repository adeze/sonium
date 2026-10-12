//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::SPEED_OF_LIGHT;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let radius = 0.05;
        let freq = 1.0e9;
        let ka = 2.0 * std::f64::consts::PI * freq * radius / SPEED_OF_LIGHT;
        assert!(ka > 1.0 && ka < 2.0);
        println!("Maxwell PEC cylinder scattering setup: ka = {:.6}", ka);
    });
    reference_result::emit(
        "reference.fem-maxwell-pec-cylinder-scattering.v1",
        result.is_ok(),
    );
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
