//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::maxwell_pec_cavity_1d_frequency;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let f = maxwell_pec_cavity_1d_frequency(1, 0.5);
        assert!((f - 299_792_458.0).abs() < 1e-6);
        println!("Maxwell 1D PEC cavity: f1 = {:.6e} Hz", f);
    });
    reference_result::emit("reference.fem-maxwell-1d-pec-cavity.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
