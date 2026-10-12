//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::eddy_cylinder_ac_low_ka;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let ratio = eddy_cylinder_ac_low_ka(0.5);
        assert!(ratio < 1.0 && ratio > 0.99);
        println!("Eddy cylinder AC low-ka field ratio = {:.6e}", ratio);
    });
    reference_result::emit("reference.fem-eddy-current-cylinder-ac.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
