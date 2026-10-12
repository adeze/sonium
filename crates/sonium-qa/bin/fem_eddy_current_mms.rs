//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::eddy_mms_a_phi;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let a = eddy_mms_a_phi(0.5, 0.5);
        assert!((a.re - 1.0).abs() < 1e-12);
        assert!(a.im.abs() < 1e-12);
        println!(
            "Eddy A-phi MMS value at center = {:.6e}+{:.6e}i",
            a.re, a.im
        );
    });
    reference_result::emit("reference.fem-eddy-current-mms.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
