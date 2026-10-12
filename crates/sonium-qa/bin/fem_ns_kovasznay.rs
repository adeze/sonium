//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::ns_kovasznay_solution;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let (u0, v0, _) = ns_kovasznay_solution(0.0, 0.0, 40.0);
        let (u1, _, _) = ns_kovasznay_solution(1.0, 0.0, 40.0);
        assert!(u0.abs() < 1e-12);
        assert!(v0.abs() < 1e-12);
        assert!(u1 > u0);
        println!("Kovasznay downstream velocity = {:.6e}", u1);
    });
    reference_result::emit("reference.fem-ns-kovasznay.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
