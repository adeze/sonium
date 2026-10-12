//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::ns_burgers_viscous_shock;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let nu = 0.02;
        let u_left = 1.0;
        let u_right = 0.0;
        let u_far_left = ns_burgers_viscous_shock(-1.0, 0.0, nu, u_left, u_right);
        let u_far_right = ns_burgers_viscous_shock(1.0, 0.0, nu, u_left, u_right);
        let u_center = ns_burgers_viscous_shock(0.0, 0.0, nu, u_left, u_right);
        assert!((u_far_left - u_left).abs() < 1e-5);
        assert!((u_far_right - u_right).abs() < 1e-5);
        assert!((u_center - 0.5).abs() < 1e-12);
        println!("Burgers shock center velocity = {:.6e}", u_center);
    });
    reference_result::emit("reference.fem-ns-1d-burgers.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
