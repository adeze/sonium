//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::electroacoustics_piston_baffle_on_axis;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let rho = 1.21;
        let c0 = 343.0;
        let k = 12.0;
        let radius = 0.08;
        let velocity = 0.01;
        let p_near = electroacoustics_piston_baffle_on_axis(rho, c0, k, radius, velocity, 0.5);
        let p_far = electroacoustics_piston_baffle_on_axis(rho, c0, k, radius, velocity, 2.0);
        assert!(p_near.norm() > p_far.norm());
        assert!(p_far.norm() > 0.0);
        println!(
            "Baffled piston on-axis pressure magnitude = {:.6e}",
            p_near.norm()
        );
    });
    reference_result::emit("reference.fem-electroacoustics-piston-baffle.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
