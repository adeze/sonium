//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::ns_taylor_green_vortex;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let (u0, _, _) = ns_taylor_green_vortex(0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.01);
        let (u1, _, _) = ns_taylor_green_vortex(0.0, std::f64::consts::FRAC_PI_2, 10.0, 0.01);
        let (_, _, p0) = ns_taylor_green_vortex(0.0, 0.0, 0.0, 0.01);
        let (_, _, p1) = ns_taylor_green_vortex(0.0, 0.0, 10.0, 0.01);
        assert!(u1.abs() < u0.abs());
        assert!(p1.abs() < p0.abs());
        println!("Taylor-Green decayed velocity = {:.6e}", u1);
    });
    reference_result::emit("reference.fem-ns-taylor-green.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
