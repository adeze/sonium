//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

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
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
