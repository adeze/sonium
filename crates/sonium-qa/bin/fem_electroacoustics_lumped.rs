//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::electroacoustics_lumped_displacement;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let x_static = electroacoustics_lumped_displacement(2.0, 0.1, 0.3, 1000.0, 0.0);
        let x_res = electroacoustics_lumped_displacement(2.0, 0.1, 0.3, 1000.0, 100.0);
        assert!((x_static.re - 0.002).abs() < 1e-12);
        assert!(x_res.norm() > x_static.norm());
        println!("Lumped electroacoustic displacement = {:.6e}", x_static.re);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
