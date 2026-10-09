//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::electroacoustics_piston_tube_pressure;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p = electroacoustics_piston_tube_pressure(1.21, 343.0, 0.01);
        let expected = 1.21 * 343.0 * 0.01;
        assert!((p - expected).abs() < 1e-12);
        println!("Piston tube pressure = {:.6e} Pa", p);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
