//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::neo_hookean_sphere_inflation_pressure;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p = neo_hookean_sphere_inflation_pressure(1.2, 1.0e6, 0.002, 0.1);
        assert!(p > 0.0);
        println!("Neo-Hookean sphere inflation pressure = {:.6e}", p);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
