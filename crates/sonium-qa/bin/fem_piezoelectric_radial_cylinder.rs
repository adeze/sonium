//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::piezo_radial_cylinder_capacitance;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let c = piezo_radial_cylinder_capacitance(1e-8, 0.01, 0.02);
        assert!(c > 0.0);
        println!("Piezo radial-cylinder capacitance per length = {:.6e}", c);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
