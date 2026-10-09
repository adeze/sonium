//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::contact_rigid_punch_average_pressure;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let p = contact_rigid_punch_average_pressure(100.0, 0.01);
        assert!(p > 0.0);
        println!("Rigid punch average pressure = {:.6e}", p);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
