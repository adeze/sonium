//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::contact_hertz_sphere_radius;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let a = contact_hertz_sphere_radius(100.0, 0.01, 1.0e9);
        assert!(a > 0.0 && a < 0.01);
        println!("Hertz sphere contact radius = {:.6e}", a);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
