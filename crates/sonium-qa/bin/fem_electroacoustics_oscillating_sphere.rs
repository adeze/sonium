//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::electroacoustics_oscillating_sphere_directivity;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let axial = electroacoustics_oscillating_sphere_directivity(0.0);
        let side = electroacoustics_oscillating_sphere_directivity(std::f64::consts::FRAC_PI_2);
        let back = electroacoustics_oscillating_sphere_directivity(std::f64::consts::PI);
        assert!((axial - 1.0).abs() < 1e-12);
        assert!(side.abs() < 1e-12);
        assert!((back + 1.0).abs() < 1e-12);
        println!("Oscillating sphere dipole directivity = {:.6e}", axial);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
