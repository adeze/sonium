//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::electroacoustics_pulsating_sphere_impedance;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let z = electroacoustics_pulsating_sphere_impedance(1.21, 343.0, 4.0, 0.2);
        assert!(z.re > 0.0);
        assert!(z.im > z.re);
        let z_high = electroacoustics_pulsating_sphere_impedance(1.21, 343.0, 200.0, 0.2);
        assert!((z_high.re / (1.21 * 343.0) - 1.0).abs() < 1e-3);
        println!("Pulsating sphere impedance = {:.6e}+{:.6e}i", z.re, z.im);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
