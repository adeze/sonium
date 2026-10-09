//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::maxwell_circular_waveguide_te11_cutoff;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let fc = maxwell_circular_waveguide_te11_cutoff(0.01);
        let expected = 8.784_923_322_365_324e9;
        assert!((fc - expected).abs() / expected < 1e-6);
        assert!((2.0 * fc - maxwell_circular_waveguide_te11_cutoff(0.005)).abs() / fc < 1e-12);
        println!("Maxwell circular TE11 cutoff: {:.6e} Hz", fc);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
