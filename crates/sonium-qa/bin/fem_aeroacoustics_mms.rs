//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::{aero_lpe_mms_laplacian, aero_lpe_mms_pressure};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let x = 0.5;
        let y = 0.5;
        let p = aero_lpe_mms_pressure(x, y);
        let lap = aero_lpe_mms_laplacian(x, y);
        assert!((p - 1.0).abs() < 1e-12);
        assert!((lap + 2.0 * std::f64::consts::PI.powi(2)).abs() < 1e-12);
        println!("Aeroacoustic MMS p = {:.6e}, laplacian = {:.6e}", p, lap);
    });
    reference_result::emit("reference.fem-aeroacoustics-mms.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
