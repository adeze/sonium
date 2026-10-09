//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::{ns_mms_pressure, ns_mms_velocity};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let h = 1e-6;
        let x = 0.37;
        let y = 0.41;
        let (up, _) = ns_mms_velocity(x + h, y);
        let (um, _) = ns_mms_velocity(x - h, y);
        let (_, vp) = ns_mms_velocity(x, y + h);
        let (_, vm) = ns_mms_velocity(x, y - h);
        let divergence = (up - um) / (2.0 * h) + (vp - vm) / (2.0 * h);
        let p = ns_mms_pressure(0.0, 0.0);
        assert!(divergence.abs() < 1e-9);
        assert!((p - 1.0).abs() < 1e-12);
        println!("Navier-Stokes MMS divergence = {:.6e}", divergence);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
