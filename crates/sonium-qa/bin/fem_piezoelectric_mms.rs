//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::{piezo_axial_charge, piezo_bimorph_tip_deflection};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let u = piezo_bimorph_tip_deflection(100e-12, 10.0, 1.0, 0.1);
        let q = piezo_axial_charge(100e-12, 1.0);
        assert!(u > 0.0 && q > 0.0);
        println!("Piezo MMS scales: u={:.6e}, q={:.6e}", u, q);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
