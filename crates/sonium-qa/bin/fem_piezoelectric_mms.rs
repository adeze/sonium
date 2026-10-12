//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::{piezo_axial_charge, piezo_bimorph_tip_deflection};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let u = piezo_bimorph_tip_deflection(100e-12, 10.0, 1.0, 0.1);
        let q = piezo_axial_charge(100e-12, 1.0);
        assert!(u > 0.0 && q > 0.0);
        println!("Piezo MMS scales: u={:.6e}, q={:.6e}", u, q);
    });
    reference_result::emit("reference.fem-piezoelectric-mms.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
