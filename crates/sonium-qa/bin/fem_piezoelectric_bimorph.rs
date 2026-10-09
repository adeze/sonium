//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::piezo_bimorph_tip_deflection;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let w = piezo_bimorph_tip_deflection(-175e-12, 100.0, 0.05, 0.001).abs();
        assert!(w > 0.0);
        println!("Piezo bimorph tip deflection scale = {:.6e}", w);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
