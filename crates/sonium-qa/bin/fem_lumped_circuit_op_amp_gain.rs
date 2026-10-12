//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::circuit_inverting_op_amp_gain;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let gain = circuit_inverting_op_amp_gain(10_000.0, 47_000.0);
        assert!((gain + 4.7).abs() < 1e-12);
        println!("Inverting op-amp gain = {:.6e}", gain);
    });
    reference_result::emit("reference.fem-lumped-circuit-op-amp-gain.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
