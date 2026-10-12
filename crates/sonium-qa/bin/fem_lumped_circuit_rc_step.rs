//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::circuit_rc_step_voltage;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let r = 1000.0;
        let c = 1e-6;
        let v = 5.0;
        assert!(circuit_rc_step_voltage(0.0, r, c, v).abs() < 1e-12);
        let v_tau = circuit_rc_step_voltage(r * c, r, c, v);
        assert!((v_tau / v - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        println!("RC step voltage at tau = {:.6e} V", v_tau);
    });
    reference_result::emit("reference.fem-lumped-circuit-rc-step.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
