//! Release-style wrapper for checked TBEM quadrature admission.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/tbem_quadrature_policy_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.tbem-quadrature-policy-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_tbem_quadrature_policy_reference);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let success = outcome.is_ok();
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("active_elements", metrics.active_elements as f64)
            .metric("checked_dofs", metrics.checked_dofs as f64)
            .minimum("checked_dofs", 1.0)
            .metric("permissive_passed", f64::from(metrics.permissive_passed))
            .minimum("permissive_passed", 1.0)
            .metric("strict_rejected", f64::from(metrics.strict_rejected))
            .minimum("strict_rejected", 1.0)
            .metric(
                "solver_permissive_passed",
                f64::from(metrics.solver_permissive_passed),
            )
            .minimum("solver_permissive_passed", 1.0)
            .metric(
                "solver_strict_rejected",
                f64::from(metrics.solver_strict_rejected),
            )
            .minimum("solver_strict_rejected", 1.0)
            .metric(
                "compressed_strict_rejected",
                f64::from(metrics.compressed_strict_rejected),
            )
            .minimum("compressed_strict_rejected", 1.0)
            .input_checksum("bem:rigid-sphere:ka-0.233:tbem-adjacent-quadrature-policy")
            .reference_checksum("analytic/deterministic-fixture:bem-tbem-quadrature-policy-v1")
            .dimension(
                "3-D constant-panel CPU TBEM; builder and solver regular/self-pair admission",
            )
            .seed("bem-tbem-quadrature-policy:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "checked TBEM quadrature policy panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference_checksum("analytic/deterministic-fixture:bem-tbem-quadrature-policy-v1")
            .dimension("3-D constant-panel CPU TBEM; checked regular/self-pair adjacent orders")
            .seed("bem-tbem-quadrature-policy:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if success {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
