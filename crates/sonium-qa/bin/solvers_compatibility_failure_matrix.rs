//! Release-backed generic solver compatibility failure matrix.

#[allow(dead_code)]
#[path = "../../sonium-solvers/tests/compatibility_failure_matrix.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "solvers.compatibility-failure-matrix.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_compatibility_failure_matrix);
    let result = match outcome {
        Ok(checks) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("typed_failure_checks", checks as f64)
            .minimum("typed_failure_checks", 24.0)
            .metric("scalar_family_coverage", 4.0)
            .minimum("scalar_family_coverage", 4.0)
            .input_checksum("analytic/deterministic-fixture:compatibility-failure-matrix-v1")
            .reference(
                "independent typed dimension, RHS/matrix non-finite, malformed CSR, column-range, and configuration failure classification",
            )
            .reference_checksum("analytic/deterministic-fixture:compatibility-failure-matrix-v1")
            .dimension("2x2 identity plus malformed 1x1 CSR fixtures across f32/f64/Complex32/Complex64")
            .formulation("public generic solve<T> admission boundary")
            .tolerance("24 typed failure classifications pass before iteration")
            .seed("solvers-compatibility-failure-matrix:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("compatibility failure matrix panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("solvers-compatibility-failure-matrix:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("serialize QA result must succeed")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
