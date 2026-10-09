//! CPU reference for the public generic compatibility facade's CG path.

#[allow(dead_code)]
#[path = "../../sonium-solvers/tests/compatibility_cg_execution.rs"]
mod reference;

use sonium_qa::QaResult;
use std::panic::catch_unwind;
use std::time::Instant;

const CASE_ID: &str = "solvers.compatibility-cg-execution.v1";

fn metric_limit(name: &str) -> f64 {
    if name.starts_with("f32_") || name.starts_with("complex32_") {
        1.0e-3
    } else {
        1.0e-11
    }
}

fn main() {
    let started = Instant::now();
    let (passed, result) = match catch_unwind(reference::run_case) {
        Ok(Ok(metrics)) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .reference("independent partial-pivot dense reference and CSR residual")
                    .reference_checksum(
                        "analytic/deterministic-fixture:compatibility-cg-execution-v1",
                    )
                    .input_checksum("analytic/deterministic-fixture:compatibility-cg-execution-v1")
                    .dimension("4x4 Hermitian positive-definite CSR through solve<T>")
                    .formulation("public high-level checked compatibility facade with CG")
                    .tolerance("f32/Complex32 <= 1e-3; f64/Complex64 <= 1e-11")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| {
                    result.metric(name, value).maximum(name, metric_limit(name))
                },
            );
            (true, result)
        }
        Ok(Err(error)) => (
            false,
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
        ),
        Err(payload) => (
            false,
            QaResult::failed(
                CASE_ID,
                format!("CG compatibility packet panicked: {payload:?}"),
            )
            .elapsed_ms(started.elapsed().as_millis() as u64),
        ),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
