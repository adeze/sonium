//! Release-backed wrapper for the sustained restarted block-GMRES workload.

#[allow(dead_code)]
#[path = "../../sonium-solvers/tests/iterative_block_workload_reference.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::panic::catch_unwind;
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-block-workload-reference.v1";

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(reference::run_case) {
        Ok(Ok(metrics)) => metrics.into_iter().fold(
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference(
                    "independent partial-pivot dense solve and residual; scalar GMRES differential",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:iterative-block-workload-reference-v1",
                )
                .input_checksum(
                    "analytic/deterministic-fixture:iterative-block-workload-reference-v1",
                )
                .dimension("12x12 dense nonsymmetric operator; 3 RHS; restarted block GMRES")
                .formulation("native apply_block versus scalar-column restarted GMRES")
                .tolerance(
                    "solution error <= 5e-8; independently recomputed relative residual <= 5e-10",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64),
            |result, (name, value)| {
                let mut result = result.metric(name, value);
                match name {
                    "block_reference_error" | "scalar_reference_error" => {
                        result = result.maximum(name, 5.0e-8);
                    }
                    "block_residual" | "scalar_residual" | "reported_true_residual" => {
                        result = result.maximum(name, 5.0e-10);
                    }
                    _ => {}
                }
                result
            },
        ),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "block workload panicked"),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
