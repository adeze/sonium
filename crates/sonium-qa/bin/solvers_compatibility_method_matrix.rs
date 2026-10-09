//! Release-facing compatibility facade method/initial-guess contract matrix.

#[path = "../../sonium-solvers/tests/compatibility_method_matrix.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::panic::catch_unwind;
use std::time::Instant;

const CASE_ID: &str = "solvers.compatibility-method-matrix.v1";

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(reference::run_case) {
        Ok(Ok(metrics)) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
            .reference("independent known solutions, nine facade strategies, scalar-family checks, and typed rejection")
                .reference_checksum("analytic/deterministic-fixture:compatibility-method-matrix-v1")
                .input_checksum("analytic/deterministic-fixture:compatibility-method-matrix-v1")
            .dimension("4x4 real SPD, 2x2 Complex64 identity, 2x2 f32/Complex32 scalar fixtures")
            .formulation(
                "all nine public GMRES/FGMRES/CG facade strategies, initial-guess and scalar-family contracts",
            )
            .tolerance("f64 solution/residual/backward error <= 1e-10; f32/Complex32 residual <= 1e-5; 15 contract checks")
                .seed("solvers-compatibility-method-matrix:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            if name == "contract_checks" {
                result = result.minimum(name, 15.0);
            } else if name == "facade_method_checks" {
                result = result.minimum(name, 9.0);
            } else if name == "f32_residual" || name == "complex32_residual" {
                result = result.maximum(name, 1.0e-5);
            } else {
                result = result.maximum(name, 1.0e-10);
                }
            }
            result
        }
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "compatibility method matrix panicked"),
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
