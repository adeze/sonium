//! Independent CPU evidence for low-level dense Cholesky transactionality.

use sonium_qa::QaResult;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{CholeskyError, cholesky_factorize};
use std::time::Instant;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![1.0e-20_f64, 0.0, 0.0, 1.0e-20]);
    let factor = cholesky_factorize(&matrix)
        .map_err(|error| format!("finite tiny SPD factor should be admitted: {error:?}"))?;

    let mut destination = [53.0_f64; 2];
    let before = destination;
    let error = factor
        .solve_into(&[1.0e-20, f64::MAX], &mut destination)
        .expect_err("finite RHS should overflow the second forward pivot");
    if !matches!(
        error,
        CholeskyError::NumericalBreakdown {
            stage: "forward substitution",
            index: 1,
        }
    ) {
        return Err(format!("unexpected Cholesky error: {error:?}"));
    }
    if destination != before {
        return Err("Cholesky failure published a partial destination".into());
    }

    Ok(vec![("failure_typed", 1.0), ("destination_preserved", 1.0)])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 2x2 tiny-pivot dense Cholesky fixture")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("2x2 dense SPD Cholesky; single RHS")
                .tolerance("typed generated-overflow rejection and destination preservation")
                .seed("solvers-cholesky-transactional:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "solvers.cholesky-transactional.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:cholesky-transactional-v1";
