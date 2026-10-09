//! Independent CPU evidence for low-level dense LU transpose transactionality.

use sonium_qa::QaResult;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{LuError, lu_factorize};
use std::time::Instant;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![1.0e-200_f64]);
    let factor = lu_factorize(&matrix)
        .map_err(|error| format!("finite tiny LU factor should be admitted: {error:?}"))?;

    let mut transpose_destination = [37.0_f64];
    let transpose_before = transpose_destination;
    let transpose_error = factor
        .solve_transpose_into(&[f64::MAX], &mut transpose_destination)
        .expect_err("finite RHS should overflow the transpose pivot");
    if !matches!(
        transpose_error,
        LuError::NumericalBreakdown {
            stage: "transpose forward substitution",
            index: 0,
        }
    ) {
        return Err(format!("unexpected transpose error: {transpose_error:?}"));
    }
    if transpose_destination != transpose_before {
        return Err("transpose failure published a partial destination".into());
    }

    let mut adjoint_destination = [41.0_f64];
    let adjoint_before = adjoint_destination;
    let adjoint_error = factor
        .solve_conj_transpose_into(&[f64::MAX], &mut adjoint_destination)
        .expect_err("finite RHS should overflow the adjoint pivot");
    if !matches!(
        adjoint_error,
        LuError::NumericalBreakdown {
            stage: "transpose forward substitution",
            index: 0,
        }
    ) {
        return Err(format!("unexpected adjoint error: {adjoint_error:?}"));
    }
    if adjoint_destination != adjoint_before {
        return Err("adjoint failure published a partial destination".into());
    }

    Ok(vec![
        ("transpose_failure_typed", 1.0),
        ("adjoint_failure_typed", 1.0),
        ("destination_preserved", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 1x1 tiny-pivot dense LU transpose fixture")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("1x1 dense LU; transpose and Hermitian-adjoint single RHS")
                .tolerance("typed generated-overflow rejection and destination preservation")
                .seed("solvers-lu-transpose-transactional:cpu:v1")
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

const CASE_ID: &str = "solvers.lu-transpose-transactional.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:lu-transpose-transactional-v1";
