//! Independent CPU evidence for checked CSR mutation arithmetic.

use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, SolverError};
use std::time::Instant;

const CASE_ID: &str = "solvers.csr-mutation-generated-nonfinite.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:csr-mutation-generated-nonfinite-v1";
const TOLERANCE: f64 = 1.0e-12;

fn run() -> Result<(f64, f64, f64, f64), String> {
    let mut scaled = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    scaled
        .try_scale(2.0)
        .map_err(|error| format!("finite CSR scale failed: {error:?}"))?;
    let finite_scale_error = (scaled.get(0, 0) - 4.0).abs();
    if !finite_scale_error.is_finite() || finite_scale_error > TOLERANCE {
        return Err(format!(
            "finite CSR scale error {finite_scale_error} exceeds tolerance"
        ));
    }

    let mut diagonal = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    diagonal
        .try_add_diagonal(0.5)
        .map_err(|error| format!("finite CSR diagonal update failed: {error:?}"))?;
    let finite_diagonal_error = (diagonal.get(0, 0) - 2.5).abs();
    if !finite_diagonal_error.is_finite() || finite_diagonal_error > TOLERANCE {
        return Err(format!(
            "finite CSR diagonal error {finite_diagonal_error} exceeds tolerance"
        ));
    }

    let mut overflowing_scale =
        CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let scale_before = overflowing_scale.values.clone();
    let scale_rejection = usize::from(matches!(
        overflowing_scale.try_scale(2.0),
        Err(SolverError::OperationFailed {
            operation: "CSR scale",
            ref reason,
        }) if reason.contains("non-finite")
    ));
    if scale_rejection != 1 || overflowing_scale.values != scale_before {
        return Err("CSR scale accepted overflow or mutated on rejection".into());
    }

    let mut overflowing_diagonal =
        CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let diagonal_before = overflowing_diagonal.values.clone();
    let diagonal_rejection = usize::from(matches!(
        overflowing_diagonal.try_add_diagonal(f64::MAX),
        Err(SolverError::OperationFailed {
            operation: "CSR add_diagonal",
            ref reason,
        }) if reason.contains("non-finite")
    ));
    if diagonal_rejection != 1 || overflowing_diagonal.values != diagonal_before {
        return Err("CSR diagonal update accepted overflow or mutated on rejection".into());
    }

    Ok((
        finite_scale_error,
        finite_diagonal_error,
        scale_rejection as f64,
        diagonal_rejection as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((finite_scale_error, finite_diagonal_error, scale_rejection, diagonal_rejection)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent finite mutation parity and transactional overflow rejection")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("1x1 CSR scale and stored-diagonal update")
                .tolerance("finite mutation errors <= 1e-12; one typed rejection per generated-value boundary")
                .metric("finite_scale_error", finite_scale_error)
                .metric("finite_diagonal_error", finite_diagonal_error)
                .metric("scale_generated_nonfinite_rejection", scale_rejection)
                .metric("add_diagonal_generated_nonfinite_rejection", diagonal_rejection)
                .maximum("finite_scale_error", TOLERANCE)
                .maximum("finite_diagonal_error", TOLERANCE)
                .minimum("scale_generated_nonfinite_rejection", 1.0)
                .minimum("add_diagonal_generated_nonfinite_rejection", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
