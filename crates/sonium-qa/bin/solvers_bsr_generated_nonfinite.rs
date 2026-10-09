//! Independent CPU evidence that checked BSR conversion rejects non-finite
//! values generated while packing finite duplicate CSR entries.

use sonium_qa::QaResult;
use sonium_solvers::sparse::{BsrMatrix, CsrMatrix};
use sonium_solvers_core::traits::SolverError;
use std::time::Instant;

const CASE_ID: &str = "solvers.bsr-generated-nonfinite.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:bsr-generated-nonfinite-v1";

fn run() -> Result<(f64, f64, usize), String> {
    let finite = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![1.25_f64, 2.75]);
    let bsr = BsrMatrix::<f64, 2>::try_from_csr(&finite)
        .map_err(|error| format!("finite duplicate BSR conversion failed: {error:?}"))?;
    let finite_action = bsr.matvec(&[1.0, 0.0])[0];
    let finite_error = (finite_action - 4.0).abs();
    if !finite_error.is_finite() || finite_error > 1.0e-12 {
        return Err(format!(
            "finite duplicate BSR action error {finite_error} exceeds tolerance"
        ));
    }

    let overflowing =
        CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![f64::MAX, f64::MAX]);
    let generated_nonfinite_rejection = usize::from(matches!(
        BsrMatrix::<f64, 2>::try_from_csr(&overflowing),
        Err(SolverError::OperationFailed {
            operation: "BSR construction",
            ref reason,
        }) if reason.contains("generated block values")
    ));
    if generated_nonfinite_rejection != 1 {
        return Err("BSR conversion accepted an overflow-generated block value".into());
    }

    Ok((finite_error, 1.0, generated_nonfinite_rejection))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((finite_error, finite_conversion_checks, generated_nonfinite_rejections)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .metric("finite_conversion_checks", finite_conversion_checks)
                .metric(
                    "generated_nonfinite_rejections",
                    generated_nonfinite_rejections as f64,
                )
                .metric("finite_action_error", finite_error)
                .maximum("finite_action_error", 1.0e-12)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
