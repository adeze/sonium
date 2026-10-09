//! Independent CPU QA for operator-retaining dense direct diagnostics.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectError, DenseDirectMethod, DirectFactorization, MatrixClass,
    factorize_dense_configured,
};
use sonium_solvers_core::traits::SolverResultContract;
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-configured-diagnostics.v1";

fn main() {
    let started = Instant::now();
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]);
    let config = DenseDirectConfig {
        matrix_class: MatrixClass::General,
        method: DenseDirectMethod::Lu,
        estimate_condition: true,
        iterative_refinement_steps: 2,
        ..DenseDirectConfig::default()
    };

    let outcome = (|| -> Result<QaResult, String> {
        let factor =
            factorize_dense_configured(&matrix, &config).map_err(|error| error.to_string())?;
        let condition_estimate = factor
            .condition_estimate()
            .ok_or_else(|| "condition estimate was not retained".to_string())?;
        let condition_error = (condition_estimate - 3.0).abs();

        let result = factor
            .solve_with_result(&[6.0, 8.0])
            .map_err(|error| error.to_string())?;
        result
            .validate_contract()
            .map_err(|error| error.to_string())?;
        let solution_error = result
            .x
            .iter()
            .zip([1.0_f64, 2.0])
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        let residual = result
            .residual
            .ok_or_else(|| "configured result omitted residual".to_string())?;
        let backward_error = result
            .backward_error
            .ok_or_else(|| "configured result omitted backward error".to_string())?;

        let invalid_tolerance = matches!(
            factor.solve_refined_with_tolerance(&[6.0, 8.0], f64::NAN, 1),
            Err(DenseDirectError::InvalidRefinementTolerance)
        );
        if !invalid_tolerance {
            return Err("invalid refinement tolerance was accepted".to_string());
        }

        Ok(QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("condition_estimate", condition_estimate)
            .metric("condition_error", condition_error)
            .metric("solution_error", solution_error)
            .metric("residual", residual)
            .metric("backward_error", backward_error)
            .metric(
                "invalid_tolerance_rejection",
                u8::from(invalid_tolerance) as f64,
            )
            .maximum("condition_error", 1.0e-12)
            .maximum("solution_error", 1.0e-12)
            .maximum("residual", 1.0e-12)
            .maximum("backward_error", 1.0e-12)
            .minimum("invalid_tolerance_rejection", 1.0)
            .elapsed_ms(started.elapsed().as_millis() as u64))
    })();

    match outcome {
        Ok(record) => {
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("DIRECT CONFIGURED DIAGNOSTICS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
