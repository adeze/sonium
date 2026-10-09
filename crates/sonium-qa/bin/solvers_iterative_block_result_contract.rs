//! Independent CPU contract checks for block GMRES/FGMRES failure records.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockFgmresSolution, BlockGmresSolution, KrylovOpCounts};
use sonium_solvers::{AccuracyStatus, SolverResultContract, SolverStatus};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-block-result-contract.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:solvers-iterative-block-result-contract-v1";

fn gmres_result(residual: f64, true_residual: f64, backward_error: f64) -> BlockGmresSolution<f64> {
    BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0]),
        iterations: 1,
        residuals: vec![residual],
        true_residuals: vec![true_residual],
        backward_errors: vec![backward_error],
        attainable_accuracy: vec![AccuracyStatus::NotAttained],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let gmres = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::NotChecked],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    let gmres_status = gmres.solver_status() == SolverStatus::NumericalError;
    gmres
        .validate_contract()
        .map_err(|error| format!("block GMRES sentinel rejected: {error}"))?;

    let fgmres = BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::NotAttained],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    let fgmres_status = fgmres.solver_status() == SolverStatus::NumericalError;
    fgmres
        .validate_contract()
        .map_err(|error| format!("block FGMRES sentinel rejected: {error}"))?;

    let invalid = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    let contradiction_rejected = invalid.validate_contract().is_err();
    let negative_diagnostic_rejections = [
        gmres_result(-1.0, 1.0, 1.0).validate_contract().is_err(),
        gmres_result(1.0, -1.0, 1.0).validate_contract().is_err(),
        gmres_result(1.0, 1.0, -1.0).validate_contract().is_err(),
    ];
    let negative_diagnostics_rejected = negative_diagnostic_rejections
        .iter()
        .all(|rejected| *rejected);

    if !negative_diagnostics_rejected {
        return Err("negative block diagnostics were accepted".into());
    }
    if !gmres_status || !fgmres_status || !contradiction_rejected {
        return Err(format!(
            "unexpected block contract state: gmres_status={gmres_status}, \
             fgmres_status={fgmres_status}, contradiction_rejected={contradiction_rejected}"
        ));
    }

    Ok(vec![
        ("gmres_numerical_error_status", gmres_status as u8 as f64),
        ("gmres_sentinel_valid", 1.0),
        ("fgmres_numerical_error_status", fgmres_status as u8 as f64),
        ("fgmres_sentinel_valid", 1.0),
        (
            "contradictory_attained_accuracy_rejected",
            contradiction_rejected as u8 as f64,
        ),
        (
            "negative_diagnostic_rejections",
            negative_diagnostic_rejections
                .iter()
                .filter(|rejected| **rejected)
                .count() as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference(
                    "independent block GMRES/FGMRES result and negative-diagnostic contract fixtures",
                )
                .reference_checksum(CHECKSUM)
                .input_checksum(CHECKSUM)
                .tolerance(
                    "all block result-contract metrics equal one; negative-diagnostic rejections = 3",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("ITERATIVE BLOCK RESULT CONTRACT FAILED");
            std::process::exit(1);
        }
    }
}
