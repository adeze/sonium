//! Release-backed FEM-to-core solver failure vocabulary reference.

use sonium_fem::solver::SolverError;
use sonium_qa::QaResult;
use sonium_solvers::traits::{AccuracyStatus, SolverFailure, SolverStatus};

const CASE_ID: &str = "fem.solver-failure-contract.v1";

fn run() -> Result<QaResult, String> {
    let cases = [
        (
            SolverError::ConvergenceFailure(7, 1.0e-3).status(),
            SolverStatus::MaxIterationsReached,
            SolverError::ConvergenceFailure(7, 1.0e-3).accuracy_status(),
            AccuracyStatus::NotAttained,
        ),
        (
            SolverError::SingularMatrix.status(),
            SolverStatus::Breakdown,
            SolverError::SingularMatrix.accuracy_status(),
            AccuracyStatus::NotChecked,
        ),
        (
            SolverError::DimensionMismatch {
                expected: 2,
                actual: 3,
            }
            .status(),
            SolverStatus::NumericalError,
            SolverError::DimensionMismatch {
                expected: 2,
                actual: 3,
            }
            .accuracy_status(),
            AccuracyStatus::NotChecked,
        ),
        (
            SolverError::InvalidConfiguration("bad tolerance".into()).status(),
            SolverStatus::NumericalError,
            SolverError::InvalidConfiguration("bad tolerance".into()).accuracy_status(),
            AccuracyStatus::NotChecked,
        ),
    ];
    let checks_passed = cases
        .iter()
        .filter(
            |(actual_status, expected_status, actual_accuracy, expected_accuracy)| {
                actual_status == expected_status && actual_accuracy == expected_accuracy
            },
        )
        .count();
    if checks_passed != cases.len() {
        return Err(format!(
            "FEM failure vocabulary checks: {checks_passed}/{}",
            cases.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("failure_cases_passed", checks_passed as f64)
        .minimum("failure_cases_passed", cases.len() as f64)
        .input_checksum("analytic/deterministic-fixture:fem-solver-failure-contract-v1")
        .reference("shared SolverStatus and AccuracyStatus mapping for FEM solver errors")
        .reference_checksum("analytic/deterministic-fixture:fem-solver-failure-contract-v1")
        .dimension("convergence, singular, dimension, and configuration FEM failures")
        .formulation("FEM SolverError implements sonium-solvers SolverFailure vocabulary")
        .tolerance("all four status/accuracy mappings match the shared contract")
        .seed("fem-solver-failure-contract:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("FEM solver failure result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("FEM solver failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
