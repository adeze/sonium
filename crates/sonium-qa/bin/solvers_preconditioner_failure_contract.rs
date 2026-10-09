//! Release-backed shared failure vocabulary for checked CPU preconditioner and SpMV setup.

use sonium_qa::QaResult;
use sonium_solvers::{AccuracyStatus, ChebyshevError, NumaSpmvError, SolverFailure, SolverStatus};

const CASE_ID: &str = "solvers.preconditioner-failure-contract.v1";

fn matches_contract<E: SolverFailure>(
    error: &E,
    expected_status: SolverStatus,
    expected_accuracy: AccuracyStatus,
) -> bool {
    error.status() == expected_status && error.accuracy_status() == expected_accuracy
}

fn run() -> Result<QaResult, String> {
    let chebyshev_errors = [
        ChebyshevError::ComplexEntries { count: 2 },
        ChebyshevError::NonPositiveDiagonal { index: 3 },
        ChebyshevError::InvalidCsrStructure,
    ];
    let chebyshev_checks = chebyshev_errors
        .iter()
        .filter(|error| {
            matches_contract(
                *error,
                SolverStatus::NumericalError,
                AccuracyStatus::NotChecked,
            )
        })
        .count();

    let numa_errors = [
        NumaSpmvError::DimensionMismatch {
            expected: 4,
            got: 3,
        },
        NumaSpmvError::InvalidPartition {
            partition: 1,
            reason: "row range is empty".to_owned(),
        },
        NumaSpmvError::NonFinite { operation: "spmv" },
        NumaSpmvError::WorkerPanicked,
    ];
    let numa_checks = numa_errors
        .iter()
        .filter(|error| {
            matches_contract(
                *error,
                SolverStatus::NumericalError,
                AccuracyStatus::NotChecked,
            )
        })
        .count();

    let checks_passed = chebyshev_checks + numa_checks;
    let expected_checks = chebyshev_errors.len() + numa_errors.len();
    if checks_passed != expected_checks {
        return Err(format!(
            "preconditioner failure vocabulary checks: {checks_passed}/{expected_checks}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .minimum("checks_passed", expected_checks as f64)
        .metric("chebyshev_checks", chebyshev_checks as f64)
        .metric("numa_checks", numa_checks as f64)
        .input_checksum("analytic/deterministic-fixture:preconditioner-failure-contract-v1")
        .reference("shared SolverFailure status and accuracy vocabulary")
        .reference_checksum("analytic/deterministic-fixture:preconditioner-failure-contract-v1")
        .formulation("public preconditioner and sparse CPU error enums implement SolverFailure")
        .seed("solvers-preconditioner-failure-contract:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
