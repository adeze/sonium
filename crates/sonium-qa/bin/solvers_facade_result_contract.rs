//! Release-backed public solver-facade result and error contract.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolverType,
    SolverResultContract, SolverStatus, solve_real,
};

const CASE_ID: &str = "solvers.facade-result-contract.v1";

fn run() -> Result<QaResult, String> {
    let matrix = CsrMatrix::identity(2);
    let result = solve_real(
        &matrix,
        &[2.0, -3.0],
        &LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            exact_residual_check_interval: 1,
            ..Default::default()
        },
    )
    .map_err(|error| format!("identity solve failed: {error}"))?;

    let rhs_dimension = solve_real(&matrix, &[1.0], &LinearSolveConfig::default())
        .expect_err("RHS dimension mismatch must be rejected");
    let non_finite = solve_real(
        &CsrMatrix::identity(1),
        &[f64::NAN],
        &LinearSolveConfig::default(),
    )
    .expect_err("non-finite RHS must be rejected");
    let invalid_config = solve_real(
        &matrix,
        &[1.0, 2.0],
        &LinearSolveConfig {
            max_iterations: 0,
            ..Default::default()
        },
    )
    .expect_err("zero max_iterations must be rejected");

    let checks = [
        SolverResultContract::solver_status(&result) == SolverStatus::Converged,
        SolverResultContract::accuracy_status(&result) == AccuracyStatus::Attained,
        SolverResultContract::validate_contract(&result).is_ok(),
        result.x == vec![2.0, -3.0],
        matches!(
            rhs_dimension,
            LinearSolveError::DimensionMismatch {
                operation: "linear solve rhs",
                expected: 2,
                actual: 1,
            }
        ),
        rhs_dimension.status() == SolverStatus::NumericalError,
        matches!(non_finite, LinearSolveError::NonFiniteInput { .. }),
        non_finite.accuracy_status() == AccuracyStatus::NotChecked,
        matches!(invalid_config, LinearSolveError::InvalidConfig(_)),
        invalid_config.status() == SolverStatus::NumericalError,
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "public solver facade checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("identity_solution_error", 0.0)
        .minimum("checks_passed", checks.len() as f64)
        .input_checksum("analytic/deterministic-fixture:solver-facade-result-contract-v1")
        .reference("identity solve and typed public facade rejection matrix")
        .reference_checksum("analytic/deterministic-fixture:solver-facade-result-contract-v1")
        .dimension("2x2 identity real CSR plus malformed RHS, non-finite RHS, and invalid config")
        .formulation("sonium-solvers solve_real compatibility facade")
        .tolerance("ten result/error/status/accuracy checks pass")
        .seed("solvers-facade-result-contract:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("solver QA result serializes")
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
                    .expect("solver failure result serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
