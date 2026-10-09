//! Independent CPU reference for duplicate-column SPD validation.

use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, LinearSolveConfig, LinearSolverType, solve_real};
use std::time::Instant;

const CASE_ID: &str = "solvers.spd-duplicate-csr.v1";
const TOLERANCE: f64 = 1.0e-10;

fn duplicate_spd_matrix() -> CsrMatrix<f64> {
    // The effective matrix after CSR duplicate summation is [[2, 1], [1, 2]].
    CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 4, 6],
        vec![0, 0, 1, 1, 0, 1],
        vec![1.0, 1.0, 0.5, 0.5, 1.0, 2.0],
    )
}

fn run() -> Result<(f64, f64, bool), String> {
    let matrix = duplicate_spd_matrix();
    let rhs = [3.0, 3.0];
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        max_iterations: 20,
        tolerance: 1.0e-12,
        ..LinearSolveConfig::default()
    };
    let result = solve_real(&matrix, &rhs, &config)
        .map_err(|error| format!("duplicate-column SPD solve: {error:?}"))?;
    if !result.converged {
        return Err("duplicate-column SPD solve did not converge".into());
    }

    // Independent dense reference for the coalesced operator, avoiding the
    // production CSR matvec implementation in the QA oracle.
    let solution_error = result
        .x
        .iter()
        .zip([1.0, 1.0])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let residual = [
        (2.0 * result.x[0] + result.x[1] - rhs[0]).abs(),
        (result.x[0] + 2.0 * result.x[1] - rhs[1]).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);

    let nonpositive = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 3, 4],
        vec![0, 0, 1, 1],
        vec![2.0, -3.0, 0.0, 1.0],
    );
    let rejected = matches!(
        solve_real(&nonpositive, &[1.0, 1.0], &config),
        Err(sonium_solvers::LinearSolveError::InvalidConfig(_))
    );
    Ok((solution_error, residual, rejected))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((solution_error, residual, rejected))
            if solution_error <= TOLERANCE && residual <= TOLERANCE && rejected =>
        {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense action of coalesced duplicate-column SPD matrix")
                .input_checksum("analytic/deterministic-fixture:spd-duplicate-csr-v1")
                .reference_checksum("analytic/deterministic-fixture:spd-duplicate-csr-v1")
                .dimension("2x2 CSR with duplicate diagonal and off-diagonal columns")
                .tolerance(
                    "solution and independent residual <= 1e-10; non-positive aggregate rejected",
                )
                .metric("duplicate_spd_solution_error", solution_error)
                .metric("duplicate_spd_residual", residual)
                .metric("nonpositive_duplicate_diagonal_rejected", 1.0)
                .maximum("duplicate_spd_solution_error", TOLERANCE)
                .maximum("duplicate_spd_residual", TOLERANCE)
                .minimum("nonpositive_duplicate_diagonal_rejected", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Ok((solution_error, residual, rejected)) => {
            let result = QaResult::failed(
                CASE_ID,
                format!(
                    "duplicate CSR contract outside tolerance: solution_error={solution_error}, residual={residual}, rejected={rejected}"
                ),
            )
            .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
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
