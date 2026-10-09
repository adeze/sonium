//! Independent eigen-solver breakdown/non-convergence packet.

use sonium_qa::QaResult;
use sonium_solvers::eigen::{LobpcgConfig, lobpcg};
use sonium_solvers::{CsrMatrix, SolverResultContract, SolverStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "solvers.eigen-breakdown-matrix.v1";
const CHECKSUM: &str = "deterministic-fixture:eigen-breakdown-matrix-v1";

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 5.0)]);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        lobpcg(
            &operator,
            &LobpcgConfig {
                num_eigenvalues: 1,
                max_iterations: 1,
                tolerance: 1.0e-12,
                block_padding: 0,
            },
        )
    }))
    .map_err(|_| "LOBPCG breakdown path panicked".to_string())?;

    let (failure_status, result_contract) = match outcome {
        Err(_) => (1.0, 1.0),
        Ok(result) => {
            result
                .validate_contract()
                .map_err(|error| format!("LOBPCG result contract: {error}"))?;
            let non_success = f64::from((result.solver_status() != SolverStatus::Converged) as u8);
            (non_success, 1.0)
        }
    };
    if failure_status != 1.0 || result_contract != 1.0 {
        return Err("LOBPCG one-step fixture reported unchecked convergence".into());
    }
    Ok(vec![
        ("one_step_non_success", failure_status),
        ("result_contract_valid", result_contract),
        ("panic_free", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .reference("independent distinct-spectrum one-step LOBPCG failure fixture")
                    .reference_checksum(CHECKSUM)
                    .input_checksum(CHECKSUM)
                    .formulation("LOBPCG max-iteration/non-convergence status contract")
                    .dimension("3x3 diagonal real operator, one eigenpair, one iteration")
                    .tolerance("one-step result is non-converged or typed failure, contract-valid, panic-free")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| result.metric(name, value),
            );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
