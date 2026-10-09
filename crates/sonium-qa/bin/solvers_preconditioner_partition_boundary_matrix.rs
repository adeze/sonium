//! Checked Schwarz core-partition topology boundary matrix.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::AdditiveSchwarzPreconditioner;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-partition-boundary-matrix.v1";
const CHECKSUM: &str = "deterministic-fixture:schwarz-core-partition-boundary-matrix-v1";

fn rejected_plain(matrix: &CsrMatrix<f64>, partitions: &[Vec<usize>]) -> Result<bool, String> {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        AdditiveSchwarzPreconditioner::try_from_csr_partitions(matrix, partitions, 1)
    }))
    .map_err(|_| "checked Schwarz partition construction panicked".to_string())?;
    Ok(outcome.is_err())
}

fn rejected_oras(matrix: &CsrMatrix<f64>, partitions: &[Vec<usize>]) -> Result<bool, String> {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        AdditiveSchwarzPreconditioner::try_from_csr_partitions_oras(matrix, partitions, 1, 1.0)
    }))
    .map_err(|_| "checked ORAS partition construction panicked".to_string())?;
    Ok(outcome.is_err())
}

fn run() -> Result<QaResult, String> {
    let matrix = CsrMatrix::<f64>::identity(4);
    let malformed = [
        vec![vec![], vec![0, 1, 2, 3]],
        vec![vec![0, 1], vec![2, 4]],
        vec![vec![0, 0], vec![1, 2, 3]],
        vec![vec![0, 1], vec![1, 2, 3]],
        vec![vec![0, 1], vec![2]],
    ];

    let mut plain_rejections = 0.0;
    let mut oras_rejections = 0.0;
    for partitions in &malformed {
        if rejected_plain(&matrix, partitions)? {
            plain_rejections += 1.0;
        }
        if rejected_oras(&matrix, partitions)? {
            oras_rejections += 1.0;
        }
    }

    let valid = vec![vec![1, 0], vec![3, 2]];
    let valid_plain = rejected_plain(&matrix, &valid)?;
    let valid_oras = rejected_oras(&matrix, &valid)?;
    if valid_plain || valid_oras {
        return Err("valid unsorted disjoint partitions were rejected".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .input_checksum(CHECKSUM)
        .reference_checksum(CHECKSUM)
        .reference("independent checked Schwarz partition-topology oracle")
        .formulation(
            "five malformed core-partition fixtures across plain Schwarz and ORAS; one valid unsorted fixture",
        )
        .tolerance(
            "all malformed fixtures reject without panic; valid disjoint coverage remains accepted",
        )
        .metric("malformed_partition_cases", malformed.len() as f64)
        .metric("plain_rejections", plain_rejections)
        .metric("oras_rejections", oras_rejections)
        .metric("valid_plain_acceptance", 1.0)
        .metric("valid_oras_acceptance", 1.0)
        .minimum("plain_rejections", malformed.len() as f64)
        .minimum("oras_rejections", malformed.len() as f64)
        .minimum("valid_plain_acceptance", 1.0)
        .minimum("valid_oras_acceptance", 1.0))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(result) => {
            let result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
