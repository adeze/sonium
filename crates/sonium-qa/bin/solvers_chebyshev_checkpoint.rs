//! Independent CPU evidence for portable Chebyshev preconditioner checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::preconditioners::{ChebyshevConfig, ChebyshevPreconditioner};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.chebyshev-checkpoint.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:chebyshev-checkpoint-v1";

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0],
    )
}

fn max_error(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let source = ChebyshevPreconditioner::try_from_csr(
        &matrix(),
        ChebyshevConfig {
            degree: 4,
            eigenvalue_ratio: 18.0,
            lambda_max: Some(5.5),
        },
    )
    .map_err(|error| format!("Chebyshev setup failed: {error}"))?;
    let rhs = [1.0, -2.0, 0.5, 3.0];
    let source_forward = source
        .try_apply(&rhs)
        .map_err(|error| format!("source forward failed: {error}"))?;
    let source_transpose = source
        .try_apply_transpose(&rhs)
        .map_err(|error| format!("source transpose failed: {error}"))?;
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error}"))?;
    let restored = ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint restore failed: {error}"))?;
    let restored_forward = restored
        .try_apply(&rhs)
        .map_err(|error| format!("restored forward failed: {error}"))?;
    let restored_transpose = restored
        .try_apply_transpose(&rhs)
        .map_err(|error| format!("restored transpose failed: {error}"))?;
    let forward_error = max_error(&source_forward, &restored_forward);
    let transpose_error = max_error(&source_transpose, &restored_transpose);

    let mut malformed_rejections = 0.0;
    for length in [0, bytes.len() / 2, bytes.len().saturating_sub(1)] {
        if ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bytes[..length]).is_err() {
            malformed_rejections += 1.0;
        }
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    if ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err() {
        malformed_rejections += 1.0;
    }
    let mut wrong_scalar = bytes;
    wrong_scalar[10] = 16;
    if ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&wrong_scalar).is_err() {
        malformed_rejections += 1.0;
    }
    Ok((
        forward_error,
        transpose_error,
        malformed_rejections,
        source.lambda_max(),
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((forward_error, transpose_error, malformed_rejections, lambda_max))
            if forward_error <= 1.0e-12
                && transpose_error <= 1.0e-12
                && malformed_rejections == 5.0
                && lambda_max.is_finite() =>
        {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .metric("forward_action_error", forward_error)
                .metric("transpose_action_error", transpose_error)
                .metric("malformed_rejections", malformed_rejections)
                .metric("lambda_max", lambda_max)
                .maximum("forward_action_error", 1.0e-12)
                .maximum("transpose_action_error", 1.0e-12)
                .minimum("malformed_rejections", 5.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Ok((forward_error, transpose_error, malformed_rejections, lambda_max)) => {
            eprintln!(
                "Chebyshev checkpoint gate failed: forward={forward_error}, transpose={transpose_error}, malformed={malformed_rejections}, lambda_max={lambda_max}"
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("Chebyshev checkpoint gate failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
