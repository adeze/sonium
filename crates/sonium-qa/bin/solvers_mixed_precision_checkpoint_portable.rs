//! Independent CPU evidence for portable mixed-precision checkpoint replay.

use sonium_qa::QaResult;
use sonium_solvers::iterative::FgmresConfig;
use sonium_solvers::mixed_precision::{MixedPrecisionCheckpoint, try_mixed_precision_solve};
use sonium_solvers::{CsrMatrix, DiagonalPreconditioner, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-checkpoint-portable.v1";
const TOLERANCE: f64 = 1.0e-11;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ],
    )
}

fn preconditioner(matrix: &CsrMatrix<f64>) -> Box<dyn Preconditioner<f32> + Send + Sync> {
    Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>()))
}

fn relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = matrix.matvec(x);
    ax.iter()
        .zip(b)
        .map(|(actual, expected)| actual - expected)
        .map(f64::abs)
        .fold(0.0, f64::max)
}

fn max_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let rhs = vec![1.0, -2.0, 0.5];
    let config = FgmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-12,
        ..FgmresConfig::default()
    };
    let cold = try_mixed_precision_solve(&matrix, &rhs, preconditioner(&matrix), &config)
        .map_err(|error| error.to_string())?;
    if !cold.converged {
        return Err(format!("cold solve did not converge: {:?}", cold.status));
    }
    let checkpoint =
        MixedPrecisionCheckpoint::from_solution(&cold).map_err(|error| error.to_string())?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored = MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    let round_trip_delta = max_delta(checkpoint.solution(), restored.solution());
    let metadata_match = f64::from(
        checkpoint.source_iterations() == restored.source_iterations()
            && checkpoint.source_restarts() == restored.source_restarts()
            && checkpoint.source_op_counts() == restored.source_op_counts(),
    );
    let resumed = restored
        .resume(&matrix, &rhs, preconditioner(&matrix), &config)
        .map_err(|error| error.to_string())?;
    let residual = relative_residual(&matrix, &resumed.x, &rhs);
    let cumulative_iterations =
        f64::from(resumed.iterations >= checkpoint.source_iterations().unwrap_or(0));
    let mut malformed = bytes;
    malformed[0] ^= 1;
    let malformed_rejected =
        f64::from(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&malformed).is_err());
    Ok(vec![
        ("round_trip_delta", round_trip_delta),
        ("metadata_match", metadata_match),
        ("residual", residual),
        ("cumulative_iterations", cumulative_iterations),
        ("malformed_rejected", malformed_rejected),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic deterministic real CSR mixed-precision checkpoint packet")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-checkpoint-portable-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-checkpoint-portable-v1",
                )
                .dimension("3x3 f64 CSR; f32 diagonal preconditioner")
                .tolerance(
                    "round-trip delta/residual <= 1e-11; metadata and malformed rejection = 1",
                )
                .seed("solvers-mixed-precision-checkpoint-portable:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name == "metadata_match"
                    || name == "cumulative_iterations"
                    || name == "malformed_rejected"
                {
                    result.minimum(name, 1.0).metric(name, value)
                } else {
                    result.maximum(name, TOLERANCE).metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-checkpoint-portable-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-checkpoint-portable-v1",
                );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
