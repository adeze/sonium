//! Independent CPU reference for checked GMRES continuation checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, GmresCheckpoint, GmresConfig, try_gmres};
use std::time::Instant;

const CASE_ID: &str = "solvers.gmres-checkpoint-reference.v1";
const TOLERANCE: f64 = 1.0e-10;

const MATRIX: [(usize, usize, f64); 7] = [
    (0, 0, 4.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
    (1, 1, 3.0),
    (1, 2, 1.0),
    (2, 1, 1.0),
    (2, 2, 2.0),
];
const FIRST_SOLUTION: [f64; 3] = [1.0, -1.0, 2.0];
const SECOND_SOLUTION: [f64; 3] = [-0.5, 2.0, 1.5];

fn matvec(x: &[f64; 3]) -> [f64; 3] {
    [
        4.0 * x[0] + x[1],
        x[0] + 3.0 * x[1] + x[2],
        x[1] + 2.0 * x[2],
    ]
}

fn relative_residual(x: &[f64], rhs: &[f64; 3]) -> f64 {
    let candidate = [x[0], x[1], x[2]];
    let actual = matvec(&candidate);
    let residual = actual
        .iter()
        .zip(rhs)
        .map(|(actual, rhs)| (actual - rhs).abs())
        .fold(0.0, f64::max);
    residual / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn max_error(actual: &[f64], expected: &[f64; 3]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = CsrMatrix::from_triplets(3, 3, MATRIX.to_vec());
    let first_rhs = matvec(&FIRST_SOLUTION);
    let second_rhs = matvec(&SECOND_SOLUTION);
    let config = GmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: TOLERANCE,
        ..GmresConfig::default()
    };

    let first = try_gmres(&matrix, &first_rhs, &config)
        .map_err(|error| format!("first GMRES solve failed: {error:?}"))?;
    if !first.converged {
        return Err(format!(
            "first GMRES solve did not converge: {:?}",
            first.status
        ));
    }
    let checkpoint = GmresCheckpoint::from_solution(&first)
        .map_err(|error| format!("checkpoint capture failed: {error:?}"))?;
    let serialized = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error:?}"))?;
    let restored = GmresCheckpoint::<f64>::from_checkpoint_bytes(&serialized)
        .map_err(|error| format!("checkpoint restore failed: {error:?}"))?;
    let resumed = restored
        .resume(&matrix, &second_rhs, &config)
        .map_err(|error| format!("checkpoint resume failed: {error:?}"))?;
    if !resumed.converged {
        return Err(format!(
            "resumed GMRES solve did not converge: {:?}",
            resumed.status
        ));
    }

    let solution_error = max_error(&resumed.x, &SECOND_SOLUTION);
    let residual = relative_residual(&resumed.x, &second_rhs);
    let checkpoint_bytes = serialized.len() as f64;
    let checkpoint_roundtrip_error = restored
        .solution()
        .iter()
        .zip(&first.x)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let source_iterations = checkpoint.source_iterations() as f64;
    if solution_error > TOLERANCE || residual > TOLERANCE || checkpoint_roundtrip_error > TOLERANCE
    {
        return Err(format!(
            "checkpoint reference failed: solution_error={solution_error:e}, residual={residual:e}"
        ));
    }

    Ok(vec![
        ("checkpoint_solution_error", solution_error),
        ("checkpoint_independent_residual", residual),
        ("checkpoint_storage_bytes", checkpoint_bytes),
        ("checkpoint_roundtrip_error", checkpoint_roundtrip_error),
        ("checkpoint_source_iterations", source_iterations),
        ("checkpoint_resume_iterations", resumed.iterations as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 3x3 SPD matrix with two independent known RHS solutions")
                .input_checksum("analytic/deterministic-fixture:gmres-checkpoint-reference-v1")
                .reference_checksum("analytic/deterministic-fixture:gmres-checkpoint-reference-v1")
                .dimension("3x3 CSR; two sequential RHS solves")
                .tolerance("solution, round-trip, and independent residual errors <= 1e-10")
                .seed("solvers-gmres-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name.contains("error") || name.contains("residual") {
                    result.maximum(name, TOLERANCE).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic 3x3 SPD matrix with two independent known RHS solutions")
                .input_checksum("analytic/deterministic-fixture:gmres-checkpoint-reference-v1")
                .reference_checksum("analytic/deterministic-fixture:gmres-checkpoint-reference-v1")
                .tolerance("solution, round-trip, and independent residual errors <= 1e-10")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
