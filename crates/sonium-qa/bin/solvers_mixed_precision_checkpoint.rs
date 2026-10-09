//! Independent CPU evidence for mixed-precision FGMRES continuation.

use sonium_qa::QaResult;
use sonium_solvers::FgmresConfig;
use sonium_solvers::mixed_precision::{MixedPrecisionCheckpoint, try_mixed_precision_solve};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
    )
}

fn rhs(matrix: &CsrMatrix<f64>, solution: &[f64; 3]) -> Vec<f64> {
    matrix.matvec(solution)
}

fn residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn max_error(actual: &[f64], expected: &[f64; 3]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let first_expected = [1.0, -2.0, 0.5];
    let second_expected = [-0.5, 1.25, 2.0];
    let first_rhs = rhs(&matrix, &first_expected);
    let second_rhs = rhs(&matrix, &second_expected);
    let config = FgmresConfig {
        max_iterations: 40,
        restart: 3,
        tolerance: 1.0e-12,
        exact_residual_check_interval: 1,
        ..FgmresConfig::default()
    };
    let first = try_mixed_precision_solve(
        &matrix,
        &first_rhs,
        Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>())),
        &config,
    )
    .map_err(|error| format!("first mixed-precision solve failed: {error}"))?;
    let checkpoint = MixedPrecisionCheckpoint::from_solution(&first)
        .map_err(|error| format!("checkpoint capture failed: {error}"))?;
    let resumed = checkpoint
        .resume(
            &matrix,
            &second_rhs,
            Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>())),
            &config,
        )
        .map_err(|error| format!("checkpoint resume failed: {error}"))?;
    let solution_error = max_error(&resumed.x, &second_expected);
    let independent_residual = residual(&matrix, &resumed.x, &second_rhs);
    if !resumed.converged
        || solution_error > TOLERANCE
        || independent_residual > TOLERANCE
        || resumed.iterations < first.iterations
        || resumed.op_counts.spmv_count < first.op_counts.spmv_count
    {
        return Err(format!(
            "continuation failed: converged={}, error={solution_error:.3e}, residual={independent_residual:.3e}, source_iterations={}, resumed_iterations={}",
            resumed.converged, first.iterations, resumed.iterations,
        ));
    }
    Ok(vec![
        ("resumed_solution_error", solution_error),
        ("resumed_independent_residual", independent_residual),
        ("source_iterations", first.iterations as f64),
        ("resumed_cumulative_iterations", resumed.iterations as f64),
        (
            "checkpoint_storage_bytes",
            checkpoint.storage_bytes() as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent analytic 3x3 SPD two-RHS mixed-precision continuation")
                .input_checksum("analytic/deterministic-fixture:mixed-precision-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:mixed-precision-checkpoint-v1")
                .dimension("3x3 CSR; f64 FGMRES with f32 diagonal preconditioner")
                .tolerance("solution error and independent residual <= 1e-10")
                .seed("solvers-mixed-precision-checkpoint:cpu:v1")
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
                .reference("independent analytic 3x3 SPD two-RHS mixed-precision continuation")
                .input_checksum("analytic/deterministic-fixture:mixed-precision-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:mixed-precision-checkpoint-v1")
                .tolerance("solution error and independent residual <= 1e-10")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
