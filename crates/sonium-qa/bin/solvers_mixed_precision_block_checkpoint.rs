//! Independent CPU evidence for mixed-precision block-FGMRES continuation.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::{
    MixedPrecisionBlockCheckpoint, try_mixed_precision_block_solve,
};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{BlockFgmresConfig, LinearOperator, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-block-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-9;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
    )
}

fn block_from_columns(columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    let mut block = DenseMatrix::from_elem(3, 2, 0.0);
    for column in 0..2 {
        for row in 0..3 {
            block[[row, column]] = columns[column][row];
        }
    }
    block
}

fn rhs(matrix: &CsrMatrix<f64>, columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    let mut block = DenseMatrix::from_elem(3, 2, 0.0);
    for column in 0..2 {
        let values = matrix.matvec(&columns[column]);
        for row in 0..3 {
            block[[row, column]] = values[row];
        }
    }
    block
}

fn preconditioner(matrix: &CsrMatrix<f64>) -> Box<dyn Preconditioner<f32> + Send + Sync> {
    Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>()))
}

fn config() -> BlockFgmresConfig<f64> {
    BlockFgmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: 1.0e-11,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    }
}

fn max_error(actual: &DenseMatrix<f64>, expected: &DenseMatrix<f64>) -> f64 {
    actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn relative_residual(
    matrix: &CsrMatrix<f64>,
    solution: &DenseMatrix<f64>,
    rhs: &DenseMatrix<f64>,
) -> f64 {
    let mut action = DenseMatrix::from_elem(solution.nrows(), solution.ncols(), 0.0);
    matrix.apply_block(solution, &mut action);
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for (actual, expected) in action.as_slice().iter().zip(rhs.as_slice()) {
        residual_squared += (actual - expected).powi(2);
        rhs_squared += expected.powi(2);
    }
    residual_squared.sqrt() / rhs_squared.sqrt().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let first_columns = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]];
    let second_columns = [[2.0, 0.25, -1.0], [1.25, -1.5, 0.75]];
    let first_rhs = rhs(&matrix, &first_columns);
    let second_rhs = rhs(&matrix, &second_columns);
    let expected = block_from_columns(&second_columns);

    let first =
        try_mixed_precision_block_solve(&matrix, &first_rhs, preconditioner(&matrix), &config())
            .map_err(|error| format!("first block solve failed: {error:?}"))?;
    let checkpoint = MixedPrecisionBlockCheckpoint::from_solution(&first)
        .map_err(|error| format!("checkpoint capture failed: {error:?}"))?;
    if checkpoint.source_residuals() != Some(first.residuals.as_slice())
        || checkpoint.source_true_residuals() != Some(first.true_residuals.as_slice())
        || checkpoint.source_backward_errors() != Some(first.backward_errors.as_slice())
        || checkpoint.source_attainable_accuracy() != Some(first.attainable_accuracy.as_slice())
    {
        return Err("block checkpoint did not preserve source residual metadata".into());
    }
    let resumed = checkpoint
        .resume(&matrix, &second_rhs, preconditioner(&matrix), &config())
        .map_err(|error| format!("checkpoint resume failed: {error:?}"))?;
    if !resumed.converged {
        return Err(format!(
            "resumed block solve did not converge: {:?}",
            resumed
        ));
    }
    let solution_error = max_error(&resumed.x, &expected);
    let independent_residual = relative_residual(&matrix, &resumed.x, &second_rhs);
    if solution_error > TOLERANCE || independent_residual > TOLERANCE {
        return Err(format!(
            "block continuation errors exceed tolerance: solution={solution_error:.3e}, residual={independent_residual:.3e}"
        ));
    }
    Ok(vec![
        ("solution_error", solution_error),
        ("independent_residual", independent_residual),
        (
            "source_iterations",
            checkpoint.source_iterations().unwrap_or_default() as f64,
        ),
        (
            "source_true_residual_max",
            checkpoint
                .source_true_residuals()
                .unwrap_or_default()
                .iter()
                .copied()
                .fold(0.0, f64::max),
        ),
        ("resumed_iterations", resumed.iterations as f64),
        ("rhs_count", checkpoint.num_rhs() as f64),
        (
            "source_spmv_count",
            checkpoint.source_op_counts().unwrap_or_default().spmv_count as f64,
        ),
        ("resumed_spmv_count", resumed.op_counts.spmv_count as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent analytic 3x3 SPD block reference")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-v1",
                )
                .dimension("3x3 CSR; f64 block FGMRES with f32 diagonal preconditioner")
                .tolerance("solution error and independent residual <= 1e-9")
                .seed("solvers-mixed-precision-block-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE).metric(name, value);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent analytic 3x3 SPD block reference")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-v1",
                )
                .tolerance("solution error and independent residual <= 1e-9")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
