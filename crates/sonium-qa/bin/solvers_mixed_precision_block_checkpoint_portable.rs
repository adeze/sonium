//! Independent CPU evidence for portable mixed-precision block checkpoints.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::{
    MixedPrecisionBlockCheckpoint, try_mixed_precision_block_solve,
};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{BlockFgmresConfig, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-block-checkpoint-portable.v1";
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

fn block_from_columns(columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    DenseMatrix::from_fn(3, 2, |row, col| columns[col][row])
}

fn rhs(matrix: &CsrMatrix<f64>, columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    DenseMatrix::from_fn(3, 2, |row, col| matrix.matvec(&columns[col])[row])
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
    let mut maximum: f64 = 0.0;
    for column in 0..solution.ncols() {
        let x = solution.column_to_vec(column);
        let action = matrix.matvec(&x);
        for (actual, expected) in action.iter().zip(rhs.column_to_vec(column)) {
            maximum = maximum.max((actual - expected).abs());
        }
    }
    maximum
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let columns = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]];
    let initial = block_from_columns(&columns);
    let rhs = rhs(&matrix, &columns);
    let cold = try_mixed_precision_block_solve(&matrix, &rhs, preconditioner(&matrix), &config())
        .map_err(|error| error.to_string())?;
    if !cold.converged {
        return Err("cold block solve did not converge".into());
    }
    let checkpoint =
        MixedPrecisionBlockCheckpoint::from_solution(&cold).map_err(|error| error.to_string())?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored = MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    let round_trip_delta = max_error(restored.solution(), checkpoint.solution());
    let metadata_match = f64::from(
        checkpoint.source_iterations() == restored.source_iterations()
            && checkpoint.source_residuals() == restored.source_residuals()
            && checkpoint.source_true_residuals() == restored.source_true_residuals()
            && checkpoint.source_backward_errors() == restored.source_backward_errors()
            && checkpoint.source_attainable_accuracy() == restored.source_attainable_accuracy()
            && checkpoint.source_op_counts() == restored.source_op_counts()
            && checkpoint.source_converged() == restored.source_converged(),
    );
    let resumed = restored
        .resume(&matrix, &rhs, preconditioner(&matrix), &config())
        .map_err(|error| error.to_string())?;
    let residual = relative_residual(&matrix, &resumed.x, &rhs);
    let cumulative_iterations =
        f64::from(resumed.iterations >= checkpoint.source_iterations().unwrap_or(0));
    let mut malformed = bytes;
    malformed[0] ^= 1;
    let malformed_rejected =
        f64::from(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&malformed).is_err());
    let iterate_round_trip = MixedPrecisionBlockCheckpoint::from_iterate(&initial)
        .map_err(|error| error.to_string())?
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let iterate_restored =
        MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&iterate_round_trip)
            .map_err(|error| error.to_string())?;
    let iterate_only_delta = max_error(&iterate_restored.solution(), &initial);
    Ok(vec![
        ("round_trip_delta", round_trip_delta),
        ("iterate_only_delta", iterate_only_delta),
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
                .reference("analytic deterministic block f64 CSR mixed-precision checkpoint packet")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-portable-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-portable-v1",
                )
                .dimension("3x3 f64 CSR; 2 RHS; f32 diagonal preconditioner")
                .tolerance(
                    "round-trip/iterate delta/residual <= 1e-10; metadata/cumulative/rejection = 1",
                )
                .seed("solvers-mixed-precision-block-checkpoint-portable:cpu:v1")
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
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-portable-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-block-checkpoint-portable-v1",
                );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
