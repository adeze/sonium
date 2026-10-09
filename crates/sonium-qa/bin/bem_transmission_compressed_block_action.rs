//! Bounded CPU QA for fused compressed-transmission block actions.

use num_complex::Complex64;
use sonium_bem::core::assembly::CompressedTransmissionTbemSystem;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::LinearOperator;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-compressed-block-action.v1";
const MAX_ERROR: f64 = 1.0e-13;

fn source_matrix() -> DenseMatrix<Complex64> {
    let mut matrix = DenseMatrix::zeros(4, 4);
    let values = [
        [
            Complex64::new(2.0, 0.1),
            Complex64::new(-0.4, 0.2),
            Complex64::new(0.5, -0.3),
            Complex64::new(0.25, 0.15),
        ],
        [
            Complex64::new(0.3, -0.2),
            Complex64::new(1.7, -0.1),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.1, -0.25),
        ],
        [
            Complex64::new(-0.1, 0.35),
            Complex64::new(0.2, -0.15),
            Complex64::new(1.4, 0.2),
            Complex64::new(-0.35, 0.1),
        ],
        [
            Complex64::new(0.15, -0.05),
            Complex64::new(-0.25, 0.3),
            Complex64::new(0.45, -0.2),
            Complex64::new(1.9, -0.15),
        ],
    ];
    for row in 0..4 {
        for column in 0..4 {
            matrix[[row, column]] = values[row][column];
        }
    }
    matrix
}

fn run() -> Result<(f64, f64, usize), String> {
    let source = source_matrix();
    let compressed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 1.0e-12, 2)
        .map_err(|error| error.to_string())?;
    let mut input = DenseMatrix::zeros(4, 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                0.2 + 0.11 * row as f64 + 0.07 * column as f64,
                -0.3 + 0.05 * row as f64 - 0.09 * column as f64,
            );
        }
    }

    let mut block_action = DenseMatrix::zeros(4, 3);
    <CompressedTransmissionTbemSystem as LinearOperator<Complex64>>::apply_block(
        &compressed,
        &input,
        &mut block_action,
    );
    let mut max_error: f64 = 0.0;
    let mut scale: f64 = f64::MIN_POSITIVE;
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected = compressed.apply(&vector);
        for row in 0..input.nrows() {
            max_error = max_error.max((block_action[[row, column]] - expected[row]).norm());
            scale = scale.max(expected[row].norm());
        }
    }
    let relative_error = max_error / scale;
    if !relative_error.is_finite() || relative_error > MAX_ERROR {
        return Err(format!("fused block action error {relative_error:e}"));
    }
    Ok((
        relative_error,
        compressed.compressed_block_count() as f64,
        input.ncols(),
    ))
}

fn main() {
    let started = Instant::now();
    let elapsed_ms = || started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok((relative_error, compressed_blocks, rhs_count))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("block_action_relative_error", relative_error)
            .metric("compressed_blocks", compressed_blocks)
            .metric("rhs_count", rhs_count as f64)
            .maximum("block_action_relative_error", MAX_ERROR)
            // This four-DOF fixture may be entirely dense; action parity is
            // the release gate while the block count remains diagnostic.
            .minimum("compressed_blocks", 0.0)
            .minimum("rhs_count", 3.0)
            .input_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-block-action-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-block-action-v1",
            )
            .dimension("4x4 two-trace compressed operator with three RHS columns")
            .formulation("fused dense/low-rank ACA block action")
            .tolerance("fused block action matches independent column actions")
            .seed("bem-transmission-compressed-block-action:cpu:v1")
            .elapsed_ms(elapsed_ms()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
        Err(_) => QaResult::failed(CASE_ID, "compressed block-action QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
    };
    println!(
        "{}",
        result
            .to_json_line()
            .expect("serialize compressed block QA")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        std::process::exit(1);
    }
}
