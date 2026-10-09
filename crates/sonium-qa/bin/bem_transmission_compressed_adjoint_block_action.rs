//! Bounded CPU QA for fused compressed-transmission Hermitian block actions.

use num_complex::Complex64;
use sonium_bem::core::assembly::CompressedTransmissionTbemSystem;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-compressed-adjoint-block-action.v1";
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

fn run() -> Result<(f64, usize, usize), String> {
    let source = source_matrix();
    let compressed = CompressedTransmissionTbemSystem::from_dense_matrix(2, &source, 2, 0.5, 1)
        .map_err(|error| error.to_string())?;
    let mut input = DenseMatrix::zeros(4, 3);
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            input[[row, column]] = Complex64::new(
                -0.17 + 0.08 * row as f64 + 0.13 * column as f64,
                0.29 - 0.04 * row as f64 + 0.06 * column as f64,
            );
        }
    }

    let block_action = compressed
        .apply_hermitian_block_checked(&input)
        .map_err(|error| error.to_string())?;
    let mut maximum_error: f64 = 0.0;
    let mut scale = f64::MIN_POSITIVE;
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected = compressed.apply_hermitian(&vector);
        for row in 0..input.nrows() {
            maximum_error = maximum_error.max((block_action[[row, column]] - expected[row]).norm());
            scale = scale.max(expected[row].norm());
        }
    }
    let relative_error = maximum_error / scale;
    if !relative_error.is_finite() || relative_error > MAX_ERROR {
        return Err(format!(
            "Hermitian block action relative error {relative_error:.3e} exceeds {MAX_ERROR:.3e}"
        ));
    }
    Ok((
        relative_error,
        compressed.compressed_block_count(),
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
            .metric("hermitian_block_action_relative_error", relative_error)
            .metric("compressed_blocks", compressed_blocks as f64)
            .metric("rhs_count", rhs_count as f64)
            .maximum("hermitian_block_action_relative_error", MAX_ERROR)
            .minimum("compressed_blocks", 1.0)
            .minimum("rhs_count", 3.0)
            .input_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-adjoint-block-action-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-compressed-adjoint-block-action-v1",
            )
            .dimension("4x4 two-trace compressed operator three RHS columns")
            .formulation("fused dense/low-rank ACA Hermitian block action")
            .tolerance("Hermitian block action matches independent column actions")
            .seed("bem-transmission-compressed-adjoint-block-action:cpu:v1")
            .elapsed_ms(elapsed_ms()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
        Err(_) => QaResult::failed(CASE_ID, "compressed Hermitian block-action QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
    };
    println!(
        "{}",
        result
            .to_json_line()
            .expect("serialize compressed Hermitian block QA")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        std::process::exit(1);
    }
}
