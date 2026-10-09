//! Bounded CPU QA for distributed-owner Hermitian transmission row actions.

use num_complex::Complex64;
use sonium_bem::core::assembly::CompressedTransmissionTbemSystem;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-distributed-hermitian-block-rows.v1";
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
                0.03 + 0.07 * row as f64 - 0.05 * column as f64,
                -0.21 + 0.09 * row as f64 + 0.02 * column as f64,
            );
        }
    }
    let owned_rows = [3, 1, 3];
    let actual = compressed
        .apply_hermitian_rows_block_checked(&input, &owned_rows)
        .map_err(|error| error.to_string())?;
    let mut error: f64 = 0.0;
    let mut scale = f64::MIN_POSITIVE;
    for column in 0..input.ncols() {
        let vector = (0..input.nrows())
            .map(|row| input[[row, column]])
            .collect::<Vec<_>>();
        let expected = compressed.apply_hermitian(&vector);
        for (slot, &row) in owned_rows.iter().enumerate() {
            error = error.max((actual[[slot, column]] - expected[row]).norm());
            scale = scale.max(expected[row].norm());
        }
    }
    let error = error / scale;
    if !error.is_finite() || error > MAX_ERROR {
        return Err(format!("Hermitian row block error is {error:.3e}"));
    }
    Ok((error, owned_rows.len(), input.ncols()))
}

fn main() {
    let started = Instant::now();
    let elapsed_ms = || started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok((error, row_count, rhs_count))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("hermitian_row_block_relative_error", error)
            .metric("owned_row_count", row_count as f64)
            .metric("rhs_count", rhs_count as f64)
            .maximum("hermitian_row_block_relative_error", MAX_ERROR)
            .minimum("owned_row_count", 3.0)
            .minimum("rhs_count", 3.0)
            .input_checksum(
                "analytic/deterministic-fixture:bem-transmission-distributed-hermitian-block-rows-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-distributed-hermitian-block-rows-v1",
            )
            .dimension("4x4 two-trace compressed operator three owned rows three RHS columns")
            .formulation("distributed-owner local fused Hermitian row action")
            .tolerance("Hermitian block row action matches independent column-wise action")
            .seed("bem-transmission-distributed-hermitian-block-rows:cpu:v1")
            .elapsed_ms(elapsed_ms()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
        Err(_) => QaResult::failed(CASE_ID, "Hermitian distributed block-row QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms()),
    };
    println!(
        "{}",
        result
            .to_json_line()
            .expect("serialize Hermitian distributed block-row QA")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        std::process::exit(1);
    }
}
