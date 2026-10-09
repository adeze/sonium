//! Deterministic matrix-free FEM--BEM trace-operator evidence.

use num_complex::Complex64;
use sonium_bem::{ConservativeTraceTransfer, CoupledTraceOperator, CoupledTraceSystem};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::sparse::{CsrMatrix, DenseBlockOperator};
use std::time::Instant;

const CASE_ID: &str = "bem.fem-bem-trace-operator.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.25), (1, 0.75)], vec![(0, 0.75), (1, 0.25)]],
        vec![1.25, 1.75],
        vec![2.0, 1.0],
    )
    .map_err(|error| error.to_string())?;
    let interior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, 0.5),
            Complex64::new(-0.2, 0.1),
            Complex64::new(0.4, -0.3),
            Complex64::new(1.5, -0.25),
        ],
    );
    let exterior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(0.8, -0.1),
            Complex64::new(0.3, 0.2),
            Complex64::new(-0.1, 0.4),
            Complex64::new(1.2, 0.6),
        ],
    );
    let scale = Complex64::new(0.7, -0.2);
    let dense =
        CoupledTraceSystem::try_new(interior.clone(), exterior.clone(), transfer.clone(), scale)
            .map_err(|error| error.to_string())?;
    let fem = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(2.0, 0.5)),
            (0, 1, Complex64::new(-0.2, 0.1)),
            (1, 0, Complex64::new(0.4, -0.3)),
            (1, 1, Complex64::new(1.5, -0.25)),
        ],
    );
    let operator = CoupledTraceOperator::try_new(
        Box::new(fem),
        Box::new(DenseBlockOperator::new(exterior)),
        transfer,
        scale,
    )
    .map_err(|error| error.to_string())?;
    let state = vec![
        Complex64::new(0.4, -0.6),
        Complex64::new(-0.2, 0.8),
        Complex64::new(0.9, 0.1),
        Complex64::new(-0.7, -0.3),
    ];
    let reference = dense.to_dense_matrix().map_err(|error| error.to_string())?;
    let expected_forward = reference.matvec(&state);
    let expected_hermitian = dense_matrix_hermitian_matvec(&reference, &state);
    let forward_error = max_error(
        &operator
            .try_apply(&state)
            .map_err(|error| error.to_string())?,
        &expected_forward,
    );
    let hermitian_error = max_error(
        &operator
            .try_apply_hermitian(&state)
            .map_err(|error| error.to_string())?,
        &expected_hermitian,
    );
    let transpose_error = max_error(
        &operator
            .try_apply_transpose(&state)
            .map_err(|error| error.to_string())?,
        &dense_matrix_transpose_matvec(&reference, &state),
    );
    if forward_error > 1.0e-12 || transpose_error > 1.0e-12 || hermitian_error > 1.0e-12 {
        return Err(format!(
            "matrix-free parity exceeded tolerance: forward={forward_error:e}, transpose={transpose_error:e}, hermitian={hermitian_error:e}"
        ));
    }
    Ok((forward_error, transpose_error, hermitian_error))
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn dense_matrix_transpose_matvec(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]] * state[row])
                .sum()
        })
        .collect()
}

fn dense_matrix_hermitian_matvec(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
) -> Vec<Complex64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| matrix[[row, column]].conj() * state[row])
                .sum()
        })
        .collect()
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((forward_error, transpose_error, hermitian_error)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("forward_error", forward_error)
            .metric("transpose_error", transpose_error)
            .metric("hermitian_error", hermitian_error)
            .input_checksum("analytic/deterministic-fixture:bem-fem-bem-trace-operator-v1")
            .reference_checksum("dense-coupled-trace-system-v1")
            .dimension("2+2 trace DOFs; sparse FEM diagonal + dense BEM diagonal")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
