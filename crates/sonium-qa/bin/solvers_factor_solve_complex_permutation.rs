//! Independent complex sparse-LU factor/solve and symmetric-permutation reference.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-complex-permutation.v1";
const SEED: &str = "solvers-factor-solve-complex-permutation:fixture-v1";
const FIXTURE_CHECKSUM: f64 = 197.75;

fn matrix() -> Vec<Vec<Complex64>> {
    vec![
        vec![
            Complex64::new(5.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.1),
        ],
        vec![
            Complex64::new(0.25, 0.2),
            Complex64::new(4.5, -0.75),
            Complex64::new(1.0, 0.3),
            Complex64::new(0.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, -0.2),
            Complex64::new(3.5, 0.4),
            Complex64::new(0.75, -0.1),
        ],
        vec![
            Complex64::new(0.1, -0.3),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.25, 0.15),
            Complex64::new(4.0, 0.6),
        ],
    ]
}

fn expected() -> Vec<Complex64> {
    vec![
        Complex64::new(0.5, -0.75),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.25, 0.5),
        Complex64::new(0.75, -0.2),
    ]
}

fn matvec(matrix: &[Vec<Complex64>], x: &[Complex64]) -> Vec<Complex64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| *a * *b).sum::<Complex64>())
        .collect()
}

fn csr(matrix: &[Vec<Complex64>]) -> CsrMatrix<Complex64> {
    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| {
                    (value != Complex64::new(0.0, 0.0)).then_some((row, column, value))
                })
        })
        .collect();
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn max_solution_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn max_residual(matrix: &[Vec<Complex64>], solution: &[Complex64], rhs: &[Complex64]) -> f64 {
    matvec(matrix, solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn fixture_checksum(matrix: &[Vec<Complex64>], expected: &[Complex64]) -> f64 {
    matrix
        .iter()
        .flatten()
        .chain(expected.iter())
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * (value.re + 0.5 * value.im))
        .sum()
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64), String> {
    let matrix = matrix();
    let expected = expected();
    let rhs = matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(
        matrix.len(),
        matrix.len(),
        matrix.iter().flatten().copied().collect(),
    );
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .map_err(|error| format!("dense setup: {error:?}"))?;
    let sparse_factor = factorize_sparse(
        &csr(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("sparse setup: {error:?}"))?;
    let dense_solution = dense_factor
        .solve(&rhs)
        .map_err(|error| format!("dense solve: {error:?}"))?;
    let sparse_solution = sparse_factor
        .solve(&rhs)
        .map_err(|error| format!("sparse solve: {error:?}"))?;

    let permutation = [2usize, 0, 3, 1];
    let permuted_matrix: Vec<Vec<_>> = permutation
        .iter()
        .map(|&row| {
            permutation
                .iter()
                .map(|&column| matrix[row][column])
                .collect()
        })
        .collect();
    let permuted_rhs: Vec<_> = permutation.iter().map(|&row| rhs[row]).collect();
    let permuted_expected: Vec<_> = permutation.iter().map(|&row| expected[row]).collect();
    let permuted_factor = factorize_sparse(
        &csr(&permuted_matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("permuted sparse setup: {error:?}"))?;
    let permuted_solution = permuted_factor
        .solve(&permuted_rhs)
        .map_err(|error| format!("permuted sparse solve: {error:?}"))?;

    let checksum = fixture_checksum(&matrix, &expected);
    if (checksum - FIXTURE_CHECKSUM).abs() > 1.0e-12 {
        return Err(format!(
            "fixture checksum changed: expected {FIXTURE_CHECKSUM}, got {checksum}"
        ));
    }

    Ok((
        max_solution_error(&dense_solution, &expected),
        max_solution_error(&sparse_solution, &expected),
        max_solution_error(&permuted_solution, &permuted_expected),
        max_residual(&matrix, &dense_solution, &rhs),
        max_residual(&matrix, &sparse_solution, &rhs),
        max_residual(&permuted_matrix, &permuted_solution, &permuted_rhs),
        checksum,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            dense_error,
            sparse_error,
            permuted_error,
            dense_residual,
            sparse_residual,
            permuted_residual,
            checksum,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .metric("dense_solution_error", dense_error)
                .metric("sparse_solution_error", sparse_error)
                .metric("permuted_sparse_solution_error", permuted_error)
                .metric("dense_residual", dense_residual)
                .metric("sparse_residual", sparse_residual)
                .metric("permuted_sparse_residual", permuted_residual)
                .metric("fixture_checksum", checksum)
                .maximum("dense_solution_error", 1.0e-11)
                .maximum("sparse_solution_error", 1.0e-11)
                .maximum("permuted_sparse_solution_error", 1.0e-11)
                .maximum("dense_residual", 1.0e-11)
                .maximum("sparse_residual", 1.0e-11)
                .maximum("permuted_sparse_residual", 1.0e-11)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("COMPLEX FACTOR/SOLVE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
