//! Independent CPU evidence for portable sparse-Cholesky factor checkpoints.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyError, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-cholesky-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn real_fixture() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 0.75),
            (2, 1, 0.75),
            (2, 2, 3.0),
        ],
    )
}

fn hermitian_fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(3.0, 0.0)),
        ],
    )
}

fn max_real_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn max_complex_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn real_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn complex_residual(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(String, f64)>, String> {
    let real_matrix = real_fixture();
    let real_expected = [1.0, -2.0, 0.5];
    let real_rhs = real_matrix.matvec(&real_expected);
    let mut metrics = Vec::new();

    for (name, factor) in [
        (
            "simplicial",
            SparseCholeskyFactorization::factorize_with_ordering(
                &real_matrix,
                CholeskyOrdering::Natural,
                false,
            )
            .map_err(|error| error.to_string())?,
        ),
        (
            "simplicial_colamd",
            SparseCholeskyFactorization::factorize_with_ordering(
                &real_matrix,
                CholeskyOrdering::Colamd,
                false,
            )
            .map_err(|error| error.to_string())?,
        ),
        (
            "simplicial_nested_dissection",
            SparseCholeskyFactorization::factorize_with_ordering(
                &real_matrix,
                CholeskyOrdering::NestedDissection,
                false,
            )
            .map_err(|error| error.to_string())?,
        ),
        (
            "supernodal",
            SparseCholeskyFactorization::factorize_supernodal(&real_matrix)
                .map_err(|error| error.to_string())?,
        ),
    ] {
        let bytes = factor
            .to_checkpoint_bytes()
            .map_err(|error| error.to_string())?;
        let restored = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(&bytes)
            .map_err(|error| error.to_string())?;
        let solution = restored
            .solve(&real_rhs)
            .map_err(|error| error.to_string())?;
        metrics.push((
            format!("{name}_solution_error"),
            max_real_error(&solution, &real_expected),
        ));
        metrics.push((
            format!("{name}_independent_residual"),
            real_residual(&real_matrix, &solution, &real_rhs),
        ));
        metrics.push((format!("{name}_checkpoint_bytes"), bytes.len() as f64));
    }

    let complex_matrix = hermitian_fixture();
    let complex_expected = [Complex64::new(1.0, 0.25), Complex64::new(-0.5, 1.0)];
    let complex_rhs = complex_matrix.matvec(&complex_expected);
    let factor = SparseCholeskyFactorization::factorize_hermitian(&complex_matrix)
        .map_err(|error| error.to_string())?;
    let bytes = factor
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored = SparseCholeskyFactorization::<Complex64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    let solution = restored
        .solve(&complex_rhs)
        .map_err(|error| error.to_string())?;
    metrics.push((
        "complex_hermitian_solution_error".into(),
        max_complex_error(&solution, &complex_expected),
    ));
    metrics.push((
        "complex_hermitian_independent_residual".into(),
        complex_residual(&complex_matrix, &solution, &complex_rhs),
    ));
    metrics.push((
        "complex_hermitian_checkpoint_bytes".into(),
        bytes.len() as f64,
    ));

    let mut corrupted = bytes;
    corrupted[0] ^= 1;
    let rejected = matches!(
        SparseCholeskyFactorization::<Complex64>::from_checkpoint_bytes(&corrupted),
        Err(SparseCholeskyError::InvalidCheckpoint(_))
    );
    if !rejected {
        return Err("corrupted checkpoint was accepted".into());
    }
    metrics.push(("corruption_rejection_checks".into(), 1.0));
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-cholesky-checkpoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-cholesky-checkpoint-v1",
                );
            for (name, value) in metrics {
                let bounded =
                    name.ends_with("solution_error") || name.ends_with("independent_residual");
                result = result.metric(name.clone(), value);
                if bounded {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-cholesky-checkpoint-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-cholesky-checkpoint-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
