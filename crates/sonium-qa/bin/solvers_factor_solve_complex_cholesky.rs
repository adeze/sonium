//! Independent Complex64 Hermitian sparse-Cholesky factor/solve property case.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-complex-cholesky.v1";
const SEED: &str = "solvers-factor-solve-complex-cholesky:fixture-v1";
const FIXTURE_CHECKSUM: f64 = 28.35;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (0, 3, Complex64::new(0.2, -0.1)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(5.0, 0.0)),
            (1, 2, Complex64::new(0.75, -0.2)),
            (2, 1, Complex64::new(0.75, 0.2)),
            (2, 2, Complex64::new(4.5, 0.0)),
            (2, 3, Complex64::new(1.0, 0.3)),
            (3, 0, Complex64::new(0.2, 0.1)),
            (3, 2, Complex64::new(1.0, -0.3)),
            (3, 3, Complex64::new(3.5, 0.0)),
        ],
    )
}

fn expected() -> Vec<Complex64> {
    vec![
        Complex64::new(0.5, -0.75),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.25, 0.5),
        Complex64::new(0.75, -0.2),
    ]
}

fn fixture_checksum(matrix: &CsrMatrix<Complex64>, expected: &[Complex64]) -> f64 {
    matrix
        .values
        .iter()
        .chain(expected)
        .map(|value| value.re.abs() + 0.5 * value.im.abs())
        .sum()
}

fn run() -> Result<(f64, f64, f64), String> {
    let matrix = matrix();
    let expected = expected();
    let checksum = fixture_checksum(&matrix, &expected);
    if (checksum - FIXTURE_CHECKSUM).abs() > 1.0e-12 {
        return Err(format!(
            "fixture checksum changed: expected {FIXTURE_CHECKSUM}, got {checksum}"
        ));
    }
    let rhs = matrix.matvec(&expected);
    let mut max_solution_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for ordering in [
        CholeskyOrdering::Natural,
        CholeskyOrdering::Colamd,
        CholeskyOrdering::NestedDissection,
    ] {
        let factor = SparseCholeskyFactorization::factorize_with_ordering(&matrix, ordering, true)
            .map_err(|error| format!("Complex64 Cholesky factorization {ordering:?}: {error}"))?;
        let solution = factor
            .solve(&rhs)
            .map_err(|error| format!("Complex64 Cholesky solve {ordering:?}: {error}"))?;
        let solution_error = solution
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        let residual = matrix
            .matvec(&solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        max_solution_error = max_solution_error.max(solution_error);
        max_residual = max_residual.max(residual);
    }
    if max_solution_error > 1.0e-12 || max_residual > 1.0e-12 {
        return Err(format!(
            "Complex64 Cholesky accuracy exceeded tolerance: solution_error={max_solution_error:e}, residual={max_residual:e}"
        ));
    }
    Ok((max_solution_error, max_residual, checksum))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((solution_error, residual, checksum)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum("analytic/deterministic-fixture:factor-solve-complex-cholesky-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:factor-solve-complex-cholesky-v1",
                )
                .reference("independent Complex64 Hermitian sparse-Cholesky ordering oracle")
                .metric("max_solution_error", solution_error)
                .metric("max_true_residual", residual)
                .metric("fixture_checksum", checksum)
                .metric("ordering_count", 3.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .seed(SEED)
                .input_checksum("analytic/deterministic-fixture:factor-solve-complex-cholesky-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:factor-solve-complex-cholesky-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
