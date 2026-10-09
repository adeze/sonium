//! Independent CPU reference for sparse-LU preprocessing and CSR policy.

use sonium_qa::QaResult;
use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-preprocessing.v1";
const TOLERANCE: f64 = 1.0e-10;
const EXPECTED: [f64; 4] = [1.0, -2.0, 0.5, 3.0];

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 7, 10],
        vec![1, 3, 0, 2, 0, 1, 2, 1, 2, 3],
        vec![2.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
    )
}

fn rhs(matrix: &CsrMatrix<f64>) -> [f64; 4] {
    let values = matrix.matvec(&EXPECTED);
    [values[0], values[1], values[2], values[3]]
}

fn max_error(actual: &[f64]) -> f64 {
    actual
        .iter()
        .zip(EXPECTED)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn max_residual(matrix: &CsrMatrix<f64>, actual: &[f64], expected_rhs: &[f64; 4]) -> f64 {
    matrix
        .matvec(actual)
        .iter()
        .zip(expected_rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, usize, f64), String> {
    let matrix = matrix();
    let expected_rhs = rhs(&matrix);
    let mut worst_error: f64 = 0.0;
    let mut worst_residual: f64 = 0.0;

    for supernodal in [false, true] {
        for equilibrate in [false, true] {
            let factor = if supernodal {
                SparseLuFactorization::factorize_supernodal_with_preprocessing(
                    &matrix,
                    1.0,
                    equilibrate,
                )
            } else {
                SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, equilibrate)
            }
            .map_err(|error| {
                format!("supernodal={supernodal} equilibrate={equilibrate} factorization: {error}")
            })?;
            let solution = factor.solve(&expected_rhs).map_err(|error| {
                format!("supernodal={supernodal} equilibrate={equilibrate} solve: {error}")
            })?;
            worst_error = worst_error.max(max_error(&solution));
            worst_residual = worst_residual.max(max_residual(&matrix, &solution, &expected_rhs));
        }
    }

    let duplicate = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![1.0, 2.0]);
    let unsorted = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![1, 0], vec![1.0, 2.0]);
    let mut rejected = 0;
    for malformed in [&duplicate, &unsorted] {
        match SparseLuFactorization::factorize(malformed) {
            Err(SparseLuError::InvalidCsrStructure(_)) => rejected += 1,
            Err(error) => return Err(format!("unexpected malformed-CSR error: {error}")),
            Ok(_) => return Err("malformed CSR unexpectedly factorized".to_string()),
        }
    }

    // Preconditioner mode intentionally admits pivots below the direct
    // solver's absolute singularity floor. Keep this independent of the
    // ordinary factorization modes above so the admission/solve contract
    // cannot regress unnoticed.
    let tiny_pivot = 1.0e-31_f64;
    let tiny_matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, tiny_pivot)]);
    let tiny_factor = SparseLuFactorization::factorize_for_preconditioner(&tiny_matrix)
        .map_err(|error| format!("tiny preconditioner factorization failed: {error}"))?;
    let tiny_solution = tiny_factor
        .solve(&[1.0])
        .map_err(|error| format!("tiny preconditioner solve failed: {error}"))?;
    let tiny_expected = 1.0 / tiny_pivot;
    let tiny_error = (tiny_solution[0] - tiny_expected).abs() / tiny_expected;

    Ok((worst_error, worst_residual, rejected, tiny_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((worst_error, worst_residual, rejected, tiny_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent 4x4 pivoted sparse-LU oracle across preprocessing modes")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-preprocessing-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-preprocessing-v1")
                .dimension("4x4 CSR plus 1x1 tiny-pivot preconditioner; simplicial/supernodal; equilibration off/on; malformed CSR")
                .tolerance("all solution and true residual errors <= 1e-10; duplicate/unsorted CSR rejected; tiny admitted pivot solves")
                .metric("mode_count", 4.0)
                .metric("worst_solution_error", worst_error)
                .metric("worst_true_residual", worst_residual)
                .metric("malformed_csr_rejections", rejected as f64)
                .metric("tiny_pivot_solution_error", tiny_error)
                .maximum("worst_solution_error", TOLERANCE)
                .maximum("worst_true_residual", TOLERANCE)
                .minimum("malformed_csr_rejections", 2.0)
                .maximum("tiny_pivot_solution_error", 1.0e-12)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent 4x4 pivoted sparse-LU oracle across preprocessing modes")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-preprocessing-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-preprocessing-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
