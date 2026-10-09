//! Independent CPU evidence for the sparse Hermitian Cholesky adjoint contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-cholesky-adjoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn fixture() -> (CsrMatrix<Complex64>, Vec<Complex64>) {
    let matrix = CsrMatrix::from_triplets(
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
    );
    let expected = vec![
        Complex64::new(0.5, -0.75),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.25, 0.5),
        Complex64::new(0.75, -0.2),
    ];
    (matrix, expected)
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn max_real_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64), String> {
    let (matrix, expected) = fixture();
    let rhs = matrix.matvec(&expected);
    let factors = [
        SparseCholeskyFactorization::factorize_hermitian(&matrix)
            .map_err(|error| error.to_string())?,
        SparseCholeskyFactorization::factorize_supernodal_hermitian(&matrix)
            .map_err(|error| error.to_string())?,
        SparseCholeskyFactorization::factorize_with_ordering(
            &matrix,
            CholeskyOrdering::NestedDissection,
            true,
        )
        .map_err(|error| error.to_string())?,
    ];
    let mut max_solution_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for factor in &factors {
        let solution = factor
            .try_apply_hermitian(&rhs)
            .map_err(|error| error.to_string())?;
        max_solution_error = max_solution_error.max(max_error(&solution, &expected));
        max_residual = max_residual.max(max_error(&matrix.matvec(&solution), &rhs));
    }
    // Real SPD factors are simultaneously symmetric and Hermitian.  Exercise
    // both checked adjoint paths independently; this catches capability
    // regressions that a complex-HPD-only fixture cannot see.
    let real_matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0_f64),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 2, 2.0),
        ],
    );
    let real_expected = vec![1.0, -0.5, 2.0];
    let real_rhs = real_matrix.matvec(&real_expected);
    let real_factor =
        SparseCholeskyFactorization::factorize(&real_matrix).map_err(|error| error.to_string())?;
    if real_factor.transpose_capability() != sonium_solvers_core::traits::AdjointCapability::Exact
        || real_factor.hermitian_capability()
            != sonium_solvers_core::traits::AdjointCapability::Exact
    {
        return Err("real SPD Cholesky did not advertise exact adjoint capabilities".into());
    }
    let real_transpose = real_factor
        .try_apply_transpose(&real_rhs)
        .map_err(|error| error.to_string())?;
    let real_hermitian = real_factor
        .try_apply_hermitian(&real_rhs)
        .map_err(|error| error.to_string())?;
    let real_adjoint_error = max_real_error(&real_transpose, &real_expected)
        .max(max_real_error(&real_hermitian, &real_expected));
    if max_solution_error > TOLERANCE || max_residual > TOLERANCE || real_adjoint_error > TOLERANCE
    {
        return Err(format!(
            "Cholesky adjoint error exceeded tolerance: solution={max_solution_error:e}, residual={max_residual:e}, real_adjoint={real_adjoint_error:e}"
        ));
    }
    Ok((max_solution_error, max_residual, real_adjoint_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((max_solution_error, max_residual, real_adjoint_error)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:sparse-cholesky-adjoint-v1")
                .reference("independent Complex64 HPD matvec residual oracle")
                .reference_checksum("analytic/deterministic-fixture:sparse-cholesky-adjoint-v1")
                .dimension(
                    "4x4 Complex64 HPD plus 3x3 real SPD CSR; simplicial/supernodal/order variants",
                )
                .tolerance("maximum Hermitian/transpose solve errors and residual <= 1e-10")
                .metric("max_solution_error", max_solution_error)
                .metric("max_residual", max_residual)
                .metric("real_adjoint_error", real_adjoint_error)
                .maximum("max_solution_error", TOLERANCE)
                .maximum("max_residual", TOLERANCE)
                .maximum("real_adjoint_error", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent Complex64 HPD matvec residual oracle")
                .input_checksum("analytic/deterministic-fixture:sparse-cholesky-adjoint-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-cholesky-adjoint-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
