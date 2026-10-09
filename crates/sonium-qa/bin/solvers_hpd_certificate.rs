//! Independent CPU reference for the scalable complex-HPD certificate path.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    AssumptionProvenance, CsrMatrix, LinearSolveConfig, LinearSolveError, LinearSolverType,
    LinearSystemAssumption, solve_with_assumption,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.hpd-scalable-certificate.v1";
const SEED: &str = "solvers-hpd-scalable-certificate:fixture-v1";
const SIZE: usize = 1_025;

fn scaled_hpd_matrix() -> CsrMatrix<Complex64> {
    let mut row_ptrs = Vec::with_capacity(SIZE + 1);
    let mut col_indices = Vec::with_capacity(3 * SIZE - 2);
    let mut values = Vec::with_capacity(3 * SIZE - 2);
    row_ptrs.push(0);
    for row in 0..SIZE {
        let row_scale = if row.is_multiple_of(2) { 0.5 } else { 2.0 };
        if row > 0 {
            let column_scale = if (row - 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row - 1);
            values.push(Complex64::new(1.0, -0.5) / (row_scale * column_scale));
        }
        col_indices.push(row);
        values.push(Complex64::new(3.0 / (row_scale * row_scale), 0.0));
        if row + 1 < SIZE {
            let column_scale = if (row + 1).is_multiple_of(2) {
                0.5
            } else {
                2.0
            };
            col_indices.push(row + 1);
            values.push(Complex64::new(1.0, 0.5) / (row_scale * column_scale));
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(SIZE, SIZE, row_ptrs, col_indices, values)
}

fn max_residual(matrix: &CsrMatrix<Complex64>, solution: &[Complex64], rhs: &[Complex64]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, bool), String> {
    let matrix = scaled_hpd_matrix();
    let expected = vec![Complex64::new(1.0, -0.5); SIZE];
    let rhs = matrix.matvec(&expected);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 256,
        tolerance: 1.0e-10,
        ..Default::default()
    };
    let result = solve_with_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSystemAssumption::PositiveDefiniteCertified,
    )
    .map_err(|error| format!("complex HPD solve: {error:?}"))?;
    if !result.converged
        || result.assumption_provenance != AssumptionProvenance::ScalableCertificate
    {
        return Err(format!(
            "complex HPD solve did not converge with scalable certificate: converged={}, provenance={:?}",
            result.converged, result.assumption_provenance
        ));
    }
    let residual = max_residual(&matrix, &result.x, &rhs);
    let error = result
        .x
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);

    let mut non_hermitian = matrix.clone();
    let lower_first = non_hermitian.row_ptrs[1];
    non_hermitian.values[lower_first] += Complex64::new(0.0, 0.2);
    let rejected = matches!(
        solve_with_assumption(
            &non_hermitian,
            &rhs,
            &config,
            LinearSystemAssumption::PositiveDefiniteCertified,
        ),
        Err(LinearSolveError::InvalidConfig(_))
    );
    if !rejected {
        return Err("non-Hermitian large matrix was not rejected".into());
    }
    Ok((residual, error, rejected))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((residual, error, rejected)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum("analytic/deterministic-fixture:hpd-scalable-certificate-v1")
                .reference_checksum("analytic/deterministic-fixture:hpd-scalable-certificate-v1")
                .reference("independent complex Hermitian positive-definite CG certificate")
                .metric("hpd_max_true_residual", residual)
                .metric("hpd_solution_error", error)
                .metric("non_hermitian_rejected", if rejected { 1.0 } else { 0.0 })
                .maximum("hpd_max_true_residual", 1.0e-7)
                .maximum("hpd_solution_error", 1.0e-7)
                .minimum("non_hermitian_rejected", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum("analytic/deterministic-fixture:hpd-scalable-certificate-v1")
                .reference_checksum("analytic/deterministic-fixture:hpd-scalable-certificate-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
