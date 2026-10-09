//! Independent CPU evidence for sparse-LU transpose/Hermitian solves.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-adjoint.v1";
const TOLERANCE: f64 = 1.0e-10;

type Entry = (usize, usize, Complex64);

fn fixture() -> (CsrMatrix<Complex64>, Vec<Entry>, Vec<Complex64>) {
    let entries = vec![
        (0, 0, Complex64::new(4.0, 1.0)),
        (0, 1, Complex64::new(1.0, -2.0)),
        (0, 2, Complex64::new(0.5, 0.25)),
        (0, 3, Complex64::new(-0.25, 0.5)),
        (1, 0, Complex64::new(2.0, 0.5)),
        (1, 1, Complex64::new(5.0, -1.0)),
        (1, 2, Complex64::new(-1.0, 0.75)),
        (1, 3, Complex64::new(0.25, -0.25)),
        (2, 0, Complex64::new(0.25, -0.5)),
        (2, 1, Complex64::new(-1.0, 2.0)),
        (2, 2, Complex64::new(3.0, 0.25)),
        (2, 3, Complex64::new(1.25, 0.0)),
        (3, 0, Complex64::new(0.75, 0.25)),
        (3, 1, Complex64::new(0.5, -0.75)),
        (3, 2, Complex64::new(-0.5, 0.5)),
        (3, 3, Complex64::new(4.0, -0.5)),
    ];
    let matrix = CsrMatrix::from_triplets(
        4,
        4,
        entries.iter().map(|&(i, j, value)| (i, j, value)).collect(),
    );
    let expected = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.75, 2.0),
        Complex64::new(-0.25, 0.5),
    ];
    (matrix, entries, expected)
}

fn adjoint_rhs(entries: &[Entry], x: &[Complex64], hermitian: bool) -> Vec<Complex64> {
    let mut rhs = vec![Complex64::new(0.0, 0.0); x.len()];
    for &(row, col, value) in entries {
        rhs[col] += if hermitian { value.conj() } else { value } * x[row];
    }
    rhs
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn residual(entries: &[Entry], x: &[Complex64], rhs: &[Complex64], hermitian: bool) -> f64 {
    max_error(rhs, &adjoint_rhs(entries, x, hermitian))
}

fn run() -> Result<(f64, f64, f64), String> {
    let (matrix, entries, expected) = fixture();
    let transpose_rhs = adjoint_rhs(&entries, &expected, false);
    let hermitian_rhs = adjoint_rhs(&entries, &expected, true);
    let factors = [
        SparseLuFactorization::factorize(&matrix).map_err(|e| e.to_string())?,
        SparseLuFactorization::factorize_supernodal(&matrix).map_err(|e| e.to_string())?,
        SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true)
            .map_err(|e| e.to_string())?,
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
            .map_err(|e| e.to_string())?,
    ];

    let mut max_solution_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for factor in &factors {
        let transpose = factor
            .solve_transpose(&transpose_rhs)
            .map_err(|e| e.to_string())?;
        let hermitian = factor
            .solve_hermitian(&hermitian_rhs)
            .map_err(|e| e.to_string())?;
        max_solution_error = max_solution_error
            .max(max_error(&transpose, &expected))
            .max(max_error(&hermitian, &expected));
        max_residual = max_residual
            .max(residual(&entries, &transpose, &transpose_rhs, false))
            .max(residual(&entries, &hermitian, &hermitian_rhs, true));
    }

    if max_solution_error > TOLERANCE || max_residual > TOLERANCE {
        return Err(format!(
            "adjoint solve error exceeded tolerance: solution={max_solution_error:e}, residual={max_residual:e}"
        ));
    }
    Ok((max_solution_error, max_residual, factors.len() as f64))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((max_solution_error, max_residual, factor_classes)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:sparse-lu-adjoint-v1")
                .reference("independent complex transpose/Hermitian residual oracle")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-adjoint-v1")
                .dimension("4x4 Complex64; simplicial/supernodal with and without equilibration")
                .tolerance("maximum solution error and independent adjoint residual <= 1e-10")
                .metric("factor_classes", factor_classes)
                .metric("max_solution_error", max_solution_error)
                .metric("max_residual", max_residual)
                .maximum("max_solution_error", TOLERANCE)
                .maximum("max_residual", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent complex transpose/Hermitian residual oracle")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-adjoint-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-adjoint-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
