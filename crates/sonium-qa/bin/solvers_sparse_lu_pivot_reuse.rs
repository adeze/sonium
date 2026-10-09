//! Independent CPU reference for sparse-LU pivoting and symbolic reuse.

use sonium_qa::QaResult;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-pivot-reuse.v1";
const TOLERANCE: f64 = 1.0e-10;

const EXPECTED: [f64; 4] = [1.0, -2.0, 0.5, 3.0];

fn base_matrix() -> CsrMatrix<f64> {
    // Zero diagonal entries force a row pivot while preserving a fixed CSR
    // pattern for the later numeric-only update.
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 7, 10],
        vec![1, 3, 0, 2, 0, 1, 2, 1, 2, 3],
        vec![2.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
    )
}

fn updated_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 7, 10],
        vec![1, 3, 0, 2, 0, 1, 2, 1, 2, 3],
        vec![2.0, 1.0, 1.0, 3.0, 4.0, 5.25, 6.0, 1.0, 2.0, 3.5],
    )
}

fn matvec(matrix: &CsrMatrix<f64>, x: &[f64; 4]) -> [f64; 4] {
    let values = matrix.matvec(x);
    [values[0], values[1], values[2], values[3]]
}

fn max_error(actual: &[f64], expected: &[f64; 4]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64; 4]) -> f64 {
    matrix
        .matvec(solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let matrix = base_matrix();
    let rhs = matvec(&matrix, &EXPECTED);

    let simplicial =
        SparseLuFactorization::factorize(&matrix).map_err(|error| error.to_string())?;
    let simplicial_solution = simplicial.solve(&rhs).map_err(|error| error.to_string())?;

    let supernodal =
        SparseLuFactorization::factorize_supernodal(&matrix).map_err(|error| error.to_string())?;
    let supernodal_solution = supernodal.solve(&rhs).map_err(|error| error.to_string())?;

    let updated = updated_matrix();
    let updated_rhs = matvec(&updated, &EXPECTED);
    let symbolic =
        SparseLuFactorization::analyze_supernodal(&matrix).map_err(|error| error.to_string())?;
    let reused =
        SparseLuFactorization::factorize_supernodal_with_symbolic(&updated, &symbolic, 1.0)
            .map_err(|error| error.to_string())?;
    let reused_solution = reused
        .solve(&updated_rhs)
        .map_err(|error| error.to_string())?;

    Ok((
        max_error(&simplicial_solution, &EXPECTED),
        max_error(&supernodal_solution, &EXPECTED),
        max_error(&reused_solution, &EXPECTED),
        residual(&matrix, &simplicial_solution, &rhs)
            .max(residual(&matrix, &supernodal_solution, &rhs))
            .max(residual(&updated, &reused_solution, &updated_rhs)),
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((simplicial_error, supernodal_error, reused_error, max_residual)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference(
                    "independent 4x4 pivoted sparse-LU solution and same-pattern numeric reuse",
                )
                .input_checksum("analytic/deterministic-fixture:sparse-lu-pivot-reuse-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-pivot-reuse-v1")
                .dimension("4x4 CSR; zero diagonal pivoting; simplicial/supernodal; symbolic reuse")
                .tolerance("solution and true residual errors <= 1e-10")
                .metric("simplicial_solution_error", simplicial_error)
                .metric("supernodal_solution_error", supernodal_error)
                .metric("reused_symbolic_solution_error", reused_error)
                .metric("max_true_residual", max_residual)
                .maximum("simplicial_solution_error", TOLERANCE)
                .maximum("supernodal_solution_error", TOLERANCE)
                .maximum("reused_symbolic_solution_error", TOLERANCE)
                .maximum("max_true_residual", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference(
                    "independent 4x4 pivoted sparse-LU solution and same-pattern numeric reuse",
                )
                .input_checksum("analytic/deterministic-fixture:sparse-lu-pivot-reuse-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-pivot-reuse-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
