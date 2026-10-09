//! Independent CPU reference for reusable sparse-LU multi-RHS refinement.

use sonium_qa::QaResult;
use sonium_solvers_direct::{SparseLuFactorization, SparseRefinementStopReason};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-refinement-many.v1";
const TOLERANCE: f64 = 1.0e-10;

const EXPECTED: [[f64; 3]; 3] = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5], [2.0, 0.25, -1.0]];

fn base_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
    )
}

fn updated_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.000001, 1.0, 1.0, 3.000001, 1.0, 1.0, 2.000001],
    )
}

fn matvec(x: &[f64; 3]) -> [f64; 3] {
    [
        4.000001 * x[0] + x[1],
        x[0] + 3.000001 * x[1] + x[2],
        x[1] + 2.000001 * x[2],
    ]
}

fn max_error(actual: &[f64], expected: &[f64; 3]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn independent_residual(actual: &[f64], rhs: &[f64; 3]) -> f64 {
    let x = [actual[0], actual[1], actual[2]];
    let ax = matvec(&x);
    ax.iter()
        .zip(rhs)
        .map(|(actual, rhs)| (actual - rhs).abs())
        .fold(0.0, f64::max)
        / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let factor = SparseLuFactorization::factorize(&base_matrix())
        .map_err(|error| format!("base sparse-LU factorization failed: {error:?}"))?;
    let matrix = updated_matrix();
    let rhs_columns: [[f64; 3]; 3] = EXPECTED.map(|solution| matvec(&solution));
    let rhs = [
        rhs_columns[0][0],
        rhs_columns[1][0],
        rhs_columns[2][0],
        rhs_columns[0][1],
        rhs_columns[1][1],
        rhs_columns[2][1],
        rhs_columns[0][2],
        rhs_columns[1][2],
        rhs_columns[2][2],
    ];
    let results = factor
        .solve_refined_many(&matrix, &rhs, 3, TOLERANCE, 64)
        .map_err(|error| format!("multi-RHS refinement failed: {error:?}"))?;
    if results.len() != EXPECTED.len() {
        return Err(format!(
            "expected 3 refinement results, got {}",
            results.len()
        ));
    }

    let mut max_solution_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    let mut total_iterations: f64 = 0.0;
    for (column, result) in results.iter().enumerate() {
        if !result.converged || result.stop_reason != SparseRefinementStopReason::ToleranceReached {
            return Err(format!("column {column} did not converge: {result:?}"));
        }
        max_solution_error = max_solution_error.max(max_error(&result.x, &EXPECTED[column]));
        max_residual = max_residual.max(independent_residual(&result.x, &rhs_columns[column]));
        total_iterations += result.iterations as f64;
    }
    if max_solution_error > TOLERANCE || max_residual > TOLERANCE {
        return Err(format!(
            "independent gate failed: solution_error={max_solution_error:e}, residual={max_residual:e}"
        ));
    }

    Ok(vec![
        ("max_solution_error", max_solution_error),
        ("max_independent_residual", max_residual),
        ("converged_columns", results.len() as f64),
        ("total_refinement_iterations", total_iterations),
        (
            "structure_fingerprint",
            factor.structure_fingerprint() as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent analytic 3x3 SPD solutions with same-pattern value update")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-refinement-many-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-refinement-many-v1")
                .dimension("3x3 CSR; one factor, same-pattern value update, three row-major RHS")
                .tolerance("solution error and independent residual <= 1e-10")
                .seed("solvers-sparse-lu-refinement-many:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name.contains("error") || name.contains("residual") {
                    result.maximum(name, TOLERANCE).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent analytic 3x3 SPD solutions with same-pattern value update")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-refinement-many-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-refinement-many-v1")
                .tolerance("solution error and independent residual <= 1e-10")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
