//! Independent CPU evidence for sparse-LU level-scheduled triangular solves.

use sonium_qa::QaResult;
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-parallel.v1";
const TOLERANCE: f64 = 1.0e-10;

fn tridiagonal(n: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::with_capacity(3 * n.saturating_sub(1) + 1);
    for row in 0..n {
        triplets.push((row, row, 4.0 + 0.001 * row as f64));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let expected: Vec<f64> = (0..1200)
        .map(|index| {
            let x = index as f64 / 1199.0;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();
    let matrix = tridiagonal(expected.len());
    let rhs = matrix.matvec(&expected);
    let factor = SparseLuFactorization::factorize(&matrix)
        .map_err(|error| format!("sparse-LU factorization failed: {error}"))?;

    let mut max_solution_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for _ in 0..3 {
        let solution = factor
            .solve(&rhs)
            .map_err(|error| format!("level-scheduled sparse-LU solve failed: {error}"))?;
        max_solution_error = max_solution_error.max(max_error(&solution, &expected));
        let residual = matrix
            .matvec(&solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        max_residual = max_residual.max(residual);
    }
    if max_solution_error > TOLERANCE || max_residual > TOLERANCE {
        return Err(format!(
            "level-scheduled sparse-LU errors exceed tolerance: solution={max_solution_error:.3e}, residual={max_residual:.3e}"
        ));
    }

    Ok(vec![
        ("dimension", expected.len() as f64),
        ("repeated_solves", 3.0),
        ("max_solution_error", max_solution_error),
        ("max_residual", max_residual),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent analytic 1200-row diagonally perturbed tridiagonal system")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-parallel-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-parallel-v1")
                .dimension("1200x1200 CSR; level-scheduled L/U triangular solve")
                .tolerance("maximum solution error and residual <= 1e-10")
                .seed("solvers-sparse-lu-parallel:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE).metric(name, value);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent analytic 1200-row diagonally perturbed tridiagonal system")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-parallel-v1")
                .reference_checksum("analytic/deterministic-fixture:sparse-lu-parallel-v1")
                .dimension("1200x1200 CSR; level-scheduled L/U triangular solve")
                .tolerance("maximum solution error and residual <= 1e-10")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
