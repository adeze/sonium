//! Independent CPU evidence for permuted large sparse-LU solves.

use sonium_qa::QaResult;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-permutation-parallel.v1";
const TOLERANCE: f64 = 1.0e-10;
const DIMENSION: usize = 1200;
const PERMUTATION_STRIDE: usize = 37;
const REPEATED_SOLVES: usize = 3;

fn permuted_tridiagonal(n: usize) -> (CsrMatrix<f64>, Vec<usize>) {
    let permutation: Vec<usize> = (0..n)
        .map(|index| (PERMUTATION_STRIDE * index) % n)
        .collect();
    let mut triplets = Vec::with_capacity(3 * n.saturating_sub(1) + 1);
    for row in 0..n {
        let permuted_row = permutation[row];
        triplets.push((permuted_row, permuted_row, 4.0 + 0.001 * row as f64));
        if row > 0 {
            triplets.push((permuted_row, permutation[row - 1], -1.0));
        }
        if row + 1 < n {
            triplets.push((permuted_row, permutation[row + 1], -1.0));
        }
    }
    (CsrMatrix::from_triplets(n, n, triplets), permutation)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64), String> {
    let base_solution: Vec<f64> = (0..DIMENSION)
        .map(|index| {
            let x = index as f64 / (DIMENSION - 1) as f64;
            0.25 + 0.5 * x - 0.2 * (3.0 * x).cos()
        })
        .collect();
    let (matrix, permutation) = permuted_tridiagonal(DIMENSION);
    let mut expected = vec![0.0; DIMENSION];
    for (index, &permuted_index) in permutation.iter().enumerate() {
        expected[permuted_index] = base_solution[index];
    }
    let rhs = matrix.matvec(&expected);
    let simplicial = SparseLuFactorization::factorize(&matrix)
        .map_err(|error| format!("permuted simplicial factorization failed: {error}"))?;
    let supernodal = SparseLuFactorization::factorize_supernodal(&matrix)
        .map_err(|error| format!("permuted supernodal factorization failed: {error}"))?;

    let mut simplicial_error: f64 = 0.0;
    let mut supernodal_error: f64 = 0.0;
    let mut max_residual: f64 = 0.0;
    for _ in 0..REPEATED_SOLVES {
        let solution = simplicial
            .solve(&rhs)
            .map_err(|error| format!("permuted simplicial solve failed: {error}"))?;
        simplicial_error = simplicial_error.max(max_error(&solution, &expected));
        max_residual = max_residual.max(max_error(&matrix.matvec(&solution), &rhs));
    }
    for _ in 0..REPEATED_SOLVES {
        let solution = supernodal
            .solve(&rhs)
            .map_err(|error| format!("permuted supernodal solve failed: {error}"))?;
        supernodal_error = supernodal_error.max(max_error(&solution, &expected));
        max_residual = max_residual.max(max_error(&matrix.matvec(&solution), &rhs));
    }
    if simplicial_error > TOLERANCE || supernodal_error > TOLERANCE || max_residual > TOLERANCE {
        return Err(format!(
            "permuted sparse-LU errors exceed tolerance: simplicial={simplicial_error:.3e}, supernodal={supernodal_error:.3e}, residual={max_residual:.3e}"
        ));
    }
    Ok((simplicial_error, supernodal_error, max_residual))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((simplicial_error, supernodal_error, max_residual)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent analytic permuted tridiagonal solution and residual oracle")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-permutation-parallel-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:sparse-lu-permutation-parallel-v1",
                )
                .dimension("1200x1200 CSR; stride-37 symmetric row/column permutation; simplicial/supernodal level-scheduled solves")
                .tolerance("maximum simplicial/supernodal solution errors and residual <= 1e-10")
                .metric("dimension", DIMENSION as f64)
                .metric("permutation_stride", PERMUTATION_STRIDE as f64)
                .metric("repeated_solves_per_backend", REPEATED_SOLVES as f64)
                .metric("simplicial_max_solution_error", simplicial_error)
                .metric("supernodal_max_solution_error", supernodal_error)
                .metric("max_residual", max_residual)
                .maximum("simplicial_max_solution_error", TOLERANCE)
                .maximum("supernodal_max_solution_error", TOLERANCE)
                .maximum("max_residual", TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent analytic permuted tridiagonal solution and residual oracle")
                .input_checksum("analytic/deterministic-fixture:sparse-lu-permutation-parallel-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:sparse-lu-permutation-parallel-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
