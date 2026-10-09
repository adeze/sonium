//! Seeded dense/sparse factor and block-RHS reconstruction evidence.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-seeded-matrix.v1";

fn matrix(seed: u64) -> Vec<Vec<f64>> {
    let n = 5;
    let mut state = seed;
    let mut matrix = vec![vec![0.0; n]; n];
    for row in 0..n {
        let mut row_sum = 0.0;
        for column in 0..n {
            if row == column {
                continue;
            }
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5;
            matrix[row][column] = value;
            row_sum += value.abs();
        }
        matrix[row][row] = 2.0 + row_sum;
    }
    matrix
}

fn csr(matrix: &[Vec<f64>]) -> CsrMatrix<f64> {
    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| (value != 0.0).then_some((row, column, value)))
        })
        .collect();
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64), String> {
    let mut dense_error = 0.0_f64;
    let mut sparse_error = 0.0_f64;
    let mut block_error = 0.0_f64;
    for seed in 0..8_u64 {
        let matrix = matrix(0xFAC7_0A ^ seed);
        let expected: Vec<_> = (0..matrix.len())
            .map(|index| (index as f64 + 1.0) * (1.0 - 0.1 * seed as f64))
            .collect();
        let rhs = matvec(&matrix, &expected);
        let dense = DenseMatrix::from_row_major_vec(
            matrix.len(),
            matrix.len(),
            matrix.iter().flatten().copied().collect(),
        );
        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .map_err(|error| format!("seed {seed} dense factorization: {error}"))?;
        let sparse_factor = factorize_sparse(
            &csr(&matrix),
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .map_err(|error| format!("seed {seed} sparse factorization: {error}"))?;
        let dense_solution = dense_factor
            .solve(&rhs)
            .map_err(|error| format!("seed {seed} dense solve: {error}"))?;
        let sparse_solution = sparse_factor
            .solve(&rhs)
            .map_err(|error| format!("seed {seed} sparse solve: {error}"))?;
        dense_error = dense_error.max(max_error(&dense_solution, &expected));
        sparse_error = sparse_error.max(max_error(&sparse_solution, &expected));

        let second_expected: Vec<_> = expected.iter().map(|value| -0.5 * value).collect();
        let second_rhs = matvec(&matrix, &second_expected);
        let block_rhs: Vec<_> = (0..matrix.len())
            .flat_map(|row| [rhs[row], second_rhs[row]])
            .collect();
        let mut block_solution = vec![f64::NAN; block_rhs.len()];
        dense_factor
            .solve_many_into(&block_rhs, &mut block_solution, 2)
            .map_err(|error| format!("seed {seed} block solve: {error}"))?;
        let expected_block: Vec<_> = (0..matrix.len())
            .flat_map(|row| [expected[row], second_expected[row]])
            .collect();
        block_error = block_error.max(max_error(&block_solution, &expected_block));
    }
    Ok((dense_error, sparse_error, block_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((dense_error, sparse_error, block_error)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("seed_count", 8.0)
                .metric("dense_solution_error", dense_error)
                .metric("sparse_solution_error", sparse_error)
                .metric("block_solution_error", block_error)
                .maximum("dense_solution_error", 1.0e-10)
                .maximum("sparse_solution_error", 1.0e-10)
                .maximum("block_solution_error", 1.0e-10)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("SEEDED FACTOR/SOLVE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
