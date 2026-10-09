//! Independent factor/solve reconstruction and block-RHS reference.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, SparseLuFactorization, factorize_dense,
    factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-reference.v1";

fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn csr(matrix: &[Vec<f64>]) -> CsrMatrix<f64> {
    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(col, &value)| (value != 0.0).then_some((row, col, value)))
        })
        .collect();
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let general = vec![
        vec![4.0, 1.0, 0.5],
        vec![2.0, 5.0, 1.0],
        vec![0.25, 1.5, 3.0],
    ];
    let spd = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let expected = [1.0, -0.5, 2.0];
    let rhs = matvec(&general, &expected);
    let dense = DenseMatrix::from_row_major_vec(3, 3, general.iter().flatten().copied().collect());
    let dense_lu = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .map_err(|error| format!("dense LU setup: {error}"))?;
    let dense_solution = dense_lu.solve(&rhs).map_err(|error| error.to_string())?;

    let general_csr = csr(&general);
    let sparse_lu = factorize_sparse(
        &general_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("sparse LU setup: {error}"))?;
    let sparse_solution = sparse_lu.solve(&rhs).map_err(|error| error.to_string())?;
    let refined = SparseLuFactorization::factorize(&general_csr)
        .map_err(|error| format!("refinement factorization: {error}"))?
        .solve_refined(&general_csr, &rhs, 1.0e-14, 4)
        .map_err(|error| format!("refinement solve: {error}"))?;
    let refinement_error = max_error(&matvec(&general, &refined.x), &rhs);

    let spd_expected = [1.0, -2.0, 0.75];
    let spd_rhs = matvec(&spd, &spd_expected);
    let spd_dense = DenseMatrix::from_row_major_vec(3, 3, spd.iter().flatten().copied().collect());
    let dense_cholesky = factorize_dense(
        &spd_dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .map_err(|error| format!("dense Cholesky setup: {error}"))?;
    let dense_cholesky_solution = dense_cholesky
        .solve(&spd_rhs)
        .map_err(|error| error.to_string())?;
    let sparse_cholesky = factorize_sparse(
        &csr(&spd),
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("sparse Cholesky setup: {error}"))?;
    let sparse_cholesky_solution = sparse_cholesky
        .solve(&spd_rhs)
        .map_err(|error| error.to_string())?;

    let x1 = [-2.0, 0.25, 1.5];
    let rhs1 = matvec(&general, &x1);
    let block_rhs = [rhs[0], rhs1[0], rhs[1], rhs1[1], rhs[2], rhs1[2]];
    let mut block_solution = vec![0.0; 6];
    dense_lu
        .solve_many_into(&block_rhs, &mut block_solution, 2)
        .map_err(|error| format!("dense block solve: {error}"))?;
    let expected_block = [expected[0], x1[0], expected[1], x1[1], expected[2], x1[2]];

    let mut sparse_block_solution = vec![0.0; 6];
    sparse_lu
        .solve_many_into(&block_rhs, &mut sparse_block_solution, 2)
        .map_err(|error| format!("sparse block solve: {error}"))?;

    let overflow_matrix = csr(&[vec![1.0e-15, 0.0], vec![0.0, 2.0]]);
    let overflow_factor = factorize_sparse(
        &overflow_matrix,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("transactional block factorization: {error}"))?;
    let overflow_rhs = [1.0, f64::MAX, 2.0, 0.0];
    let mut transactional_destination = vec![41.0; 4];
    let transactional_before = transactional_destination.clone();
    let transactional_block_failure = if overflow_factor
        .solve_many_into(&overflow_rhs, &mut transactional_destination, 2)
        .is_err()
        && transactional_destination == transactional_before
    {
        1.0
    } else {
        0.0
    };
    if transactional_block_failure < 1.0 {
        return Err("multi-RHS numerical failure published partial destination".into());
    }

    let permutation = [2usize, 0, 1];
    let permuted_spd: Vec<Vec<f64>> = permutation
        .iter()
        .map(|&row| permutation.iter().map(|&col| spd[row][col]).collect())
        .collect();
    let permuted_rhs: Vec<f64> = permutation.iter().map(|&index| spd_rhs[index]).collect();
    let permuted_factor = factorize_sparse(
        &csr(&permuted_spd),
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("permuted sparse Cholesky setup: {error}"))?;
    let permuted_solution = permuted_factor
        .solve(&permuted_rhs)
        .map_err(|error| format!("permuted sparse Cholesky solve: {error}"))?;
    let permutation_error = max_error(
        &permuted_solution,
        &permutation
            .iter()
            .map(|&index| spd_expected[index])
            .collect::<Vec<_>>(),
    );

    let complex_matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(4.0, -1.0),
        ],
    );
    let complex_expected = [Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)];
    let complex_rhs = [
        complex_matrix[(0, 0)] * complex_expected[0] + complex_matrix[(0, 1)] * complex_expected[1],
        complex_matrix[(1, 0)] * complex_expected[0] + complex_matrix[(1, 1)] * complex_expected[1],
    ];
    let complex_factor = factorize_dense(
        &complex_matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .map_err(|error| format!("complex LU setup: {error}"))?;
    let complex_solution = complex_factor
        .solve(&complex_rhs)
        .map_err(|error| error.to_string())?;
    let complex_error = complex_solution
        .iter()
        .zip(complex_expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let complex_residual = [
        complex_matrix[(0, 0)] * complex_solution[0] + complex_matrix[(0, 1)] * complex_solution[1],
        complex_matrix[(1, 0)] * complex_solution[0] + complex_matrix[(1, 1)] * complex_solution[1],
    ];
    let complex_residual_error = complex_residual
        .iter()
        .zip(complex_rhs)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);

    Ok((
        max_error(&dense_solution, &expected),
        max_error(&sparse_solution, &expected),
        max_error(&dense_cholesky_solution, &spd_expected),
        max_error(&sparse_cholesky_solution, &spd_expected),
        max_error(&block_solution, &expected_block),
        complex_error,
        max_error(&matvec(&general, &dense_solution), &rhs),
        max_error(&sparse_block_solution, &expected_block),
        permutation_error,
        complex_residual_error,
        refinement_error,
        transactional_block_failure,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            dense_error,
            sparse_error,
            dense_cholesky_error,
            sparse_cholesky_error,
            block_error,
            complex_error,
            residual_error,
            sparse_block_error,
            permutation_error,
            complex_residual_error,
            refinement_error,
            transactional_block_failure,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("dense_lu_solution_error", dense_error)
                .metric("sparse_lu_solution_error", sparse_error)
                .metric("dense_cholesky_solution_error", dense_cholesky_error)
                .metric("sparse_cholesky_solution_error", sparse_cholesky_error)
                .metric("block_rhs_error", block_error)
                .metric("complex_lu_error", complex_error)
                .metric("complex_lu_residual_error", complex_residual_error)
                .metric("sparse_block_rhs_error", sparse_block_error)
                .metric("symmetric_permutation_error", permutation_error)
                .metric("independent_residual_error", residual_error)
                .metric("refinement_residual_error", refinement_error)
                .metric("transactional_block_failure", transactional_block_failure)
                .maximum("dense_lu_solution_error", 1e-12)
                .maximum("sparse_lu_solution_error", 1e-12)
                .maximum("dense_cholesky_solution_error", 1e-12)
                .maximum("sparse_cholesky_solution_error", 1e-12)
                .maximum("block_rhs_error", 1e-12)
                .maximum("complex_lu_error", 1e-12)
                .maximum("complex_lu_residual_error", 1e-12)
                .maximum("sparse_block_rhs_error", 1e-12)
                .maximum("symmetric_permutation_error", 1e-12)
                .maximum("independent_residual_error", 1e-12)
                .maximum("refinement_residual_error", 1e-12)
                .minimum("transactional_block_failure", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FACTOR/SOLVE REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
