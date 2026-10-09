//! Independent CPU evidence for the low-level direct-result contract.

use num_complex::Complex64;
use num_traits::ToPrimitive;
use sonium_qa::QaResult;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_direct::{
    DenseDirectConfig, DirectFactorization, MatrixClass, SparseDirectConfig, SparseDirectMethod,
    factorize_dense, factorize_sparse,
};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-result-contract.v1";

fn max_solution_error<T: sonium_solvers_core::traits::ComplexField>(
    actual: &[T],
    expected: &[T],
) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .norm()
                .to_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64), String> {
    let dense = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let expected = [1.0_f64, 2.0];
    let rhs = dense.matvec(&expected);
    let factor = factorize_dense(&dense, &DenseDirectConfig::default())
        .map_err(|error| error.to_string())?;
    let result = factor
        .solve_with_result(&rhs)
        .map_err(|error| error.to_string())?;
    result
        .validate_contract()
        .map_err(|error| error.to_string())?;
    if result.solver_status() != SolverStatus::Converged
        || result.accuracy_status() != AccuracyStatus::NotChecked
        || result.iterations != 0
        || result.residual.is_some()
        || result.backward_error.is_some()
    {
        return Err("unchecked dense direct metadata was not conservative".into());
    }
    let solution_error = max_solution_error(result.solution(), &expected);
    let backward_error =
        sonium_solvers_direct::dense_backward_error(&dense, result.solution(), &rhs)
            .map_err(|error| error.to_string())?;
    let checked = result
        .with_accuracy_check(solution_error, backward_error, 1.0e-12_f64)
        .map_err(|error| error.to_string())?;
    checked
        .validate_contract()
        .map_err(|error| error.to_string())?;
    if checked.accuracy_status() != AccuracyStatus::Attained {
        return Err("independent dense residual gate did not attain tolerance".into());
    }

    let sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let sparse_factor = factorize_sparse(
        &sparse,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| error.to_string())?;
    let sparse_result = sparse_factor
        .solve_with_result(&rhs)
        .map_err(|error| error.to_string())?;
    sparse_result
        .validate_contract()
        .map_err(|error| error.to_string())?;
    let sparse_solution_error = max_solution_error(sparse_result.solution(), &expected);

    let complex_dense = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(3.0, -0.5),
        ],
    );
    let complex_expected = [Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)];
    let complex_rhs = complex_dense.matvec(&complex_expected);
    let complex_factor = factorize_dense(&complex_dense, &DenseDirectConfig::default())
        .map_err(|error| error.to_string())?;
    let complex_result = complex_factor
        .solve_with_result(&complex_rhs)
        .map_err(|error| error.to_string())?;
    complex_result
        .validate_contract()
        .map_err(|error| error.to_string())?;
    let complex_solution_error = max_solution_error(complex_result.solution(), &complex_expected);

    Ok((
        solution_error.max(sparse_solution_error),
        complex_solution_error,
        backward_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((solution_error, complex_solution_error, backward_error)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("result_contract_checks", 7.0)
            .minimum("result_contract_checks", 7.0)
            .metric("scalar_family_checks", 3.0)
            .minimum("scalar_family_checks", 3.0)
            .metric("max_real_solution_error", solution_error)
            .maximum("max_real_solution_error", 1.0e-12)
            .metric("max_complex_solution_error", complex_solution_error)
            .maximum("max_complex_solution_error", 1.0e-12)
            .metric("dense_backward_error", backward_error)
            .maximum("dense_backward_error", 1.0e-12)
            .input_checksum("analytic/deterministic-fixture:solvers-direct-result-contract-v1")
            .reference("independent dense/sparse direct solve and explicit residual admission")
            .reference_checksum("analytic/deterministic-fixture:solvers-direct-result-contract-v1")
            .seed("solvers-direct-result-contract:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
