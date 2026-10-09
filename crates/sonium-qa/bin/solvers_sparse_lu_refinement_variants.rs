//! Independent CPU evidence for sparse-LU refinement compatibility variants.

use sonium_qa::QaResult;
use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-refinement-variants.v1";
const TOLERANCE: f64 = 1.0e-10;

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

fn expected_columns() -> [[f64; 3]; 2] {
    [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]]
}

fn rhs(matrix: &CsrMatrix<f64>, columns: &[[f64; 3]; 2]) -> Vec<f64> {
    let values = columns.map(|column| matrix.matvec(&column));
    let mut result = Vec::with_capacity(6);
    for row in 0..3 {
        for column in &values {
            result.push(column[row]);
        }
    }
    result
}

fn max_error(actual: &[f64], expected: &[f64; 3]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn max_residual(matrix: &CsrMatrix<f64>, actual: &[f64], expected_rhs: &[f64]) -> f64 {
    matrix
        .matvec(actual)
        .iter()
        .zip(expected_rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn check_factor(
    name: &str,
    factor: &SparseLuFactorization<f64>,
    matrix: &CsrMatrix<f64>,
    columns: &[[f64; 3]; 2],
) -> Result<Vec<(String, f64)>, String> {
    let expected_rhs = columns.map(|column| matrix.matvec(&column));
    let results = factor
        .solve_refined_many(matrix, &rhs(matrix, columns), 2, 1.0e-12, 3)
        .map_err(|error| format!("{name} refinement failed: {error:?}"))?;
    let mut metrics = Vec::new();
    for (index, (result, expected)) in results.iter().zip(columns).enumerate() {
        if !result.converged || result.relative_residual > 1.0e-12 {
            return Err(format!(
                "{name} column {index} did not converge: {:?}",
                result.stop_reason
            ));
        }
        metrics.push((
            format!("{name}_column_{index}_solution_error"),
            max_error(&result.x, expected),
        ));
        metrics.push((
            format!("{name}_column_{index}_residual"),
            max_residual(matrix, &result.x, &expected_rhs[index]),
        ));
    }
    Ok(metrics)
}

fn run() -> Result<Vec<(String, f64)>, String> {
    let matrix = base_matrix();
    let columns = expected_columns();
    let factors = [
        (
            "simplicial",
            SparseLuFactorization::factorize(&matrix)
                .map_err(|error| format!("simplicial factorization failed: {error:?}"))?,
        ),
        (
            "supernodal",
            SparseLuFactorization::factorize_supernodal(&matrix)
                .map_err(|error| format!("supernodal factorization failed: {error:?}"))?,
        ),
        (
            "simplicial_preprocessed",
            SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true)
                .map_err(|error| format!("simplicial preprocessing failed: {error:?}"))?,
        ),
        (
            "supernodal_preprocessed",
            SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
                .map_err(|error| format!("supernodal preprocessing failed: {error:?}"))?,
        ),
    ];
    let mut metrics = Vec::new();
    for (name, factor) in &factors {
        metrics.extend(check_factor(name, factor, &matrix, &columns)?);
        let updated = updated_matrix();
        let updated_rhs = rhs(&updated, &columns);
        let updated_results = factor
            .solve_refined_many(&updated, &updated_rhs, 2, 1.0e-12, 3)
            .map_err(|error| format!("{name} value update failed: {error:?}"))?;
        for (index, result) in updated_results.iter().enumerate() {
            let expected_rhs = [
                updated_rhs[index],
                updated_rhs[index + 2],
                updated_rhs[index + 4],
            ];
            metrics.push((
                format!("{name}_value_update_{index}_residual"),
                max_residual(&updated, &result.x, &expected_rhs),
            ));
        }
    }

    let duplicate = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 5, 7],
        vec![0, 0, 1, 0, 1, 1, 2],
        vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0],
    );
    let unsorted = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![1, 0, 0, 1, 2, 1, 2],
        vec![1.0, 4.0, 1.0, 3.0, 1.0, 1.0, 2.0],
    );
    let mut rejection_checks = 0usize;
    for (_, factor) in &factors {
        for invalid in [&duplicate, &unsorted] {
            if matches!(
                factor.solve_refined(invalid, &[1.0, 2.0, 3.0], 1.0e-12, 1),
                Err(SparseLuError::InvalidCsrStructure(_))
            ) {
                rejection_checks += 1;
            }
            if matches!(
                factor.solve_refined_many(invalid, &[1.0, 2.0, 3.0], 1, 1.0e-12, 1),
                Err(SparseLuError::InvalidCsrStructure(_))
            ) {
                rejection_checks += 1;
            }
        }
    }
    if rejection_checks != 16 {
        return Err(format!(
            "expected 16 duplicate/unsorted rejection checks, got {rejection_checks}"
        ));
    }
    metrics.push((
        "duplicate_unsorted_rejection_checks".into(),
        rejection_checks as f64,
    ));
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-refinement-variants-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-refinement-variants-v1",
                );
            for (name, value) in metrics {
                let bounded = name.contains("solution_error") || name.contains("residual");
                result = result.metric(name.clone(), value);
                if bounded {
                    result = result.maximum(name, TOLERANCE);
                }
            }
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-refinement-variants-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-refinement-variants-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
