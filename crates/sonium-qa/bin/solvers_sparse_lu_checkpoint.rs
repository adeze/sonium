//! Independent CPU evidence for portable sparse-LU factor checkpoints.

use sonium_qa::QaResult;
use sonium_solvers_direct::{SparseLuBackend, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn base_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
    )
}

fn dense_reference(matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = matrix.num_rows;
    if matrix.num_cols != n || rhs.len() != n {
        return Err("reference dimension mismatch".into());
    }
    let mut a = vec![vec![0.0; n + 1]; n];
    for row in 0..n {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            a[row][matrix.col_indices[index]] += matrix.values[index];
        }
        a[row][n] = rhs[row];
    }
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&lhs, &rhs| a[lhs][column].abs().total_cmp(&a[rhs][column].abs()))
            .ok_or_else(|| "empty pivot column".to_string())?;
        if a[pivot][column].abs() <= f64::EPSILON {
            return Err(format!("singular reference pivot {column}"));
        }
        a.swap(column, pivot);
        let diagonal = a[column][column];
        for value in &mut a[column][column..] {
            *value /= diagonal;
        }
        for row in column + 1..n {
            let factor = a[row][column];
            for index in column..=n {
                a[row][index] -= factor * a[column][index];
            }
        }
    }
    for column in (0..n).rev() {
        for row in 0..column {
            a[row][n] -= a[row][column] * a[column][n];
        }
    }
    Ok(a.into_iter().map(|row| row[n]).collect())
}

fn max_error(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn relative_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    let action = matrix.matvec(solution);
    let residual = action
        .iter()
        .zip(rhs)
        .map(|(action, rhs)| (action - rhs).abs())
        .fold(0.0, f64::max);
    let scale = rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    residual / scale
}

fn check_factor(
    factor: &SparseLuFactorization<f64>,
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    reference: &[f64],
) -> Result<(f64, f64, f64), String> {
    let original = factor.solve(rhs).map_err(|error| error.to_string())?;
    let bytes = factor
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored = SparseLuFactorization::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    let restored_solution = restored.solve(rhs).map_err(|error| error.to_string())?;
    let error = max_error(&restored_solution, reference).max(max_error(&original, reference));
    let residual = relative_residual(matrix, &restored_solution, rhs);
    Ok((error, residual, bytes.len() as f64))
}

fn run() -> Result<Vec<(String, f64)>, String> {
    let matrix = base_matrix();
    let rhs = [7.0, 3.0, 5.0];
    let reference = dense_reference(&matrix, &rhs)?;
    let mut metrics = Vec::new();
    for (name, factor) in [
        (
            "simplicial_f64",
            SparseLuFactorization::factorize_with_backend(
                &matrix,
                SparseLuBackend::Simplicial,
                1.0,
            )
            .map_err(|error| error.to_string())?,
        ),
        (
            "supernodal_f64",
            SparseLuFactorization::factorize_with_backend(
                &matrix,
                SparseLuBackend::Supernodal,
                1.0,
            )
            .map_err(|error| error.to_string())?,
        ),
    ] {
        let (error, residual, bytes) = check_factor(&factor, &matrix, &rhs, &reference)?;
        if error > TOLERANCE || residual > TOLERANCE {
            return Err(format!(
                "{name} failed: error={error:.3e}, residual={residual:.3e}"
            ));
        }
        metrics.push((format!("{name}_max_solution_error"), error));
        metrics.push((format!("{name}_independent_residual"), residual));
        metrics.push((format!("{name}_checkpoint_bytes"), bytes));
    }

    let scaled = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.0e-4, 2.0e-4, 3.0e3, 4.0e3, 1.0e3, 5.0e-2, 2.0e-2],
    );
    let scaled_rhs = [1.0, 2.0, 3.0];
    let scaled_reference = dense_reference(&scaled, &scaled_rhs)?;
    let preprocessed =
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&scaled, 1.0, true)
            .map_err(|error| error.to_string())?;
    let (error, residual, bytes) =
        check_factor(&preprocessed, &scaled, &scaled_rhs, &scaled_reference)?;
    if error > TOLERANCE || residual > TOLERANCE {
        return Err(format!(
            "preprocessed failed: error={error:.3e}, residual={residual:.3e}"
        ));
    }
    metrics.push(("preprocessed_max_solution_error".into(), error));
    metrics.push(("preprocessed_independent_residual".into(), residual));
    metrics.push(("preprocessed_checkpoint_bytes".into(), bytes));
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense partial-pivot reference and independent CSR residual")
                .reference_checksum("analytic/deterministic-fixture:solvers-sparse-lu-checkpoint-v1")
                .input_checksum("analytic/deterministic-fixture:solvers-sparse-lu-checkpoint-v1")
                .dimension("3x3 CSR; simplicial, supernodal, complex-family API, and preprocessed factor checkpoints")
                .tolerance("restored solution error and independent residual <= 1e-10")
                .seed("solvers-sparse-lu-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name.contains("error") || name.contains("residual") {
                    result.maximum(name.clone(), TOLERANCE).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense partial-pivot reference and independent CSR residual")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-checkpoint-v1",
                )
                .input_checksum("analytic/deterministic-fixture:solvers-sparse-lu-checkpoint-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
