//! Independent differential reference for the CPU solver adapters.
//!
//! The reference implementation deliberately does not call any Sonium solver:
//! it is a small partial-pivot Gaussian elimination used to cross-check the
//! dense, sparse, and iterative public paths on the same deterministic system.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{GmresConfig, LinearOperator, SolverResultContract, try_gmres};
use std::time::Instant;

const CASE_ID: &str = "solvers.differential-reference.v1";
const REFERENCE_CHECKSUM: &str = "deterministic-fixture:solver-differential-reference-v1";

#[derive(Clone)]
struct MatrixOperator {
    matrix: Vec<Vec<f64>>,
}

impl LinearOperator<f64> for MatrixOperator {
    fn num_rows(&self) -> usize {
        self.matrix.len()
    }

    fn num_cols(&self) -> usize {
        self.matrix.first().map_or(0, Vec::len)
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        for (row, output) in self.matrix.iter().zip(y.iter_mut()) {
            *output = row.iter().zip(x).map(|(a, x)| a * x).sum();
        }
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for (row, x) in self.matrix.iter().zip(x) {
            for (column, value) in row.iter().enumerate() {
                y[column] += value * x;
            }
        }
    }
}

fn matrix() -> Vec<Vec<f64>> {
    vec![
        vec![7.0, 1.0, 0.5, 0.0],
        vec![2.0, 8.0, 1.0, 0.2],
        vec![0.0, 1.0, 6.0, 1.0],
        vec![0.3, 0.0, 2.0, 9.0],
    ]
}

fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

/// Independent reference solve. This is intentionally kept local to this QA
/// binary so it cannot accidentally share pivoting or stopping code with the
/// solver implementations under test.
fn reference_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) || rhs.len() != n {
        return Err("reference dimensions".into());
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for pivot in 0..n {
        let (pivot_row, pivot_abs) = (pivot..n)
            .map(|row| (row, a[row][pivot].abs()))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| "reference pivot search".to_string())?;
        if !pivot_abs.is_finite() || pivot_abs <= 1.0e-14 {
            return Err(format!("reference singular pivot {pivot}"));
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        for row in (pivot + 1)..n {
            let factor = a[row][pivot] / a[pivot][pivot];
            a[row][pivot] = 0.0;
            for column in (pivot + 1)..n {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = ((row + 1)..n)
            .map(|column| a[row][column] * solution[column])
            .sum();
        solution[row] = (b[row] - tail) / a[row][row];
    }
    Ok(solution)
}

fn max_error(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let expected = [1.0, -2.0, 0.5, 3.0];
    let rhs = matvec(&matrix, &expected);
    let reference = reference_solve(&matrix, &rhs)?;
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
    .map_err(|error| format!("dense factorization: {error}"))?;
    let dense_solution = dense_factor
        .solve(&rhs)
        .map_err(|error| format!("dense solve: {error}"))?;

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
    let sparse = CsrMatrix::from_triplets(matrix.len(), matrix.len(), triplets);
    let sparse_factor = factorize_sparse(
        &sparse,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("sparse factorization: {error}"))?;
    let sparse_solution = sparse_factor
        .solve(&rhs)
        .map_err(|error| format!("sparse solve: {error}"))?;

    let operator = MatrixOperator { matrix };
    let gmres = try_gmres(
        &operator,
        &rhs,
        &GmresConfig {
            restart: 4,
            max_iterations: 40,
            tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .map_err(|error| format!("GMRES: {error}"))?;
    gmres
        .validate_contract()
        .map_err(|error| format!("GMRES result contract: {error}"))?;

    let dense_error = max_error(&dense_solution, &reference);
    let sparse_error = max_error(&sparse_solution, &reference);
    let gmres_error = max_error(gmres.x.as_slice(), &reference);
    let gmres_residual = max_error(&matvec(&operator.matrix, gmres.x.as_slice()), &rhs);
    if !dense_error.is_finite()
        || !sparse_error.is_finite()
        || !gmres_error.is_finite()
        || !gmres_residual.is_finite()
        || dense_error > 1.0e-11
        || sparse_error > 1.0e-11
        || gmres_error > 1.0e-9
        || gmres_residual > 1.0e-10
    {
        return Err(format!(
            "differential mismatch dense={dense_error} sparse={sparse_error} gmres={gmres_error} residual={gmres_residual}"
        ));
    }
    Ok(vec![
        ("dense_reference_error", dense_error),
        ("sparse_reference_error", sparse_error),
        ("gmres_reference_error", gmres_error),
        ("gmres_independent_residual", gmres_residual),
        ("reference_solution_finite", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .reference("independent partial-pivot Gaussian elimination")
                    .reference_checksum(REFERENCE_CHECKSUM)
                    .input_checksum(REFERENCE_CHECKSUM)
                    .formulation("dense LU, sparse LU, and GMRES versus independent reference")
                    .dimension("deterministic 4x4 real nonsymmetric system")
                    .tolerance("dense/sparse reference error <= 1e-11; GMRES error <= 1e-9; residual <= 1e-10")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| result.metric(name, value),
            );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
