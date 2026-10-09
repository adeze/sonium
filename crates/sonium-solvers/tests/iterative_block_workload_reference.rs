//! Independent differential coverage for a sustained restarted block-GMRES workload.
//!
//! This is deliberately separate from the compatibility-CG, scalar-execution,
//! result-contract, and small block-vs-column cases.  The matrix and solution
//! reference are owned by this test: no direct solver or solver residual is
//! used as the oracle.

use sonium_solvers::{
    AccuracyStatus, BlockGmresConfig, GmresConfig, LinearOperator, SolverResultContract,
    SolverStatus, block_gmres, dense::DenseMatrix, try_gmres,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const N: usize = 12;
const RHS_COUNT: usize = 3;
const RESTART: usize = 2;
const MAX_ITERATIONS: usize = 32;
const TOLERANCE: f64 = 1.0e-10;

#[derive(Clone)]
struct DenseFixture {
    matrix: [[f64; N]; N],
    block_calls: Arc<AtomicUsize>,
    single_calls: Arc<AtomicUsize>,
}

impl DenseFixture {
    fn new(matrix: [[f64; N]; N]) -> Self {
        Self {
            matrix,
            block_calls: Arc::new(AtomicUsize::new(0)),
            single_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn apply_matrix(&self, input: &[f64], output: &mut [f64], transpose: bool) {
        for row in 0..N {
            output[row] = (0..N)
                .map(|column| {
                    let value = if transpose {
                        self.matrix[column][row]
                    } else {
                        self.matrix[row][column]
                    };
                    value * input[column]
                })
                .sum();
        }
    }
}

impl LinearOperator<f64> for DenseFixture {
    fn num_rows(&self) -> usize {
        N
    }

    fn num_cols(&self) -> usize {
        N
    }

    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        self.single_calls.fetch_add(1, Ordering::Relaxed);
        self.apply_matrix(input, output, false);
    }

    fn apply_block(&self, input: &DenseMatrix<f64>, output: &mut DenseMatrix<f64>) {
        self.block_calls.fetch_add(1, Ordering::Relaxed);
        for column in 0..input.ncols() {
            for row in 0..N {
                output[[row, column]] = (0..N)
                    .map(|inner| self.matrix[row][inner] * input[[inner, column]])
                    .sum();
            }
        }
    }

    fn apply_transpose_into(&self, input: &[f64], output: &mut [f64]) {
        self.apply_matrix(input, output, true);
    }
}

fn matrix() -> [[f64; N]; N] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if row == column {
                1.4 + 0.12 * row as f64
            } else {
                let distance = row.abs_diff(column) as f64;
                0.045 * (1 + ((row * 5 + column * 3) % 7)) as f64 / (1.0 + distance)
            }
        })
    })
}

fn exact_solution() -> [[f64; RHS_COUNT]; N] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let base = (row + 1) as f64;
            match column {
                0 => 0.08 * base - 0.31,
                1 => -0.05 * base + 0.012 * (row * row) as f64,
                _ => 0.17 - 0.04 * base + 0.003 * (row * row) as f64,
            }
        })
    })
}

fn multiply(matrix: &[[f64; N]; N], solution: &[[f64; RHS_COUNT]; N]) -> [[f64; RHS_COUNT]; N] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..N)
                .map(|inner| matrix[row][inner] * solution[inner][column])
                .sum()
        })
    })
}

fn reference_solve(
    matrix: &[[f64; N]; N],
    rhs: &[[f64; RHS_COUNT]; N],
) -> Result<[[f64; RHS_COUNT]; N], String> {
    let mut augmented = vec![vec![0.0; N + RHS_COUNT]; N];
    for row in 0..N {
        for column in 0..N {
            augmented[row][column] = matrix[row][column];
        }
        for column in 0..RHS_COUNT {
            augmented[row][N + column] = rhs[row][column];
        }
    }

    for pivot in 0..N {
        let pivot_row = (pivot..N)
            .max_by(|&left, &right| {
                augmented[left][pivot]
                    .abs()
                    .partial_cmp(&augmented[right][pivot].abs())
                    .expect("finite reference matrix")
            })
            .ok_or_else(|| "empty reference pivot range".to_string())?;
        if augmented[pivot_row][pivot].abs() < 1.0e-14 {
            return Err(format!("reference matrix is singular at pivot {pivot}"));
        }
        augmented.swap(pivot, pivot_row);

        for row in (pivot + 1)..N {
            let factor = augmented[row][pivot] / augmented[pivot][pivot];
            augmented[row][pivot] = 0.0;
            for column in (pivot + 1)..(N + RHS_COUNT) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }

    let mut solution = [[0.0; RHS_COUNT]; N];
    for row in (0..N).rev() {
        for column in 0..RHS_COUNT {
            let tail: f64 = ((row + 1)..N)
                .map(|inner| augmented[row][inner] * solution[inner][column])
                .sum();
            solution[row][column] = (augmented[row][N + column] - tail) / augmented[row][row];
        }
    }
    Ok(solution)
}

fn relative_residual(matrix: &[[f64; N]; N], solution: &[f64], rhs: &[f64]) -> f64 {
    let residual_squared: f64 = (0..N)
        .map(|row| {
            let actual: f64 = (0..N)
                .map(|column| matrix[row][column] * solution[column])
                .sum();
            (actual - rhs[row]).powi(2)
        })
        .sum();
    let rhs_squared: f64 = rhs.iter().map(|value| value * value).sum();
    residual_squared.sqrt() / rhs_squared.sqrt().max(1.0e-30)
}

fn scalar_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: MAX_ITERATIONS,
        restart: RESTART,
        tolerance: TOLERANCE,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    }
}

pub fn run_case() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let expected = exact_solution();
    let rhs = multiply(&matrix, &expected);
    let reference = reference_solve(&matrix, &rhs)?;

    let mut rhs_values = Vec::with_capacity(N * RHS_COUNT);
    for row in &rhs {
        rhs_values.extend(row);
    }
    let rhs_matrix = DenseMatrix::from_row_major_vec(N, RHS_COUNT, rhs_values);

    let block_operator = DenseFixture::new(matrix);
    let block = block_gmres(
        &block_operator,
        &rhs_matrix,
        &BlockGmresConfig {
            max_iterations: MAX_ITERATIONS,
            restart: RESTART,
            tolerance: TOLERANCE,
            true_residual_check: true,
            exact_residual_check_interval: 1,
        },
    );

    if block.solver_status() != SolverStatus::Converged {
        return Err(format!("block GMRES status: {:?}", block.solver_status()));
    }
    if block.accuracy_status() != AccuracyStatus::Attained {
        return Err(format!(
            "block GMRES accuracy: {:?}",
            block.accuracy_status()
        ));
    }
    if !block.converged
        || block
            .attainable_accuracy
            .iter()
            .any(|status| *status != AccuracyStatus::Attained)
    {
        return Err("block GMRES did not attain every RHS".to_string());
    }
    if block.iterations < 3 || block_operator.block_calls.load(Ordering::Relaxed) < 3 {
        return Err(format!(
            "block workload was not sustained through native block applications: iterations={}, block={}, single={}",
            block.iterations,
            block_operator.block_calls.load(Ordering::Relaxed),
            block_operator.single_calls.load(Ordering::Relaxed),
        ));
    }

    let mut max_block_reference_error: f64 = 0.0;
    let mut max_block_residual: f64 = 0.0;
    let mut max_reported_true_residual: f64 = 0.0;
    let mut max_scalar_reference_error: f64 = 0.0;
    let mut max_scalar_residual: f64 = 0.0;
    let mut total_scalar_applications = 0usize;

    for column in 0..RHS_COUNT {
        let scalar_operator = DenseFixture::new(matrix);
        let scalar_rhs: Vec<_> = (0..N).map(|row| rhs[row][column]).collect();
        let scalar = try_gmres(&scalar_operator, &scalar_rhs, &scalar_config())
            .map_err(|error| format!("scalar GMRES failed for RHS {column}: {error}"))?;
        if scalar.solver_status() != SolverStatus::Converged {
            return Err(format!(
                "scalar GMRES status for RHS {column}: {:?}",
                scalar.solver_status()
            ));
        }
        if scalar.accuracy_status() != AccuracyStatus::Attained {
            return Err(format!(
                "scalar GMRES accuracy for RHS {column}: {:?}",
                scalar.accuracy_status()
            ));
        }

        let mut block_column = vec![0.0; N];
        let mut reference_column = vec![0.0; N];
        for row in 0..N {
            block_column[row] = block.x[[row, column]];
            reference_column[row] = reference[row][column];
        }
        let block_error = block_column
            .iter()
            .zip(&reference_column)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        let scalar_error = scalar
            .x
            .iter()
            .zip(&reference_column)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        max_block_reference_error = max_block_reference_error.max(block_error);
        max_scalar_reference_error = max_scalar_reference_error.max(scalar_error);
        max_block_residual =
            max_block_residual.max(relative_residual(&matrix, &block_column, &scalar_rhs));
        max_scalar_residual =
            max_scalar_residual.max(relative_residual(&matrix, &scalar.x, &scalar_rhs));
        max_reported_true_residual = max_reported_true_residual.max(block.true_residuals[column]);
        total_scalar_applications += scalar_operator.single_calls.load(Ordering::Relaxed);
        if scalar_operator.block_calls.load(Ordering::Relaxed) != 0 {
            return Err("scalar GMRES unexpectedly used block applications".to_string());
        }
    }

    if max_block_reference_error > 5.0e-8
        || max_scalar_reference_error > 5.0e-8
        || max_block_residual > 5.0e-10
        || max_scalar_residual > 5.0e-10
        || max_reported_true_residual > 5.0e-10
    {
        return Err(format!(
            "dense differential thresholds failed: block_error={max_block_reference_error:e}, scalar_error={max_scalar_reference_error:e}, block_residual={max_block_residual:e}, scalar_residual={max_scalar_residual:e}, reported={max_reported_true_residual:e}"
        ));
    }

    Ok(vec![
        ("block_reference_error", max_block_reference_error),
        ("scalar_reference_error", max_scalar_reference_error),
        ("block_residual", max_block_residual),
        ("scalar_residual", max_scalar_residual),
        ("reported_true_residual", max_reported_true_residual),
        ("block_iterations", block.iterations as f64),
        (
            "block_apply_calls",
            block_operator.block_calls.load(Ordering::Relaxed) as f64,
        ),
        (
            "block_single_apply_calls",
            block_operator.single_calls.load(Ordering::Relaxed) as f64,
        ),
        ("scalar_apply_calls", total_scalar_applications as f64),
        ("status_checks", (RHS_COUNT + 1) as f64),
    ])
}

#[test]
fn sustained_block_gmres_matches_dense_reference_and_scalar_columns() {
    let metrics = run_case().expect("sustained block-GMRES differential workload");
    assert!(metrics.iter().any(|(name, _)| *name == "block_apply_calls"));
}
