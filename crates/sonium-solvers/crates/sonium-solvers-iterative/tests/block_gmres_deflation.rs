//! Block-GMRES rank-drop deflation with true-residual interval checks.
//!
//! Two identical right-hand sides make the initial residual block exactly
//! rank-deficient, forcing the deflation path (block shrink mid-restart).
//! With `exact_residual_check_interval > 0` and `true_residual_check` on,
//! the projected Arnoldi estimates and the recomputed true residuals must
//! agree that every RHS converged — projected-vs-true divergence must not
//! hide until exit.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_iterative::{BlockGmresConfig, block_gmres};
use sonium_solvers_sparse::CsrMatrix;

fn tridiagonal_spd(dimension: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::new();
    for row in 0..dimension {
        triplets.push((row, row, 2.0));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < dimension {
            triplets.push((row, row + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(dimension, dimension, triplets)
}

#[test]
fn block_gmres_deflates_duplicate_rhs_with_true_residual_gating() {
    let n = 6;
    let operator = tridiagonal_spd(n);
    // Columns 0 and 1 are identical (forces rank-drop deflation); column 2
    // is independent.
    let mut data = Vec::with_capacity(n * 3);
    for row in 0..n {
        let shared = (row + 1) as f64;
        let third = if row % 2 == 0 { 1.0 } else { -0.5 };
        data.extend([shared, shared, third]);
    }
    let rhs = DenseMatrix::from_row_major_vec(n, 3, data);
    let config = BlockGmresConfig {
        max_iterations: 20,
        restart: 4,
        tolerance: 1.0e-10,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let solution = block_gmres(&operator, &rhs, &config);
    assert!(solution.converged, "deflated block solve must converge");
    assert_eq!(solution.true_residuals.len(), 3);

    // Every reported true residual must independently verify: recompute
    // A·x_j − b_j column by column (deflation must not corrupt bookkeeping
    // of which column is which).
    for column in 0..3 {
        let x: Vec<f64> = (0..n).map(|row| solution.x[[row, column]]).collect();
        let b: Vec<f64> = (0..n).map(|row| rhs[[row, column]]).collect();
        let ax = operator.apply(&x);
        let residual = ax
            .iter()
            .zip(&b)
            .map(|(actual, expected)| (expected - actual).powi(2))
            .sum::<f64>()
            .sqrt()
            / b.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
        assert!(
            residual < 1.0e-9,
            "column {column}: independent residual {residual:.3e} too large"
        );
        assert!(
            (residual - solution.true_residuals[column]).abs() < 1.0e-9,
            "column {column}: reported true residual disagrees with recomputation"
        );
        // Projected estimates must not claim convergence the truth denies.
        assert!(
            solution.residuals[column] < 1.0e-6,
            "column {column}: projected residual {:?} diverged from truth",
            solution.residuals[column]
        );
    }

    // Duplicate RHS columns must produce (near-)identical solutions.
    for row in 0..n {
        assert!(
            (solution.x[[row, 0]] - solution.x[[row, 1]]).abs() < 1.0e-9,
            "duplicate RHS diverged at row {row}"
        );
    }
}
