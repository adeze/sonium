//! Seeded metamorphic checks for the public sparse-solver contract.
//!
//! These relations do not require an external answer: scaling both `A` and
//! `b`, or applying the same symmetric permutation to both, must preserve the
//! physical solution.  They catch indexing, scaling, and ordering regressions
//! that a single known-answer fixture can miss.

use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{GmresConfig, gmres};

mod common;

const TOLERANCE: f64 = 2.0e-10;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![6.0, -1.0, 0.5, -1.0, 5.0, 1.0, 0.5, 1.0, 4.0],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 80,
        tolerance: 1.0e-12,
        restart: 3,
        convergence_check_interval: 1,
        ..Default::default()
    }
}

fn scaled(matrix: &CsrMatrix<f64>, factor: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        matrix.num_rows,
        matrix.num_cols,
        matrix.row_ptrs.clone(),
        matrix.col_indices.clone(),
        matrix.values.iter().map(|value| value * factor).collect(),
    )
}

fn permuted_symmetric(matrix: &CsrMatrix<f64>, permutation: [usize; 3]) -> CsrMatrix<f64> {
    let mut dense = vec![0.0; 9];
    for row in 0..3 {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            dense[row * 3 + matrix.col_indices[index]] = matrix.values[index];
        }
    }

    let mut values = vec![0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            values[row * 3 + col] = dense[permutation[row] * 3 + permutation[col]];
        }
    }
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        values,
    )
}

fn permute(values: &[f64; 3], permutation: [usize; 3]) -> Vec<f64> {
    permutation.iter().map(|&index| values[index]).collect()
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0),
            "{label}[{index}] mismatch: actual={actual}, expected={expected}"
        );
    }
}

fn assert_true_residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64], label: &str) {
    let ax = matrix
        .try_matvec(x)
        .unwrap_or_else(|error| panic!("{label}: matvec failed: {error}"));
    let residual = ax
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(
        residual / rhs_norm.max(1.0) <= 5.0e-10,
        "{label}: true residual {residual} exceeds tolerance"
    );
}

#[test]
fn gmres_is_equivariant_under_positive_scaling() {
    let matrix = matrix();
    let rhs = vec![2.0, -1.0, 3.0];
    let base = gmres(&matrix, &rhs, &config());
    assert!(base.converged, "base solve did not converge");
    assert_true_residual(&matrix, &base.x, &rhs, "base residual");

    let factor = 7.25;
    let scaled_result = gmres(
        &scaled(&matrix, factor),
        &rhs.iter().map(|v| factor * v).collect::<Vec<_>>(),
        &config(),
    );
    assert!(scaled_result.converged, "scaled solve did not converge");
    assert_true_residual(
        &scaled(&matrix, factor),
        &scaled_result.x,
        &rhs.iter().map(|v| factor * v).collect::<Vec<_>>(),
        "scaled residual",
    );
    assert_close(&scaled_result.x, &base.x, "scale equivariance");
}

#[test]
fn gmres_is_equivariant_under_symmetric_permutation() {
    let matrix = matrix();
    let rhs = [2.0, -1.0, 3.0];
    let base = gmres(&matrix, &rhs, &config());
    assert!(base.converged, "base solve did not converge");
    assert_true_residual(&matrix, &base.x, &rhs, "permutation base residual");

    let permutation = [2, 0, 1];
    let permuted = permuted_symmetric(&matrix, permutation);
    let permuted_rhs = permute(&rhs, permutation);
    let permuted_result = gmres(&permuted, &permuted_rhs, &config());
    assert!(permuted_result.converged, "permuted solve did not converge");
    assert_true_residual(
        &permuted,
        &permuted_result.x,
        &permuted_rhs,
        "permuted residual",
    );
    assert_close(
        &permuted_result.x,
        &permute(&[base.x[0], base.x[1], base.x[2]], permutation),
        "permutation equivariance",
    );
}

#[test]
fn seeded_diagonally_dominant_fixtures_preserve_scaling_relation() {
    for seed in 0..8_u64 {
        let matrix = common::random_diagonally_dominant(0x5EED ^ seed, 4);
        let rhs = vec![1.0, -2.0, 0.5, 3.0];
        let base = gmres(&matrix, &rhs, &config());
        assert!(base.converged, "seed {seed} base solve did not converge");
        assert_true_residual(
            &matrix,
            &base.x,
            &rhs,
            &format!("seed {seed} base residual"),
        );

        let factor = 1.5 + seed as f64;
        let scaled_matrix = scaled(&matrix, factor);
        let scaled_rhs: Vec<_> = rhs.iter().map(|value| factor * value).collect();
        let scaled_result = gmres(&scaled_matrix, &scaled_rhs, &config());
        assert!(
            scaled_result.converged,
            "seed {seed} scaled solve did not converge"
        );
        assert_true_residual(
            &scaled_matrix,
            &scaled_result.x,
            &scaled_rhs,
            &format!("seed {seed} scaled residual"),
        );
        assert_close(&scaled_result.x, &base.x, "seeded scale equivariance");
    }
}
