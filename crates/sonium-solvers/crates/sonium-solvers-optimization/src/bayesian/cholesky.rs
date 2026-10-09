use sonium_solvers_core::dense::DenseMatrix;

use super::consts::CHOLESKY_ATTEMPTS;
use super::misc::kernel_matrix;
use super::solve::solve_lower;
use super::solve::solve_upper_from_lower;
use crate::util::dense_linalg::cholesky;

pub(super) fn negative_log_marginal_likelihood(
    x: &[Vec<f64>],
    y: &[f64],
    lengthscales: &[f64],
    kernel_variance: f64,
    noise: f64,
) -> Option<f64> {
    if x.is_empty() || x.len() != y.len() {
        return None;
    }
    let n = y.len();
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let variance = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;
    let y_scale = variance.sqrt().max(1e-12);
    let y_norm: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_scale).collect();
    let k = kernel_matrix(x, lengthscales, kernel_variance);
    let (chol_l, _) = cholesky_with_jitter(&k, noise.max(1e-12))?;
    let alpha = cholesky_solve(&chol_l, &y_norm);
    let data_fit = 0.5 * dot(&y_norm, &alpha);
    let complexity = (0..n).map(|i| chol_l[(i, i)].ln()).sum::<f64>();
    let normalizer = 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    let score = data_fit + complexity + normalizer;
    score.is_finite().then_some(score)
}

pub(super) fn cholesky_with_jitter(
    matrix: &DenseMatrix<f64>,
    base_jitter: f64,
) -> Option<(DenseMatrix<f64>, f64)> {
    // `dense_linalg::cholesky` reports non-positive pivots via `None` but lets
    // NaN through (`NaN <= 0.0` is false); the hand-rolled factorisation this
    // replaces also rejected non-finite pivots, so guard the input up front.
    if !matrix.as_slice().iter().all(|v| v.is_finite()) {
        return None;
    }
    for attempt in 0..CHOLESKY_ATTEMPTS {
        let jitter = base_jitter.max(1e-14) * 10f64.powi(attempt as i32);
        let mut shifted = matrix.clone();
        for i in 0..shifted.nrows() {
            shifted[(i, i)] += jitter;
        }
        if let Some(l) = cholesky(&shifted) {
            return Some((l, jitter));
        }
    }
    None
}

pub(super) fn cholesky_solve(l: &DenseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let y = solve_lower(l, b);
    solve_upper_from_lower(l, &y)
}

pub(super) fn sample_multivariate_normal(
    mean: &[f64],
    cov: &DenseMatrix<f64>,
    normals: &[f64],
) -> Vec<f64> {
    let q = mean.len();
    if q == 0 {
        return Vec::new();
    }
    if normals.len() != q || cov.nrows() != q || cov.ncols() != q {
        return mean.to_vec();
    }

    let mut sym = DenseMatrix::zeros(q, q);
    for i in 0..q {
        for j in 0..q {
            sym[(i, j)] = 0.5 * (cov[(i, j)] + cov[(j, i)]);
        }
    }
    if let Some((l, _)) = cholesky_with_jitter(&sym, 1e-12) {
        let correlated = matvec(&l, normals);
        return mean
            .iter()
            .enumerate()
            .map(|(i, m)| m + correlated[i])
            .collect();
    }

    mean.iter()
        .enumerate()
        .map(|(i, m)| m + cov[(i, i)].max(1e-14).sqrt() * normals[i])
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

fn matvec(a: &DenseMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..a.nrows())
        .map(|i| (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum())
        .collect()
}
