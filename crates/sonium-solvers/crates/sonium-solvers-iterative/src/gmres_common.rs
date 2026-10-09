//! Shared helpers for GMRES-family solvers

use super::cgs_orth::KrylovOpCounts;
use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::traits::{ComplexField, LinearOperator};

#[inline]
pub fn exact_residual_check_due(interval: usize, iteration: usize) -> bool {
    // Public solver results must always be based on the true residual. Keep
    // the interval argument for source compatibility with existing configs;
    // it may still be used for optional progress diagnostics, but never to
    // suppress the correctness check on a success/failure return path.
    interval > 0 && iteration > 0 && iteration.is_multiple_of(interval)
}

pub fn exact_relative_residual<T, A>(
    operator: &A,
    x: &[T],
    b: &[T],
    work: &mut [T],
    reference_norm: T::Real,
    op_counts: &mut KrylovOpCounts,
) -> T::Real
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    operator.apply_into(x, work);
    op_counts.spmv_count += 1;
    op_counts.exact_residual_checks += 1;

    let mut residual = T::Real::zero();
    for k in 0..b.len() {
        let rk = b[k] - work[k];
        residual += rk.norm_sqr();
    }
    op_counts.norm_count += 1;

    // Keep the metric finite even when a zero-RHS compatibility path passes
    // a zero reference norm.  The smallest positive value is only a divide
    // guard; unlike a unit floor it does not distort tiny physical scales.
    residual.sqrt() / reference_norm.max(T::Real::min_positive_value())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Identity;

    impl LinearOperator<f64> for Identity {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            y[0] = x[0];
        }

        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            y[0] = x[0];
        }
    }

    #[test]
    fn exact_residual_zero_reference_is_finite() {
        let mut work = [0.0];
        let mut counts = KrylovOpCounts::default();
        let residual =
            exact_relative_residual(&Identity, &[0.0], &[0.0], &mut work, 0.0, &mut counts);
        assert_eq!(residual, 0.0);
        assert_eq!(counts.exact_residual_checks, 1);
    }
}

/// Compute a Givens rotation that zeroes out `b` in the vector `[a; b]`.
///
/// Returns `(c, s)` both of type `T` (complex-valued for complex fields) such that:
/// ```text
/// [c.conj()  s.conj()] [a]   [r]
/// [-s        c       ] [b] = [0]
/// ```
/// where `r = sqrt(|a|^2 + |b|^2)`.
///
/// This convention works correctly for both real and complex scalar types.
#[inline]
pub fn givens_rotation<T: ComplexField>(a: T, b: T) -> (T, T) {
    let tol = T::Real::from_f64(1e-30).unwrap();
    if b.norm() < tol {
        return (T::one(), T::zero());
    }
    if a.norm() < tol {
        return (T::zero(), T::one());
    }

    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
    let c = a * T::from_real(T::Real::one() / r);
    let s = b * T::from_real(T::Real::one() / r);

    (c, s)
}

/// Solve upper triangular system Hy = g
pub fn solve_upper_triangular<T: ComplexField>(
    h: &sonium_solvers_core::dense::DenseMatrix<T>,
    g: &[T],
    k: usize,
) -> Vec<T> {
    let mut y = vec![T::zero(); k];
    let tol = T::Real::from_f64(1e-30).unwrap();

    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= h[[i, j]] * y[j];
        }
        if h[[i, i]].norm() > tol {
            y[i] = sum * h[[i, i]].inv();
        }
    }

    y
}
