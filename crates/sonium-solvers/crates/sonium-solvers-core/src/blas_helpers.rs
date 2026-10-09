//! BLAS-accelerated linear algebra operations on plain slices.
//!
//! Thin wrappers around `ComplexField` BLAS-dispatch methods. The underlying
//! kernels live in `sonium_compute::cpu::blas1` (NEON on AArch64, scalar
//! fallback elsewhere).

use crate::traits::ComplexField;
use num_traits::{Float, Zero};

/// Compute inner product (x, y) = Σ conj(x_i) * y_i
#[inline]
pub fn inner_product<T: ComplexField>(x: &[T], y: &[T]) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "Vector lengths must match for inner product"
    );
    T::vec_dot(x, y)
}

/// Compute unconjugated inner product (x, y) = Σ x_i * y_i
///
/// Used for bi-orthogonal Lanczos polynomials in complex BiCGSTAB to prevent breakdown.
#[inline]
pub fn dot_unconjugated<T: ComplexField>(x: &[T], y: &[T]) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "Vector lengths must match for dot_unconjugated"
    );
    let mut sum = T::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        sum += *xi * *yi;
    }
    sum
}

/// Compute vector 2-norm: ||x||_2 = sqrt(Σ |x_i|^2)
#[inline]
pub fn vector_norm<T: ComplexField>(x: &[T]) -> T::Real
where
    T::Real: Float,
{
    T::vec_norm(x)
}

/// Compute vector norm squared: ||x||_2^2 = Σ |x_i|^2
#[inline]
#[allow(dead_code)]
pub fn vector_norm_sqr<T: ComplexField>(x: &[T]) -> T::Real {
    T::vec_norm_sqr(x)
}

/// Compute `(||x||_2, (x, y))` with no threading.
///
/// Iterative solvers need both a norm (convergence test) and an inner
/// product (recurrence coefficient) every iteration. Running the two
/// bandwidth-bound sweeps under a `rayon::join` saved no time even
/// uncontended and collapsed under oversubscription (per-iteration
/// rendezvous of a many-thread pool). This runs the same two SIMD kernels
/// back to back on the calling thread: bit-identical to
/// `(vector_norm(x), inner_product(x, y))`, with no fork-join overhead.
#[inline]
pub fn norm_dot<T: ComplexField>(x: &[T], y: &[T]) -> (T::Real, T)
where
    T::Real: Float,
{
    (vector_norm(x), inner_product(x, y))
}

/// Scale vector by scalar: y = α * x
#[allow(dead_code)]
#[inline]
pub fn scale_vector<T: ComplexField>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = alpha * *xi;
    }
}

/// Compute axpy: y = α * x + y
#[inline]
pub fn axpy<T: ComplexField>(alpha: T, x: &[T], y: &mut [T]) {
    T::vec_axpy(alpha, x, y);
}

/// Compute the scaled vector addition: z = α * x + β * y
#[allow(dead_code)]
#[inline]
pub fn axpby<T: ComplexField>(alpha: T, x: &[T], beta: T, y: &[T], z: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
        *zi = alpha * *xi + beta * *yi;
    }
}

/// Compute `out = base + alpha * x` and return `||out||_2` in one pass.
#[inline]
pub fn copy_axpy_norm<T: ComplexField>(base: &[T], alpha: T, x: &[T], out: &mut [T]) -> T::Real
where
    T::Real: Float,
{
    debug_assert_eq!(base.len(), x.len());
    debug_assert_eq!(base.len(), out.len());

    let mut norm_sqr = T::Real::zero();
    for ((base_i, x_i), out_i) in base.iter().zip(x.iter()).zip(out.iter_mut()) {
        let value = *base_i + alpha * *x_i;
        *out_i = value;
        norm_sqr += value.norm_sqr();
    }
    norm_sqr.sqrt()
}

/// Compute vector scale in-place: x = α * x
#[allow(dead_code)]
#[inline]
pub fn scale_inplace<T: ComplexField>(x: &mut [T], alpha: T) {
    T::vec_scale(x, alpha);
}

/// Compute the dot product of two real f64 vectors
#[allow(dead_code)]
#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    sonium_compute::cpu::blas1::dot_f64(x, y)
}

/// Compute the dot product of two real f32 vectors
#[allow(dead_code)]
#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    sonium_compute::cpu::blas1::dot_f32(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_inner_product_real() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let y = vec![4.0_f64, 5.0, 6.0];

        let ip = inner_product(&x, &y);
        assert_relative_eq!(ip, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product_complex() {
        let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let ip = inner_product(&x, &y);
        let expected = Complex64::new(70.0, -8.0);
        assert_relative_eq!(ip.re, expected.re, epsilon = 1e-10);
        assert_relative_eq!(ip.im, expected.im, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_norm_real() {
        let x = vec![3.0_f64, 4.0];
        let norm = vector_norm(&x);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_norm_complex() {
        let x = vec![Complex64::new(3.0, 0.0), Complex64::new(0.0, 4.0)];
        let norm = vector_norm(&x);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_norm_zero() {
        let x = vec![0.0_f64, 0.0, 0.0];
        let norm = vector_norm(&x);
        assert_relative_eq!(norm, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_scale_vector() {
        let alpha = 2.0_f64;
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![0.0_f64; 3];

        scale_vector(alpha, &x, &mut y);

        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_axpy() {
        let alpha = 2.0_f64;
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![1.0_f64; 3];

        axpy(alpha, &x, &mut y);

        assert_relative_eq!(y[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_inplace() {
        let mut x = vec![1.0_f64, 2.0, 3.0];
        let alpha = 0.5_f64;

        scale_inplace(&mut x, alpha);

        assert_relative_eq!(x[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_norm_sqr() {
        let x = vec![3.0_f64, 4.0];
        let norm_sqr = vector_norm_sqr(&x);
        assert_relative_eq!(norm_sqr, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_f64() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0_f64, 5.0, 6.0];

        let d = dot_f64(&x, &y);
        assert_relative_eq!(d, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_axpby() {
        let alpha = 2.0_f64;
        let beta = 0.5_f64;
        let x = vec![1.0_f64, 2.0, 3.0];
        let y = vec![4.0_f64, 5.0, 6.0];
        let mut z = vec![0.0_f64; 3];

        axpby(alpha, &x, beta, &y, &mut z);

        assert_relative_eq!(z[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(z[1], 6.5, epsilon = 1e-10);
        assert_relative_eq!(z[2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_copy_axpy_norm() {
        let base = vec![Complex64::new(1.0, 1.0), Complex64::new(3.0, -1.0)];
        let x = vec![Complex64::new(2.0, 0.5), Complex64::new(-1.0, 4.0)];
        let mut out = vec![Complex64::new(0.0, 0.0); 2];

        let norm = copy_axpy_norm(&base, Complex64::new(0.5, 0.0), &x, &mut out);

        assert_relative_eq!(out[0].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(out[0].im, 1.25, epsilon = 1e-10);
        assert_relative_eq!(out[1].re, 2.5, epsilon = 1e-10);
        assert_relative_eq!(out[1].im, 1.0, epsilon = 1e-10);
        assert_relative_eq!(norm, vector_norm(&out), epsilon = 1e-10);
    }

    #[test]
    fn test_norm_dot_matches_separate_calls() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y = vec![5.0_f64, 6.0, 7.0, 8.0];

        let (norm, dot) = norm_dot(&x, &y);

        assert_relative_eq!(norm, vector_norm(&x), epsilon = 1e-12);
        assert_relative_eq!(dot, inner_product(&x, &y), epsilon = 1e-12);
        assert_relative_eq!(norm, 30.0_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(dot, 70.0, epsilon = 1e-12);
    }

    #[test]
    fn test_norm_dot_complex_matches_separate_calls() {
        let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
        let y = vec![Complex64::new(0.5, 1.0), Complex64::new(-2.0, 0.5)];

        let (norm, dot) = norm_dot(&x, &y);
        let expected_dot = inner_product(&x, &y);

        assert_relative_eq!(norm, vector_norm(&x), epsilon = 1e-12);
        assert_relative_eq!(dot.re, expected_dot.re, epsilon = 1e-12);
        assert_relative_eq!(dot.im, expected_dot.im, epsilon = 1e-12);
    }
}
