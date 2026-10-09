//! Numerical differentiation utilities.
//!
//! Forward and central difference gradient approximation and Hessian-vector
//! products via gradient differencing. Used for testing and as a fallback
//! when the user does not supply analytical derivatives.

use crate::types::{OptimError, RealScalar};
use rayon::prelude::*;

/// Minimum coordinate count before a `*_parallel` finite-difference helper
/// fans evaluations out over rayon. One stencil evaluation is usually far
/// costlier than thread dispatch, but trivial objectives (a few flops) never
/// amortize it — tune per machine via `SONIUM_FD_PAR_COORDS`. Below the
/// floor the `*_parallel` entry points run the serial algorithm, so results
/// are identical either way.
pub const PAR_FD_THRESHOLD: usize = 8;

fn par_fd_threshold() -> usize {
    std::env::var("SONIUM_FD_PAR_COORDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(PAR_FD_THRESHOLD)
}

fn numerical_error(message: &str) -> OptimError {
    OptimError::NumericalError {
        iteration: 0,
        message: message.into(),
    }
}

fn validate_jacobian_inputs<T: RealScalar>(x: &[T], m: usize, h: T) -> Result<usize, OptimError> {
    if !h.is_finite() || h <= T::zero() {
        return Err(numerical_error(
            "finite-difference step must be positive and finite",
        ));
    }
    let entries = m
        .checked_mul(x.len())
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<T>())
        .ok_or_else(|| numerical_error("Jacobian storage size overflows"))?;
    // Reject invalid stencils before invoking a potentially expensive callback.
    for &xi in x {
        central_step(xi, h)?;
    }
    Ok(entries)
}

fn central_step<T: RealScalar>(xi: T, h: T) -> Result<T, OptimError> {
    let hi = h * T::max(T::one(), xi.abs());
    let plus = xi + hi;
    let minus = xi - hi;
    let two = T::one() + T::one();
    if !xi.is_finite()
        || !hi.is_finite()
        || hi <= T::zero()
        || !(two * hi).is_finite()
        || !plus.is_finite()
        || !minus.is_finite()
        || plus == xi
        || minus == xi
    {
        return Err(numerical_error(
            "finite-difference stencil is nonfinite or unrepresentable",
        ));
    }
    Ok(hi)
}

fn central_column<T: RealScalar>(fp: &[T], fm: &[T], m: usize, h: T) -> Result<Vec<T>, OptimError> {
    for values in [fp, fm] {
        if values.len() != m {
            return Err(OptimError::DimensionMismatch {
                expected: m,
                got: values.len(),
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(numerical_error(
                "finite-difference callback returned a nonfinite value",
            ));
        }
    }
    let two = T::one() + T::one();
    fp.iter()
        .zip(fm)
        .map(|(&p, &m)| {
            let value = (p - m) / (two * h);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(numerical_error("finite-difference derivative is nonfinite"))
            }
        })
        .collect()
}

/// Compute the gradient of `f` at `x` using forward differences.
///
/// `h` is the *base* step (typically `sqrt(machine_epsilon)`); the effective
/// per-coordinate step is `hi = h * max(1, |x_i|)`, scaled inside. Do not
/// pre-scale `h` by `max(1, |x_i|)` yourself — that would double-scale.
pub fn gradient_forward<T, F>(f: &F, x: &[T], h: T) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let n = x.len();
    let f0 = f(x);
    let mut grad = vec![T::zero(); n];
    let mut x_pert = x.to_vec();

    for i in 0..n {
        let xi_orig = x_pert[i];
        let hi = h * T::max(T::one(), xi_orig.abs());
        x_pert[i] = xi_orig + hi;
        grad[i] = (f(&x_pert) - f0) / hi;
        x_pert[i] = xi_orig;
    }
    grad
}

/// Compute the gradient of `f` at `x` using central differences.
///
/// More accurate than forward differences (O(h^2) vs O(h)) but costs 2n evaluations.
pub fn gradient_central<T, F>(f: &F, x: &[T], h: T) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let n = x.len();
    let mut grad = vec![T::zero(); n];
    let mut x_pert = x.to_vec();
    let two = T::from_f64(2.0).unwrap();

    for i in 0..n {
        let xi_orig = x_pert[i];
        let hi = h * T::max(T::one(), xi_orig.abs());
        x_pert[i] = xi_orig + hi;
        let fp = f(&x_pert);
        x_pert[i] = xi_orig - hi;
        let fm = f(&x_pert);
        grad[i] = (fp - fm) / (two * hi);
        x_pert[i] = xi_orig;
    }
    grad
}

/// Compute the Jacobian of a vector-valued function `f: R^n -> R^m` at `x`
/// using central differences. Row `k` of the result is the gradient of `f_k`.
///
/// Costs `2n` evaluations of `f`. Returns [`OptimError::DimensionMismatch`]
/// if `f` does not return exactly `m` components, instead of silently
/// dropping constraints. Invalid/nonrepresentable steps, nonfinite coordinates,
/// callback values, derivatives, or overflowing storage return a numerical error.
pub fn try_jacobian_central<T, F>(
    f: &F,
    x: &[T],
    m: usize,
    h: T,
) -> Result<sonium_solvers_core::dense::DenseMatrix<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> Vec<T>,
{
    let n = x.len();
    let entries = validate_jacobian_inputs(x, m, h)?;
    let mut data = vec![T::zero(); entries];
    let mut x_pert = x.to_vec();

    for j in 0..n {
        let xj_orig = x_pert[j];
        let hj = central_step(xj_orig, h)?;
        x_pert[j] = xj_orig + hj;
        let fp = f(&x_pert);
        x_pert[j] = xj_orig - hj;
        let fm = f(&x_pert);
        x_pert[j] = xj_orig;
        for (k, value) in central_column(&fp, &fm, m, hj)?.into_iter().enumerate() {
            data[k * n + j] = value;
        }
    }
    Ok(sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
        m, n, data,
    ))
}

/// Compute the Jacobian of a vector-valued function `f: R^n -> R^m` at `x`
/// using central differences.
///
/// Panicking wrapper around [`try_jacobian_central`]: fails loudly if `f`
/// does not return exactly `m` components. Prefer `try_jacobian_central`
/// in fallible pipelines.
pub fn jacobian_central<T, F>(
    f: &F,
    x: &[T],
    m: usize,
    h: T,
) -> sonium_solvers_core::dense::DenseMatrix<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> Vec<T>,
{
    try_jacobian_central(f, x, m, h).expect("invalid central-difference Jacobian")
}

/// Compute the Jacobian using central differences with automatic step size.
pub fn jacobian_central_auto<T, F>(
    f: &F,
    x: &[T],
    m: usize,
) -> sonium_solvers_core::dense::DenseMatrix<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> Vec<T>,
{
    let h = T::machine_epsilon().powf(T::from_f64(1.0 / 3.0).unwrap());
    jacobian_central(f, x, m, h)
}

/// Compute the Hessian-vector product `H(x) * v` via gradient differencing.
///
/// `H(x) * v ≈ (∇f(x + h*v) - ∇f(x)) / h`
///
/// Requires the user-provided gradient function.
pub fn hessian_vec_product<T, G>(grad_fn: &G, x: &[T], v: &[T], h: T) -> Vec<T>
where
    T: RealScalar,
    G: Fn(&[T]) -> Vec<T>,
{
    let v_norm = T::vec_norm(v);
    if v_norm < T::machine_epsilon() {
        return vec![T::zero(); x.len()];
    }
    // Normalise v so the perturbation is of order h regardless of ||v||
    let v_unit = v.iter().map(|&vi| vi / v_norm).collect::<Vec<_>>();
    let g0 = grad_fn(x);
    let perturbation = v_unit.iter().map(|&vi| vi * h).collect::<Vec<_>>();
    let x_pert = sonium_solvers_core::vec_ops::vec_add(x, &perturbation);
    let g1 = grad_fn(&x_pert);
    sonium_solvers_core::vec_ops::vec_mapv(&sonium_solvers_core::vec_ops::vec_sub(&g1, &g0), |gi| {
        gi * v_norm / h
    })
}

/// Compute the gradient using forward differences with automatic step size.
pub fn gradient_auto<T, F>(f: &F, x: &[T]) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    gradient_forward(f, x, T::sqrt_epsilon())
}

/// Compute the gradient using central differences with automatic step size.
pub fn gradient_central_auto<T, F>(f: &F, x: &[T]) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
{
    let h = T::machine_epsilon().powf(T::from_f64(1.0 / 3.0).unwrap());
    gradient_central(f, x, h)
}

/// Compute the gradient of `f` at `x` using forward differences, evaluating
/// stencil points in parallel when `x.len()` reaches the
/// [`PAR_FD_THRESHOLD`] floor (overridable via `SONIUM_FD_PAR_COORDS`).
///
/// Requires `F: Sync` so evaluations can run on rayon worker threads; a
/// panic inside `f` propagates through the join. Results are bit-identical
/// to [`gradient_forward`] (each coordinate is computed the same way).
pub fn gradient_forward_parallel<T, F>(f: &F, x: &[T], h: T) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + Sync,
{
    let n = x.len();
    if n < par_fd_threshold() {
        return gradient_forward(f, x, h);
    }
    let f0 = f(x);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut x_pert = x.to_vec();
            let hi = h * T::max(T::one(), x_pert[i].abs());
            x_pert[i] += hi;
            (f(&x_pert) - f0) / hi
        })
        .collect()
}

/// Parallel central-difference gradient (see [`gradient_forward_parallel`]
/// for the threshold, determinism, and panic contract).
pub fn gradient_central_parallel<T, F>(f: &F, x: &[T], h: T) -> Vec<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T + Sync,
{
    let n = x.len();
    if n < par_fd_threshold() {
        return gradient_central(f, x, h);
    }
    let two = T::from_f64(2.0).unwrap();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut x_pert = x.to_vec();
            let hi = h * T::max(T::one(), x_pert[i].abs());
            let xi_orig = x_pert[i];
            x_pert[i] = xi_orig + hi;
            let fp = f(&x_pert);
            x_pert[i] = xi_orig - hi;
            let fm = f(&x_pert);
            (fp - fm) / (two * hi)
        })
        .collect()
}

/// Parallel central-difference Jacobian (see [`gradient_forward_parallel`]
/// for the threshold, determinism, and panic contract). Shape violations
/// propagate as [`OptimError::DimensionMismatch`], matching
/// [`try_jacobian_central`].
pub fn try_jacobian_central_parallel<T, F>(
    f: &F,
    x: &[T],
    m: usize,
    h: T,
) -> Result<sonium_solvers_core::dense::DenseMatrix<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> Vec<T> + Sync,
{
    let n = x.len();
    if n < par_fd_threshold() {
        return try_jacobian_central(f, x, m, h);
    }
    let entries = validate_jacobian_inputs(x, m, h)?;
    let columns: Result<Vec<Vec<T>>, OptimError> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut x_pert = x.to_vec();
            let hj = central_step(x_pert[j], h)?;
            let xj_orig = x_pert[j];
            x_pert[j] = xj_orig + hj;
            let fp = f(&x_pert);
            x_pert[j] = xj_orig - hj;
            let fm = f(&x_pert);
            central_column(&fp, &fm, m, hj)
        })
        .collect();
    let columns = columns?;
    let mut data = vec![T::zero(); entries];
    for (j, column) in columns.iter().enumerate() {
        for (k, value) in column.iter().enumerate() {
            data[k * n + j] = *value;
        }
    }
    Ok(sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(
        m, n, data,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        sonium_solvers_core::array![dx, dy]
    }

    #[test]
    fn test_jacobian_central_linear() {
        // f(x) = [x0 + 2*x1, x0^2] at x = [3, 4]
        // J = [[1, 2], [2*x0, 0]] = [[1, 2], [6, 0]]
        let f = |x: &[f64]| sonium_solvers_core::array![x[0] + 2.0 * x[1], x[0] * x[0]];
        let x = sonium_solvers_core::array![3.0, 4.0];
        let h = f64::EPSILON.powf(1.0 / 3.0);
        let jac = jacobian_central(&f, &x, 2, h);
        assert_relative_eq!(jac[[0, 0]], 1.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[0, 1]], 2.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[1, 0]], 6.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[1, 1]], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_jacobian_central_auto_quadratic() {
        // f(x) = [x0^2 + x1, x0 * x1] at x = [2, -1]
        // J = [[2*x0, 1], [x1, x0]] = [[4, 1], [-1, 2]]
        let f = |x: &[f64]| sonium_solvers_core::array![x[0] * x[0] + x[1], x[0] * x[1]];
        let x = sonium_solvers_core::array![2.0, -1.0];
        let jac = jacobian_central_auto(&f, &x, 2);
        assert_relative_eq!(jac[[0, 0]], 4.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[0, 1]], 1.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[1, 0]], -1.0, epsilon = 1e-7);
        assert_relative_eq!(jac[[1, 1]], 2.0, epsilon = 1e-7);
    }

    #[test]
    #[should_panic(expected = "expected: 2")]
    fn test_jacobian_central_wrong_dim_fails_loudly() {
        // A constraint function returning the wrong dimension must panic,
        // never be silently ignored.
        let f = |x: &[f64]| sonium_solvers_core::array![x[0]];
        let x = sonium_solvers_core::array![1.0, 2.0];
        let _ = jacobian_central_auto(&f, &x, 2);
    }

    #[test]
    fn test_gradient_forward_quadratic() {
        // f(x) = x0^2 + x1^2, grad = [2*x0, 2*x1]
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = sonium_solvers_core::array![3.0, 4.0];
        let g = gradient_forward(&f, &x, f64::EPSILON.sqrt());
        assert_relative_eq!(g[0], 6.0, epsilon = 1e-6);
        assert_relative_eq!(g[1], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_stencils_match_serial() {
        // Sum-of-squares in 10 dimensions: above PAR_FD_THRESHOLD, so the
        // rayon path runs; every entry must be bit-identical to serial.
        let f = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let x: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let h = f64::EPSILON.sqrt();
        assert_eq!(
            gradient_forward_parallel(&f, &x, h),
            gradient_forward(&f, &x, h)
        );
        assert_eq!(
            gradient_central_parallel(&f, &x, h),
            gradient_central(&f, &x, h)
        );
        let congestion = |x: &[f64]| x.iter().map(|v| v * v).collect::<Vec<_>>();
        let serial = try_jacobian_central(&congestion, &x, 10, h).unwrap();
        let parallel = try_jacobian_central_parallel(&congestion, &x, 10, h).unwrap();
        assert_eq!(serial.as_slice(), parallel.as_slice());
        // Below the floor the parallel entry still agrees (serial fallback).
        let tiny = vec![3.0, 4.0];
        assert_eq!(
            gradient_forward_parallel(&f, &tiny, h),
            gradient_forward(&f, &tiny, h)
        );
        // Shape violations propagate through the parallel Jacobian too.
        let bad = |_: &[f64]| vec![1.0];
        assert!(try_jacobian_central_parallel(&bad, &x, 10, h).is_err());
    }

    #[test]
    fn test_gradient_central_quadratic() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = sonium_solvers_core::array![3.0, 4.0];
        let h = f64::EPSILON.powf(1.0 / 3.0);
        let g = gradient_central(&f, &x, h);
        assert_relative_eq!(g[0], 6.0, epsilon = 1e-8);
        assert_relative_eq!(g[1], 8.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gradient_forward_rosenbrock() {
        let x = sonium_solvers_core::array![-1.2, 1.0];
        let g_num = gradient_auto(&rosenbrock, &x);
        let g_exact = rosenbrock_grad(&x);
        for i in 0..2 {
            assert_relative_eq!(g_num[i], g_exact[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_gradient_central_rosenbrock() {
        let x = sonium_solvers_core::array![-1.2, 1.0];
        let g_num = gradient_central_auto(&rosenbrock, &x);
        let g_exact = rosenbrock_grad(&x);
        for i in 0..2 {
            assert_relative_eq!(g_num[i], g_exact[i], epsilon = 1e-7);
        }
    }

    #[test]
    fn test_hessian_vec_product() {
        // f(x) = 0.5 * (x0^2 + 2*x1^2), H = [[1, 0], [0, 2]]
        let grad_fn = |x: &[f64]| sonium_solvers_core::array![x[0], 2.0 * x[1]];
        let x = sonium_solvers_core::array![1.0, 1.0];
        let v = sonium_solvers_core::array![1.0, 0.5];
        let hv = hessian_vec_product(&grad_fn, &x, &v, f64::EPSILON.sqrt());
        // H * v = [1*1 + 0*0.5, 0*1 + 2*0.5] = [1.0, 1.0]
        assert_relative_eq!(hv[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(hv[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gradient_forward_quadratic_f32() {
        // f(x) = x0^2 + x1^2, grad = [2*x0, 2*x1]
        let f = |x: &[f32]| x[0] * x[0] + x[1] * x[1];
        let x = sonium_solvers_core::array![3.0_f32, 4.0_f32];
        let g = gradient_forward(&f, &x, f32::EPSILON.sqrt());
        assert!((g[0] - 6.0_f32).abs() < 1e-2, "g[0]={}", g[0]);
        assert!((g[1] - 8.0_f32).abs() < 1e-2, "g[1]={}", g[1]);
    }

    #[test]
    fn test_hessian_vec_zero_direction() {
        // v = [0, 0] should return [0, 0] without evaluating the gradient
        // (the function checks ||v|| < eps and returns zeros early).
        let grad_fn = |x: &[f64]| sonium_solvers_core::array![x[0], 2.0 * x[1]];
        let x = sonium_solvers_core::array![1.0, 1.0];
        let v = sonium_solvers_core::array![0.0, 0.0];
        let hv = hessian_vec_product(&grad_fn, &x, &v, f64::EPSILON.sqrt());
        assert_relative_eq!(hv[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(hv[1], 0.0, epsilon = 1e-15);
    }
}
