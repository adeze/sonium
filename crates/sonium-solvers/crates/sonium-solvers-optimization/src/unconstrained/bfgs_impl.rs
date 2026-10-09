//! BFGS quasi-Newton method.
//!
//! Maintains a dense n×n inverse Hessian approximation. For large problems
//! (n > ~1000), prefer [`lbfgs`](super::lbfgs).

use crate::line_search::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use crate::types::{LineSearchMethod, OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for BFGS.
#[derive(Debug, Clone)]
pub struct BfgsConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub fun_tolerance: T,
    pub step_tolerance: T,
    pub line_search: LineSearchMethod,
}

impl Default for BfgsConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-6,
            fun_tolerance: 1e-12,
            step_tolerance: 1e-12,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

impl Default for BfgsConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-4,
            fun_tolerance: 1e-6,
            step_tolerance: 1e-6,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

/// Minimise `f(x)` using the BFGS quasi-Newton method.
pub fn bfgs<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &BfgsConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    super::validate_gradient_start(f, grad_fn, x0)?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bfgs_impl(f, grad_fn, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "BFGS callback".into(),
    })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn bfgs_impl<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &BfgsConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    if x0.is_empty() {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if let Some(index) = x0.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("initial point contains a non-finite value at index {index}"),
        });
    }
    if config.max_iterations == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "max_iterations must be greater than zero".to_string(),
        });
    }
    for (name, value) in [
        ("grad_tolerance", config.grad_tolerance),
        ("fun_tolerance", config.fun_tolerance),
        ("step_tolerance", config.step_tolerance),
    ] {
        if !value.is_finite() || value <= T::zero() {
            return Err(OptimError::NumericalError {
                iteration: 0,
                message: format!("{name} must be finite and positive"),
            });
        }
    }

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    if !fx.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "objective returned a non-finite value".to_string(),
        });
    }
    let mut g = grad_fn(&x);
    if g.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            got: g.len(),
        });
    }
    if let Some(index) = g.iter().position(|value| !value.is_finite()) {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: format!("gradient contains a non-finite value at index {index}"),
        });
    }
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;

    // Start with identity inverse Hessian approximation
    let mut h_inv: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::eye(n);

    // Pre-allocate work buffers (reused across iterations, zero allocation in loop)
    let mut d = vec![T::zero(); n];
    let mut s_buf = vec![T::zero(); n];
    let mut hy_buf = vec![T::zero(); n];

    for iter in 0..config.max_iterations {
        let grad_norm = g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);

        if grad_norm < config.grad_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(g),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Search direction: d = -H^{-1} * g (zero allocation via pre-allocated buffer)
        for (i, slot) in d.iter_mut().enumerate() {
            let mut sum = T::zero();
            let row = h_inv.row(i);
            for j in 0..n {
                sum += row[j] * g[j];
            }
            *slot = -sum;
        }

        // Line search
        let ls_result = match config.line_search {
            LineSearchMethod::StrongWolfe => {
                let ls_config = StrongWolfeConfig::default();
                let r = strong_wolfe(f, grad_fn, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                grad_evals += r.grad_evals;
                r
            }
            LineSearchMethod::Backtracking => {
                let ls_config = BacktrackingConfig::default();
                let r = backtracking(f, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                r
            }
            LineSearchMethod::HagerZhang => {
                let ls_config = HagerZhangConfig::default();
                let r = hager_zhang_line_search(f, grad_fn, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                grad_evals += r.grad_evals;
                r
            }
        };

        if !ls_result.success {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(g),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: false,
                status: OptimStatus::LineSearchFailed,
                kkt: None,
            });
        }

        let alpha = ls_result.alpha;
        // s = alpha * d (zero allocation via pre-allocated buffer)
        s_buf.copy_from_slice(&d);
        sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut s_buf, |di| di * alpha);
        let x_new = sonium_solvers_core::vec_ops::vec_add(&x, &s_buf);
        if let Some(index) = x_new.iter().position(|value| !value.is_finite()) {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: format!("line search produced a non-finite point at index {index}"),
            });
        }
        let g_new = ls_result.grad.unwrap_or_else(|| {
            grad_evals += 1;
            grad_fn(&x_new)
        });
        if !ls_result.fun.is_finite() {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: "line search returned a non-finite objective value".to_string(),
            });
        }
        if g_new.len() != n {
            return Err(OptimError::DimensionMismatch {
                expected: n,
                got: g_new.len(),
            });
        }
        if let Some(index) = g_new.iter().position(|value| !value.is_finite()) {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: format!("gradient contains a non-finite value at index {index}"),
            });
        }
        let fx_new = ls_result.fun;

        // Check function tolerance
        if (fx - fx_new).abs() < config.fun_tolerance * (T::one() + fx.abs()) {
            return Ok(OptimResult {
                x: x_new,
                fun: fx_new,
                grad: Some(g_new),
                iterations: iter + 1,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::FunctionTolerance,
                kkt: None,
            });
        }

        // Check step tolerance
        if T::vec_norm(&s_buf) < config.step_tolerance * (T::one() + T::vec_norm(&x)) {
            return Ok(OptimResult {
                x: x_new,
                fun: fx_new,
                grad: Some(g_new),
                iterations: iter + 1,
                fun_evals,
                grad_evals,
                hess_evals: 0,
                grad_norm,
                converged: true,
                status: OptimStatus::StepTolerance,
                kkt: None,
            });
        }

        // BFGS update of inverse Hessian
        let y = sonium_solvers_core::vec_ops::vec_sub(&g_new, &g); // gradient difference
        let sy = T::vec_dot(&s_buf, &y);

        // Skip update if curvature condition is violated
        if sy > T::machine_epsilon() * T::vec_norm(&s_buf) * T::vec_norm(&y) {
            let rho = T::one() / sy;
            // H = (I - rho * s * y^T) * H * (I - rho * y * s^T) + rho * s * s^T
            // hy = H^{-1} * y (zero allocation via pre-allocated buffer)
            for (i, slot) in hy_buf.iter_mut().enumerate() {
                let mut sum = T::zero();
                let row = h_inv.row(i);
                for j in 0..n {
                    sum += row[j] * y[j];
                }
                *slot = sum;
            }
            let yhy = T::vec_dot(&y, &hy_buf);

            // Sherman-Morrison-Woodbury form (more numerically stable):
            // H_new = H + (sy + y^T H y) / sy^2 * s s^T - (H y s^T + s y^T H) / sy
            let factor1 = (sy + yhy) * rho * rho;
            for i in 0..n {
                for j in 0..=i {
                    let update = factor1 * s_buf[i] * s_buf[j]
                        - rho * (hy_buf[i] * s_buf[j] + s_buf[i] * hy_buf[j]);
                    h_inv[[i, j]] += update;
                    if i != j {
                        h_inv[[j, i]] += update;
                    }
                }
            }
        }

        x = x_new;
        fx = fx_new;
        g = g_new;
    }

    let grad_norm = g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(g),
        iterations: config.max_iterations,
        fun_evals,
        grad_evals,
        hess_evals: 0,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bfgs_rejects_invalid_input_and_callbacks() {
        let config = BfgsConfig::<f64>::default();
        let objective = |x: &[f64]| x.iter().map(|value| value * value).sum();
        let gradient = |x: &[f64]| x.iter().map(|value| 2.0 * value).collect::<Vec<_>>();

        assert!(matches!(
            bfgs(&objective, &gradient, &[], &config),
            Err(OptimError::DimensionMismatch {
                expected: 1,
                got: 0
            })
        ));
        assert!(matches!(
            bfgs(&objective, &gradient, &[f64::NAN], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let bad_gradient = |_x: &[f64]| vec![f64::NAN];
        assert!(matches!(
            bfgs(&objective, &bad_gradient, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let bad_objective = |_x: &[f64]| f64::INFINITY;
        assert!(matches!(
            bfgs(&bad_objective, &gradient, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));
    }

    #[test]
    fn test_bfgs_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = BfgsConfig::default();
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-12);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let g = |x: &[f64]| {
            let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dy = 200.0 * (x[1] - x[0] * x[0]);
            sonium_solvers_core::array![dx, dy]
        };
        let x0 = sonium_solvers_core::array![-1.2, 1.0];
        let config = BfgsConfig::default();
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_bfgs_booth() {
        let f = |x: &[f64]| {
            let a = x[0] + 2.0 * x[1] - 7.0;
            let b = 2.0 * x[0] + x[1] - 5.0;
            a * a + b * b
        };
        let g = |x: &[f64]| {
            let a = x[0] + 2.0 * x[1] - 7.0;
            let b = 2.0 * x[0] + x[1] - 5.0;
            sonium_solvers_core::array![2.0 * a + 4.0 * b, 4.0 * a + 2.0 * b]
        };
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = BfgsConfig::default();
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_already_at_min() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = BfgsConfig::default();
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_bfgs_backtracking() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = BfgsConfig {
            line_search: LineSearchMethod::Backtracking,
            ..Default::default()
        };
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_bfgs_curvature_skip() {
        // Use a very flat function where the curvature condition sy <= eps*||s||*||y||
        // can be triggered. f(x) = 1e-15 * x.dot(x) has extremely small gradients,
        // so s and y are tiny and the BFGS update may be skipped.
        // Start near the minimum where s and y become nearly orthogonal.
        let f = |x: &[f64]| 1e-15 * x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2e-15 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![1.0, 1.0];
        let config = BfgsConfig {
            grad_tolerance: 1e-30,
            fun_tolerance: 1e-30,
            step_tolerance: 1e-30,
            max_iterations: 500,
            ..Default::default()
        };
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        // The algorithm should not crash even when curvature updates are skipped.
        // It should either converge or hit max iterations gracefully.
        assert!(
            result.fun <= f(&x0),
            "BFGS should not increase the objective: initial={}, final={}",
            f(&x0),
            result.fun
        );
    }

    #[test]
    fn test_bfgs_line_search_failure() {
        // f(x) = exp(100 * x[0]) + exp(100 * x[1]): extremely steep.
        // The strong Wolfe line search may fail to find a step satisfying the conditions.
        let f = |x: &[f64]| (100.0 * x[0]).exp() + (100.0 * x[1]).exp();
        let g = |x: &[f64]| {
            sonium_solvers_core::array![100.0 * (100.0 * x[0]).exp(), 100.0 * (100.0 * x[1]).exp()]
        };
        let x0 = sonium_solvers_core::array![10.0, 10.0];
        let config = BfgsConfig {
            line_search: LineSearchMethod::StrongWolfe,
            max_iterations: 50,
            ..Default::default()
        };
        match bfgs(&f, &g, &x0, &config) {
            Ok(result) => assert!(
                result.status == OptimStatus::LineSearchFailed
                    || result.status == OptimStatus::MaxIterationsReached
                    || result.converged,
                "Expected LineSearchFailed or graceful termination, got {:?}",
                result.status
            ),
            Err(OptimError::NumericalError { .. }) => {}
            Err(error) => panic!("unexpected BFGS failure: {error:?}"),
        }
    }

    #[test]
    fn test_bfgs_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: BfgsConfig<f32> = BfgsConfig::default();
        let result = bfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }
}
