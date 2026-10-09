//! SR1 and DFP quasi-Newton methods.
//!
//! - **SR1** (Symmetric Rank-1): Can produce indefinite Hessian approximations,
//!   which sometimes provides better search directions near saddle points.
//! - **DFP** (Davidon-Fletcher-Powell): The dual of BFGS — updates the
//!   Hessian approximation directly instead of its inverse.

use std::cell::Cell;

use crate::line_search::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use crate::types::{LineSearchMethod, OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Which quasi-Newton update to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuasiNewtonUpdate {
    /// Symmetric Rank-1 update.
    SR1,
    /// Davidon-Fletcher-Powell update.
    DFP,
}

/// Configuration for SR1/DFP.
#[derive(Debug, Clone)]
pub struct QuasiNewtonConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub fun_tolerance: T,
    pub update: QuasiNewtonUpdate,
    /// SR1 skip threshold: skip update if `|s^T (y - B*s)| < r * ||s|| * ||y - B*s||`.
    pub sr1_skip_threshold: T,
    pub line_search: LineSearchMethod,
}

impl Default for QuasiNewtonConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-6,
            fun_tolerance: 1e-12,
            update: QuasiNewtonUpdate::SR1,
            sr1_skip_threshold: 1e-8,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

impl Default for QuasiNewtonConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-4,
            fun_tolerance: 1e-6,
            update: QuasiNewtonUpdate::SR1,
            sr1_skip_threshold: 1e-4,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

/// Minimise `f(x)` using SR1 or DFP quasi-Newton.
pub fn quasi_newton<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &QuasiNewtonConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    super::validate_gradient_start(f, grad_fn, x0)?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        quasi_newton_impl(f, grad_fn, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "quasi-Newton callback".into(),
    })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn quasi_newton_impl<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &QuasiNewtonConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut g = grad_fn(&x);
    if !fx.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "quasi-Newton objective returned a non-finite value".into(),
        });
    }
    super::validate_gradient_output(&g, x.len(), 0, "quasi-Newton gradient")?;
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;

    // Inverse Hessian approximation (starts as identity)
    let mut h_inv = sonium_solvers_core::dense::DenseMatrix::eye(n);
    let callback_non_finite = Cell::new(false);
    let checked_f = |point: &[T]| {
        let value = f(point);
        if !value.is_finite() {
            callback_non_finite.set(true);
        }
        value
    };
    let checked_grad = |point: &[T]| {
        let gradient = grad_fn(point);
        if gradient.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
        }
        gradient
    };

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

        // Search direction
        let d = sonium_solvers_core::vec_ops::vec_neg(&h_inv.dot(&g));

        // If not descent, fall back to steepest descent
        let dg = T::vec_dot(&d, &g);
        let d = if dg >= T::zero() {
            sonium_solvers_core::vec_ops::vec_neg(&g)
        } else {
            d
        };
        super::validate_finite_vector(&d, iter, "quasi-Newton direction")?;

        // Line search
        let ls_result = match config.line_search {
            LineSearchMethod::StrongWolfe => {
                let ls_config = StrongWolfeConfig::default();
                let r = strong_wolfe(&checked_f, &checked_grad, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                grad_evals += r.grad_evals;
                r
            }
            LineSearchMethod::Backtracking => {
                let ls_config = BacktrackingConfig::default();
                let r = backtracking(&checked_f, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                r
            }
            LineSearchMethod::HagerZhang => {
                let ls_config = HagerZhangConfig::default();
                let r =
                    hager_zhang_line_search(&checked_f, &checked_grad, &x, &d, fx, &g, &ls_config);
                fun_evals += r.fun_evals;
                grad_evals += r.grad_evals;
                r
            }
        };
        if callback_non_finite.get() {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: "quasi-Newton line search callback returned a non-finite value".into(),
            });
        }
        super::validate_line_search_result(&ls_result, x.len(), iter, "quasi-Newton line search")?;

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
        let s = d.iter().map(|&di| di * alpha).collect::<Vec<_>>();
        let x_new = sonium_solvers_core::vec_ops::vec_add(&x, &s);
        super::validate_finite_vector(&x_new, iter, "quasi-Newton iterate")?;
        let g_new = ls_result.grad.unwrap_or_else(|| {
            grad_evals += 1;
            grad_fn(&x_new)
        });
        super::validate_gradient_output(&g_new, x.len(), iter, "quasi-Newton gradient")?;
        let fx_new = ls_result.fun;

        // Function tolerance
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

        // Update inverse Hessian
        let y = sonium_solvers_core::vec_ops::vec_sub(&g_new, &g);

        match config.update {
            QuasiNewtonUpdate::SR1 => {
                // SR1: H += (s - H*y)(s - H*y)^T / (s - H*y)^T y
                let hy = h_inv.dot(&y);
                let v = sonium_solvers_core::vec_ops::vec_sub(&s, &hy);
                let vy = T::vec_dot(&v, &y);

                // Skip condition
                if vy.abs() > config.sr1_skip_threshold * T::vec_norm(&v) * T::vec_norm(&y) {
                    let rho = T::one() / vy;
                    for i in 0..n {
                        for j in 0..=i {
                            let update = rho * v[i] * v[j];
                            h_inv[[i, j]] += update;
                            if i != j {
                                h_inv[[j, i]] += update;
                            }
                        }
                    }
                }
            }
            QuasiNewtonUpdate::DFP => {
                // DFP: H = (I - sy^T/y^Ts) H (I - ys^T/y^Ts) + ss^T/y^Ts
                // Equivalent: H += s*s^T / (s^T*y) - H*y*y^T*H / (y^T*H*y)
                let sy = T::vec_dot(&s, &y);
                if sy > T::machine_epsilon() * T::vec_norm(&s) * T::vec_norm(&y) {
                    let hy = h_inv.dot(&y);
                    let yhy = T::vec_dot(&y, &hy);

                    let rho_s = T::one() / sy;
                    let rho_y = if yhy.abs() > T::machine_epsilon() {
                        T::one() / yhy
                    } else {
                        T::zero()
                    };

                    for i in 0..n {
                        for j in 0..=i {
                            let update = rho_s * s[i] * s[j] - rho_y * hy[i] * hy[j];
                            h_inv[[i, j]] += update;
                            if i != j {
                                h_inv[[j, i]] += update;
                            }
                        }
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
    fn test_sr1_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = QuasiNewtonConfig {
            update: QuasiNewtonUpdate::SR1,
            ..Default::default()
        };
        let result = quasi_newton(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn quasi_newton_rejects_non_finite_generated_line_search_state() {
        let f = |x: &[f64]| x[0] * x[0];
        let gradient_calls = std::cell::Cell::new(0);
        let g = |_x: &[f64]| {
            let calls = gradient_calls.get() + 1;
            gradient_calls.set(calls);
            if calls <= 2 {
                vec![-1.0]
            } else {
                vec![f64::NAN]
            }
        };
        let config = QuasiNewtonConfig {
            max_iterations: 1,
            ..QuasiNewtonConfig::default()
        };

        let error = quasi_newton(&f, &g, &[0.0], &config)
            .expect_err("generated non-finite line-search state must fail");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn test_dfp_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = QuasiNewtonConfig {
            update: QuasiNewtonUpdate::DFP,
            ..Default::default()
        };
        let result = quasi_newton(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_sr1_rosenbrock() {
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
        let config = QuasiNewtonConfig::default();
        let result = quasi_newton(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_dfp_rosenbrock() {
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
        let config = QuasiNewtonConfig {
            update: QuasiNewtonUpdate::DFP,
            ..Default::default()
        };
        let result = quasi_newton(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_sr1_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: QuasiNewtonConfig<f32> = QuasiNewtonConfig {
            update: QuasiNewtonUpdate::SR1,
            ..Default::default()
        };
        let result = quasi_newton(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }
}
