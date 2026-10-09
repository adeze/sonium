//! L-BFGS (limited-memory BFGS) quasi-Newton method.
//!
//! Stores the last `m` correction pairs `{s_k, y_k}` and uses the two-loop
//! recursion to compute the search direction. O(mn) per iteration vs O(n^2)
//! for full BFGS.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::line_search::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use crate::types::{LineSearchMethod, OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for L-BFGS.
#[derive(Debug, Clone)]
pub struct LbfgsConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub fun_tolerance: T,
    pub step_tolerance: T,
    /// Number of correction pairs to store (m, typically 5-20).
    pub memory: usize,
    pub line_search: LineSearchMethod,
}

impl<T: RealScalar> Default for LbfgsConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: T::from_f64(1e-6).unwrap(),
            fun_tolerance: T::from_f64(1e-12).unwrap(),
            step_tolerance: T::from_f64(1e-12).unwrap(),
            memory: 10,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

/// Minimise `f(x)` using the L-BFGS quasi-Newton method.
pub fn lbfgs<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &LbfgsConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    super::validate_gradient_start(f, grad_fn, x0)?;
    let result =
        catch_unwind(AssertUnwindSafe(|| lbfgs_impl(f, grad_fn, x0, config))).map_err(|_| {
            OptimError::CallbackPanicked {
                iteration: 0,
                stage: "L-BFGS callback".into(),
            }
        })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn lbfgs_impl<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &LbfgsConfig<T>,
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
    if config.max_iterations == 0 || config.memory == 0 {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "max_iterations and memory must be greater than zero".to_string(),
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

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    if !fx.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "objective returned a non-finite value".to_string(),
        });
    }
    let mut g = grad_fn(&x);
    if g.len() != x.len() {
        return Err(OptimError::DimensionMismatch {
            expected: x.len(),
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

    // Correction pairs ring buffer
    let mut s_history: VecDeque<Vec<T>> = VecDeque::with_capacity(config.memory);
    let mut y_history: VecDeque<Vec<T>> = VecDeque::with_capacity(config.memory);
    let mut rho_history: VecDeque<T> = VecDeque::with_capacity(config.memory);

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

        // Two-loop recursion to compute d = -H_k * g_k
        let d = two_loop_recursion(&g, &s_history, &y_history, &rho_history);

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
        let s = d.iter().map(|&di| di * alpha).collect::<Vec<_>>();
        let x_new = sonium_solvers_core::vec_ops::vec_add(&x, &s);
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
        let fx_new = ls_result.fun;
        if !fx_new.is_finite() {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: "line search returned a non-finite objective value".to_string(),
            });
        }
        if g_new.len() != x.len() {
            return Err(OptimError::DimensionMismatch {
                expected: x.len(),
                got: g_new.len(),
            });
        }
        if let Some(index) = g_new.iter().position(|value| !value.is_finite()) {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: format!("gradient contains a non-finite value at index {index}"),
            });
        }

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

        // Step tolerance (reuse the step norm below in the curvature test).
        let s_norm = T::vec_norm(&s);
        if s_norm < config.step_tolerance * (T::one() + T::vec_norm(&x)) {
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

        // Store correction pair
        let y = sonium_solvers_core::vec_ops::vec_sub(&g_new, &g);
        let sy = T::vec_dot(&s, &y);

        if sy > T::machine_epsilon() * s_norm * T::vec_norm(&y) {
            if s_history.len() >= config.memory {
                s_history.pop_front();
                y_history.pop_front();
                rho_history.pop_front();
            }
            rho_history.push_back(T::one() / sy);
            s_history.push_back(s);
            y_history.push_back(y);
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

/// L-BFGS two-loop recursion.
///
/// Computes `d = -H_k * g` where `H_k` is the limited-memory inverse Hessian.
fn two_loop_recursion<T: RealScalar>(
    g: &[T],
    s_hist: &VecDeque<Vec<T>>,
    y_hist: &VecDeque<Vec<T>>,
    rho_hist: &VecDeque<T>,
) -> Vec<T> {
    let m = s_hist.len();
    let mut q = g.to_vec();
    let mut alphas = vec![T::zero(); m];

    // First loop: newest to oldest
    for i in (0..m).rev() {
        alphas[i] = rho_hist[i] * T::vec_dot(&s_hist[i], &q);
        T::vec_axpy(-alphas[i], &y_hist[i], &mut q);
    }

    // Initial Hessian scaling: H_0 = gamma * I
    // gamma = s^T y / y^T y for the most recent pair
    if m > 0 {
        let sy = T::vec_dot(&s_hist[m - 1], &y_hist[m - 1]);
        let yy = T::vec_dot(&y_hist[m - 1], &y_hist[m - 1]);
        if yy > T::zero() {
            let gamma = sy / yy;
            sonium_solvers_core::vec_ops::vec_mapv_inplace(&mut q, |qi| qi * gamma);
        }
    }

    // Second loop: oldest to newest
    for i in 0..m {
        let beta = rho_hist[i] * T::vec_dot(&y_hist[i], &q);
        T::vec_axpy(alphas[i] - beta, &s_hist[i], &mut q);
    }

    // Return -H*g
    q.iter().map(|&qi| -qi).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lbfgs_rejects_invalid_input_and_callbacks() {
        let config = LbfgsConfig::<f64>::default();
        let objective = |x: &[f64]| x.iter().map(|value| value * value).sum();
        let gradient = |x: &[f64]| x.iter().map(|value| 2.0 * value).collect::<Vec<_>>();

        assert!(matches!(
            lbfgs(&objective, &gradient, &[], &config),
            Err(OptimError::DimensionMismatch {
                expected: 1,
                got: 0
            })
        ));
        assert!(matches!(
            lbfgs(&objective, &gradient, &[f64::NAN], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let bad_gradient = |_x: &[f64]| vec![f64::NAN];
        assert!(matches!(
            lbfgs(&objective, &bad_gradient, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));

        let bad_objective = |_x: &[f64]| f64::INFINITY;
        assert!(matches!(
            lbfgs(&bad_objective, &gradient, &[1.0], &config),
            Err(OptimError::NumericalError { iteration: 0, .. })
        ));
    }

    #[test]
    fn test_lbfgs_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = LbfgsConfig::default();
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-12);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
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
        let config = LbfgsConfig::default();
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lbfgs_high_dim_rosenbrock() {
        let n = 20;
        let f = |x: &[f64]| {
            let mut val = 0.0;
            for i in 0..(x.len() - 1) {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                val += a * a + 100.0 * b * b;
            }
            val
        };
        let g = |x: &[f64]| {
            let n = x.len();
            let mut grad = vec![0.0; n];
            for i in 0..(n - 1) {
                grad[i] += -2.0 * (1.0 - x[i]) - 400.0 * x[i] * (x[i + 1] - x[i] * x[i]);
                grad[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
            }
            grad
        };
        let mut x0 = vec![1.0; n];
        for i in (0..n).step_by(2) {
            x0[i] = -1.2;
        }
        let config = LbfgsConfig {
            max_iterations: 5000,
            ..Default::default()
        };
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_lbfgs_already_at_min() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = LbfgsConfig::default();
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_lbfgs_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: LbfgsConfig<f32> = LbfgsConfig::default();
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }

    #[test]
    fn test_lbfgs_step_tolerance() {
        // Use very tight fun_tolerance so it won't trigger, but very loose
        // step_tolerance (1.0) so the solver converges via StepTolerance quickly.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = LbfgsConfig {
            fun_tolerance: 1e-30,
            step_tolerance: 1.0, // very loose: triggers as soon as step < 1.0*(1+||x||)
            ..Default::default()
        };
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "Expected convergence, got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            OptimStatus::StepTolerance,
            "Expected StepTolerance, got {:?}",
            result.status
        );
        // Should converge in very few iterations due to the loose step tolerance
        assert!(
            result.iterations <= 5,
            "Expected quick convergence via StepTolerance, got {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_lbfgs_memory_1() {
        // Minimal memory (m=1) should still converge on the sphere
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = LbfgsConfig {
            memory: 1,
            ..Default::default()
        };
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "L-BFGS with memory=1 failed: {:?}",
            result.status
        );
        assert!(result.fun < 1e-12);
    }

    #[test]
    fn test_lbfgs_memory_overflow() {
        // memory=2 on 5D Rosenbrock: the ring buffer should cycle correctly
        // as total iterations will exceed memory size
        let f = |x: &[f64]| {
            let mut val = 0.0;
            for i in 0..(x.len() - 1) {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                val += a * a + 100.0 * b * b;
            }
            val
        };
        let g = |x: &[f64]| {
            let n = x.len();
            let mut grad = vec![0.0; n];
            for i in 0..(n - 1) {
                grad[i] += -2.0 * (1.0 - x[i]) - 400.0 * x[i] * (x[i + 1] - x[i] * x[i]);
                grad[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
            }
            grad
        };
        let x0 = sonium_solvers_core::array![-1.2, 1.0, -1.2, 1.0, -1.2];
        let config = LbfgsConfig {
            memory: 2,
            max_iterations: 10_000,
            ..Default::default()
        };
        let result = lbfgs(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "L-BFGS memory=2 on 5D Rosenbrock failed: {:?}, fun={}",
            result.status, result.fun
        );
        assert!(result.fun < 1e-4, "fun={}", result.fun);
        // Verify ring buffer cycled: iterations should exceed memory size
        assert!(
            result.iterations > 2,
            "Expected more than 2 iterations, got {}",
            result.iterations
        );
    }

    #[test]
    fn lbfgs_maps_callback_panics_to_typed_error() {
        let objective = |_x: &[f64]| -> f64 { panic!("objective failure") };
        let gradient = |_x: &[f64]| vec![0.0];
        let error = lbfgs(&objective, &gradient, &[0.0], &LbfgsConfig::default())
            .expect_err("callback panic must not escape the public API");
        assert!(matches!(error, OptimError::CallbackPanicked { .. }));
    }
}
