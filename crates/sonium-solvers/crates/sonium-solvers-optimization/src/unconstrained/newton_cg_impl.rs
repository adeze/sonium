//! Newton-CG (truncated Newton / Hessian-free Newton).
//!
//! Uses CG to approximately solve the Newton system `H*d = -g`, where the
//! Hessian-vector product `H*v` is provided by the user (or computed via
//! finite differences of the gradient).

use std::cell::Cell;

use crate::line_search::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use crate::types::{LineSearchMethod, OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for Newton-CG.
#[derive(Debug, Clone)]
pub struct NewtonCgConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    /// Maximum CG inner iterations per Newton step.
    pub max_cg_iterations: usize,
    /// CG inner loop tolerance factor (Eisenstat-Walker): `||r|| <= eta * ||g||`
    /// where `eta = min(0.5, sqrt(||g||))`.
    pub cg_tolerance_factor: T,
    pub line_search: LineSearchMethod,
}

impl Default for NewtonCgConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-6,
            max_cg_iterations: 200,
            cg_tolerance_factor: 0.5,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

impl Default for NewtonCgConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            grad_tolerance: 1e-4,
            max_cg_iterations: 200,
            cg_tolerance_factor: 0.5,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

/// Minimise `f(x)` using Newton-CG (truncated Newton).
///
/// `hess_vec_fn(x, v)` computes the Hessian-vector product `H(x) * v`.
pub fn newton_cg<T, F, G, Hv>(
    f: &F,
    grad_fn: &G,
    hess_vec_fn: &Hv,
    x0: &[T],
    config: &NewtonCgConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    super::validate_gradient_start(f, grad_fn, x0)?;
    super::validate_hessian_vector_start(hess_vec_fn, x0, "Newton-CG Hessian-vector preflight")?;
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
        if gradient.len() != point.len() || gradient.iter().any(|value| !value.is_finite()) {
            callback_non_finite.set(true);
            vec![T::zero(); point.len()]
        } else {
            gradient
        }
    };
    let checked_hess_vec = |point: &[T], vector: &[T]| {
        let value = hess_vec_fn(point, vector);
        if value.len() != point.len() || value.iter().any(|entry| !entry.is_finite()) {
            callback_non_finite.set(true);
            vec![T::zero(); point.len()]
        } else {
            value
        }
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        newton_cg_impl(&checked_f, &checked_grad, &checked_hess_vec, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "Newton-CG callback".into(),
    })??;
    if callback_non_finite.get() {
        return Err(OptimError::NumericalError {
            iteration: result.iterations,
            message: "Newton-CG callback returned a non-finite or malformed value".into(),
        });
    }
    super::validate_result(&result)?;
    Ok(result)
}

fn newton_cg_impl<T, F, G, Hv>(
    f: &F,
    grad_fn: &G,
    hess_vec_fn: &Hv,
    x0: &[T],
    config: &NewtonCgConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut g = grad_fn(&x);
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;
    let mut hess_evals = 0_usize;

    for iter in 0..config.max_iterations {
        let grad_norm = T::vec_norm(&g);
        let grad_inf_norm = g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);

        if grad_inf_norm < config.grad_tolerance {
            return Ok(OptimResult {
                x,
                fun: fx,
                grad: Some(g),
                iterations: iter,
                fun_evals,
                grad_evals,
                hess_evals,
                grad_norm: grad_inf_norm,
                converged: true,
                status: OptimStatus::Converged,
                kkt: None,
            });
        }

        // Eisenstat-Walker forcing term
        let eta = T::min(config.cg_tolerance_factor, grad_norm.sqrt());
        let cg_tol = eta * grad_norm;

        // Inner CG loop: solve H*d = -g approximately
        let (d, inner_hess_evals) =
            cg_steihaug(&x, &g, hess_vec_fn, cg_tol, config.max_cg_iterations);
        hess_evals += inner_hess_evals;

        // Ensure descent direction
        let dg = T::vec_dot(&d, &g);
        let d = if dg >= T::zero() {
            // CG gave non-descent; fall back to steepest descent
            sonium_solvers_core::vec_ops::vec_neg(&g)
        } else {
            d
        };

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
                hess_evals,
                grad_norm: grad_inf_norm,
                converged: false,
                status: OptimStatus::LineSearchFailed,
                kkt: None,
            });
        }

        let alpha = ls_result.alpha;
        x = sonium_solvers_core::vec_ops::vec_add_scaled(&x, &d, alpha);
        fx = ls_result.fun;
        g = ls_result.grad.unwrap_or_else(|| {
            grad_evals += 1;
            grad_fn(&x)
        });
    }

    let grad_norm = g.iter().map(|&gi| gi.abs()).fold(T::zero(), T::max);
    Ok(OptimResult {
        x,
        fun: fx,
        grad: Some(g),
        iterations: config.max_iterations,
        fun_evals,
        grad_evals,
        hess_evals,
        grad_norm,
        converged: false,
        status: OptimStatus::MaxIterationsReached,
        kkt: None,
    })
}

/// CG-Steihaug: approximately solve `H*d = -g` with early termination
/// on negative curvature (returns steepest descent direction scaled).
fn cg_steihaug<T, Hv>(
    x: &[T],
    g: &[T],
    hess_vec_fn: &Hv,
    tol: T,
    max_iters: usize,
) -> (Vec<T>, usize)
where
    T: RealScalar,
    Hv: Fn(&[T], &[T]) -> Vec<T>,
{
    let n = g.len();
    let mut d = vec![T::zero(); n];
    let mut r = g.to_vec(); // residual = g + H*d = g (since d=0)
    let mut p = sonium_solvers_core::vec_ops::vec_neg(&r);
    let mut hess_evals = 0;

    let r_norm_sq = T::vec_dot(&r, &r);
    if r_norm_sq.sqrt() < tol {
        return (d, 0);
    }

    let mut rr = r_norm_sq;

    for _ in 0..max_iters {
        let hp = hess_vec_fn(x, &p);
        hess_evals += 1;
        let php = T::vec_dot(&p, &hp);

        // Negative curvature: take the steepest descent direction
        if php <= T::zero() {
            if T::vec_dot(&d, &d) == T::zero() {
                return (sonium_solvers_core::vec_ops::vec_neg(g), hess_evals);
            }
            return (d, hess_evals);
        }

        let alpha = rr / php;
        T::vec_axpy(alpha, &p, &mut d);
        T::vec_axpy(alpha, &hp, &mut r);

        let rr_new = T::vec_dot(&r, &r);
        if rr_new.sqrt() < tol {
            break;
        }

        let beta = rr_new / rr;
        p = sonium_solvers_core::vec_ops::vec_add(
            &sonium_solvers_core::vec_ops::vec_neg(&r),
            &sonium_solvers_core::vec_ops::vec_scale(&p, beta),
        );
        rr = rr_new;
    }

    (d, hess_evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_newton_cg_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let hv = |_x: &[f64], v: &[f64]| v.iter().map(|&vi| 2.0 * vi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = NewtonCgConfig::default();
        let result = newton_cg(&f, &g, &hv, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-12);
    }

    #[test]
    fn test_newton_cg_rosenbrock() {
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
        let hv = |x: &[f64], v: &[f64]| {
            let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
            let h01 = -400.0 * x[0];
            let h11 = 200.0;
            sonium_solvers_core::array![h00 * v[0] + h01 * v[1], h01 * v[0] + h11 * v[1]]
        };
        let x0 = sonium_solvers_core::array![-1.2, 1.0];
        let config = NewtonCgConfig::default();
        let result = newton_cg(&f, &g, &hv, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_newton_cg_already_at_min() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let hv = |_x: &[f64], v: &[f64]| v.iter().map(|&vi| 2.0 * vi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![0.0, 0.0];
        let config = NewtonCgConfig::default();
        let result = newton_cg(&f, &g, &hv, &x0, &config).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_newton_cg_negative_curvature() {
        // f(x) = -x.dot(x) + x[0]^4 + x[1]^4
        // This is indefinite near the origin: the Hessian of -x.dot(x) is -2I,
        // so the quartic terms dominate far from origin but near zero the Hessian
        // is negative definite. The CG inner loop should detect negative curvature
        // and fall back to steepest descent.
        let f = |x: &[f64]| -x.dot(x) + x[0].powi(4) + x[1].powi(4);
        let g = |x: &[f64]| {
            sonium_solvers_core::array![
                -2.0 * x[0] + 4.0 * x[0].powi(3),
                -2.0 * x[1] + 4.0 * x[1].powi(3),
            ]
        };
        // Hessian-vector product: H = diag(-2 + 12*x_i^2), so Hv = (-2 + 12*x_i^2)*v_i
        // We intentionally return -2*v to simulate a negative definite Hessian
        // (as if evaluating near the origin where x_i ~ 0).
        let hv = |_x: &[f64], v: &[f64]| v.iter().map(|&vi| -2.0 * vi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![0.5, 0.5];
        let f0 = f(&x0);
        let config = NewtonCgConfig {
            max_iterations: 200,
            ..Default::default()
        };
        let result = newton_cg(&f, &g, &hv, &x0, &config).unwrap();
        // The solver should still make progress (function decreases).
        assert!(
            result.fun < f0,
            "Expected function decrease: initial={}, final={}",
            f0,
            result.fun
        );
    }

    #[test]
    fn test_newton_cg_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let hv = |_x: &[f32], v: &[f32]| v.iter().map(|&vi| 2.0_f32 * vi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: NewtonCgConfig<f32> = NewtonCgConfig::default();
        let result = newton_cg(&f, &g, &hv, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }

    #[test]
    fn newton_cg_rejects_non_finite_generated_callback_values() {
        let objective_calls = std::cell::Cell::new(0usize);
        let objective = |x: &[f64]| {
            let call = objective_calls.get();
            objective_calls.set(call + 1);
            if call >= 2 {
                f64::NAN
            } else {
                x.iter().map(|value| value * value).sum()
            }
        };
        let gradient = |x: &[f64]| x.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
        let hessian_vector =
            |_x: &[f64], vector: &[f64]| vector.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
        let error = newton_cg(
            &objective,
            &gradient,
            &hessian_vector,
            &[1.0, 1.0],
            &NewtonCgConfig::default(),
        )
        .expect_err("generated non-finite objective must be rejected");
        assert!(matches!(error, OptimError::NumericalError { .. }));

        let hessian_calls = std::cell::Cell::new(0usize);
        let hessian_vector = |_x: &[f64], vector: &[f64]| {
            let call = hessian_calls.get();
            hessian_calls.set(call + 1);
            if call >= 1 {
                vec![f64::INFINITY; vector.len()]
            } else {
                vector.iter().map(|value| 2.0 * value).collect::<Vec<_>>()
            }
        };
        let error = newton_cg(
            &|x: &[f64]| x.iter().map(|value| value * value).sum(),
            &gradient,
            &hessian_vector,
            &[1.0, 1.0],
            &NewtonCgConfig::default(),
        )
        .expect_err("generated non-finite Hessian-vector value must be rejected");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }
}
