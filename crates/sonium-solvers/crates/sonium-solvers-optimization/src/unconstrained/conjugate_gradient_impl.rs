//! Nonlinear conjugate gradient methods.
//!
//! Variants: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, PR+ (guaranteed
//! descent), and Hager–Zhang (CG-DESCENT, guaranteed descent independent of
//! the line search accuracy).

use std::cell::Cell;

use crate::line_search::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use crate::types::{LineSearchMethod, OptimError, OptimResult, OptimStatus, RealScalar};
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// CG beta formula variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgVariant {
    FletcherReeves,
    PolakRibiere,
    HestenesStiefel,
    /// Polak-Ribiere with `max(beta, 0)` (guaranteed descent).
    PolakRibierePlus,
    /// Hager–Zhang (2005) CG-DESCENT update,
    /// `beta = (y - 2 d ||y||^2 / dᵀy)ᵀ g_new / dᵀy`, truncated below by
    /// `eta = -1 / (||d|| min(0.01, ||g_new||))`, which guarantees descent
    /// (`gᵀd <= -7/8 ||g||²`) for any line search. Pairs by default with
    /// the Hager–Zhang approximate-Wolfe line search: when this variant is
    /// selected and `line_search` is left at its default
    /// ([`LineSearchMethod::StrongWolfe`]), the Hager–Zhang search is used
    /// instead; set `line_search` explicitly to override.
    HagerZhang,
}

/// Configuration for nonlinear CG.
#[derive(Debug, Clone)]
pub struct NonlinearCgConfig<T: RealScalar> {
    pub max_iterations: usize,
    pub grad_tolerance: T,
    pub variant: CgVariant,
    /// Restart CG every `restart_period` iterations (0 = no forced restart).
    pub restart_period: usize,
    pub line_search: LineSearchMethod,
}

impl Default for NonlinearCgConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-6,
            variant: CgVariant::PolakRibierePlus,
            restart_period: 0,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

impl Default for NonlinearCgConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tolerance: 1e-4,
            variant: CgVariant::PolakRibierePlus,
            restart_period: 0,
            line_search: LineSearchMethod::StrongWolfe,
        }
    }
}

/// Minimise `f(x)` using nonlinear conjugate gradient.
pub fn nonlinear_cg<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &NonlinearCgConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    super::validate_gradient_start(f, grad_fn, x0)?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nonlinear_cg_impl(f, grad_fn, x0, config)
    }))
    .map_err(|_| OptimError::CallbackPanicked {
        iteration: 0,
        stage: "nonlinear-CG callback".into(),
    })??;
    super::validate_result(&result)?;
    Ok(result)
}

fn nonlinear_cg_impl<T, F, G>(
    f: &F,
    grad_fn: &G,
    x0: &[T],
    config: &NonlinearCgConfig<T>,
) -> Result<OptimResult<T>, OptimError>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut g = grad_fn(&x);
    if !fx.is_finite() {
        return Err(OptimError::NumericalError {
            iteration: 0,
            message: "nonlinear-CG objective returned a non-finite value".into(),
        });
    }
    super::validate_gradient_output(&g, x.len(), 0, "nonlinear-CG gradient")?;
    let mut fun_evals = 1_usize;
    let mut grad_evals = 1_usize;

    let mut d = sonium_solvers_core::vec_ops::vec_neg(&g); // initial direction = steepest descent
    #[allow(unused_assignments)]
    let mut g_prev = g.clone();

    // The Hager–Zhang beta pairs by default with the Hager–Zhang line
    // search (CG-DESCENT): promote the default StrongWolfe selection, but
    // respect an explicit override (e.g. Backtracking).
    let effective_ls = match (config.variant, config.line_search) {
        (CgVariant::HagerZhang, LineSearchMethod::StrongWolfe) => LineSearchMethod::HagerZhang,
        (_, method) => method,
    };

    // Warm start for the Hager–Zhang search (CG-DESCENT psi_2 heuristic):
    // alpha_init = alpha_prev * ||d_prev|| / ||d||. None on the first
    // iteration (alpha_init = 1).
    let mut warm_alpha: Option<T> = None;
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

    let run_line_search = |x: &[T],
                           d: &[T],
                           fx: T,
                           g: &[T],
                           warm_alpha: Option<T>,
                           fun_evals: &mut usize,
                           grad_evals: &mut usize|
     -> crate::line_search::LineSearchResult<T> {
        match effective_ls {
            LineSearchMethod::StrongWolfe => {
                let ls_config = StrongWolfeConfig {
                    c2: T::from_f64(0.1).unwrap(),
                    ..StrongWolfeConfig::default()
                };
                let r = strong_wolfe(&checked_f, &checked_grad, x, d, fx, g, &ls_config);
                *fun_evals += r.fun_evals;
                *grad_evals += r.grad_evals;
                r
            }
            LineSearchMethod::Backtracking => {
                let ls_config = BacktrackingConfig::default();
                let r = backtracking(&checked_f, x, d, fx, g, &ls_config);
                *fun_evals += r.fun_evals;
                r
            }
            LineSearchMethod::HagerZhang => {
                let ls_config = HagerZhangConfig {
                    alpha_init: warm_alpha.unwrap_or_else(T::one),
                    ..HagerZhangConfig::default()
                };
                let r = hager_zhang_line_search(&checked_f, &checked_grad, x, d, fx, g, &ls_config);
                *fun_evals += r.fun_evals;
                *grad_evals += r.grad_evals;
                r
            }
        }
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

        // Line search
        let d_norm_ls = T::vec_dot(&d, &d).sqrt();
        let ls_result =
            run_line_search(&x, &d, fx, &g, warm_alpha, &mut fun_evals, &mut grad_evals);
        if callback_non_finite.get() {
            return Err(OptimError::NumericalError {
                iteration: iter,
                message: "nonlinear-CG line search callback returned a non-finite value".into(),
            });
        }
        super::validate_line_search_result(&ls_result, x.len(), iter, "nonlinear-CG line search")?;

        if !ls_result.success {
            // Restart with steepest descent
            d = sonium_solvers_core::vec_ops::vec_neg(&g);
            let ls_retry =
                run_line_search(&x, &d, fx, &g, warm_alpha, &mut fun_evals, &mut grad_evals);
            if callback_non_finite.get() {
                return Err(OptimError::NumericalError {
                    iteration: iter,
                    message: "nonlinear-CG restart callback returned a non-finite value".into(),
                });
            }
            super::validate_line_search_result(
                &ls_retry,
                x.len(),
                iter,
                "nonlinear-CG restart line search",
            )?;

            if !ls_retry.success {
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

            let alpha = ls_retry.alpha;
            // Steepest-descent restart: the direction changed completely,
            // so the scaled warm start is not meaningful.
            warm_alpha = Some(alpha);
            x = sonium_solvers_core::vec_ops::vec_add_scaled(&x, &d, alpha);
            super::validate_finite_vector(&x, iter, "nonlinear-CG restart iterate")?;
            fx = ls_retry.fun;
            #[allow(unused_assignments)]
            {
                g_prev = g;
            }
            let next_gradient = ls_retry.grad.unwrap_or_else(|| {
                grad_evals += 1;
                grad_fn(&x)
            });
            super::validate_gradient_output(
                &next_gradient,
                x.len(),
                iter,
                "nonlinear-CG restart gradient",
            )?;
            g = next_gradient;
            d = sonium_solvers_core::vec_ops::vec_neg(&g);
            continue;
        }

        let alpha = ls_result.alpha;
        x = sonium_solvers_core::vec_ops::vec_add_scaled(&x, &d, alpha);
        super::validate_finite_vector(&x, iter, "nonlinear-CG iterate")?;
        fx = ls_result.fun;
        g_prev = g;
        let next_gradient = ls_result.grad.unwrap_or_else(|| {
            grad_evals += 1;
            grad_fn(&x)
        });
        super::validate_gradient_output(&next_gradient, x.len(), iter, "nonlinear-CG gradient")?;
        g = next_gradient;

        // Compute beta
        let restart = config.restart_period > 0 && (iter + 1) % config.restart_period == 0;
        let beta = if restart {
            T::zero()
        } else {
            compute_beta(&g, &g_prev, &d, config.variant)
        };

        // Update direction: d = -g + beta * d
        d = sonium_solvers_core::vec_ops::vec_add(
            &sonium_solvers_core::vec_ops::vec_neg(&g),
            &sonium_solvers_core::vec_ops::vec_scale(&d, beta),
        );
        super::validate_finite_vector(&d, iter, "nonlinear-CG direction")?;

        // If direction is not descent, restart
        if T::vec_dot(&d, &g) >= T::zero() {
            d = sonium_solvers_core::vec_ops::vec_neg(&g);
        }

        // CG-DESCENT psi_2 warm start for the next line search:
        // alpha_init = 2 * alpha * ||d_used|| / ||d_new||.
        let d_norm_new = T::vec_dot(&d, &d).sqrt();
        warm_alpha = if d_norm_new > T::machine_epsilon() {
            Some(T::from_f64(2.0).unwrap() * alpha * d_norm_ls / d_norm_new)
        } else {
            None
        };
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

fn compute_beta<T: RealScalar>(g: &[T], g_prev: &[T], d: &[T], variant: CgVariant) -> T {
    let gg = T::vec_dot(g, g);
    let gg_prev = T::vec_dot(g_prev, g_prev);
    let y = sonium_solvers_core::vec_ops::vec_sub(g, g_prev);

    match variant {
        CgVariant::FletcherReeves => {
            if gg_prev.abs() < T::machine_epsilon() {
                T::zero()
            } else {
                gg / gg_prev
            }
        }
        CgVariant::PolakRibiere => {
            if gg_prev.abs() < T::machine_epsilon() {
                T::zero()
            } else {
                T::vec_dot(g, &y) / gg_prev
            }
        }
        CgVariant::HestenesStiefel => {
            let dy = T::vec_dot(d, &y);
            if dy.abs() < T::machine_epsilon() {
                T::zero()
            } else {
                T::vec_dot(g, &y) / dy
            }
        }
        CgVariant::PolakRibierePlus => {
            if gg_prev.abs() < T::machine_epsilon() {
                T::zero()
            } else {
                T::max(T::zero(), T::vec_dot(g, &y) / gg_prev)
            }
        }
        CgVariant::HagerZhang => {
            // Hager–Zhang (2005), CG-DESCENT:
            //   beta = (1 / dᵀy) (y − 2 d ||y||² / dᵀy)ᵀ g_new
            // truncated below by eta = −1/(||d|| min(0.01, ||g_new||)),
            // which guarantees g_newᵀd_new ≤ −(7/8)||g_new||² regardless
            // of line search accuracy (Thm. 3.1 of the paper).
            let dy = T::vec_dot(d, &y);
            if dy.abs() < T::machine_epsilon() {
                T::zero()
            } else {
                let two = T::from_f64(2.0).unwrap();
                let yy = T::vec_dot(&y, &y);
                let beta = (T::vec_dot(&y, g) - two * T::vec_dot(d, g) * yy / dy) / dy;
                let d_norm = T::vec_dot(d, d).sqrt();
                let g_norm = gg.sqrt();
                let eta = -T::one() / (d_norm * T::min(T::from_f64(0.01).unwrap(), g_norm));
                T::max(beta, eta)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_sphere() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0];
        let config = NonlinearCgConfig::default();
        let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn nonlinear_cg_rejects_non_finite_generated_line_search_state() {
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
        let config = NonlinearCgConfig {
            max_iterations: 1,
            ..NonlinearCgConfig::default()
        };

        let error = nonlinear_cg(&f, &g, &[0.0], &config)
            .expect_err("generated non-finite line-search state must fail");
        assert!(matches!(error, OptimError::NumericalError { .. }));
    }

    #[test]
    fn test_cg_rosenbrock() {
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
        let config = NonlinearCgConfig {
            max_iterations: 5000,
            ..Default::default()
        };
        let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "status: {:?}, fun: {}",
            result.status, result.fun
        );
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cg_variants() {
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0, -2.0];

        for variant in [
            CgVariant::FletcherReeves,
            CgVariant::PolakRibiere,
            CgVariant::HestenesStiefel,
            CgVariant::PolakRibierePlus,
            CgVariant::HagerZhang,
        ] {
            let config = NonlinearCgConfig {
                variant,
                ..Default::default()
            };
            let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
            assert!(
                result.converged,
                "variant {:?} failed: {:?}",
                variant, result.status
            );
        }
    }

    #[test]
    fn test_cg_hager_zhang_rosenbrock() {
        // CG-DESCENT (Hager–Zhang beta + Hager–Zhang line search, the
        // default pairing for this variant) on the Rosenbrock banana.
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

        let hz = nonlinear_cg(
            &f,
            &g,
            &x0,
            &NonlinearCgConfig {
                max_iterations: 5000,
                variant: CgVariant::HagerZhang,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            hz.converged,
            "HZ status: {:?}, fun: {}, grad_norm: {}",
            hz.status, hz.fun, hz.grad_norm
        );
        assert_relative_eq!(hz.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(hz.x[1], 1.0, epsilon = 1e-3);

        // Reference: PR+ with its default strong-Wolfe search. The
        // iteration/eval comparison is reported (not asserted): HZ is
        // expected to be competitive or better on this problem.
        let pr = nonlinear_cg(
            &f,
            &g,
            &x0,
            &NonlinearCgConfig {
                max_iterations: 5000,
                variant: CgVariant::PolakRibierePlus,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(pr.converged, "PR+ status: {:?}", pr.status);
        println!(
            "Rosenbrock: HZ iters={} f_evals={} g_evals={} | PR+ iters={} f_evals={} g_evals={}",
            hz.iterations, hz.fun_evals, hz.grad_evals, pr.iterations, pr.fun_evals, pr.grad_evals
        );
    }

    #[test]
    fn test_cg_hager_zhang_explicit_line_search() {
        // Explicitly selecting LineSearchMethod::HagerZhang must work for
        // any beta variant.
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0, -2.0];
        let config = NonlinearCgConfig {
            line_search: LineSearchMethod::HagerZhang,
            ..Default::default()
        };
        let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_cg_determinism() {
        // Two identical runs must produce bit-identical iterates.
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
        for variant in [CgVariant::HagerZhang, CgVariant::PolakRibierePlus] {
            let config = NonlinearCgConfig {
                max_iterations: 5000,
                variant,
                ..Default::default()
            };
            let r1 = nonlinear_cg(&f, &g, &x0, &config).unwrap();
            let r2 = nonlinear_cg(&f, &g, &x0, &config).unwrap();
            assert_eq!(r1.x, r2.x, "variant {variant:?} not deterministic");
            assert_eq!(r1.iterations, r2.iterations);
            assert_eq!(r1.fun_evals, r2.fun_evals);
        }
    }

    #[test]
    fn test_cg_restart_period() {
        // Verify that with restart_period=2, the solver still converges on the sphere.
        // This exercises the restart path (beta set to zero every 2 iterations).
        let f = |x: &[f64]| x.dot(x);
        let g = |x: &[f64]| x.iter().map(|&xi| 2.0 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0, 3.0, -2.0];
        let config = NonlinearCgConfig {
            restart_period: 2,
            ..Default::default()
        };
        let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
        assert!(
            result.converged,
            "CG with restart_period=2 failed: {:?}",
            result.status
        );
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_cg_sphere_f32() {
        let f = |x: &[f32]| x.dot(x);
        let g = |x: &[f32]| x.iter().map(|&xi| 2.0_f32 * xi).collect::<Vec<_>>();
        let x0 = sonium_solvers_core::array![5.0_f32, 3.0_f32];
        let config: NonlinearCgConfig<f32> = NonlinearCgConfig::default();
        let result = nonlinear_cg(&f, &g, &x0, &config).unwrap();
        assert!(result.converged, "status: {:?}", result.status);
        assert!(result.fun < 1e-2, "fun={}", result.fun);
        assert!(result.x[0].abs() < 1e-3, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1]={}", result.x[1]);
    }
}
