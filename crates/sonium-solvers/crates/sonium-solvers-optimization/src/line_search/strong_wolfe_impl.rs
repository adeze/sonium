//! More-Thuente line search satisfying strong Wolfe conditions.
//!
//! Finds alpha such that:
//!   (i)  f(x + alpha*d) <= f(x) + c1 * alpha * g^T d      (sufficient decrease)
//!   (ii) |g(x + alpha*d)^T d| <= c2 * |g(x)^T d|          (curvature condition)

use crate::line_search::LineSearchResult;
use crate::types::RealScalar;
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for the strong Wolfe line search.
#[derive(Debug, Clone)]
pub struct StrongWolfeConfig<T: RealScalar> {
    /// Sufficient decrease parameter (c1, typically 1e-4).
    pub c1: T,
    /// Curvature condition parameter (c2, typically 0.9 for quasi-Newton, 0.1 for CG).
    pub c2: T,
    /// Maximum number of function evaluations.
    pub max_fun_evals: usize,
    /// Initial step size.
    pub alpha_init: T,
    /// Maximum step size.
    pub alpha_max: T,
}

impl<T: RealScalar> Default for StrongWolfeConfig<T> {
    fn default() -> Self {
        Self {
            c1: T::from_f64(1e-4).unwrap(),
            c2: T::from_f64(0.9).unwrap(),
            max_fun_evals: 40,
            alpha_init: T::one(),
            alpha_max: T::from_f64(1e10).unwrap(),
        }
    }
}

/// Strong Wolfe line search (More-Thuente style bracket-zoom).
///
/// Requires both objective and gradient.
pub fn strong_wolfe<T, F, G>(
    f: &F,
    grad_fn: &G,
    x: &[T],
    d: &[T],
    fun0: T,
    grad0: &[T],
    config: &StrongWolfeConfig<T>,
) -> LineSearchResult<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    if x.len() != d.len()
        || x.len() != grad0.len()
        || !fun0.is_finite()
        || x.iter().any(|value| !value.is_finite())
        || d.iter().any(|value| !value.is_finite())
        || grad0.iter().any(|value| !value.is_finite())
        || !config.c1.is_finite()
        || config.c1 <= T::zero()
        || config.c1 >= T::one()
        || !config.c2.is_finite()
        || config.c2 <= config.c1
        || config.c2 >= T::one()
        || !config.alpha_init.is_finite()
        || config.alpha_init <= T::zero()
        || !config.alpha_max.is_finite()
        || config.alpha_max < config.alpha_init
    {
        return LineSearchResult::numerical_failure();
    }

    let dg0 = T::vec_dot(grad0, d);
    if !dg0.is_finite() {
        return LineSearchResult::numerical_failure();
    }
    let two = T::from_f64(2.0).unwrap();

    // Phase 1: Bracketing — find an interval [alpha_lo, alpha_hi] containing
    // a step satisfying the strong Wolfe conditions.
    let mut alpha_prev = T::zero();
    let mut f_prev = fun0;
    let mut alpha = config.alpha_init;
    let mut fun_evals = 0_usize;
    let mut grad_evals = 0_usize;
    // One trial buffer shared by bracketing and zoom (bit-identical to
    // per-trial allocation; see `write_trial_point`).
    let mut x_new = vec![T::zero(); x.len()];

    let max_bracket_iters = config.max_fun_evals / 2;

    for i in 0..max_bracket_iters {
        super::write_trial_point(x, d, alpha, &mut x_new);
        let f_new = f(&x_new);
        fun_evals += 1;
        if !f_new.is_finite() {
            return LineSearchResult::numerical_failure();
        }

        // Check Armijo violation or non-descent from previous
        if f_new > fun0 + config.c1 * alpha * dg0 || (i > 0 && f_new >= f_prev) {
            return zoom(
                f, grad_fn, x, d, fun0, dg0, alpha_prev, f_prev, None, alpha, f_new, None, config,
                fun_evals, grad_evals, &mut x_new,
            );
        }

        let g_new = grad_fn(&x_new);
        grad_evals += 1;
        if g_new.len() != grad0.len() || g_new.iter().any(|value| !value.is_finite()) {
            return LineSearchResult::numerical_failure();
        }
        let dg_new = T::vec_dot(&g_new, d);
        if !dg_new.is_finite() {
            return LineSearchResult::numerical_failure();
        }

        // Strong Wolfe curvature satisfied
        if dg_new.abs() <= config.c2 * dg0.abs() {
            return LineSearchResult {
                alpha,
                fun: f_new,
                grad: Some(g_new),
                fun_evals,
                grad_evals,
                success: true,
            };
        }

        // If derivative is non-negative, we've passed the minimum
        if dg_new >= T::zero() {
            return zoom(
                f,
                grad_fn,
                x,
                d,
                fun0,
                dg0,
                alpha,
                f_new,
                Some(dg_new),
                alpha_prev,
                f_prev,
                None,
                config,
                fun_evals,
                grad_evals,
                &mut x_new,
            );
        }

        alpha_prev = alpha;
        f_prev = f_new;
        alpha = T::min(alpha * two, config.alpha_max);
    }

    // Bracketing failed — return best step found
    super::write_trial_point(x, d, alpha, &mut x_new);
    let f_new = f(&x_new);
    let g_new = grad_fn(&x_new);
    if !f_new.is_finite()
        || g_new.len() != grad0.len()
        || g_new.iter().any(|value| !value.is_finite())
    {
        return LineSearchResult::numerical_failure();
    }
    LineSearchResult {
        alpha,
        fun: f_new,
        grad: Some(g_new),
        fun_evals: fun_evals + 1,
        grad_evals: grad_evals + 1,
        success: false,
    }
}

/// Phase 2: Zoom — shrink the bracket to find a point satisfying strong Wolfe.
///
/// Trial steps use safeguarded cubic interpolation (More-Thuente / Nocedal &
/// Wright eq. 3.59) when directional derivatives are known at both ends,
/// quadratic interpolation when only the low end's derivative is known, and
/// bisection as a fallback. Every trial is clamped into the central 80% of the
/// bracket so the interval shrinks by at least 10% per iteration.
#[allow(clippy::too_many_arguments)]
fn zoom<T, F, G>(
    f: &F,
    grad_fn: &G,
    x: &[T],
    d: &[T],
    fun0: T,
    dg0: T,
    mut alpha_lo: T,
    mut f_lo: T,
    mut dg_lo: Option<T>,
    mut alpha_hi: T,
    mut f_hi: T,
    mut dg_hi: Option<T>,
    config: &StrongWolfeConfig<T>,
    mut fun_evals: usize,
    mut grad_evals: usize,
    x_new: &mut [T],
) -> LineSearchResult<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
    G: Fn(&[T]) -> Vec<T>,
{
    let two = T::from_f64(2.0).unwrap();
    let max_zoom_iters = config.max_fun_evals.saturating_sub(fun_evals + grad_evals);

    for _ in 0..max_zoom_iters {
        let width = (alpha_hi - alpha_lo).abs();
        if width < T::machine_epsilon() {
            break;
        }
        let tenth = T::from_f64(0.1).unwrap();
        let lo_min = T::min(alpha_lo, alpha_hi);
        let hi_max = T::max(alpha_lo, alpha_hi);
        let guard_lo = lo_min + tenth * width;
        let guard_hi = hi_max - tenth * width;

        // Safeguarded interpolation with bisection fallback.
        let mid = (alpha_lo + alpha_hi) / two;
        let alpha_j = match interpolate(alpha_lo, f_lo, dg_lo, alpha_hi, f_hi, dg_hi) {
            Some(a) if a > guard_lo && a < guard_hi => a,
            _ => mid,
        };

        super::write_trial_point(x, d, alpha_j, x_new);
        let f_j = f(x_new);
        fun_evals += 1;
        if !f_j.is_finite() {
            return LineSearchResult::numerical_failure();
        }

        if f_j > fun0 + config.c1 * alpha_j * dg0 || f_j >= f_lo {
            alpha_hi = alpha_j;
            f_hi = f_j;
            dg_hi = None; // derivative at the new hi is unknown
        } else {
            let g_j = grad_fn(x_new);
            grad_evals += 1;
            if g_j.len() != d.len() || g_j.iter().any(|value| !value.is_finite()) {
                return LineSearchResult::numerical_failure();
            }
            let dg_j = T::vec_dot(&g_j, d);
            if !dg_j.is_finite() {
                return LineSearchResult::numerical_failure();
            }

            if dg_j.abs() <= config.c2 * dg0.abs() {
                return LineSearchResult {
                    alpha: alpha_j,
                    fun: f_j,
                    grad: Some(g_j),
                    fun_evals,
                    grad_evals,
                    success: true,
                };
            }

            if dg_j * (alpha_hi - alpha_lo) >= T::zero() {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
                dg_hi = dg_lo;
            }

            alpha_lo = alpha_j;
            f_lo = f_j;
            dg_lo = Some(dg_j);
        }

        // Convergence on bracket width
        if (alpha_hi - alpha_lo).abs() < T::machine_epsilon() * alpha_lo.abs() {
            break;
        }
    }

    // Return the best we found
    super::write_trial_point(x, d, alpha_lo, x_new);
    let f_best = f(x_new);
    let g_best = grad_fn(x_new);
    LineSearchResult {
        alpha: alpha_lo,
        fun: f_best,
        grad: Some(g_best),
        fun_evals: fun_evals + 1,
        grad_evals: grad_evals + 1,
        success: false,
    }
}

/// Interpolated trial step inside the bracket.
///
/// Cubic interpolation (Nocedal & Wright eq. 3.59) when the directional
/// derivative is known at both ends, quadratic interpolation (eq. 3.57) when
/// only the low end's derivative is known, `None` (use bisection) otherwise or
/// when the interpolation is degenerate.
fn interpolate<T: RealScalar>(
    a_lo: T,
    f_lo: T,
    dg_lo: Option<T>,
    a_hi: T,
    f_hi: T,
    dg_hi: Option<T>,
) -> Option<T> {
    let da = a_hi - a_lo;
    if da.abs() < T::machine_epsilon() {
        return None;
    }
    match (dg_lo, dg_hi) {
        (Some(dlo), Some(dhi)) => {
            let three = T::from_f64(3.0).unwrap();
            let two = T::from_f64(2.0).unwrap();
            let d1 = dlo + dhi - three * (f_lo - f_hi) / (a_lo - a_hi);
            let disc = d1 * d1 - dlo * dhi;
            if disc < T::zero() {
                return None;
            }
            let d2 = disc.sqrt() * (a_hi - a_lo).signum();
            let denom = dhi - dlo + two * d2;
            if denom.abs() < T::machine_epsilon() {
                return None;
            }
            Some(a_hi - (a_hi - a_lo) * (dhi + d2 - d1) / denom)
        }
        (Some(dlo), None) => {
            // Quadratic through (a_lo, f_lo) with slope dlo at a_lo and value
            // f_hi at a_hi.
            let two = T::from_f64(2.0).unwrap();
            let denom = two * (f_hi - f_lo - dlo * da);
            if denom.abs() < T::machine_epsilon() {
                return None;
            }
            Some(a_lo - dlo * da * da / denom)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_strong_wolfe_quadratic() {
        // f(x) = 0.5 * x^T x
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let g = |x: &[f64]| x.to_vec();

        let x = sonium_solvers_core::array![2.0, 2.0];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config = StrongWolfeConfig::default();
        let result = strong_wolfe(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.alpha > 0.0);

        // Verify Wolfe conditions
        let dg0 = sonium_solvers_core::traits::ComplexField::vec_dot(&grad, &d);
        assert!(result.fun <= f0 + config.c1 * result.alpha * dg0);
        if let Some(ref g_new) = result.grad {
            let dg_new = sonium_solvers_core::traits::ComplexField::vec_dot(g_new, &d);
            assert!(dg_new.abs() <= config.c2 * dg0.abs());
        }
    }

    #[test]
    fn test_strong_wolfe_exact_step_quadratic() {
        // For f(x) = 0.5*x^T*x, steepest descent with alpha=1 is exact
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let g = |x: &[f64]| x.to_vec();

        let x = sonium_solvers_core::array![1.0, 1.0];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config = StrongWolfeConfig::default();
        let result = strong_wolfe(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert_relative_eq!(result.alpha, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_strong_wolfe_rosenbrock_direction() {
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

        let x = sonium_solvers_core::array![-1.2, 1.0];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config = StrongWolfeConfig::default();
        let result = strong_wolfe(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.fun < f0);
    }

    #[test]
    fn test_strong_wolfe_wolfe_conditions_verified() {
        // After line search, explicitly verify both Armijo and curvature conditions
        // using a non-trivial function (Rosenbrock along steepest descent)
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

        let x = sonium_solvers_core::array![-1.2, 1.0];
        let grad0 = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad0);
        let f0 = f(&x);
        let dg0 = sonium_solvers_core::traits::ComplexField::vec_dot(&grad0, &d);

        let config = StrongWolfeConfig::default();
        let result = strong_wolfe(&f, &g, &x, &d, f0, &grad0, &config);

        assert!(result.success, "Line search did not succeed");

        // (i) Armijo sufficient decrease condition:
        //     f(x + alpha*d) <= f(x) + c1 * alpha * g0^T d
        let armijo_bound = f0 + config.c1 * result.alpha * dg0;
        assert!(
            result.fun <= armijo_bound + 1e-14,
            "Armijo violated: f_new={} > bound={}",
            result.fun,
            armijo_bound,
        );

        // (ii) Strong Wolfe curvature condition:
        //     |g(x + alpha*d)^T d| <= c2 * |g0^T d|
        let grad_new = result.grad.expect("Strong Wolfe should return gradient");
        let dg_new = sonium_solvers_core::traits::ComplexField::vec_dot(&grad_new, &d);
        let curvature_bound = config.c2 * dg0.abs();
        assert!(
            dg_new.abs() <= curvature_bound + 1e-14,
            "Curvature violated: |dg_new|={} > c2*|dg0|={}",
            dg_new.abs(),
            curvature_bound,
        );
    }

    #[test]
    fn test_strong_wolfe_quadratic_f32() {
        let f = |x: &[f32]| 0.5_f32 * x.dot(x);
        let g = |x: &[f32]| x.to_vec();

        let x = sonium_solvers_core::array![2.0_f32, 2.0_f32];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config: StrongWolfeConfig<f32> = StrongWolfeConfig::default();
        let result = strong_wolfe(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.alpha > 0.0_f32);
        assert!(result.fun < f0);
    }

    /// Reference line search with a bisection-only zoom, used to quantify the
    /// evaluation savings of the interpolated zoom. Mirrors `strong_wolfe`'s
    /// bracketing phase exactly.
    fn strong_wolfe_bisection_zoom(
        f: &dyn Fn(&[f64]) -> f64,
        grad_fn: &dyn Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
        d: &[f64],
        fun0: f64,
        grad0: &[f64],
        config: &StrongWolfeConfig<f64>,
    ) -> LineSearchResult<f64> {
        use sonium_solvers_core::traits::ComplexField;
        let dg0 = f64::vec_dot(grad0, d);
        let mut fun_evals = 0_usize;
        let mut grad_evals = 0_usize;
        let mut alpha_prev = 0.0;
        let mut f_prev = fun0;
        let mut alpha = config.alpha_init;

        for i in 0..config.max_fun_evals / 2 {
            let x_new = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, alpha);
            let f_new = f(&x_new);
            fun_evals += 1;
            let armijo_violated = f_new > fun0 + config.c1 * alpha * dg0;
            let non_descent = i > 0 && f_new >= f_prev;

            let mut dg_new = None;
            if !armijo_violated && !non_descent {
                let g_new = grad_fn(&x_new);
                grad_evals += 1;
                let dg = f64::vec_dot(&g_new, d);
                if dg.abs() <= config.c2 * dg0.abs() {
                    return LineSearchResult {
                        alpha,
                        fun: f_new,
                        grad: Some(g_new),
                        fun_evals,
                        grad_evals,
                        success: true,
                    };
                }
                dg_new = Some(dg);
            }

            if armijo_violated || non_descent || dg_new.is_some_and(|dg| dg >= 0.0) {
                // Bisection-only zoom over [alpha_prev, alpha] (order by value).
                let (mut lo, mut hi) = if armijo_violated || non_descent {
                    (alpha_prev, alpha)
                } else {
                    (alpha, alpha_prev)
                };
                let mut f_lo = if armijo_violated || non_descent {
                    f_prev
                } else {
                    f_new
                };
                for _ in 0..config.max_fun_evals {
                    let mid = 0.5 * (lo + hi);
                    let x_m = sonium_solvers_core::vec_ops::vec_add_scaled(x, d, mid);
                    let f_m = f(&x_m);
                    fun_evals += 1;
                    if f_m > fun0 + config.c1 * mid * dg0 || f_m >= f_lo {
                        hi = mid;
                    } else {
                        let g_m = grad_fn(&x_m);
                        grad_evals += 1;
                        let dg_m = f64::vec_dot(&g_m, d);
                        if dg_m.abs() <= config.c2 * dg0.abs() {
                            return LineSearchResult {
                                alpha: mid,
                                fun: f_m,
                                grad: Some(g_m),
                                fun_evals,
                                grad_evals,
                                success: true,
                            };
                        }
                        if dg_m * (hi - lo) >= 0.0 {
                            hi = lo;
                        }
                        lo = mid;
                        f_lo = f_m;
                    }
                    if (hi - lo).abs() < 1e-16 * lo.abs().max(1.0) {
                        break;
                    }
                }
                return LineSearchResult {
                    alpha: lo,
                    fun: f_lo,
                    grad: None,
                    fun_evals,
                    grad_evals,
                    success: false,
                };
            }

            alpha_prev = alpha;
            f_prev = f_new;
            alpha = (alpha * 2.0).min(config.alpha_max);
        }
        LineSearchResult {
            alpha,
            fun: f_prev,
            grad: None,
            fun_evals,
            grad_evals,
            success: false,
        }
    }

    #[test]
    fn test_zoom_interpolation_fewer_evals_than_bisection() {
        // Smooth 1-D merit phi(a) = exp(0.3 a) - 2 a, minimum at a ~ 6.32.
        // With a tight curvature parameter (c2 = 0.1) the bracketing phase
        // cannot satisfy strong Wolfe, forcing the zoom phase. Cubic/quadratic
        // interpolation must locate an acceptable step with strictly fewer
        // evaluations than pure bisection.
        let phi = |a: f64| (0.3 * a).exp() - 2.0 * a;
        let dphi = |a: f64| 0.3 * (0.3 * a).exp() - 2.0;

        let f = |x: &[f64]| phi(x[0]);
        let g = move |x: &[f64]| sonium_solvers_core::array![dphi(x[0])];

        let x = sonium_solvers_core::array![0.0_f64];
        let d = sonium_solvers_core::array![1.0_f64];
        let grad0 = g(&x);
        let f0 = f(&x);

        let config = StrongWolfeConfig {
            c2: 0.1,
            ..StrongWolfeConfig::default()
        };

        let interp = strong_wolfe(&f, &g, &x, &d, f0, &grad0, &config);
        assert!(interp.success, "interpolated zoom failed: {interp:?}");

        let bisect = strong_wolfe_bisection_zoom(&f, &g, &x, &d, f0, &grad0, &config);
        assert!(bisect.success, "bisection zoom failed: {bisect:?}");

        let interp_total = interp.fun_evals + interp.grad_evals;
        let bisect_total = bisect.fun_evals + bisect.grad_evals;
        assert!(
            interp_total < bisect_total,
            "interpolation should use fewer evals: interp {interp_total} vs bisection {bisect_total}"
        );

        // The accepted step must satisfy both strong Wolfe conditions.
        let dg0 = sonium_solvers_core::traits::ComplexField::vec_dot(&grad0, &d);
        let armijo_bound = f0 + config.c1 * interp.alpha * dg0;
        assert!(interp.fun <= armijo_bound + 1e-14);
        let g_new = interp.grad.expect("gradient");
        let dg_new = sonium_solvers_core::traits::ComplexField::vec_dot(&g_new, &d);
        assert!(dg_new.abs() <= config.c2 * dg0.abs() + 1e-14);
        // And it should be close to the true 1-D minimiser.
        assert_relative_eq!(interp.alpha, 6.3218, epsilon = 0.5);
    }
}
