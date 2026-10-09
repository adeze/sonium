//! Hager–Zhang (2005) line search.
//!
//! Implements the core of the scheme from Hager & Zhang, "A New Conjugate
//! Gradient Method with Guaranteed Descent and an Efficient Line Search",
//! SIAM J. Optim. 16(1), 2005 — the line search shipped with CG_DESCENT:
//!
//! - acceptance via the **approximate Wolfe conditions**
//!   `(2δ − 1) φ'(0) ≥ φ'(α) ≥ σ φ'(0)` together with the perturbation
//!   bound `φ(α) ≤ φ(0) + ε_k`, `ε_k = ε |f_k|`, with a fallback to the
//!   standard Wolfe conditions;
//! - a **bracketing phase** that geometrically expands the trial step. Each
//!   trial evaluates `f` first; the gradient is only evaluated when the
//!   function value alone cannot decide the outcome (i.e. when
//!   `φ(α) ≤ φ(0) + ε_k`). An `f`-only overshoot trial (`φ(α) > φ(0) +
//!   ε_k`, or non-finite `φ`) already brackets a stationary point —
//!   `φ'(a) < 0` at the left endpoint and the mean value theorem force a
//!   zero of `φ'` inside — so its gradient is never paid for;
//! - a **zoom phase** with the **secant² update** (two successive secant
//!   steps per iteration, the second taken on the bracket refined by the
//!   first) and the **U1–U3 update rules**: for a trial `c` in a bracket
//!   `[a, b]` with `φ'(a) < 0`,
//!   - U1: `φ'(c) ≥ 0`                          → `[a, c]`
//!   - U2: `φ'(c) < 0` and `φ(c) ≤ φ(0) + ε_k`  → `[c, b]`
//!   - U3: `φ'(c) < 0` and `φ(c) > φ(0) + ε_k`  → `[a, c]`
//!
//!   with bisection as safeguard whenever a secant step is not strictly
//!   interior, an endpoint derivative is unavailable (an `f`-only
//!   overshoot endpoint), or the bracket fails to shrink.
//!
//! What is *not* implemented relative to the full CG_DESCENT code: the
//! initial-step heuristics other than ψ0 (`ψ1/ψ2`, quadratic fit of the
//! first step — the caller supplies `alpha_init`; the ψ0 warm start,
//! reusing the previous accepted step as `alpha_init`, is done by
//! [`crate::unconstrained::nonlinear_cg`] for its
//! Hager–Zhang pairing), and the approximate-Wolfe perturbation tracking
//! of `ε_k` across iterates (here `ε_k` is recomputed per call from the
//! supplied `fun0`).

use crate::line_search::LineSearchResult;
use crate::types::RealScalar;
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for the Hager-Zhang line search.
#[derive(Debug, Clone)]
pub struct HagerZhangConfig<T: RealScalar> {
    /// Wolfe sufficient decrease (delta, default 0.1).
    pub delta: T,
    /// Wolfe curvature (sigma, default 0.9).
    pub sigma: T,
    /// Approximate Wolfe tolerance (epsilon, default 1e-6).
    pub epsilon: T,
    /// Bisection parameter (theta, default 0.5).
    pub theta: T,
    /// Expansion factor for initial bracket (rho, default 5.0).
    pub rho: T,
    /// Maximum function evaluations.
    pub max_fun_evals: usize,
    /// Initial step size.
    pub alpha_init: T,
}

impl<T: RealScalar> Default for HagerZhangConfig<T> {
    fn default() -> Self {
        Self {
            delta: T::from_f64(0.1).unwrap(),
            sigma: T::from_f64(0.9).unwrap(),
            epsilon: T::from_f64(1e-6).unwrap(),
            theta: T::from_f64(0.5).unwrap(),
            rho: T::from_f64(5.0).unwrap(),
            max_fun_evals: 50,
            alpha_init: T::one(),
        }
    }
}

/// Hager-Zhang line search.
///
/// Finds a step satisfying the approximate Wolfe conditions:
///   (AW1)  (2*delta - 1) * dg0 >= dg(alpha) >= sigma * dg0
///   (AW2)  f(alpha) <= f0 + epsilon_k
///
/// Falls back to standard Wolfe when the approximate conditions fail.
pub fn hager_zhang_line_search<T, F, G>(
    f: &F,
    grad_fn: &G,
    x: &[T],
    d: &[T],
    fun0: T,
    grad0: &[T],
    config: &HagerZhangConfig<T>,
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
        || !config.delta.is_finite()
        || config.delta <= T::zero()
        || config.delta >= T::from_f64(0.5).unwrap()
        || !config.sigma.is_finite()
        || config.sigma <= config.delta
        || config.sigma >= T::one()
        || !config.epsilon.is_finite()
        || config.epsilon < T::zero()
        || !config.theta.is_finite()
        || config.theta <= T::zero()
        || config.theta >= T::one()
        || !config.rho.is_finite()
        || config.rho <= T::one()
        || !config.alpha_init.is_finite()
        || config.alpha_init <= T::zero()
    {
        return LineSearchResult::numerical_failure();
    }

    let dg0 = T::vec_dot(grad0, d);
    if !dg0.is_finite() {
        return LineSearchResult::numerical_failure();
    }

    // Non-descent direction: nothing to search along.
    if dg0 >= T::zero() {
        return LineSearchResult {
            alpha: T::zero(),
            fun: fun0,
            grad: Some(grad0.to_vec()),
            fun_evals: 0,
            grad_evals: 0,
            success: false,
        };
    }

    let epsilon_k = config.epsilon * fun0.abs();
    let mut fun_evals = 0_usize;
    let mut grad_evals = 0_usize;
    // One trial buffer shared by expansion and zoom (bit-identical to
    // per-trial allocation; see `write_trial_point`).
    let mut x_new = vec![T::zero(); x.len()];

    // Phase 1: bracketing by geometric expansion. `a` is the left endpoint
    // of the developing bracket; its function value and derivative are
    // always known, with dphi(a) < 0.
    let mut a = T::zero();
    let mut fa = fun0;
    let mut dga = dg0;
    let mut alpha = config.alpha_init;

    let max_expansions = (config.max_fun_evals / 2).max(1);
    for _ in 0..max_expansions {
        // f-only probe first: an overshoot already brackets a stationary
        // point (dphi(a) < 0, phi(alpha) > phi(a) ≥ ... ⇒ φ' vanishes in
        // between by the mean value theorem), so the gradient at this
        // trial is never needed to establish the bracket.
        super::write_trial_point(x, d, alpha, &mut x_new);
        let f_alpha = f(&x_new);
        fun_evals += 1;

        if !f_alpha.is_finite() {
            return LineSearchResult::numerical_failure();
        }
        if f_alpha > fun0 + epsilon_k {
            // U3-type bracket: [a, alpha] with the upper endpoint known
            // only through its function value. Zoom handles the missing
            // derivative lazily.
            return hz_zoom(
                f,
                grad_fn,
                x,
                d,
                fun0,
                dg0,
                BracketEnd::with_derivative(a, fa, dga),
                BracketEnd::without_derivative(alpha, f_alpha),
                config,
                fun_evals,
                grad_evals,
                &mut x_new,
            );
        }

        // f does not rule the trial out: the gradient is needed both for
        // the Wolfe test and to decide the bracket.
        let g_alpha = grad_fn(&x_new);
        grad_evals += 1;
        if g_alpha.len() != grad0.len() || g_alpha.iter().any(|value| !value.is_finite()) {
            return LineSearchResult::numerical_failure();
        }
        let dg_alpha = T::vec_dot(&g_alpha, d);
        if !dg_alpha.is_finite() {
            return LineSearchResult::numerical_failure();
        }

        if satisfies_approx_wolfe(config, fun0, dg0, epsilon_k, f_alpha, dg_alpha)
            || satisfies_wolfe(config, fun0, dg0, alpha, f_alpha, dg_alpha)
        {
            return LineSearchResult {
                alpha,
                fun: f_alpha,
                grad: Some(g_alpha),
                fun_evals,
                grad_evals,
                success: true,
            };
        }

        if dg_alpha >= T::zero() {
            // U1-type bracket: [a, alpha] with dphi(a) < 0 <= dphi(alpha).
            return hz_zoom(
                f,
                grad_fn,
                x,
                d,
                fun0,
                dg0,
                BracketEnd::with_derivative(a, fa, dga),
                BracketEnd::with_derivative(alpha, f_alpha, dg_alpha),
                config,
                fun_evals,
                grad_evals,
                &mut x_new,
            );
        }

        // Still descending with an acceptable function value: expand.
        a = alpha;
        fa = f_alpha;
        dga = dg_alpha;
        alpha *= config.rho;
    }

    // Expansion exhausted while still descending (very flat direction or
    // unbounded-below along d). Return the last evaluated point.
    LineSearchResult {
        alpha: a,
        fun: fa,
        grad: None,
        fun_evals,
        grad_evals,
        success: false,
    }
}

/// Approximate Wolfe conditions (Hager–Zhang 2005, eq. (4.1)–(4.2)).
fn satisfies_approx_wolfe<T: RealScalar>(
    config: &HagerZhangConfig<T>,
    fun0: T,
    dg0: T,
    epsilon_k: T,
    fa: T,
    dga: T,
) -> bool {
    let two = T::from_f64(2.0).unwrap();
    (two * config.delta - T::one()) * dg0 >= dga
        && dga >= config.sigma * dg0
        && fa <= fun0 + epsilon_k
}

/// Standard Wolfe conditions (fallback).
fn satisfies_wolfe<T: RealScalar>(
    config: &HagerZhangConfig<T>,
    fun0: T,
    dg0: T,
    alpha: T,
    fa: T,
    dga: T,
) -> bool {
    fa <= fun0 + config.delta * alpha * dg0 && dga >= config.sigma * dg0
}

/// One end of a zoom bracket. The derivative may be unknown for an
/// endpoint that was only probed with `f` (overshoot trial); the zoom
/// phase falls back to bisection until the endpoint is replaced by a
/// trial whose derivative is known.
#[derive(Clone, Copy)]
struct BracketEnd<T: RealScalar> {
    alpha: T,
    fun: T,
    dfun: Option<T>,
}

impl<T: RealScalar> BracketEnd<T> {
    fn with_derivative(alpha: T, fun: T, dfun: T) -> Self {
        Self {
            alpha,
            fun,
            dfun: Some(dfun),
        }
    }
    fn without_derivative(alpha: T, fun: T) -> Self {
        Self {
            alpha,
            fun,
            dfun: None,
        }
    }
}

/// Zoom phase: secant² (two successive secant steps per iteration) with
/// the U1–U3 bracket update rules and bisection safeguards.
///
/// Bracket invariant: `a` has `dphi(a) < 0`; `b` either has
/// `dphi(b) >= 0` (U1) or `phi(b) > phi(0) + epsilon_k` (U3).
#[allow(clippy::too_many_arguments)]
fn hz_zoom<T, F, G>(
    f: &F,
    grad_fn: &G,
    x: &[T],
    d: &[T],
    fun0: T,
    dg0: T,
    mut a: BracketEnd<T>,
    mut b: BracketEnd<T>,
    config: &HagerZhangConfig<T>,
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
    let epsilon_k = config.epsilon * fun0.abs();
    // Track the best (lowest-f) trial for the failure fallback. Its
    // gradient is cached when it was computed.
    let mut best = a;
    if b.fun < best.fun {
        best = b;
    }

    let max_iters = config.max_fun_evals.saturating_sub(fun_evals + grad_evals);
    let mut prev_width = (b.alpha - a.alpha).abs();

    let mut iter = 0_usize;
    while iter < max_iters {
        // Secant²: two successive secant steps per zoom iteration, the
        // second taken on the bracket refined by the first. Each secant
        // trial is a genuine candidate, so acceptance is tested after
        // every step.
        for _ in 0..2 {
            if iter >= max_iters {
                break;
            }
            iter += 1;

            let (lo, hi) = if a.alpha < b.alpha {
                (a.alpha, b.alpha)
            } else {
                (b.alpha, a.alpha)
            };
            let width = hi - lo;

            // Trial point: secant on the endpoints when both derivatives
            // are known, bisection otherwise. Safeguarded to stay strictly
            // interior with a 10% margin.
            let margin = T::from_f64(0.1).unwrap() * width;
            let c = match (a.dfun, b.dfun) {
                (Some(dfa), Some(dfb)) => {
                    let denom = dfa - dfb;
                    let c_secant = if denom.abs() > T::machine_epsilon() {
                        a.alpha - dfa * (a.alpha - b.alpha) / denom
                    } else {
                        (a.alpha + b.alpha) / two
                    };
                    if c_secant <= lo + margin || c_secant >= hi - margin {
                        (a.alpha + b.alpha) / two
                    } else {
                        c_secant
                    }
                }
                _ => (a.alpha + b.alpha) / two,
            };

            super::write_trial_point(x, d, c, x_new);
            let fc = f(x_new);
            fun_evals += 1;
            if !fc.is_finite() {
                return LineSearchResult::numerical_failure();
            }
            // The U-rules and the Wolfe test both need the directional
            // derivative, so the gradient is always required at zoom
            // trials — unless f alone is enough to reject via U3, but the
            // U3 update keeps c's derivative, so compute it regardless.
            let gc = grad_fn(x_new);
            grad_evals += 1;
            if gc.len() != d.len() || gc.iter().any(|value| !value.is_finite()) {
                return LineSearchResult::numerical_failure();
            }
            let dgc = T::vec_dot(&gc, d);
            if !dgc.is_finite() {
                return LineSearchResult::numerical_failure();
            }

            if satisfies_approx_wolfe(config, fun0, dg0, epsilon_k, fc, dgc)
                || satisfies_wolfe(config, fun0, dg0, c, fc, dgc)
            {
                return LineSearchResult {
                    alpha: c,
                    fun: fc,
                    grad: Some(gc),
                    fun_evals,
                    grad_evals,
                    success: true,
                };
            }

            if fc < best.fun {
                best = BracketEnd::with_derivative(c, fc, dgc);
            }

            // U1–U3 bracket update.
            if dgc >= T::zero() {
                // U1: c is past the minimum — new upper endpoint.
                b = BracketEnd::with_derivative(c, fc, dgc);
            } else if fc <= fun0 + epsilon_k {
                // U2: still descending, acceptable f — new lower endpoint.
                a = BracketEnd::with_derivative(c, fc, dgc);
            } else {
                // U3: still descending but f too large — new upper
                // endpoint (its derivative stays negative).
                b = BracketEnd::with_derivative(c, fc, dgc);
            }

            let new_width = (b.alpha - a.alpha).abs();
            if new_width < T::machine_epsilon() * T::max(lo.abs().max(hi.abs()), T::one()) {
                break;
            }
        }

        // Safeguard: if a full secant² iteration failed to shrink the
        // bracket appreciably, drop one endpoint's derivative so the next
        // trial is a bisection.
        let width = (b.alpha - a.alpha).abs();
        if width > T::from_f64(0.9).unwrap() * prev_width {
            b.dfun = None;
        }
        prev_width = width;
    }

    // Failure: return the best point found (gradient only if cached).
    LineSearchResult {
        alpha: best.alpha,
        fun: best.fun,
        grad: None,
        fun_evals,
        grad_evals,
        success: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hager_zhang_quadratic() {
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let g = |x: &[f64]| x.to_vec();

        let x = sonium_solvers_core::array![2.0, 3.0];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.fun < f0);
    }

    #[test]
    fn test_hager_zhang_rosenbrock() {
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

        let config = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.fun < f0);
    }

    #[test]
    fn test_hager_zhang_quadratic_f32() {
        let f = |x: &[f32]| 0.5_f32 * x.dot(x);
        let g = |x: &[f32]| x.to_vec();

        let x = sonium_solvers_core::array![2.0_f32, 3.0_f32];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config: HagerZhangConfig<f32> = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);

        assert!(result.success);
        assert!(result.fun < f0);
    }

    #[test]
    fn test_hager_zhang_wolfe_on_ill_conditioned_quartic() {
        // f(x) = (x1^2 + 1000 x2^2)^2: a quartic whose Hessian at the
        // solution is singular and whose level sets are extremely
        // elongated (condition number 1e3 in the quadratic factor).
        let f = |x: &[f64]| {
            let q = x[0] * x[0] + 1000.0 * x[1] * x[1];
            q * q
        };
        let g = |x: &[f64]| {
            let q = x[0] * x[0] + 1000.0 * x[1] * x[1];
            sonium_solvers_core::array![4.0 * q * x[0], 4000.0 * q * x[1]]
        };

        let x = sonium_solvers_core::array![1.0, 0.05];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);
        let dg0: f64 = grad.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();

        let config = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);
        assert!(result.success, "line search failed on quartic");

        // Verify the Wolfe conditions at the returned step.
        let alpha = result.alpha;
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let f_new = f(&x_new);
        let g_new = g(&x_new);
        let dg_new: f64 = g_new.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();

        // Sufficient decrease: every acceptance path (approximate Wolfe
        // or standard Wolfe) implies f <= f0 + epsilon_k.
        let epsilon_k = config.epsilon * f0.abs();
        assert!(
            f_new <= f0 + epsilon_k + 1e-12,
            "decrease violated: f_new={f_new}, f0+eps={}",
            f0 + epsilon_k
        );
        // Curvature: both acceptance paths enforce dg >= sigma * dg0.
        assert!(
            dg_new >= config.sigma * dg0 - 1e-12,
            "curvature violated: dg_new={dg_new}, sigma*dg0={}",
            config.sigma * dg0
        );
    }

    #[test]
    fn test_hager_zhang_f_only_probes_skip_gradient() {
        // With an aggressively large initial step the first trial
        // overshoots (f > f0 + epsilon_k); the bracket is established
        // from f alone, so the overshoot trial must not pay for a
        // gradient evaluation.
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let g = |x: &[f64]| x.to_vec();

        let x = sonium_solvers_core::array![1.0, 1.0];
        let grad = g(&x);
        let d = sonium_solvers_core::vec_ops::vec_neg(&grad);
        let f0 = f(&x);

        let config = HagerZhangConfig {
            alpha_init: 10.0, // minimum of phi is at alpha = 1
            ..HagerZhangConfig::default()
        };
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);
        assert!(result.success);
        assert!(result.fun < f0);
        // One function evaluation at the overshoot trial, no gradient
        // there; the accepted point comes from zoom. Total gradient
        // evaluations must be strictly fewer than function evaluations.
        assert!(
            result.grad_evals < result.fun_evals,
            "gradient paid at f-only probes: f={}, g={}",
            result.fun_evals,
            result.grad_evals
        );
    }

    #[test]
    fn test_hager_zhang_non_descent_direction() {
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let g = |x: &[f64]| x.to_vec();
        let x = sonium_solvers_core::array![1.0, 1.0];
        let grad = g(&x);
        let d = grad.clone(); // ascent direction
        let f0 = f(&x);
        let config = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &g, &x, &d, f0, &grad, &config);
        assert!(!result.success);
        assert_eq!(result.fun_evals, 0);
        assert_eq!(result.grad_evals, 0);
    }
}
