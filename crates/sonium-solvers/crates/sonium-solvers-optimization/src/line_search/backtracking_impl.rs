//! Armijo backtracking line search.

use crate::line_search::LineSearchResult;
use crate::types::RealScalar;
#[cfg(test)]
use sonium_solvers_core::vec_ops::SliceDot;

/// Configuration for Armijo backtracking.
#[derive(Debug, Clone)]
pub struct BacktrackingConfig<T: RealScalar> {
    /// Sufficient decrease parameter (c1, typically 1e-4).
    pub c1: T,
    /// Step contraction factor (typically 0.5).
    pub rho: T,
    /// Maximum number of backtracking steps.
    pub max_steps: usize,
    /// Initial step size.
    pub alpha_init: T,
}

impl<T: RealScalar> Default for BacktrackingConfig<T> {
    fn default() -> Self {
        Self {
            c1: T::from_f64(1e-4).unwrap(),
            rho: T::from_f64(0.5).unwrap(),
            max_steps: 40,
            alpha_init: T::one(),
        }
    }
}

/// Armijo backtracking line search.
///
/// Finds `alpha` such that `f(x + alpha*d) <= f(x) + c1 * alpha * g^T d`.
pub fn backtracking<T, F>(
    f: &F,
    x: &[T],
    d: &[T],
    fun0: T,
    grad0: &[T],
    config: &BacktrackingConfig<T>,
) -> LineSearchResult<T>
where
    T: RealScalar,
    F: Fn(&[T]) -> T,
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
        || !config.rho.is_finite()
        || config.rho <= T::zero()
        || config.rho >= T::one()
        || !config.alpha_init.is_finite()
        || config.alpha_init <= T::zero()
    {
        return LineSearchResult::numerical_failure();
    }

    let dg0 = T::vec_dot(grad0, d);
    if !dg0.is_finite() {
        return LineSearchResult::numerical_failure();
    }
    let mut alpha = config.alpha_init;
    let mut fun_evals = 0;
    // One trial buffer for the whole search (bit-identical to per-trial
    // allocation; see `write_trial_point`).
    let mut x_new = vec![T::zero(); x.len()];

    for _ in 0..config.max_steps {
        super::write_trial_point(x, d, alpha, &mut x_new);
        let f_new = f(&x_new);
        fun_evals += 1;
        if !f_new.is_finite() {
            return LineSearchResult::numerical_failure();
        }

        // Armijo sufficient decrease
        if f_new <= fun0 + config.c1 * alpha * dg0 {
            return LineSearchResult {
                alpha,
                fun: f_new,
                grad: None,
                fun_evals,
                grad_evals: 0,
                success: true,
            };
        }

        alpha *= config.rho;
    }

    // Failed: return best alpha found
    super::write_trial_point(x, d, alpha, &mut x_new);
    let f_new = f(&x_new);
    fun_evals += 1;
    if !f_new.is_finite() {
        return LineSearchResult::numerical_failure();
    }

    LineSearchResult {
        alpha,
        fun: f_new,
        grad: None,
        fun_evals,
        grad_evals: 0,
        success: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_backtracking_quadratic() {
        // f(x) = 0.5 * (x0^2 + x1^2)
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let x = sonium_solvers_core::array![2.0, 2.0];
        let g = sonium_solvers_core::array![2.0, 2.0]; // grad at x
        let d = sonium_solvers_core::array![-2.0, -2.0]; // steepest descent
        let f0 = f(&x);

        let config = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);

        assert!(result.success);
        assert!(result.alpha > 0.0);
        // Verify Armijo condition
        let dg0 = sonium_solvers_core::traits::ComplexField::vec_dot(&g, &d);
        assert!(result.fun <= f0 + config.c1 * result.alpha * dg0);
    }

    #[test]
    fn test_backtracking_unit_step_for_quadratic() {
        // For a quadratic with Hessian = I, steepest descent with alpha=1 is exact
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let x = sonium_solvers_core::array![1.0, 1.0];
        let g = x.clone();
        let d = sonium_solvers_core::vec_ops::vec_neg(&g);
        let f0 = f(&x);

        let config = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);

        assert!(result.success);
        assert_relative_eq!(result.alpha, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_backtracking_steep_function() {
        // f(x) = 100 * x.dot(x), steep curvature forces multiple backtracks
        // when starting with alpha_init = 1
        let f = |x: &[f64]| 100.0 * x.dot(x);
        let x = sonium_solvers_core::array![2.0, 2.0];
        let g = sonium_solvers_core::array![400.0, 400.0]; // grad = 200 * x
        let d = sonium_solvers_core::vec_ops::vec_neg(&g); // steepest descent
        let f0 = f(&x);

        let config = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);

        assert!(result.success);
        // With such a large gradient direction, the initial unit step overshoots badly,
        // so the line search must backtrack multiple times
        assert!(
            result.fun_evals > 1,
            "Expected multiple backtracks, got fun_evals={}",
            result.fun_evals
        );
        // Verify the result is a descent
        assert!(result.fun < f0);
    }

    #[test]
    fn test_backtracking_quadratic_f32() {
        let f = |x: &[f32]| 0.5_f32 * x.dot(x);
        let x = sonium_solvers_core::array![2.0_f32, 2.0_f32];
        let g = sonium_solvers_core::array![2.0_f32, 2.0_f32];
        let d = sonium_solvers_core::array![-2.0_f32, -2.0_f32];
        let f0 = f(&x);

        let config: BacktrackingConfig<f32> = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);

        assert!(result.success);
        assert!(result.alpha > 0.0_f32);
        assert!(result.fun < f0);
    }

    #[test]
    fn test_backtracking_failure() {
        // Use an ascent direction (d = +gradient). The Armijo condition can never
        // be satisfied for c1 > 0 when d is an ascent direction, so the line
        // search should fail.
        let f = |x: &[f64]| 0.5 * x.dot(x);
        let x = sonium_solvers_core::array![2.0, 2.0];
        let g = sonium_solvers_core::array![2.0, 2.0]; // gradient at x
        let d = g.clone(); // ascent direction: d = +g
        let f0 = f(&x);

        let config = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);

        assert!(
            !result.success,
            "Expected line search failure for ascent direction, but got success with alpha={}",
            result.alpha
        );
    }
}
