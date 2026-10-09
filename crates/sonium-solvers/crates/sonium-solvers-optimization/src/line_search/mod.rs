//! Line search algorithms for gradient-based optimisation.

mod backtracking_impl;
mod brent_impl;
mod hager_zhang_impl;
mod strong_wolfe_impl;

use crate::types::RealScalar;
use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverResultContract, SolverStatus,
};

pub use backtracking_impl::{BacktrackingConfig, backtracking};
pub use brent_impl::{BrentConfig, BrentResult, BrentStatus, brent_minimize};
pub use hager_zhang_impl::{HagerZhangConfig, hager_zhang_line_search};
pub use strong_wolfe_impl::{StrongWolfeConfig, strong_wolfe};

/// Fill `out` with the trial point `x + alpha * d`, element-wise.
///
/// Line searches probe dozens of trial points per call; reusing one caller
/// buffer instead of allocating per trial removes that churn. The arithmetic
/// is identical to `vec_ops::vec_add_scaled(x, d, alpha)` (same per-element
/// operation order), so results are bit-for-bit equal. Callers must pass a
/// buffer of length `x.len()`; lengths are asserted, never silently
/// truncated.
pub(crate) fn write_trial_point<T: RealScalar>(x: &[T], d: &[T], alpha: T, out: &mut [T]) {
    assert_eq!(x.len(), d.len(), "vector length mismatch");
    assert_eq!(x.len(), out.len(), "vector length mismatch");
    for ((o, &xi), &di) in out.iter_mut().zip(x.iter()).zip(d.iter()) {
        *o = xi + alpha * di;
    }
}

impl<T: RealScalar> LineSearchResult<T> {
    /// Construct the canonical result for invalid or non-finite line-search
    /// inputs.  The line-search APIs intentionally return a result value
    /// (rather than `Result`) for compatibility, so a non-finite payload is
    /// how callers observe a typed [`SolverStatus::NumericalError`].
    pub(crate) fn numerical_failure() -> Self {
        Self {
            alpha: T::nan(),
            fun: T::nan(),
            grad: None,
            fun_evals: 0,
            grad_evals: 0,
            success: false,
        }
    }

    /// Return whether all scalar and optional gradient outputs are finite.
    pub fn is_finite(&self) -> bool {
        self.alpha.is_finite()
            && self.fun.is_finite()
            && self
                .grad
                .as_ref()
                .is_none_or(|gradient| gradient.iter().all(|value| value.is_finite()))
    }

    /// Return the canonical status for a completed line-search attempt.
    pub fn solver_status(&self) -> SolverStatus {
        if !self.is_finite() {
            SolverStatus::NumericalError
        } else if self.success {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// A successful line search satisfies its local acceptance conditions.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        match self.solver_status() {
            SolverStatus::Converged => AccuracyStatus::Attained,
            SolverStatus::MaxIterationsReached => AccuracyStatus::NotAttained,
            _ => AccuracyStatus::NotChecked,
        }
    }

    /// Alias matching the checked result vocabulary used by other families.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: RealScalar> SolverResultContract for LineSearchResult<T> {
    fn solver_status(&self) -> SolverStatus {
        LineSearchResult::solver_status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        LineSearchResult::accuracy_status(self)
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        if self.is_finite() {
            Ok(())
        } else {
            Err(SolverError::OperationFailed {
                operation: "line-search result contract",
                reason: "step, objective, or gradient is non-finite".into(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(success: bool, alpha: f64) -> LineSearchResult<f64> {
        LineSearchResult {
            alpha,
            fun: 1.0,
            grad: Some(vec![0.0]),
            fun_evals: 1,
            grad_evals: 1,
            success,
        }
    }

    #[test]
    fn line_search_result_reports_canonical_statuses() {
        let converged = result(true, 0.5);
        assert!(converged.is_finite());
        assert_eq!(converged.status(), SolverStatus::Converged);
        assert_eq!(converged.accuracy_status(), AccuracyStatus::Attained);

        let failed = result(false, 0.5);
        assert_eq!(failed.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(failed.accuracy_status(), AccuracyStatus::NotAttained);

        let non_finite = result(true, f64::NAN);
        assert!(!non_finite.is_finite());
        assert_eq!(non_finite.status(), SolverStatus::NumericalError);
        assert_eq!(non_finite.accuracy_status(), AccuracyStatus::NotChecked);
    }

    #[test]
    fn backtracking_rejects_nonfinite_configuration_as_numerical_failure() {
        let f = |x: &[f64]| 0.5 * x.iter().map(|value| value * value).sum::<f64>();
        let x = [1.0, 1.0];
        let direction = [-1.0, -1.0];
        let gradient = [1.0, 1.0];
        let mut config = BacktrackingConfig::default();
        config.c1 = f64::NAN;

        let result = backtracking(&f, &x, &direction, 1.0, &gradient, &config);

        assert_eq!(result.status(), SolverStatus::NumericalError);
        assert!(!result.success);
        assert_eq!(result.fun_evals, 0);
    }

    #[test]
    fn strong_wolfe_rejects_invalid_initial_state_as_numerical_failure() {
        let f = |x: &[f64]| 0.5 * x.iter().map(|value| value * value).sum::<f64>();
        let gradient_fn = |x: &[f64]| x.to_vec();
        let x = [1.0, 1.0];
        let direction = [-1.0, -1.0];
        let gradient = [1.0, 1.0];
        let config = StrongWolfeConfig {
            alpha_init: f64::NAN,
            ..StrongWolfeConfig::default()
        };

        let result = strong_wolfe(&f, &gradient_fn, &x, &direction, 1.0, &gradient, &config);

        assert_eq!(result.status(), SolverStatus::NumericalError);
        assert!(!result.success);
        assert_eq!(result.fun_evals, 0);
    }

    #[test]
    fn write_trial_point_matches_allocating_form_bitwise() {
        // The reused-buffer helper must be bit-for-bit equal to the
        // allocating form it replaces.
        let x = [1.5_f64, -2.0, 0.75, 3.25, -0.0];
        let d = [-0.5_f64, 1.0, 2.0, -1.25, 4.0];
        for alpha in [1.0, 0.03125, 1e-8, 123.456] {
            let mut buf = vec![f64::NAN; x.len()];
            super::write_trial_point(&x, &d, alpha, &mut buf);
            let expected = sonium_solvers_core::vec_ops::vec_add_scaled(&x, &d, alpha);
            assert_eq!(
                buf, expected,
                "trial buffer mismatch at alpha = {alpha} (stale or partial write)"
            );
        }
    }

    #[test]
    fn backtracking_probes_follow_the_geometric_step_sequence() {
        // Backtracking alphas are exactly alpha_init * rho^k: every recorded
        // probe must equal x + rho^k * d bitwise, proving the reused trial
        // buffer is fully overwritten per probe.
        use std::cell::RefCell;
        let x = [2.0_f64, 2.0];
        let d = [-400.0_f64, -400.0];
        let trials = RefCell::new(Vec::new());
        let f = |t: &[f64]| {
            trials.borrow_mut().push(t.to_vec());
            100.0 * t.iter().map(|v| v * v).sum::<f64>()
        };
        let g = [400.0_f64, 400.0];
        let f0 = 100.0 * (x[0] * x[0] + x[1] * x[1]);
        let config = BacktrackingConfig::default();
        let result = backtracking(&f, &x, &d, f0, &g, &config);
        assert!(result.success);
        let trials = trials.borrow();
        assert!(
            trials.len() >= 2,
            "fixture must force multiple probes, got {}",
            trials.len()
        );
        let mut alpha = config.alpha_init;
        for (k, t) in trials.iter().enumerate() {
            let mut expected = vec![0.0; x.len()];
            super::write_trial_point(&x, &d, alpha, &mut expected);
            assert_eq!(*t, expected, "probe {k} is not x + rho^{k} * d");
            alpha *= config.rho;
        }
    }

    /// Every trial point handed to `f`/`grad` must lie on the ray `x + αd`.
    /// A stale or partially overwritten trial buffer would fail this badly;
    /// rounding noise is ~1e-16 relative, the tolerance is 1e-9.
    fn assert_trials_on_ray(x: &[f64], d: &[f64], trials: &[Vec<f64>]) {
        assert!(
            trials.len() >= 2,
            "fixture must force multiple probes, got {}",
            trials.len()
        );
        let j = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(j, _)| j)
            .unwrap();
        for t in trials {
            let alpha = (t[j] - x[j]) / d[j];
            for i in 0..x.len() {
                let expected = x[i] + alpha * d[i];
                let tol = 1e-9 * (1.0 + t[i].abs());
                assert!(
                    (t[i] - expected).abs() <= tol,
                    "trial {t:?} is off the ray x + αd (recovered α = {alpha})"
                );
            }
        }
    }

    #[test]
    fn strong_wolfe_zoom_probes_stay_on_search_ray() {
        // phi(a) = exp(0.3a) - 2a with tight c2 forces the zoom phase
        // (same fixture as the interpolation test), exercising the shared
        // bracketing/zoom trial buffer across many probes.
        use std::cell::RefCell;
        let trials = RefCell::new(Vec::new());
        let f = |t: &[f64]| {
            trials.borrow_mut().push(t.to_vec());
            (0.3 * t[0]).exp() - 2.0 * t[0]
        };
        let g_trials = RefCell::new(Vec::new());
        let grad = |t: &[f64]| {
            g_trials.borrow_mut().push(t.to_vec());
            vec![0.3 * (0.3 * t[0]).exp() - 2.0, 0.0]
        };
        let x = [0.0_f64, 0.0];
        let d = [1.0_f64, 1.0];
        let f0 = f(&x);
        let grad0 = vec![0.3 * (0.3 * x[0]).exp() - 2.0, 0.0];
        let config = StrongWolfeConfig {
            c2: 0.1,
            ..StrongWolfeConfig::default()
        };
        let result = strong_wolfe(&f, &grad, &x, &d, f0, &grad0, &config);
        assert!(result.success);
        let mut all = trials.borrow().clone();
        all.extend(g_trials.borrow().iter().cloned());
        assert_trials_on_ray(&x, &d, &all);
    }

    #[test]
    fn hager_zhang_probes_stay_on_search_ray() {
        // Ill-conditioned quartic forces Hager-Zhang through expansion and
        // zoom, exercising the shared trial buffer across many probes.
        use std::cell::RefCell;
        let trials = RefCell::new(Vec::new());
        let f = |t: &[f64]| {
            trials.borrow_mut().push(t.to_vec());
            t[0].powi(4) + 100.0 * t[1] * t[1]
        };
        let g_trials = RefCell::new(Vec::new());
        let grad = |t: &[f64]| {
            g_trials.borrow_mut().push(t.to_vec());
            vec![4.0 * t[0].powi(3), 200.0 * t[1]]
        };
        let x = [2.0_f64, 1.0];
        let grad0 = grad(&x);
        let d = [-grad0[0], -grad0[1]];
        let f0 = f(&x);
        let config = HagerZhangConfig::default();
        let result = hager_zhang_line_search(&f, &grad, &x, &d, f0, &grad0, &config);
        assert!(result.success);
        let mut all = trials.borrow().clone();
        all.extend(g_trials.borrow().iter().cloned());
        assert_trials_on_ray(&x, &d, &all);
    }

    #[test]
    fn hager_zhang_rejects_invalid_curvature_configuration_as_numerical_failure() {
        let f = |x: &[f64]| 0.5 * x.iter().map(|value| value * value).sum::<f64>();
        let gradient_fn = |x: &[f64]| x.to_vec();
        let x = [1.0, 1.0];
        let direction = [-1.0, -1.0];
        let gradient = [1.0, 1.0];
        let config = HagerZhangConfig {
            sigma: 0.05,
            ..HagerZhangConfig::default()
        };

        let result =
            hager_zhang_line_search(&f, &gradient_fn, &x, &direction, 1.0, &gradient, &config);

        assert_eq!(result.status(), SolverStatus::NumericalError);
        assert!(!result.success);
        assert_eq!(result.fun_evals, 0);
    }
}

/// Result of a line search.
#[derive(Debug, Clone)]
pub struct LineSearchResult<T: RealScalar> {
    /// Step size alpha satisfying the chosen conditions.
    pub alpha: T,
    /// Function value at `x + alpha * d`.
    pub fun: T,
    /// Gradient at `x + alpha * d` (if computed).
    pub grad: Option<Vec<T>>,
    /// Number of function evaluations used.
    pub fun_evals: usize,
    /// Number of gradient evaluations used.
    pub grad_evals: usize,
    /// Whether the line search succeeded.
    pub success: bool,
}
