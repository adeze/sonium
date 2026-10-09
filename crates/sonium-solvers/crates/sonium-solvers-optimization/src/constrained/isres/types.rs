use std::sync::Arc;

/// Erased inequality constraint closure: feasible when `<= 0`.
pub type IsresConstraintFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A single inequality constraint `fun(x) <= 0`.
#[derive(Clone)]
pub struct IsresConstraint {
    /// Constraint closure. Feasible when ≤ 0; positive values count as the
    /// violation magnitude used in stochastic ranking.
    pub fun: IsresConstraintFn,
}

#[derive(Clone)]
pub(super) struct Individual {
    pub(super) x: Vec<f64>,
    pub(super) sigma: Vec<f64>,
    pub(super) fun: f64,
    pub(super) max_violation: f64,
}

/// Maximum constraint violation across all constraints. Zero means feasible.
pub(super) fn max_violation_of(x: &[f64], constraints: &[IsresConstraint]) -> f64 {
    let mut worst = 0.0_f64;
    for c in constraints {
        let v = (c.fun)(x);
        if v > worst {
            worst = v;
        }
    }
    worst
}
