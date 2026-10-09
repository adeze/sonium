use super::deconfig::DEConfig;
use super::types::LinearPenalty;

/// SciPy-like linear constraint helper: lb <= A x <= ub.
#[derive(Debug, Clone)]
pub struct LinearConstraintHelper {
    /// Constraint matrix A (m x n).
    pub a: Vec<Vec<f64>>,
    /// Lower bounds vector (m elements).
    pub lb: Vec<f64>,
    /// Upper bounds vector (m elements).
    pub ub: Vec<f64>,
}

impl LinearConstraintHelper {
    /// Apply helper by merging into DEConfig.linear_penalty (stacking rows if already present)
    pub fn apply_to(&self, cfg: &mut DEConfig, weight: f64) {
        use super::stack_linear_penalty::stack_linear_penalty;

        let new_lp = LinearPenalty {
            a: self.a.clone(),
            lb: self.lb.clone(),
            ub: self.ub.clone(),
            weight,
        };
        match &mut cfg.linear_penalty {
            Some(existing) => stack_linear_penalty(existing, &new_lp),
            None => cfg.linear_penalty = Some(new_lp),
        }
    }
}
