//! Adjoint-based differentiation bridge for FEM optimization.
//!
//! Wraps FEM assembly and solve operations as differentiable functions whose
//! gradient is computed by the analytical adjoint, so topology / shape
//! optimisation can call back into FEM solves and pull `dF/dp` out.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::optimization::autograd::*;
//!
//! let mut rho = DiffDesignVars::new(initial_densities);
//! let obj = DiffFemObjective::new(forward_fn, gradient_fn);
//! let value = obj.evaluate(&rho);
//! obj.backward(&mut rho);
//! ```

/// Differentiable design variables wrapper.
///
/// Wraps a vector of design parameters with autograd tracking so that
/// gradients can be computed through FEM assembly and solve operations.
#[derive(Debug, Clone)]
pub struct DiffDesignVars {
    /// Current values.
    pub values: Vec<f64>,
    /// Accumulated gradients (populated after backward pass).
    pub gradients: Option<Vec<f64>>,
}

impl DiffDesignVars {
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            gradients: None,
        }
    }

    /// Number of design variables.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Differentiable FEM objective: wraps a forward solve + objective evaluation
/// as a function whose gradient can be computed via the adjoint method.
///
/// The gradient computation uses the analytical adjoint (not tape-based AD)
/// for the linear solve, combined with autograd for the parameter-to-matrix
/// mapping.
pub struct DiffFemObjective<F, G> {
    /// Forward solve: design_vars → objective_value
    pub forward: F,
    /// Gradient: design_vars → dF/dp (via adjoint)
    pub gradient: G,
}

impl<F, G> DiffFemObjective<F, G>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    pub fn new(forward: F, gradient: G) -> Self {
        Self { forward, gradient }
    }

    /// Evaluate the objective.
    pub fn evaluate(&self, vars: &DiffDesignVars) -> f64 {
        (self.forward)(&vars.values)
    }

    /// Compute gradients and store them in the design variables.
    pub fn backward(&self, vars: &mut DiffDesignVars) {
        vars.gradients = Some((self.gradient)(&vars.values));
    }
}

/// Differentiable SIMP interpolation.
///
/// Maps density `rho` to effective stiffness ratio and its derivative.
pub fn simp_interpolation(rho: f64, penalty: f64) -> (f64, f64) {
    let val = rho.powf(penalty);
    let deriv = penalty * rho.powf(penalty - 1.0);
    (val, deriv)
}

/// Chain rule helper: given dF/dK (sensitivity w.r.t. stiffness) and dK/dp
/// (stiffness w.r.t. design parameter), compute dF/dp.
///
/// This is the core operation that connects the adjoint-computed sensitivity
/// (w.r.t. assembled matrix) to the design parameter sensitivity.
pub fn chain_rule_sensitivity(
    df_dk_elements: &[f64], // dF/dK_e for each element (from adjoint)
    dk_dp_elements: &[f64], // dK_e/dp_e for each element (from SIMP)
) -> Vec<f64> {
    df_dk_elements
        .iter()
        .zip(dk_dp_elements)
        .map(|(&df_dk, &dk_dp)| df_dk * dk_dp)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_design_vars() {
        let vars = DiffDesignVars::new(vec![0.5, 0.8, 1.0]);
        assert_eq!(vars.len(), 3);
        assert!(vars.gradients.is_none());
    }

    #[test]
    fn test_diff_fem_objective() {
        let obj = DiffFemObjective::new(
            |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>(), // f = sum(x^2)
            |p: &[f64]| p.iter().map(|x| 2.0 * x).collect(),  // df/dx = 2x
        );

        let mut vars = DiffDesignVars::new(vec![1.0, 2.0, 3.0]);
        let val = obj.evaluate(&vars);
        assert!((val - 14.0).abs() < 1e-14); // 1 + 4 + 9

        obj.backward(&mut vars);
        let grad = vars.gradients.unwrap();
        assert!((grad[0] - 2.0).abs() < 1e-14);
        assert!((grad[1] - 4.0).abs() < 1e-14);
        assert!((grad[2] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_chain_rule() {
        let df_dk = vec![1.0, 2.0, 3.0];
        let dk_dp = vec![0.5, 0.5, 0.5];
        let df_dp = chain_rule_sensitivity(&df_dk, &dk_dp);
        assert!((df_dp[0] - 0.5).abs() < 1e-14);
        assert!((df_dp[1] - 1.0).abs() < 1e-14);
        assert!((df_dp[2] - 1.5).abs() < 1e-14);
    }

    #[test]
    fn test_simp_interpolation() {
        let (val, deriv) = simp_interpolation(0.5, 3.0);
        assert!((val - 0.125).abs() < 1e-14);
        assert!((deriv - 0.75).abs() < 1e-14); // 3 * 0.5^2
    }
}
