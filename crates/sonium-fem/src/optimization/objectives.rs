//! Objective functions and their sensitivities

/// An objective function that can compute its value and gradient.
///
/// The gradient is computed via the adjoint method: one forward solve + one adjoint solve,
/// regardless of the number of design variables.
pub trait Objective: Send + Sync {
    /// Evaluate the objective function value given the current state.
    fn evaluate(&self, state: &ObjectiveState) -> f64;

    /// Compute the sensitivity dF/dp for each design variable.
    ///
    /// Uses the adjoint method:
    /// 1. Solve forward: `K(p) u = f` → get `u`
    /// 2. Solve adjoint: `K(p)^T lambda = -dF/du` → get `lambda`
    /// 3. Sensitivity: `dF/dp_i = lambda^T (dK/dp_i) u + dF/dp_i|_explicit`
    fn sensitivity(&self, state: &ObjectiveState) -> Vec<f64>;
}

/// State data passed to objective functions.
#[derive(Debug, Clone)]
pub struct ObjectiveState {
    /// Design variable values (densities, shape params, etc.).
    pub design_vars: Vec<f64>,
    /// Forward solution (displacement, pressure, potential, etc.).
    pub solution: Vec<f64>,
    /// Adjoint solution (computed by the adjoint solver).
    pub adjoint: Option<Vec<f64>>,
    /// Element-wise quantities (stiffness contributions, etc.).
    pub element_data: Vec<f64>,
}

/// Structural compliance: `C = u^T K u = f^T u`
///
/// Minimizing compliance maximizes structural stiffness.
/// Sensitivity: `dC/dp_e = -p_e^{p-1} * u_e^T K_0_e u_e` (SIMP)
#[derive(Debug, Clone)]
pub struct ComplianceObjective;

impl Objective for ComplianceObjective {
    fn evaluate(&self, state: &ObjectiveState) -> f64 {
        // C = sum of element compliances
        state.element_data.iter().sum()
    }

    fn sensitivity(&self, state: &ObjectiveState) -> Vec<f64> {
        // For compliance, the adjoint is lambda = -u, so
        // dC/dp_e = -u_e^T (dK_e/dp_e) u_e
        // With SIMP: dK/dp = p * p^{p-1} * K_0 → dC/dp = -p * p^{p-1} * u_e^T K_0 u_e
        // The element_data stores u_e^T K_0_e u_e for each element.
        state
            .element_data
            .iter()
            .zip(state.design_vars.iter())
            .map(|(&ce, &rho)| -ce / rho.max(1e-9))
            .collect()
    }
}

/// Acoustic frequency response objective: minimize `|p(omega) - p_target(omega)|^2`
///
/// For loudspeaker design: flatten the frequency response in a target band.
#[derive(Debug, Clone)]
pub struct FrequencyResponseObjective {
    /// Target pressure at each frequency (complex magnitude).
    pub target: Vec<f64>,
    /// Frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Frequency weights (importance of each frequency).
    pub weights: Vec<f64>,
}

impl FrequencyResponseObjective {
    pub fn flat_response(frequencies: Vec<f64>, target_spl_db: f64) -> Self {
        let p_ref = 20e-6; // 20 µPa reference
        let target_pa = p_ref * 10.0_f64.powf(target_spl_db / 20.0);
        let n = frequencies.len();
        Self {
            target: vec![target_pa; n],
            frequencies,
            weights: vec![1.0; n],
        }
    }
}

impl Objective for FrequencyResponseObjective {
    fn evaluate(&self, state: &ObjectiveState) -> f64 {
        // Sum of weighted squared errors at each frequency
        state
            .element_data
            .iter()
            .zip(self.target.iter())
            .zip(self.weights.iter())
            .map(|((&actual, &target), &w)| w * (actual - target).powi(2))
            .sum()
    }

    fn sensitivity(&self, state: &ObjectiveState) -> Vec<f64> {
        // Placeholder: actual sensitivity requires adjoint solve at each frequency
        vec![0.0; state.design_vars.len()]
    }
}

/// Magnetic BL maximization objective.
///
/// Maximize `BL(x=0)` — the force factor at rest position.
#[derive(Debug, Clone)]
pub struct MaxBLObjective;

impl Objective for MaxBLObjective {
    fn evaluate(&self, state: &ObjectiveState) -> f64 {
        // BL is stored as the first element_data entry
        -state.element_data.first().copied().unwrap_or(0.0) // negate for minimization
    }

    fn sensitivity(&self, state: &ObjectiveState) -> Vec<f64> {
        // dBL/dp from adjoint of the magnetostatic solve
        // Placeholder: requires adjoint implementation for magnetostatics
        vec![0.0; state.design_vars.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_sensitivity_sign() {
        let state = ObjectiveState {
            design_vars: vec![0.5, 0.8, 1.0],
            solution: vec![],
            adjoint: None,
            element_data: vec![10.0, 20.0, 30.0], // element compliances
        };

        let obj = ComplianceObjective;
        let sens = obj.sensitivity(&state);

        // Sensitivity should be negative (increasing density decreases compliance)
        for &s in &sens {
            assert!(s < 0.0, "Compliance sensitivity should be negative");
        }
    }
}
