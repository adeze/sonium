//! Optimization constraints

/// A constraint `g(p) <= 0` (inequality) or `h(p) = 0` (equality).
pub trait Constraint: Send + Sync {
    /// Evaluate constraint value. Feasible when <= 0 for inequality.
    fn evaluate(&self, design_vars: &[f64]) -> f64;
    /// Constraint gradient dg/dp.
    fn gradient(&self, design_vars: &[f64]) -> Vec<f64>;
}

/// Volume fraction constraint: `sum(rho * V) / sum(V) <= vf_max`.
#[derive(Debug, Clone)]
pub struct VolumeConstraint {
    /// Maximum volume fraction.
    pub max_fraction: f64,
    /// Element volumes.
    pub element_volumes: Vec<f64>,
    /// Total volume.
    total_volume: f64,
}

impl VolumeConstraint {
    pub fn new(max_fraction: f64, element_volumes: Vec<f64>) -> Self {
        let total_volume: f64 = element_volumes.iter().sum();
        Self {
            max_fraction,
            element_volumes,
            total_volume,
        }
    }
}

impl Constraint for VolumeConstraint {
    fn evaluate(&self, design_vars: &[f64]) -> f64 {
        let vol: f64 = design_vars
            .iter()
            .zip(&self.element_volumes)
            .map(|(&rho, &v)| rho * v)
            .sum();
        vol / self.total_volume - self.max_fraction
    }

    fn gradient(&self, _design_vars: &[f64]) -> Vec<f64> {
        self.element_volumes
            .iter()
            .map(|&v| v / self.total_volume)
            .collect()
    }
}

/// Stress constraint: max von Mises stress <= sigma_max.
///
/// Uses p-norm aggregation to make it differentiable:
/// `g = (sum_e (sigma_e / sigma_max)^p)^{1/p} - 1 <= 0`
#[derive(Debug, Clone)]
pub struct StressConstraint {
    /// Maximum allowable stress.
    pub sigma_max: f64,
    /// P-norm exponent (higher = closer to max, typically 8-16).
    pub p_norm: f64,
}

impl StressConstraint {
    pub fn new(sigma_max: f64) -> Self {
        Self {
            sigma_max,
            p_norm: 8.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_constraint() {
        let volumes = vec![1.0; 4];
        let constraint = VolumeConstraint::new(0.5, volumes);

        // At rho = 0.3: vol_frac = 0.3 < 0.5 → feasible (g < 0)
        assert!(constraint.evaluate(&[0.3, 0.3, 0.3, 0.3]) < 0.0);

        // At rho = 0.7: vol_frac = 0.7 > 0.5 → infeasible (g > 0)
        assert!(constraint.evaluate(&[0.7, 0.7, 0.7, 0.7]) > 0.0);

        // Gradient: V_e / V_total = 0.25 for each
        let grad = constraint.gradient(&[0.5; 4]);
        for &g in &grad {
            assert!((g - 0.25).abs() < 1e-14);
        }
    }
}
