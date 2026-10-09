//! Design variable representations for topology optimization
//!
//! SIMP (Solid Isotropic Material with Penalization) and RAMP interpolation
//! schemes for density-based topology optimization.

/// Material interpolation scheme for topology optimization.
#[derive(Debug, Clone, Copy)]
pub enum InterpolationScheme {
    /// SIMP: `E(rho) = E_min + rho^p * (E_0 - E_min)`
    Simp {
        /// Penalization exponent (typically 3).
        penalty: f64,
    },
    /// RAMP: `E(rho) = E_min + rho / (1 + q*(1-rho)) * (E_0 - E_min)`
    Ramp {
        /// Penalization parameter (typically 3).
        penalty: f64,
    },
}

impl InterpolationScheme {
    /// Standard SIMP with penalty p=3.
    pub fn simp() -> Self {
        Self::Simp { penalty: 3.0 }
    }

    /// SIMP with custom penalty.
    pub fn simp_with_penalty(p: f64) -> Self {
        Self::Simp { penalty: p }
    }

    /// Standard RAMP with q=3.
    pub fn ramp() -> Self {
        Self::Ramp { penalty: 3.0 }
    }

    /// Interpolated stiffness ratio: `E(rho) / E_0` (excluding E_min).
    pub fn interpolate(&self, rho: f64) -> f64 {
        match self {
            Self::Simp { penalty } => rho.powf(*penalty),
            Self::Ramp { penalty } => rho / (1.0 + penalty * (1.0 - rho)),
        }
    }

    /// Derivative of interpolation: `d/d(rho) [E(rho) / E_0]`.
    pub fn interpolate_derivative(&self, rho: f64) -> f64 {
        match self {
            Self::Simp { penalty } => penalty * rho.powf(penalty - 1.0),
            Self::Ramp { penalty } => (1.0 + penalty) / (1.0 + penalty * (1.0 - rho)).powi(2),
        }
    }
}

/// Element-wise design variables for topology optimization.
#[derive(Debug, Clone)]
pub struct DensityField {
    /// Density value per element (0 = void, 1 = solid).
    pub densities: Vec<f64>,
    /// Minimum density to avoid singularity.
    pub rho_min: f64,
    /// Interpolation scheme.
    pub scheme: InterpolationScheme,
    /// Volume fraction constraint target (0-1).
    pub target_volume_fraction: f64,
}

impl DensityField {
    /// Initialize with uniform density at the target volume fraction.
    pub fn uniform(n_elements: usize, target_vf: f64, scheme: InterpolationScheme) -> Self {
        Self {
            densities: vec![target_vf; n_elements],
            rho_min: 1e-3,
            scheme,
            target_volume_fraction: target_vf,
        }
    }

    /// Current volume fraction: `sum(rho * V_e) / sum(V_e)`.
    pub fn volume_fraction(&self, element_volumes: &[f64]) -> f64 {
        let total_vol: f64 = element_volumes.iter().sum();
        if total_vol < 1e-15 {
            return 0.0;
        }
        let weighted: f64 = self
            .densities
            .iter()
            .zip(element_volumes)
            .map(|(&rho, &v)| rho * v)
            .sum();
        weighted / total_vol
    }

    /// Interpolated stiffness ratio for each element.
    pub fn stiffness_ratios(&self) -> Vec<f64> {
        self.densities
            .iter()
            .map(|&rho| {
                let rho_eff = rho.max(self.rho_min);
                self.rho_min + self.scheme.interpolate(rho_eff) * (1.0 - self.rho_min)
            })
            .collect()
    }

    /// Stiffness ratio derivatives for each element.
    pub fn stiffness_ratio_derivatives(&self) -> Vec<f64> {
        self.densities
            .iter()
            .map(|&rho| {
                let rho_eff = rho.max(self.rho_min);
                self.scheme.interpolate_derivative(rho_eff) * (1.0 - self.rho_min)
            })
            .collect()
    }

    /// Clamp all densities to [rho_min, 1.0].
    pub fn clamp(&mut self) {
        for rho in &mut self.densities {
            *rho = rho.clamp(self.rho_min, 1.0);
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.densities.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.densities.is_empty()
    }
}

/// Magnetic topology design variables.
///
/// Each element can be: air (0), iron (1), or magnet (separate flag).
#[derive(Debug, Clone)]
pub struct MagneticDensityField {
    /// Iron density per element (0 = air, 1 = iron).
    pub iron_density: Vec<f64>,
    /// Magnet density per element (0 = no magnet, 1 = full magnet).
    pub magnet_density: Vec<f64>,
    /// Magnet orientation angle per element (radians).
    pub magnet_orientation: Vec<f64>,
    /// Interpolation scheme for iron.
    pub scheme: InterpolationScheme,
}

impl MagneticDensityField {
    pub fn new(n_elements: usize) -> Self {
        Self {
            iron_density: vec![0.5; n_elements],
            magnet_density: vec![0.0; n_elements],
            magnet_orientation: vec![0.0; n_elements],
            scheme: InterpolationScheme::simp(),
        }
    }

    /// Interpolated permeability ratio for each element.
    /// Air: mu_r = 1, Iron: mu_r = mu_iron.
    pub fn permeability_ratios(&self, mu_r_iron: f64) -> Vec<f64> {
        self.iron_density
            .iter()
            .map(|&rho| 1.0 + self.scheme.interpolate(rho.max(1e-3)) * (mu_r_iron - 1.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simp_interpolation() {
        let simp = InterpolationScheme::simp();
        assert!((simp.interpolate(0.0) - 0.0).abs() < 1e-14);
        assert!((simp.interpolate(1.0) - 1.0).abs() < 1e-14);
        assert!((simp.interpolate(0.5) - 0.125).abs() < 1e-14); // 0.5^3
    }

    #[test]
    fn test_ramp_interpolation() {
        let ramp = InterpolationScheme::ramp();
        assert!((ramp.interpolate(0.0) - 0.0).abs() < 1e-14);
        assert!((ramp.interpolate(1.0) - 1.0).abs() < 1e-14);
        // RAMP is less aggressive than SIMP at intermediate densities
        assert!(ramp.interpolate(0.5) > InterpolationScheme::simp().interpolate(0.5));
    }

    #[test]
    fn test_volume_fraction() {
        let field = DensityField::uniform(4, 0.5, InterpolationScheme::simp());
        let volumes = vec![1.0; 4];
        assert!((field.volume_fraction(&volumes) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_simp_derivative() {
        let simp = InterpolationScheme::simp();
        let rho = 0.5;
        let eps = 1e-7;
        let numerical = (simp.interpolate(rho + eps) - simp.interpolate(rho - eps)) / (2.0 * eps);
        let analytical = simp.interpolate_derivative(rho);
        assert!(
            (analytical - numerical).abs() < 1e-5,
            "SIMP derivative: analytical={analytical}, numerical={numerical}"
        );
    }
}
