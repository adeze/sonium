//! Membrane-thickness coupling for large-bending shells.
//!
//! Thin-shell kinematics usually freeze the thickness, which misses the
//! Poisson thinning under membrane tension and the thickening under
//! compression. This module updates the thickness from the membrane strain
//! state: elastic plane-stress thinning for small strains and plastic
//! incompressibility once yielding starts. Element stiffnesses are rebuilt
//! from the updated thickness on the next assembly.
//!
//! Reference: Belytschko, Liu, and Moran, "Nonlinear Finite Elements for
//! Continua and Structures", 2nd ed., Chapter 8 (shell thickness update).

// Rust guideline compliant 2026-02-21

/// Thickness update rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThicknessRule {
    /// Elastic plane stress: eps_zz = -nu/(1-nu) (eps_xx + eps_yy).
    PlaneStress {
        /// Poisson's ratio numerator, stored as 1e6 * nu for Eq.
        nu_micro: u32,
    },
    /// Plastic incompressibility: eps_zz = -(eps_xx + eps_yy).
    Incompressible,
}

impl ThicknessRule {
    /// Plane-stress rule for a Poisson ratio.
    pub fn plane_stress(poisson_ratio: f64) -> Self {
        Self::PlaneStress {
            nu_micro: (poisson_ratio * 1.0e6).round() as u32,
        }
    }

    /// Transverse strain for membrane strains [eps_xx, eps_yy, gamma_xy].
    pub fn transverse_strain(&self, eps_membrane: &[f64; 3]) -> f64 {
        match *self {
            Self::PlaneStress { nu_micro } => {
                let nu = f64::from(nu_micro) / 1.0e6;
                -nu / (1.0 - nu) * (eps_membrane[0] + eps_membrane[1])
            }
            Self::Incompressible => -(eps_membrane[0] + eps_membrane[1]),
        }
    }
}

/// Updated thickness from a transverse strain (logarithmic update).
pub fn apply_transverse_strain(thickness: f64, eps_zz: f64) -> f64 {
    assert!(thickness > 0.0, "thickness must stay positive");
    thickness * eps_zz.exp()
}

/// One-step thickness update from membrane strains.
pub fn update_thickness(
    thickness: f64,
    eps_membrane: &[f64; 3],
    rule: ThicknessRule,
) -> f64 {
    apply_transverse_strain(thickness, rule.transverse_strain(eps_membrane))
}

/// Per-element thickness state for a shell mesh.
#[derive(Debug, Clone)]
pub struct ThicknessField {
    /// Reference thicknesses in m.
    reference: Vec<f64>,
    /// Current thicknesses in m.
    current: Vec<f64>,
    /// Update rule per element.
    rules: Vec<ThicknessRule>,
}

impl ThicknessField {
    /// Create from reference thicknesses with one shared rule.
    pub fn new(reference: Vec<f64>, rule: ThicknessRule) -> Self {
        assert!(
            reference.iter().all(|&t| t > 0.0),
            "reference thicknesses must be positive"
        );
        let rules = vec![rule; reference.len()];
        Self {
            current: reference.clone(),
            reference,
            rules,
        }
    }

    /// Create with per-element rules.
    pub fn with_rules(reference: Vec<f64>, rules: Vec<ThicknessRule>) -> Self {
        assert_eq!(reference.len(), rules.len(), "thickness/rule length mismatch");
        assert!(
            reference.iter().all(|&t| t > 0.0),
            "reference thicknesses must be positive"
        );
        Self {
            current: reference.clone(),
            reference,
            rules,
        }
    }

    /// Number of tracked elements.
    pub fn len(&self) -> usize {
        self.reference.len()
    }

    /// Whether the field is empty.
    pub fn is_empty(&self) -> bool {
        self.reference.is_empty()
    }

    /// Current thickness of an element.
    pub fn current(&self, elem: usize) -> f64 {
        self.current[elem]
    }

    /// Reference thickness of an element.
    pub fn reference(&self, elem: usize) -> f64 {
        self.reference[elem]
    }

    /// Commit a membrane strain state for one element.
    ///
    /// The update is total (from the reference thickness), so repeated
    /// commits with the same strain are idempotent.
    pub fn commit_element(&mut self, elem: usize, eps_membrane: &[f64; 3]) {
        let eps_zz = self.rules[elem].transverse_strain(eps_membrane);
        self.current[elem] = apply_transverse_strain(self.reference[elem], eps_zz);
    }

    /// Logarithmic thickness strain of an element.
    pub fn thickness_strain(&self, elem: usize) -> f64 {
        (self.current[elem] / self.reference[elem]).ln()
    }

    /// Maximum absolute thickness change ratio over all elements.
    pub fn max_change_ratio(&self) -> f64 {
        self.current
            .iter()
            .zip(self.reference.iter())
            .map(|(&c, &r)| (c / r - 1.0).abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniaxial_tension_thins_by_poisson() {
        let t0 = 0.01;
        let nu = 0.3;
        let eps = [1e-3, -nu * 1e-3, 0.0];
        let t = update_thickness(t0, &eps, ThicknessRule::plane_stress(nu));
        let expected = t0 * (-nu / (1.0 - nu) * (eps[0] + eps[1])).exp();
        assert!((t - expected).abs() / t0 < 1e-12);
        assert!(t < t0, "tension must thin the shell");
    }

    #[test]
    fn incompressible_update_preserves_volume() {
        // Equibiaxial 1%: area grows by ~2%, thickness must shrink by ~2%.
        let t0 = 0.01;
        let eps = [0.01, 0.01, 0.0];
        let t = update_thickness(t0, &eps, ThicknessRule::Incompressible);
        let expected = t0 * (-0.02_f64).exp();
        assert!((t - expected).abs() / t0 < 1e-12);
    }

    #[test]
    fn thickness_field_commit_is_idempotent() {
        let mut field = ThicknessField::new(vec![0.01, 0.02], ThicknessRule::plane_stress(0.3));
        let eps = [2e-3, 1e-3, 0.0];
        field.commit_element(0, &eps);
        let once = field.current(0);
        field.commit_element(0, &eps);
        assert!((field.current(0) - once).abs() < 1e-15);
        assert!((field.current(1) - 0.02).abs() < 1e-15);
        assert!(field.max_change_ratio() > 0.0);
        assert!((field.thickness_strain(1)).abs() < 1e-15);
    }

    #[test]
    fn compression_thickens_shell() {
        let t = update_thickness(0.01, &[-1e-3, -1e-3, 0.0], ThicknessRule::plane_stress(0.3));
        assert!(t > 0.01, "compression must thicken the shell");
    }
}
