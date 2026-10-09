//! Shell cross-section properties

/// Shell section properties.
#[derive(Debug, Clone, Copy)]
pub struct ShellSection {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
    /// Shell thickness (m).
    pub thickness: f64,
    /// Material density (kg/m^3).
    pub density: f64,
}

impl ShellSection {
    pub fn new(youngs_modulus: f64, poisson_ratio: f64, thickness: f64, density: f64) -> Self {
        Self {
            youngs_modulus,
            poisson_ratio,
            thickness,
            density,
        }
    }

    /// Loudspeaker cone: paper or polypropylene.
    pub fn cone(thickness: f64) -> Self {
        // Paper cone: E ~= 3 GPa, nu ~= 0.3, rho ~= 600 kg/m^3
        Self::new(3e9, 0.3, thickness, 600.0)
    }

    /// Loudspeaker surround: rubber/foam.
    pub fn surround(thickness: f64) -> Self {
        Self::new(5e6, 0.45, thickness, 1100.0)
    }

    /// Membrane stiffness: Et/(1-nu^2)
    pub fn membrane_stiffness(&self) -> f64 {
        self.youngs_modulus * self.thickness / (1.0 - self.poisson_ratio.powi(2))
    }

    /// Bending stiffness: Et^3/(12(1-nu^2))
    pub fn bending_stiffness(&self) -> f64 {
        self.youngs_modulus * self.thickness.powi(3) / (12.0 * (1.0 - self.poisson_ratio.powi(2)))
    }

    /// Mass per unit area: rho * t
    pub fn mass_per_area(&self) -> f64 {
        self.density * self.thickness
    }

    /// Plane stress elasticity matrix (3x3, Voigt notation).
    pub fn membrane_constitutive(&self) -> [f64; 9] {
        let d = self.membrane_stiffness();
        let nu = self.poisson_ratio;
        [
            d,
            d * nu,
            0.0,
            d * nu,
            d,
            0.0,
            0.0,
            0.0,
            d * (1.0 - nu) / 2.0,
        ]
    }

    /// Bending constitutive matrix (3x3, Voigt notation for curvatures).
    pub fn bending_constitutive(&self) -> [f64; 9] {
        let d = self.bending_stiffness();
        let nu = self.poisson_ratio;
        [
            d,
            d * nu,
            0.0,
            d * nu,
            d,
            0.0,
            0.0,
            0.0,
            d * (1.0 - nu) / 2.0,
        ]
    }
}
