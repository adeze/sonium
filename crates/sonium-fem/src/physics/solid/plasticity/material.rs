//! Elastoplastic material definitions

/// Hardening law for plasticity.
#[derive(Debug, Clone, Copy)]
pub enum HardeningLaw {
    /// Perfect plasticity: no hardening after yield.
    Perfect,
    /// Linear isotropic hardening: yield stress grows linearly with accumulated plastic strain.
    /// `sigma_y(ep) = sigma_y0 + H * ep`
    LinearIsotropic {
        /// Hardening modulus H (Pa).
        modulus: f64,
    },
    /// Nonlinear isotropic (Voce/saturation) hardening:
    /// `sigma_y(ep) = sigma_y0 + Q * (1 - exp(-b * ep))`
    VoceSaturation {
        /// Saturation stress increment Q (Pa).
        q: f64,
        /// Saturation rate b.
        b: f64,
    },
    /// Linear kinematic (Prager) hardening: back-stress evolves linearly.
    /// `alpha_dot = (2/3) * C * ep_dot`
    LinearKinematic {
        /// Kinematic hardening modulus C (Pa).
        modulus: f64,
    },
    /// Combined isotropic + kinematic hardening.
    Combined {
        /// Isotropic hardening modulus H (Pa).
        isotropic_modulus: f64,
        /// Kinematic hardening modulus C (Pa).
        kinematic_modulus: f64,
    },
}

/// J2 (von Mises) elastoplastic material.
#[derive(Debug, Clone, Copy)]
pub struct J2Material {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
    /// Initial yield stress (Pa).
    pub yield_stress: f64,
    /// Hardening law.
    pub hardening: HardeningLaw,
}

impl J2Material {
    /// Create with linear isotropic hardening.
    pub fn new(e: f64, nu: f64, sigma_y: f64, hardening: HardeningLaw) -> Self {
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            yield_stress: sigma_y,
            hardening,
        }
    }

    /// Mild steel: E=210 GPa, nu=0.3, sigma_y=250 MPa, H=1 GPa.
    pub fn mild_steel() -> Self {
        Self::new(
            210e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        )
    }

    /// Aluminum 6061-T6: E=70 GPa, nu=0.33, sigma_y=276 MPa.
    pub fn aluminum_6061() -> Self {
        Self::new(
            70e9,
            0.33,
            276e6,
            HardeningLaw::LinearIsotropic { modulus: 0.5e9 },
        )
    }

    /// Shear modulus G = E / (2(1+nu)).
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus K = E / (3(1-2nu)).
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Current yield stress given accumulated plastic strain.
    pub fn current_yield_stress(&self, equiv_plastic_strain: f64) -> f64 {
        let ep = equiv_plastic_strain;
        match self.hardening {
            HardeningLaw::Perfect => self.yield_stress,
            HardeningLaw::LinearIsotropic { modulus } => self.yield_stress + modulus * ep,
            HardeningLaw::VoceSaturation { q, b } => {
                self.yield_stress + q * (1.0 - (-b * ep).exp())
            }
            HardeningLaw::LinearKinematic { .. } => self.yield_stress, // isotropic part unchanged
            HardeningLaw::Combined {
                isotropic_modulus, ..
            } => self.yield_stress + isotropic_modulus * ep,
        }
    }

    /// Isotropic hardening slope d(sigma_y)/d(ep).
    pub fn isotropic_hardening_slope(&self, equiv_plastic_strain: f64) -> f64 {
        match self.hardening {
            HardeningLaw::Perfect => 0.0,
            HardeningLaw::LinearIsotropic { modulus } => modulus,
            HardeningLaw::VoceSaturation { q, b } => q * b * (-b * equiv_plastic_strain).exp(),
            HardeningLaw::LinearKinematic { .. } => 0.0,
            HardeningLaw::Combined {
                isotropic_modulus, ..
            } => isotropic_modulus,
        }
    }

    /// Kinematic hardening modulus C.
    pub fn kinematic_hardening_modulus(&self) -> f64 {
        match self.hardening {
            HardeningLaw::LinearKinematic { modulus } => modulus,
            HardeningLaw::Combined {
                kinematic_modulus, ..
            } => kinematic_modulus,
            _ => 0.0,
        }
    }
}

/// State variables at a material integration point.
#[derive(Debug, Clone)]
pub struct PlasticState {
    /// Cauchy stress (Voigt notation): 6 components for 3D, 3 for plane stress/strain.
    pub stress: Vec<f64>,
    /// Back-stress for kinematic hardening (same size as stress).
    pub back_stress: Vec<f64>,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Whether the point is currently yielding.
    pub yielding: bool,
}

impl PlasticState {
    /// Initial (unstressed) state.
    pub fn new(n_stress_components: usize) -> Self {
        Self {
            stress: vec![0.0; n_stress_components],
            back_stress: vec![0.0; n_stress_components],
            equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    /// 3D state (6 Voigt components).
    pub fn new_3d() -> Self {
        Self::new(6)
    }

    /// 2D plane stress/strain state (3 Voigt components).
    pub fn new_2d() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_stress_linear_hardening() {
        let mat = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        );
        assert!((mat.current_yield_stress(0.0) - 250e6).abs() < 1.0);
        assert!((mat.current_yield_stress(0.01) - 260e6).abs() < 1.0);
    }

    #[test]
    fn test_voce_saturation() {
        let mat = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::VoceSaturation { q: 100e6, b: 10.0 },
        );
        // At ep=0: sigma_y = 250 MPa
        assert!((mat.current_yield_stress(0.0) - 250e6).abs() < 1.0);
        // At large ep: sigma_y -> 250 + 100 = 350 MPa
        assert!((mat.current_yield_stress(10.0) - 350e6).abs() < 1e3);
    }

    #[test]
    fn test_perfect_plasticity() {
        let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        assert!((mat.current_yield_stress(0.0) - 250e6).abs() < 1.0);
        assert!((mat.current_yield_stress(1.0) - 250e6).abs() < 1.0);
        assert_eq!(mat.isotropic_hardening_slope(0.0), 0.0);
    }
}
