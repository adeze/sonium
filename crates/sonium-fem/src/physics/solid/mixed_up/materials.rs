//! Isotropic material data for the mixed u-p formulation.

// Rust guideline compliant 2026-02-21

use std::fmt;

/// Error for invalid mixed u-p material parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum MixedUpMaterialError {
    /// Young's modulus must be finite and positive.
    InvalidYoungsModulus(f64),
    /// Poisson's ratio must lie in `(0, 1/2]`.
    InvalidPoissonRatio(f64),
}

impl fmt::Display for MixedUpMaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidYoungsModulus(value) => {
                write!(
                    f,
                    "Young's modulus must be finite and positive, got {value}"
                )
            }
            Self::InvalidPoissonRatio(value) => {
                write!(f, "Poisson's ratio must lie in (0, 1/2], got {value}")
            }
        }
    }
}

impl std::error::Error for MixedUpMaterialError {}

/// Shear modulus, first Lamé parameter, and volumetric compliance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedElasticMaterial {
    /// Shear modulus `μ` in Pa.
    pub mu: f64,
    /// First Lamé parameter `λ` in Pa (`+∞` at `ν = 1/2`).
    pub lambda: f64,
    /// Volumetric compliance `c = 1/λ` in 1/Pa (exactly `0` at `ν = 1/2`).
    pub compliance: f64,
}

impl MixedElasticMaterial {
    /// Build from engineering constants.
    ///
    /// Poisson's ratio must lie in `(0, 1/2]`: `ν ≤ 0` has no volumetric
    /// coupling to split off, and `ν > 1/2` is unphysical. At `ν = 1/2`
    /// the compliance is exactly zero and the pressure needs one pinning
    /// constraint (handled by the study, not the forms).
    ///
    /// # Errors
    ///
    /// Returns [`MixedUpMaterialError`] for non-positive or non-finite
    /// Young's modulus, or Poisson's ratio outside `(0, 1/2]`.
    pub fn from_young_poisson(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
    ) -> Result<Self, MixedUpMaterialError> {
        if !youngs_modulus_pa.is_finite() || youngs_modulus_pa <= 0.0 {
            return Err(MixedUpMaterialError::InvalidYoungsModulus(
                youngs_modulus_pa,
            ));
        }
        if !poisson_ratio.is_finite() || poisson_ratio <= 0.0 || poisson_ratio > 0.5 {
            return Err(MixedUpMaterialError::InvalidPoissonRatio(poisson_ratio));
        }
        let mu = youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio));
        // Compliance first: exact 0 at nu = 1/2, no 1/inf arithmetic.
        let compliance = (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
            / (youngs_modulus_pa * poisson_ratio);
        let lambda = if compliance == 0.0 {
            f64::INFINITY
        } else {
            1.0 / compliance
        };
        Ok(Self {
            mu,
            lambda,
            compliance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lame_parameters_match_closed_forms() {
        let material = MixedElasticMaterial::from_young_poisson(2.0e6, 0.3).expect("valid");
        assert!((material.mu - 2.0e6 / 2.6).abs() < 1.0e-6);
        assert!((material.lambda - 2.0e6 * 0.3 / (1.3 * 0.4)).abs() < 1.0e-3);
        assert!((material.compliance - 1.0 / material.lambda).abs() < 1.0e-18);
    }

    #[test]
    fn incompressible_limit_has_zero_compliance() {
        let material = MixedElasticMaterial::from_young_poisson(2.0e6, 0.5).expect("valid");
        assert_eq!(material.compliance, 0.0);
        assert!(material.lambda.is_infinite());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(MixedElasticMaterial::from_young_poisson(0.0, 0.3).is_err());
        assert!(MixedElasticMaterial::from_young_poisson(-1.0, 0.3).is_err());
        assert!(MixedElasticMaterial::from_young_poisson(f64::NAN, 0.3).is_err());
        assert!(MixedElasticMaterial::from_young_poisson(2.0e6, 0.0).is_err());
        assert!(MixedElasticMaterial::from_young_poisson(2.0e6, -0.1).is_err());
        assert!(MixedElasticMaterial::from_young_poisson(2.0e6, 0.5001).is_err());
    }
}
