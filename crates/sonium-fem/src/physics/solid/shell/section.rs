//! Shell cross-section properties

// Rust guideline compliant 2026-02-21

/// Reissner shear correction for plates.
///
/// Scales the transverse-shear stiffness `kappa * G * t`; 5/6 matches
/// the parabolic through-thickness shear distribution of plate theory.
/// Changing it rescales every transverse-shear force linearly.
pub const SHEAR_CORRECTION: f64 = 5.0 / 6.0;

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
    /// Builds a section from elastic constants and thickness.
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

    /// Shear modulus `G = E / (2 (1 + nu))` in Pa.
    #[must_use]
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Plate longitudinal wave speed in m/s.
    ///
    /// `c = sqrt(E / (rho (1 - nu^2)))` is the membrane dilatational
    /// speed of plane stress; explicit shell dynamics bounds the
    /// stable step by `Lc / c` per element (OpenCourant Theory Manual
    /// "Stability Time Step"). Returns NaN for non-positive density.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::shell::ShellSection;
    ///
    /// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
    /// let speed = section.plate_sound_speed();
    /// assert!((speed - 5309.0).abs() < 1.0);
    /// ```
    #[must_use]
    pub fn plate_sound_speed(&self) -> f64 {
        (self.youngs_modulus / (self.density * (1.0 - self.poisson_ratio.powi(2)))).sqrt()
    }

    /// Rotary inertia per unit area: rho * t^3 / 12.
    ///
    /// Lumped explicit shells carry this inertia on the rotational
    /// DOFs (Theory Manual "Mass and Inertia", pp. 115-116).
    #[must_use]
    pub fn rotary_inertia_per_area(&self) -> f64 {
        self.density * self.thickness.powi(3) / 12.0
    }

    /// Transverse-shear stiffness `kappa * G * t` in N/m.
    ///
    /// Uses the Reissner [`SHEAR_CORRECTION`]; thin Kirchhoff shells
    /// ignore it, shear-carrying hooks scale resultants with it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::shell::{SHEAR_CORRECTION, ShellSection};
    ///
    /// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
    /// let g = 200.0e9 / 2.6;
    /// assert!((section.transverse_shear_stiffness() - SHEAR_CORRECTION * g * 0.002).abs() < 1.0);
    /// ```
    #[must_use]
    pub fn transverse_shear_stiffness(&self) -> f64 {
        SHEAR_CORRECTION * self.shear_modulus() * self.thickness
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

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> ShellSection {
        ShellSection::new(200.0e9, 0.3, 0.002, 7800.0)
    }

    #[test]
    fn shear_modulus_matches_isotropic_identity() {
        let section = steel();
        assert!((section.shear_modulus() - 200.0e9 / 2.6).abs() < 1.0);
    }

    #[test]
    fn plate_sound_speed_matches_closed_form() {
        let section = steel();
        let expected = f64::sqrt(200.0e9 / (7800.0 * (1.0 - 0.09)));
        assert!((section.plate_sound_speed() - expected).abs() < 1e-9);
        assert!(section.plate_sound_speed() > 5300.0);
    }

    #[test]
    fn inertias_scale_with_thickness() {
        let section = steel();
        assert!(
            (section.rotary_inertia_per_area() - 7800.0 * 0.002f64.powi(3) / 12.0).abs() < 1e-18
        );
        assert!((section.mass_per_area() - 7800.0 * 0.002).abs() < 1e-12);
    }

    #[test]
    fn shear_stiffness_uses_reissner_correction() {
        let section = steel();
        let expected = SHEAR_CORRECTION * section.shear_modulus() * 0.002;
        assert!((section.transverse_shear_stiffness() - expected).abs() < 1e-6);
        assert_eq!(SHEAR_CORRECTION, 5.0 / 6.0);
    }
}
