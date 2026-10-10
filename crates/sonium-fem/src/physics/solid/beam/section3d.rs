//! Cross-section properties for 3-D beam elements.
//!
//! A [`Section3D`] extends the planar [`BeamSection`](super::section::BeamSection)
//! with the second bending axis, Saint-Venant torsion, and per-plane shear
//! correction. All quantities use SI units. Local cross-section axes follow the
//! element frame: `x` along the beam, `y` and `z` across the section.

// Rust guideline compliant 2026-02-21

use super::section::{BeamSection, HSection};

/// Cross-section properties for a 3-D Timoshenko space-frame element.
///
/// Bending about the local `z` axis acts in the `x-y` plane and uses `iz`;
/// bending about the local `y` axis acts in the `x-z` plane and uses `iy`.
#[derive(Debug, Clone, Copy)]
pub struct Section3D {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Shear modulus in Pa.
    pub shear_modulus: f64,
    /// Cross-sectional area in m^2.
    pub area: f64,
    /// Second moment of area about the local `y` axis in m^4.
    pub iy: f64,
    /// Second moment of area about the local `z` axis in m^4.
    pub iz: f64,
    /// Saint-Venant torsion constant in m^4.
    pub torsion_constant: f64,
    /// Warping constant `I_omega` in m^6 for thin-walled open sections.
    ///
    /// Zero means warping stiffness is neglected (plain Saint-Venant
    /// torsion). Only [`Section3D::i_beam`] and [`Section3D::with_warping`]
    /// set a nonzero value today.
    pub warping_constant: f64,
    /// Shear correction factor for shear along `y`. Zero selects Euler-Bernoulli.
    pub kappa_y: f64,
    /// Shear correction factor for shear along `z`. Zero selects Euler-Bernoulli.
    pub kappa_z: f64,
    /// Material density in kg/m^3.
    pub density: f64,
}

impl Section3D {
    /// Build a section from explicit properties and Poisson ratio.
    ///
    /// The shear modulus follows from `e` and `nu`. Both shear factors default
    /// to zero, which selects Euler-Bernoulli bending in both planes.
    ///
    /// # Examples
    ///
    /// ```
    /// use sonium_fem::physics::solid::beam::Section3D;
    /// let section = Section3D::new(200.0e9, 0.3, 1.0e-4, 1.0e-8, 2.0e-8, 3.0e-8, 7800.0);
    /// assert!((section.ea() - 200.0e9 * 1.0e-4).abs() < 1.0e-6);
    /// ```
    #[must_use]
    pub fn new(
        e: f64,
        nu: f64,
        area: f64,
        iy: f64,
        iz: f64,
        torsion_constant: f64,
        rho: f64,
    ) -> Self {
        Self {
            youngs_modulus: e,
            shear_modulus: e / (2.0 * (1.0 + nu)),
            area,
            iy,
            iz,
            torsion_constant,
            warping_constant: 0.0,
            kappa_y: 0.0,
            kappa_z: 0.0,
            density: rho,
        }
    }

    /// Attach a warping constant `I_omega` in m^6.
    ///
    /// The warping element reads this through [`Section3D::warping_stiffness`];
    /// plain [`BeamElement3D`](super::element3d::BeamElement3D) assembly ignores it.
    #[must_use]
    pub fn with_warping(mut self, warping_constant: f64) -> Self {
        self.warping_constant = warping_constant;
        self
    }

    /// Enable Timoshenko shear in both planes with Cowper style factors.
    #[must_use]
    pub fn with_shear(mut self, kappa_y: f64, kappa_z: f64) -> Self {
        self.kappa_y = kappa_y;
        self.kappa_z = kappa_z;
        self
    }

    /// Rectangular section with width `b` along `y` and height `h` along `z`.
    ///
    /// The torsion constant uses the thin-rectangle series truncated after the
    /// cubic term, accurate to a few percent for `b / h > 1.5`. When
    /// `timoshenko` is set, both shear factors use `kappa = 5 / 6`.
    #[must_use]
    pub fn rectangular(e: f64, nu: f64, b: f64, h: f64, rho: f64, timoshenko: bool) -> Self {
        let area = b * h;
        let iy = b * h.powi(3) / 12.0;
        let iz = h * b.powi(3) / 12.0;
        // Long side first so the 0.21 side ratio correction stays below one.
        let (long, short) = if b >= h { (b, h) } else { (h, b) };
        let ratio = short / long;
        let torsion_constant =
            long * short.powi(3) * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio.powi(4) / 12.0));
        let mut section = Self::new(e, nu, area, iy, iz, torsion_constant, rho);
        if timoshenko {
            section = section.with_shear(5.0 / 6.0, 5.0 / 6.0);
        }
        section
    }

    /// Solid circular section of radius `r`.
    ///
    /// Uses `J = pi r^4 / 2` and the Cowper shear factor
    /// `6 (1 + nu) / (7 + 6 nu)` when `timoshenko` is set.
    #[must_use]
    pub fn circular(e: f64, nu: f64, r: f64, rho: f64, timoshenko: bool) -> Self {
        let area = std::f64::consts::PI * r * r;
        let second = std::f64::consts::PI * r.powi(4) / 4.0;
        let torsion = 2.0 * second;
        let mut section = Self::new(e, nu, area, second, second, torsion, rho);
        if timoshenko {
            let kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu);
            section = section.with_shear(kappa, kappa);
        }
        section
    }

    /// Doubly symmetric I/H section from flange width, web height, thickness.
    ///
    /// The strong-axis inertia reuses [`HSection`]; the weak-axis inertia sums
    /// flange and web rectangles, and the torsion constant sums `b t^3 / 3`
    /// over the three thin walls, the standard open-section approximation.
    /// The warping constant uses `I_omega = I_y h_w^2 / 4` with `h_w = h + t`
    /// the distance between flange midlines (Vlasov thin-walled theory).
    #[must_use]
    pub fn i_beam(e: f64, nu: f64, b: f64, h: f64, t: f64, rho: f64, timoshenko: bool) -> Self {
        let h_section = HSection::new(b, h, t);
        let area = h_section.area();
        // Strong axis: bending in the web plane.
        let iz = h_section.moment_of_inertia();
        // Weak axis: two flange rectangles plus the thin web contribution.
        let iy = (2.0 * t * b.powi(3) + h * t.powi(3)) / 12.0;
        // Open thin-walled torsion constant.
        let torsion_constant = (2.0 * b * t.powi(3) + h * t.powi(3)) / 3.0;
        let mut section = Self::new(e, nu, area, iy, iz, torsion_constant, rho);
        let flange_spacing = h + t;
        section.warping_constant = iy * flange_spacing * flange_spacing / 4.0;
        if timoshenko {
            // Shear flows mostly through the web; flange shear is neglected.
            let web_area = h * t;
            let kappa_z = web_area / area;
            section = section.with_shear(5.0 / 6.0, kappa_z);
        }
        section
    }

    /// Axial stiffness `E A` in N.
    #[must_use]
    pub fn ea(&self) -> f64 {
        self.youngs_modulus * self.area
    }

    /// Bending stiffness about local `y` in N m^2.
    #[must_use]
    pub fn eiy(&self) -> f64 {
        self.youngs_modulus * self.iy
    }

    /// Bending stiffness about local `z` in N m^2.
    #[must_use]
    pub fn eiz(&self) -> f64 {
        self.youngs_modulus * self.iz
    }

    /// Torsional stiffness `G J` in N m^2.
    #[must_use]
    pub fn gj(&self) -> f64 {
        self.shear_modulus * self.torsion_constant
    }

    /// Warping stiffness `E I_omega` in N m^4.
    ///
    /// Zero for sections without a warping constant, which reduces the
    /// warping element to Saint-Venant torsion.
    #[must_use]
    pub fn warping_stiffness(&self) -> f64 {
        self.youngs_modulus * self.warping_constant
    }

    /// Mass per unit length in kg/m.
    #[must_use]
    pub fn mass_per_length(&self) -> f64 {
        self.density * self.area
    }

    /// Shear stiffness along local `y` in N.
    #[must_use]
    pub fn shear_stiffness_y(&self) -> f64 {
        self.kappa_y * self.shear_modulus * self.area
    }

    /// Shear stiffness along local `z` in N.
    #[must_use]
    pub fn shear_stiffness_z(&self) -> f64 {
        self.kappa_z * self.shear_modulus * self.area
    }

    /// Timoshenko shear parameter for bending about `z` at length `l`.
    ///
    /// Returns zero for Euler-Bernoulli sections, which recovers the cubic
    /// Hermite stiffness exactly.
    #[must_use]
    pub fn phi_z(&self, l: f64) -> f64 {
        if self.kappa_y <= 0.0 {
            return 0.0;
        }
        12.0 * self.eiz() / (self.shear_stiffness_y() * l * l)
    }

    /// Timoshenko shear parameter for bending about `y` at length `l`.
    #[must_use]
    pub fn phi_y(&self, l: f64) -> f64 {
        if self.kappa_z <= 0.0 {
            return 0.0;
        }
        12.0 * self.eiy() / (self.shear_stiffness_z() * l * l)
    }

    /// Whether either bending plane includes shear deformation.
    #[must_use]
    pub fn is_timoshenko(&self) -> bool {
        self.kappa_y > 0.0 || self.kappa_z > 0.0
    }

    /// Reduce to the planar section used by [`BeamElement2D`](super::element::BeamElement2D).
    ///
    /// The planar element bends in its `x-y` plane, so the reduction keeps the
    /// local `z` bending axis and the `y` shear correction.
    #[must_use]
    pub fn to_planar_xy(&self) -> BeamSection {
        BeamSection {
            youngs_modulus: self.youngs_modulus,
            moment_of_inertia: self.iz,
            area: self.area,
            density: self.density,
            shear_correction: self.kappa_y,
            shear_modulus: self.shear_modulus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_matches_closed_form() {
        let section = Section3D::rectangular(200.0e9, 0.3, 0.02, 0.01, 7800.0, true);
        assert!((section.area - 2.0e-4).abs() < 1.0e-12);
        assert!((section.iy - 0.02 * 0.01_f64.powi(3) / 12.0).abs() < 1.0e-15);
        assert!((section.iz - 0.01 * 0.02_f64.powi(3) / 12.0).abs() < 1.0e-15);
        assert!((section.kappa_y - 5.0 / 6.0).abs() < 1.0e-15);
        assert!(section.is_timoshenko());
    }

    #[test]
    fn circular_torsion_is_twice_bending() {
        let section = Section3D::circular(70.0e9, 0.33, 0.05, 2700.0, false);
        assert!((section.torsion_constant - 2.0 * section.iy).abs() < 1.0e-15);
        assert!((section.iy - section.iz).abs() < 1.0e-18);
        assert!(!section.is_timoshenko());
        assert_eq!(section.phi_y(1.0), 0.0);
        assert_eq!(section.phi_z(2.0), 0.0);
    }

    #[test]
    fn i_beam_reuses_h_section_area() {
        let section = Section3D::i_beam(210.0e9, 0.3, 0.1, 0.2, 0.01, 7800.0, false);
        let expected = HSection::new(0.1, 0.2, 0.01);
        assert!((section.area - expected.area()).abs() < 1.0e-15);
        assert!((section.iz - expected.moment_of_inertia()).abs() < 1.0e-15);
        assert!(section.iy > 0.0 && section.torsion_constant > 0.0);
    }

    #[test]
    fn i_beam_warping_matches_thin_wall_formula() {
        let section = Section3D::i_beam(210.0e9, 0.3, 0.1, 0.2, 0.01, 7800.0, false);
        // I_omega = t b^3 h_w^2 / 24 with h_w = h + t between flange midlines.
        let expected = 0.01 * 0.1_f64.powi(3) * 0.21_f64.powi(2) / 24.0;
        let actual = section.warping_constant;
        assert!(
            (actual - expected).abs() / expected < 0.02,
            "warping {actual:.4e} vs thin-wall {expected:.4e}"
        );
        // Default sections carry no warping stiffness.
        let plain = Section3D::rectangular(200.0e9, 0.3, 0.02, 0.01, 7800.0, false);
        assert_eq!(plain.warping_constant, 0.0);
        assert_eq!(plain.warping_stiffness(), 0.0);
    }
}
