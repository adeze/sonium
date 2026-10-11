//! Radioss spring property cards mapped onto beam sections.
//!
//! `/PROP/TYPE8 (SPR_GENE)`, `/PROP/TYPE25 (SPR_AXI)`, and `/PROP/TYPE32
//! (SPR_PRE)` (Reference Guide pp. 1490-1493, 1531-1534, 1563-1567)
//! define joint and spring behavior through per-mode stiffness `K`,
//! damping `C`, mass, and failure limits. This module holds the linear
//! coefficients of each card and maps them onto the beam seeds
//! ([`Section3D`], [`BeamSection`]) so a spring joint can be replaced by
//! an equivalent beam, and converts beam sections back to equivalent
//! spring rates for round-trip checks.
//!
//! Only the linear coefficients map; nonlinear `/FUNCT` curves stay on
//! the frozen [`Funct1D`](super::super::plasticity::funct::Funct1D) type
//! consumed by [`Spring1D`](super::spring::Spring1D), and sensor or
//! failure state machines belong to the driver.

// Rust guideline compliant 2026-02-21

use super::section::BeamSection;
use super::section3d::Section3D;

/// One deformation mode of a spring property card.
///
/// Holds the linear stiffness `K`, viscous damping `C`, and the
/// displacement, force, and energy failure limits (`A`, `B`, `D` on the
/// TYPE8/TYPE25 cards; zero disables a limit).
#[derive(Debug, Clone, Copy)]
pub struct SpringMode {
    /// Linear stiffness in N/m (translations) or N m/rad (rotations).
    pub stiffness: f64,
    /// Viscous damping in N s/m or N m s/rad.
    pub damping: f64,
    /// Displacement/rotation rupture limit (0 disables).
    pub fail_deformation: f64,
    /// Force/moment rupture limit (0 disables).
    pub fail_force: f64,
    /// Internal-energy rupture limit in J (0 disables).
    pub fail_energy: f64,
}

impl SpringMode {
    /// Builds a mode from stiffness and damping with no failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::SpringMode;
    ///
    /// let mode = SpringMode::new(1000.0, 5.0);
    /// assert!(!mode.failed(100.0, 0.01, 1.0));
    /// assert!(mode.with_fail_force(50.0).failed(100.0, 0.01, 1.0));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for negative stiffness or damping.
    #[must_use]
    pub fn new(stiffness: f64, damping: f64) -> Self {
        assert!(
            stiffness.is_finite() && stiffness >= 0.0,
            "stiffness must be finite non-negative"
        );
        assert!(
            damping.is_finite() && damping >= 0.0,
            "damping must be finite non-negative"
        );
        Self {
            stiffness,
            damping,
            fail_deformation: 0.0,
            fail_force: 0.0,
            fail_energy: 0.0,
        }
    }

    /// Sets the displacement rupture limit (same units as deformation).
    #[must_use]
    pub fn with_fail_deformation(mut self, limit: f64) -> Self {
        self.fail_deformation = limit;
        self
    }

    /// Sets the force rupture limit in N or N m.
    #[must_use]
    pub fn with_fail_force(mut self, limit: f64) -> Self {
        self.fail_force = limit;
        self
    }

    /// Sets the internal-energy rupture limit in J.
    #[must_use]
    pub fn with_fail_energy(mut self, limit: f64) -> Self {
        self.fail_energy = limit;
        self
    }

    /// Reports whether deformation, force, or energy trips rupture.
    ///
    /// Zero limits never trip; all inputs compare by magnitude.
    #[must_use]
    pub fn failed(&self, force: f64, deformation: f64, energy: f64) -> bool {
        (self.fail_force > 0.0 && force.abs() >= self.fail_force)
            || (self.fail_deformation > 0.0 && deformation.abs() >= self.fail_deformation)
            || (self.fail_energy > 0.0 && energy >= self.fail_energy)
    }
}

/// `/PROP/TYPE8 (SPR_GENE)` general spring (Reference Guide pp. 1490-1493).
///
/// Six independent modes: translations X, Y, Z then rotations X, Y, Z.
/// Only the linear `K`/`C` coefficients map to beams; curve, sensor,
/// and equilibrium fields stay driver-side.
#[derive(Debug, Clone, Copy)]
pub struct SpringGeneProp {
    /// Spring mass in kg.
    pub mass: f64,
    /// Spring rotary inertia in kg m^2.
    pub inertia: f64,
    /// Translation-X (axial) mode.
    pub kx: SpringMode,
    /// Translation-Y mode.
    pub ky: SpringMode,
    /// Translation-Z mode.
    pub kz: SpringMode,
    /// Rotation-X (torsion) mode.
    pub rx: SpringMode,
    /// Rotation-Y (bending) mode.
    pub ry: SpringMode,
    /// Rotation-Z (bending) mode.
    pub rz: SpringMode,
}

impl SpringGeneProp {
    /// Builds a TYPE8 card from mass and six modes.
    ///
    /// # Panics
    ///
    /// Panics for negative mass or inertia.
    #[must_use]
    pub fn new(
        mass: f64,
        inertia: f64,
        kx: SpringMode,
        ky: SpringMode,
        kz: SpringMode,
        rx: SpringMode,
        ry: SpringMode,
        rz: SpringMode,
    ) -> Self {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "mass must be finite non-negative"
        );
        assert!(
            inertia.is_finite() && inertia >= 0.0,
            "inertia must be finite non-negative"
        );
        Self {
            mass,
            inertia,
            kx,
            ky,
            kz,
            rx,
            ry,
            rz,
        }
    }

    /// Maps the card onto an equivalent [`Section3D`] at a length.
    ///
    /// The beam of length `l` reproduces each linear spring rate:
    /// `EA = Kx l`, `GJ = Krx l`, `EIy = Kry l`, `EIz = Krz l`, and
    /// shear `kappa G A = K l` per plane. The area follows from the
    /// card mass (`A = mass / (rho l)`); the effective modulus then
    /// follows from the axial rate (`E = EA / A`) so the section
    /// reproduces every rate exactly. With zero mass the area falls
    /// back to `A = EA / e` with the passed reference modulus. Shear
    /// factors solve `kappa = K l / (G A)` and clamp to `[0, 1]`; a
    /// zero transverse rate selects Euler-Bernoulli in that plane. The
    /// shear modulus follows from the effective modulus and `nu`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::{SpringGeneProp, SpringMode};
    ///
    /// let mode = SpringMode::new(1.0e6, 0.0);
    /// let prop = SpringGeneProp::new(7.8, 0.0, mode, mode, mode, mode, mode, mode);
    /// let section = prop.to_section(1.0, 200.0e9, 0.3, 7800.0);
    /// assert!((section.ea() - 1.0e6).abs() < 1.0e-3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for non-positive length, reference modulus, density, or
    /// axial rate with mass.
    #[must_use]
    pub fn to_section(&self, length: f64, e: f64, nu: f64, rho: f64) -> Section3D {
        assert!(
            length.is_finite() && length > 0.0,
            "length must be positive"
        );
        assert!(e.is_finite() && e > 0.0, "modulus must be positive");
        assert!(rho.is_finite() && rho > 0.0, "density must be positive");
        let ea = self.kx.stiffness * length;
        let (area, e_eff) = if self.mass > 0.0 {
            let area = self.mass / (rho * length);
            assert!(ea > 0.0, "axial rate must be positive with mass");
            (area, ea / area)
        } else {
            (ea / e, e)
        };
        let g = e_eff / (2.0 * (1.0 + nu));
        let torsion = self.rx.stiffness * length / g;
        let iy = self.ry.stiffness * length / e_eff;
        let iz = self.rz.stiffness * length / e_eff;
        let kappa = |k: f64| {
            if k <= 0.0 || area <= 0.0 {
                0.0
            } else {
                (k * length / (g * area)).clamp(0.0, 1.0)
            }
        };
        Section3D::new(e_eff, nu, area, iy, iz, torsion, rho)
            .with_shear(kappa(self.ky.stiffness), kappa(self.kz.stiffness))
    }
}

/// `/PROP/TYPE25 (SPR_AXI)` axisymmetric spring (Reference Guide pp. 1531-1534).
///
/// Four modes: tension/compression along the axis, radial shear,
/// torsion about the axis, and radial bending. The radial modes are
/// isotropic in the cross-section plane.
#[derive(Debug, Clone, Copy)]
pub struct SpringAxiProp {
    /// Spring mass in kg.
    pub mass: f64,
    /// Spring rotary inertia in kg m^2.
    pub inertia: f64,
    /// Tension/compression (axial) mode.
    pub tension: SpringMode,
    /// Radial shear mode.
    pub shear: SpringMode,
    /// Torsion mode.
    pub torsion: SpringMode,
    /// Radial bending mode.
    pub bend: SpringMode,
}

impl SpringAxiProp {
    /// Builds a TYPE25 card from mass and four modes.
    ///
    /// # Panics
    ///
    /// Panics for negative mass or inertia.
    #[must_use]
    pub fn new(
        mass: f64,
        inertia: f64,
        tension: SpringMode,
        shear: SpringMode,
        torsion: SpringMode,
        bend: SpringMode,
    ) -> Self {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "mass must be finite non-negative"
        );
        assert!(
            inertia.is_finite() && inertia >= 0.0,
            "inertia must be finite non-negative"
        );
        Self {
            mass,
            inertia,
            tension,
            shear,
            torsion,
            bend,
        }
    }

    /// Maps the card onto an equivalent [`Section3D`] at a length.
    ///
    /// Uses the TYPE8 mapping with the radial modes shared by both
    /// planes: `EA = K_t l`, `GJ = K_tor l`, `EIy = EIz = K_b l`, and
    /// one shear factor from `K_s` for both planes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::{SpringAxiProp, SpringMode};
    ///
    /// let stiff = SpringMode::new(2.0e6, 0.0);
    /// let soft = SpringMode::new(1.0e6, 0.0);
    /// let prop = SpringAxiProp::new(7.8, 0.0, stiff, soft, soft, soft);
    /// let section = prop.to_section(1.0, 200.0e9, 0.3, 7800.0);
    /// assert!((section.ea() - 2.0e6).abs() < 1.0e-3);
    /// assert!((section.eiy() - section.eiz()).abs() < 1.0e-9);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for non-positive length, modulus, or density.
    #[must_use]
    pub fn to_section(&self, length: f64, e: f64, nu: f64, rho: f64) -> Section3D {
        let gene = SpringGeneProp::new(
            self.mass,
            self.inertia,
            self.tension,
            self.shear,
            self.shear,
            self.torsion,
            self.bend,
            self.bend,
        );
        gene.to_section(length, e, nu, rho)
    }
}

/// Lock behavior of a `/PROP/TYPE32 (SPR_PRE)` pretension spring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreLock {
    /// Locked once shortened by the slide length or at zero force.
    OnSlideOrZeroForce = 1,
    /// Locked after any unloading.
    OnUnload = 2,
}

impl PreLock {
    /// Parses the `Ilock` deck flag (0 selects the default 1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::PreLock;
    ///
    /// assert_eq!(PreLock::from_flag(0), Ok(PreLock::OnSlideOrZeroForce));
    /// assert_eq!(PreLock::from_flag(2), Ok(PreLock::OnUnload));
    /// assert!(PreLock::from_flag(7).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a message for flags outside `{0, 1, 2}`.
    pub fn from_flag(flag: i32) -> Result<Self, String> {
        match flag {
            0 | 1 => Ok(Self::OnSlideOrZeroForce),
            2 => Ok(Self::OnUnload),
            other => Err(format!("TYPE32 Ilock must be 0, 1, or 2, got {other}")),
        }
    }
}

/// `/PROP/TYPE32 (SPR_PRE)` pretension spring (Reference Guide pp. 1563-1567).
///
/// Two-node axial spring with sensor-activated pretension: the piston
/// slides by `slide`, pulling with `force1` at activation, and locks
/// per `lock`. The linear model needs any two of force, slide, energy,
/// and loading stiffness; nonlinear `/FUNCT` loading stays driver-side
/// on [`Funct1D`](super::super::plasticity::funct::Funct1D).
#[derive(Debug, Clone, Copy)]
pub struct SpringPreProp {
    /// Spring mass in kg.
    pub mass: f64,
    /// Lock behavior flag.
    pub lock: PreLock,
    /// Stiffness before activation and after locking in N/m.
    pub stiff0: f64,
    /// Pretension force at activation in N.
    pub force1: f64,
    /// Piston slide length in m.
    pub slide: f64,
    /// Initial internal energy at activation in J.
    pub energy1: f64,
    /// Loading stiffness after activation in N/m.
    pub stiff1: f64,
}

impl SpringPreProp {
    /// Builds a TYPE32 card from mass, lock, and linear parameters.
    ///
    /// # Panics
    ///
    /// Panics for negative mass or stiffnesses, or negative slide.
    #[must_use]
    pub fn new(
        mass: f64,
        lock: PreLock,
        stiff0: f64,
        force1: f64,
        slide: f64,
        energy1: f64,
        stiff1: f64,
    ) -> Self {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "mass must be finite non-negative"
        );
        assert!(
            stiff0.is_finite() && stiff0 >= 0.0,
            "stiff0 must be finite non-negative"
        );
        assert!(
            stiff1.is_finite() && stiff1 >= 0.0,
            "stiff1 must be finite non-negative"
        );
        assert!(
            slide.is_finite() && slide >= 0.0,
            "slide must be finite non-negative"
        );
        Self {
            mass,
            lock,
            stiff0,
            force1,
            slide,
            energy1,
            stiff1,
        }
    }

    /// Maps the axial stiffness onto a planar [`BeamSection`].
    ///
    /// Uses `Stif0` (before activation / after locking) with `EA =
    /// Stif0 l` and the area from the card mass, like [`SpringGeneProp`].
    /// The effective modulus follows from `E = EA / A` so the section
    /// reproduces the rate; with zero mass the area falls back to
    /// `A = EA / e` with the passed reference modulus.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::beam::{PreLock, SpringPreProp};
    ///
    /// let prop = SpringPreProp::new(0.05, PreLock::OnSlideOrZeroForce, 1.0e5, 100.0, 0.25, 0.0, 0.0);
    /// let section = prop.to_beam_section(0.25, 200.0e9, 7800.0);
    /// assert!((section.ea() - 25000.0).abs() < 1.0e-6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for non-positive length, modulus, or density.
    #[must_use]
    pub fn to_beam_section(&self, length: f64, e: f64, rho: f64) -> BeamSection {
        assert!(
            length.is_finite() && length > 0.0,
            "length must be positive"
        );
        assert!(e.is_finite() && e > 0.0, "modulus must be positive");
        assert!(rho.is_finite() && rho > 0.0, "density must be positive");
        let ea = self.stiff0 * length;
        let (area, e_eff) = if self.mass > 0.0 {
            let area = self.mass / (rho * length);
            assert!(ea > 0.0, "axial rate must be positive with mass");
            (area, ea / area)
        } else {
            (ea / e, e)
        };
        // Axial-only card: bending inertia comes from the solid-circle
        // equivalent of the area so the section stays usable in 3-D.
        let radius = (area / std::f64::consts::PI).sqrt();
        let inertia = std::f64::consts::PI * radius.powi(4) / 4.0;
        BeamSection::euler_bernoulli(e_eff, inertia, area, rho)
    }

    /// Pretension force and locked stiffness for a [`Spring1D`](super::spring::Spring1D).
    ///
    /// Returns `(stiffness, pretension)`: the locked stiffness `Stif0`
    /// with the activation force as pretension. Slide-length locking
    /// stays driver-side (it needs the activation-time length).
    #[must_use]
    pub fn locked_spring(&self) -> (f64, f64) {
        (self.stiff0, self.force1)
    }
}

/// Equivalent spring rates of a beam section at a length.
///
/// Inverts the TYPE8 mapping: `K = EA / l` axially, `K = kappa G A /
/// l` in shear, `K = GJ / l` in torsion, and `K = EI / l` in bending.
/// Sections store no damping, so callers add viscous rates per mode.
#[derive(Debug, Clone, Copy)]
pub struct BeamRates {
    /// Axial rate in N/m.
    pub axial: f64,
    /// Shear rate along y in N/m.
    pub shear_y: f64,
    /// Shear rate along z in N/m.
    pub shear_z: f64,
    /// Torsion rate in N m/rad.
    pub torsion: f64,
    /// Bending rate about y in N m/rad.
    pub bend_y: f64,
    /// Bending rate about z in N m/rad.
    pub bend_z: f64,
}

/// Converts a [`Section3D`] to equivalent spring rates at a length.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::beam::{Section3D, beam_rates_from_section};
///
/// let section = Section3D::circular(200.0e9, 0.3, 0.01, 7800.0, false);
/// let rates = beam_rates_from_section(&section, 2.0);
/// assert!((rates.axial - section.ea() / 2.0).abs() < 1.0e-6);
/// ```
///
/// # Panics
///
/// Panics for non-positive length.
#[must_use]
pub fn beam_rates_from_section(section: &Section3D, length: f64) -> BeamRates {
    assert!(
        length.is_finite() && length > 0.0,
        "length must be positive"
    );
    BeamRates {
        axial: section.ea() / length,
        shear_y: section.shear_stiffness_y() / length,
        shear_z: section.shear_stiffness_z() / length,
        torsion: section.gj() / length,
        bend_y: section.eiy() / length,
        bend_z: section.eiz() / length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gene_mapping_round_trips_rates() {
        let stiff = SpringMode::new(3.0e6, 100.0);
        let shear = SpringMode::new(1.0e6, 50.0);
        let torsion = SpringMode::new(2.0e4, 10.0);
        let bend = SpringMode::new(5.0e4, 20.0);
        let prop = SpringGeneProp::new(7.8, 0.1, stiff, shear, shear, torsion, bend, bend);
        let section = prop.to_section(2.0, 200.0e9, 0.3, 7800.0);
        let rates = beam_rates_from_section(&section, 2.0);
        assert!((rates.axial - 3.0e6).abs() < 1.0e-3);
        assert!((rates.torsion - 2.0e4).abs() < 1e-6);
        assert!((rates.bend_y - 5.0e4).abs() < 1e-6);
        assert!((rates.bend_z - 5.0e4).abs() < 1e-6);
        // Area from mass: 7.8 / (7800 * 2) = 5e-4 m^2.
        assert!((section.area - 5.0e-4).abs() < 1e-15);
        // Shear clamps to Euler-Bernoulli when the rate exceeds G A / l.
        assert!(section.kappa_y <= 1.0);
    }

    #[test]
    fn gene_zero_shear_selects_euler_bernoulli() {
        let axial = SpringMode::new(1.0e6, 0.0);
        let none = SpringMode::new(0.0, 0.0);
        let prop = SpringGeneProp::new(0.0, 0.0, axial, none, none, none, axial, axial);
        let section = prop.to_section(1.0, 200.0e9, 0.3, 7800.0);
        assert!(!section.is_timoshenko());
        assert!((section.area - 1.0e6 / 200.0e9).abs() < 1e-15);
    }

    #[test]
    fn axi_radial_modes_are_isotropic() {
        let tension = SpringMode::new(2.0e6, 0.0);
        let shear = SpringMode::new(8.0e5, 0.0);
        let torsion = SpringMode::new(1.0e4, 0.0);
        let bend = SpringMode::new(3.0e4, 0.0);
        let prop = SpringAxiProp::new(3.9, 0.0, tension, shear, torsion, bend);
        let section = prop.to_section(1.0, 200.0e9, 0.3, 7800.0);
        assert!((section.iy - section.iz).abs() < 1e-12);
        assert!((section.kappa_y - section.kappa_z).abs() < 1e-15);
        let rates = beam_rates_from_section(&section, 1.0);
        assert!((rates.axial - 2.0e6).abs() < 1e-3);
        assert!((rates.torsion - 1.0e4).abs() < 1e-9);
    }

    #[test]
    fn pre_card_maps_locked_stiffness() {
        let prop = SpringPreProp::new(
            0.05,
            PreLock::OnSlideOrZeroForce,
            1.0e5,
            100.0,
            0.25,
            12.5,
            400.0,
        );
        let section = prop.to_beam_section(0.25, 200.0e9, 7800.0);
        assert!((section.ea() - 25000.0).abs() < 1e-6);
        assert!((section.area - 0.05 / (7800.0 * 0.25)).abs() < 1e-15);
        assert_eq!(prop.locked_spring(), (1.0e5, 100.0));
        assert_eq!(PreLock::from_flag(0).unwrap(), PreLock::OnSlideOrZeroForce);
        assert!(PreLock::from_flag(3).is_err());
    }

    #[test]
    fn mode_failure_checks_all_limits() {
        let mode = SpringMode::new(1000.0, 0.0)
            .with_fail_force(500.0)
            .with_fail_deformation(0.1)
            .with_fail_energy(25.0);
        assert!(mode.failed(500.0, 0.0, 0.0));
        assert!(mode.failed(-500.0, 0.0, 0.0));
        assert!(mode.failed(0.0, 0.1, 0.0));
        assert!(mode.failed(0.0, 0.0, 25.0));
        assert!(!mode.failed(499.0, 0.09, 24.9));
    }

    #[test]
    fn circular_section_rates_match_closed_form() {
        let section = Section3D::circular(200.0e9, 0.3, 0.02, 7800.0, false);
        let rates = beam_rates_from_section(&section, 1.5);
        let area = std::f64::consts::PI * 0.02 * 0.02;
        assert!((rates.axial - 200.0e9 * area / 1.5).abs() < 1e-3);
        assert!((rates.torsion - section.gj() / 1.5).abs() < 1e-9);
        assert!((rates.bend_y - section.eiy() / 1.5).abs() < 1e-9);
        assert_eq!(rates.shear_y, 0.0);
    }
}
