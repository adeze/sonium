//! Bao-Xue-Wierzbicki triaxiality-Lode fracture locus (`/FAIL/WIERZBICKI`).
//!
//! Implements the 3D ductile fracture criterion: the fracture strain is an
//! elliptical blend between upper and lower triaxiality branches, driven by
//! the normalized third invariant. Damage accumulates linearly and the
//! point fails at unity, the shared cumulative-damage convention.
//!
//! - Locus `ef = {emax^n - [emax^n - emin^n](1 - |xi|^m)^(1/m)}^(1/n)`
//!   (Theory Manual (1067), Reference Guide (974)) with the branches
//!   `emax = C1 exp(-C2 eta)` (1068) and `emin = C3 exp(-C4 eta)` (1069).
//!   The locus is symmetric in `xi` (Theory Manual Figure 245), so the
//!   implementation folds `|xi|`; this matches the card exactly for the
//!   customary even `m` and extends it continuously otherwise.
//! - Solids with `Imoy = 0` use instantaneous `eta = sm/sVM` and
//!   `xi = 27 J3/(2 sVM^3)`; `Imoy = 1` averages both over the plastic
//!   strain path. Shells use `xi = -(27/2) eta (eta^2 - 1/3)`, the
//!   plane-stress link between the invariants.
//! - Equivalent plastic strain, triaxiality, Lode parameter, and step time
//!   arrive as P3C3 scalars; the mean-state struct carries `Imoy = 1`
//!   path averages without touching material state.
//!
//! # Example
//!
//! ```rust
//! use sonium_fem::physics::solid::fracture::wierzbicki::{Wierzbicki, xi_shell};
//!
//! let locus = Wierzbicki::deck_shell();
//! // Uniaxial shell tension sits at xi = 1, on the upper branch.
//! let eta = 1.0 / 3.0;
//! assert!((xi_shell(eta) - 1.0).abs() < 1e-12);
//! let ef = locus.fracture_strain(eta, xi_shell(eta));
//! assert!((ef - locus.emax(eta)).abs() < 1e-12);
//! ```

// Rust guideline compliant 2026-02-21

use crate::physics::solid::plasticity::update::{StressUpdate, von_mises_c1};

/// Floor for the von Mises stress in invariant ratios.
///
/// Keeps triaxiality and the normalized third invariant bounded at zero
/// stress; small enough to never shift a physical locus evaluation.
const STRESS_FLOOR: f64 = 1e-30;

/// Floor for the fracture strain in damage accumulation.
///
/// Matches the tabulated-failure floor so both lane legs treat
/// degenerate loci identically.
const FRACTURE_STRAIN_FLOOR: f64 = 1e-9;

/// Floor for positive step times when estimating strain rate.
const STEP_TIME_FLOOR: f64 = 1e-15;

/// Determinant of the deviatoric part of a C1 Voigt stress.
///
/// Returns `J3 = det(s)` with `s = sigma - sm I`, the third invariant
/// feeding the normalized Lode coordinate. Closed form in the tensor
/// components, exact to rounding.
pub fn j3_c1(stress: &[f64; 6]) -> f64 {
    let mean = (stress[0] + stress[1] + stress[2]) / 3.0;
    let s0 = stress[0] - mean;
    let s1 = stress[1] - mean;
    let s2 = stress[2] - mean;
    let (s3, s4, s5) = (stress[3], stress[4], stress[5]);
    s0 * s1 * s2 + 2.0 * s3 * s4 * s5 - s0 * s4 * s4 - s1 * s5 * s5 - s2 * s3 * s3
}

/// Normalized third invariant of a C1 Voigt stress.
///
/// Returns `xi = 27 J3/(2 q^3)` with the von Mises stress `q`, the solid
/// `Imoy = 0` Lode coordinate (Theory Manual p.378). Clamped to [-1, 1];
/// zero stress maps to 0 (pure shear limit).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::wierzbicki::xi_solid;
///
/// // Uniaxial tension is axisymmetric: xi = 1.
/// let uni = [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0];
/// assert!((xi_solid(&uni) - 1.0).abs() < 1e-12);
/// // Pure shear sits at the locus minimum: xi = 0.
/// let shear = [0.0, 0.0, 0.0, 100.0e6, 0.0, 0.0];
/// assert!(xi_solid(&shear).abs() < 1e-12);
/// ```
pub fn xi_solid(stress: &[f64; 6]) -> f64 {
    let q = von_mises_c1(stress);
    if q <= 0.0 {
        return 0.0;
    }
    let xi = 27.0 * j3_c1(stress) / (2.0 * q * q * q.max(STRESS_FLOOR));
    xi.clamp(-1.0, 1.0)
}

/// Shell Lode coordinate from stress triaxiality.
///
/// Returns `xi = -(27/2) eta (eta^2 - 1/3)`, the plane-stress relation
/// (Theory Manual p.379, Reference Guide p.1464). Clamped to [-1, 1];
/// uniaxial tension (`eta = 1/3`) maps to 1, shear to 0.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::wierzbicki::xi_shell;
///
/// assert!((xi_shell(1.0 / 3.0) - 1.0).abs() < 1e-12);
/// assert!(xi_shell(0.0).abs() < 1e-15);
/// ```
pub fn xi_shell(triaxiality: f64) -> f64 {
    let eta = triaxiality;
    (-13.5 * eta * (eta * eta - 1.0 / 3.0)).clamp(-1.0, 1.0)
}

/// Bao-Xue-Wierzbicki fracture locus with linear damage.
///
/// Stores the five locus constants plus the hardening exponent; all are
/// dimensionless. Damage accumulates as `D += dep/ef(eta, xi)` and the
/// point fails at `D = 1`, the cumulative convention shared with the
/// Johnson-Cook law (Theory Manual p.376).
#[derive(Debug, Clone, Copy)]
pub struct Wierzbicki {
    /// Upper-branch prefactor `C1`, dimensionless.
    pub c1: f64,
    /// Upper-branch triaxiality sensitivity `C2`, dimensionless.
    pub c2: f64,
    /// Lower-branch prefactor `C3`, dimensionless.
    pub c3: f64,
    /// Lower-branch triaxiality sensitivity `C4`, dimensionless.
    pub c4: f64,
    /// Lode exponent `m`, positive (customarily even).
    pub m: f64,
    /// Hardening exponent `n`, positive, doubling as locus exponent.
    pub n: f64,
}

impl Wierzbicki {
    /// Builds a locus from the six card parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::wierzbicki::Wierzbicki;
    ///
    /// let locus = Wierzbicki::new(0.87, 1.77, 0.21, 0.56, 2.0, 0.135).unwrap();
    /// assert!(locus.emax(0.0) > locus.emin(0.0));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a message when any input is non-finite, a prefactor is
    /// negative, or an exponent is not positive.
    pub fn new(c1: f64, c2: f64, c3: f64, c4: f64, m: f64, n: f64) -> Result<Self, String> {
        for (name, value) in [
            ("C1", c1),
            ("C2", c2),
            ("C3", c3),
            ("C4", c4),
            ("m", m),
            ("n", n),
        ] {
            if !value.is_finite() {
                return Err(format!("wierzbicki {name} must be finite"));
            }
        }
        if c1 < 0.0 || c3 < 0.0 {
            return Err("wierzbicki prefactors C1 and C3 must be non-negative".to_string());
        }
        if m <= 0.0 || n <= 0.0 {
            return Err("wierzbicki exponents m and n must be positive".to_string());
        }
        Ok(Self {
            c1,
            c2,
            c3,
            c4,
            m,
            n,
        })
    }

    /// Miniqa `input_51` shell locus for rupture validation.
    ///
    /// Bakes the `RUPTURE/Wierzbcki/User_law/shell` deck parameters
    /// (`C1` 0.87, `C2` 1.77, `C3` 0.21, `C4` 0.56, `m` 2, `n` 0.135).
    ///
    /// # Panics
    ///
    /// Panics when the baked deck parameters violate validation; they
    /// are fixed literals proven by the constructor tests.
    pub fn deck_shell() -> Self {
        Self::new(0.87, 1.77, 0.21, 0.56, 2.0, 0.135).expect("deck locus is valid")
    }

    /// Upper fracture branch `emax = C1 exp(-C2 eta)` (1068).
    ///
    /// Governs axisymmetric states (`|xi| = 1`).
    pub fn emax(&self, triaxiality: f64) -> f64 {
        self.c1 * (-self.c2 * triaxiality).exp()
    }

    /// Lower fracture branch `emin = C3 exp(-C4 eta)` (1069).
    ///
    /// Governs generalized shear (`xi = 0`).
    pub fn emin(&self, triaxiality: f64) -> f64 {
        self.c3 * (-self.c4 * triaxiality).exp()
    }

    /// Fracture strain at triaxiality and Lode coordinate (974).
    ///
    /// Evaluates the elliptical blend between the branches, floored to
    /// keep damage accumulation finite. Exact at the endpoints: `|xi| = 1`
    /// returns `emax`, `xi = 0` returns `emin`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::wierzbicki::Wierzbicki;
    ///
    /// let locus = Wierzbicki::deck_shell();
    /// assert!((locus.fracture_strain(0.25, 1.0) - locus.emax(0.25)).abs() < 1e-12);
    /// assert!((locus.fracture_strain(0.25, 0.0) - locus.emin(0.25)).abs() < 1e-12);
    /// ```
    pub fn fracture_strain(&self, triaxiality: f64, xi: f64) -> f64 {
        let top = self.emax(triaxiality).max(0.0).powf(self.n);
        let bottom = self.emin(triaxiality).max(0.0).powf(self.n);
        let span = (top - bottom).max(0.0);
        // Symmetric locus (Figure 245): fold |xi| so odd or fractional m
        // stays continuous; identical to xi^m for even m.
        let lode = 1.0 - xi.abs().clamp(0.0, 1.0).powf(self.m);
        let blended = top - span * lode.max(0.0).powf(1.0 / self.m);
        blended
            .max(0.0)
            .powf(1.0 / self.n)
            .max(FRACTURE_STRAIN_FLOOR)
    }

    /// Damage increment at an explicit state (P3C3 criterion).
    ///
    /// Divides the plastic increment from `update` by the locus value at
    /// the passed triaxiality and Lode coordinate. For `Imoy = 1` paths
    /// the caller passes the running means from [`WierzbickiMean`].
    /// Non-positive increments yield zero; `dt` only guards degenerate
    /// steps since the locus is rate-independent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::wierzbicki::Wierzbicki;
    /// use sonium_fem::physics::solid::plasticity::update::StressUpdate;
    ///
    /// let locus = Wierzbicki::deck_shell();
    /// let update = StressUpdate {
    ///     stress: [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0],
    ///     tangent: [0.0; 36],
    ///     delta_equiv_plastic_strain: 0.01,
    ///     yielding: true,
    /// };
    /// let step = locus.increment(&update, 0.1, 1.0 / 3.0, 1.0, 1.0e-3);
    /// assert!((step - 0.01 / locus.emax(1.0 / 3.0)).abs() < 1e-15);
    /// ```
    pub fn increment(
        &self,
        update: &StressUpdate,
        ep: f64,
        triaxiality: f64,
        lode: f64,
        dt: f64,
    ) -> f64 {
        let _ = (ep, dt.max(STEP_TIME_FLOOR));
        let dep = update.delta_equiv_plastic_strain;
        if dep <= 0.0 {
            return 0.0;
        }
        dep / self.fracture_strain(triaxiality, lode)
    }

    /// Reports whether accumulated damage fractures the point.
    ///
    /// Fracture trips at `damage >= 1`, the cumulative-damage convention.
    pub fn fractured(&self, damage: f64) -> bool {
        let _ = self;
        damage >= 1.0
    }
}

/// Plastic-strain-averaged invariants for `Imoy = 1` paths.
///
/// Accumulates `eta . dep` and `xi . dep` numerators plus the strain
/// denominator, so [`means`](Self::means) returns the path averages
/// `etabar` and `xibar` (Theory Manual p.379). Virgin state reports the
/// shear limit (0, 0).
#[derive(Debug, Clone, Copy)]
pub struct WierzbickiMean {
    /// Accumulated `eta . dep` numerator.
    num_eta: f64,
    /// Accumulated `xi . dep` numerator.
    num_xi: f64,
    /// Accumulated plastic strain denominator.
    ep: f64,
}

impl WierzbickiMean {
    /// Creates virgin averaging state.
    pub fn new() -> Self {
        Self {
            num_eta: 0.0,
            num_xi: 0.0,
            ep: 0.0,
        }
    }

    /// Observes one plastic step at instantaneous invariants.
    ///
    /// Adds `eta . dep` and `xi . dep` to the numerators; non-positive
    /// increments leave the state untouched.
    pub fn observe(&mut self, triaxiality: f64, xi: f64, dep: f64) {
        if dep <= 0.0 {
            return;
        }
        self.num_eta += triaxiality * dep;
        self.num_xi += xi * dep;
        self.ep += dep;
    }

    /// Returns the path-averaged `(etabar, xibar)` pair.
    ///
    /// Virgin state yields (0, 0), the shear limit.
    pub fn means(&self) -> (f64, f64) {
        if self.ep <= 0.0 {
            return (0.0, 0.0);
        }
        (self.num_eta / self.ep, self.num_xi / self.ep)
    }

    /// Reports the accumulated plastic strain.
    pub fn strain(&self) -> f64 {
        self.ep
    }
}

impl Default for WierzbickiMean {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branches_match_closed_forms() {
        let locus = Wierzbicki::deck_shell();
        let eta = 1.0f64 / 3.0;
        assert!((locus.emax(eta) - 0.87 * (-1.77 * eta).exp()).abs() < 1e-15);
        assert!((locus.emin(eta) - 0.21 * (-0.56 * eta).exp()).abs() < 1e-15);
        // Compression raises both branches (negative exponent slope).
        assert!(locus.emax(-0.5) > locus.emax(0.5));
        assert!(locus.emin(-0.5) > locus.emin(0.5));
    }

    #[test]
    fn locus_hits_endpoints_and_stays_between() {
        let locus = Wierzbicki::deck_shell();
        for eta in [-0.5, 0.0, 1.0 / 3.0, 0.6] {
            let top = locus.emax(eta);
            let bottom = locus.emin(eta);
            assert!(
                (locus.fracture_strain(eta, 1.0) - top).abs() < 1e-12,
                "eta {eta}"
            );
            assert!(
                (locus.fracture_strain(eta, -1.0) - top).abs() < 1e-12,
                "eta {eta}"
            );
            assert!(
                (locus.fracture_strain(eta, 0.0) - bottom).abs() < 1e-12,
                "eta {eta}"
            );
            let mid = locus.fracture_strain(eta, 0.5);
            assert!(mid > bottom && mid < top, "eta {eta}: mid {mid}");
        }
    }

    #[test]
    fn locus_matches_hand_derivation() {
        // Independent evaluation of (974) with m = 2, n = 0.135.
        let locus = Wierzbicki::deck_shell();
        let eta = 0.5f64;
        let xi = xi_shell(eta);
        assert!((xi - 0.5625).abs() < 1e-12, "xi {xi}");
        let top = 0.87f64 * (-1.77 * eta).exp();
        let bottom = 0.21f64 * (-0.56 * eta).exp();
        let expected = (top.powf(0.135)
            - (top.powf(0.135) - bottom.powf(0.135)) * (1.0 - xi * xi).sqrt())
        .powf(1.0 / 0.135);
        let got = locus.fracture_strain(eta, xi);
        assert!((got - expected).abs() < 1e-15, "got {got}, want {expected}");
    }

    #[test]
    fn shell_lode_covers_reference_states() {
        assert!((xi_shell(1.0 / 3.0) - 1.0).abs() < 1e-12);
        assert!((xi_shell(-1.0 / 3.0) + 1.0).abs() < 1e-12);
        assert!((xi_shell(2.0 / 3.0) + 1.0).abs() < 1e-12);
        assert!(xi_shell(0.0).abs() < 1e-15);
    }

    #[test]
    fn solid_xi_matches_principal_states() {
        // Uniaxial, equibiaxial, and hydrostatic-tension limits.
        let uni = [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((xi_solid(&uni) - 1.0).abs() < 1e-12);
        let equi = [200.0e6, 200.0e6, 0.0, 0.0, 0.0, 0.0];
        assert!((xi_solid(&equi) + 1.0).abs() < 1e-12);
        // Hydrostatic stress has no direction: shear limit.
        assert_eq!(xi_solid(&[100.0e6, 100.0e6, 100.0e6, 0.0, 0.0, 0.0]), 0.0);
        assert_eq!(xi_solid(&[0.0; 6]), 0.0);
    }

    #[test]
    fn constructor_rejects_bad_cards() {
        assert!(Wierzbicki::new(0.87, 1.77, 0.21, 0.56, 2.0, 0.135).is_ok());
        assert!(Wierzbicki::new(-0.1, 1.77, 0.21, 0.56, 2.0, 0.135).is_err());
        assert!(Wierzbicki::new(0.87, 1.77, 0.21, 0.56, 0.0, 0.135).is_err());
        assert!(Wierzbicki::new(0.87, 1.77, 0.21, 0.56, 2.0, -1.0).is_err());
        assert!(Wierzbicki::new(f64::NAN, 1.77, 0.21, 0.56, 2.0, 0.135).is_err());
    }

    #[test]
    fn mean_state_averages_path_invariants() {
        let mut mean = WierzbickiMean::new();
        assert_eq!(mean.means(), (0.0, 0.0));
        mean.observe(0.5, 1.0, 0.2);
        mean.observe(-0.1, 0.0, 0.2);
        let (eta, xi) = mean.means();
        assert!((eta - 0.2).abs() < 1e-15);
        assert!((xi - 0.5).abs() < 1e-15);
        assert!((mean.strain() - 0.4).abs() < 1e-15);
        // Elastic steps never dilute the averages.
        mean.observe(99.0, 99.0, 0.0);
        assert!((mean.means().0 - 0.2).abs() < 1e-15);
    }

    #[test]
    fn increment_accumulates_linearly_and_fractures_at_one() {
        let locus = Wierzbicki::deck_shell();
        let mk = |dep| StressUpdate {
            stress: [0.0; 6],
            tangent: [0.0; 36],
            delta_equiv_plastic_strain: dep,
            yielding: true,
        };
        assert_eq!(locus.increment(&mk(0.0), 0.0, 0.0, 0.0, 1.0e-3), 0.0);
        let ef = locus.emin(0.0);
        let step = locus.increment(&mk(0.01), 0.0, 0.0, 0.0, 1.0e-3);
        assert!((step - 0.01 / ef).abs() < 1e-15);
        assert!(!locus.fractured(0.999));
        assert!(locus.fractured(1.0));
    }
}
