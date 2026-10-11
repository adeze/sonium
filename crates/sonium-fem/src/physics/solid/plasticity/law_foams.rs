//! Foam and extended crash material laws.
//!
//! Implements `/MAT` laws 50 (`VISC_HONEY`), 51 (`MULTIMAT`), 68
//! (`COSSER`), 76 (`SAMP`), 28 (`HONEYCOMB`), 19 (`FABRI`), 58
//! (nonlinear fabric), 24 (`CONC`), and 33 (`FOAM_PLAS`) as frozen C1
//! [`StressUpdateModel`](super::update::StressUpdateModel) materials.
//! Honeycomb theory comes from Theory Manual p.304-307 equations
//! 847-853; concrete from p.345-348 equation 966; foam visco response
//! from p.365-366 equations 1024-1027; SAMP from Reference Guide
//! p.613-622 equations 296-300; MULTIMAT from p.1136 equations
//! 717-721; JWL pressure from p.1054-1055 equations 651-654. Each law
//! cites its exact source at the item.
//!
//! Laws holding curve data reuse the frozen
//! [`Funct1D`](super::funct::Funct1D) and [`Table2D`](super::funct::Table2D)
//! types read-only, and isotropic crushable-foam checks reuse the
//! [`CrushableFoamMaterial`](super::crash::CrushableFoamMaterial) seed.
//! Every law exposes a native `step` method plus the C1 `update` and
//! `commit` pair. Laws tracking totals (strains, energies, volumes)
//! advance them in `advance_totals` after `commit`, because `commit`
//! never sees the strain increment; drivers must call it once per
//! accepted step or use `step`, which bundles all three.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::funct::Funct1D;
//! use sonium_fem::physics::solid::plasticity::law_foams::{HoneycombLaw, HoneycombState};
//! use sonium_fem::physics::solid::plasticity::update::StressUpdateModel;
//!
//! let flat = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
//! let law = HoneycombLaw::law28([2.0; 6], [flat.clone(), flat.clone(), flat.clone(),
//!     flat.clone(), flat.clone(), flat.clone()]).unwrap();
//! let mut state = HoneycombState::new();
//! let elastic = law.update(&state, &[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! assert!(!elastic.yielding);
//! law.commit(&mut state, &elastic);
//! let capped = law.update(&state, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! assert!(capped.yielding);
//! ```
//!
//! # Panics
//!
//! Constructors return `Result` and never panic; stepping never
//! panics on finite inputs. Non-finite inputs propagate NaN.
//!
//! # Safety
//!
//! Pure safe Rust with no unsafe code.

// Rust guideline compliant 2026-02-21

use super::crash::{CrushableFoamMaterial, CrushableFoamState, UniaxialPoint, elastic_tangent_c1};
use super::funct::{Funct1D, Table2D};
use super::update::{StressUpdate, StressUpdateModel, voigt_equivalent_increment_c1, von_mises_c1};

/// C1 Voigt order `[xx, yy, zz, xy, yz, zx]` used everywhere here.
pub type Voigt6 = [f64; 6];

/// Pressure from a C1 stress, positive in compression (Pa).
///
/// Matches Reference Guide equation 297 (`P = -(xx + yy + zz) / 3`).
#[inline]
pub fn pressure_c1(stress: &Voigt6) -> f64 {
    -(stress[0] + stress[1] + stress[2]) / 3.0
}

/// Deviatoric part of a C1 stress in Voigt order (Pa).
#[inline]
pub fn deviator_c1(stress: &Voigt6) -> Voigt6 {
    let mean = (stress[0] + stress[1] + stress[2]) / 3.0;
    [
        stress[0] - mean,
        stress[1] - mean,
        stress[2] - mean,
        stress[3],
        stress[4],
        stress[5],
    ]
}

/// Principal stresses of a symmetric C1 stress via Jacobi iteration.
///
/// Runs twelve cyclic sweeps, enough for full double precision on
/// 3x3 matrices; returns values in no particular order.
pub fn principal_stresses_c1(stress: &Voigt6) -> [f64; 3] {
    let (values, _) = jacobi_eigen_c1(stress, 12);
    values
}

/// Symmetric 3x3 eigen-decomposition for principal caps.
///
/// Returns `(values, vectors)` with eigenvectors in columns. Used by
/// the LAW33 principal-stress cap; twelve sweeps converge 3x3 fully.
pub fn jacobi_eigen_c1(stress: &Voigt6, sweeps: usize) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [
        [stress[0], stress[3], stress[5]],
        [stress[3], stress[1], stress[4]],
        [stress[5], stress[4], stress[2]],
    ];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..sweeps {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
            let (c, s) = (phi.cos(), phi.sin());
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            // Every swept pair keeps p < q, so split borrows both rows.
            let (head, tail) = a.split_at_mut(q);
            let row_p = &mut head[p];
            let row_q = &mut tail[0];
            for (apk, aqk) in row_p.iter_mut().zip(row_q.iter_mut()) {
                let (old_p, old_q) = (*apk, *aqk);
                *apk = c * old_p - s * old_q;
                *aqk = s * old_p + c * old_q;
            }
            for row in v.iter_mut() {
                let vkp = row[p];
                let vkq = row[q];
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Diagonal C1 tangent from six direction moduli, row-major.
///
/// Honeycomb and fabric laws use this elastic branch; rows and
/// columns follow `[xx, yy, zz, xy, yz, zx]`.
pub fn diag_tangent_c1(moduli: &[f64; 6]) -> [f64; 36] {
    let mut tangent = [0.0; 36];
    for (dir, modulus) in moduli.iter().enumerate() {
        tangent[dir * 6 + dir] = *modulus;
    }
    tangent
}

/// Algorithmic C1 tangent by central differences of a stress map.
///
/// Mirrors the [`CrushableFoamMaterial`] yielding branch: perturbs
/// each strain component by `1e-9`, differences the returned
/// stresses, and enforces major symmetry. Serves quasi-static
/// Newton loops; explicit dynamics uses the elastic stiffness.
pub fn numeric_tangent_c1(stress_map: impl Fn(&Voigt6) -> Voigt6, dstrain: &Voigt6) -> [f64; 36] {
    let eps = 1e-9;
    let mut tangent = [0.0; 36];
    for j in 0..6 {
        let mut plus = *dstrain;
        let mut minus = *dstrain;
        plus[j] += eps;
        minus[j] -= eps;
        let s_plus = stress_map(&plus);
        let s_minus = stress_map(&minus);
        for i in 0..6 {
            tangent[i * 6 + j] = (s_plus[i] - s_minus[i]) / (2.0 * eps);
        }
    }
    for i in 0..6 {
        for j in (i + 1)..6 {
            let mean = 0.5 * (tangent[i * 6 + j] + tangent[j * 6 + i]);
            tangent[i * 6 + j] = mean;
            tangent[j * 6 + i] = mean;
        }
    }
    tangent
}

/// C1 update of a failed point: zero stress and tangent.
///
/// Failed points report `yielding` so drivers keep stepping
/// over deleted elements without dividing by stiffness.
pub fn zeroed_update() -> StressUpdate {
    StressUpdate {
        stress: [0.0; 6],
        tangent: [0.0; 36],
        delta_equiv_plastic_strain: 0.0,
        yielding: true,
    }
}

/// Rebuilds a Voigt stress from principal values and directions.
///
/// `vectors` holds eigenvectors in columns, as returned by
/// [`jacobi_eigen_c1`].
pub fn voigt_from_principal(values: &[f64; 3], vectors: &[[f64; 3]; 3]) -> Voigt6 {
    let mut full = [[0.0; 3]; 3];
    for (k, value) in values.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                full[i][j] += value * vectors[i][k] * vectors[j][k];
            }
        }
    }
    [
        full[0][0], full[1][1], full[2][2], full[0][1], full[1][2], full[0][2],
    ]
}

/// Checks `von_mises_c1` against the direct formula for doctests.
#[cfg(test)]
pub(crate) fn von_mises_direct(stress: &Voigt6) -> f64 {
    let s = deviator_c1(stress);
    (1.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .sqrt()
}

// ---------------------------------------------------------------------------
// Honeycomb pseudo-plastic orthotropic solids (LAW28, LAW50)
// ---------------------------------------------------------------------------

/// Abscissa convention of one honeycomb yield curve family.
///
/// Theory Manual p.305 Figure 193 with Reference Guide `/MAT/LAW50`
/// `Iflag1`/`Iflag2`: volumetric strain `mu` counts compression
/// positive, component strain counts tension positive, and -1 keeps
/// the volumetric sign convention on component values by negation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrainBasis {
    /// Yield versus volumetric strain `mu = rho / rho0 - 1`.
    Volumetric,
    /// Yield versus signed component strain.
    Component,
    /// Yield versus negated component strain.
    Negated,
}

impl StrainBasis {
    /// Converts the card flag 0, 1, or -1.
    ///
    /// # Errors
    ///
    /// Returns a message for any other flag value.
    pub fn from_flag(flag: i32) -> Result<Self, String> {
        match flag {
            0 => Ok(Self::Volumetric),
            1 => Ok(Self::Component),
            -1 => Ok(Self::Negated),
            other => Err(format!(
                "honeycomb Iflag must be -1, 0, or 1, found {other}"
            )),
        }
    }

    /// Evaluates the curve abscissa from totals.
    #[inline]
    pub fn abscissa(&self, component_strain: f64, mu: f64) -> f64 {
        match self {
            Self::Volumetric => mu,
            Self::Component => component_strain,
            Self::Negated => -component_strain,
        }
    }
}

/// One direction yield curve, optionally rate dependent.
///
/// `Single` is LAW28 input (one function per direction).
/// `Rated` is LAW50 input (up to five `(function, strain rate)`
/// pairs); evaluation interpolates linearly between the bracketing
/// rate curves and clamps outside, matching bilinear table lookup
/// on identical grids.
#[derive(Debug, Clone)]
pub enum YieldTable {
    /// Rate-independent yield curve.
    Single(Funct1D),
    /// Rate-interpolated yield curves with ascending rates.
    Rated(Vec<(Funct1D, f64)>),
}

impl YieldTable {
    /// Evaluates yield stress at abscissa and rate (Pa).
    pub fn eval(&self, x: f64, rate: f64) -> f64 {
        match self {
            Self::Single(funct) => funct.eval(x),
            Self::Rated(curves) => rated_eval(curves, x, rate),
        }
    }
}

/// Interpolates rated curves at abscissa and rate.
///
/// Shared by [`YieldTable::Rated`] and laws holding bare
/// `(function, rate)` lists; clamps outside the rate range.
pub fn rated_eval(curves: &[(Funct1D, f64)], x: f64, rate: f64) -> f64 {
    if curves.is_empty() {
        return 0.0;
    }
    let first = curves[0].1;
    if rate <= first || curves.len() == 1 {
        return curves[0].0.eval(x);
    }
    let last = curves.len() - 1;
    if rate >= curves[last].1 {
        return curves[last].0.eval(x);
    }
    let upper = curves.partition_point(|(_, r)| *r <= rate);
    let (flo, rlo) = &curves[upper - 1];
    let (fhi, rhi) = &curves[upper];
    let frac = (rate - rlo) / (rhi - rlo);
    flo.eval(x) + frac * (fhi.eval(x) - flo.eval(x))
}

/// Strain-rate grouping of a honeycomb law.
///
/// `iratetime = 2` filters each direction independently (the card
/// default); `1` filters one equivalent rate shared by all
/// directions, `sqrt(sum normal^2 + sum shear^2 / 2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateMode {
    /// Independent filtered rate per direction.
    #[default]
    PerDirection,
    /// One filtered equivalent rate for all directions.
    Common,
}

/// Honeycomb pseudo-plastic orthotropic solid.
///
/// Diagonal orthotropic elasticity per Theory Manual equation 847
/// with independent per-direction yield curves per Figure 193:
/// each component advances elastically, then its magnitude caps at
/// the curve value with the trial sign kept. `law28` builds the
/// rate-independent LAW28 form; `law50` builds the rate-dependent
/// LAW50 (`VISC_HONEY`) form. Curves stay positive; failure
/// compares tension totals and half shear totals against limits.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
/// use sonium_fem::physics::solid::plasticity::law_foams::HoneycombLaw;
///
/// let cap = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
/// let tables = [cap.clone(), cap.clone(), cap.clone(),
///     cap.clone(), cap.clone(), cap];
/// let law = HoneycombLaw::law28([2.0; 6], tables).unwrap();
/// assert!((law.yield_stress(0, 0.0, 0.0) - 0.5).abs() < 1.0e-15);
/// ```
///
/// # Errors
///
/// Constructors reject non-positive moduli and empty rate lists.
#[derive(Debug, Clone)]
pub struct HoneycombLaw {
    /// Diagonal moduli `[E11, E22, E33, G12, G23, G31]` (Pa).
    pub moduli: [f64; 6],
    /// Yield curves per C1 direction.
    pub tables: [YieldTable; 6],
    /// Abscissa basis of the normal directions.
    pub basis_normal: StrainBasis,
    /// Abscissa basis of the shear directions.
    pub basis_shear: StrainBasis,
    /// Failure strain per direction (`INFINITY` disables).
    pub failure: [f64; 6],
    /// Rate grouping for LAW50 tables.
    pub rate_mode: RateMode,
    /// Strain-rate filter cutoff (1/s); `INFINITY` disables filtering.
    pub rate_cutoff: f64,
}

impl HoneycombLaw {
    /// Builds the LAW28 rate-independent form from six curves.
    ///
    /// Orders curves in C1 order `[11, 22, 33, 12, 23, 31]`.
    /// Defaults use component strain bases, no failure, and no
    /// filtering.
    ///
    /// # Errors
    ///
    /// Returns a message when any modulus is not positive.
    pub fn law28(moduli: [f64; 6], curves: [Funct1D; 6]) -> Result<Self, String> {
        Self::checked(moduli, curves.map(YieldTable::Single))
    }

    /// Builds the LAW50 rate-dependent form from rated curves.
    ///
    /// Each direction holds `(function, strain rate)` pairs with
    /// ascending rates; per Reference Guide `/MAT/LAW50` the
    /// default bases negate component strain (`Iflag = -1`).
    ///
    /// # Errors
    ///
    /// Returns a message for non-positive moduli, empty rate
    /// lists, or non-ascending rates.
    pub fn law50(moduli: [f64; 6], rated: [Vec<(Funct1D, f64)>; 6]) -> Result<Self, String> {
        for (dir, curves) in rated.iter().enumerate() {
            if curves.is_empty() {
                return Err(format!(
                    "law50 direction {dir} needs at least one rated curve"
                ));
            }
            for pair in curves.windows(2) {
                if pair[1].1 <= pair[0].1 {
                    return Err(format!("law50 direction {dir} rates must ascend"));
                }
            }
        }
        let tables = rated.map(YieldTable::Rated);
        let mut law = Self::checked(moduli, tables)?;
        law.basis_normal = StrainBasis::Negated;
        law.basis_shear = StrainBasis::Negated;
        Ok(law)
    }

    /// Validates moduli and installs defaults.
    fn checked(moduli: [f64; 6], tables: [YieldTable; 6]) -> Result<Self, String> {
        for (dir, modulus) in moduli.iter().enumerate() {
            if !modulus.is_finite() || *modulus <= 0.0 {
                return Err(format!(
                    "honeycomb modulus {dir} must be positive, found {modulus}"
                ));
            }
        }
        Ok(Self {
            moduli,
            tables,
            basis_normal: StrainBasis::Component,
            basis_shear: StrainBasis::Component,
            failure: [f64::INFINITY; 6],
            rate_mode: RateMode::PerDirection,
            rate_cutoff: f64::INFINITY,
        })
    }

    /// Yield stress of one direction at abscissa and rate (Pa).
    pub fn yield_stress(&self, dir: usize, x: f64, rate: f64) -> f64 {
        self.tables[dir].eval(x, rate).abs()
    }

    /// Failure flag for totals per Reference Guide card notes.
    ///
    /// Normal directions fail when the tensile total exceeds its
    /// limit; shear directions fail when half the total shear
    /// exceeds the limit.
    pub fn failed(&self, strains: &Voigt6) -> bool {
        strains.iter().enumerate().any(|(dir, strain)| {
            self.failure[dir].is_finite() && self.failure_measure(dir, *strain) > self.failure[dir]
        })
    }

    /// Failure measure of one direction (tension or half shear).
    fn failure_measure(&self, dir: usize, strain: f64) -> f64 {
        if dir < 3 {
            strain.max(0.0)
        } else {
            0.5 * strain.abs()
        }
    }

    /// Filtered rate factor from cutoff and time step.
    ///
    /// Matches the `smorth3.F` exponential smoothing kernel
    /// `alpha = 4 pi fc dt / (1 + 4 pi fc dt)`.
    fn rate_factor(&self, dt: f64) -> f64 {
        if !self.rate_cutoff.is_finite() || self.rate_cutoff <= 0.0 {
            1.0
        } else {
            let omega = 4.0 * core::f64::consts::PI * self.rate_cutoff * dt;
            (omega / (1.0 + omega)).clamp(0.0, 1.0)
        }
    }
}

/// State of one honeycomb integration point.
///
/// `strains` holds total engineering strains per C1 direction,
/// which double as plastic totals because each direction keeps
/// only one strain history. Uses `Default`.
#[derive(Debug, Clone, Default)]
pub struct HoneycombState {
    /// Committed stresses per C1 direction (Pa).
    pub stresses: Voigt6,
    /// Total strains per C1 direction.
    pub strains: Voigt6,
    /// Filtered strain rates per C1 direction (1/s).
    pub rates: Voigt6,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic: f64,
    /// Element deletion flag from strain limits.
    pub failed: bool,
}

impl HoneycombState {
    /// Builds a virgin state with zero stress and strain.
    pub fn new() -> Self {
        Self::default()
    }
}

impl HoneycombLaw {
    /// Steps one increment with time step, bundling C1 calls.
    ///
    /// `dt` scales raw rates before filtering; pass any positive
    /// value when tables ignore rates. Failed points return zero
    /// stress and keep prior totals.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    /// use sonium_fem::physics::solid::plasticity::law_foams::{HoneycombLaw, HoneycombState};
    ///
    /// let cap = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
    /// let tables = [cap.clone(), cap.clone(), cap.clone(),
    ///     cap.clone(), cap.clone(), cap];
    /// let law = HoneycombLaw::law28([2.0; 6], tables).unwrap();
    /// let mut state = HoneycombState::new();
    /// let update = law.step(&mut state, &[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    /// assert!((update.stress[0] - 0.2).abs() < 1.0e-15);
    /// assert!(!update.yielding);
    /// ```
    pub fn step(&self, state: &mut HoneycombState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain, dt);
        update
    }

    /// Advances totals and filter state after `commit`.
    ///
    /// Updates strains and filtered rates; call once per accepted
    /// step. Drivers must use `step`, which bundles all three.
    pub fn advance_totals(&self, state: &mut HoneycombState, dstrain: &Voigt6, dt: f64) {
        let factor = self.rate_factor(dt);
        let target = self.raw_rates(dstrain, dt);
        for dir in 0..6 {
            state.strains[dir] += dstrain[dir];
            state.rates[dir] += factor * (target[dir] - state.rates[dir]);
        }
        state.failed = state.failed || self.failed(&state.strains);
    }

    /// Raw unfiltered rates from an increment and step time.
    ///
    /// Per-direction mode returns component magnitudes; common
    /// mode (`iratetime = 1`) repeats the equivalent rate
    /// `sqrt(sum normal^2 + sum shear^2 / 2)` six times.
    fn raw_rates(&self, dstrain: &Voigt6, dt: f64) -> Voigt6 {
        if dt <= 0.0 {
            return [0.0; 6];
        }
        match self.rate_mode {
            RateMode::PerDirection => core::array::from_fn(|dir| (dstrain[dir] / dt).abs()),
            RateMode::Common => {
                let raw: Voigt6 = core::array::from_fn(|dir| dstrain[dir] / dt);
                let common = (raw[0].powi(2)
                    + raw[1].powi(2)
                    + raw[2].powi(2)
                    + 0.5 * (raw[3].powi(2) + raw[4].powi(2) + raw[5].powi(2)))
                .sqrt();
                [common; 6]
            }
        }
    }

    /// Pure per-direction solve at given rates.
    ///
    /// Shared by `update` (stored filtered rates) and
    /// `update_with_rate` (fresh rates). Returns capped stress,
    /// the yield flag, and the plastic increment norm, which is
    /// zero unless some direction capped.
    fn solve(
        &self,
        state: &HoneycombState,
        dstrain: &Voigt6,
        rates: &Voigt6,
    ) -> (Voigt6, bool, f64) {
        let trial_strains: Voigt6 = core::array::from_fn(|dir| state.strains[dir] + dstrain[dir]);
        if self.failed(&trial_strains) {
            return ([0.0; 6], true, 0.0);
        }
        let mu = -(trial_strains[0] + trial_strains[1] + trial_strains[2]);
        let mut stress = [0.0; 6];
        let mut yielding = false;
        let mut plastic_norm = 0.0;
        for dir in 0..6 {
            let basis = if dir < 3 {
                self.basis_normal
            } else {
                self.basis_shear
            };
            let x = basis.abscissa(trial_strains[dir], mu);
            let trial = state.stresses[dir] + self.moduli[dir] * dstrain[dir];
            let (capped, was_capped) = self.cap_component(dir, trial, x, rates[dir]);
            yielding = yielding || was_capped;
            let plastic = dstrain[dir] - (capped - state.stresses[dir]) / self.moduli[dir];
            plastic_norm += plastic * plastic;
            stress[dir] = capped;
        }
        (stress, yielding, plastic_norm.sqrt())
    }

    /// Builds the C1 update from a pure solve at given rates.
    fn build_update(
        &self,
        state: &HoneycombState,
        dstrain: &Voigt6,
        rates: &Voigt6,
    ) -> StressUpdate {
        let (stress, yielding, dep) = self.solve(state, dstrain, rates);
        let tangent = if yielding {
            numeric_tangent_c1(|trial| self.solve(state, trial, rates).0, dstrain)
        } else {
            diag_tangent_c1(&self.moduli)
        };
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep.max(0.0),
            yielding,
        }
    }

    /// Caps one trial component at its yield value.
    ///
    /// Theory Manual p.305 Figure 193: stress advances elastically
    /// to the trial value, then its magnitude limits to the curve
    /// ordinate while keeping the trial sign. Positive curves give
    /// non-negative caps. Returns the value and a capped flag.
    fn cap_component(&self, dir: usize, trial: f64, x: f64, rate: f64) -> (f64, bool) {
        let cap = self.yield_stress(dir, x, rate);
        if trial.abs() <= cap {
            (trial, false)
        } else {
            (trial.signum() * cap, true)
        }
    }
}

impl StressUpdateModel for HoneycombLaw {
    type State = HoneycombState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        if state.failed {
            return zeroed_update();
        }
        self.build_update(state, dstrain, &state.rates)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        dstrain: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        if state.failed {
            return zeroed_update();
        }
        let rates = self.raw_rates(dstrain, dt);
        self.build_update(state, dstrain, &rates)
    }

    fn is_rate_dependent(&self) -> bool {
        self.rate_cutoff.is_finite()
            || self
                .tables
                .iter()
                .any(|table| matches!(table, YieldTable::Rated(_)))
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stresses = update.stress;
        state.equiv_plastic += update.delta_equiv_plastic_strain.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Isotropic visco foam (LAW33)
// ---------------------------------------------------------------------------

/// Isotropic foam with viscous and air-pressure response.
///
/// Follows Theory Manual p.365-366 equations 1024-1027: volumetric
/// strain `mu = rho / rho0 - 1` weights the relaxation moduli
/// `G(mu)` and `K(mu)` through frozen curves, viscous shear and
/// bulk moduli add rate-proportional stress, optional per-direction
/// air pressure `P_air` pushes components toward their stress
/// caps, and each principal trial stress caps independently at the
/// tension and compression limits of `funct_pcap` versus `mu`.
/// Reference Guide `/MAT/LAW33` sets the card defaults used here:
/// `GAM = 1.4`, `P0 = 101325 Pa`, and `Pmin = -1e30`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
/// use sonium_fem::physics::solid::plasticity::law_foams::ViscoFoam33;
///
/// let gm = Funct1D::new(vec![0.0, 1.0], vec![1.0, 1.0]).unwrap();
/// let pcap = Funct1D::new(vec![0.0, 1.0], vec![2.0, 2.0]).unwrap();
/// let mut law = ViscoFoam33::new(3.0, 0.1, pcap, gm, None).unwrap();
/// law.density_ratio = Some(1.0);
/// assert!((law.air_pressure(0.5) - law.air_pressure(0.0)) > 0.0);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive shear or bulk moduli.
#[derive(Debug, Clone)]
pub struct ViscoFoam33 {
    /// Elastic shear modulus (Pa).
    pub shear: f64,
    /// Elastic bulk modulus (Pa).
    pub bulk: f64,
    /// Volumetric yield-versus-`mu` curve (Pa).
    pub funct_pcap: Funct1D,
    /// Relaxation modulus multiplier versus `mu`.
    pub funct_gm: Funct1D,
    /// Optional rate table as `(function, strain rate)` pairs.
    pub funct_rate: Option<Vec<(Funct1D, f64)>>,
    /// Trapped-air adiabatic exponent, default 1.4.
    pub gamma: f64,
    /// Ambient pressure (Pa), default 101325.
    pub pressure0: f64,
    /// Minimum (most tensile) pressure cutoff (Pa).
    pub pressure_min: f64,
    /// Optional density ratio enabling air pressure.
    pub density_ratio: Option<f64>,
    /// Optional air-pressure exponent override.
    pub air_exponent: Option<f64>,
}

impl ViscoFoam33 {
    /// Builds the law with card defaults for air and cutoffs.
    ///
    /// # Errors
    ///
    /// Returns a message when either modulus is not positive.
    pub fn new(
        shear: f64,
        bulk: f64,
        funct_pcap: Funct1D,
        funct_gm: Funct1D,
        funct_rate: Option<Vec<(Funct1D, f64)>>,
    ) -> Result<Self, String> {
        if !shear.is_finite() || shear <= 0.0 {
            return Err(format!(
                "visco foam shear modulus must be positive, found {shear}"
            ));
        }
        if !bulk.is_finite() || bulk <= 0.0 {
            return Err(format!(
                "visco foam bulk modulus must be positive, found {bulk}"
            ));
        }
        Ok(Self {
            shear,
            bulk,
            funct_pcap,
            funct_gm,
            funct_rate,
            gamma: 1.4,
            pressure0: 101_325.0,
            pressure_min: -1.0e30,
            density_ratio: None,
            air_exponent: None,
        })
    }

    /// Trapped-air pressure at volumetric strain (Pa).
    ///
    /// Scales ambient pressure by the density ratio when the
    /// card supplies it; otherwise returns ambient pressure.
    /// Compression positive `mu` raises pressure through the
    /// adiabatic exponent.
    pub fn air_pressure(&self, mu: f64) -> f64 {
        let Some(ratio) = self.density_ratio else {
            return self.pressure0;
        };
        let exponent = self.air_exponent.unwrap_or(self.gamma);
        let rho_ratio = (1.0 + mu) / ratio.max(f64::MIN_POSITIVE);
        self.pressure0 * rho_ratio.max(f64::MIN_POSITIVE).powf(exponent)
    }

    /// Rate multiplier at abscissa and rate from the optional table.
    ///
    /// Scales the principal caps; matches the engine use of the
    /// previous-cycle filtered equivalent rate, stored in state.
    fn rate_factor_table(&self, x: f64, rate: f64) -> f64 {
        let Some(curves) = self.funct_rate.as_ref() else {
            return 1.0;
        };
        rated_eval(curves, x, rate).max(0.0)
    }
}

/// State of one visco-foam integration point.
///
/// Tracks totals needed by equations 1024-1027: the deviatoric
/// strain history enters implicitly through committed stresses,
/// while `mu` accumulates volumetric strain and `vol_plastic`
/// accumulates the plastic part. Uses `Default`.
#[derive(Debug, Clone, Default)]
pub struct ViscoFoamState {
    /// Committed stresses per C1 direction (Pa).
    pub stresses: Voigt6,
    /// Total strains per C1 direction.
    pub strains: Voigt6,
    /// Volumetric strain `mu`, compression positive.
    pub mu: f64,
    /// Accumulated plastic volumetric strain.
    pub vol_plastic: f64,
    /// Maximum principal compression reached.
    pub max_compression: f64,
    /// Previous-step equivalent strain rate (1/s).
    pub rate: f64,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic: f64,
}

impl ViscoFoamState {
    /// Builds a virgin state with zero stress and strain.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ViscoFoam33 {
    /// Steps one increment with time step, bundling C1 calls.
    pub fn step(&self, state: &mut ViscoFoamState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain, dt);
        update
    }

    /// Advances totals after `commit`; call once per accepted step.
    ///
    /// Stores the equivalent rate of this increment for the next
    /// `update`, matching the engine lag of one cycle.
    pub fn advance_totals(&self, state: &mut ViscoFoamState, dstrain: &Voigt6, dt: f64) {
        for (total, increment) in state.strains.iter_mut().zip(dstrain.iter()) {
            *total += *increment;
        }
        let dmu = -(dstrain[0] + dstrain[1] + dstrain[2]);
        state.mu += dmu;
        state.vol_plastic += dmu.max(0.0);
        if dt > 0.0 {
            state.rate = voigt_equivalent_increment_c1(dstrain) / dt;
        }
        let compression = -(state.strains[0].min(state.strains[1]).min(state.strains[2]));
        state.max_compression = state.max_compression.max(compression.max(0.0));
    }

    /// Pure trial-plus-cap solve at a given rate.
    ///
    /// Returns capped stress, the yield flag, and the plastic
    /// volumetric increment, zero unless a cap bound.
    fn solve(&self, state: &ViscoFoamState, dstrain: &Voigt6, rate: f64) -> (Voigt6, bool, f64) {
        let trial = self.trial_stress(state, dstrain);
        let mu = state.mu - (dstrain[0] + dstrain[1] + dstrain[2]);
        let (stress, yielding) = self.cap_principals(&trial, mu, rate);
        let dmu = -(dstrain[0] + dstrain[1] + dstrain[2]);
        let dep = if yielding { dmu.max(0.0) } else { 0.0 };
        (stress, yielding, dep)
    }

    /// Builds the C1 update from a pure solve at a given rate.
    fn build_update(&self, state: &ViscoFoamState, dstrain: &Voigt6, rate: f64) -> StressUpdate {
        let (stress, yielding, dep) = self.solve(state, dstrain, rate);
        let tangent = if yielding {
            numeric_tangent_c1(|trial| self.solve(state, trial, rate).0, dstrain)
        } else {
            elastic_tangent_c1(self.bulk, self.shear)
        };
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    /// Trial stress from relaxation-weighted elasticity.
    ///
    /// Equations 1025-1026: deviatoric trial scales shear modulus
    /// by `funct_gm(mu)`, volumetric trial scales bulk modulus by
    /// the same multiplier, and air pressure adds to the pressure.
    fn trial_stress(&self, state: &ViscoFoamState, dstrain: &Voigt6) -> Voigt6 {
        let mu = state.mu - (dstrain[0] + dstrain[1] + dstrain[2]);
        let relax = self.funct_gm.eval(mu).max(0.0);
        let dstrain_mean = (dstrain[0] + dstrain[1] + dstrain[2]) / 3.0;
        let air = self.air_pressure(mu) - self.pressure0;
        core::array::from_fn(|dir| {
            let elastic = if dir < 3 {
                2.0 * self.shear * (dstrain[dir] - dstrain_mean) + self.bulk * 3.0 * dstrain_mean
            } else {
                self.shear * dstrain[dir]
            };
            let trial = state.stresses[dir] + relax * elastic;
            if dir < 3 { trial - air } else { trial }
        })
    }

    /// Caps principal stresses at tension and compression limits.
    ///
    /// Equation 1027 with Reference Guide card defaults: the curve
    /// value at `mu = 0` (clamped to its first point when the grid
    /// starts above zero) limits maximum principal tension, the
    /// curve value at trial `mu` limits principal compression, the
    /// rate table scales both caps, and pressure below `Pmin`
    /// resets to `Pmin`. Returns capped stress and a yield flag.
    fn cap_principals(&self, trial: &Voigt6, mu: f64, rate: f64) -> (Voigt6, bool) {
        let (mut values, vectors) = jacobi_eigen_c1(trial, 12);
        let factor = self.rate_factor_table(mu, rate);
        let tension = factor * self.funct_pcap.eval(0.0).abs();
        let compression = factor * self.funct_pcap.eval(mu).abs();
        let mut yielding = false;
        for value in &mut values {
            if *value > tension {
                *value = tension;
                yielding = true;
            } else if *value < -compression {
                *value = -compression;
                yielding = true;
            }
        }
        let mut stress = voigt_from_principal(&values, &vectors);
        let pressure = pressure_c1(&stress);
        if pressure < self.pressure_min {
            let shift = self.pressure_min - pressure;
            stress[0] -= shift;
            stress[1] -= shift;
            stress[2] -= shift;
            yielding = true;
        }
        (stress, yielding)
    }
}

impl StressUpdateModel for ViscoFoam33 {
    type State = ViscoFoamState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        self.build_update(state, dstrain, state.rate)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        dstrain: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        let rate = if dt > 0.0 {
            voigt_equivalent_increment_c1(dstrain) / dt
        } else {
            0.0
        };
        self.build_update(state, dstrain, rate)
    }

    fn is_rate_dependent(&self) -> bool {
        self.funct_rate.is_some()
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stresses = update.stress;
        state.equiv_plastic += update.delta_equiv_plastic_strain.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Cosserat and crushable-foam solids (LAW68)
// ---------------------------------------------------------------------------

/// Cosserat couple-stress solid with foam symmetric response.
///
/// Theory Manual p.306-307 equations 849-853: the symmetric
/// stress follows the crushable-foam seed
/// ([`CrushableFoamMaterial`]) while the skew-symmetric couple
/// stress advances elastically from curvature with modulus
/// `Gc = E_mc / (2 (1 + nu_mc))` and caps radially at the foam
/// crush strength. The card supplies no separate couple yield,
/// so the couple cap reuses the compaction-dependent crush
/// strength; this choice is explicit, not from the manual. C1
/// Voigt increments carry no curvature, so `update` advances
/// only the symmetric part and `step_curve` adds curvature.
///
/// Couple and curvature vectors order skew components as
/// `[m_xy, m_yz, m_zx]`, matching C1 shear positions `[3, 4, 5]`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::{Cosserat68, CosseratState};
///
/// let law = Cosserat68::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6, 1.0e9, 0.1).unwrap();
/// let mut state = CosseratState::new();
/// let update = law.step_curve(&mut state, &[0.0; 6], &[1.0e-4, 0.0, 0.0], 1.0);
/// assert!(state.couple[0] > 0.0);
/// assert!(!update.yielding);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive moduli and Poisson
/// ratios outside `(-1, 0.5)`.
#[derive(Debug, Clone, Copy)]
pub struct Cosserat68 {
    /// Symmetric crushable-foam response.
    pub foam: CrushableFoamMaterial,
    /// Deviatoric couple stiffness (Pa).
    pub couple_modulus: f64,
}

impl Cosserat68 {
    /// Builds the law from foam and couple elastic parameters.
    ///
    /// # Errors
    ///
    /// Returns a message for non-positive moduli or Poisson
    /// ratios outside `(-1, 0.5)`.
    pub fn new(
        young: f64,
        poisson: f64,
        alpha: f64,
        initial_yield: f64,
        hardening_modulus: f64,
        couple_young: f64,
        couple_poisson: f64,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("young", young),
            ("initial yield", initial_yield),
            ("hardening modulus", hardening_modulus),
            ("couple young", couple_young),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("cosserat {name} must be positive, found {value}"));
            }
        }
        for (name, value) in [("poisson", poisson), ("couple poisson", couple_poisson)] {
            if !value.is_finite() || value <= -1.0 || value >= 0.5 {
                return Err(format!(
                    "cosserat {name} must lie in (-1, 0.5), found {value}"
                ));
            }
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(format!(
                "cosserat alpha must be non-negative, found {alpha}"
            ));
        }
        Ok(Self {
            foam: CrushableFoamMaterial::new(
                young,
                poisson,
                alpha,
                initial_yield,
                hardening_modulus,
            ),
            couple_modulus: couple_young / (2.0 * (1.0 + couple_poisson)),
        })
    }

    /// Steps one symmetric increment, bundling C1 calls.
    ///
    /// Leaves couple stress and curvature untouched; use
    /// `step_curve` when curvature increments exist.
    pub fn step(&self, state: &mut CosseratState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        update
    }

    /// Steps symmetric strain plus curvature increments.
    ///
    /// Advances the foam response, then the couple trial
    /// `m + 2 Gc dkappa` with radial return at the crush
    /// strength, accumulating couple plastic flow into the
    /// returned increment and the state totals.
    pub fn step_curve(
        &self,
        state: &mut CosseratState,
        dstrain: &Voigt6,
        dcurv: &[f64; 3],
        dt: f64,
    ) -> StressUpdate {
        let mut update = self.step(state, dstrain, dt);
        let trial =
            core::array::from_fn(|i| state.couple[i] + 2.0 * self.couple_modulus * dcurv[i]);
        let norm = (trial[0] * trial[0] + trial[1] * trial[1] + trial[2] * trial[2]).sqrt();
        let cap = self.foam.crush_strength(state.foam.compaction);
        let couple_dep = if norm > cap {
            let scale = cap / norm;
            state.couple = [trial[0] * scale, trial[1] * scale, trial[2] * scale];
            (norm - cap) / (2.0 * self.couple_modulus)
        } else {
            state.couple = trial;
            0.0
        };
        for (total, increment) in state.curvature.iter_mut().zip(dcurv.iter()) {
            *total += *increment;
        }
        state.equiv_plastic += couple_dep.max(0.0);
        update.delta_equiv_plastic_strain += couple_dep.max(0.0);
        update.yielding = update.yielding || couple_dep > 0.0;
        update
    }
}

/// State of one Cosserat integration point.
#[derive(Debug, Clone)]
pub struct CosseratState {
    /// Symmetric foam history (stress, compaction, yield flag).
    pub foam: CrushableFoamState,
    /// Skew couple stress `[m_xy, m_yz, m_zx]` (Pa).
    pub couple: [f64; 3],
    /// Total curvature matching the couple order (1/m).
    pub curvature: [f64; 3],
    /// Accumulated equivalent plastic strain with couple flow.
    pub equiv_plastic: f64,
}

impl CosseratState {
    /// Builds a virgin state with zero stress and curvature.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CosseratState {
    fn default() -> Self {
        Self {
            foam: CrushableFoamState::new(),
            couple: [0.0; 3],
            curvature: [0.0; 3],
            equiv_plastic: 0.0,
        }
    }
}

impl StressUpdateModel for Cosserat68 {
    type State = CosseratState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        self.foam.update(&state.foam, dstrain)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.foam.commit(&mut state.foam, update);
        state.equiv_plastic += update.delta_equiv_plastic_strain.max(0.0);
    }
}

/// Crushable-foam mode of LAW68.
///
/// Thin wrapper reusing the frozen [`CrushableFoamMaterial`]
/// seed and its state directly; the card selects this branch
/// for foam parts without Cosserat enrichment.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::CrushableFoamState;
/// use sonium_fem::physics::solid::plasticity::law_foams::CrushFoam68;
///
/// let law = CrushFoam68::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6);
/// let mut state = CrushableFoamState::new();
/// let update = law.step(&mut state, &[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
/// assert!(update.stress[0] < 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrushFoam68 {
    /// Reused crushable-foam response.
    pub foam: CrushableFoamMaterial,
}

impl CrushFoam68 {
    /// Builds the law from the seed material parameters.
    pub fn new(
        young: f64,
        poisson: f64,
        alpha: f64,
        initial_yield: f64,
        hardening_modulus: f64,
    ) -> Self {
        Self {
            foam: CrushableFoamMaterial::new(
                young,
                poisson,
                alpha,
                initial_yield,
                hardening_modulus,
            ),
        }
    }

    /// Steps one increment, bundling C1 calls.
    pub fn step(&self, state: &mut CrushableFoamState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        update
    }
}

impl StressUpdateModel for CrushFoam68 {
    type State = CrushableFoamState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        self.foam.update(state, dstrain)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.foam.commit(state, update);
    }
}

// ---------------------------------------------------------------------------
// Damaged concrete with viscous regularization (LAW24)
// ---------------------------------------------------------------------------

/// Mazars scalar damage parameters.
///
/// Equivalent strain `eq = sqrt(<e1>^2 + <e2>^2 + <e3>^2)`
/// from positive principal strains drives tension and
/// compression branches `D = 1 - (1-A) e0/eq - A exp(-B (eq -
/// e0))`, blended by squared principal-strain weights with
/// exponent 1.06. Theory Manual p.345-348 equation 966
/// introduces this damage framework for concrete.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::MazarsDamage;
///
/// let mazars = MazarsDamage { strain0: 1.0e-4, at: 0.8, bt: 1.5e4, ac: 1.0, bc: 2.0e3 };
/// assert!(mazars.damage(&[2.0e-4, 0.0, 0.0]) > 0.0);
/// assert!((mazars.damage(&[5.0e-5, 0.0, 0.0])).abs() < 1.0e-15);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MazarsDamage {
    /// Damage threshold equivalent strain.
    pub strain0: f64,
    /// Tension damage asymptote in [0, 1].
    pub at: f64,
    /// Tension damage rate (1/strain).
    pub bt: f64,
    /// Compression damage asymptote in [0, 1].
    pub ac: f64,
    /// Compression damage rate (1/strain).
    pub bc: f64,
}

impl MazarsDamage {
    /// Weight exponent blending tension and compression damage.
    const BLEND: f64 = 1.06;

    /// Scalar damage at principal total strains in [0, 1].
    pub fn damage(&self, principals: &[f64; 3]) -> f64 {
        let positive: f64 = principals.iter().map(|e| e.max(0.0).powi(2)).sum();
        let negative: f64 = principals.iter().map(|e| (-e).max(0.0).powi(2)).sum();
        let total = positive + negative;
        if total <= 0.0 {
            return 0.0;
        }
        let equiv = positive.sqrt();
        let tension = self.branch(self.at, self.bt, equiv);
        let compression = self.branch(self.ac, self.bc, equiv);
        let weight_t = (positive / total).powf(Self::BLEND);
        let weight_c = (negative / total).powf(Self::BLEND);
        (weight_t * tension + weight_c * compression).clamp(0.0, 1.0)
    }

    /// One damage branch at equivalent strain.
    fn branch(&self, a: f64, b: f64, equiv: f64) -> f64 {
        if equiv <= self.strain0 {
            0.0
        } else {
            let ratio = self.strain0 / equiv;
            (1.0 - (1.0 - a) * ratio - a * (-b * (equiv - self.strain0)).exp()).clamp(0.0, 1.0)
        }
    }
}

/// Concrete with Mazars damage, viscous delay, and a cap.
///
/// Isotropic elasticity degrades by `(1 - D)` with irreversible
/// Mazars damage; Duvaut-Lions regularization blends the
/// inviscid stress toward the previous stress over relaxation
/// time `tau`, and pressure caps at `cap0 + slope * mu`.
/// `update` returns the inviscid response while
/// `update_with_rate` applies the viscous blend.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::{Concrete24, ConcreteState, MazarsDamage};
///
/// let mazars = MazarsDamage { strain0: 1.0e-4, at: 0.8, bt: 1.5e4, ac: 1.0, bc: 2.0e3 };
/// let law = Concrete24::new(30.0e9, 0.2, mazars, 0.0, 30.0e6, 1.0e9).unwrap();
/// let mut state = ConcreteState::new();
/// let update = law.step(&mut state, &[1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
/// assert!(update.stress[0] > 0.0);
/// assert!(update.yielding);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive moduli, threshold, or
/// cap, Poisson ratios outside `(-1, 0.5)`, asymptotes outside
/// `[0, 1]`, non-positive damage rates, and negative viscous
/// time or cap slope.
#[derive(Debug, Clone, Copy)]
pub struct Concrete24 {
    /// Young modulus (Pa).
    pub young: f64,
    /// Poisson ratio.
    pub poisson: f64,
    /// Mazars damage parameters.
    pub damage: MazarsDamage,
    /// Duvaut-Lions relaxation time (s); 0 disables viscosity.
    pub tau: f64,
    /// Cap pressure at zero volumetric strain (Pa).
    pub cap0: f64,
    /// Cap hardening slope (Pa per unit `mu`).
    pub cap_slope: f64,
}

impl Concrete24 {
    /// Builds the law with validated parameters.
    ///
    /// # Errors
    ///
    /// Returns a message for any out-of-range parameter.
    pub fn new(
        young: f64,
        poisson: f64,
        damage: MazarsDamage,
        tau: f64,
        cap0: f64,
        cap_slope: f64,
    ) -> Result<Self, String> {
        if !young.is_finite() || young <= 0.0 {
            return Err(format!(
                "concrete young modulus must be positive, found {young}"
            ));
        }
        if !poisson.is_finite() || poisson <= -1.0 || poisson >= 0.5 {
            return Err(format!(
                "concrete poisson must lie in (-1, 0.5), found {poisson}"
            ));
        }
        if !damage.strain0.is_finite() || damage.strain0 <= 0.0 {
            return Err(format!(
                "concrete damage threshold must be positive, found {}",
                damage.strain0
            ));
        }
        for (name, value) in [("at", damage.at), ("ac", damage.ac)] {
            if !value.is_finite() || value < 0.0 || value > 1.0 {
                return Err(format!("concrete {name} must lie in [0, 1], found {value}"));
            }
        }
        for (name, value) in [("bt", damage.bt), ("bc", damage.bc)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("concrete {name} must be positive, found {value}"));
            }
        }
        if !tau.is_finite() || tau < 0.0 {
            return Err(format!("concrete tau must be non-negative, found {tau}"));
        }
        if !cap0.is_finite() || cap0 <= 0.0 {
            return Err(format!("concrete cap0 must be positive, found {cap0}"));
        }
        if !cap_slope.is_finite() || cap_slope < 0.0 {
            return Err(format!(
                "concrete cap slope must be non-negative, found {cap_slope}"
            ));
        }
        Ok(Self {
            young,
            poisson,
            damage,
            tau,
            cap0,
            cap_slope,
        })
    }

    /// Bulk modulus from Young modulus and Poisson ratio (Pa).
    pub fn bulk(&self) -> f64 {
        self.young / (3.0 * (1.0 - 2.0 * self.poisson))
    }

    /// Shear modulus from Young modulus and Poisson ratio (Pa).
    pub fn shear(&self) -> f64 {
        self.young / (2.0 * (1.0 + self.poisson))
    }

    /// Cap pressure at volumetric strain (Pa).
    pub fn cap_pressure(&self, mu: f64) -> f64 {
        self.cap0 + self.cap_slope * mu.max(0.0)
    }

    /// Steps one increment with time step, bundling C1 calls.
    pub fn step(&self, state: &mut ConcreteState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain);
        update
    }

    /// Advances totals after `commit`; call once per accepted step.
    pub fn advance_totals(&self, state: &mut ConcreteState, dstrain: &Voigt6) {
        for (total, increment) in state.strains.iter_mut().zip(dstrain.iter()) {
            *total += *increment;
        }
        let principals = principal_stresses_c1(&state.strains);
        state.damage = state.damage.max(self.damage.damage(&principals));
    }

    /// Isotropic elastic trial from committed stress.
    fn elastic_trial(&self, state: &ConcreteState, dstrain: &Voigt6) -> Voigt6 {
        let bulk = self.bulk();
        let shear = self.shear();
        let mean = (dstrain[0] + dstrain[1] + dstrain[2]) / 3.0;
        core::array::from_fn(|dir| {
            let elastic = if dir < 3 {
                2.0 * shear * (dstrain[dir] - mean) + 3.0 * bulk * mean
            } else {
                shear * dstrain[dir]
            };
            state.stresses[dir] + elastic
        })
    }

    /// Pure damage-plus-cap solve with optional viscous blend.
    ///
    /// `dt = None` returns the inviscid damaged response;
    /// `Some(dt)` blends toward the previous stress with weight
    /// `tau / dt`, the Duvaut-Lions update. Returns stress, the
    /// yield flag, the plastic increment, and new damage.
    fn solve(
        &self,
        state: &ConcreteState,
        dstrain: &Voigt6,
        dt: Option<f64>,
    ) -> (Voigt6, bool, f64, f64) {
        let totals: Voigt6 = core::array::from_fn(|dir| state.strains[dir] + dstrain[dir]);
        let principals = principal_stresses_c1(&totals);
        let damage_new = state.damage.max(self.damage.damage(&principals));
        let trial = self.elastic_trial(state, dstrain);
        let mut stress = core::array::from_fn(|dir| (1.0 - damage_new) * trial[dir]);
        if let Some(weight) = dt
            .filter(|step| self.tau > 0.0 && *step > 0.0)
            .map(|step| self.tau / step)
        {
            for (comp, old) in stress.iter_mut().zip(state.stresses.iter()) {
                *comp = (*comp + weight * *old) / (1.0 + weight);
            }
        }
        let mu = -(totals[0] + totals[1] + totals[2]);
        let pressure = pressure_c1(&stress);
        let cap = self.cap_pressure(mu);
        let mut yielding = damage_new > state.damage;
        let mut dep = (damage_new - state.damage).max(0.0);
        if pressure > cap {
            let excess = pressure - cap;
            stress[0] += excess;
            stress[1] += excess;
            stress[2] += excess;
            yielding = true;
            dep += (excess / self.bulk()).max(0.0);
        }
        (stress, yielding, dep, damage_new)
    }

    /// Builds the C1 update from a pure solve.
    fn build_update(
        &self,
        state: &ConcreteState,
        dstrain: &Voigt6,
        dt: Option<f64>,
    ) -> StressUpdate {
        let (stress, yielding, dep, _) = self.solve(state, dstrain, dt);
        let tangent = if yielding {
            numeric_tangent_c1(|trial| self.solve(state, trial, dt).0, dstrain)
        } else {
            elastic_tangent_c1(self.bulk(), self.shear())
        };
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep.max(0.0),
            yielding,
        }
    }
}

/// State of one concrete integration point.
///
/// Uses `Default`.
#[derive(Debug, Clone, Default)]
pub struct ConcreteState {
    /// Committed stresses per C1 direction (Pa).
    pub stresses: Voigt6,
    /// Total strains per C1 direction.
    pub strains: Voigt6,
    /// Irreversible Mazars damage in [0, 1].
    pub damage: f64,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic: f64,
}

impl ConcreteState {
    /// Builds a virgin state with zero stress and strain.
    pub fn new() -> Self {
        Self::default()
    }
}

impl StressUpdateModel for Concrete24 {
    type State = ConcreteState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        self.build_update(state, dstrain, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        dstrain: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.build_update(state, dstrain, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        self.tau > 0.0
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stresses = update.stress;
        state.equiv_plastic += update.delta_equiv_plastic_strain.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Woven fabric membranes (LAW19, LAW58)
// ---------------------------------------------------------------------------

/// Nonlinear warp or weft direction for LAW58.
///
/// Loading follows the frozen curve versus total strain past the
/// historic maximum; unloading and reloading return along the
/// unloading modulus from the historic maximum point. Theory
/// Manual p.304 equations 843-846 describe this fabric model.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
/// use sonium_fem::physics::solid::plasticity::law_foams::FabricDirection;
///
/// let loading = Funct1D::new(vec![0.0, 0.1], vec![0.0, 1.0e6]).unwrap();
/// let dir = FabricDirection::new(loading, 20.0e6).unwrap();
/// assert!((dir.unloading - 20.0e6).abs() < 1.0e-6);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive unloading moduli.
#[derive(Debug, Clone)]
pub struct FabricDirection {
    /// Loading curve stress versus total strain (Pa).
    pub loading: Funct1D,
    /// Unloading and reloading modulus (Pa).
    pub unloading: f64,
}

impl FabricDirection {
    /// Builds the direction with a validated modulus.
    ///
    /// # Errors
    ///
    /// Returns a message when the unloading modulus is not positive.
    pub fn new(loading: Funct1D, unloading: f64) -> Result<Self, String> {
        if !unloading.is_finite() || unloading <= 0.0 {
            return Err(format!(
                "fabric unloading modulus must be positive, found {unloading}"
            ));
        }
        Ok(Self { loading, unloading })
    }
}

/// Woven fabric with tension-compression split and failure.
///
/// `law19` builds the linear form with separate tension and
/// compression moduli per in-plane direction; `law58` adds
/// nonlinear warp/weft directions. Shear stays linear, and the
/// law embeds as plane stress: out-of-plane components return
/// zero stress with zero tangent rows. Trial totals breaching
/// any failure limit zero the response that step; `yielding`
/// reports failure only, since fabric has no yield surface.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::{FabricLaw, FabricState};
///
/// let law = FabricLaw::law19([10.0e6, 10.0e6], [1.0e6, 1.0e6], 2.0e6,
///     [0.1, 0.1], [0.2, 0.2], 0.3).unwrap();
/// let mut state = FabricState::new();
/// let update = law.step(&mut state, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
/// assert!((update.stress[0] - 100.0e3).abs() < 1.0e-6);
/// ```
///
/// # Errors
///
/// Constructors reject non-positive moduli and failure limits.
#[derive(Debug, Clone)]
pub struct FabricLaw {
    /// Tension moduli `[E1t, E2t]` (Pa).
    pub tension: [f64; 2],
    /// Compression moduli `[E1c, E2c]` (Pa).
    pub compression: [f64; 2],
    /// In-plane shear modulus (Pa).
    pub shear: f64,
    /// Tension failure strains per direction.
    pub fail_tension: [f64; 2],
    /// Compression failure magnitudes per direction.
    pub fail_compression: [f64; 2],
    /// Shear failure magnitude.
    pub fail_shear: f64,
    /// Nonlinear warp/weft directions (`None` selects LAW19).
    pub nonlinear: [Option<FabricDirection>; 2],
}

impl FabricLaw {
    /// Builds the linear LAW19 form.
    ///
    /// # Errors
    ///
    /// Returns a message for non-positive moduli or limits.
    pub fn law19(
        tension: [f64; 2],
        compression: [f64; 2],
        shear: f64,
        fail_tension: [f64; 2],
        fail_compression: [f64; 2],
        fail_shear: f64,
    ) -> Result<Self, String> {
        Self::checked(
            tension,
            compression,
            shear,
            fail_tension,
            fail_compression,
            fail_shear,
            [None, None],
        )
    }

    /// Builds the nonlinear LAW58 form with warp/weft curves.
    ///
    /// # Errors
    ///
    /// Returns a message for non-positive moduli or limits.
    #[allow(clippy::too_many_arguments)]
    pub fn law58(
        tension: [f64; 2],
        compression: [f64; 2],
        shear: f64,
        fail_tension: [f64; 2],
        fail_compression: [f64; 2],
        fail_shear: f64,
        warp: FabricDirection,
        weft: FabricDirection,
    ) -> Result<Self, String> {
        Self::checked(
            tension,
            compression,
            shear,
            fail_tension,
            fail_compression,
            fail_shear,
            [Some(warp), Some(weft)],
        )
    }

    /// Validates moduli and limits, then installs directions.
    fn checked(
        tension: [f64; 2],
        compression: [f64; 2],
        shear: f64,
        fail_tension: [f64; 2],
        fail_compression: [f64; 2],
        fail_shear: f64,
        nonlinear: [Option<FabricDirection>; 2],
    ) -> Result<Self, String> {
        for (name, value) in tension
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("tension[{i}]"), *v))
            .chain(
                compression
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (format!("compression[{i}]"), *v)),
            )
            .chain(core::iter::once(("shear".to_string(), shear)))
        {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "fabric {name} modulus must be positive, found {value}"
                ));
            }
        }
        for (name, value) in fail_tension
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("fail_tension[{i}]"), *v))
            .chain(
                fail_compression
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (format!("fail_compression[{i}]"), *v)),
            )
            .chain(core::iter::once(("fail_shear".to_string(), fail_shear)))
        {
            if value.is_nan() || value <= 0.0 {
                return Err(format!(
                    "fabric {name} limit must be positive, found {value}"
                ));
            }
        }
        Ok(Self {
            tension,
            compression,
            shear,
            fail_tension,
            fail_compression,
            fail_shear,
            nonlinear,
        })
    }

    /// Failure flag for trial totals.
    fn breached(&self, totals: &Voigt6) -> bool {
        totals[0] > self.fail_tension[0]
            || totals[1] > self.fail_tension[1]
            || -totals[0] > self.fail_compression[0]
            || -totals[1] > self.fail_compression[1]
            || totals[3].abs() > self.fail_shear
    }

    /// In-plane stress and tangent modulus of one direction.
    ///
    /// Linear directions pick the tension or compression modulus
    /// by trial-total sign; nonlinear directions evaluate the
    /// loading curve past the historic maximum and return along
    /// the unloading modulus below it.
    fn direction_response(
        &self,
        dir: usize,
        state: &FabricState,
        totals: &Voigt6,
        dstrain: &Voigt6,
    ) -> (f64, f64) {
        let trial = totals[dir];
        if let Some(nonlinear) = &self.nonlinear[dir] {
            if trial > state.max_strain[dir] {
                (
                    nonlinear.loading.eval(trial),
                    nonlinear.loading.slope(trial),
                )
            } else {
                let stress = state.stress_at_max[dir]
                    - nonlinear.unloading * (state.max_strain[dir] - trial);
                (stress, nonlinear.unloading)
            }
        } else {
            let modulus = if trial >= 0.0 {
                self.tension[dir]
            } else {
                self.compression[dir]
            };
            (state.stresses[dir] + modulus * dstrain[dir], modulus)
        }
    }

    /// Steps one increment with time step, bundling C1 calls.
    pub fn step(&self, state: &mut FabricState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain);
        update
    }

    /// Advances totals after `commit`; call once per accepted step.
    pub fn advance_totals(&self, state: &mut FabricState, dstrain: &Voigt6) {
        for (total, increment) in state.strains.iter_mut().zip(dstrain.iter()) {
            *total += *increment;
        }
        for dir in 0..2 {
            if self.nonlinear[dir].is_some() && state.strains[dir] > state.max_strain[dir] {
                state.max_strain[dir] = state.strains[dir];
                state.stress_at_max[dir] = state.stresses[dir];
            }
        }
        state.failed = state.failed || self.breached(&state.strains);
    }
}

/// State of one fabric integration point.
///
/// Uses `Default`.
#[derive(Debug, Clone, Default)]
pub struct FabricState {
    /// Committed stresses per C1 direction (Pa).
    pub stresses: Voigt6,
    /// Total strains per C1 direction.
    pub strains: Voigt6,
    /// Historic maximum total strain per in-plane direction.
    pub max_strain: [f64; 2],
    /// Committed stress at each historic maximum (Pa).
    pub stress_at_max: [f64; 2],
    /// Accumulated equivalent plastic strain (always zero).
    pub equiv_plastic: f64,
    /// Element deletion flag from strain limits.
    pub failed: bool,
}

impl FabricState {
    /// Builds a virgin state with zero stress and strain.
    pub fn new() -> Self {
        Self::default()
    }
}

impl StressUpdateModel for FabricLaw {
    type State = FabricState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        if state.failed {
            return zeroed_update();
        }
        let totals: Voigt6 = core::array::from_fn(|dir| state.strains[dir] + dstrain[dir]);
        if self.breached(&totals) {
            return zeroed_update();
        }
        let mut stress = [0.0; 6];
        let mut moduli = [0.0; 6];
        for dir in 0..2 {
            let (value, modulus) = self.direction_response(dir, state, &totals, dstrain);
            stress[dir] = value;
            moduli[dir] = modulus;
        }
        stress[3] = state.stresses[3] + self.shear * dstrain[3];
        moduli[3] = self.shear;
        StressUpdate {
            stress,
            tangent: diag_tangent_c1(&moduli),
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stresses = update.stress;
    }
}

// ---------------------------------------------------------------------------
// Semi-analytical plasticity with triaxiality (LAW76)
// ---------------------------------------------------------------------------

/// Equivalent-plastic-strain failure limits per stress mode.
#[derive(Debug, Clone, Copy)]
pub struct SampFailure {
    /// Tension-dominated limit (`INFINITY` disables).
    pub tension: f64,
    /// Compression-dominated limit.
    pub compression: f64,
    /// Shear-dominated limit.
    pub shear: f64,
}

impl SampFailure {
    /// Disables all failure limits.
    pub fn none() -> Self {
        Self {
            tension: f64::INFINITY,
            compression: f64::INFINITY,
            shear: f64::INFINITY,
        }
    }
}

/// Semi-analytical model for polymers (SAMP).
///
/// Reference Guide `/MAT/LAW76` equations 296-300: isotropic
/// elasticity with pressure `P = -(xx + yy + zz) / 3` per
/// equation 297, yield interpolated piecewise-linearly across
/// triaxiality stations `T = P / q` with biaxial tension at
/// `-2/3`, uniaxial tension at `-1/3`, shear at `0`, and
/// uniaxial compression at `1/3`, radial return at fixed
/// pressure, optional rate scaling from a frozen table over
/// `(strain rate, triaxiality)`, and mode-dependent failure on
/// equivalent plastic strain.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
/// use sonium_fem::physics::solid::plasticity::law_foams::{Samp76, SampState};
///
/// let flat = Funct1D::new(vec![0.0, 1.0], vec![10.0, 10.0]).unwrap();
/// let shear = Funct1D::new(vec![0.0, 1.0], vec![10.0 / 1.7320508075688772, 10.0]).unwrap();
/// let law = Samp76::new(1000.0, 0.3, flat.clone(), flat, shear).unwrap();
/// let mut state = SampState::new();
/// let update = law.step(&mut state, &[0.02, -0.006, -0.006, 0.0, 0.0, 0.0], 1.0);
/// assert!((update.stress[0] - 40.0 / 3.0).abs() < 1.0e-9);
/// assert!(update.yielding);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive moduli.
#[derive(Debug, Clone)]
pub struct Samp76 {
    /// Young modulus (Pa).
    pub young: f64,
    /// Poisson ratio.
    pub poisson: f64,
    /// Yield versus plastic strain in tension (Pa).
    pub tension: Funct1D,
    /// Yield versus plastic strain in compression (Pa).
    pub compression: Funct1D,
    /// Yield versus plastic strain in shear (Pa).
    pub shear: Funct1D,
    /// Optional yield curve for biaxial tension (Pa).
    pub biaxial: Option<Funct1D>,
    /// Optional rate scale over `(rate, triaxiality)`.
    pub rate_table: Option<Table2D>,
    /// Failure limits per mode.
    pub failure: SampFailure,
}

impl Samp76 {
    /// Builds the law with no biaxial curve, rate table, or failure.
    ///
    /// # Errors
    ///
    /// Returns a message for non-positive moduli or Poisson
    /// ratios outside `(-1, 0.5)`.
    pub fn new(
        young: f64,
        poisson: f64,
        tension: Funct1D,
        compression: Funct1D,
        shear: Funct1D,
    ) -> Result<Self, String> {
        if !young.is_finite() || young <= 0.0 {
            return Err(format!(
                "samp young modulus must be positive, found {young}"
            ));
        }
        if !poisson.is_finite() || poisson <= -1.0 || poisson >= 0.5 {
            return Err(format!(
                "samp poisson must lie in (-1, 0.5), found {poisson}"
            ));
        }
        Ok(Self {
            young,
            poisson,
            tension,
            compression,
            shear,
            biaxial: None,
            rate_table: None,
            failure: SampFailure::none(),
        })
    }

    /// Bulk modulus from Young modulus and Poisson ratio (Pa).
    pub fn bulk(&self) -> f64 {
        self.young / (3.0 * (1.0 - 2.0 * self.poisson))
    }

    /// Shear modulus from Young modulus and Poisson ratio (Pa).
    pub fn shear_modulus(&self) -> f64 {
        self.young / (2.0 * (1.0 + self.poisson))
    }

    /// Yield stress at plastic strain and triaxiality (Pa).
    ///
    /// Interpolates piecewise-linearly across stations
    /// `[-2/3, -1/3, 0, 1/3, 2/3]` and clamps outside; the
    /// biaxial curve feeds the `-2/3` station when present.
    pub fn yield_at(&self, ep: f64, triax: f64) -> f64 {
        let tension = self.tension.eval(ep);
        let compression = self.compression.eval(ep);
        let shear = self.shear.eval(ep);
        let biaxial = self
            .biaxial
            .as_ref()
            .map(|curve| curve.eval(ep))
            .unwrap_or(tension);
        let stations = [-2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0];
        let values = [biaxial, tension, shear, compression, compression];
        if triax <= stations[0] {
            return values[0].max(0.0);
        }
        for i in 1..stations.len() {
            if triax <= stations[i] {
                let frac = (triax - stations[i - 1]) / (stations[i] - stations[i - 1]);
                return (values[i - 1] + frac * (values[i] - values[i - 1])).max(0.0);
            }
        }
        values[values.len() - 1].max(0.0)
    }

    /// Rate scale at strain rate and triaxiality.
    fn rate_scale(&self, rate: f64, triax: f64) -> f64 {
        self.rate_table
            .as_ref()
            .map(|table| table.eval(rate, triax).max(0.0))
            .unwrap_or(1.0)
    }

    /// Failure limit for the triaxiality mode.
    fn ep_limit(&self, triax: f64) -> f64 {
        if triax < -1.0 / 6.0 {
            self.failure.tension
        } else if triax > 1.0 / 6.0 {
            self.failure.compression
        } else {
            self.failure.shear
        }
    }

    /// Isotropic elastic trial from committed stress.
    fn elastic_trial(&self, state: &SampState, dstrain: &Voigt6) -> Voigt6 {
        let bulk = self.bulk();
        let shear = self.shear_modulus();
        let mean = (dstrain[0] + dstrain[1] + dstrain[2]) / 3.0;
        core::array::from_fn(|dir| {
            let elastic = if dir < 3 {
                2.0 * shear * (dstrain[dir] - mean) + 3.0 * bulk * mean
            } else {
                shear * dstrain[dir]
            };
            state.stresses[dir] + elastic
        })
    }

    /// Pure radial-return solve at a given rate.
    ///
    /// Returns stress, the yield flag, and the plastic
    /// multiplier. Fixed-point iteration (eight sweeps) tracks
    /// the triaxiality drift as `q` falls at fixed pressure.
    fn solve(&self, state: &SampState, dstrain: &Voigt6, rate: f64) -> (Voigt6, bool, f64) {
        let trial = self.elastic_trial(state, dstrain);
        let pressure = pressure_c1(&trial);
        let q_trial = von_mises_c1(&trial);
        if q_trial <= 0.0 {
            return (trial, false, 0.0);
        }
        let triax_trial = pressure / q_trial;
        let scale = self.rate_scale(rate, triax_trial);
        let shear = self.shear_modulus();
        let yield_trial = self.yield_at(state.equiv_plastic, triax_trial) * scale;
        if q_trial <= yield_trial {
            return (trial, false, 0.0);
        }
        let mut dep = ((q_trial - yield_trial) / (3.0 * shear)).max(0.0);
        for _ in 0..8 {
            let q_new = (q_trial - 3.0 * shear * dep).max(0.0);
            let triax = if q_new > 0.0 {
                pressure / q_new
            } else {
                triax_trial
            };
            let yield_now = self.yield_at(state.equiv_plastic + dep, triax) * scale;
            dep = ((q_trial - yield_now) / (3.0 * shear)).max(0.0);
        }
        let factor = ((q_trial - 3.0 * shear * dep) / q_trial).clamp(0.0, 1.0);
        let deviator = deviator_c1(&trial);
        let mean = (trial[0] + trial[1] + trial[2]) / 3.0;
        let stress = core::array::from_fn(|dir| {
            if dir < 3 {
                mean + factor * deviator[dir]
            } else {
                factor * deviator[dir]
            }
        });
        (stress, true, dep)
    }

    /// Builds the C1 update from a pure solve at a given rate.
    fn build_update(&self, state: &SampState, dstrain: &Voigt6, rate: f64) -> StressUpdate {
        let (stress, yielding, dep) = self.solve(state, dstrain, rate);
        if yielding {
            let q = von_mises_c1(&stress);
            let triax = if q > 0.0 {
                pressure_c1(&stress) / q
            } else {
                0.0
            };
            if state.equiv_plastic + dep > self.ep_limit(triax) {
                // Zeroed stress keeps the increment so commit latches failure.
                return StressUpdate {
                    stress: [0.0; 6],
                    tangent: [0.0; 36],
                    delta_equiv_plastic_strain: dep.max(0.0),
                    yielding: true,
                };
            }
        }
        let tangent = if yielding {
            numeric_tangent_c1(|trial| self.solve(state, trial, rate).0, dstrain)
        } else {
            elastic_tangent_c1(self.bulk(), self.shear_modulus())
        };
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep.max(0.0),
            yielding,
        }
    }

    /// Confirms a zeroed step breached its mode limit.
    ///
    /// Re-solves from pre-commit state so degenerate zero
    /// returns (zero yield with pure shear trial) never latch
    /// failure; exact where `advance_totals` must guess the mode
    /// from committed stress.
    fn breach_confirmed(&self, state: &SampState, dstrain: &Voigt6, dt: f64) -> bool {
        let rate = if dt > 0.0 {
            voigt_equivalent_increment_c1(dstrain) / dt
        } else {
            0.0
        };
        let (solved, yielding, dep) = self.solve(state, dstrain, rate);
        if !yielding {
            return false;
        }
        let q = von_mises_c1(&solved);
        let triax = if q > 0.0 {
            pressure_c1(&solved) / q
        } else {
            0.0
        };
        state.equiv_plastic + dep > self.ep_limit(triax)
    }

    /// Steps one increment with time step, bundling C1 calls.
    pub fn step(&self, state: &mut SampState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        let breached = !state.failed
            && update.stress == [0.0; 6]
            && update.delta_equiv_plastic_strain > 0.0
            && self.breach_confirmed(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain, dt);
        state.failed = state.failed || breached;
        update
    }

    /// Advances totals after `commit`; call once per accepted step.
    pub fn advance_totals(&self, state: &mut SampState, dstrain: &Voigt6, dt: f64) {
        if dt > 0.0 {
            state.rate = voigt_equivalent_increment_c1(dstrain) / dt;
        }
        let q = von_mises_c1(&state.stresses);
        let triax = if q > 0.0 {
            pressure_c1(&state.stresses) / q
        } else {
            0.0
        };
        state.failed = state.failed || state.equiv_plastic > self.ep_limit(triax);
    }
}

/// State of one SAMP integration point.
///
/// Uses `Default`.
#[derive(Debug, Clone, Default)]
pub struct SampState {
    /// Committed stresses per C1 direction (Pa).
    pub stresses: Voigt6,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic: f64,
    /// Previous-step equivalent strain rate (1/s).
    pub rate: f64,
    /// Element deletion flag from strain limits.
    pub failed: bool,
}

impl SampState {
    /// Builds a virgin state with zero stress and strain.
    pub fn new() -> Self {
        Self::default()
    }
}

impl StressUpdateModel for Samp76 {
    type State = SampState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        if state.failed {
            return zeroed_update();
        }
        self.build_update(state, dstrain, state.rate)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        dstrain: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        if state.failed {
            return zeroed_update();
        }
        let rate = if dt > 0.0 {
            voigt_equivalent_increment_c1(dstrain) / dt
        } else {
            0.0
        };
        self.build_update(state, dstrain, rate)
    }

    fn is_rate_dependent(&self) -> bool {
        self.rate_table.is_some()
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stresses = update.stress;
        state.equiv_plastic += update.delta_equiv_plastic_strain.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Multi-material mixture with JWL equations of state (LAW51)
// ---------------------------------------------------------------------------

/// Jones-Wilkins-Lee equation of state.
///
/// Reference Guide p.1054-1055 equations 651-654 with
/// compression ratio `eta = 1 + mu`:
/// `P = A (1 - w eta / R1) e^{-R1 / eta} + B (1 - w eta / R2)
/// e^{-R2 / eta} + w rho0 eta e`, where `e` is specific
/// internal energy. Theory Manual section 9 (`MULTIMAT`)
/// equations 717-721 mix sub-material pressures by volume.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::JwlEos;
///
/// let eos = JwlEos::new(1.0, 0.0, 4.0, 1.0, 0.3, 1000.0, 0.0).unwrap();
/// let pressure = eos.pressure(0.0, 0.0);
/// assert!((pressure - 0.925 * (-4.0f64).exp()).abs() < 1.0e-15);
/// assert!((eos.dp_de(0.0) - 300.0).abs() < 1.0e-9);
/// ```
///
/// # Errors
///
/// The constructor rejects non-positive `R1`, `R2`, `rho0`,
/// negative `A`, `B`, `omega`, and non-finite `e0`.
#[derive(Debug, Clone, Copy)]
pub struct JwlEos {
    /// Linear coefficient (Pa).
    pub a: f64,
    /// Second linear coefficient (Pa).
    pub b: f64,
    /// First exponent denominator.
    pub r1: f64,
    /// Second exponent denominator.
    pub r2: f64,
    /// Gruneisen coefficient.
    pub omega: f64,
    /// Reference density (kg/m3).
    pub rho0: f64,
    /// Initial specific internal energy (J/kg).
    pub e0: f64,
}

impl JwlEos {
    /// Builds the EOS with validated parameters.
    ///
    /// # Errors
    ///
    /// Returns a message for any out-of-range parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: f64,
        b: f64,
        r1: f64,
        r2: f64,
        omega: f64,
        rho0: f64,
        e0: f64,
    ) -> Result<Self, String> {
        for (name, value) in [("r1", r1), ("r2", r2), ("rho0", rho0)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("jwl {name} must be positive, found {value}"));
            }
        }
        for (name, value) in [("a", a), ("b", b), ("omega", omega)] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("jwl {name} must be non-negative, found {value}"));
            }
        }
        if !e0.is_finite() {
            return Err(format!("jwl e0 must be finite, found {e0}"));
        }
        Ok(Self {
            a,
            b,
            r1,
            r2,
            omega,
            rho0,
            e0,
        })
    }

    /// Pressure at volumetric strain and specific energy (Pa).
    pub fn pressure(&self, mu: f64, energy: f64) -> f64 {
        let eta = (1.0 + mu).max(1.0e-12);
        let term_a = self.a * (1.0 - self.omega * eta / self.r1) * (-self.r1 / eta).exp();
        let term_b = self.b * (1.0 - self.omega * eta / self.r2) * (-self.r2 / eta).exp();
        term_a + term_b + self.omega * self.rho0 * eta * energy
    }

    /// Pressure derivative over volumetric strain (Pa).
    pub fn dp_dmu(&self, mu: f64, energy: f64) -> f64 {
        let eta = (1.0 + mu).max(1.0e-12);
        let da = self.a
            * (-self.r1 / eta).exp()
            * (self.r1 / (eta * eta) - self.omega / eta - self.omega / self.r1);
        let db = self.b
            * (-self.r2 / eta).exp()
            * (self.r2 / (eta * eta) - self.omega / eta - self.omega / self.r2);
        da + db + self.omega * self.rho0 * energy
    }

    /// Pressure derivative over specific energy (Pa per J/kg).
    pub fn dp_de(&self, mu: f64) -> f64 {
        let eta = (1.0 + mu).max(1.0e-12);
        self.omega * self.rho0 * eta
    }
}

/// Two-material programmed-burn mixture.
///
/// Mixes unreacted and reacted JWL pressures by burn fraction
/// `F`: `P = (1 - F) Pu + F Pr`, returning hydrostatic stress
/// `-(P - Pext) I`. Burn ramps linearly from `detonation_time`
/// over `burn_time` (non-positive burns instantly). Energies
/// advance adiabatically, `de = P dmu / (rho0 eta^2)` per
/// material. Rate independent with zero plastic flow.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_foams::{JwlEos, Multimat51, MultimatState};
///
/// let eos = JwlEos::new(1.0e9, 0.0, 4.0, 1.0, 0.3, 1000.0, 0.0).unwrap();
/// let law = Multimat51::new(eos, eos, 0.0, 0.0, 0.0).unwrap();
/// let mut state = MultimatState::new(eos.e0, eos.e0);
/// let update = law.step(&mut state, &[-0.1, -0.1, -0.1, 0.0, 0.0, 0.0], 1.0);
/// assert!(update.stress[0] < 0.0);
/// ```
///
/// # Errors
///
/// The constructor rejects negative external pressure and
/// non-finite detonation or burn times.
#[derive(Debug, Clone, Copy)]
pub struct Multimat51 {
    /// Unreacted equation of state.
    pub unreacted: JwlEos,
    /// Reacted equation of state.
    pub reacted: JwlEos,
    /// Ambient external pressure (Pa).
    pub pressure_ext: f64,
    /// Programmed detonation time (s).
    pub detonation_time: f64,
    /// Burn ramp duration (s); non-positive burns instantly.
    pub burn_time: f64,
}

impl Multimat51 {
    /// Builds the mixture with validated scalars.
    ///
    /// # Errors
    ///
    /// Returns a message for negative external pressure or
    /// non-finite detonation or burn times.
    pub fn new(
        unreacted: JwlEos,
        reacted: JwlEos,
        pressure_ext: f64,
        detonation_time: f64,
        burn_time: f64,
    ) -> Result<Self, String> {
        if !pressure_ext.is_finite() || pressure_ext < 0.0 {
            return Err(format!(
                "multimat external pressure must be non-negative, found {pressure_ext}"
            ));
        }
        if !detonation_time.is_finite() {
            return Err(format!(
                "multimat detonation time must be finite, found {detonation_time}"
            ));
        }
        if !burn_time.is_finite() {
            return Err(format!(
                "multimat burn time must be finite, found {burn_time}"
            ));
        }
        Ok(Self {
            unreacted,
            reacted,
            pressure_ext,
            detonation_time,
            burn_time,
        })
    }

    /// Burn fraction at time in [0, 1].
    pub fn burn_at(&self, time: f64) -> f64 {
        if self.burn_time <= 0.0 {
            if time >= self.detonation_time {
                1.0
            } else {
                0.0
            }
        } else {
            ((time - self.detonation_time) / self.burn_time).clamp(0.0, 1.0)
        }
    }

    /// Mixture pressure at state values (Pa).
    pub fn mixture_pressure(&self, mu: f64, energy_u: f64, energy_r: f64, burn: f64) -> f64 {
        let pu = self.unreacted.pressure(mu, energy_u);
        let pr = self.reacted.pressure(mu, energy_r);
        (1.0 - burn) * pu + burn * pr
    }

    /// Mixture pressure derivative over volumetric strain (Pa).
    fn mixture_dp_dmu(&self, mu: f64, energy_u: f64, energy_r: f64, burn: f64) -> f64 {
        let dpu = self.unreacted.dp_dmu(mu, energy_u);
        let dpr = self.reacted.dp_dmu(mu, energy_r);
        (1.0 - burn) * dpu + burn * dpr
    }

    /// Steps one increment with time step, bundling C1 calls.
    pub fn step(&self, state: &mut MultimatState, dstrain: &Voigt6, dt: f64) -> StressUpdate {
        let update = self.update_with_rate(state, dstrain, dt);
        self.commit(state, &update);
        self.advance_totals(state, dstrain, dt);
        update
    }

    /// Advances totals after `commit`; call once per accepted step.
    pub fn advance_totals(&self, state: &mut MultimatState, dstrain: &Voigt6, dt: f64) {
        let dmu = -(dstrain[0] + dstrain[1] + dstrain[2]);
        let mu = state.mu + dmu;
        let eta = (1.0 + mu).max(1.0e-12);
        let pu = self.unreacted.pressure(mu, state.energy_u);
        let pr = self.reacted.pressure(mu, state.energy_r);
        state.energy_u += pu / (self.unreacted.rho0 * eta * eta) * dmu;
        state.energy_r += pr / (self.reacted.rho0 * eta * eta) * dmu;
        state.mu = mu;
        state.time += dt.max(0.0);
        state.burn = self.burn_at(state.time);
    }
}

/// State of one multimat integration point.
#[derive(Debug, Clone)]
pub struct MultimatState {
    /// Volumetric strain, compression positive.
    pub mu: f64,
    /// Unreacted specific internal energy (J/kg).
    pub energy_u: f64,
    /// Reacted specific internal energy (J/kg).
    pub energy_r: f64,
    /// Burn fraction in [0, 1].
    pub burn: f64,
    /// Simulation time (s).
    pub time: f64,
    /// Last committed mixture pressure (Pa).
    pub pressure: f64,
    /// Accumulated equivalent plastic strain (always zero).
    pub equiv_plastic: f64,
}

impl MultimatState {
    /// Builds a virgin unburned state from initial energies.
    pub fn new(energy_u: f64, energy_r: f64) -> Self {
        Self {
            mu: 0.0,
            energy_u,
            energy_r,
            burn: 0.0,
            time: 0.0,
            pressure: 0.0,
            equiv_plastic: 0.0,
        }
    }
}

impl StressUpdateModel for Multimat51 {
    type State = MultimatState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, dstrain: &Self::StrainIncrement) -> StressUpdate {
        let mu = state.mu - (dstrain[0] + dstrain[1] + dstrain[2]);
        let pressure = self.mixture_pressure(mu, state.energy_u, state.energy_r, state.burn);
        let net = pressure - self.pressure_ext;
        let stress = [-net, -net, -net, 0.0, 0.0, 0.0];
        let slope = self.mixture_dp_dmu(mu, state.energy_u, state.energy_r, state.burn);
        let mut tangent = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                tangent[i * 6 + j] = slope;
            }
        }
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.pressure = -update.stress[0] + self.pressure_ext;
    }
}

// ---------------------------------------------------------------------------
// Material-point drivers
// ---------------------------------------------------------------------------

/// Bundled single-step advance for material-point drivers.
///
/// Every foam law implements this by delegating to its inherent
/// `step`, so one driver serves all laws while the C1 pair
/// stays host-compatible.
pub trait PointStepper: StressUpdateModel<StrainIncrement = [f64; 6]> {
    /// Advances one accepted increment with step time.
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate;
}

impl PointStepper for HoneycombLaw {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for ViscoFoam33 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for Cosserat68 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for CrushFoam68 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for Concrete24 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for FabricLaw {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for Samp76 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

impl PointStepper for Multimat51 {
    fn step_point(&self, state: &mut Self::State, dstrain: &[f64; 6], dt: f64) -> StressUpdate {
        self.step(state, dstrain, dt)
    }
}

/// Drives a foam law through uniaxial stress at fixed step time.
///
/// Holds axial increments at `target / steps`, solves lateral
/// strains by diagonal Newton (up to 100 sweeps) so transverse
/// stresses vanish, and records frozen [`UniaxialPoint`]
/// history. Diagonal Newton (instead of the seed 2x2 block)
/// supports membrane laws with zero tangent rows and converges
/// linearly for diagonally-dominant and isotropic tangents.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
/// use sonium_fem::physics::solid::plasticity::law_foams::{HoneycombLaw, HoneycombState, drive_uniaxial_foam};
///
/// let cap = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
/// let tables = [cap.clone(), cap.clone(), cap.clone(),
///     cap.clone(), cap.clone(), cap];
/// let law = HoneycombLaw::law28([2.0; 6], tables).unwrap();
/// let mut state = HoneycombState::new();
/// let curve = drive_uniaxial_foam(&law, &mut state, -0.5, 5, 1.0);
/// assert_eq!(curve.len(), 6);
/// assert!((curve[5].stress_xx + 0.5).abs() < 1.0e-12);
/// ```
///
/// # Panics
///
/// Panics when `steps` is zero or `dt` is not positive.
pub fn drive_uniaxial_foam<M: PointStepper>(
    law: &M,
    state: &mut M::State,
    target_xx: f64,
    steps: usize,
    dt: f64,
) -> Vec<UniaxialPoint> {
    assert!(steps > 0, "uniaxial driver needs at least one step");
    assert!(dt > 0.0, "uniaxial driver needs a positive step time");
    let mut curve = Vec::with_capacity(steps + 1);
    curve.push(UniaxialPoint {
        strain_xx: 0.0,
        stress_xx: 0.0,
        equiv_plastic_strain: 0.0,
        yielding: false,
    });
    let dx = target_xx / steps as f64;
    let mut lateral = [0.0, 0.0];
    let mut ep = 0.0;
    for step in 1..=steps {
        // Seed lateral strains from the previous step for fast Newton.
        let mut dlat = lateral;
        let mut trial = law.update_with_rate(state, &[dx, dlat[0], dlat[1], 0.0, 0.0, 0.0], dt);
        for _ in 0..100 {
            let scale = trial.tangent[0]
                .abs()
                .max(trial.tangent[7].abs())
                .max(trial.tangent[14].abs())
                .max(1.0);
            let r0 = trial.stress[1];
            let r1 = trial.stress[2];
            if !r0.is_finite() || !r1.is_finite() {
                break;
            }
            if r0.abs() < 1e-9 * scale && r1.abs() < 1e-9 * scale {
                break;
            }
            let mut moved = false;
            if trial.tangent[7].abs() > 0.0 {
                dlat[0] -= r0 / trial.tangent[7];
                moved = true;
            }
            if trial.tangent[14].abs() > 0.0 {
                dlat[1] -= r1 / trial.tangent[14];
                moved = true;
            }
            if !moved {
                break;
            }
            trial = law.update_with_rate(state, &[dx, dlat[0], dlat[1], 0.0, 0.0, 0.0], dt);
        }
        lateral = dlat;
        let update = law.step_point(state, &[dx, lateral[0], lateral[1], 0.0, 0.0, 0.0], dt);
        ep += update.delta_equiv_plastic_strain;
        curve.push(UniaxialPoint {
            strain_xx: dx * step as f64,
            stress_xx: update.stress[0],
            equiv_plastic_strain: ep,
            yielding: update.yielding,
        });
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_funct(value: f64) -> Funct1D {
        Funct1D::new(vec![-1.0, 1.0], vec![value, value]).unwrap()
    }

    fn honeycomb_flat(modulus: f64, cap: f64) -> HoneycombLaw {
        HoneycombLaw::law28([modulus; 6], core::array::from_fn(|_| flat_funct(cap))).unwrap()
    }

    fn rated_honeycomb() -> HoneycombLaw {
        let lo = flat_funct(1.0);
        let hi = flat_funct(2.0);
        let rated = core::array::from_fn(|_| vec![(lo.clone(), 0.0), (hi.clone(), 10.0)]);
        HoneycombLaw::law50([100.0; 6], rated).unwrap()
    }

    fn visco_flat() -> ViscoFoam33 {
        ViscoFoam33::new(3.0, 1.0, flat_funct(2.0), flat_funct(1.0), None).unwrap()
    }

    #[test]
    fn pressure_and_deviator_split() {
        let stress = [3.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((pressure_c1(&stress) + 1.0).abs() < 1.0e-15);
        let dev = deviator_c1(&stress);
        assert!((dev[0] + dev[1] + dev[2]).abs() < 1.0e-15);
        assert!((dev[3] - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn jacobi_roundtrips_stress() {
        let stress = [5.0, 2.0, 1.0, 0.5, -0.25, 0.75];
        let (values, vectors) = jacobi_eigen_c1(&stress, 12);
        let rebuilt = voigt_from_principal(&values, &vectors);
        for dir in 0..6 {
            assert!((rebuilt[dir] - stress[dir]).abs() < 1.0e-12);
        }
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(sorted[0] < sorted[2]);
    }

    #[test]
    fn jacobi_diagonal_is_exact() {
        let values = principal_stresses_c1(&[4.0, 1.0, 3.0, 0.0, 0.0, 0.0]);
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1.0e-12);
        assert!((sorted[1] - 3.0).abs() < 1.0e-12);
        assert!((sorted[2] - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn von_mises_matches_direct_formula() {
        let stress = [5.0, 2.0, 1.0, 0.5, -0.25, 0.75];
        assert!((von_mises_c1(&stress) - von_mises_direct(&stress)).abs() < 1.0e-12);
    }

    #[test]
    fn strain_basis_flags_and_abscissa() {
        assert_eq!(StrainBasis::from_flag(0).unwrap(), StrainBasis::Volumetric);
        assert_eq!(StrainBasis::from_flag(1).unwrap(), StrainBasis::Component);
        assert_eq!(StrainBasis::from_flag(-1).unwrap(), StrainBasis::Negated);
        assert!(StrainBasis::from_flag(2).is_err());
        assert!((StrainBasis::Volumetric.abscissa(0.3, 0.1) - 0.1).abs() < 1.0e-15);
        assert!((StrainBasis::Component.abscissa(0.3, 0.1) - 0.3).abs() < 1.0e-15);
        assert!((StrainBasis::Negated.abscissa(0.3, 0.1) + 0.3).abs() < 1.0e-15);
    }

    #[test]
    fn yield_table_interpolates_and_clamps() {
        let table = YieldTable::Rated(vec![(flat_funct(1.0), 0.0), (flat_funct(3.0), 10.0)]);
        assert!((table.eval(0.0, 5.0) - 2.0).abs() < 1.0e-15);
        assert!((table.eval(0.0, -1.0) - 1.0).abs() < 1.0e-15);
        assert!((table.eval(0.0, 99.0) - 3.0).abs() < 1.0e-15);
        assert!((rated_eval(&[], 0.0, 0.0)).abs() < 1.0e-15);
    }

    #[test]
    fn honeycomb_elastic_then_caps() {
        let law = honeycomb_flat(2.0, 0.5);
        let mut state = HoneycombState::new();
        let elastic = law.step(&mut state, &[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((elastic.stress[0] - 0.2).abs() < 1.0e-15);
        assert!(!elastic.yielding);
        assert!((elastic.tangent[0] - 2.0).abs() < 1.0e-15);
        let capped = law.step(&mut state, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((capped.stress[0] - 0.5).abs() < 1.0e-15);
        assert!(capped.yielding);
        assert!(capped.delta_equiv_plastic_strain > 0.0);
        assert!(state.equiv_plastic > 0.0);
    }

    #[test]
    fn honeycomb_yielding_tangent_stays_symmetric() {
        let law = honeycomb_flat(2.0, 0.5);
        let state = HoneycombState::new();
        let update = law.update(&state, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(update.yielding);
        for i in 0..6 {
            for j in 0..6 {
                assert!((update.tangent[i * 6 + j] - update.tangent[j * 6 + i]).abs() < 1.0e-18);
            }
        }
    }

    #[test]
    fn honeycomb_failure_zeroes_response() {
        let mut law = honeycomb_flat(2.0, 50.0);
        law.failure[0] = 0.05;
        let mut state = HoneycombState::new();
        let update = law.step(&mut state, &[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(state.failed);
        assert!(update.stress.iter().all(|s| s.abs() < 1.0e-15));
        let repeat = law.update(&state, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(repeat.stress.iter().all(|s| s.abs() < 1.0e-15));
        assert!(repeat.yielding);
    }

    #[test]
    fn honeycomb_rate_tables_respond_to_rate() {
        let law = rated_honeycomb();
        assert!(law.is_rate_dependent());
        assert!(!honeycomb_flat(2.0, 0.5).is_rate_dependent());
        let state = HoneycombState::new();
        let slow = law.update_with_rate(&state, &[0.05, 0.0, 0.0, 0.0, 0.0, 0.0], f64::INFINITY);
        let fast = law.update_with_rate(&state, &[0.05, 0.0, 0.0, 0.0, 0.0, 0.0], 0.005);
        assert!(slow.yielding && fast.yielding);
        assert!(fast.stress[0] > slow.stress[0]);
        assert!((slow.stress[0] - 1.0).abs() < 1.0e-12);
        assert!((fast.stress[0] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn honeycomb_rate_filter_lags_raw_rate() {
        let mut law = rated_honeycomb();
        law.rate_cutoff = 0.01;
        let mut state = HoneycombState::new();
        law.step(&mut state, &[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(state.rates[0] > 0.0);
        assert!(state.rates[0] < 0.1);
    }

    #[test]
    fn visco_air_pressure_rises_in_compression() {
        let mut law = visco_flat();
        assert!((law.air_pressure(0.5) - law.pressure0).abs() < 1.0e-9);
        law.density_ratio = Some(1.0);
        assert!(law.air_pressure(0.5) > law.air_pressure(0.0));
        assert!(law.air_pressure(0.0) > 0.0);
    }

    #[test]
    fn visco_caps_principal_tension() {
        let law = visco_flat();
        let mut state = ViscoFoamState::new();
        let update = law.step(&mut state, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(update.yielding);
        assert!((update.stress[0] - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn visco_pressure_floor_binds() {
        let mut law = visco_flat();
        law.pressure_min = 0.0;
        let mut state = ViscoFoamState::new();
        let update = law.step(&mut state, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(pressure_c1(&update.stress) > -1.0e-9);
    }

    #[test]
    fn visco_elastic_tangent_matches_seed() {
        let law = visco_flat();
        let state = ViscoFoamState::new();
        let update = law.update(&state, &[1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(!update.yielding);
        let expected = elastic_tangent_c1(1.0, 3.0);
        for (got, want) in update.tangent.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-15);
        }
    }

    #[test]
    fn visco_rate_table_scales_caps() {
        let plain = visco_flat();
        let mut scaled = visco_flat();
        scaled.funct_rate = Some(vec![(flat_funct(2.0), 0.0)]);
        assert!(scaled.is_rate_dependent());
        assert!(!plain.is_rate_dependent());
        let dstrain = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let base = plain.update(&ViscoFoamState::new(), &dstrain);
        let boosted = scaled.update(&ViscoFoamState::new(), &dstrain);
        assert!(base.yielding && boosted.yielding);
        assert!((boosted.stress[0] - 2.0 * base.stress[0]).abs() < 1.0e-9);
    }

    fn cosserat() -> Cosserat68 {
        Cosserat68::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6, 1.0e9, 0.1).unwrap()
    }

    #[test]
    fn cosserat_couple_advances_elastically() {
        let law = cosserat();
        let mut state = CosseratState::new();
        let update = law.step_curve(&mut state, &[0.0; 6], &[1.0e-4, 0.0, 0.0], 1.0);
        assert!(!update.yielding);
        assert!((state.couple[0] - 200_000.0 / 2.2).abs() < 1.0e-6);
        assert!((state.curvature[0] - 1.0e-4).abs() < 1.0e-18);
        assert!(update.delta_equiv_plastic_strain < 1.0e-18);
    }

    #[test]
    fn cosserat_couple_caps_at_crush_strength() {
        let law = cosserat();
        let mut state = CosseratState::new();
        let update = law.step_curve(&mut state, &[0.0; 6], &[0.1, 0.0, 0.0], 1.0);
        assert!(update.yielding);
        let norm = (state.couple[0] * state.couple[0]
            + state.couple[1] * state.couple[1]
            + state.couple[2] * state.couple[2])
            .sqrt();
        assert!(((norm - 5.0e6) / 5.0e6).abs() < 1.0e-12);
        assert!(update.delta_equiv_plastic_strain > 0.0);
        assert!(state.equiv_plastic > 0.0);
    }

    #[test]
    fn cosserat_delegates_symmetric_response() {
        let law = cosserat();
        let state = CosseratState::new();
        let dstrain = [-0.01, 0.002, 0.001, 0.0, 0.0, 0.0];
        let outer = law.update(&state, &dstrain);
        let inner = law.foam.update(&state.foam, &dstrain);
        for dir in 0..6 {
            assert!((outer.stress[dir] - inner.stress[dir]).abs() < 1.0e-18);
        }
        assert_eq!(outer.yielding, inner.yielding);
    }

    #[test]
    fn crushfoam68_matches_seed() {
        let law = CrushFoam68::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6);
        let seed = CrushableFoamMaterial::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6);
        let state = CrushableFoamState::new();
        let dstrain = [-0.01, 0.002, 0.001, 0.0, 0.0, 0.0];
        let outer = law.update(&state, &dstrain);
        let inner = seed.update(&state, &dstrain);
        for dir in 0..6 {
            assert!((outer.stress[dir] - inner.stress[dir]).abs() < 1.0e-18);
        }
    }

    fn mazars() -> MazarsDamage {
        MazarsDamage {
            strain0: 1.0e-4,
            at: 0.8,
            bt: 1.5e4,
            ac: 1.0,
            bc: 2.0e3,
        }
    }

    fn concrete() -> Concrete24 {
        Concrete24::new(30.0e9, 0.2, mazars(), 0.0, 30.0e6, 0.0).unwrap()
    }

    #[test]
    fn concrete_damage_is_irreversible() {
        let law = concrete();
        let mut state = ConcreteState::new();
        let loaded = law.step(&mut state, &[1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(loaded.yielding);
        assert!(state.damage > 0.5);
        let damaged = state.damage;
        law.step(&mut state, &[-1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((state.damage - damaged).abs() < 1.0e-15);
    }

    #[test]
    fn concrete_visco_blend_halves_first_step() {
        let slow = Concrete24::new(30.0e9, 0.2, mazars(), 1.0, 30.0e6, 0.0).unwrap();
        assert!(slow.is_rate_dependent());
        assert!(!concrete().is_rate_dependent());
        let state = ConcreteState::new();
        let dstrain = [1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let inviscid = slow.update(&state, &dstrain);
        let blended = slow.update_with_rate(&state, &dstrain, 1.0);
        assert!((blended.stress[0] - 0.5 * inviscid.stress[0]).abs() < 1.0e-6);
    }

    #[test]
    fn concrete_cap_binds_hydrostatic() {
        let law = concrete();
        let mut state = ConcreteState::new();
        let update = law.step(&mut state, &[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0], 1.0);
        assert!(update.yielding);
        assert!(((pressure_c1(&update.stress) - 30.0e6) / 30.0e6).abs() < 1.0e-9);
    }

    #[test]
    fn concrete_rejects_bad_params() {
        assert!(Concrete24::new(-1.0, 0.2, mazars(), 0.0, 30.0e6, 0.0).is_err());
        assert!(Concrete24::new(30.0e9, 0.6, mazars(), 0.0, 30.0e6, 0.0).is_err());
        let mut bad = mazars();
        bad.at = 2.0;
        assert!(Concrete24::new(30.0e9, 0.2, bad, 0.0, 30.0e6, 0.0).is_err());
        bad = mazars();
        bad.bt = -1.0;
        assert!(Concrete24::new(30.0e9, 0.2, bad, 0.0, 30.0e6, 0.0).is_err());
        assert!(Concrete24::new(30.0e9, 0.2, mazars(), -1.0, 30.0e6, 0.0).is_err());
        assert!(Concrete24::new(30.0e9, 0.2, mazars(), 0.0, 0.0, 0.0).is_err());
        assert!(Concrete24::new(30.0e9, 0.2, mazars(), 0.0, 30.0e6, -1.0).is_err());
    }

    fn fabric19() -> FabricLaw {
        FabricLaw::law19(
            [10.0e6, 10.0e6],
            [1.0e6, 1.0e6],
            2.0e6,
            [0.1, 0.1],
            [0.2, 0.2],
            0.3,
        )
        .unwrap()
    }

    #[test]
    fn fabric_splits_tension_compression() {
        let law = fabric19();
        let mut tension = FabricState::new();
        let pulled = law.step(&mut tension, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((pulled.stress[0] - 100.0e3).abs() < 1.0e-6);
        let mut compression = FabricState::new();
        let pushed = law.step(&mut compression, &[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((pushed.stress[0] + 10.0e3).abs() < 1.0e-9);
        let mut shear = FabricState::new();
        let sheared = law.step(&mut shear, &[0.0, 0.0, 0.0, 0.01, 0.0, 0.0], 1.0);
        assert!((sheared.stress[3] - 20.0e3).abs() < 1.0e-9);
        assert!(sheared.stress[2].abs() < 1.0e-18);
    }

    #[test]
    fn fabric_failure_zeroes_response() {
        let law = fabric19();
        let mut tension = FabricState::new();
        let pulled = law.step(&mut tension, &[0.15, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!(tension.failed);
        assert!(pulled.stress.iter().all(|s| s.abs() < 1.0e-18));
        let mut shear = FabricState::new();
        law.step(&mut shear, &[0.0, 0.0, 0.0, 0.5, 0.0, 0.0], 1.0);
        assert!(shear.failed);
    }

    #[test]
    fn fabric_law58_unloads_along_modulus() {
        let warp = FabricDirection::new(
            Funct1D::new(vec![0.0, 0.1], vec![0.0, 1.0e6]).unwrap(),
            20.0e6,
        )
        .unwrap();
        let law = FabricLaw::law58(
            [10.0e6, 10.0e6],
            [1.0e6, 1.0e6],
            2.0e6,
            [0.1, 0.1],
            [0.2, 0.2],
            0.3,
            warp.clone(),
            warp,
        )
        .unwrap();
        let mut state = FabricState::new();
        let loaded = law.step(&mut state, &[0.05, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((loaded.stress[0] - 0.5e6).abs() < 1.0e-6);
        let unloaded = law.step(&mut state, &[-0.03, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert!((unloaded.stress[0] + 0.1e6).abs() < 1.0e-6);
        assert!((state.max_strain[0] - 0.05).abs() < 1.0e-15);
    }

    fn samp_flat() -> Samp76 {
        Samp76::new(
            1000.0,
            0.3,
            flat_funct(10.0),
            flat_funct(20.0),
            flat_funct(6.0),
        )
        .unwrap()
    }

    #[test]
    fn samp_yield_interpolates_stations() {
        let law = samp_flat();
        assert!((law.yield_at(0.0, -1.0 / 3.0) - 10.0).abs() < 1.0e-12);
        assert!((law.yield_at(0.0, 0.0) - 6.0).abs() < 1.0e-12);
        assert!((law.yield_at(0.0, 1.0 / 3.0) - 20.0).abs() < 1.0e-12);
        assert!((law.yield_at(0.0, -1.0 / 6.0) - 8.0).abs() < 1.0e-12);
        assert!((law.yield_at(0.0, -2.0 / 3.0) - 10.0).abs() < 1.0e-12);
        assert!((law.yield_at(0.0, 1.0) - 20.0).abs() < 1.0e-12);
        let mut biaxial = samp_flat();
        biaxial.biaxial = Some(flat_funct(15.0));
        assert!((biaxial.yield_at(0.0, -2.0 / 3.0) - 15.0).abs() < 1.0e-12);
    }

    #[test]
    fn samp_failure_zeroes_and_latches() {
        let mut law = samp_flat();
        law.failure.shear = 0.001;
        let mut state = SampState::new();
        let update = law.step(&mut state, &[0.0, 0.0, 0.0, 0.1, 0.0, 0.0], 1.0);
        assert!(update.stress.iter().all(|s| s.abs() < 1.0e-18));
        assert!(state.failed);
        assert!(state.equiv_plastic > 0.001);
    }

    fn jwl() -> JwlEos {
        JwlEos::new(2.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, 1.0e5).unwrap()
    }

    #[test]
    fn jwl_pressure_matches_hand_formula() {
        let eos = jwl();
        let eta: f64 = 1.3;
        let energy = 2.0e5;
        let first = 2.0e9 * (1.0 - 0.25 * eta / 4.5) * (-4.5 / eta).exp();
        let second = 5.0e8 * (1.0 - 0.25 * eta / 1.2) * (-1.2 / eta).exp();
        let expected = first + second + 0.25 * 1500.0 * eta * energy;
        assert!(((eos.pressure(0.3, energy) - expected) / expected).abs() < 1.0e-12);
    }

    #[test]
    fn jwl_derivatives_match_numeric() {
        let eos = jwl();
        let mu = 0.2;
        let energy = 2.0e5;
        let h = 1.0e-7;
        let numeric_mu = (eos.pressure(mu + h, energy) - eos.pressure(mu - h, energy)) / (2.0 * h);
        assert!(((eos.dp_dmu(mu, energy) - numeric_mu) / numeric_mu).abs() < 1.0e-6);
        let he = 1.0;
        let numeric_e =
            (eos.pressure(mu, energy + he) - eos.pressure(mu, energy - he)) / (2.0 * he);
        assert!(((eos.dp_de(mu) - numeric_e) / numeric_e).abs() < 1.0e-9);
    }

    #[test]
    fn multimat_mixes_by_burn() {
        let unreacted = JwlEos::new(1.0e9, 0.0, 4.0, 1.0, 0.3, 1000.0, 0.0).unwrap();
        let reacted = JwlEos::new(2.0e9, 0.0, 4.0, 1.0, 0.3, 1000.0, 0.0).unwrap();
        let law = Multimat51::new(unreacted, reacted, 1.0e5, 0.0, 0.0).unwrap();
        let fresh = MultimatState::new(0.0, 0.0);
        let zero = [0.0; 6];
        let unburned = law.update(&fresh, &zero);
        assert!((unburned.stress[0] + unreacted.pressure(0.0, 0.0) - 1.0e5).abs() < 1.0e-6);
        let mut burned = fresh.clone();
        burned.burn = 1.0;
        let reacted_update = law.update(&burned, &zero);
        assert!((reacted_update.stress[0] + reacted.pressure(0.0, 0.0) - 1.0e5).abs() < 1.0e-6);
    }

    #[test]
    fn multimat_burn_ramps_and_energy_rises() {
        let eos = JwlEos::new(1.0e9, 0.0, 4.0, 1.0, 0.3, 1000.0, 0.0).unwrap();
        let ramp = Multimat51::new(eos, eos, 0.0, 2.0, 2.0).unwrap();
        assert!((ramp.burn_at(1.0)).abs() < 1.0e-15);
        assert!((ramp.burn_at(3.0) - 0.5).abs() < 1.0e-15);
        assert!((ramp.burn_at(5.0) - 1.0).abs() < 1.0e-15);
        let instant = Multimat51::new(eos, eos, 0.0, 2.0, 0.0).unwrap();
        assert!((instant.burn_at(1.99)).abs() < 1.0e-15);
        assert!((instant.burn_at(2.0) - 1.0).abs() < 1.0e-15);
        let mut state = MultimatState::new(0.0, 0.0);
        ramp.step(&mut state, &[-0.05, -0.05, -0.05, 0.0, 0.0, 0.0], 3.0);
        assert!(state.energy_u > 0.0);
        assert!((state.burn - 0.5).abs() < 1.0e-15);
        assert!((state.time - 3.0).abs() < 1.0e-15);
    }

    #[test]
    fn zeroed_update_has_no_stiffness() {
        let update = zeroed_update();
        assert!(update.stress.iter().all(|s| s.abs() < 1.0e-18));
        assert!(update.tangent.iter().all(|t| t.abs() < 1.0e-18));
        assert!(update.yielding);
    }

    #[test]
    fn driver_tracks_honeycomb_curve() {
        let law = honeycomb_flat(2.0, 0.5);
        let mut state = HoneycombState::new();
        let curve = drive_uniaxial_foam(&law, &mut state, -0.5, 5, 1.0);
        assert_eq!(curve.len(), 6);
        assert!((curve[5].strain_xx + 0.5).abs() < 1.0e-15);
        assert!((curve[5].stress_xx + 0.5).abs() < 1.0e-12);
        assert!(!curve[1].yielding);
        assert!(curve[5].yielding);
        assert!(curve[5].equiv_plastic_strain > 0.0);
    }

    #[test]
    fn driver_converges_samp_uniaxial_stress() {
        let law = Samp76::new(
            1000.0,
            0.3,
            flat_funct(10.0),
            flat_funct(10.0),
            flat_funct(10.0),
        )
        .unwrap();
        let mut state = SampState::new();
        let curve = drive_uniaxial_foam(&law, &mut state, 0.02, 4, 1.0);
        assert_eq!(curve.len(), 5);
        assert!((curve[4].stress_xx - 10.0).abs() < 1.0e-2);
        assert!(curve[4].yielding);
        assert!(curve[4].equiv_plastic_strain > 0.0);
    }

    #[test]
    fn driver_handles_fabric_membrane() {
        let law = fabric19();
        let mut state = FabricState::new();
        let curve = drive_uniaxial_foam(&law, &mut state, 0.05, 5, 1.0);
        assert_eq!(curve.len(), 6);
        assert!((curve[5].stress_xx - 500.0e3).abs() < 1.0e-6);
    }
}
