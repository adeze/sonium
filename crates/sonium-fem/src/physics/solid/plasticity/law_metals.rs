//! Crash metal laws behind the frozen C1 contract.
//!
//! Implements the Radioss metal plasticity laws owned by Phase-3 lane 3B:
//! [LAW2](Law2) (Johnson-Cook, full form with temperature, caps, rupture,
//! combined hardening, and the Zerilli-Armstrong variant), [LAW22](Law22)
//! (ductile damage), [LAW27](Law27) (brittle damage), [LAW36](Law36)
//! (tabulated piecewise linear), [LAW44](Law44) (Cowper-Symonds),
//! [LAW62](Law62) (hyper-viscoelastic Ogden-Prony), and [LAW66](Law66)
//! (tension-compression tabulated). Every law implements the frozen
//! [`StressUpdateModel`](super::update::StressUpdateModel) with C1 Voigt
//! increments `[f64; 6]` in SI units, reuses the shared
//! [`j2_return_c1`](super::crash#) radial return for isotropic paths plus
//! the frozen [`Funct1D`](super::funct::Funct1D)/[`Table2D`](super::funct::Table2D)
//! readers for tabulated data, and stays pure in `update` (history moves
//! only in `commit`).
//!
//! # Equation sources
//!
//! Flow and damage equations come from the Altair Radioss 2022 Theory
//! Manual with section and equation citations on each item; card-level
//! details (defaults, flags, rupture behavior) come from the matching
//! `/MAT` card in the Reference Guide. Two deliberate deviations are
//! documented where they occur: the Zhao/LAW44 tensile-reduction law is
//! applied as a total scaling rather than the step-recursive product
//! (which would depend on the step count), and tabulated laws extrapolate
//! linearly past their last knot where the manual requires extrapolation
//! even though the frozen readers clamp (the extension reuses the frozen
//! end slope, so no reader behavior changes).
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::law_metals::{Law2, Law2State};
//! use sonium_fem::physics::solid::plasticity::StressUpdateModel;
//!
//! let steel = Law2::c4340_steel();
//! let state = Law2State::new(298.0);
//! let update = steel.update(&state, &[0.01, -0.003, -0.003, 0.0, 0.0, 0.0]);
//! assert!(update.yielding);
//! ```

// Rust guideline compliant 2026-02-21

use super::crash::{consistent_tangent_j2_c1, elastic_tangent_c1, j2_return_c1};
use super::funct::{Funct1D, Table2D};
use super::update::{StressUpdate, StressUpdateModel, voigt_equivalent_increment_c1, von_mises_c1};
use crate::physics::solid::fracture::ductile::JohnsonCookDamage;
use std::f64::consts::PI;

/// J2 radial return with combined isotropic-kinematic hardening.
///
/// Behaves exactly like the shared [`j2_return_c1`] return when `chard`
/// is zero and `back` vanishes; otherwise the yield surface grows with
/// `(1 - chard)` of the hardening curve above `sig0` while the
/// back-stress translates with Prager modulus `chard * slope`. The trial
/// direction comes from the relative stress (trial deviator minus `back`),
/// and the consistent tangent reuses the shared Simo-Hughes form with the
/// summed moduli. Returns `(stress, tangent, dep, yielding, back_new)`.
#[expect(clippy::too_many_arguments, reason = "return mappings take flat state")]
fn j2_return_combined_c1(
    e: f64,
    nu: f64,
    stress: &[f64; 6],
    back: &[f64; 6],
    ep: f64,
    chard: f64,
    sig0: f64,
    d_eps: &[f64; 6],
    sig_y: impl Fn(f64) -> f64,
    slope: impl Fn(f64) -> f64,
) -> ([f64; 6], [f64; 36], f64, bool, [f64; 6]) {
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    let g2 = 2.0 * g;

    let vol = d_eps[0] + d_eps[1] + d_eps[2];
    let p_old = (stress[0] + stress[1] + stress[2]) / 3.0;
    let p_trial = p_old + k * vol;
    let mut s_trial = [0.0; 6];
    for i in 0..3 {
        s_trial[i] = stress[i] - p_old + g2 * (d_eps[i] - vol / 3.0);
    }
    for i in 3..6 {
        s_trial[i] = stress[i] + g2 * 0.5 * d_eps[i];
    }
    // Relative trial stress and its von Mises norm (the hydrostatic part
    // added here is removed again inside von_mises_c1).
    let xi: [f64; 6] = std::array::from_fn(|i| s_trial[i] - back[i]);
    let xi_full = [
        xi[0] + p_trial,
        xi[1] + p_trial,
        xi[2] + p_trial,
        xi[3],
        xi[4],
        xi[5],
    ];
    let q_xi = von_mises_c1(&xi_full);
    let iso = |epe: f64| sig0 + (1.0 - chard) * (sig_y(epe) - sig0);

    if q_xi <= iso(ep) {
        let mut elastic = s_trial;
        elastic[0] += p_trial;
        elastic[1] += p_trial;
        elastic[2] += p_trial;
        return (elastic, elastic_tangent_c1(k, g), 0.0, false, *back);
    }

    // Consistency: q_xi - (3G + Ck) dep - iso(ep + dep) = 0 with the
    // Prager modulus Ck = chard * slope at the end of the step. Same
    // bracket-then-bisect solve as the shared isotropic return.
    let residual = |dep: f64| {
        let ck = chard * slope(ep + dep).max(0.0);
        q_xi - (3.0 * g + ck) * dep - iso(ep + dep)
    };
    let mut lo = 0.0;
    let mut hi = ((q_xi - iso(ep)) / (3.0 * g)).max(1e-15);
    for _ in 0..60 {
        if residual(hi) <= 0.0 {
            break;
        }
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if residual(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dep = 0.5 * (lo + hi);
    let ck = chard * slope(ep + dep).max(0.0);
    // Radial update along the relative-trial direction: stress recoils
    // with 3G dep while the back-stress advances with Ck dep.
    let mut new_stress = [0.0; 6];
    let mut back_new = [0.0; 6];
    for i in 0..6 {
        let dir = xi[i] / q_xi;
        let recoil = 3.0 * g * dep * dir;
        back_new[i] = back[i] + ck * dep * dir;
        new_stress[i] = s_trial[i] - recoil + if i < 3 { p_trial } else { 0.0 };
    }
    let h_total = (1.0 - chard) * slope(ep + dep).max(0.0) + ck;
    let tangent = consistent_tangent_j2_c1(k, g, q_xi, dep, &xi, h_total);
    (new_stress, tangent, dep, true, back_new)
}

/// Principal values/vectors of a symmetric C1 stress tensor.
///
/// Uses cyclic Jacobi rotations (at most 16 sweeps with a scale-aware
/// stop), which is bulletproof for 3x3 and needs no branching on
/// degeneracies. Returns unsorted values with eigenvector columns.
/// Stress slots 3..6 already hold tensor shear; strain callers must use
/// [`principal_strain_c1`] (engineering shear is twice tensor shear).
#[allow(
    clippy::needless_range_loop,
    reason = "Jacobi rotations read clearest with matrix indices"
)]
fn principal_system_c1(t: &[f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [[t[0], t[3], t[5]], [t[3], t[1], t[4]], [t[5], t[4], t[2]]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let scale = (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()).max(1.0);
    for _ in 0..16 {
        let mut p = 0;
        let mut q = 1;
        let mut big = 0.0;
        for i in 0..3 {
            for j in (i + 1)..3 {
                let magnitude = a[i][j].abs();
                if magnitude > big {
                    big = magnitude;
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1e-15 * scale {
            break;
        }
        // Stable rotation angle via the tau/t/c/s form (no atan2 branch
        // cut near equal diagonals).
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t_rot = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t_rot * t_rot).sqrt();
        let s = t_rot * c;
        for i in 0..3 {
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[i][q] = s * aip + c * aiq;
        }
        for j in 0..3 {
            let apj = a[p][j];
            let aqj = a[q][j];
            a[p][j] = c * apj - s * aqj;
            a[q][j] = s * apj + c * aqj;
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let mean = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = mean;
                a[j][i] = mean;
            }
        }
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Principal system of a C1 strain tensor (shear slots halved).
///
/// Same Jacobi eigensolver as [`principal_system_c1`] with the
/// engineering-shear slots converted to tensor shear first; values and
/// vectors both describe the true strain tensor.
fn principal_strain_c1(t: &[f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    principal_system_c1(&[t[0], t[1], t[2], 0.5 * t[3], 0.5 * t[4], 0.5 * t[5]])
}

/// Largest principal value of a C1 strain tensor.
fn max_principal_strain_c1(t: &[f64; 6]) -> f64 {
    let (values, _) = principal_strain_c1(t);
    values[0].max(values[1]).max(values[2])
}

/// Rebuilds a C1 tensor from principal values and eigenvector columns.
#[cfg(test)]
fn c1_from_principal(values: &[f64; 3], vecs: &[[f64; 3]; 3]) -> [f64; 6] {
    let mut full = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            full[i][j] = vecs[i][0] * values[0] * vecs[j][0]
                + vecs[i][1] * values[1] * vecs[j][1]
                + vecs[i][2] * values[2] * vecs[j][2];
        }
    }
    [
        full[0][0], full[1][1], full[2][2], full[0][1], full[1][2], full[0][2],
    ]
}

/// Johnson-Cook rate factor with a quiet quasi-static branch.
///
/// Returns exactly 1 when the rate sits at or below the reference rate
/// (Reference Guide `/MAT/LAW2` comment 8: no rate effect there) or when
/// `c` vanishes; otherwise `1 + c ln(rate / ref)`, floored to stay
/// positive so the yield surface never inverts.
fn jc_rate_factor(rate: f64, c: f64, ref_rate: f64) -> f64 {
    if c == 0.0 || ref_rate <= 0.0 || rate <= ref_rate {
        1.0
    } else {
        // Floor mirrors crash.rs (1e-3): negative c with a huge rate ratio
        // could otherwise flip the yield sign and break the return solve.
        (1.0 + c * (rate / ref_rate).ln()).max(1e-3)
    }
}

/// Cowper-Symonds rate factor `1 + (rate / c)^(1 / p)`.
///
/// Returns 1 when `c <= 0` (Reference Guide `/MAT/LAW44`: `c = 0` means
/// no rate effect) or the rate is non-positive; `p` stays positive by
/// construction.
fn cs_rate_factor(rate: f64, c: f64, p: f64) -> f64 {
    if c <= 0.0 || rate <= 0.0 {
        1.0
    } else {
        1.0 + (rate / c).powf(1.0 / p)
    }
}

/// Advances the Radioss strain-rate filter one step.
///
/// Implements Theory Manual Eq. 921, `ef(t) = a e(t) + (1 - a) ef(t-1)`
/// with `a = 2 pi dt Fcut`, clamped to 1 so coarse steps stay stable.
/// Drivers apply this between steps and feed the smoothed rate to the
/// filtered-rate entry points; the default `Fcut = 1e30` collapses to the
/// raw rate, which matches the card default of no filtering.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::smooth_rate_step;
///
/// let first = smooth_rate_step(0.0, 100.0, 1.0e-6, 10.0e3);
/// assert!(first > 0.0 && first < 100.0);
/// let off = smooth_rate_step(0.0, 100.0, 1.0e-6, 0.0);
/// assert_eq!(off, 100.0);
/// ```
pub fn smooth_rate_step(prev_filtered: f64, raw: f64, dt: f64, fcut: f64) -> f64 {
    if fcut <= 0.0 || dt <= 0.0 {
        return raw;
    }
    let a = (2.0 * PI * dt * fcut).min(1.0);
    a * raw + (1.0 - a) * prev_filtered
}

/// Tensile softening scale between rupture strain limits.
///
/// Returns 1 below `t1`, 0 above `t2`, and the linear ramp `(t2 - e1) /
/// (t2 - t1)` between them. Applies the Zhao/LAW44 Eq. 239 reduction as a
/// total scaling of the trial stress: the manual's step-recursive product
/// would decay with the step count, while the ramp reaches zero exactly
/// at `t2` for any stepping.
fn tensile_scale(first_principal: f64, t1: f64, t2: f64) -> f64 {
    if first_principal <= t1 {
        1.0
    } else if first_principal >= t2 {
        0.0
    } else {
        (t2 - first_principal) / (t2 - t1)
    }
}

/// Volumetric-only tangent for ruptured points (bulk block, else zero).
fn ruptured_tangent_c1(k: f64) -> [f64; 36] {
    let mut tangent = [0.0; 36];
    for i in 0..3 {
        for j in 0..3 {
            tangent[i * 6 + j] = k;
        }
    }
    tangent
}

/// Pressure-only update for a ruptured solid point.
///
/// Keeps the hydrostatic response elastic (`p += K dV`) with the
/// deviatoric stress permanently zero, per the `/MAT/LAW2` comment 5
/// solid-element rule shared by the metal laws.
fn ruptured_update_c1(k: f64, stress: &[f64; 6], d_eps: &[f64; 6]) -> StressUpdate {
    let p_old = (stress[0] + stress[1] + stress[2]) / 3.0;
    let p_new = p_old + k * (d_eps[0] + d_eps[1] + d_eps[2]);
    StressUpdate {
        stress: [p_new, p_new, p_new, 0.0, 0.0, 0.0],
        tangent: ruptured_tangent_c1(k),
        delta_equiv_plastic_strain: 0.0,
        yielding: true,
    }
}

/// Rate effect on the maximum flow stress (`ICC` card flag).
///
/// `Scaled` (ICC = 1) rates the cap with the same factor as the flow
/// stress; `Fixed` (ICC = 2) holds the cap at its input value. Shared by
/// LAW2, LAW22, and LAW44 (Reference Guide Figures 74, 77, 87).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxRateEffect {
    /// Maximum stress scales with the strain-rate factor.
    Scaled,
    /// Maximum stress stays at its input value.
    Fixed,
}

/// LAW2 flow-stress branch: Johnson-Cook or Zerilli-Armstrong.
///
/// Both branches share the reference rate, thermal data, caps, rupture,
/// and hardening split of [`Law2`]; only the flow evaluation differs
/// (Theory Manual Eq. 899 vs Eq. 922).
#[derive(Debug, Clone, Copy)]
pub enum Law2Flow {
    /// Johnson-Cook parameters `a`, `b`, `n`, `c` (SI: Pa, Pa, -, -).
    JohnsonCook {
        /// Initial yield strength in Pa.
        a: f64,
        /// Hardening coefficient in Pa.
        b: f64,
        /// Hardening exponent, at most 1.
        n: f64,
        /// Rate sensitivity, dimensionless.
        c: f64,
    },
    /// Zerilli-Armstrong parameters (SI: Pa, Pa, 1/K, 1/K, Pa, -).
    ZerilliArmstrong {
        /// Yield stress in Pa.
        c0: f64,
        /// Rate-temperature coefficient in Pa.
        c1: f64,
        /// Thermal softening in 1/K.
        c3: f64,
        /// Rate-temperature coupling in 1/K.
        c4: f64,
        /// Hardening coefficient in Pa.
        c5: f64,
        /// Hardening exponent, at most 1.
        n: f64,
    },
}

impl Law2Flow {
    /// Flow stress at plastic strain, rate, and temperature.
    ///
    /// Takes equivalent plastic strain (dimensionless), equivalent
    /// plastic strain rate (1/s), and temperature (K); returns Pa. The
    /// Johnson-Cook branch applies Eq. 899 without the thermal factor
    /// (the caller multiplies it); the Zerilli-Armstrong branch evaluates
    /// Eq. 922 with the absolute temperature.
    pub fn flow_stress(&self, ep: f64, rate: f64, temperature: f64, ref_rate: f64) -> f64 {
        match *self {
            Law2Flow::JohnsonCook { a, b, n, c } => {
                let hardening = a + b * ep.max(0.0).powf(n);
                hardening * jc_rate_factor(rate, c, ref_rate)
            }
            Law2Flow::ZerilliArmstrong {
                c0,
                c1,
                c3,
                c4,
                c5,
                n,
            } => {
                let star = (rate.max(1e-12) / ref_rate.max(1e-30)).max(1e-9);
                c0 + c1 * (-c3 * temperature + c4 * temperature * star.ln()).exp()
                    + c5 * ep.max(0.0).powf(n)
            }
        }
    }

    /// Hardening slope `d sigma / d ep` at plastic strain.
    ///
    /// Analytic derivative with the plastic strain floored at 1e-12 (keeps
    /// the singular `n < 1` slope finite); the return mapping solves the
    /// consistency equation exactly, so the slope only feeds the tangent
    /// and the kinematic split.
    pub fn hardening_slope(&self, ep: f64, rate: f64, _temperature: f64, ref_rate: f64) -> f64 {
        // Temperature enters the flow level, never its strain derivative.
        let ep_safe = ep.max(1e-12);
        match *self {
            Law2Flow::JohnsonCook { b, n, c, .. } => {
                b * n * ep_safe.powf(n - 1.0) * jc_rate_factor(rate, c, ref_rate)
            }
            Law2Flow::ZerilliArmstrong { c5, n, .. } => c5 * n * ep_safe.powf(n - 1.0),
        }
    }
}

/// Full Johnson-Cook metal law (`/MAT/LAW2`, `PLAS_JOHNS`).
///
/// Flow stress follows Theory Manual Eq. 899 with the quiet
/// below-reference rate branch, cap `sig_max0` with `ICC` scaling,
/// rupture at `eps_max` (solids keep pressure, lose deviatoric stress),
/// combined isotropic-kinematic hardening via `chard`, adiabatic heating
/// from the plastic work (Reference Guide Eq. 210), and optional
/// Johnson-Cook damage accumulation. Set `flow` to the
/// Zerilli-Armstrong branch for Eq. 922 instead.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law2;
///
/// let steel = Law2::c4340_steel();
/// let quasi = steel.flow_stress(0.01, 1.0, 298.0);
/// let fast = steel.flow_stress(0.01, 1000.0, 298.0);
/// assert!(fast > quasi);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Law2 {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Flow-stress branch parameters.
    pub flow: Law2Flow,
    /// Reference strain rate in 1/s.
    pub reference_strain_rate: f64,
    /// Thermal softening exponent, dimensionless.
    pub m: f64,
    /// Melt temperature in K (`<= ref` disables temperature).
    pub melt_temperature: f64,
    /// Reference temperature in K (card default 298).
    pub reference_temperature: f64,
    /// Volumetric heat capacity in J/(m^3 K); 0 disables heating.
    pub volumetric_heat_capacity: f64,
    /// Maximum flow stress in Pa (card default 1e30).
    pub max_stress: f64,
    /// Rate effect on the maximum stress (`ICC`).
    pub max_rate: MaxRateEffect,
    /// Rupture plastic strain (card default 1e30).
    pub rupture_strain: f64,
    /// Iso-kinematic split in `[0, 1]` (0 isotropic, 1 kinematic).
    pub chard: f64,
    /// Strain-rate filter cutoff in Hz (huge disables).
    pub fcut: f64,
    /// Optional Johnson-Cook damage accumulation model.
    pub damage: Option<JohnsonCookDamage>,
}

impl Law2 {
    /// Creates a LAW2 material from the full card-equivalent inputs.
    ///
    /// Thermal softening applies only when `melt_temperature` exceeds
    /// the reference temperature with `m > 0` (Reference Guide: `Tmelt =
    /// 0` means no temperature effect); adiabatic heating only when the
    /// volumetric heat capacity is positive.
    ///
    /// # Panics
    ///
    /// Panics when the hardening exponent exceeds 1 (card comment 6) or
    /// `chard` leaves `[0, 1]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "material cards take flat parameter lists"
    )]
    pub fn new(
        e: f64,
        nu: f64,
        flow: Law2Flow,
        reference_strain_rate: f64,
        m: f64,
        melt_temperature: f64,
        reference_temperature: f64,
        volumetric_heat_capacity: f64,
        max_stress: f64,
        max_rate: MaxRateEffect,
        rupture_strain: f64,
        chard: f64,
        fcut: f64,
    ) -> Self {
        let n = match flow {
            Law2Flow::JohnsonCook { n, .. } | Law2Flow::ZerilliArmstrong { n, .. } => n,
        };
        assert!(n <= 1.0, "LAW2 hardening exponent must be <= 1, got {n}");
        assert!(
            (0.0..=1.0).contains(&chard),
            "LAW2 chard must sit in [0, 1], got {chard}"
        );
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            flow,
            reference_strain_rate,
            m,
            melt_temperature,
            reference_temperature,
            volumetric_heat_capacity,
            max_stress,
            max_rate,
            rupture_strain,
            chard,
            fcut,
            damage: None,
        }
    }

    /// AISI 4340 steel after Johnson and Cook (1983).
    ///
    /// Card-exact defaults: reference temperature 298 K, caps open at
    /// 1e30, isotropic hardening, filtering off, adiabatic heating on
    /// with steel handbook heat data.
    pub fn c4340_steel() -> Self {
        Self::new(
            200.0e9,
            0.3,
            Law2Flow::JohnsonCook {
                a: 792.0e6,
                b: 510.0e6,
                n: 0.26,
                c: 0.014,
            },
            1.0,
            1.03,
            1793.0,
            298.0,
            7830.0 * 475.0,
            1.0e30,
            MaxRateEffect::Scaled,
            1.0e30,
            0.0,
            1.0e30,
        )
    }

    /// Whether thermal softening applies to this material.
    pub fn has_thermal(&self) -> bool {
        self.m > 0.0 && self.melt_temperature > self.reference_temperature
    }

    /// Homologous temperature in `[0, 1]`.
    pub fn homologous_temperature(&self, temperature: f64) -> f64 {
        if !self.has_thermal() {
            return 0.0;
        }
        let span = self.melt_temperature - self.reference_temperature;
        ((temperature - self.reference_temperature) / span).clamp(0.0, 1.0)
    }

    /// Thermal factor `1 - T*^m` (1 when thermal effects are off).
    pub fn thermal_factor(&self, temperature: f64) -> f64 {
        if !self.has_thermal() {
            return 1.0;
        }
        // Zerilli-Armstrong carries temperature inside its exponential, so
        // no extra thermal factor applies to that branch (Eq. 922).
        if matches!(self.flow, Law2Flow::ZerilliArmstrong { .. }) {
            return 1.0;
        }
        1.0 - self.homologous_temperature(temperature).powf(self.m)
    }

    /// Capped flow stress at plastic strain, rate, and temperature.
    ///
    /// Takes equivalent plastic strain, equivalent plastic strain rate
    /// (1/s), and temperature (K); returns Pa. The cap rates with
    /// `MaxRateEffect::Scaled` and stays fixed with `Fixed`.
    pub fn flow_stress(&self, ep: f64, rate: f64, temperature: f64) -> f64 {
        let raw = self
            .flow
            .flow_stress(ep, rate, temperature, self.reference_strain_rate)
            * self.thermal_factor(temperature);
        raw.min(self.rated_cap(rate))
    }

    /// Capped hardening slope at plastic strain, rate, temperature.
    ///
    /// Returns 0 past the cap (perfectly plastic tangent there); the rate
    /// factor stays frozen for the slope, matching the explicit update.
    pub fn hardening_slope(&self, ep: f64, rate: f64, temperature: f64) -> f64 {
        let raw = self
            .flow
            .flow_stress(ep, rate, temperature, self.reference_strain_rate)
            * self.thermal_factor(temperature);
        if raw >= self.rated_cap(rate) {
            return 0.0;
        }
        self.flow
            .hardening_slope(ep, rate, temperature, self.reference_strain_rate)
            * self.thermal_factor(temperature)
    }

    /// Maximum stress with the `ICC` rate treatment applied.
    pub fn rated_cap(&self, rate: f64) -> f64 {
        match self.max_rate {
            MaxRateEffect::Fixed => self.max_stress,
            MaxRateEffect::Scaled => {
                let factor = match self.flow {
                    Law2Flow::JohnsonCook { c, .. } => {
                        jc_rate_factor(rate, c, self.reference_strain_rate)
                    }
                    // The Zerilli-Armstrong card shares the ICC flag with
                    // the same log form on the cap (Figure 74 applies to
                    // both LAW2 branches).
                    Law2Flow::ZerilliArmstrong { c4, .. } => {
                        jc_rate_factor(rate, c4, self.reference_strain_rate)
                    }
                };
                self.max_stress * factor
            }
        }
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus in Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Accumulates Johnson-Cook damage for a committed step.
    ///
    /// Adds `dep / ef(triax, rate, T*)` to `state.damage` when the
    /// material carries a damage model; a no-op otherwise. Drivers call
    /// this after [`commit`](StressUpdateModel::commit) with the step
    /// strain rate in 1/s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::law_metals::{Law2, Law2State};
    /// use sonium_fem::physics::solid::plasticity::StressUpdateModel;
    /// use sonium_fem::physics::solid::fracture::ductile::JohnsonCookDamage;
    ///
    /// let mut steel = Law2::c4340_steel();
    /// steel.damage = Some(JohnsonCookDamage::c4340_steel());
    /// let mut state = Law2State::new(298.0);
    /// let update = steel.update(&state, &[0.01, -0.003, -0.003, 0.0, 0.0, 0.0]);
    /// steel.commit(&mut state, &update);
    /// steel.accumulate_damage(&mut state, &update, 1.0);
    /// assert!(state.damage > 0.0);
    /// ```
    pub fn accumulate_damage(
        &self,
        state: &mut Law2State,
        update: &StressUpdate,
        strain_rate: f64,
    ) {
        let Some(damage) = self.damage else {
            return;
        };
        if update.delta_equiv_plastic_strain <= 0.0 {
            return;
        }
        let p = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
        let q = von_mises_c1(&update.stress);
        let triax = p / q.max(1e-9 * p.abs().max(1.0));
        state.damage += damage.increment(
            triax,
            update.delta_equiv_plastic_strain,
            strain_rate,
            self.homologous_temperature(state.temperature),
        );
        state.damage = state.damage.clamp(0.0, 1.0);
    }

    /// Trial update at an explicit strain rate estimate.
    fn update_at_rate(&self, state: &Law2State, d_eps: &[f64; 6], rate: f64) -> StressUpdate {
        if state.ruptured {
            return ruptured_update_c1(self.bulk_modulus(), &state.stress, d_eps);
        }
        let temp = state.temperature;
        let sig0 = self.flow_stress(0.0, rate, temp);
        let (stress, tangent, dep, yielding, _) = if self.chard == 0.0 {
            let (s, t, d, y) = j2_return_c1(
                self.youngs_modulus,
                self.poisson_ratio,
                &state.stress,
                state.equiv_plastic_strain,
                d_eps,
                |ep| self.flow_stress(ep, rate, temp),
                |ep| self.hardening_slope(ep, rate, temp),
            );
            (s, t, d, y, state.back_stress)
        } else {
            j2_return_combined_c1(
                self.youngs_modulus,
                self.poisson_ratio,
                &state.stress,
                &state.back_stress,
                state.equiv_plastic_strain,
                self.chard,
                sig0,
                d_eps,
                |ep| self.flow_stress(ep, rate, temp),
                |ep| self.hardening_slope(ep, rate, temp),
            )
        };
        // Rupture zeroes the deviatoric trial the same step it trips.
        if state.equiv_plastic_strain + dep >= self.rupture_strain {
            let p = (stress[0] + stress[1] + stress[2]) / 3.0;
            return StressUpdate {
                stress: [p, p, p, 0.0, 0.0, 0.0],
                tangent: ruptured_tangent_c1(self.bulk_modulus()),
                delta_equiv_plastic_strain: dep,
                yielding: true,
            };
        }
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }
}

/// History state of one LAW2 material point.
#[derive(Debug, Clone)]
pub struct Law2State {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Back-stress in C1 Voigt order (Pa, deviatoric).
    pub back_stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Accumulated Johnson-Cook damage in `[0, 1]`.
    pub damage: f64,
    /// Whether the point has ruptured (deviatoric response gone).
    pub ruptured: bool,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl Law2State {
    /// Creates a virgin state at the given temperature in K.
    pub fn new(temperature: f64) -> Self {
        Self {
            stress: [0.0; 6],
            back_stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            temperature,
            damage: 0.0,
            ruptured: false,
            yielding: false,
        }
    }
}

impl StressUpdateModel for Law2 {
    type State = Law2State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_at_rate(state, d_eps, self.reference_strain_rate)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        // Explicit codes estimate the plastic rate from the total
        // increment; the elastic part of the error is negligible once
        // yielding. Callers smooth this rate with fcut between steps.
        let rate = voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
        self.update_at_rate(state, d_eps, rate)
    }

    fn is_rate_dependent(&self) -> bool {
        true
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
        if state.equiv_plastic_strain >= self.rupture_strain {
            state.ruptured = true;
        }
        // Textbook Prager back-stress from the exact rated hardening
        // increment (exact under proportional loading): the quasi-static
        // surface difference rated by the recovered step factor
        // telescopes along the path for any curve shape.
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && self.chard > 0.0 {
            let q = von_mises_c1(&update.stress);
            let ep_new = state.equiv_plastic_strain;
            let ep_old = ep_new - update.delta_equiv_plastic_strain;
            let temp = state.temperature;
            let quasi_new = self.flow_stress(ep_new, self.reference_strain_rate, temp);
            let ratio = commit_rate_ratio(q, quasi_new);
            let cap = match self.max_rate {
                MaxRateEffect::Fixed => self.max_stress,
                MaxRateEffect::Scaled => self.max_stress * ratio,
            };
            let step = kinematic_step_c1(
                self.chard,
                self.flow_stress(ep_old.max(0.0), self.reference_strain_rate, temp),
                quasi_new,
                ratio,
                cap,
                self.hardening_slope(ep_new, self.reference_strain_rate, temp)
                    .max(0.0)
                    * update.delta_equiv_plastic_strain,
            );
            let dir = if step > 0.0 {
                flow_direction_c1(&update.stress)
            } else {
                None
            };
            if let Some(dir) = dir {
                accumulate_c1(&mut state.back_stress, step, &dir);
            }
        }
        // Adiabatic heating converts the full plastic work (Eq. 210 has no
        // Taylor-Quinney split: T = Tr + Eint / (rho Cp V)).
        if update.yielding
            && update.delta_equiv_plastic_strain > 0.0
            && self.volumetric_heat_capacity > 0.0
        {
            let q = von_mises_c1(&update.stress);
            state.temperature +=
                q * update.delta_equiv_plastic_strain / self.volumetric_heat_capacity;
        }
    }
}

/// Johnson-Cook law with isotropic ductile damage (`/MAT/LAW22`, `DAMA`).
///
/// Runs a temperature-free [`Law2`] core (Reference Guide Eq. 219 has no
/// thermal factor) with damage-degraded moduli `E_dam = (1 - d) E`,
/// `nu_dam = d / 2 + (1 - d) nu` (Eq. 220) and the `(1 - d)`-scaled
/// surface (comment 4 applies damage to the deviatoric stress itself).
/// Below `damage_strain` the law is identical to LAW2. Damage
/// integrates so the uniaxial tangent follows the softening slope `Et`
/// (Reference Guide Figure 78 draws `Et` against total strain, so the
/// ep-space rate converts it with the undamaged elastic follower:
/// `d' = ((1 - d) H' - Et / (1 + |Et| / E)) / S`).
/// For solids the damage touches the deviatoric response only (bulk
/// modulus is exactly preserved: `K_dam = K`); at `d = 1` the deviatoric
/// stress vanishes while pressure still rides the undamaged bulk modulus.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law22;
///
/// let alu = Law22::card_aluminum();
/// assert_eq!(alu.damage_factor(0.05), 0.0);
/// assert!(alu.damage_factor(0.15) > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Law22 {
    /// Johnson-Cook core (thermal effects and `chard` unused).
    pub base: Law2,
    /// Plastic strain where damage starts (card default 0.15).
    pub damage_strain: f64,
    /// Softening slope in Pa (`-E < Et <= 0`, default 0).
    pub softening_slope: f64,
}

impl Law22 {
    /// Creates a LAW22 material from a core, damage strain, slope.
    ///
    /// Resets the core `chard`, thermal, heating, and JC-damage inputs:
    /// the LAW22 card carries none, so inherited values would silently
    /// change the response.
    ///
    /// # Panics
    ///
    /// Panics when the softening slope is positive (card: `-E < Et <=
    /// 0`) or the damage strain is negative.
    pub fn new(mut base: Law2, damage_strain: f64, softening_slope: f64) -> Self {
        assert!(
            softening_slope <= 0.0,
            "LAW22 softening slope must be <= 0, got {softening_slope}"
        );
        assert!(damage_strain >= 0.0, "LAW22 damage strain must be >= 0");
        base.chard = 0.0;
        base.m = 0.0;
        base.volumetric_heat_capacity = 0.0;
        base.damage = None;
        Self {
            base,
            damage_strain,
            softening_slope,
        }
    }

    /// Aluminum after the `/MAT/LAW22` card example (p.500).
    ///
    /// Perfectly plastic (`b = 0`) with `Eps_max = 0.2`, `SIGMA_max0 =
    /// 100 MPa`, damage from `Eps_dam = 0.1` with `Et = -2000 MPa`, no
    /// rate effects. Values transcribe the card example into SI.
    pub fn card_aluminum() -> Self {
        let base = Law2::new(
            70.0e9,
            0.3,
            Law2Flow::JohnsonCook {
                a: 100.0e6,
                b: 0.0,
                n: 1.0,
                c: 0.0,
            },
            1.0,
            0.0,
            0.0,
            298.0,
            0.0,
            100.0e6,
            MaxRateEffect::Scaled,
            0.2,
            0.0,
            1.0e30,
        );
        Self::new(base, 0.1, -2000.0e6)
    }

    /// Shear-modulus ratio `G_dam / G` at damage `d`.
    ///
    /// Follows from `E_dam = (1 - d) E` and `nu_dam = d / 2 + (1 - d)
    /// nu`; exactly 1 at `d = 0`, exactly 0 at `d = 1`.
    pub fn shear_ratio(&self, damage: f64) -> f64 {
        let nu = self.base.poisson_ratio;
        let nu_dam = 0.5 * damage + (1.0 - damage) * nu;
        (1.0 - damage) * (1.0 + nu) / (1.0 + nu_dam)
    }

    /// Softening slope converted to plastic-strain space (negative).
    ///
    /// Figure 78 draws `Et` against total strain; the damage rate needs
    /// it against plastic strain. Dividing by the undamaged elastic
    /// follower `(1 + |Et| / E)` reproduces tangent `Et` at damage
    /// onset and the engine's linear damage accumulation afterwards.
    fn ep_softening_slope(&self) -> f64 {
        self.softening_slope / (1.0 + (-self.softening_slope / self.base.youngs_modulus))
    }

    /// Closed-form damage at plastic strain (perfectly plastic core).
    ///
    /// Integrates `d' = -Et_ep / S` with constant strength `S`, giving
    /// the linear ramp used by the card-aluminum seed; hardening cores
    /// use the incremental form in [`trial_damage`](Self::trial_damage).
    /// Pure helper for tests and drivers, not the update path.
    pub fn damage_factor(&self, ep: f64) -> f64 {
        if ep <= self.damage_strain || self.softening_slope >= 0.0 {
            return 0.0;
        }
        let strength = self
            .base
            .flow_stress(
                ep,
                self.base.reference_strain_rate,
                self.base.reference_temperature,
            )
            .max(1e-9);
        (-self.ep_softening_slope() * (ep - self.damage_strain) / strength).clamp(0.0, 1.0)
    }

    /// Trial damage after a plastic increment (explicit Euler on `d`).
    ///
    /// Advances `d_old` with `d' = ((1 - d) H' - Et_ep) / S` evaluated
    /// at the end-of-step plastic strain with the quasi-static strength
    /// (the manual gives the damage law no rate dependence). Pure:
    /// `update` and `commit` share it deterministically.
    fn trial_damage(&self, ep_new: f64, dep: f64, d_old: f64) -> f64 {
        if ep_new <= self.damage_strain || dep <= 0.0 || self.softening_slope >= 0.0 {
            return d_old;
        }
        let rate = self.base.reference_strain_rate;
        let temp = self.base.reference_temperature;
        let strength = self.base.flow_stress(ep_new, rate, temp).max(1e-9);
        let slope = self.base.hardening_slope(ep_new, rate, temp);
        let rate_d = ((1.0 - d_old) * slope - self.ep_softening_slope()) / strength;
        (d_old + rate_d * dep).clamp(0.0, 1.0)
    }

    /// Johnson-Cook core with start-of-step damage applied (explicit).
    ///
    /// Degrades the moduli (Eq. 220) and scales the strength parameters
    /// (flow coefficients and cap) by `(1 - d)`: comment 4 applies the
    /// damage law to the deviatoric stress itself, so the surface keeps
    /// the full `(1 - d)` factor rather than the shear-modulus ratio.
    fn damaged_core(&self, damage: f64) -> Law2 {
        let mut core = self.base;
        core.youngs_modulus = (1.0 - damage) * self.base.youngs_modulus;
        core.poisson_ratio = 0.5 * damage + (1.0 - damage) * self.base.poisson_ratio;
        let keep = 1.0 - damage;
        core.flow = match self.base.flow {
            Law2Flow::JohnsonCook { a, b, n, c } => Law2Flow::JohnsonCook {
                a: keep * a,
                b: keep * b,
                n,
                c,
            },
            Law2Flow::ZerilliArmstrong {
                c0,
                c1,
                c3,
                c4,
                c5,
                n,
            } => Law2Flow::ZerilliArmstrong {
                c0: keep * c0,
                c1: keep * c1,
                c3,
                c4,
                c5: keep * c5,
                n,
            },
        };
        core.max_stress = keep * self.base.max_stress;
        core
    }

    /// Shared trial path with optional step time for the rate.
    fn update_inner(&self, state: &Law22State, d_eps: &[f64; 6], dt: Option<f64>) -> StressUpdate {
        if state.damage >= 1.0 - 1e-12 {
            // Fully damaged: zero strength, pressure on the undamaged
            // bulk modulus (bulk is damage-invariant), plastic strain
            // frozen. The engine skips the plastic update at zero
            // strength (harvested PLAS pins once damage saturates), so
            // the rupture strain cannot trip past full damage either.
            let p_old =
                (state.inner.stress[0] + state.inner.stress[1] + state.inner.stress[2]) / 3.0;
            let bulk = self.base.bulk_modulus();
            let p_new = p_old + bulk * (d_eps[0] + d_eps[1] + d_eps[2]);
            return StressUpdate {
                stress: [p_new, p_new, p_new, 0.0, 0.0, 0.0],
                tangent: ruptured_tangent_c1(bulk),
                delta_equiv_plastic_strain: 0.0,
                yielding: true,
            };
        }
        let core = self.damaged_core(state.damage);
        match dt {
            Some(dt) => core.update_with_rate(&state.inner, d_eps, dt),
            None => core.update(&state.inner, d_eps),
        }
    }
}

/// History state of one LAW22 material point.
#[derive(Debug, Clone)]
pub struct Law22State {
    /// Johnson-Cook core state.
    pub inner: Law2State,
    /// Isotropic ductile damage in `[0, 1]`.
    pub damage: f64,
}

impl Law22State {
    /// Creates a virgin state at the given temperature in K.
    pub fn new(temperature: f64) -> Self {
        Self {
            inner: Law2State::new(temperature),
            damage: 0.0,
        }
    }
}

impl StressUpdateModel for Law22 {
    type State = Law22State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_inner(state, d_eps, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.update_inner(state, d_eps, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        true
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        // The trial already ran on damaged moduli, so the core commits
        // directly (its chard/heating stay off by construction); then the
        // damage variable advances deterministically from the step.
        self.base.commit(&mut state.inner, update);
        let ep_new = state.inner.equiv_plastic_strain;
        state.damage = self.trial_damage(ep_new, update.delta_equiv_plastic_strain, state.damage);
    }
}

/// Full 3x3 matrix from a C1 tensor.
fn full_from_c1(t: &[f64; 6]) -> [[f64; 3]; 3] {
    [[t[0], t[3], t[5]], [t[3], t[1], t[4]], [t[5], t[4], t[2]]]
}

/// C1 tensor from a full 3x3 matrix (symmetric part).
fn c1_from_full(m: &[[f64; 3]; 3]) -> [f64; 6] {
    [
        m[0][0],
        m[1][1],
        m[2][2],
        0.5 * (m[0][1] + m[1][0]),
        0.5 * (m[1][2] + m[2][1]),
        0.5 * (m[0][2] + m[2][0]),
    ]
}

/// Elastic strain of a C1 stress with bulk and shear moduli.
fn elastic_strain_c1(stress: &[f64; 6], k: f64, g: f64) -> [f64; 6] {
    let p = (stress[0] + stress[1] + stress[2]) / 3.0;
    std::array::from_fn(|i| {
        let dev = if i < 3 { stress[i] - p } else { stress[i] };
        dev / (2.0 * g) + if i < 3 { p / (3.0 * k) } else { 0.0 }
    })
}

/// Unit flow direction of a C1 stress (deviatoric, `None` if tiny).
///
/// Returns `dev / q` (unit in the von Mises norm): the backstress
/// direction. Plastic strain advances along [`plastic_flow_c1`] instead
/// (unit in the equivalent-strain norm); the two differ by 3/2.
fn flow_direction_c1(stress: &[f64; 6]) -> Option<[f64; 6]> {
    let q = von_mises_c1(stress);
    // Scale-aware floor: q below 1e-12 of a 1 Pa reference carries no
    // meaningful direction (ruptured or virgin states).
    if q <= 1e-12 {
        return None;
    }
    let p = (stress[0] + stress[1] + stress[2]) / 3.0;
    Some(std::array::from_fn(|i| {
        let dev = if i < 3 { stress[i] - p } else { stress[i] };
        // Radial-return direction shares the stress-deviatoric line, so
        // scaling by 1/q matches the trial direction exactly under
        // proportional loading.
        dev / q
    }))
}

/// Associative plastic-flow direction of a C1 stress (`None` if tiny).
///
/// Unit-equivalent direction `(3/2) dev / q`: advancing the plastic
/// tensor with `dep` along it dissipates exactly `q dep` per step and
/// reproduces the uniaxial plastic tensor `[1, -1/2, -1/2] dep`.
fn plastic_flow_c1(stress: &[f64; 6]) -> Option<[f64; 6]> {
    flow_direction_c1(stress).map(|dir| dir.map(|v| 1.5 * v))
}

/// Commit-side rate ratio recovered from the returned stress level.
///
/// Rate-multiplicative laws sit on the surface as `q ~= sy_quasi * R`,
/// so the ratio recovers the step rate factor the commit otherwise
/// cannot see (exact away from caps; Fixed-cap corners and post-scaled
/// stresses lag one step, noted on the laws that scale trials).
fn commit_rate_ratio(q_new: f64, quasi_yield: f64) -> f64 {
    if quasi_yield <= 1e-30 {
        1.0
    } else {
        (q_new / quasi_yield).clamp(1e-3, 1e3)
    }
}

/// Kinematic hardening increment for one committed step (exact).
///
/// Textbook Prager backstress advances with the hardening increment
/// `d alpha = chard * d(rated flow) * dir`. Rating the quasi-static
/// surface difference by the recovered step factor keeps the
/// accumulation exact for any curve shape: an end-point slope times
/// `dep` instead undercounts concave hardening (up to half the first
/// step for square-root laws, whose slope diverges at zero plastic
/// strain). `cap` clamps the rated increment at the estimated cap;
/// `fallback` (slope times `dep`) covers degenerate non-positive
/// quasi-static surfaces.
fn kinematic_step_c1(
    chard: f64,
    quasi_old: f64,
    quasi_new: f64,
    ratio: f64,
    cap: f64,
    fallback: f64,
) -> f64 {
    if quasi_new <= 0.0 {
        return chard * fallback.max(0.0);
    }
    let rated_new = (quasi_new * ratio).min(cap);
    let rated_old = (quasi_old.max(0.0) * ratio).min(cap);
    chard * (rated_new - rated_old).max(0.0)
}

/// Textbook equivalent of a C1 strain increment (booking use).
///
/// Full double contraction `sqrt((2/3) e:e)` with engineering shear;
/// dep booking requires the physical value (frozen rate estimates use a
/// lighter shear weight, which is fine for rate factors but not for
/// accumulated plastic strain).
fn equiv_strain_c1(d_eps: &[f64; 6]) -> f64 {
    let mean = (d_eps[0] + d_eps[1] + d_eps[2]) / 3.0;
    let d0 = d_eps[0] - mean;
    let d1 = d_eps[1] - mean;
    let d2 = d_eps[2] - mean;
    let shear = 0.5 * (d_eps[3] * d_eps[3] + d_eps[4] * d_eps[4] + d_eps[5] * d_eps[5]);
    ((2.0 / 3.0) * (d0 * d0 + d1 * d1 + d2 * d2 + shear)).sqrt()
}

/// Equivalent of a deviatoric C1 tensor with tensor shear slots.
///
/// Plastic-accumulation tensors store tensor shear (flow directions come
/// from stress deviators), so the contraction doubles the shear slots.
fn equiv_deviatoric_c1(t: &[f64; 6]) -> f64 {
    let shear = 2.0 * (t[3] * t[3] + t[4] * t[4] + t[5] * t[5]);
    ((2.0 / 3.0) * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + shear)).sqrt()
}

/// Plastic-flow direction with an accumulated-flow fallback.
///
/// Associative [`plastic_flow_c1`] while stress is nonzero; at zero
/// stress (fully softened or just-ruptured response) the flow continues
/// along the accumulated plastic direction (unit equivalent), the
/// standard zero-strength idealization. Without the fallback the
/// plastic tensor would freeze at zero stress and total-strain rupture
/// triggers would pin one step below their threshold forever. Returns
/// `None` only for a virgin point, where no direction exists yet.
fn plastic_flow_or_accumulated_c1(stress: &[f64; 6], plastic: &[f64; 6]) -> Option<[f64; 6]> {
    if let Some(dir) = plastic_flow_c1(stress) {
        return Some(dir);
    }
    let eq = equiv_deviatoric_c1(plastic);
    if eq <= 1e-30 {
        return None;
    }
    Some(std::array::from_fn(|i| plastic[i] / eq))
}

/// Accumulates a scaled direction into a C1 tensor.
fn accumulate_c1(tensor: &mut [f64; 6], scale: f64, dir: &[f64; 6]) {
    for (slot, d) in tensor.iter_mut().zip(dir.iter()) {
        *slot += scale * d;
    }
}

/// Trial total strain: elastic part plus plastic accumulation.
///
/// Adds the accumulated tensor and the step flow along the trial
/// direction (no direction at zero stress: plastic alone). Pure: all
/// inputs come from the state, the increment response, and the return.
fn trial_total_c1(
    plastic: &[f64; 6],
    stress: &[f64; 6],
    dep: f64,
    bulk: f64,
    shear: f64,
) -> [f64; 6] {
    let elastic = elastic_strain_c1(stress, bulk, shear);
    let dir = plastic_flow_c1(stress);
    std::array::from_fn(|i| elastic[i] + plastic[i] + dep * dir.map(|d| d[i]).unwrap_or(0.0))
}

/// Brittle damage parameters of one principal direction (LAW27).
///
/// Linear damage `d = (e - et) / (em - et)` between the onset strain
/// `eps_t` and the full-damage strain `eps_m` (Theory Manual Eq. 963),
/// capped at `dmax`, with element deletion at `eps_f` (card: shell
/// deletion; the material point raises `ruptured` instead).
#[derive(Debug, Clone, Copy)]
pub struct BrittleDir {
    /// Tensile onset strain (damage starts above it).
    pub eps_t: f64,
    /// Full-damage strain (damage caps at `dmax` there).
    pub eps_m: f64,
    /// Maximum damage factor in `[0, 1]` (card default 0.999).
    pub dmax: f64,
    /// Deletion strain (material point ruptures above it).
    pub eps_f: f64,
}

impl BrittleDir {
    /// Creates one brittle direction, checking the strain ordering.
    ///
    /// # Panics
    ///
    /// Panics when `eps_m <= eps_t`, `dmax` leaves `[0, 1]`, or `eps_f <
    /// eps_m`.
    pub fn new(eps_t: f64, eps_m: f64, dmax: f64, eps_f: f64) -> Self {
        assert!(eps_m > eps_t, "LAW27 needs eps_m > eps_t");
        assert!((0.0..=1.0).contains(&dmax), "LAW27 dmax must sit in [0, 1]");
        assert!(eps_f >= eps_m, "LAW27 needs eps_f >= eps_m");
        Self {
            eps_t,
            eps_m,
            dmax,
            eps_f,
        }
    }

    /// Damage at a principal strain (tension only, 0 in compression).
    pub fn damage_at(&self, principal: f64) -> f64 {
        if principal <= self.eps_t {
            0.0
        } else if principal >= self.eps_m {
            self.dmax
        } else {
            ((principal - self.eps_t) / (self.eps_m - self.eps_t)).min(self.dmax)
        }
    }
}

/// Johnson-Cook law with orthotropic brittle damage (`/MAT/LAW27`).
///
/// Runs the [`Law2`] core (isotropic, no thermal part: the card carries
/// neither `Chard` nor temperature inputs) and degrades the stress in the
/// two largest principal-strain directions with [`BrittleDir`] damage
/// (Theory Manual Eq. 957-965). In the strain-principal basis each stress
/// component scales with `sqrt((1 - d_i)(1 - d_j))`, the symmetric 3-D
/// form of the shell law; damage and rupture trigger on tension only.
/// Deletion at `eps_f` raises `ruptured` and zeroes the full stress (the
/// brittle point carries no pressure after deletion). Damaged steps
/// book their full equivalent increment as plastic flow (zero-strength
/// idealization), keeping total-strain triggers exact on monotonic paths.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law27;
///
/// let glass = Law27::steel_brittle();
/// assert_eq!(glass.dir1.damage_at(0.01), 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Law27 {
    /// Johnson-Cook core (`chard`, thermal, damage inputs reset).
    pub base: Law2,
    /// Brittle parameters of the largest principal direction.
    pub dir1: BrittleDir,
    /// Brittle parameters of the middle principal direction.
    pub dir2: BrittleDir,
}

impl Law27 {
    /// Creates a LAW27 material from a core and two directions.
    ///
    /// Resets the core `chard`, thermal exponent, and JC damage inputs:
    /// the LAW27 card carries none, so inherited values would silently
    /// change the response.
    pub fn new(mut base: Law2, dir1: BrittleDir, dir2: BrittleDir) -> Self {
        base.chard = 0.0;
        base.m = 0.0;
        base.damage = None;
        Self { base, dir1, dir2 }
    }

    /// Illustrative brittle steel (JC base plus symmetric damage dirs).
    ///
    /// Uses a 250 MPa yield core with onset at 8%, full damage at 15%,
    /// and deletion at 16% in both directions. Parameters are
    /// illustrative: the card ships no numeric example.
    pub fn steel_brittle() -> Self {
        let base = Law2::new(
            200.0e9,
            0.3,
            Law2Flow::JohnsonCook {
                a: 250.0e6,
                b: 500.0e6,
                n: 0.5,
                c: 0.0,
            },
            1.0,
            0.0,
            0.0,
            298.0,
            0.0,
            1.0e30,
            MaxRateEffect::Scaled,
            1.0e30,
            0.0,
            1.0e30,
        );
        let dir = BrittleDir::new(0.08, 0.15, 0.999, 0.16);
        Self::new(base, dir, dir)
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus(&self) -> f64 {
        self.base.shear_modulus()
    }

    /// Bulk modulus in Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.base.bulk_modulus()
    }

    /// Directional damages and sorted principal basis of total strain.
    ///
    /// Returns `(d1, d2, values, vectors)` with values largest first;
    /// both damages vanish unless the leading principal strain is
    /// tensile (card: damage occurs only in tension).
    fn damage_system(&self, total: &[f64; 6]) -> (f64, f64, [f64; 3], [[f64; 3]; 3]) {
        let (values, vecs) = principal_strain_c1(total);
        let mut order = [0, 1, 2];
        order.sort_by(|x, y| values[*y].total_cmp(&values[*x]));
        let sorted = [values[order[0]], values[order[1]], values[order[2]]];
        let mut vec_sorted = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                vec_sorted[i][k] = vecs[i][order[k]];
            }
        }
        let d1 = self.dir1.damage_at(sorted[0]);
        let d2 = self.dir2.damage_at(sorted[1]);
        (d1, d2, sorted, vec_sorted)
    }

    /// Applies directional damage to a trial stress tensor.
    fn apply_brittle(&self, stress: &[f64; 6], d1: f64, d2: f64, vecs: &[[f64; 3]; 3]) -> [f64; 6] {
        if d1 <= 0.0 && d2 <= 0.0 {
            return *stress;
        }
        // Rotate into the strain-principal basis, scale symmetrically,
        // rotate back. The third direction stays intact (d3 = 0): the
        // card defines damage for two in-plane directions only.
        let keep = [1.0 - d1, 1.0 - d2, 1.0];
        let full = full_from_c1(stress);
        let mut hat = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut entry = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        entry += vecs[a][i] * full[a][b] * vecs[b][j];
                    }
                }
                hat[i][j] = (keep[i] * keep[j]).sqrt() * entry;
            }
        }
        let mut back = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut entry = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        entry += vecs[a][i] * hat[i][j] * vecs[b][j];
                    }
                }
                back[a][b] = entry;
            }
        }
        c1_from_full(&back)
    }
}

/// History state of one LAW27 material point.
#[derive(Debug, Clone)]
pub struct Law27State {
    /// Johnson-Cook core state.
    pub inner: Law2State,
    /// Accumulated plastic strain tensor (C1, for total-strain rebuild).
    pub plastic: [f64; 6],
    /// Directional damages `[d1, d2]`.
    pub damage: [f64; 2],
    /// Whether brittle deletion tripped (full stress zeroed).
    pub ruptured: bool,
}

impl Law27State {
    /// Creates a virgin state at the given temperature in K.
    pub fn new(temperature: f64) -> Self {
        Self {
            inner: Law2State::new(temperature),
            plastic: [0.0; 6],
            damage: [0.0; 2],
            ruptured: false,
        }
    }

    /// Rebuilds total strain as elastic plus accumulated plastic parts.
    fn total_strain(&self, law: &Law27) -> [f64; 6] {
        let elastic =
            elastic_strain_c1(&self.inner.stress, law.bulk_modulus(), law.shear_modulus());
        std::array::from_fn(|i| elastic[i] + self.plastic[i])
    }
}

impl StressUpdateModel for Law27 {
    type State = Law27State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_inner(state, d_eps, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.update_inner(state, d_eps, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        true
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.base.commit(&mut state.inner, update);
        // Advance the plastic tensor along the associative direction
        // (exact under proportional loading), falling back to the
        // accumulated direction at zero stress so the rupture trigger
        // cannot pin one step below its threshold, then refresh damage
        // and the rupture latch from committed totals.
        if let Some(dir) = plastic_flow_or_accumulated_c1(&update.stress, &state.plastic) {
            accumulate_c1(&mut state.plastic, update.delta_equiv_plastic_strain, &dir);
        }
        let total = state.total_strain(self);
        let (d1, d2, values, _) = self.damage_system(&total);
        state.damage = [d1, d2];
        if values[0] >= self.dir1.eps_f || values[1] >= self.dir2.eps_f {
            state.ruptured = true;
        }
    }
}

impl Law27 {
    /// Shared trial path with optional step time for the rate.
    fn update_inner(&self, state: &Law27State, d_eps: &[f64; 6], dt: Option<f64>) -> StressUpdate {
        if state.ruptured || state.inner.ruptured {
            // Brittle deletion zeroes the full stress (no pressure past
            // deletion); a tripped JC core alone keeps the solid rule.
            if state.ruptured {
                return StressUpdate {
                    stress: [0.0; 6],
                    tangent: [0.0; 36],
                    delta_equiv_plastic_strain: 0.0,
                    yielding: true,
                };
            }
            return ruptured_update_c1(self.bulk_modulus(), &state.inner.stress, d_eps);
        }
        let mut update = match dt {
            Some(dt) => self.base.update_with_rate(&state.inner, d_eps, dt),
            None => self.base.update(&state.inner, d_eps),
        };
        // Trial total strain: elastic part of the trial plus plastic
        // accumulation along the trial direction.
        let total = trial_total_c1(
            &state.plastic,
            &update.stress,
            update.delta_equiv_plastic_strain,
            self.bulk_modulus(),
            self.shear_modulus(),
        );
        let (mut d1, mut d2, mut values, mut vecs) = self.damage_system(&total);
        if d1 > 0.0 || d2 > 0.0 {
            // Zero-strength-flow booking: once brittle damage opens, the
            // secant response hides strain from the core elastic/plastic
            // split (the core trial starts from scaled stress), so the
            // step books its full equivalent increment as plastic flow.
            // Keeps the total-strain rebuild (and the deletion trigger)
            // exact along monotonic paths up to a bounded softening lag
            // (order surface/E absolute); undamaged steps keep the
            // physical core increment.
            update.delta_equiv_plastic_strain = equiv_strain_c1(d_eps);
            update.yielding = true;
            let re_total = trial_total_c1(
                &state.plastic,
                &update.stress,
                update.delta_equiv_plastic_strain,
                self.bulk_modulus(),
                self.shear_modulus(),
            );
            (d1, d2, values, vecs) = self.damage_system(&re_total);
        }
        if values[0] >= self.dir1.eps_f || values[1] >= self.dir2.eps_f {
            return StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: update.delta_equiv_plastic_strain,
                yielding: true,
            };
        }
        update.stress = self.apply_brittle(&update.stress, d1, d2, &vecs);
        // Tangent approximation: mean directional keep factor (exact when
        // undamaged; explicit dynamics never consumes this tangent).
        let keep = ((1.0 - d1) + (1.0 - d2) + 1.0) / 3.0;
        if keep < 1.0 {
            for entry in update.tangent.iter_mut() {
                *entry *= keep;
            }
        }
        update
    }
}

/// Rebuilds total strain from stress plus accumulated plastic tensor.
fn total_strain_c1(stress: &[f64; 6], plastic: &[f64; 6], k: f64, g: f64) -> [f64; 6] {
    let elastic = elastic_strain_c1(stress, k, g);
    std::array::from_fn(|i| elastic[i] + plastic[i])
}

/// One tabulated hardening curve at a fixed strain rate (LAW36).
///
/// Holds the frozen [`Funct1D`] reader plus the last abscissa (the reader
/// exposes no grid, so the manual-required linear extrapolation past the
/// last knot is applied here from the frozen end slope).
#[derive(Debug, Clone)]
pub struct RateCurve {
    /// Strain rate of this curve in 1/s.
    pub rate: f64,
    /// Yield stress versus plastic strain (Pa over -).
    pub curve: Funct1D,
    /// Ordinate scale factor (card `Fscalei`).
    pub scale: f64,
    /// Last strain knot (extrapolation starts above it).
    pub xmax: f64,
}

impl RateCurve {
    /// Creates one rate curve from raw knots and a scale factor.
    ///
    /// # Errors
    ///
    /// Returns the [`Funct1D`] validation message for bad knots.
    pub fn new(rate: f64, x: Vec<f64>, y: Vec<f64>, scale: f64) -> Result<Self, String> {
        let xmax = *x.last().unwrap_or(&0.0);
        Ok(Self {
            rate,
            curve: Funct1D::new(x, y)?,
            scale,
            xmax,
        })
    }

    /// Yield value at plastic strain with end-slope extrapolation.
    ///
    /// Evaluates the frozen reader inside the tabulated range and extends
    /// the last segment linearly beyond it (Theory Manual p.334: curves
    /// are extrapolated past the maximum plastic strain).
    pub fn yield_at(&self, ep: f64) -> f64 {
        let base = self.curve.eval(ep);
        if ep > self.xmax {
            base + (ep - self.xmax) * self.curve.slope(self.xmax)
        } else {
            base
        }
    }

    /// Hardening slope at plastic strain (end slope past the grid).
    pub fn slope_at(&self, ep: f64) -> f64 {
        self.curve.slope(ep.min(self.xmax).max(0.0))
    }
}

/// Tabulated piecewise-linear metal law (`/MAT/LAW36`, `PLAS_TAB`).
///
/// Yield stress interpolates linearly in plastic strain along one curve
/// per strain rate, and linearly between curves in rate (Theory Manual
/// p.334); past the last rate the yield extrapolates linearly (card
/// comment 6), and past the last strain knot each curve extends with its
/// end slope. There is no maximum stress. Options cover combined
/// iso-kinematic hardening (`chard`), the Zhao-type tensile limits
/// (`eps_t1`, `eps_t2`), rupture at `eps_max`, deletion at `eps_del`, a
/// pressure-factor function, and a strain-dependent Young's modulus
/// scale. The strain rate is the largest-absolute principal value (the
/// 3-D analog of Eq. 928; the manual states the maximum, and the absolute
/// form keeps compression-dominated steps rated).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law36;
///
/// let alu = Law36::card_aluminum();
/// let y0 = alu.nominal_yield(0.0, 0.0);
/// assert!(y0 > 80.0e6 && y0 < 100.0e6);
/// ```
#[derive(Debug, Clone)]
pub struct Law36 {
    /// Young's modulus in Pa (scaled by `e_scale` when set).
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Hardening curves with strictly ascending rates.
    pub curves: Vec<RateCurve>,
    /// Iso-kinematic split in `[0, 1]` (0 isotropic, 1 kinematic).
    pub chard: f64,
    /// Rupture plastic strain (card default 1e20).
    pub rupture_strain: f64,
    /// Tensile strain where stress starts reducing (default 1e20).
    pub tensile_t1: f64,
    /// Tensile strain where stress reaches zero (default 2e20).
    pub tensile_t2: f64,
    /// Tensile strain for element deletion (default 3e20).
    pub deletion_strain: f64,
    /// Yield factor versus pressure (`fct_IDp` times `Fscale`).
    pub pressure_factor: Option<(Funct1D, f64)>,
    /// Young's modulus scale versus plastic strain (`fct_IDE`).
    pub e_scale: Option<Funct1D>,
    /// Rate source: plastic (`VP = 1`) or total (`VP = 0`, default).
    pub plastic_rate: bool,
}

impl Law36 {
    /// Creates a LAW36 material from moduli, curves, and limits.
    ///
    /// # Panics
    ///
    /// Panics when no curve is given, rates are not strictly ascending
    /// (card comment 12), or `chard` leaves `[0, 1]`.
    pub fn new(
        e: f64,
        nu: f64,
        curves: Vec<RateCurve>,
        chard: f64,
        rupture_strain: f64,
        tensile_t1: f64,
        tensile_t2: f64,
        deletion_strain: f64,
    ) -> Self {
        assert!(!curves.is_empty(), "LAW36 needs at least one curve");
        for pair in curves.windows(2) {
            assert!(
                pair[1].rate > pair[0].rate,
                "LAW36 rates must strictly ascend"
            );
        }
        assert!(
            (0.0..=1.0).contains(&chard),
            "LAW36 chard must sit in [0, 1], got {chard}"
        );
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            curves,
            chard,
            rupture_strain,
            tensile_t1,
            tensile_t2,
            deletion_strain,
            pressure_factor: None,
            e_scale: None,
            plastic_rate: false,
        }
    }

    /// Aluminum after the `/MAT/LAW36` card example (p.523).
    ///
    /// Single tabulated curve (90 to 170 MPa over 7.5% strain, `E = 60.4
    /// GPa`, `nu = 0.33`) with all limits open. Values transcribe the
    /// card example into SI.
    pub fn card_aluminum() -> Self {
        let x = vec![
            0.0, 2.5e-4, 0.001, 0.009, 0.017, 0.021, 0.036, 0.045, 0.055, 0.072, 0.075,
        ];
        let y_mpa = vec![
            90.0, 100.0, 104.0, 121.0, 136.0, 143.0, 156.0, 162.0, 165.0, 170.0, 170.0,
        ];
        let y: Vec<f64> = y_mpa.iter().map(|v| v * 1.0e6).collect();
        let curve = RateCurve::new(0.0, x, y, 1.0).expect("card knots must validate");
        Self::new(
            60.4e9,
            0.33,
            vec![curve],
            0.0,
            1.0e20,
            1.0e20,
            2.0e20,
            3.0e20,
        )
    }

    /// Current Young's modulus at plastic strain (explicit start value).
    pub fn current_modulus(&self, ep: f64) -> f64 {
        match &self.e_scale {
            Some(f) => self.youngs_modulus * f.eval(ep.max(0.0)),
            None => self.youngs_modulus,
        }
    }

    /// Shear modulus in Pa at plastic strain.
    pub fn shear_modulus(&self, ep: f64) -> f64 {
        self.current_modulus(ep) / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus in Pa at plastic strain.
    pub fn bulk_modulus(&self, ep: f64) -> f64 {
        self.current_modulus(ep) / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Rate-bracket indices and weight (weight may exceed 1 past last).
    fn bracket(&self, rate: f64) -> (usize, usize, f64) {
        let n = self.curves.len();
        if n == 1 || rate <= self.curves[0].rate {
            return (0, 0, 0.0);
        }
        if rate >= self.curves[n - 1].rate {
            // Linear extrapolation in rate between the last two curves
            // (card comment 6); the caller is warned it can run away.
            let span = self.curves[n - 1].rate - self.curves[n - 2].rate;
            return (n - 2, n - 1, (rate - self.curves[n - 2].rate) / span);
        }
        for pair in 0..n - 1 {
            if rate <= self.curves[pair + 1].rate {
                let span = self.curves[pair + 1].rate - self.curves[pair].rate;
                return (pair, pair + 1, (rate - self.curves[pair].rate) / span);
            }
        }
        (n - 2, n - 1, 1.0)
    }

    /// Nominal (pressure-free) yield at plastic strain and rate.
    pub fn nominal_yield(&self, ep: f64, rate: f64) -> f64 {
        let (lo, hi, w) = self.bracket(rate);
        let y_lo = self.curves[lo].yield_at(ep) * self.curves[lo].scale;
        let y_hi = self.curves[hi].yield_at(ep) * self.curves[hi].scale;
        y_lo + w * (y_hi - y_lo)
    }

    /// Nominal hardening slope at plastic strain and rate.
    pub fn nominal_slope(&self, ep: f64, rate: f64) -> f64 {
        let (lo, hi, w) = self.bracket(rate);
        let s_lo = self.curves[lo].slope_at(ep) * self.curves[lo].scale;
        let s_hi = self.curves[hi].slope_at(ep) * self.curves[hi].scale;
        s_lo + w * (s_hi - s_lo)
    }

    /// Pressure factor at Radioss pressure (positive in compression).
    pub fn pressure_scale(&self, pressure: f64) -> f64 {
        match &self.pressure_factor {
            Some((f, scale)) => f.eval(pressure) * scale,
            None => 1.0,
        }
    }

    /// Effective yield at strain, rate, and trial pressure.
    pub fn flow_stress(&self, ep: f64, rate: f64, pressure: f64) -> f64 {
        self.nominal_yield(ep, rate) * self.pressure_scale(pressure)
    }

    /// Effective hardening slope (pressure held fixed for the slope).
    pub fn hardening_slope(&self, ep: f64, rate: f64, pressure: f64) -> f64 {
        self.nominal_slope(ep, rate) * self.pressure_scale(pressure)
    }
}

/// History state of one LAW36 material point.
#[derive(Debug, Clone)]
pub struct Law36State {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Back-stress in C1 Voigt order (Pa, deviatoric).
    pub back_stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Accumulated plastic strain tensor (C1, for total-strain rebuild).
    pub plastic: [f64; 6],
    /// Whether rupture tripped (deviatoric response gone).
    pub ruptured: bool,
    /// Whether deletion tripped (full stress zeroed).
    pub deleted: bool,
    /// Start-of-step secant tensile scale (explicit Zhao reduction).
    pub tensile_scale: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl Law36State {
    /// Creates a virgin state.
    pub fn new() -> Self {
        Self {
            stress: [0.0; 6],
            back_stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            plastic: [0.0; 6],
            ruptured: false,
            deleted: false,
            tensile_scale: 1.0,
            yielding: false,
        }
    }
}

impl Default for Law36State {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for Law36 {
    type State = Law36State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_inner(state, d_eps, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.update_inner(state, d_eps, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        self.curves.len() > 1
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
        // Accumulated-flow fallback at zero stiffness (see Law27).
        if let Some(dir) = plastic_flow_or_accumulated_c1(&update.stress, &state.plastic) {
            accumulate_c1(&mut state.plastic, update.delta_equiv_plastic_strain, &dir);
        }
        // Secant rebuild with the start-of-step scale (explicit scheme):
        // the plastic tensor already absorbed the step flow, and the
        // elastic leg unloads on the softened moduli.
        let e_sec = state.tensile_scale * self.current_modulus(state.equiv_plastic_strain);
        let g = e_sec / (2.0 * (1.0 + self.poisson_ratio));
        let k = e_sec / (3.0 * (1.0 - 2.0 * self.poisson_ratio));
        let pressure = -(state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        // Textbook Prager back-stress from the exact rated hardening
        // increment (exact: the secant scale rides along in the stress
        // ratio; tabulated curves have no cap).
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && self.chard > 0.0 {
            let q = von_mises_c1(&update.stress);
            let ep_new = state.equiv_plastic_strain;
            let ep_old = ep_new - update.delta_equiv_plastic_strain;
            let quasi_new = self.flow_stress(ep_new, 0.0, pressure);
            let ratio = commit_rate_ratio(q, quasi_new);
            let step = kinematic_step_c1(
                self.chard,
                self.flow_stress(ep_old.max(0.0), 0.0, pressure),
                quasi_new,
                ratio,
                f64::INFINITY,
                self.hardening_slope(ep_new, 0.0, pressure).max(0.0)
                    * update.delta_equiv_plastic_strain,
            );
            let dir = if step > 0.0 {
                flow_direction_c1(&update.stress)
            } else {
                None
            };
            if let Some(dir) = dir {
                accumulate_c1(&mut state.back_stress, step, &dir);
            }
        }
        let total = if state.tensile_scale <= 1e-12 {
            // Zero stiffness: plastic tensor plus the elastic-limit leg
            // (quasi-static surface over base modulus along accumulated
            // flow), continuous with the secant rebuild as s -> 0. The
            // volumetric leg stays undefined at zero strength.
            let mut tot = state.plastic;
            let eq = equiv_deviatoric_c1(&state.plastic);
            if eq > 1e-30 {
                let sweep = (self.flow_stress(state.equiv_plastic_strain, 0.0, pressure)
                    / self.current_modulus(state.equiv_plastic_strain))
                .max(0.0);
                accumulate_c1(&mut tot, sweep / eq, &state.plastic);
            }
            tot
        } else {
            total_strain_c1(&state.stress, &state.plastic, k, g)
        };
        let first = max_principal_strain_c1(&total);
        state.tensile_scale = tensile_scale(first, self.tensile_t1, self.tensile_t2);
        if first >= self.deletion_strain {
            state.deleted = true;
        }
        if state.equiv_plastic_strain >= self.rupture_strain {
            state.ruptured = true;
        }
    }
}

impl Law36 {
    /// Shared trial path with optional step time for the rate.
    fn update_inner(&self, state: &Law36State, d_eps: &[f64; 6], dt: Option<f64>) -> StressUpdate {
        if state.deleted {
            return StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: 0.0,
                yielding: true,
            };
        }
        // Secant tensile scale from the start of the step (explicit): the
        // trial runs on softened moduli with the scaled surface, so the
        // Zhao-type reduction emerges from the return itself and the
        // plastic bookkeeping stays exact.
        let s_old = state.tensile_scale;
        let e = s_old * self.current_modulus(state.equiv_plastic_strain);
        let g = e / (2.0 * (1.0 + self.poisson_ratio));
        let k = e / (3.0 * (1.0 - 2.0 * self.poisson_ratio));
        if state.ruptured {
            // Ruptured pressure rides the scaled bulk (tensile cracking
            // degrades the full stress); the commit still evolves s.
            return ruptured_update_c1(k, &state.stress, d_eps);
        }
        if s_old <= 1e-12 {
            // Zero strength: full zero with full flow booked so the
            // total-strain rebuild (plastic-carried) still trips deletion.
            return StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: equiv_strain_c1(d_eps),
                yielding: true,
            };
        }
        // Start-of-step pressure feeds the pressure factor (explicit).
        let p_state = -(state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        let rate = match dt {
            None => 0.0,
            Some(dt) if self.plastic_rate => {
                // Fixed-point plastic rate: three passes from the total
                // estimate (VP = 1 has no filtering by card design).
                let mut rate = voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
                for _ in 0..3 {
                    let trial = self.return_at_rate(state, d_eps, rate, p_state, e, s_old);
                    rate = trial.2 / dt.max(1e-15);
                }
                rate
            }
            Some(dt) => {
                let (values, _) = principal_strain_c1(d_eps);
                values.iter().map(|v| v.abs()).fold(0.0, f64::max) / dt.max(1e-15)
            }
        };
        let (stress, tangent, dep, yielding, _) =
            self.return_at_rate(state, d_eps, rate, p_state, e, s_old);
        // Trial totals for the tensile checks (same rebuild as commit).
        let elastic = elastic_strain_c1(&stress, k, g);
        let mut total = state.plastic;
        let dir = plastic_flow_c1(&stress);
        for i in 0..6 {
            total[i] += elastic[i] + dep * dir.map(|d| d[i]).unwrap_or(0.0);
        }
        let first = max_principal_strain_c1(&total);
        if first >= self.deletion_strain {
            return StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: dep,
                yielding: true,
            };
        }
        let ep_new = state.equiv_plastic_strain + dep;
        if ep_new >= self.rupture_strain || self.flow_stress(ep_new, rate, p_state) <= 0.0 {
            let p = (stress[0] + stress[1] + stress[2]) / 3.0;
            return StressUpdate {
                stress: [p, p, p, 0.0, 0.0, 0.0],
                tangent: ruptured_tangent_c1(k),
                delta_equiv_plastic_strain: dep,
                yielding: true,
            };
        }
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    /// J2 return at fixed rate, pressure, secant scale, modulus.
    fn return_at_rate(
        &self,
        state: &Law36State,
        d_eps: &[f64; 6],
        rate: f64,
        pressure: f64,
        e: f64,
        s_old: f64,
    ) -> ([f64; 6], [f64; 36], f64, bool, [f64; 6]) {
        let sig0 = s_old * self.flow_stress(0.0, rate, pressure);
        if self.chard == 0.0 {
            let (s, t, d, y) = j2_return_c1(
                e,
                self.poisson_ratio,
                &state.stress,
                state.equiv_plastic_strain,
                d_eps,
                |ep| s_old * self.flow_stress(ep, rate, pressure),
                |ep| s_old * self.hardening_slope(ep, rate, pressure),
            );
            (s, t, d, y, state.back_stress)
        } else {
            j2_return_combined_c1(
                e,
                self.poisson_ratio,
                &state.stress,
                &state.back_stress,
                state.equiv_plastic_strain,
                self.chard,
                sig0,
                d_eps,
                |ep| s_old * self.flow_stress(ep, rate, pressure),
                |ep| s_old * self.hardening_slope(ep, rate, pressure),
            )
        }
    }
}

/// Strain-rate source for the Cowper-Symonds factor (`VP` flag).
///
/// `Plastic` (VP = 1) iterates the plastic rate to fixed point;
/// `Total` (VP = 2, default) uses the tensor Frobenius rate;
/// `Deviatoric` (VP = 3) uses the von Mises equivalent rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law44Rate {
    /// Plastic strain rate (fixed-point iteration, no filtering).
    Plastic,
    /// Total strain rate (default).
    Total,
    /// Deviatoric (equivalent) strain rate.
    Deviatoric,
}

/// Cowper-Symonds metal law (`/MAT/LAW44`, `COWPER`).
///
/// Flow stress `(a + b ep^n)(1 + (rate / c)^(1 / p))` (Theory Manual Eq.
/// 923) with combined iso-kinematic hardening (`chard`), cap `sig_max0`
/// with `ICC` scaling, rupture at `eps_max`, Zhao-type tensile limits
/// (`eps_t1`, `eps_t2`, Eq. 239 as a total scaling), and an optional
/// tabulated yield function (`fct_IDy`, Eq. 236 additive when `a = 0`,
/// Eq. 237 multiplicative when `a > 0`). The rate source follows `VP`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law44;
///
/// let metal = Law44::card_metal();
/// let quasi = metal.flow_stress(0.01, 0.0);
/// let fast = metal.flow_stress(0.01, 1000.0);
/// assert!(fast > quasi);
/// ```
#[derive(Debug, Clone)]
pub struct Law44 {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Initial yield strength in Pa.
    pub a: f64,
    /// Hardening coefficient in Pa.
    pub b: f64,
    /// Hardening exponent, below 1.
    pub n: f64,
    /// Iso-kinematic split in `[0, 1]` (0 isotropic, 1 kinematic).
    pub chard: f64,
    /// Maximum flow stress in Pa (card default 1e20).
    pub max_stress: f64,
    /// Rate coefficient in 1/s (0 disables rate effects).
    pub c: f64,
    /// Rate exponent, positive.
    pub p: f64,
    /// Rate effect on the maximum stress (`ICC`).
    pub max_rate: MaxRateEffect,
    /// Strain-rate source (`VP`).
    pub rate_source: Law44Rate,
    /// Strain-rate filter cutoff in Hz (huge disables).
    pub fcut: f64,
    /// Optional tabulated yield function (`fct_IDy` times `Fscaley`).
    pub yield_function: Option<(Funct1D, f64)>,
    /// Rupture plastic strain (card default 1e20).
    pub rupture_strain: f64,
    /// Tensile strain where stress starts reducing (default 1e20).
    pub tensile_t1: f64,
    /// Tensile strain where stress reaches zero (default 2e20).
    pub tensile_t2: f64,
}

impl Law44 {
    /// Creates a LAW44 material from the full card-equivalent inputs.
    ///
    /// # Panics
    ///
    /// Panics when `n >= 1` (card comment 4), `p <= 0`, or `chard`
    /// leaves `[0, 1]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "material cards take flat parameter lists"
    )]
    pub fn new(
        e: f64,
        nu: f64,
        a: f64,
        b: f64,
        n: f64,
        chard: f64,
        max_stress: f64,
        c: f64,
        p: f64,
        max_rate: MaxRateEffect,
        rate_source: Law44Rate,
        rupture_strain: f64,
        tensile_t1: f64,
        tensile_t2: f64,
    ) -> Self {
        assert!(n < 1.0, "LAW44 hardening exponent must be < 1, got {n}");
        assert!(p > 0.0, "LAW44 rate exponent must be > 0, got {p}");
        assert!(
            (0.0..=1.0).contains(&chard),
            "LAW44 chard must sit in [0, 1], got {chard}"
        );
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            a,
            b,
            n,
            chard,
            max_stress,
            c,
            p,
            max_rate,
            rate_source,
            fcut: 1.0e30,
            yield_function: None,
            rupture_strain,
            tensile_t1,
            tensile_t2,
        }
    }

    /// Metal after the `/MAT/LAW44` card example (p.539).
    ///
    /// `a = 50`, `b = 100`, `n = 0.5` MPa with `Chard = 1`, `SIGMA_max0
    /// = 90 MPa`, `c = 100 /s`, `p = 5`, `ICC = 1`, and open rupture
    /// limits. Values transcribe the card example into SI.
    pub fn card_metal() -> Self {
        Self::new(
            20.5e9,
            0.3,
            50.0e6,
            100.0e6,
            0.5,
            1.0,
            90.0e6,
            100.0,
            5.0,
            MaxRateEffect::Scaled,
            Law44Rate::Total,
            1.0e20,
            1.0e20,
            2.0e20,
        )
    }

    /// Cowper-Symonds factor `1 + (rate / c)^(1 / p)` at a rate.
    pub fn rate_factor(&self, rate: f64) -> f64 {
        cs_rate_factor(rate, self.c, self.p)
    }

    /// Maximum stress with the `ICC` rate treatment applied.
    pub fn rated_cap(&self, rate: f64) -> f64 {
        match self.max_rate {
            MaxRateEffect::Fixed => self.max_stress,
            MaxRateEffect::Scaled => self.max_stress * self.rate_factor(rate),
        }
    }

    /// Static (rate-free) yield at plastic strain.
    fn static_yield(&self, ep: f64) -> f64 {
        match &self.yield_function {
            // Eq. 237 branch shares the tabulated curve; the rate factor
            // multiplies outside.
            Some((f, scale)) if self.a > 0.0 => f.eval(ep.max(0.0)) * scale,
            Some((f, scale)) => f.eval(ep.max(0.0)) * scale,
            None => self.a + self.b * ep.max(0.0).powf(self.n),
        }
    }

    /// Capped flow stress at plastic strain and rate.
    ///
    /// Selects Eq. 236 (additive overstress, tabulated curve with `a =
    /// 0`), Eq. 237 (multiplicative, tabulated curve with `a > 0`), or
    /// Eq. 238 (analytic), then caps at the rated maximum.
    pub fn flow_stress(&self, ep: f64, rate: f64) -> f64 {
        let factor = self.rate_factor(rate);
        let over = factor - 1.0;
        let raw = match &self.yield_function {
            Some((f, scale)) if self.a == 0.0 => {
                f.eval(ep.max(0.0)) * scale + self.b * ep.max(0.0).powf(self.n) * over
            }
            Some(_) => self.static_yield(ep) * factor,
            None => (self.a + self.b * ep.max(0.0).powf(self.n)) * factor,
        };
        raw.min(self.rated_cap(rate))
    }

    /// Hardening slope at plastic strain and rate (0 past the cap).
    pub fn hardening_slope(&self, ep: f64, rate: f64) -> f64 {
        let factor = self.rate_factor(rate);
        let over = factor - 1.0;
        let ep_safe = ep.max(1e-12);
        let raw = self.flow_stress(ep, rate);
        if raw >= self.rated_cap(rate) {
            return 0.0;
        }
        match &self.yield_function {
            Some((_, _)) if self.a == 0.0 => self.b * self.n * ep_safe.powf(self.n - 1.0) * over,
            Some((f, scale)) => f.slope(ep.max(0.0)) * scale * factor,
            None => self.b * self.n * ep_safe.powf(self.n - 1.0) * factor,
        }
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus in Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }
}

/// History state of one LAW44 material point.
#[derive(Debug, Clone)]
pub struct Law44State {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Back-stress in C1 Voigt order (Pa, deviatoric).
    pub back_stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Accumulated plastic strain tensor (C1, for total-strain rebuild).
    pub plastic: [f64; 6],
    /// Whether rupture tripped (deviatoric response gone).
    pub ruptured: bool,
    /// Start-of-step secant tensile scale (explicit Eq. 239 reduction).
    pub tensile_scale: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl Law44State {
    /// Creates a virgin state.
    pub fn new() -> Self {
        Self {
            stress: [0.0; 6],
            back_stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            plastic: [0.0; 6],
            ruptured: false,
            tensile_scale: 1.0,
            yielding: false,
        }
    }
}

impl Default for Law44State {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for Law44 {
    type State = Law44State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_inner(state, d_eps, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.update_inner(state, d_eps, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        self.c > 0.0
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
        // Accumulated-flow fallback at zero stiffness (see Law27).
        if let Some(dir) = plastic_flow_or_accumulated_c1(&update.stress, &state.plastic) {
            accumulate_c1(&mut state.plastic, update.delta_equiv_plastic_strain, &dir);
        }
        // Secant rebuild with the start-of-step scale (explicit scheme),
        // then the end-of-step scale for the next trial.
        let e_sec = state.tensile_scale * self.youngs_modulus;
        let total = if state.tensile_scale <= 1e-12 {
            // Zero stiffness: plastic tensor plus the elastic-limit leg
            // (quasi-static surface over modulus along accumulated
            // flow), continuous with the secant rebuild as s -> 0. The
            // volumetric leg stays undefined at zero strength.
            let mut tot = state.plastic;
            let eq = equiv_deviatoric_c1(&state.plastic);
            if eq > 1e-30 {
                let sweep = (self.flow_stress(state.equiv_plastic_strain, 0.0)
                    / self.youngs_modulus)
                    .max(0.0);
                accumulate_c1(&mut tot, sweep / eq, &state.plastic);
            }
            tot
        } else {
            let g = e_sec / (2.0 * (1.0 + self.poisson_ratio));
            let k = e_sec / (3.0 * (1.0 - 2.0 * self.poisson_ratio));
            total_strain_c1(&state.stress, &state.plastic, k, g)
        };
        let first = max_principal_strain_c1(&total);
        state.tensile_scale = tensile_scale(first, self.tensile_t1, self.tensile_t2);
        // Textbook Prager back-stress from the exact rated hardening
        // increment (exact: the secant scale rides along in the stress
        // ratio).
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && self.chard > 0.0 {
            let q = von_mises_c1(&update.stress);
            let ep_new = state.equiv_plastic_strain;
            let ep_old = ep_new - update.delta_equiv_plastic_strain;
            let quasi_new = self.flow_stress(ep_new, 0.0);
            let ratio = commit_rate_ratio(q, quasi_new);
            let cap = match self.max_rate {
                MaxRateEffect::Fixed => self.max_stress,
                MaxRateEffect::Scaled => self.max_stress * ratio,
            };
            let step = kinematic_step_c1(
                self.chard,
                self.flow_stress(ep_old.max(0.0), 0.0),
                quasi_new,
                ratio,
                cap,
                self.hardening_slope(ep_new, 0.0).max(0.0) * update.delta_equiv_plastic_strain,
            );
            let dir = if step > 0.0 {
                flow_direction_c1(&update.stress)
            } else {
                None
            };
            if let Some(dir) = dir {
                accumulate_c1(&mut state.back_stress, step, &dir);
            }
        }
        if state.equiv_plastic_strain >= self.rupture_strain {
            state.ruptured = true;
        }
    }
}

impl Law44 {
    /// Shared trial path with optional step time for the rate.
    fn update_inner(&self, state: &Law44State, d_eps: &[f64; 6], dt: Option<f64>) -> StressUpdate {
        // Secant tensile scale from the start of the step (explicit): the
        // trial runs on softened moduli with the scaled surface, so the
        // Eq. 239 reduction emerges from the return itself and the
        // plastic bookkeeping stays exact.
        let s_old = state.tensile_scale;
        let e = s_old * self.youngs_modulus;
        let k = e / (3.0 * (1.0 - 2.0 * self.poisson_ratio));
        if state.ruptured {
            // Ruptured pressure rides the scaled bulk (tensile cracking
            // degrades the full stress); the commit still evolves s.
            return ruptured_update_c1(k, &state.stress, d_eps);
        }
        if s_old <= 1e-12 {
            // Zero strength: full zero with full flow booked (comment 9
            // keeps the element alive at zero stress).
            return StressUpdate {
                stress: [0.0; 6],
                tangent: [0.0; 36],
                delta_equiv_plastic_strain: equiv_strain_c1(d_eps),
                yielding: true,
            };
        }
        let rate = match dt {
            None => 0.0,
            Some(dt) => match self.rate_source {
                Law44Rate::Plastic => {
                    // Fixed-point plastic rate: three passes from the
                    // total estimate (VP = 1 has no filtering).
                    let mut rate = voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
                    for _ in 0..3 {
                        let trial = self.return_at_rate(state, d_eps, rate, s_old);
                        rate = trial.2 / dt.max(1e-15);
                    }
                    rate
                }
                // Tensor Frobenius rate of the full increment: tensor
                // shear is half the engineering shear, contracted twice.
                Law44Rate::Total => {
                    let shear =
                        0.5 * (d_eps[3] * d_eps[3] + d_eps[4] * d_eps[4] + d_eps[5] * d_eps[5]);
                    (d_eps[0] * d_eps[0] + d_eps[1] * d_eps[1] + d_eps[2] * d_eps[2] + shear).sqrt()
                        / dt.max(1e-15)
                }
                Law44Rate::Deviatoric => voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15),
            },
        };
        let (stress, tangent, dep, yielding, _) = self.return_at_rate(state, d_eps, rate, s_old);
        let ep_new = state.equiv_plastic_strain + dep;
        if ep_new >= self.rupture_strain {
            let p = (stress[0] + stress[1] + stress[2]) / 3.0;
            return StressUpdate {
                stress: [p, p, p, 0.0, 0.0, 0.0],
                tangent: ruptured_tangent_c1(k),
                delta_equiv_plastic_strain: dep,
                yielding: true,
            };
        }
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    /// J2 return at a fixed rate with secant moduli and surface.
    fn return_at_rate(
        &self,
        state: &Law44State,
        d_eps: &[f64; 6],
        rate: f64,
        s_old: f64,
    ) -> ([f64; 6], [f64; 36], f64, bool, [f64; 6]) {
        let e = s_old * self.youngs_modulus;
        let sig0 = s_old * self.flow_stress(0.0, rate);
        if self.chard == 0.0 {
            let (s, t, d, y) = j2_return_c1(
                e,
                self.poisson_ratio,
                &state.stress,
                state.equiv_plastic_strain,
                d_eps,
                |ep| s_old * self.flow_stress(ep, rate),
                |ep| s_old * self.hardening_slope(ep, rate),
            );
            (s, t, d, y, state.back_stress)
        } else {
            j2_return_combined_c1(
                e,
                self.poisson_ratio,
                &state.stress,
                &state.back_stress,
                state.equiv_plastic_strain,
                self.chard,
                sig0,
                d_eps,
                |ep| s_old * self.flow_stress(ep, rate),
                |ep| s_old * self.hardening_slope(ep, rate),
            )
        }
    }
}

/// Solves a dense 6x6 system by Gauss elimination (partial pivot).
///
/// Returns `None` when the matrix is numerically singular; used only to
/// recover the strain increment in the LAW62 commit, where the consistent
/// tangent is positive definite by construction.
fn solve6(matrix: &[f64; 36], rhs: &[f64; 6]) -> Option<[f64; 6]> {
    let mut a = *matrix;
    let mut x = *rhs;
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if a[row * 6 + col].abs() > a[pivot * 6 + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * 6 + col].abs() <= 1e-30 * a[col * 6 + col].abs().max(1.0) {
            return None;
        }
        if pivot != col {
            for j in col..6 {
                a.swap(col * 6 + j, pivot * 6 + j);
            }
            x.swap(col, pivot);
        }
        let inv = 1.0 / a[col * 6 + col];
        for row in (col + 1)..6 {
            let factor = a[row * 6 + col] * inv;
            for j in col..6 {
                a[row * 6 + j] -= factor * a[col * 6 + j];
            }
            x[row] -= factor * x[col];
        }
    }
    let mut out = [0.0; 6];
    for i in (0..6).rev() {
        let mut sum = x[i];
        for j in (i + 1)..6 {
            sum -= a[i * 6 + j] * out[j];
        }
        out[i] = sum / a[i * 6 + i];
    }
    Some(out)
}

/// Hyper-viscoelastic Ogden-Prony law (`/MAT/LAW62`, `VISC_HYP`).
///
/// Hyperelastic part: compressible Ogden strain energy (Theory Manual Eq.
/// 1028) evaluated exactly in principal stretches, with principal Cauchy
/// stresses `s_a = sum_i (2 mu_i / (al_i J)) (lam_a^al_i - J^(-al_i be))`
/// and `be = nu / (1 - 2 nu)`. Viscous part: Simo finite viscoelasticity
/// (Eq. 1032-1034): `S = gamma_inf S^0 + sum H_i` with branch overstresses
/// `dH_i + H_i / tau_i = gamma_i dS^0` driven by the instantaneous Ogden
/// deviatoric stress rate (not the linear strain rate: the two agree only
/// near `alpha = +-2`). Steps integrate exactly for a piecewise-linear
/// Ogden path. Long-term share `gamma_inf = 1 - sum gamma_i` (Reference
/// Guide Eq. 436); `M = 0` branches recovers pure Ogden. Branch
/// viscosities cap at `max_viscosity` (card `mu_max`).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law62;
///
/// let rubber = Law62::card_rubber();
/// assert!((rubber.ground_shear() - 3.0e6).abs() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Law62 {
    /// Poisson ratio (card: neither 0 nor 1/2).
    pub poisson_ratio: f64,
    /// Ogden pairs `(mu_i, alpha_i)` (Pa, -), at least one.
    pub ogden: Vec<(f64, f64)>,
    /// Prony branches `(gamma_i, tau_i)` (-, s).
    pub prony: Vec<(f64, f64)>,
    /// Maximum branch viscosity in Pa s (card default 1e30).
    pub max_viscosity: f64,
}

impl Law62 {
    /// Creates a LAW62 material from Poisson ratio and parameter pairs.
    ///
    /// # Panics
    ///
    /// Panics when no Ogden pair is given, any `alpha` vanishes, the
    /// shear sum is not positive, any `gamma` leaves `[0, 1]` or sums to
    /// 1 or more, any `tau` is not positive, or `nu` is 1/2 (`nu = 0`
    /// is the fully compressible limit: the volumetric Ogden term is
    /// unity there).
    pub fn new(
        nu: f64,
        ogden: Vec<(f64, f64)>,
        prony: Vec<(f64, f64)>,
        max_viscosity: f64,
    ) -> Self {
        assert!(!ogden.is_empty(), "LAW62 needs at least one Ogden pair");
        assert!(
            ogden.iter().all(|(_, al)| *al != 0.0),
            "LAW62 Ogden exponents must be nonzero"
        );
        let g0: f64 = ogden.iter().map(|(mu, _)| mu).sum();
        assert!(g0 > 0.0, "LAW62 shear sum must be positive, got {g0}");
        let gamma_sum: f64 = prony.iter().map(|(gam, _)| gam).sum();
        assert!(
            prony.iter().all(|(gam, _)| (0.0..=1.0).contains(gam)),
            "LAW62 stiffness ratios must sit in [0, 1]"
        );
        assert!(gamma_sum < 1.0, "LAW62 stiffness ratios must sum below 1");
        assert!(
            prony.iter().all(|(_, tau)| *tau > 0.0),
            "LAW62 relaxation times must be positive"
        );
        assert!(nu != 0.5, "LAW62 Poisson ratio must differ from 1/2");
        Self {
            poisson_ratio: nu,
            ogden,
            prony,
            max_viscosity,
        }
    }

    /// Hyperelastic rubber after the `/MAT/LAW62` card example (p.764).
    ///
    /// Two Ogden pairs (`mu = 2, 1 MPa`, `alpha = 2, -2`), `nu =
    /// 0.495`, no Prony branches. Values transcribe the card into SI.
    pub fn card_rubber() -> Self {
        Self::new(0.495, vec![(2.0e6, 2.0), (1.0e6, -2.0)], Vec::new(), 1.0e30)
    }

    /// Illustrative rubber with one Prony branch (30% stiffness, 10 ms).
    ///
    /// Extends [`card_rubber`](Self::card_rubber) with viscous relaxation;
    /// branch data is illustrative (the card example is hyperelastic).
    pub fn rubber_prony() -> Self {
        Self::new(
            0.495,
            vec![(2.0e6, 2.0), (1.0e6, -2.0)],
            vec![(0.3, 0.01)],
            1.0e30,
        )
    }

    /// Initial (ground) shear modulus `G0 = sum mu_i` in Pa.
    pub fn ground_shear(&self) -> f64 {
        self.ogden.iter().map(|(mu, _)| mu).sum()
    }

    /// Long-term stiffness share `1 - sum gamma_i`.
    pub fn gamma_infinity(&self) -> f64 {
        1.0 - self.prony.iter().map(|(gam, _)| gam).sum::<f64>()
    }

    /// Effective branch moduli with the viscosity cap applied.
    fn branch_moduli(&self) -> Vec<f64> {
        let g0 = self.ground_shear();
        self.prony
            .iter()
            .map(|(gam, tau)| {
                let capped = (self.max_viscosity / tau).max(0.0);
                (gam * g0).min(capped)
            })
            .collect()
    }

    /// Ogden volumetric exponent `be = nu / (1 - 2 nu)`.
    fn ogden_beta(&self) -> f64 {
        self.poisson_ratio / (1.0 - 2.0 * self.poisson_ratio)
    }

    /// Bulk modulus in Pa (from `G0` and `nu`, tangent use).
    pub fn bulk_modulus(&self) -> f64 {
        2.0 * self.ground_shear() * (1.0 + self.poisson_ratio)
            / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Exact Ogden principal Cauchy stresses at principal stretches.
    ///
    /// Closed form of Eq. 1028 via `s_a = (lam_a / J) dW / dlam_a`; the
    /// small-strain limit recovers linear elasticity with `G0` exactly.
    pub fn ogden_principal(&self, lambda: &[f64; 3]) -> [f64; 3] {
        let beta = self.ogden_beta();
        let jacobian = lambda[0] * lambda[1] * lambda[2];
        let mut out = [0.0; 3];
        for (mu, alpha) in &self.ogden {
            let j_term = jacobian.powf(-alpha * beta);
            let prefactor = 2.0 * mu / (alpha * jacobian);
            for a in 0..3 {
                out[a] += prefactor * (lambda[a].powf(*alpha) - j_term);
            }
        }
        out
    }

    /// Instantaneous Ogden deviatoric stress in C1 from a principal system.
    ///
    /// Evaluates [`ogden_principal`](Self::ogden_principal) at the stretches,
    /// removes the mean (Simo internal variables see the deviatoric part
    /// only), and rotates back to C1 order. Drives the branch overstresses.
    fn ogden_dev_from_system(&self, values: &[f64; 3], vecs: &[[f64; 3]; 3]) -> [f64; 6] {
        let lambda = [values[0].exp(), values[1].exp(), values[2].exp()];
        let ogden = self.ogden_principal(&lambda);
        let mean = (ogden[0] + ogden[1] + ogden[2]) / 3.0;
        let dev = [ogden[0] - mean, ogden[1] - mean, ogden[2] - mean];
        let mut back = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let mut entry = 0.0;
                for a in 0..3 {
                    entry += vecs[r][a] * dev[a] * vecs[c][a];
                }
                back[r][c] = entry;
            }
        }
        c1_from_full(&back)
    }

    /// Pure forward trial from state, increment, and optional step time.
    ///
    /// Returns `(stress, tangent, hencky_trial, branches_trial)`. With no
    /// step time the branches respond instantaneously (the `dt -> 0`
    /// limit); with `dt` they integrate exactly for the step. Branches
    /// follow the instantaneous Ogden deviatoric increment (Simo), not
    /// the linear strain increment.
    fn trial(
        &self,
        state: &Law62State,
        d_eps: &[f64; 6],
        dt: Option<f64>,
    ) -> ([f64; 6], [f64; 36], [f64; 6], Vec<[f64; 6]>) {
        assert_eq!(
            state.branches.len(),
            self.prony.len(),
            "LAW62 state branch count must match the material"
        );
        let hencky: [f64; 6] = std::array::from_fn(|i| state.hencky[i] + d_eps[i]);
        let (values, vecs) = principal_strain_c1(&hencky);
        let lambda = [values[0].exp(), values[1].exp(), values[2].exp()];
        let ogden = self.ogden_principal(&lambda);
        let p_og = (ogden[0] + ogden[1] + ogden[2]) / 3.0;
        let s_og = [ogden[0] - p_og, ogden[1] - p_og, ogden[2] - p_og];
        // Simo overstress driver: the instantaneous Ogden deviatoric
        // increment over the step (Eq. 1032-1034 convolve the Ogden
        // stress, so branches track its rate exactly, stiffening included).
        let s0_trial = self.ogden_dev_from_system(&values, &vecs);
        let s0_old = if self.prony.is_empty() {
            [0.0; 6]
        } else {
            let (old_values, old_vecs) = principal_strain_c1(&state.hencky);
            self.ogden_dev_from_system(&old_values, &old_vecs)
        };
        let g0 = self.ground_shear();
        let moduli = self.branch_moduli();
        let mut branches = Vec::with_capacity(self.prony.len());
        let mut tangent_shear = self.gamma_infinity() * g0;
        for (i, (_, tau)) in self.prony.iter().enumerate() {
            let (decay, gain) = match dt {
                Some(dt) => {
                    let beta = dt.max(0.0) / tau;
                    let decay = (-beta).exp();
                    let gain = if beta > 1e-12 {
                        (1.0 - decay) / beta
                    } else {
                        1.0
                    };
                    (decay, gain)
                }
                None => (1.0, 1.0),
            };
            tangent_shear += moduli[i] * gain;
            let share = moduli[i] / g0;
            let mut branch = [0.0; 6];
            for k in 0..6 {
                branch[k] = decay * state.branches[i][k] + share * gain * (s0_trial[k] - s0_old[k]);
            }
            branches.push(branch);
        }
        // Assemble in the trial principal basis: long-term Ogden diagonal
        // plus rotated branch overstresses, then rotate back to C1.
        let gamma_inf = self.gamma_infinity();
        let mut full = [[0.0; 3]; 3];
        for branch in branches.iter() {
            let b = full_from_c1(branch);
            for a in 0..3 {
                for bb in 0..3 {
                    let mut entry = 0.0;
                    for r in 0..3 {
                        for c in 0..3 {
                            entry += vecs[r][a] * b[r][c] * vecs[c][bb];
                        }
                    }
                    full[a][bb] += entry;
                }
            }
        }
        for a in 0..3 {
            full[a][a] += gamma_inf * s_og[a] + p_og;
        }
        let mut back = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let mut entry = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        entry += vecs[r][a] * full[a][b] * vecs[c][b];
                    }
                }
                back[r][c] = entry;
            }
        }
        let stress = c1_from_full(&back);
        // Linear gain-weighted shear: encodes the step time for the
        // commit recovery (which inverts this exact formula) and seeds
        // its Newton solve. It understates the true Ogden tangent at
        // large |alpha|, but explicit dynamics never consumes it and
        // the commit differentiates the true trial by finite differences.
        let tangent = elastic_tangent_c1(self.bulk_modulus(), tangent_shear);
        (stress, tangent, hencky, branches)
    }
}

/// History state of one LAW62 material point.
#[derive(Debug, Clone)]
pub struct Law62State {
    /// Hencky (logarithmic) total strain in C1 order.
    pub hencky: [f64; 6],
    /// Prony branch overstresses in C1 order (one per branch).
    pub branches: Vec<[f64; 6]>,
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
}

impl Law62State {
    /// Creates a virgin state with the given branch count.
    ///
    /// # Panics
    ///
    /// Panics only indirectly: [`Law62`] asserts the count matches on
    /// every update.
    pub fn new(branches: usize) -> Self {
        Self {
            hencky: [0.0; 6],
            branches: vec![[0.0; 6]; branches],
            stress: [0.0; 6],
        }
    }
}

impl StressUpdateModel for Law62 {
    type State = Law62State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let (stress, tangent, _, _) = self.trial(state, d_eps, None);
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        let (stress, tangent, _, _) = self.trial(state, d_eps, Some(dt));
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn is_rate_dependent(&self) -> bool {
        !self.prony.is_empty()
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        // The C1 commit carries neither the strain increment nor the step
        // time, but the hyperelastic history needs both. Recover them from
        // the update itself: dt bisects out of the tangent shear (the
        // gain-weighted sum decreases strictly in dt), then the increment
        // Newton-solves out of the stress delta with a finite-difference
        // Jacobian of the true trial. Re-running the trial from the
        // recovered pair stores exactly the history the update evaluated
        // (proven by the recovery unit tests).
        let dt = self.recover_dt(update);
        let delta: [f64; 6] = std::array::from_fn(|i| update.stress[i] - state.stress[i]);
        // Exact Newton with a central-difference Jacobian: the analytic
        // tangent drops the Ogden spin terms and grossly understates
        // the stiffness for large exponents (alpha ~ 30 stiffens the
        // trial several-fold by gamma ~ 0.1), which diverges a chord
        // iteration. Finite differences differentiate the true trial
        // (spin terms included) at 12 trial evaluations per pass;
        // backtracking keeps every accepted iterate an improvement.
        let stress_scale = update.stress.iter().map(|v| v.abs()).fold(1.0, f64::max);
        let trial_stress_of = |inc: &[f64; 6]| {
            let (trial_stress, _, _, _) = self.trial(state, inc, dt);
            trial_stress
        };
        let residual_of = |inc: &[f64; 6]| {
            let trial_stress = trial_stress_of(inc);
            let vector: [f64; 6] = std::array::from_fn(|i| update.stress[i] - trial_stress[i]);
            let norm: f64 = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
            (vector, norm)
        };
        let mut best = solve6(&update.tangent, &delta).unwrap_or([0.0; 6]);
        let mut best_norm = residual_of(&best).1;
        for _ in 0..20 {
            if best_norm <= 1e-13 * stress_scale {
                break;
            }
            let mut jacobian = [0.0; 36];
            for col in 0..6 {
                let h = 1e-7 * (1.0 + best[col].abs());
                let mut plus = best;
                plus[col] += h;
                let mut minus = best;
                minus[col] -= h;
                let up = trial_stress_of(&plus);
                let down = trial_stress_of(&minus);
                for row in 0..6 {
                    jacobian[row * 6 + col] = (up[row] - down[row]) / (2.0 * h);
                }
            }
            let (residual, _) = residual_of(&best);
            let Some(correction) = solve6(&jacobian, &residual) else {
                break;
            };
            if !correction.iter().all(|v| v.is_finite()) {
                break;
            }
            let corr_norm: f64 = correction.iter().map(|v| v * v).sum::<f64>().sqrt();
            let best_size: f64 = best.iter().map(|v| v * v).sum::<f64>().sqrt();
            if corr_norm <= 1e-15 * (1.0 + best_size) {
                break;
            }
            let mut step = 1.0;
            let mut improved = false;
            for _ in 0..10 {
                let candidate: [f64; 6] = std::array::from_fn(|i| best[i] + step * correction[i]);
                let candidate_norm = residual_of(&candidate).1;
                if candidate_norm.is_finite() && candidate_norm < best_norm {
                    best = candidate;
                    best_norm = candidate_norm;
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            if !improved {
                break;
            }
        }
        let recovered = best;
        let (_, _, hencky, branches) = self.trial(state, &recovered, dt);
        state.hencky = hencky;
        state.branches = branches;
        state.stress = update.stress;
    }
}

impl Law62 {
    /// Recovers the step time from the tangent shear of an update.
    ///
    /// The shear-block diagonal carries `G_inf + sum G_i gain_i(dt)`
    /// with each gain decreasing strictly from 1 (dt = 0) to 0, so one
    /// scalar bisection inverts it. Returns `None` for the instantaneous
    /// path (pure Ogden or `update` without step time).
    fn recover_dt(&self, update: &StressUpdate) -> Option<f64> {
        if self.prony.is_empty() {
            return None;
        }
        let target = update.tangent[3 * 6 + 3];
        let g0 = self.ground_shear();
        let moduli = self.branch_moduli();
        let shear_at = |dt: f64| {
            let mut total = self.gamma_infinity() * g0;
            for (i, (_, tau)) in self.prony.iter().enumerate() {
                let beta = dt / tau;
                let gain = if beta > 1e-12 {
                    (1.0 - (-beta).exp()) / beta
                } else {
                    1.0
                };
                total += moduli[i] * gain;
            }
            total
        };
        // Instantaneous path (or virgin tangent): gains all read 1.
        if target >= shear_at(0.0) * (1.0 - 1e-12) {
            return None;
        }
        let floor = self.gamma_infinity() * g0;
        if target <= floor {
            // Fully relaxed: gains all read 0 (dt -> infinity, where the
            // trial evaluates decay 0 / gain 0 exactly).
            return Some(f64::INFINITY);
        }
        let mut lo = 0.0;
        let mut hi = self.prony.iter().map(|(_, tau)| *tau).fold(0.0, f64::max);
        while shear_at(hi) > target {
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if shear_at(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }
}

/// Strain-rate treatment of the LAW66 yield stress (`Iyld_rate`).
///
/// `Static` (0) ignores rate; `CowperSymonds` (1, default) and `LogRate`
/// (2) scale the blended static yield; `ScaleCurves` (3) scales each side
/// with its own rate function; `MultiTable` (4) reads each side from a
/// `(plastic strain, rate)` table. The card's ragged multi-curve form of
/// mode 4 restricts here to rectangular [`Table2D`] grids.
#[derive(Debug, Clone)]
pub enum Law66Rate {
    /// No rate effect.
    Static,
    /// Cowper-Symonds factor `1 + (rate / ref)^(1 / c)`.
    CowperSymonds {
        /// Reference strain rate in 1/s.
        ref_rate: f64,
        /// Rate exponent parameter.
        c: f64,
    },
    /// Logarithmic factor `1 + c ln(rate / ref)`.
    LogRate {
        /// Reference strain rate in 1/s.
        ref_rate: f64,
        /// Rate sensitivity, dimensionless.
        c: f64,
    },
    /// Per-side rate scale functions (of strain rate).
    ScaleCurves {
        /// Tension scale versus rate.
        tension: Funct1D,
        /// Compression scale versus rate.
        compression: Funct1D,
        /// Tension ordinate scale.
        fscale_t: f64,
        /// Compression ordinate scale.
        fscale_c: f64,
    },
    /// Per-side yield tables over `(plastic strain, rate)`.
    MultiTable {
        /// Tension yield over `(ep, rate)`.
        tension: Table2D,
        /// Compression yield over `(ep, rate)`.
        compression: Table2D,
    },
}

/// Tension-compression tabulated metal law (`/MAT/LAW66`).
///
/// Blends separate tension and compression hardening curves by pressure:
/// `sy = al st(ep) + (1 - al) sc(ep)` with `al = (Pc - P) / (Pc + Pt)`
/// inside `[-Pt, Pc]` (Reference Guide Eq. 280, pressure positive in
/// compression); outside the range, or when both limits vanish, the
/// tension curve rules at `P <= 0` and the compression curve at `P > 0`.
/// Rate effects follow [`Law66Rate`]; the Cowper-Symonds form adds the
/// overstress to `initial_yield` when it is positive and multiplies
/// otherwise (card p.595). Combined hardening via `chard`; no rupture
/// inputs on the card (failure models stay in `/FAIL`).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::law_metals::Law66;
///
/// let alu = Law66::card_aluminum();
/// let tensile = alu.static_yield(0.01, -100.0e6);
/// let compressive = alu.static_yield(0.01, 100.0e6);
/// assert!((tensile - compressive).abs() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Law66 {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Iso-kinematic split in `[0, 1]` (0 isotropic, 1 kinematic).
    pub chard: f64,
    /// Limit pressure in compression, Pa (positive).
    pub pc: f64,
    /// Limit pressure in tension, Pa (positive).
    pub pt: f64,
    /// Tension hardening curve (`fct_IDt` times `Fscalet`).
    pub tension: (Funct1D, f64),
    /// Compression hardening curve (`fct_IDc` times `Fscalec`).
    pub compression: (Funct1D, f64),
    /// Strain-rate treatment.
    pub rate: Law66Rate,
    /// Rate source: plastic (`VP = 1`) or total (`VP = 0`, default).
    pub plastic_rate: bool,
    /// Initial yield for the additive rate form (0 multiplies).
    pub initial_yield: f64,
}

impl Law66 {
    /// Creates a LAW66 material from moduli, curves, limits, rate.
    ///
    /// # Panics
    ///
    /// Panics when `chard` leaves `[0, 1]` or a limit pressure is
    /// negative.
    #[expect(
        clippy::too_many_arguments,
        reason = "material cards take flat parameter lists"
    )]
    pub fn new(
        e: f64,
        nu: f64,
        chard: f64,
        pc: f64,
        pt: f64,
        tension: (Funct1D, f64),
        compression: (Funct1D, f64),
        rate: Law66Rate,
        plastic_rate: bool,
        initial_yield: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&chard),
            "LAW66 chard must sit in [0, 1], got {chard}"
        );
        assert!(pc >= 0.0 && pt >= 0.0, "LAW66 limit pressures must be >= 0");
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            chard,
            pc,
            pt,
            tension,
            compression,
            rate,
            plastic_rate,
            initial_yield,
        }
    }

    /// Aluminum after the `/MAT/LAW66` card example (p.594).
    ///
    /// Symmetric 90-to-170 MPa curves with `E = 60.4 GPa`, `nu = 0.33`,
    /// `Pc = 500 MPa`, `Pt = 600 MPa`, static response. Values transcribe
    /// the card example into SI.
    pub fn card_aluminum() -> Self {
        let x = vec![0.0, 0.08];
        let y = vec![90.0e6, 170.0e6];
        let tension = Funct1D::new(x.clone(), y.clone()).expect("card knots must validate");
        let compression = Funct1D::new(x, y).expect("card knots must validate");
        Self::new(
            60.4e9,
            0.33,
            0.0,
            500.0e6,
            600.0e6,
            (tension, 1.0),
            (compression, 1.0),
            Law66Rate::Static,
            false,
            0.0,
        )
    }

    /// Tension weight `al` at Radioss pressure (positive compression).
    pub fn blend(&self, pressure: f64) -> f64 {
        if self.pc + self.pt <= 0.0 {
            return if pressure <= 0.0 { 1.0 } else { 0.0 };
        }
        if pressure < -self.pt {
            return 1.0;
        }
        if pressure > self.pc {
            return 0.0;
        }
        (self.pc - pressure) / (self.pc + self.pt)
    }

    /// Static blended yield at plastic strain and pressure.
    pub fn static_yield(&self, ep: f64, pressure: f64) -> f64 {
        let al = self.blend(pressure);
        let yt = self.tension.0.eval(ep.max(0.0)) * self.tension.1;
        let yc = self.compression.0.eval(ep.max(0.0)) * self.compression.1;
        al * yt + (1.0 - al) * yc
    }

    /// Static blended slope at plastic strain and pressure.
    fn static_slope(&self, ep: f64, pressure: f64) -> f64 {
        let al = self.blend(pressure);
        let st = self.tension.0.slope(ep.max(0.0)) * self.tension.1;
        let sc = self.compression.0.slope(ep.max(0.0)) * self.compression.1;
        al * st + (1.0 - al) * sc
    }

    /// Flow stress at plastic strain, rate, and pressure.
    ///
    /// Takes equivalent plastic strain, strain rate (1/s), and Radioss
    /// pressure (Pa, positive in compression); returns Pa.
    pub fn flow_stress(&self, ep: f64, rate: f64, pressure: f64) -> f64 {
        let al = self.blend(pressure);
        match &self.rate {
            Law66Rate::Static => self.static_yield(ep, pressure),
            Law66Rate::CowperSymonds { ref_rate, c } => {
                let factor = if *ref_rate <= 0.0 || *c <= 0.0 || rate <= 0.0 {
                    1.0
                } else {
                    1.0 + (rate / ref_rate).powf(1.0 / c)
                };
                let base = self.static_yield(ep, pressure);
                if self.initial_yield > 0.0 {
                    base + self.initial_yield * (factor - 1.0)
                } else {
                    base * factor
                }
            }
            Law66Rate::LogRate { ref_rate, c } => {
                self.static_yield(ep, pressure) * jc_rate_factor(rate, *c, *ref_rate)
            }
            Law66Rate::ScaleCurves {
                tension,
                compression,
                fscale_t,
                fscale_c,
            } => {
                let yt = self.tension.0.eval(ep.max(0.0)) * self.tension.1;
                let yc = self.compression.0.eval(ep.max(0.0)) * self.compression.1;
                al * yt * tension.eval(rate.max(0.0)) * fscale_t
                    + (1.0 - al) * yc * compression.eval(rate.max(0.0)) * fscale_c
            }
            Law66Rate::MultiTable {
                tension,
                compression,
            } => {
                let yt = tension.eval(ep.max(0.0), rate.max(0.0)) * self.tension.1;
                let yc = compression.eval(ep.max(0.0), rate.max(0.0)) * self.compression.1;
                al * yt + (1.0 - al) * yc
            }
        }
    }

    /// Hardening slope at strain, rate, pressure (rate frozen).
    pub fn hardening_slope(&self, ep: f64, rate: f64, pressure: f64) -> f64 {
        let al = self.blend(pressure);
        match &self.rate {
            Law66Rate::Static => self.static_slope(ep, pressure),
            Law66Rate::CowperSymonds { ref_rate, c } => {
                let factor = if *ref_rate <= 0.0 || *c <= 0.0 || rate <= 0.0 {
                    1.0
                } else {
                    1.0 + (rate / ref_rate).powf(1.0 / c)
                };
                if self.initial_yield > 0.0 {
                    self.static_slope(ep, pressure)
                } else {
                    self.static_slope(ep, pressure) * factor
                }
            }
            Law66Rate::LogRate { ref_rate, c } => {
                self.static_slope(ep, pressure) * jc_rate_factor(rate, *c, *ref_rate)
            }
            Law66Rate::ScaleCurves {
                tension,
                compression,
                fscale_t,
                fscale_c,
            } => {
                let st = self.tension.0.slope(ep.max(0.0)) * self.tension.1;
                let sc = self.compression.0.slope(ep.max(0.0)) * self.compression.1;
                al * st * tension.eval(rate.max(0.0)) * fscale_t
                    + (1.0 - al) * sc * compression.eval(rate.max(0.0)) * fscale_c
            }
            // Table2D exposes no slope: central differences (tangent and
            // kinematic-modulus use only; the return solve reads yield).
            Law66Rate::MultiTable {
                tension,
                compression,
            } => {
                let h = 1e-6 * (1.0 + ep.abs());
                let hi = self.blended_table(ep + h, rate, al, tension, compression);
                let lo = self.blended_table((ep - h).max(0.0), rate, al, tension, compression);
                (hi - lo) / ((ep + h) - (ep - h).max(0.0)).max(1e-30)
            }
        }
    }

    /// Blended table yield shared by the multi-table slope stencil.
    fn blended_table(
        &self,
        ep: f64,
        rate: f64,
        al: f64,
        tension: &Table2D,
        compression: &Table2D,
    ) -> f64 {
        al * tension.eval(ep, rate) * self.tension.1
            + (1.0 - al) * compression.eval(ep, rate) * self.compression.1
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus in Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }
}

/// History state of one LAW66 material point.
#[derive(Debug, Clone)]
pub struct Law66State {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Back-stress in C1 Voigt order (Pa, deviatoric).
    pub back_stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl Law66State {
    /// Creates a virgin state.
    pub fn new() -> Self {
        Self {
            stress: [0.0; 6],
            back_stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }
}

impl Default for Law66State {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for Law66 {
    type State = Law66State;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_inner(state, d_eps, None)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        self.update_inner(state, d_eps, Some(dt))
    }

    fn is_rate_dependent(&self) -> bool {
        !matches!(self.rate, Law66Rate::Static)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
        // Textbook Prager back-stress from the exact rated hardening
        // increment (exact for the multiplicative rate branches; table
        // branches lag one step here; the blend has no cap).
        let pressure = -(state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && self.chard > 0.0 {
            let q = von_mises_c1(&update.stress);
            let ep_new = state.equiv_plastic_strain;
            let ep_old = ep_new - update.delta_equiv_plastic_strain;
            let quasi_new = self.flow_stress(ep_new, 0.0, pressure);
            let ratio = commit_rate_ratio(q, quasi_new);
            let step = kinematic_step_c1(
                self.chard,
                self.flow_stress(ep_old.max(0.0), 0.0, pressure),
                quasi_new,
                ratio,
                f64::INFINITY,
                self.hardening_slope(ep_new, 0.0, pressure).max(0.0)
                    * update.delta_equiv_plastic_strain,
            );
            let dir = if step > 0.0 {
                flow_direction_c1(&update.stress)
            } else {
                None
            };
            if let Some(dir) = dir {
                accumulate_c1(&mut state.back_stress, step, &dir);
            }
        }
    }
}

impl Law66 {
    /// Shared trial path with optional step time for the rate.
    fn update_inner(&self, state: &Law66State, d_eps: &[f64; 6], dt: Option<f64>) -> StressUpdate {
        // Start-of-step pressure feeds the tension/compression blend and
        // the rate-free static yield (explicit, like LAW36).
        let p_state = -(state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        let rate = match dt {
            None => 0.0,
            Some(dt) if self.plastic_rate => {
                let mut rate = voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
                for _ in 0..3 {
                    let trial = self.return_at_rate(state, d_eps, rate, p_state);
                    rate = trial.2 / dt.max(1e-15);
                }
                rate
            }
            Some(dt) => voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15),
        };
        let (stress, tangent, dep, yielding, _) = self.return_at_rate(state, d_eps, rate, p_state);
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    /// J2 return at a fixed rate and pressure.
    fn return_at_rate(
        &self,
        state: &Law66State,
        d_eps: &[f64; 6],
        rate: f64,
        pressure: f64,
    ) -> ([f64; 6], [f64; 36], f64, bool, [f64; 6]) {
        let sig0 = self.flow_stress(0.0, rate, pressure);
        if self.chard == 0.0 {
            let (s, t, d, y) = j2_return_c1(
                self.youngs_modulus,
                self.poisson_ratio,
                &state.stress,
                state.equiv_plastic_strain,
                d_eps,
                |ep| self.flow_stress(ep, rate, pressure),
                |ep| self.hardening_slope(ep, rate, pressure),
            );
            (s, t, d, y, state.back_stress)
        } else {
            j2_return_combined_c1(
                self.youngs_modulus,
                self.poisson_ratio,
                &state.stress,
                &state.back_stress,
                state.equiv_plastic_strain,
                self.chard,
                sig0,
                d_eps,
                |ep| self.flow_stress(ep, rate, pressure),
                |ep| self.hardening_slope(ep, rate, pressure),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::crash::{
        drive_uniaxial_stress, drive_uniaxial_stress_rate,
    };

    #[test]
    fn combined_matches_isotropic_when_chard_zero() {
        let stress = [100.0e6, 20.0e6, -30.0e6, 40.0e6, 0.0, 10.0e6];
        let back = [0.0; 6];
        let d_eps = [0.004, -0.001, -0.001, 0.001, 0.0, 0.0];
        let sig_y = |ep: f64| 250.0e6 + 1.0e9 * ep;
        let slope = |_ep: f64| 1.0e9;
        let (s_iso, t_iso, d_iso, y_iso) =
            j2_return_c1(200.0e9, 0.3, &stress, 0.01, &d_eps, sig_y, slope);
        let sig_y = |ep: f64| 250.0e6 + 1.0e9 * ep;
        let slope = |_ep: f64| 1.0e9;
        let (s_comb, t_comb, d_comb, y_comb, back_new) = j2_return_combined_c1(
            200.0e9, 0.3, &stress, &back, 0.01, 0.0, 250.0e6, &d_eps, sig_y, slope,
        );
        assert_eq!(y_iso, y_comb);
        assert!((d_iso - d_comb).abs() < 1e-15);
        for i in 0..6 {
            assert!((s_iso[i] - s_comb[i]).abs() < 1e-6 * s_iso[i].abs().max(1.0));
            assert!(back_new[i].abs() < 1e-12);
        }
        for i in 0..36 {
            assert!((t_iso[i] - t_comb[i]).abs() < 1e-6 * t_iso[i].abs().max(1.0));
        }
    }

    #[test]
    fn jacobi_recovers_rotated_spectrum() {
        // 30-degree rotation about z applied to diag(1, 2, 3).
        let (c, s) = (0.5 * PI / 3.0 * 2.0_f64).sin_cos();
        let (c, s) = (s, c);
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let d = [1.0, 2.0, 3.0];
        let mut full = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                full[i][j] =
                    r[i][0] * d[0] * r[j][0] + r[i][1] * d[1] * r[j][1] + r[i][2] * d[2] * r[j][2];
            }
        }
        let t = c1_from_full(&full);
        let (values, vecs) = principal_system_c1(&t);
        let mut sorted = values;
        sorted.sort_by(|x, y| x.total_cmp(y));
        for (got, want) in sorted.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "value {got} vs {want}");
        }
        // Eigenvectors stay orthonormal.
        for i in 0..3 {
            for j in 0..3 {
                let dot =
                    vecs[0][i] * vecs[0][j] + vecs[1][i] * vecs[1][j] + vecs[2][i] * vecs[2][j];
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "orthonormality ({i},{j})");
            }
        }
        // Round trip through the principal rebuild.
        let rebuilt = c1_from_principal(&values, &vecs);
        for i in 0..6 {
            assert!((rebuilt[i] - t[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn jc_rate_quiet_below_reference() {
        assert_eq!(jc_rate_factor(0.5, 0.014, 1.0), 1.0);
        assert_eq!(jc_rate_factor(1.0, 0.014, 1.0), 1.0);
        assert_eq!(jc_rate_factor(1000.0, 0.0, 1.0), 1.0);
        let fast = jc_rate_factor(1000.0, 0.014, 1.0);
        assert!((fast - (1.0 + 0.014 * 1000.0_f64.ln())).abs() < 1e-15);
        assert_eq!(cs_rate_factor(0.0, 40.0, 5.0), 1.0);
        assert_eq!(cs_rate_factor(100.0, 0.0, 5.0), 1.0);
        let cs = cs_rate_factor(100.0, 40.0, 5.0);
        assert!((cs - (1.0 + (100.0_f64 / 40.0).powf(0.2))).abs() < 1e-15);
    }

    #[test]
    fn smooth_filter_matches_eq921() {
        let prev = 10.0;
        let raw = 100.0;
        let a = 2.0 * PI * 1.0e-6 * 10.0e3;
        let got = smooth_rate_step(prev, raw, 1.0e-6, 10.0e3);
        assert!((got - (a * raw + (1.0 - a) * prev)).abs() < 1e-9);
        assert_eq!(smooth_rate_step(prev, raw, 1.0e-6, 0.0), raw);
        assert_eq!(smooth_rate_step(prev, raw, 1.0e-6, 1.0e30), raw);
    }

    #[test]
    fn tensile_scale_ramps() {
        assert_eq!(tensile_scale(0.05, 0.1, 0.2), 1.0);
        assert_eq!(tensile_scale(0.25, 0.1, 0.2), 0.0);
        assert!((tensile_scale(0.15, 0.1, 0.2) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn law2_quasistatic_driver_follows_closed_form() {
        let steel = Law2::c4340_steel();
        let mut state = Law2State::new(298.0);
        let curve = drive_uniaxial_stress(&steel, &mut state, 0.05, 100);
        let last = curve.last().unwrap();
        assert!(last.yielding);
        let expected = steel.flow_stress(last.equiv_plastic_strain, 1.0, 298.0);
        assert!(
            (last.stress_xx - expected).abs() / expected < 0.01,
            "stress {} vs closed form {expected}",
            last.stress_xx
        );
    }

    #[test]
    fn law2_cap_holds_and_rupture_zeroes_deviatoric() {
        let mut capped = Law2::c4340_steel();
        capped.max_stress = 900.0e6;
        let mut state = Law2State::new(298.0);
        let curve = drive_uniaxial_stress(&capped, &mut state, 0.05, 100);
        let last = curve.last().unwrap();
        assert!(
            (last.stress_xx - 900.0e6).abs() / 900.0e6 < 0.01,
            "capped stress {}",
            last.stress_xx
        );
        let mut rupturable = Law2::c4340_steel();
        rupturable.rupture_strain = 0.02;
        let mut rstate = Law2State::new(298.0);
        drive_uniaxial_stress(&rupturable, &mut rstate, 0.05, 100);
        assert!(rstate.ruptured);
        assert!(von_mises_c1(&rstate.stress) < 1.0);
    }

    #[test]
    fn law2_za_branch_orders_rate_and_temperature() {
        let za = Law2::new(
            200.0e9,
            0.3,
            Law2Flow::ZerilliArmstrong {
                c0: 100.0e6,
                c1: 500.0e6,
                c3: 0.002,
                c4: 0.0002,
                c5: 300.0e6,
                n: 0.5,
            },
            1.0,
            0.0,
            0.0,
            298.0,
            0.0,
            1.0e30,
            MaxRateEffect::Scaled,
            1.0e30,
            0.0,
            1.0e30,
        );
        let slow = za.flow_stress(0.01, 1.0, 300.0);
        let fast = za.flow_stress(0.01, 1000.0, 300.0);
        assert!(fast > slow);
        let cold = za.flow_stress(0.01, 1.0, 300.0);
        let hot = za.flow_stress(0.01, 1.0, 500.0);
        assert!(hot < cold);
        assert!(za.hardening_slope(0.01, 1.0, 300.0) > 0.0);
    }

    #[test]
    fn law2_kinematic_shows_bauschinger() {
        let mut kine = Law2::c4340_steel();
        kine.chard = 1.0;
        let mut state = Law2State::new(298.0);
        let forward = drive_uniaxial_stress(&kine, &mut state, 0.02, 40);
        let peak = forward.last().unwrap().stress_xx;
        assert!(state.back_stress[0] > 0.0);
        let back_after_forward = state.back_stress[0];
        let reverse = drive_uniaxial_stress(&kine, &mut state, -0.04, 80);
        let trough = reverse.iter().map(|p| p.stress_xx).fold(0.0, f64::min);
        assert!(trough.abs() < peak, "trough {trough} vs peak {peak}");
        assert!(state.back_stress[0] < back_after_forward);
    }

    #[test]
    fn law2_update_pure_and_commit_heats() {
        let steel = Law2::c4340_steel();
        let state = Law2State::new(298.0);
        let before = state.clone();
        let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let update = steel.update(&state, &d_eps);
        assert_eq!(state.stress, before.stress);
        assert_eq!(state.back_stress, before.back_stress);
        assert_eq!(state.equiv_plastic_strain, before.equiv_plastic_strain);
        assert_eq!(state.temperature, before.temperature);
        let mut committed = state.clone();
        steel.commit(&mut committed, &update);
        assert!(committed.temperature > 298.0);
    }

    #[test]
    fn law22_identity_below_damage_strain() {
        let law = Law22::card_aluminum();
        let state = Law22State::new(298.0);
        let core_state = Law2State::new(298.0);
        // Small step stays below Eps_dam = 0.1: identical to LAW2.
        let d_eps = [0.02, -0.006, -0.006, 0.0, 0.0, 0.0];
        let got = law.update(&state, &d_eps);
        let want = law.base.update(&core_state, &d_eps);
        for i in 0..6 {
            assert!((got.stress[i] - want.stress[i]).abs() < 1e-9);
        }
        assert_eq!(law.damage_factor(0.05), 0.0);
    }

    #[test]
    fn law22_softening_slope_matches_et() {
        let law = Law22::card_aluminum();
        let mut state = Law22State::new(298.0);
        // Card aluminum fully damages at ep = 0.15; stop at 0.13 for a
        // mid-range damage with a settled softening tangent.
        let curve = drive_uniaxial_stress(&law, &mut state, 0.13, 260);
        // Tangent between strain 0.115 and 0.1295 (damage mid-range).
        let slope = (curve[259].stress_xx - curve[230].stress_xx)
            / (curve[259].strain_xx - curve[230].strain_xx);
        assert!(
            (slope - law.softening_slope).abs() / law.softening_slope.abs() < 0.10,
            "slope {slope} vs Et {}",
            law.softening_slope
        );
        assert!(
            state.damage > 0.3 && state.damage < 0.8,
            "damage {}",
            state.damage
        );
    }

    #[test]
    fn law22_bulk_preserved_and_full_damage_zeroes() {
        let law = Law22::card_aluminum();
        // K_dam = E_dam / (3 (1 - 2 nu_dam)) equals K at any damage.
        let e = law.base.youngs_modulus;
        let k0 = e / (3.0 * (1.0 - 2.0 * law.base.poisson_ratio));
        for d in [0.0, 0.25, 0.5, 0.9] {
            let e_dam = (1.0 - d) * e;
            let nu_dam = 0.5 * d + (1.0 - d) * law.base.poisson_ratio;
            let k_dam = e_dam / (3.0 * (1.0 - 2.0 * nu_dam));
            assert!((k_dam - k0).abs() / k0 < 1e-12, "bulk drift at d={d}");
        }
        assert!((law.shear_ratio(0.0) - 1.0).abs() < 1e-15);
        assert!(law.shear_ratio(1.0) < 1e-15);
        // Card aluminum fully damages before Eps_max = 0.2.
        let mut state = Law22State::new(298.0);
        drive_uniaxial_stress(&law, &mut state, 0.19, 380);
        assert!((state.damage - 1.0).abs() < 1e-9);
        assert!(von_mises_c1(&state.inner.stress) < 1.0e6);
    }

    #[test]
    fn law22_freezes_plastic_strain_at_full_damage() {
        let law = Law22::card_aluminum();
        let mut state = Law22State::new(298.0);
        drive_uniaxial_stress(&law, &mut state, 0.19, 380);
        assert!((state.damage - 1.0).abs() < 1e-9);
        // Past full damage the engine skips the plastic update: the
        // equivalent strain pins while straining continues at zero
        // deviatoric stress (harvested LOI22 PLAS behavior).
        let pinned = state.inner.equiv_plastic_strain;
        drive_uniaxial_stress(&law, &mut state, 0.05, 100);
        assert!((state.inner.equiv_plastic_strain - pinned).abs() < 1e-12);
        assert!(von_mises_c1(&state.inner.stress) < 1.0e6);
    }

    #[test]
    fn law62_nu_zero_is_fully_compressible() {
        // Deck LOI62 runs nu = 0: the volumetric Ogden term is unity
        // and updates stay finite.
        let foam = Law62::new(0.0, vec![(2.0e6, 2.0)], Vec::new(), 1.0e30);
        let principal = foam.ogden_principal(&[1.2, 1.0, 1.0]);
        // J = 1.2, beta = 0: s1 = 2 mu / 2 (lam^2 - 1) / J.
        let want = 2.0e6 * (1.44 - 1.0) / 1.2;
        assert!((principal[0] - want).abs() < 1.0);
        let mut state = Law62State::new(0);
        let update = foam.update(&state, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);
        foam.commit(&mut state, &update);
        assert!(state.stress.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn law27_quiet_in_compression() {
        let law = Law27::steel_brittle();
        let mut state = Law27State::new(298.0);
        let mut core = Law2State::new(298.0);
        let curve = drive_uniaxial_stress(&law, &mut state, -0.05, 100);
        let reference = drive_uniaxial_stress(&law.base, &mut core, -0.05, 100);
        assert_eq!(state.damage, [0.0, 0.0]);
        assert!(!state.ruptured);
        let last = curve.last().unwrap();
        let want = reference.last().unwrap();
        assert!((last.stress_xx - want.stress_xx).abs() < 1.0);
    }

    #[test]
    fn law27_damages_and_deletes_in_tension() {
        let law = Law27::steel_brittle();
        let mut state = Law27State::new(298.0);
        let curve = drive_uniaxial_stress(&law, &mut state, 0.10, 200);
        assert!(state.damage[0] > 0.1);
        // Damaged response sits below the undamaged core at same strain.
        let mut core = Law2State::new(298.0);
        let reference = drive_uniaxial_stress(&law.base, &mut core, 0.10, 200);
        let last = curve.last().unwrap();
        let want = reference.last().unwrap();
        assert!(last.stress_xx < 0.95 * want.stress_xx);
        assert!(last.stress_xx > 0.0);
        // Past eps_f = 0.16 the point deletes with zero stress.
        drive_uniaxial_stress(&law, &mut state, 0.08, 160);
        assert!(state.ruptured);
        assert!(state.inner.stress.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn law27_update_pure() {
        let law = Law27::steel_brittle();
        let state = Law27State::new(298.0);
        let before = state.clone();
        let _ = law.update(&state, &[0.05, -0.015, -0.015, 0.0, 0.0, 0.0]);
        assert_eq!(state.inner.stress, before.inner.stress);
        assert_eq!(state.plastic, before.plastic);
        assert_eq!(state.damage, before.damage);
    }

    #[test]
    fn law36_hits_card_knots_and_driver() {
        let alu = Law36::card_aluminum();
        assert!((alu.nominal_yield(0.0, 0.0) - 90.0e6).abs() < 1.0);
        assert!((alu.nominal_yield(0.075, 0.0) - 170.0e6).abs() < 1.0);
        let mut state = Law36State::new();
        let curve = drive_uniaxial_stress(&alu, &mut state, 0.05, 100);
        let last = curve.last().unwrap();
        let expected = alu.nominal_yield(last.equiv_plastic_strain, 0.0);
        assert!(
            (last.stress_xx - expected).abs() / expected < 0.01,
            "stress {} vs tabulated {expected}",
            last.stress_xx
        );
    }

    #[test]
    fn law36_rate_interp_and_strain_extrapolation() {
        let lo = RateCurve::new(0.0, vec![0.0, 0.1], vec![100.0e6, 200.0e6], 1.0).unwrap();
        let hi = RateCurve::new(100.0, vec![0.0, 0.1], vec![150.0e6, 250.0e6], 1.0).unwrap();
        let law = Law36::new(
            200.0e9,
            0.3,
            vec![lo, hi],
            0.0,
            1.0e20,
            1.0e20,
            2.0e20,
            3.0e20,
        );
        assert!((law.nominal_yield(0.05, 50.0) - 175.0e6).abs() < 1.0);
        assert!((law.nominal_yield(0.05, 150.0) - 225.0e6).abs() < 1.0);
        // End-slope extrapolation past the last strain knot.
        assert!((law.nominal_yield(0.15, 0.0) - 250.0e6).abs() < 1.0);
        assert!(law.is_rate_dependent());
        assert!(!Law36::card_aluminum().is_rate_dependent());
    }

    #[test]
    fn law36_tensile_limits_and_deletion() {
        let curve = RateCurve::new(0.0, vec![0.0, 0.2], vec![100.0e6, 200.0e6], 1.0).unwrap();
        let law = Law36::new(200.0e9, 0.3, vec![curve], 0.0, 1.0e20, 0.05, 0.10, 0.12);
        let mut state = Law36State::new();
        let mid = drive_uniaxial_stress(&law, &mut state, 0.075, 150);
        let last = mid.last().unwrap();
        // Tensile ramp at e1 ~ 0.075 reads (0.10 - 0.075) / 0.05 = 0.5.
        let full = law.nominal_yield(last.equiv_plastic_strain, 0.0);
        assert!(
            (last.stress_xx - 0.5 * full).abs() / full < 0.10,
            "scaled {} vs {}",
            last.stress_xx,
            0.5 * full
        );
        drive_uniaxial_stress(&law, &mut state, 0.06, 120);
        assert!(state.deleted);
        assert!(state.stress.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn law44_closed_form_and_cap() {
        let metal = Law44::card_metal();
        let quasi = metal.flow_stress(0.01, 0.0);
        assert!((quasi - 60.0e6).abs() < 1.0e3);
        let factor = 1.0 + (1000.0_f64 / 100.0).powf(0.2);
        let fast = metal.flow_stress(0.01, 1000.0);
        assert!((fast - 60.0e6 * factor).abs() < 1.0e3);
        assert!((metal.rated_cap(0.0) - 90.0e6).abs() < 1.0);
        assert!((metal.rated_cap(1000.0) - 90.0e6 * factor).abs() < 1.0e3);
    }

    #[test]
    fn law44_fast_driver_scales_with_cs_factor() {
        let metal = Law44::card_metal();
        let mut quasi_state = Law44State::new();
        let quasi = drive_uniaxial_stress(&metal, &mut quasi_state, 0.05, 100);
        let mut fast_state = Law44State::new();
        let fast = drive_uniaxial_stress_rate(&metal, &mut fast_state, 0.05, 100, Some(5.0e-5));
        let q = quasi.last().unwrap();
        let f = fast.last().unwrap();
        // Mean total rate ~ 10 /s; both legs share chard = 1, so the
        // ratio reads the Cowper-Symonds factor at matched strain.
        let rate = 10.0;
        let expected_ratio = metal.rate_factor(rate);
        let got_ratio = f.stress_xx / q.stress_xx;
        assert!(
            (got_ratio - expected_ratio).abs() / expected_ratio < 0.05,
            "ratio {got_ratio} vs factor {expected_ratio}"
        );
        assert!((q.equiv_plastic_strain - f.equiv_plastic_strain).abs() < 0.005);
    }

    #[test]
    fn law44_fct_branches_follow_eq236_237() {
        let f = Funct1D::new(vec![0.0, 0.1], vec![100.0e6, 150.0e6]).unwrap();
        // Additive branch (a = 0): fct + b ep^n (factor - 1).
        let mut additive = Law44::card_metal();
        additive.a = 0.0;
        additive.yield_function = Some((f.clone(), 1.0));
        let factor = additive.rate_factor(1000.0);
        let got_add = additive.flow_stress(0.04, 1000.0);
        let want_add = 120.0e6 + 100.0e6 * 0.04_f64.powf(0.5) * (factor - 1.0);
        assert!((got_add - want_add.min(additive.rated_cap(1000.0))).abs() < 1.0e3);
        // Multiplicative branch (a > 0): fct * factor.
        let mut mult = Law44::card_metal();
        mult.yield_function = Some((f, 1.0));
        let got_mult = mult.flow_stress(0.04, 1000.0);
        assert!((got_mult - (120.0e6 * factor).min(mult.rated_cap(1000.0))).abs() < 1.0e3);
    }

    #[test]
    fn law44_plastic_rate_self_consistent() {
        let mut law = Law44::card_metal();
        law.chard = 0.0;
        law.rate_source = Law44Rate::Plastic;
        law.rupture_strain = 1.0e30;
        let state = Law44State::new();
        let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let update = law.update_with_rate(&state, &d_eps, 0.001);
        assert!(update.yielding);
        let rate_est = update.delta_equiv_plastic_strain / 0.001;
        let q = von_mises_c1(&update.stress);
        let expected = law.flow_stress(update.delta_equiv_plastic_strain, rate_est);
        assert!(
            (q - expected).abs() / expected < 0.03,
            "q {q} vs fixed-point flow {expected}"
        );
    }

    #[test]
    fn law62_ogden_linear_limit() {
        let rubber = Law62::card_rubber();
        let state = Law62State::new(0);
        // Constrained uniaxial increment: sigma_xx = (K + 4G/3) eps.
        let update = rubber.update(&state, &[1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let g0 = rubber.ground_shear();
        let constrained = rubber.bulk_modulus() + 4.0 * g0 / 3.0;
        assert!(
            (update.stress[0] - constrained * 1.0e-6).abs() / (constrained * 1.0e-6) < 0.005,
            "stress {} vs linear {}",
            update.stress[0],
            constrained * 1.0e-6
        );
        // Engineering shear 2e-6 = tensor shear 1e-6: sigma_xy = 2G eps.
        let shear = rubber.update(&state, &[0.0, 0.0, 0.0, 2.0e-6, 0.0, 0.0]);
        assert!((shear.stress[3] - 2.0 * g0 * 1.0e-6).abs() < 1.0e-3);
    }

    #[test]
    fn law62_ogden_uniaxial_closed_form() {
        let rubber = Law62::card_rubber();
        let l2 = 1.0 / 1.5_f64.sqrt();
        let got = rubber.ogden_principal(&[1.5, l2, l2]);
        // J = 1: s_a = sum (2 mu / al) (lam_a^al - 1).
        let want1 = 2.0e6 * (1.5_f64.powi(2) - 1.0) - 1.0e6 * (1.5_f64.powi(-2) - 1.0);
        assert!((got[0] - want1).abs() / want1 < 1e-12);
        assert!((got[1] - got[2]).abs() < 1e-9);
    }

    #[test]
    fn law62_prony_relaxes_to_long_term() {
        let rubber = Law62::rubber_prony();
        let mut state = Law62State::new(1);
        let step = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        let update = rubber.update_with_rate(&state, &step, 1.0e-4);
        rubber.commit(&mut state, &update);
        let instant = state.stress;
        let instant_mean = (instant[0] + instant[1] + instant[2]) / 3.0;
        let instant_dev = instant[0] - instant_mean;
        // Hold 10 relaxation times with zero increments.
        let mut previous = instant[0];
        for _ in 0..100 {
            let hold = rubber.update_with_rate(&state, &[0.0; 6], 0.001);
            rubber.commit(&mut state, &hold);
            // Newton recovery noise sits near 1e-12 absolute strain, so
            // the monotonicity gate allows 1e-9 relative slack per step.
            assert!(state.stress[0] <= previous * (1.0 + 1e-9));
            previous = state.stress[0];
        }
        // Long-term deviatoric = gamma_inf share of the Ogden stress.
        let (values, _) = principal_strain_c1(&state.hencky);
        let lambda = [values[0].exp(), values[1].exp(), values[2].exp()];
        let ogden = rubber.ogden_principal(&lambda);
        let p_og = (ogden[0] + ogden[1] + ogden[2]) / 3.0;
        let branch_free = rubber.gamma_infinity() * (ogden[0] - p_og) + p_og;
        // Rotation-free hold keeps the xx slot on the first principal.
        assert!(
            (state.stress[0] - branch_free).abs() / branch_free.abs().max(1.0) < 0.02,
            "relaxed {} vs long-term {branch_free}",
            state.stress[0]
        );
        let relaxed_mean = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        let relaxed_dev = state.stress[0] - relaxed_mean;
        // Physical check: the deviatoric overstress (instant 1.0 share)
        // decays toward the long-term gamma_inf share (0.71), cleanly
        // separated from the non-relaxing volumetric pressure.
        assert!(
            relaxed_dev > 0.65 * instant_dev && relaxed_dev < 0.78 * instant_dev,
            "deviatoric {relaxed_dev} vs instant {instant_dev}"
        );
    }

    #[test]
    fn law62_commit_recovers_history_exactly() {
        for law in [Law62::card_rubber(), Law62::rubber_prony()] {
            let branches = law.prony.len();
            let mut state = Law62State::new(branches);
            let d_eps = [0.004, -0.001, 0.0005, 0.001, -0.0005, 0.0002];
            let update = law.update_with_rate(&state, &d_eps, 0.003);
            law.commit(&mut state, &update);
            for (i, (hencky, increment)) in state.hencky.iter().zip(d_eps.iter()).enumerate() {
                assert!(
                    (hencky - increment).abs() < 1e-12,
                    "hencky drift in slot {i}"
                );
            }
            // Branch history matches the exact Simo step: the share of
            // the Ogden deviatoric increment (virgin start, decay from
            // zero branches), not the linear strain increment.
            let (values, vecs) = principal_strain_c1(&d_eps);
            let s0 = law.ogden_dev_from_system(&values, &vecs);
            for (b, (gam, tau)) in state.branches.iter().zip(law.prony.iter()) {
                let beta = 0.003 / tau;
                let gain = (1.0 - (-beta).exp()) / beta;
                for k in 0..6 {
                    let want = gam * gain * s0[k];
                    assert!(
                        (b[k] - want).abs() < 1e-9 * want.abs().max(1.0),
                        "branch drift in slot {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn law62_update_pure() {
        let rubber = Law62::rubber_prony();
        let state = Law62State::new(1);
        let before = state.clone();
        let _ = rubber.update_with_rate(&state, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 0.001);
        assert_eq!(state.hencky, before.hencky);
        assert_eq!(state.branches, before.branches);
        assert_eq!(state.stress, before.stress);
    }

    #[test]
    fn law66_blend_matches_eq280() {
        let alu = Law66::card_aluminum();
        assert!((alu.blend(0.0) - 500.0 / 1100.0).abs() < 1e-15);
        assert_eq!(alu.blend(-600.0e6), 1.0);
        assert_eq!(alu.blend(500.0e6), 0.0);
        assert_eq!(alu.blend(-700.0e6), 1.0);
        assert_eq!(alu.blend(600.0e6), 0.0);
        let mut sharp = alu.clone();
        sharp.pc = 0.0;
        sharp.pt = 0.0;
        assert_eq!(sharp.blend(-1.0), 1.0);
        assert_eq!(sharp.blend(1.0), 0.0);
    }

    #[test]
    fn law66_split_selects_sides() {
        let tension = Funct1D::new(vec![0.0, 0.1], vec![100.0e6, 150.0e6]).unwrap();
        let compression = Funct1D::new(vec![0.0, 0.1], vec![300.0e6, 350.0e6]).unwrap();
        let law = Law66::new(
            200.0e9,
            0.3,
            0.0,
            0.0,
            0.0,
            (tension, 1.0),
            (compression, 1.0),
            Law66Rate::Static,
            false,
            0.0,
        );
        assert!((law.static_yield(0.05, -10.0) - 125.0e6).abs() < 1.0);
        assert!((law.static_yield(0.05, 10.0) - 325.0e6).abs() < 1.0);
        // Uniaxial drivers land on their own sides.
        let mut tensile = Law66State::new();
        let pull = drive_uniaxial_stress(&law, &mut tensile, 0.05, 100);
        let last = pull.last().unwrap();
        assert!(
            (last.stress_xx - 125.0e6).abs() / 125.0e6 < 0.02,
            "tensile {}",
            last.stress_xx
        );
        let mut compressive = Law66State::new();
        let push = drive_uniaxial_stress(&law, &mut compressive, -0.05, 100);
        let first = push.last().unwrap();
        assert!(
            (first.stress_xx + 325.0e6).abs() / 325.0e6 < 0.02,
            "compressive {}",
            first.stress_xx
        );
    }

    #[test]
    fn law66_cs_additive_and_multiplicative() {
        let base = Law66::card_aluminum();
        let static_y = base.static_yield(0.01, 0.0);
        let mut mult = base.clone();
        mult.rate = Law66Rate::CowperSymonds {
            ref_rate: 1.0,
            c: 5.0,
        };
        let factor = 1.0 + 1000.0_f64.powf(0.2);
        assert!((mult.flow_stress(0.01, 1000.0, 0.0) - static_y * factor).abs() < 1.0e3);
        let mut add = base.clone();
        add.rate = Law66Rate::CowperSymonds {
            ref_rate: 1.0,
            c: 5.0,
        };
        add.initial_yield = 90.0e6;
        let want = static_y + 90.0e6 * (factor - 1.0);
        assert!((add.flow_stress(0.01, 1000.0, 0.0) - want).abs() < 1.0e3);
        // Log-rate branch quiets below its reference.
        let mut log = base.clone();
        log.rate = Law66Rate::LogRate {
            ref_rate: 1.0,
            c: 0.05,
        };
        assert!((log.flow_stress(0.01, 0.5, 0.0) - static_y).abs() < 1.0);
        assert!(log.flow_stress(0.01, 100.0, 0.0) > static_y);
    }

    #[test]
    fn law66_multitable_bilinear() {
        // Flat row-major layout over (rate, ep): v(y, x) in grid order.
        let table = Table2D::new(
            vec![0.0, 0.1],
            vec![0.0, 100.0],
            vec![100.0e6, 150.0e6, 200.0e6, 250.0e6],
        )
        .unwrap();
        let mut law = Law66::card_aluminum();
        law.rate = Law66Rate::MultiTable {
            tension: table.clone(),
            compression: table,
        };
        assert!((law.flow_stress(0.05, 50.0, -1.0) - 175.0e6).abs() < 1.0);
        // Slope stencil reads d/d ep = 500 MPa per unit strain.
        let slope = law.hardening_slope(0.05, 50.0, -1.0);
        assert!((slope - 500.0e6).abs() / 500.0e6 < 0.01, "slope {slope}");
    }

    #[test]
    fn law44_chard_one_matches_flow_on_monotonic() {
        // Chard only splits isotropic/kinematic hardening: on a
        // monotonic path chard = 1 must reproduce the chard = 0
        // response (the exact-increment commit telescopes there).
        let mut got = Vec::new();
        for chard in [0.0, 1.0] {
            let mut law = Law44::card_metal();
            law.chard = chard;
            let mut state = Law44State::new();
            let path = drive_uniaxial_stress_rate(&law, &mut state, 0.05, 200, Some(5.0e-5));
            got.push(path.last().unwrap().stress_xx.abs());
        }
        assert!(
            (got[1] - got[0]).abs() / got[0] < 0.01,
            "chard legs {} vs {}",
            got[1],
            got[0]
        );
    }

    #[test]
    fn law62_branches_track_ogden_rate() {
        // Simo branches follow the Ogden stress increment (deck-like
        // alpha = 30.5 stiffens 44% past linear at this shear, so a
        // linear driver would miss by far).
        let law = Law62::new(0.0, vec![(1.0e6, 30.5)], vec![(0.3, 10.0)], 1.0e30);
        let mut state = Law62State::new(1);
        let d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1];
        let update = law.update_with_rate(&state, &d, 1.0);
        law.commit(&mut state, &update);
        let (values, vecs) = principal_strain_c1(&d);
        let s0 = law.ogden_dev_from_system(&values, &vecs);
        let gain = (1.0 - (-0.1f64).exp()) / 0.1;
        let want = 0.3 * gain * s0[5];
        assert!(
            (state.branches[0][5] - want).abs() < 1e-9 * want.abs(),
            "branch {} vs Ogden-driven {want}",
            state.branches[0][5]
        );
        // Linear driver would give 2 G0 gain (d/2): 44% below here.
        let linear = 2.0 * 0.3 * 1.0e6 * gain * 0.05;
        assert!(
            (want - linear).abs() / linear.abs() > 0.2,
            "test needs Ogden/linear separation: {want} vs {linear}"
        );
    }
}
