//! Rate-dependent crash plasticity models through contract C1.
//!
//! Implements the crash-material chain of the FEM elasticity program: the
//! Johnson-Cook thermo-viscoplastic model, Cowper-Symonds rate scaling,
//! piecewise-linear plasticity, and Deshpande-Fleck crushable foam. Every
//! model implements [`StressUpdateModel`](super::update::StressUpdateModel)
//! with C1 Voigt increments `[f64; 6]`, so beam fibers, shell layers, and
//! the explicit dynamics engine consume them without concrete coupling.
//!
//! Rate dependence enters only through the additive C1 provided methods
//! [`update_with_rate`](super::update::StressUpdateModel::update_with_rate)
//! and [`is_rate_dependent`](super::update::StressUpdateModel::is_rate_dependent).
//! The plain `update` always returns the quasi-static response at the
//! reference strain rate, which keeps single-element drivers and the
//! quasi-static QA meaningful for every crash model.

// Rust guideline compliant 2026-02-21

use super::material::{J2Material, PlasticState};
use super::update::{StressUpdate, StressUpdateModel, von_mises_c1};
use crate::physics::solid::fracture::ductile::JohnsonCookDamage;

/// J2 radial return with a nonlinear isotropic yield curve.
///
/// Advances `stress` (C1 Voigt, Pa) by `d_eps` (C1 Voigt) with elastic
/// moduli `e` and `nu`, starting from accumulated plastic strain `ep`. The
/// yield function `sig_y` and its slope `slope` are evaluated at the
/// end-of-step plastic strain, so power-law and tabulated hardening are
/// exact rather than linearized. Solves the scalar consistency equation by
/// bisection and returns `(stress, tangent, dep, yielding)` with the
/// Simo-Hughes consistent tangent in C1 order.
pub(crate) fn j2_return_c1(
    e: f64,
    nu: f64,
    stress: &[f64; 6],
    ep: f64,
    d_eps: &[f64; 6],
    sig_y: impl Fn(f64) -> f64,
    slope: impl Fn(f64) -> f64,
) -> ([f64; 6], [f64; 36], f64, bool) {
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
    // von_mises_c1 removes the hydrostatic part internally.
    let trial_full = [
        s_trial[0] + p_trial,
        s_trial[1] + p_trial,
        s_trial[2] + p_trial,
        s_trial[3],
        s_trial[4],
        s_trial[5],
    ];
    let q_trial = von_mises_c1(&trial_full);

    if q_trial <= sig_y(ep) {
        let mut elastic = s_trial;
        elastic[0] += p_trial;
        elastic[1] += p_trial;
        elastic[2] += p_trial;
        return (elastic, elastic_tangent_c1(k, g), 0.0, false);
    }

    // Consistency: q_trial - 3 G dep - sig_y(ep + dep) = 0.
    let three_g = 3.0 * g;
    let mut lo = 0.0;
    let mut hi = ((q_trial - sig_y(ep)) / three_g).max(1e-15);
    // The perfect-plasticity root brackets hardening curves; expand the
    // bracket geometrically so softening segments still trap the root.
    for _ in 0..60 {
        if q_trial - three_g * hi - sig_y(ep + hi) <= 0.0 {
            break;
        }
        hi *= 2.0;
    }
    // Bisection converges linearly; 80 passes drive dep below 1e-15
    // relative even for strongly nonlinear power-law hardening.
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let residual = q_trial - three_g * mid - sig_y(ep + mid);
        if residual > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dep = 0.5 * (lo + hi);
    let ratio = 1.0 - three_g * dep / q_trial;
    let mut new_stress = [0.0; 6];
    for i in 0..3 {
        new_stress[i] = ratio * s_trial[i] + p_trial;
    }
    for i in 3..6 {
        new_stress[i] = ratio * s_trial[i];
    }
    let h_iso = slope(ep + dep).max(0.0);
    let tangent = consistent_tangent_j2_c1(k, g, q_trial, dep, &s_trial, h_iso);
    (new_stress, tangent, dep, true)
}

/// Isotropic elastic tangent in C1 Voigt order, row-major 6x6.
pub(crate) fn elastic_tangent_c1(k: f64, g: f64) -> [f64; 36] {
    let lambda = k - 2.0 * g / 3.0;
    let mut c = [0.0; 36];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 6 + j] = if i == j { lambda + 2.0 * g } else { lambda };
        }
    }
    for i in 3..6 {
        c[i * 6 + i] = g;
    }
    c
}

/// Simo-Hughes consistent tangent for J2 plasticity in C1 order.
///
/// Follows Simo and Hughes, Computational Inelasticity, Eq. 3.7.10: the
/// volumetric part stays elastic, the deviatoric part scales by `theta`,
/// and the flow-direction correction uses `theta_bar` with the isotropic
/// slope. Structurally identical to the legacy internal-ordering version
/// because both key on normal (index < 3) versus shear entries.
pub(crate) fn consistent_tangent_j2_c1(
    k: f64,
    g: f64,
    q_trial: f64,
    dep: f64,
    s_trial: &[f64; 6],
    h_iso: f64,
) -> [f64; 36] {
    let theta = 1.0 - 3.0 * g * dep / q_trial;
    let theta_bar = 1.0 / (1.0 + h_iso / (3.0 * g)) - (1.0 - theta);
    let g2 = 2.0 * g;
    let n: [f64; 6] = std::array::from_fn(|i| s_trial[i] / q_trial);
    let mut c = [0.0; 36];
    for i in 0..6 {
        let w_i = if i < 3 { 1.0 } else { 2.0_f64.sqrt() };
        for j in 0..6 {
            let w_j = if j < 3 { 1.0 } else { 2.0_f64.sqrt() };
            let mut entry = 0.0;
            if i < 3 && j < 3 {
                entry += k;
            }
            let i_dev = if i == j {
                if i < 3 { 2.0 / 3.0 } else { 0.5 }
            } else if i < 3 && j < 3 {
                -1.0 / 3.0
            } else {
                0.0
            };
            entry += g2 * theta * i_dev;
            entry -= g2 * theta_bar * w_i * w_j * n[i] * n[j];
            c[i * 6 + j] = entry;
        }
    }
    c
}

/// Johnson-Cook thermo-viscoplastic material.
///
/// Flow stress `sigma = (A + B ep^n) (1 + C ln e*) (1 - T*^m)` with
/// normalized rate `e* = max(ep_dot, tiny) / ep_dot_0` and homologous
/// temperature `T* = clamp((T - T_room) / (T_melt - T_room), 0, 1)`.
/// Above melt the strength is zero. Units are SI (Pa, K, 1/s).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::JohnsonCookMaterial;
///
/// let steel = JohnsonCookMaterial::c4340_steel();
/// let quasi = steel.flow_stress(0.01, 1.0, 293.0);
/// let fast = steel.flow_stress(0.01, 1000.0, 293.0);
/// assert!(fast > quasi);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JohnsonCookMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Initial yield strength A in Pa.
    pub a: f64,
    /// Hardening coefficient B in Pa.
    pub b: f64,
    /// Hardening exponent n, dimensionless.
    pub n: f64,
    /// Rate sensitivity C, dimensionless.
    pub c: f64,
    /// Thermal softening exponent m, dimensionless.
    pub m: f64,
    /// Reference strain rate in 1/s.
    pub reference_strain_rate: f64,
    /// Room temperature in K.
    pub room_temperature: f64,
    /// Melt temperature in K.
    pub melt_temperature: f64,
    /// Mass density in kg/m^3 (adiabatic heating).
    pub density: f64,
    /// Specific heat in J/(kg K) (adiabatic heating).
    pub specific_heat: f64,
    /// Taylor-Quinney fraction of plastic work converted to heat.
    pub taylor_quinney: f64,
    /// Optional Johnson-Cook damage accumulation model.
    pub damage: Option<JohnsonCookDamage>,
}

impl JohnsonCookMaterial {
    /// Creates a Johnson-Cook material from mechanical parameters.
    ///
    /// Thermal capacity defaults to zero, which disables adiabatic heating;
    /// set `density`, `specific_heat`, and `taylor_quinney` for coupled runs.
    /// Damage stays `None` until assigned.
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
        c: f64,
        m: f64,
        reference_strain_rate: f64,
        room_temperature: f64,
        melt_temperature: f64,
    ) -> Self {
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            a,
            b,
            n,
            c,
            m,
            reference_strain_rate,
            room_temperature,
            melt_temperature,
            density: 0.0,
            specific_heat: 0.0,
            taylor_quinney: 0.9,
            damage: None,
        }
    }

    /// AISI 4340 steel after Johnson and Cook (1983).
    pub fn c4340_steel() -> Self {
        Self::new(
            200.0e9, 0.3, 792.0e6, 510.0e6, 0.26, 0.014, 1.03, 1.0, 293.0, 1793.0,
        )
    }

    /// OFHC copper after Johnson and Cook (1983).
    pub fn ofhc_copper() -> Self {
        Self::new(
            117.0e9, 0.35, 90.0e6, 292.0e6, 0.31, 0.025, 1.09, 1.0, 293.0, 1356.0,
        )
    }

    /// AISI 4340 steel with adiabatic heating enabled.
    ///
    /// Same flow parameters as [`c4340_steel`](Self::c4340_steel) plus
    /// handbook mass data, so `commit` accumulates the adiabatic
    /// temperature rise of the plastic work.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::crash::JohnsonCookMaterial;
    ///
    /// let steel = JohnsonCookMaterial::c4340_steel_thermal();
    /// assert!(steel.density * steel.specific_heat > 0.0);
    /// ```
    pub fn c4340_steel_thermal() -> Self {
        // Handbook mass data for low-alloy steel: density 7830 kg/m^3,
        // specific heat 475 J/(kg K). Raising either slows the adiabatic
        // temperature rise proportionally; both feed only the commit-side
        // heating term, never the flow stress directly.
        let mut material = Self::c4340_steel();
        material.density = 7830.0;
        material.specific_heat = 475.0;
        material
    }

    /// OFHC copper with adiabatic heating enabled.
    ///
    /// Same flow parameters as [`ofhc_copper`](Self::ofhc_copper) plus
    /// handbook mass data for the commit-side temperature rise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::crash::JohnsonCookMaterial;
    ///
    /// let copper = JohnsonCookMaterial::ofhc_copper_thermal();
    /// assert!(copper.density * copper.specific_heat > 0.0);
    /// ```
    pub fn ofhc_copper_thermal() -> Self {
        // Handbook mass data for OFHC copper: density 8960 kg/m^3,
        // specific heat 383 J/(kg K). Same heating-only role as above.
        let mut material = Self::ofhc_copper();
        material.specific_heat = 383.0;
        material.density = 8960.0;
        material
    }

    /// Homologous temperature in `[0, 1]`.
    pub fn homologous_temperature(&self, temperature: f64) -> f64 {
        let span = (self.melt_temperature - self.room_temperature).max(1e-9);
        ((temperature - self.room_temperature) / span).clamp(0.0, 1.0)
    }

    /// Rate factor `1 + C ln(e*)`, floored at 1e-3.
    pub fn rate_factor(&self, strain_rate: f64) -> f64 {
        let star = (strain_rate.max(1e-12) / self.reference_strain_rate).max(1e-9);
        (1.0 + self.c * star.ln()).max(1e-3)
    }

    /// Thermal factor `1 - T*^m`.
    pub fn thermal_factor(&self, temperature: f64) -> f64 {
        1.0 - self.homologous_temperature(temperature).powf(self.m)
    }

    /// Flow stress at plastic strain, rate, and temperature.
    ///
    /// Takes equivalent plastic strain (dimensionless), equivalent plastic
    /// strain rate (1/s), and temperature (K); returns stress in Pa.
    pub fn flow_stress(&self, ep: f64, strain_rate: f64, temperature: f64) -> f64 {
        let hardening = self.a + self.b * ep.max(0.0).powf(self.n);
        hardening * self.rate_factor(strain_rate) * self.thermal_factor(temperature)
    }

    /// Hardening slope `d sigma / d ep` at plastic strain.
    ///
    /// Evaluates the analytic derivative with the plastic strain floored at
    /// 1e-12, which keeps the singular `n < 1` slope finite; the return
    /// mapping solves the consistency equation exactly, so the slope only
    /// feeds the tangent.
    pub fn hardening_slope(&self, ep: f64, strain_rate: f64, temperature: f64) -> f64 {
        let ep_safe = ep.max(1e-12);
        self.b
            * self.n
            * ep_safe.powf(self.n - 1.0)
            * self.rate_factor(strain_rate)
            * self.thermal_factor(temperature)
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
    /// Adds `dep / ef(triax, rate, T*)` to `state.damage` from the trial
    /// stress triaxiality when the material carries a damage model; a no-op
    /// otherwise. Drivers call this after [`commit`](StressUpdateModel::commit)
    /// with the step strain rate. Takes the accepted update and the
    /// equivalent plastic strain rate in 1/s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::plasticity::crash::{JohnsonCookMaterial, JohnsonCookState};
    /// use sonium_fem::physics::solid::plasticity::StressUpdateModel;
    /// use sonium_fem::physics::solid::fracture::ductile::JohnsonCookDamage;
    ///
    /// let mut material = JohnsonCookMaterial::c4340_steel();
    /// material.damage = Some(JohnsonCookDamage::c4340_steel());
    /// let mut state = JohnsonCookState::new(293.0);
    /// let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
    /// let update = material.update(&state, &d_eps);
    /// material.commit(&mut state, &update);
    /// material.accumulate_damage(&mut state, &update, 1.0);
    /// assert!(state.damage > 0.0);
    /// ```
    pub fn accumulate_damage(
        &self,
        state: &mut JohnsonCookState,
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
}

/// History state of one Johnson-Cook material point.
#[derive(Debug, Clone)]
pub struct JohnsonCookState {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Accumulated Johnson-Cook damage in `[0, 1]`.
    pub damage: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl JohnsonCookState {
    /// Creates a virgin state at the given temperature in K.
    pub fn new(temperature: f64) -> Self {
        Self {
            stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            temperature,
            damage: 0.0,
            yielding: false,
        }
    }
}

impl JohnsonCookMaterial {
    /// Trial update at an explicit strain rate estimate.
    fn update_at_rate(
        &self,
        state: &JohnsonCookState,
        d_eps: &[f64; 6],
        strain_rate: f64,
    ) -> StressUpdate {
        let temp = state.temperature;
        let (stress, tangent, dep, yielding) = j2_return_c1(
            self.youngs_modulus,
            self.poisson_ratio,
            &state.stress,
            state.equiv_plastic_strain,
            d_eps,
            |ep| self.flow_stress(ep, strain_rate, temp),
            |ep| self.hardening_slope(ep, strain_rate, temp),
        );
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }
}

impl StressUpdateModel for JohnsonCookMaterial {
    type State = JohnsonCookState;
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
        // Explicit codes estimate the plastic rate from the total increment;
        // the elastic part of the error is negligible once yielding.
        let rate = super::update::voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
        self.update_at_rate(state, d_eps, rate)
    }

    fn is_rate_dependent(&self) -> bool {
        true
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
        // Adiabatic heating from the plastic work of this step.
        let capacity = self.density * self.specific_heat;
        if update.yielding && update.delta_equiv_plastic_strain > 0.0 && capacity > 0.0 {
            let q = von_mises_c1(&update.stress);
            state.temperature +=
                self.taylor_quinney * q * update.delta_equiv_plastic_strain / capacity;
        }
    }
}

/// Cowper-Symonds rate scaling over a J2 base material.
///
/// Scales the quasi-static yield and hardening slope of `base` by the
/// dynamic factor `1 + (ep_dot / D)^(1/q)`. Only the isotropic part of
/// the base material is used; back-stress stays untouched. The plain
/// `update` returns the quasi-static response (factor exactly 1 at zero
/// rate); `update_with_rate` evaluates the factor at the estimated rate.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::CowperSymondsJ2;
/// use sonium_fem::physics::solid::plasticity::J2Material;
///
/// let scaled = CowperSymondsJ2::new(J2Material::mild_steel(), 40.0, 5.0);
/// assert!((scaled.dynamic_factor(0.0) - 1.0).abs() < 1e-15);
/// assert!(scaled.dynamic_factor(100.0) > 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CowperSymondsJ2 {
    /// Quasi-static J2 base material.
    pub base: J2Material,
    /// Rate coefficient D in 1/s.
    pub d_rate: f64,
    /// Rate exponent q, dimensionless.
    pub q_exponent: f64,
}

impl CowperSymondsJ2 {
    /// Creates a Cowper-Symonds model from a base and rate parameters.
    pub fn new(base: J2Material, d_rate: f64, q_exponent: f64) -> Self {
        Self {
            base,
            d_rate,
            q_exponent,
        }
    }

    /// Mild steel with textbook rate parameters D=40/s, q=5.
    pub fn mild_steel() -> Self {
        Self::new(J2Material::mild_steel(), 40.0, 5.0)
    }

    /// Dynamic yield factor `1 + (ep_dot / D)^(1/q)` at a rate.
    pub fn dynamic_factor(&self, strain_rate: f64) -> f64 {
        1.0 + (strain_rate.max(0.0) / self.d_rate.max(1e-30)).powf(1.0 / self.q_exponent.max(1e-9))
    }

    /// Trial update at an explicit strain rate estimate.
    fn update_at_rate(
        &self,
        state: &PlasticState,
        d_eps: &[f64; 6],
        strain_rate: f64,
    ) -> StressUpdate {
        let internal: [f64; 6] = state.stress.as_slice().try_into().unwrap_or_else(|_| {
            panic!("CowperSymondsJ2 needs a 3D PlasticState with 6 components");
        });
        let stress = super::update::internal_to_c1_6(&internal);
        let factor = self.dynamic_factor(strain_rate);
        let ep = state.equiv_plastic_strain;
        let (new_stress, tangent, dep, yielding) = j2_return_c1(
            self.base.youngs_modulus,
            self.base.poisson_ratio,
            &stress,
            ep,
            d_eps,
            |e| self.base.current_yield_stress(e) * factor,
            |e| self.base.isotropic_hardening_slope(e) * factor,
        );
        StressUpdate {
            stress: new_stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }
}

impl StressUpdateModel for CowperSymondsJ2 {
    type State = PlasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.update_at_rate(state, d_eps, 0.0)
    }

    fn update_with_rate(
        &self,
        state: &Self::State,
        d_eps: &Self::StrainIncrement,
        dt: f64,
    ) -> StressUpdate {
        let rate = super::update::voigt_equivalent_increment_c1(d_eps) / dt.max(1e-15);
        self.update_at_rate(state, d_eps, rate)
    }

    fn is_rate_dependent(&self) -> bool {
        true
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        assert!(
            state.stress.len() == 6,
            "CowperSymondsJ2 commit needs a 3D PlasticState"
        );
        let internal = super::update::c1_to_internal_6(&update.stress);
        state.stress.copy_from_slice(&internal);
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
    }
}

/// Piecewise-linear isotropic plasticity from a hardening curve.
///
/// Interpolates the yield stress linearly between `(ep, sigma)` knots and
/// uses each segment slope as the hardening modulus. Beyond the last knot
/// the response holds the last stress (perfect plasticity). The knot list
/// must be non-empty with strictly increasing strains.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::PiecewiseLinearPlasticity;
///
/// let material = PiecewiseLinearPlasticity::mild_steel();
/// assert!((material.yield_stress(0.0) - 250.0e6).abs() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct PiecewiseLinearPlasticity {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Hardening curve knots `(ep, sigma)` with increasing strain.
    pub curve: Vec<[f64; 2]>,
}

impl PiecewiseLinearPlasticity {
    /// Creates a piecewise-linear model from moduli and a curve.
    ///
    /// # Panics
    ///
    /// Panics when the curve is empty or its strains do not increase.
    pub fn new(e: f64, nu: f64, curve: Vec<[f64; 2]>) -> Self {
        assert!(!curve.is_empty(), "piecewise-linear curve needs knots");
        for pair in curve.windows(2) {
            assert!(
                pair[1][0] > pair[0][0],
                "piecewise-linear strains must strictly increase"
            );
        }
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            curve,
        }
    }

    /// Mild steel with a two-segment hardening curve.
    pub fn mild_steel() -> Self {
        Self::new(
            210.0e9,
            0.3,
            vec![[0.0, 250.0e6], [0.02, 270.0e6], [0.10, 300.0e6]],
        )
    }

    /// Yield stress in Pa at plastic strain by linear interpolation.
    pub fn yield_stress(&self, ep: f64) -> f64 {
        let curve = &self.curve;
        if ep <= curve[0][0] {
            return curve[0][1];
        }
        for pair in curve.windows(2) {
            if ep <= pair[1][0] {
                let weight = (ep - pair[0][0]) / (pair[1][0] - pair[0][0]);
                return pair[0][1] + weight * (pair[1][1] - pair[0][1]);
            }
        }
        curve[curve.len() - 1][1]
    }

    /// Hardening slope in Pa at plastic strain (segment slope).
    pub fn hardening_slope(&self, ep: f64) -> f64 {
        let curve = &self.curve;
        for pair in curve.windows(2) {
            if ep < pair[1][0] {
                return (pair[1][1] - pair[0][1]) / (pair[1][0] - pair[0][0]);
            }
        }
        0.0
    }
}

impl StressUpdateModel for PiecewiseLinearPlasticity {
    type State = PlasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let internal: [f64; 6] = state.stress.as_slice().try_into().unwrap_or_else(|_| {
            panic!("PiecewiseLinearPlasticity needs a 3D PlasticState with 6 components");
        });
        let stress = super::update::internal_to_c1_6(&internal);
        let (new_stress, tangent, dep, yielding) = j2_return_c1(
            self.youngs_modulus,
            self.poisson_ratio,
            &stress,
            state.equiv_plastic_strain,
            d_eps,
            |e| self.yield_stress(e),
            |e| self.hardening_slope(e),
        );
        StressUpdate {
            stress: new_stress,
            tangent,
            delta_equiv_plastic_strain: dep,
            yielding,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        assert!(
            state.stress.len() == 6,
            "PiecewiseLinearPlasticity commit needs a 3D PlasticState"
        );
        let internal = super::update::c1_to_internal_6(&update.stress);
        state.stress.copy_from_slice(&internal);
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
    }
}

/// Deshpande-Fleck crushable foam and honeycomb model.
///
/// Yield function `Phi = sqrt(q^2 + alpha^2 p^2) / B - Y(ev)` with
/// `B = sqrt(1 + (alpha/3)^2)` and linear compaction hardening `Y =
/// y0 + H ev`. The shape factor `alpha` sets the hydrostatic strength:
/// `alpha -> 0` recovers von Mises, large `alpha` crushes under
/// hydrostatic load. Associative flow integrates along the fixed trial
/// direction (explicit-code standard); the tangent is the algorithmic
/// derivative of the update by central differences. The C1 equivalent
/// increment carries the volumetric compaction increment, which is the
/// hardening and erosion variable for foams.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::{CrushableFoamMaterial, CrushableFoamState};
/// use sonium_fem::physics::solid::plasticity::StressUpdateModel;
///
/// let foam = CrushableFoamMaterial::aluminum_foam();
/// let state = CrushableFoamState::new();
/// let d_eps = [-0.01, -0.01, -0.01, 0.0, 0.0, 0.0];
/// let update = foam.update(&state, &d_eps);
/// assert!(update.yielding);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrushableFoamMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Deshpande-Fleck shape factor, dimensionless.
    pub alpha: f64,
    /// Initial crush strength in Pa.
    pub initial_yield: f64,
    /// Compaction hardening modulus in Pa.
    pub hardening_modulus: f64,
    /// Compaction at the densification knee (`INFINITY` disables).
    pub lock_strain: f64,
    /// Extra hardening slope in Pa past the knee (0 disables).
    pub lock_modulus: f64,
}

impl CrushableFoamMaterial {
    /// Creates a crushable foam from moduli, shape, and hardening.
    ///
    /// The densification knee starts disabled; use
    /// [`with_densification`](CrushableFoamMaterial::with_densification)
    /// to enable it.
    pub fn new(e: f64, nu: f64, alpha: f64, initial_yield: f64, hardening_modulus: f64) -> Self {
        Self {
            youngs_modulus: e,
            poisson_ratio: nu,
            alpha,
            initial_yield,
            hardening_modulus,
            lock_strain: f64::INFINITY,
            lock_modulus: 0.0,
        }
    }

    /// Enables the densification knee at lock strain and modulus.
    ///
    /// Takes the knee compaction `c_lock` (dimensionless) and the
    /// post-knee slope `H_lock` (Pa); the crush strength gains
    /// `H_lock * max(0, c - c_lock)`.
    pub fn with_densification(mut self, lock_strain: f64, lock_modulus: f64) -> Self {
        self.lock_strain = lock_strain;
        self.lock_modulus = lock_modulus;
        self
    }

    /// Aluminum foam with moderate crush strength.
    pub fn aluminum_foam() -> Self {
        Self::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6)
    }

    /// Normalizer `B = sqrt(1 + (alpha/3)^2)`.
    pub fn normalizer(&self) -> f64 {
        (1.0 + (self.alpha / 3.0) * (self.alpha / 3.0)).sqrt()
    }

    /// Crush strength in Pa at compaction `ev`.
    ///
    /// `Y(c) = y0 + H c + H_lock max(0, c - c_lock)` with the
    /// compaction floored at 0 and the strength floored at 1e-9 Pa.
    /// Below the knee the response is the plain linear law; past the
    /// knee the lock modulus adds a steep densification leg. With the
    /// default (`c_lock = INFINITY`, `H_lock = 0`) the lock term is
    /// exactly 0 and the linear law is unchanged.
    pub fn crush_strength(&self, compaction: f64) -> f64 {
        let c = compaction.max(0.0);
        let lock = self.lock_modulus * (c - self.lock_strain).max(0.0);
        (self.initial_yield + self.hardening_modulus * c + lock).max(1e-9)
    }

    /// Yield function value (negative inside, zero on the surface).
    ///
    /// Takes pressure and von Mises stress in Pa plus compaction.
    pub fn yield_function(&self, p: f64, q: f64, compaction: f64) -> f64 {
        let s = (q * q + self.alpha * self.alpha * p * p).sqrt();
        s / self.normalizer() - self.crush_strength(compaction)
    }

    /// Shear modulus in Pa.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus in Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Stress-only update shared by the C1 impl and the tangent.
    ///
    /// Returns `(stress, compaction_increment, yielding)`.
    fn solve_stress(
        &self,
        stress: &[f64; 6],
        compaction: f64,
        d_eps: &[f64; 6],
    ) -> ([f64; 6], f64, bool) {
        let g = self.shear_modulus();
        let k = self.bulk_modulus();
        let vol = d_eps[0] + d_eps[1] + d_eps[2];
        let p_old = (stress[0] + stress[1] + stress[2]) / 3.0;
        let p_trial = p_old + k * vol;
        let mut s_trial = [0.0; 6];
        for i in 0..3 {
            s_trial[i] = stress[i] - p_old + 2.0 * g * (d_eps[i] - vol / 3.0);
        }
        for i in 3..6 {
            s_trial[i] = stress[i] + g * d_eps[i];
        }
        let trial_full = [
            s_trial[0] + p_trial,
            s_trial[1] + p_trial,
            s_trial[2] + p_trial,
            s_trial[3],
            s_trial[4],
            s_trial[5],
        ];
        let q_trial = von_mises_c1(&trial_full);
        let b = self.normalizer();
        let s_trial_norm = (q_trial * q_trial + self.alpha * self.alpha * p_trial * p_trial).sqrt();
        if s_trial_norm / b <= self.crush_strength(compaction) {
            return (trial_full, 0.0, false);
        }
        // Fixed-direction associative return from the trial state.
        let denom = (b * s_trial_norm).max(1e-30);
        let dir_q = q_trial / denom;
        let dir_p = self.alpha * self.alpha * p_trial / denom;
        let three_g = 3.0 * g;
        // Compaction rate along the direction (positive in compression).
        let compaction_rate = -dir_p;
        let residual = |lambda: f64| {
            let p = p_trial - k * lambda * dir_p;
            let q = (q_trial - three_g * lambda * dir_q).max(0.0);
            let ev = compaction + lambda * compaction_rate;
            (q * q + self.alpha * self.alpha * p * p).sqrt() / b - self.crush_strength(ev)
        };
        // The residual is V-shaped (the return path crosses p = 0 from
        // far trial states), so coarse bracket doubling can step over the
        // valley. Scan finely for the first crossing, then bisect: the
        // physical root always precedes the s-minimum because F decreases
        // monotonically until the return path bottoms out.
        let aq = three_g * dir_q;
        let ap = k * dir_p;
        let a_quad = aq * aq + self.alpha * self.alpha * ap * ap;
        let stiffness = (three_g * q_trial * q_trial + k * self.alpha.powi(4) * p_trial * p_trial)
            / denom.max(1e-30)
            + self.hardening_modulus * compaction_rate.abs();
        let lam_smin = if a_quad > 0.0 {
            (q_trial * aq + self.alpha * self.alpha * p_trial * ap) / a_quad
        } else {
            0.0
        };
        let f0 = residual(0.0);
        let lam_max = (2.0 * lam_smin)
            .max(4.0 * f0 / stiffness.max(1e-30))
            .max(1e-12);
        let scan = 4000;
        let mut lo = 0.0;
        let mut hi = lam_max;
        let mut best = (f0, 0.0);
        let mut bracketed = false;
        for i in 1..=scan {
            let lam = lam_max * i as f64 / scan as f64;
            let f = residual(lam);
            if f < best.0 {
                best = (f, lam);
            }
            if !bracketed && f <= 0.0 {
                lo = lam_max * (i - 1) as f64 / scan as f64;
                hi = lam;
                bracketed = true;
            }
        }
        let lambda = if bracketed {
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if residual(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        } else {
            // No crossing (pathological hardening): closest approach.
            best.1
        };
        let p_new = p_trial - k * lambda * dir_p;
        let q_new = (q_trial - three_g * lambda * dir_q).max(0.0);
        let ratio = if q_trial > 1e-30 {
            q_new / q_trial
        } else {
            0.0
        };
        let new_stress = [
            ratio * s_trial[0] + p_new,
            ratio * s_trial[1] + p_new,
            ratio * s_trial[2] + p_new,
            ratio * s_trial[3],
            ratio * s_trial[4],
            ratio * s_trial[5],
        ];
        (new_stress, lambda * compaction_rate, true)
    }
}

/// History state of one crushable foam point.
#[derive(Debug, Clone)]
pub struct CrushableFoamState {
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Volumetric compaction, positive in compression.
    pub compaction: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl CrushableFoamState {
    /// Creates a virgin foam state.
    pub fn new() -> Self {
        Self {
            stress: [0.0; 6],
            compaction: 0.0,
            yielding: false,
        }
    }
}

impl Default for CrushableFoamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for CrushableFoamMaterial {
    type State = CrushableFoamState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let (stress, dev, yielding) = self.solve_stress(&state.stress, state.compaction, d_eps);
        let tangent = if yielding {
            // Algorithmic tangent by central differences of the update.
            // Explicit dynamics uses the elastic stiffness instead; the
            // differentiated tangent serves quasi-static Newton loops.
            let eps = 1e-9;
            let mut tangent = [0.0; 36];
            for j in 0..6 {
                let mut plus = *d_eps;
                let mut minus = *d_eps;
                plus[j] += eps;
                minus[j] -= eps;
                let (s_plus, _, _) = self.solve_stress(&state.stress, state.compaction, &plus);
                let (s_minus, _, _) = self.solve_stress(&state.stress, state.compaction, &minus);
                for i in 0..6 {
                    tangent[i * 6 + j] = (s_plus[i] - s_minus[i]) / (2.0 * eps);
                }
            }
            // Enforce major symmetry lost to differencing noise.
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let mean = 0.5 * (tangent[i * 6 + j] + tangent[j * 6 + i]);
                    tangent[i * 6 + j] = mean;
                    tangent[j * 6 + i] = mean;
                }
            }
            tangent
        } else {
            elastic_tangent_c1(self.bulk_modulus(), self.shear_modulus())
        };
        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: dev.max(0.0),
            yielding,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.compaction += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;
    }
}

/// One point of a driven uniaxial stress-strain curve.
#[derive(Debug, Clone, Copy)]
pub struct UniaxialPoint {
    /// Total axial strain, dimensionless.
    pub strain_xx: f64,
    /// Axial Cauchy stress in Pa.
    pub stress_xx: f64,
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Whether the step ended yielding.
    pub yielding: bool,
}

/// Drives a C1 model through uniaxial stress tension or compression.
///
/// Advances `target_xx` total axial strain in `steps` increments, solving
/// the lateral strains for zero transverse stress by Newton iteration on
/// the tangent sub-block each step. Returns the stress-strain curve
/// including the initial point. Rate-dependent models run at the
/// reference rate here; use [`drive_uniaxial_stress_rate`] for imposed
/// rates. Tolerance scales with the axial tangent stiffness.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::crash::{drive_uniaxial_stress, PiecewiseLinearPlasticity};
/// use sonium_fem::physics::solid::plasticity::PlasticState;
///
/// let material = PiecewiseLinearPlasticity::mild_steel();
/// let mut state = PlasticState::new_3d();
/// let curve = drive_uniaxial_stress(&material, &mut state, 0.05, 50);
/// assert!(curve.last().unwrap().stress_xx > 250.0e6);
/// ```
pub fn drive_uniaxial_stress<M>(
    material: &M,
    state: &mut M::State,
    target_xx: f64,
    steps: usize,
) -> Vec<UniaxialPoint>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    drive_uniaxial_stress_rate(material, state, target_xx, steps, None)
}

/// Drives a C1 model through uniaxial stress at an imposed rate.
///
/// Identical to [`drive_uniaxial_stress`] but evaluates rate-dependent
/// models with [`update_with_rate`](StressUpdateModel::update_with_rate)
/// at `dt` seconds per step. Rate-independent models ignore `dt` and
/// behave exactly like [`drive_uniaxial_stress`].
///
/// # Panics
///
/// Panics when `steps` is zero or `dt` is not positive.
pub fn drive_uniaxial_stress_rate<M>(
    material: &M,
    state: &mut M::State,
    target_xx: f64,
    steps: usize,
    dt: Option<f64>,
) -> Vec<UniaxialPoint>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    assert!(steps > 0, "uniaxial driver needs at least one step");
    if let Some(dt) = dt {
        assert!(dt > 0.0, "uniaxial driver needs a positive step time");
    }
    let trial = |state: &M::State, d_eps: &[f64; 6]| match dt {
        Some(dt) => material.update_with_rate(state, d_eps, dt),
        None => material.update(state, d_eps),
    };
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
        let mut d_lat = lateral;
        let mut update = trial(state, &[dx, d_lat[0], d_lat[1], 0.0, 0.0, 0.0]);
        for _ in 0..25 {
            let scale = update.tangent[0].abs().max(1.0);
            if update.stress[1].abs() < 1e-9 * scale && update.stress[2].abs() < 1e-9 * scale {
                break;
            }
            // 2x2 Newton on the transverse block [[C11,C12],[C21,C22]].
            let a = update.tangent[7];
            let b = update.tangent[8];
            let c = update.tangent[13];
            let d = update.tangent[14];
            let det = a * d - b * c;
            if det.abs() < 1e-30 {
                break;
            }
            let r0 = update.stress[1];
            let r1 = update.stress[2];
            d_lat[0] -= (d * r0 - b * r1) / det;
            d_lat[1] -= (a * r1 - c * r0) / det;
            update = trial(state, &[dx, d_lat[0], d_lat[1], 0.0, 0.0, 0.0]);
        }
        lateral = d_lat;
        ep += update.delta_equiv_plastic_strain;
        let yielding = update.yielding;
        let sx = update.stress[0];
        material.commit(state, &update);
        curve.push(UniaxialPoint {
            strain_xx: dx * step as f64,
            stress_xx: sx,
            equiv_plastic_strain: ep,
            yielding,
        });
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::super::material::HardeningLaw;
    use super::*;

    #[test]
    fn jc_quasistatic_uniaxial_follows_closed_form() {
        let material = JohnsonCookMaterial::c4340_steel();
        let mut state = JohnsonCookState::new(293.0);
        let curve = drive_uniaxial_stress(&material, &mut state, 0.05, 100);
        let last = curve.last().unwrap();
        assert!(last.yielding);
        let expected = material.flow_stress(last.equiv_plastic_strain, 1.0, 293.0);
        assert!(
            (last.stress_xx - expected).abs() / expected < 0.01,
            "stress {} vs closed form {expected}",
            last.stress_xx
        );
    }

    #[test]
    fn jc_rate_hardening_matches_factor() {
        let material = JohnsonCookMaterial::c4340_steel();
        let mut slow_state = JohnsonCookState::new(293.0);
        // One slow step and one fast step from virgin states.
        let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let slow = material.update(&slow_state, &d_eps);
        let fast = material.update_with_rate(&slow_state, &d_eps, 1.0e-5);
        assert!(material.is_rate_dependent());
        assert!(fast.yielding && slow.yielding);
        let rate = super::super::update::voigt_equivalent_increment_c1(&d_eps) / 1.0e-5;
        let factor = material.rate_factor(rate);
        let got = von_mises_c1(&fast.stress) / von_mises_c1(&slow.stress);
        assert!(
            (got - factor).abs() / factor < 0.02,
            "got {got}, want {factor}"
        );
        material.commit(&mut slow_state, &slow);
    }

    #[test]
    fn jc_temperature_softens_to_zero_at_melt() {
        let material = JohnsonCookMaterial::c4340_steel();
        assert!((material.thermal_factor(293.0) - 1.0).abs() < 1e-15);
        assert!(material.flow_stress(0.01, 1.0, 1793.0) < 1.0);
        let warm = material.flow_stress(0.01, 1.0, 900.0);
        let cold = material.flow_stress(0.01, 1.0, 293.0);
        assert!(warm < cold && warm > 0.0);
    }

    #[test]
    fn jc_damage_accumulates_to_fracture() {
        let mut material = JohnsonCookMaterial::c4340_steel();
        material.damage = Some(JohnsonCookDamage::c4340_steel());
        let mut state = JohnsonCookState::new(293.0);
        // Proportional tensile straining accumulates damage to the clamp.
        // The imposed path carries residual transverse compression from
        // the elastic-plastic Poisson mismatch, so fracture lands above
        // the ideal uniaxial ef(1/3); exact ef math is covered by the
        // ductile unit tests and the C2 golden instead.
        for _ in 0..2000 {
            let update = material.update(&state, &[0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0]);
            material.commit(&mut state, &update);
            material.accumulate_damage(&mut state, &update, 1.0);
            if state.damage >= 1.0 {
                break;
            }
        }
        assert!(
            (state.damage - 1.0).abs() < 1e-12,
            "damage {}",
            state.damage
        );
        assert!(
            (1.0..8.0).contains(&state.equiv_plastic_strain),
            "fracture strain {} outside 4340 bounds",
            state.equiv_plastic_strain
        );
    }

    #[test]
    fn jc_damage_absent_is_noop() {
        let material = JohnsonCookMaterial::c4340_steel();
        let mut state = JohnsonCookState::new(293.0);
        let update = material.update(&state, &[0.01, -0.003, -0.003, 0.0, 0.0, 0.0]);
        material.commit(&mut state, &update);
        material.accumulate_damage(&mut state, &update, 1.0);
        assert_eq!(state.damage, 0.0);
    }

    #[test]
    fn jc_update_is_pure_and_adiabatic_commit_heats() {
        let mut material = JohnsonCookMaterial::c4340_steel();
        material.density = 7830.0;
        material.specific_heat = 475.0;
        let state = JohnsonCookState::new(293.0);
        let d_eps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let before = state.clone();
        let update = material.update(&state, &d_eps);
        assert_eq!(state.stress, before.stress);
        assert_eq!(state.temperature, before.temperature);
        let mut committed = state.clone();
        material.commit(&mut committed, &update);
        assert!(committed.temperature > 293.0);
    }

    #[test]
    fn cowper_symonds_matches_dynamic_factor() {
        let scaled = CowperSymondsJ2::mild_steel();
        assert!((scaled.dynamic_factor(0.0) - 1.0).abs() < 1e-15);
        let expected = 1.0 + (100.0_f64 / 40.0).powf(1.0 / 5.0);
        assert!((scaled.dynamic_factor(100.0) - expected).abs() < 1e-12);
        // Driven fast response exceeds quasi-static by the factor.
        let mut slow_state = PlasticState::new_3d();
        let mut fast_state = PlasticState::new_3d();
        let slow_curve = drive_uniaxial_stress(&scaled, &mut slow_state, 0.02, 40);
        let fast_curve =
            drive_uniaxial_stress_rate(&scaled, &mut fast_state, 0.02, 40, Some(2.0e-5));
        let slow_last = slow_curve.last().unwrap();
        let fast_last = fast_curve.last().unwrap();
        assert!(fast_last.stress_xx > 1.05 * slow_last.stress_xx);
    }

    #[test]
    fn piecewise_linear_hits_knots() {
        let material = PiecewiseLinearPlasticity::mild_steel();
        assert!((material.yield_stress(0.02) - 270.0e6).abs() < 1.0);
        assert!((material.yield_stress(0.05) - 281.25e6).abs() < 1.0);
        assert!((material.hardening_slope(0.05) - 375.0e6).abs() < 1.0);
        let mut state = PlasticState::new_3d();
        let curve = drive_uniaxial_stress(&material, &mut state, 0.06, 120);
        let last = curve.last().unwrap();
        let expected = material.yield_stress(last.equiv_plastic_strain);
        assert!(
            (last.stress_xx - expected).abs() / expected < 0.01,
            "stress {} vs curve {expected}",
            last.stress_xx
        );
    }

    #[test]
    fn foam_hydrostatic_crush_matches_closed_form() {
        let foam = CrushableFoamMaterial::aluminum_foam();
        let mut state = CrushableFoamState::new();
        // Hydrostatic compression in small steps tracks the crush curve.
        for _ in 0..50 {
            let d_eps = [-0.001, -0.001, -0.001, 0.0, 0.0, 0.0];
            let update = foam.update(&state, &d_eps);
            foam.commit(&mut state, &update);
        }
        assert!(state.yielding);
        let p = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0;
        let expected_p = -foam.crush_strength(state.compaction) * foam.normalizer() / foam.alpha;
        assert!(
            (p - expected_p).abs() / expected_p.abs() < 0.02,
            "pressure {p} vs closed form {expected_p}"
        );
    }

    #[test]
    fn foam_tangent_is_symmetric_and_softer() {
        let foam = CrushableFoamMaterial::aluminum_foam();
        let state = CrushableFoamState::new();
        let d_eps = [-0.02, -0.02, -0.02, 0.0, 0.0, 0.0];
        let update = foam.update(&state, &d_eps);
        assert!(update.yielding);
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert!(
                    (update.tangent[i * 6 + j] - update.tangent[j * 6 + i]).abs() < 1e-6,
                    "tangent asymmetric at ({i},{j})"
                );
            }
        }
        let elastic = elastic_tangent_c1(foam.bulk_modulus(), foam.shear_modulus());
        assert!(update.tangent[0] < elastic[0]);
    }

    #[test]
    fn thermal_seeds_heat_but_keep_flow() {
        for material in [
            JohnsonCookMaterial::c4340_steel_thermal(),
            JohnsonCookMaterial::ofhc_copper_thermal(),
        ] {
            // Thermal seeds share the dry flow response exactly.
            let dry = if material.a > 500.0e6 {
                JohnsonCookMaterial::c4340_steel()
            } else {
                JohnsonCookMaterial::ofhc_copper()
            };
            let got = material.flow_stress(0.01, 1.0, 293.0);
            let want = dry.flow_stress(0.01, 1.0, 293.0);
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "flow differs: {got} vs {want}"
            );
            // One plastic step warms the coupled state.
            let mut state = JohnsonCookState::new(293.0);
            let update = material.update(&state, &[0.01, -0.003, -0.003, 0.0, 0.0, 0.0]);
            assert!(update.yielding);
            material.commit(&mut state, &update);
            assert!(state.temperature > 293.0);
        }
    }

    #[test]
    fn driver_recovers_j2_yield() {
        let material = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        // Drive through the crate J2 adapter path via Cowper-Symonds base.
        let wrapped = CowperSymondsJ2::new(material, 1.0e30, 5.0);
        let mut state = PlasticState::new_3d();
        let curve = drive_uniaxial_stress(&wrapped, &mut state, 0.01, 50);
        let last = curve.last().unwrap();
        assert!(
            (last.stress_xx - 250.0e6).abs() / 250.0e6 < 0.01,
            "stress {}",
            last.stress_xx
        );
    }

    #[test]
    fn crush_strength_rises_steeply_past_densification_knee() {
        let foam = CrushableFoamMaterial::aluminum_foam().with_densification(0.3, 1.0e9);
        // Below the knee: plain linear law y0 + H c.
        let below = foam.crush_strength(0.1);
        assert!((below - (5.0e6 + 50.0e6 * 0.1)).abs() < 1.0);
        // At the knee the lock leg starts continuously.
        let knee = foam.crush_strength(0.3);
        assert!((knee - (5.0e6 + 50.0e6 * 0.3)).abs() < 1.0);
        // Far above the knee the lock modulus dominates.
        let above = foam.crush_strength(0.5);
        assert!((above - (5.0e6 + 50.0e6 * 0.5 + 1.0e9 * 0.2)).abs() < 1.0);
        assert!(above > 10.0 * below);
        // Disabled by default: lock term is exactly zero.
        let plain = CrushableFoamMaterial::aluminum_foam();
        assert!((plain.crush_strength(0.5) - (5.0e6 + 50.0e6 * 0.5)).abs() < 1.0);
    }
}
