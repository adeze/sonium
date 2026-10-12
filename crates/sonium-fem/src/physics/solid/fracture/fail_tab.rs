//! Tabulated triaxiality failure with damage accumulation (`/FAIL/TAB1`).
//!
//! Implements the OpenCourant tabulated failure model: the plastic failure
//! strain comes from user tables over stress triaxiality, strain rate,
//! and Lode angle, scaled by element-size and temperature factors, and
//! damage accumulates per step until it exceeds the critical value.
//! Equations follow Reference Guide `/FAIL/TAB1` comments 3-5 and the
//! Theory Manual failure-model chapter.
//!
//! - Failure strain `ef = Yscale . Table(eta, rate/Xscale, xi) . fel . fT`
//!   (Reference Guide (951)); size factor (952) applies only inside the
//!   triaxiality window (953); temperature factor uses (954)-(955).
//! - Below the first tabulated rate the first curve holds; above the last
//!   rate the value extrapolates linearly from the last two rows
//!   (Reference Guide `/FAIL/TAB1` comment 3). Triaxiality ends clamp
//!   through the frozen [`Funct1D`](super::super::plasticity::funct::Funct1D)
//!   reader, the OpenCourant default.
//! - Damage with parameter input is `dD = (dep/ef) . n . Dp^(1-1/n)`
//!   (Reference Guide (959)); with curve input `dD = (dep/ef) . fd(ep)`
//!   ((960)). Rupture trips when accumulated damage exceeds `Dcrit` ((956)).
//! - Strain rates are per second (SI-consistent). The legacy `/FAIL/TAB`
//!   deck card is a deprecated alias of `/FAIL/TAB1` (Reference Guide
//!   conversion table, v13.0), so the same equations cover it.
//!
//! # Example
//!
//! ```rust
//! use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
//! use sonium_fem::physics::solid::plasticity::funct::Funct1D;
//!
//! let low = Funct1D::new(vec![0.0, 1.0], vec![0.15, 0.18]).unwrap();
//! let high = Funct1D::new(vec![0.0, 1.0], vec![0.30, 0.36]).unwrap();
//! let tab = FailTab::new(vec![1.0e-5, 0.5], vec![0.0], vec![low, high]).unwrap();
//! let ef = tab.failure_strain(1.0 / 3.0, 0.5, 0.0, None, None);
//! assert!((ef - 0.32).abs() < 1e-12);
//! ```

// Rust guideline compliant 2026-02-21

use crate::physics::solid::plasticity::funct::Funct1D;
use crate::physics::solid::plasticity::update::StressUpdate;

/// Floor for the failure strain in damage accumulation.
///
/// Keeps `dep/ef` finite under extreme compression tables; matches the
/// Johnson-Cook damage floor in
/// [`ductile`](super::ductile::JohnsonCookDamage::fracture_strain) so both
/// failure legs treat degenerate loci identically.
const FAILURE_STRAIN_FLOOR: f64 = 1e-9;

/// Floor for positive step times when estimating strain rate.
///
/// Mirrors the `dt` guard in
/// [`JohnsonCookMaterial`](super::super::plasticity::crash::JohnsonCookMaterial)
/// rate updates so a zero step time yields a finite quasi-static rate.
const STEP_TIME_FLOOR: f64 = 1e-15;

/// Plastic failure strain table over triaxiality, rate, and Lode angle.
///
/// Holds one [`Funct1D`] failure-strain-vs-triaxiality curve per
/// (rate, Lode) grid node in row-major order
/// (`curves[rate * nlode + lode]`), the `/TABLE/1` layout where the first
/// variable is triaxiality, the second strain rate, and the third the Lode
/// parameter (Reference Guide `/FAIL/TAB1` comment 8). Shell tables carry
/// a single Lode node, which reduces evaluation to triaxiality-by-rate.
#[derive(Debug, Clone)]
pub struct FailTab {
    /// Strictly increasing strain-rate grid in 1/s.
    rates: Vec<f64>,
    /// Strictly increasing Lode-parameter grid, dimensionless.
    lodes: Vec<f64>,
    /// Failure-strain curves, one per grid node.
    curves: Vec<Funct1D>,
    /// Scale factor on the tabulated failure strain (`Yscale1`).
    pub y_scale: f64,
    /// Scale factor on the rate abscissa (`Xscale1`).
    pub x_scale: f64,
    /// Critical accumulated damage (`Dcrit`, default 0.999).
    pub dcrit: f64,
    /// Damage accumulation parameter (`Dp`, default 1.0).
    pub dp: f64,
    /// Damage accumulation exponent (`n`, default 1.0).
    pub exponent: f64,
    /// Optional damage-scale curve `fd(ep)` (`fct_IDd`, (960)).
    pub damage_curve: Option<Funct1D>,
    /// Optional element-size factor curve (`fct_IDel`, (952)).
    pub size_curve: Option<Funct1D>,
    /// Scale on the element-size factor (`Fscaleel`).
    pub size_scale: f64,
    /// Reference element size (`El_ref`).
    pub size_ref: f64,
    /// Shear triaxiality limit for size scaling (`Shrf`).
    pub size_lo: f64,
    /// Biaxial triaxiality limit for size scaling (`Biaxf`).
    pub size_hi: f64,
    /// Optional temperature factor curve (`fct_IDT`, (954)).
    pub temp_curve: Option<Funct1D>,
    /// Scale on the temperature factor (`FscaleT`).
    pub temp_scale: f64,
}

impl FailTab {
    /// Builds a tabulated failure model from a rate-Lode grid.
    ///
    /// Takes strictly increasing rate (1/s) and Lode grids plus one
    /// triaxiality curve per grid node in row-major order. Scales default
    /// to 1, `Dcrit` to 0.999, `Dp` and `n` to 1.0, and all optional
    /// curves to `None` (neutral factors).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    ///
    /// let curve = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
    /// let tab = FailTab::new(vec![1.0e-5], vec![0.0], vec![curve]).unwrap();
    /// let ef = tab.failure_strain(0.0, 0.0, 0.0, None, None);
    /// assert!((ef - 0.5).abs() < 1e-15);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a message when either grid is empty, non-finite, or not
    /// strictly increasing, or the curve count misses `nrate * nlode`.
    pub fn new(rates: Vec<f64>, lodes: Vec<f64>, curves: Vec<Funct1D>) -> Result<Self, String> {
        if rates.is_empty() || lodes.is_empty() {
            return Err("fail-tab needs at least one rate and one Lode node".to_string());
        }
        if !rates.iter().chain(lodes.iter()).all(|v| v.is_finite()) {
            return Err("fail-tab grids must be finite".to_string());
        }
        if rates.windows(2).any(|w| w[1] <= w[0]) {
            return Err("fail-tab rates must strictly increase".to_string());
        }
        if lodes.windows(2).any(|w| w[1] <= w[0]) {
            return Err("fail-tab Lode nodes must strictly increase".to_string());
        }
        if curves.len() != rates.len() * lodes.len() {
            return Err(format!(
                "fail-tab needs {} curves, got {}",
                rates.len() * lodes.len(),
                curves.len()
            ));
        }
        if rates.iter().any(|r| *r < 0.0) {
            return Err("fail-tab rates must be non-negative".to_string());
        }
        Ok(Self {
            rates,
            lodes,
            curves,
            y_scale: 1.0,
            x_scale: 1.0,
            dcrit: 0.999,
            dp: 1.0,
            exponent: 1.0,
            damage_curve: None,
            size_curve: None,
            size_scale: 1.0,
            size_ref: 1.0,
            size_lo: -1.0,
            size_hi: 1.0,
            temp_curve: None,
            temp_scale: 1.0,
        })
    }

    /// ELEM_SAMP deck failure table for rupture validation.
    ///
    /// Bakes the miniqa `RUPTURE/FAIL_TAB/ELEM_SAMP` `/FUNCT/20` failure
    /// strain vs triaxiality curve with the deck rate rows
    /// (`Fscale` 0.5 at 1e-5/s, 1.0 at 0.5/s) and `Dcrit` 1.1, `Dp` 1,
    /// `n` 1. Deck size and temperature functions are flat 1.0, so both
    /// scale curves stay `None`.
    ///
    /// # Panics
    ///
    /// Panics when the baked deck nodes violate grid validation; the
    /// nodes are fixed literals proven by the constructor tests.
    pub fn elem_samp_steel() -> Self {
        // Deck FUNCT/20 pairs (triaxiality, failure strain).
        let nodes: [[f64; 2]; 28] = [
            [-1.0, 3.088],
            [-0.666666667, 3.088],
            [-0.57735, 2.173],
            [-0.5, 1.604],
            [-0.45, 1.319],
            [-0.4, 1.086],
            [-0.35, 0.894],
            [-0.3333333, 0.838],
            [-0.3, 0.737],
            [-0.25, 0.608],
            [-0.2, 0.502],
            [-0.15, 0.416],
            [-0.1, 0.344],
            [-0.05, 0.286],
            [0.0, 0.300],
            [0.05, 0.320],
            [0.1, 0.330],
            [0.15, 0.340],
            [0.2, 0.350],
            [0.25, 0.355],
            [0.3, 0.360],
            [0.3333333, 0.360],
            [0.35, 0.340],
            [0.4, 0.290],
            [0.45, 0.200],
            [0.5, 0.100],
            [0.57735, 0.042],
            [0.666666667, 0.050],
        ];
        // Deck tail node at triaxiality 1.0 extends the flat end.
        let mut x: Vec<f64> = nodes.iter().map(|pair| pair[0]).collect();
        let mut y: Vec<f64> = nodes.iter().map(|pair| pair[1]).collect();
        x.push(1.0);
        y.push(0.050);
        let base = Funct1D::new(x, y).expect("deck funct nodes are valid");
        let curves = vec![base.scaled_ordinates(0.5), base];
        let mut tab = Self::new(vec![1.0e-5, 0.5], vec![0.0], curves).expect("deck grid is valid");
        tab.dcrit = 1.1;
        tab
    }

    /// Interpolates the tabulated failure strain without scale factors.
    ///
    /// Blends the rate bracket linearly (Reference Guide `/FAIL/TAB1`
    /// comment 3): rates below the first row hold the first curve, rates
    /// above the last row extrapolate linearly from the last two rows,
    /// and Lode blends linearly with clamped ends.
    pub fn table_strain(&self, triaxiality: f64, rate: f64, lode: f64) -> f64 {
        let nl = self.lodes.len();
        let nr = self.rates.len();
        let (l0, l1, fl) = bracket(&self.lodes, lode);
        let value = |ri: usize, li: usize| self.curves[ri * nl + li].eval(triaxiality);
        let at_rate = |ri: usize| value(ri, l0) + fl * (value(ri, l1) - value(ri, l0));
        if nr == 1 || rate <= self.rates[0] {
            return at_rate(0);
        }
        let last = nr - 1;
        if rate >= self.rates[last] && nr >= 2 {
            let span = self.rates[last] - self.rates[last - 1];
            let slope = (at_rate(last) - at_rate(last - 1)) / span;
            return at_rate(last) + slope * (rate - self.rates[last]);
        }
        let upper = self.rates.partition_point(|node| *node <= rate);
        let lo = upper - 1;
        let frac = (rate - self.rates[lo]) / (self.rates[upper] - self.rates[lo]);
        at_rate(lo) + frac * (at_rate(upper) - at_rate(lo))
    }

    /// Element-size scale factor from (952)-(953).
    ///
    /// Returns `Fscaleel . fel(size/El_ref)` inside the open triaxiality
    /// window (`Shrf`, `Biaxf`) and 1.0 outside it or when no size curve
    /// is set. The reference size is floored at [`STEP_TIME_FLOOR`]-scale
    /// to keep the ratio finite for degenerate cards.
    pub fn size_factor(&self, triaxiality: f64, size: f64) -> f64 {
        let Some(curve) = &self.size_curve else {
            return 1.0;
        };
        if triaxiality <= self.size_lo || triaxiality >= self.size_hi {
            return 1.0;
        }
        self.size_scale * curve.eval(size / self.size_ref.max(STEP_TIME_FLOOR))
    }

    /// Temperature scale factor from (954)-(955).
    ///
    /// Evaluates `FscaleT . fT(T*)` with the normalized temperature
    /// `T* = (T - Tini)/(Tmelt - Tini)`; returns 1.0 when no temperature
    /// curve is set. A degenerate melt span clamps `T*` to zero.
    pub fn temp_factor(&self, temperature: f64, temp_init: f64, temp_melt: f64) -> f64 {
        let Some(curve) = &self.temp_curve else {
            return 1.0;
        };
        let span = temp_melt - temp_init;
        let star = if span > 0.0 {
            (temperature - temp_init) / span
        } else {
            0.0
        };
        self.temp_scale * curve.eval(star)
    }

    /// Scaled plastic failure strain from (951).
    ///
    /// Multiplies the table value (evaluated at `rate/x_scale`) by the
    /// ordinate scale and the element-size and temperature factors.
    /// Callers without size or temperature data pass `None` for either,
    /// which selects the neutral factor of 1.0.
    pub fn failure_strain(
        &self,
        triaxiality: f64,
        rate: f64,
        lode: f64,
        size: Option<f64>,
        temperature: Option<(f64, f64, f64)>,
    ) -> f64 {
        let table = self.table_strain(triaxiality, rate / self.x_scale.max(STEP_TIME_FLOOR), lode);
        let fel = size.map_or(1.0, |s| self.size_factor(triaxiality, s));
        let ft = temperature.map_or(1.0, |(t, ti, tm)| self.temp_factor(t, ti, tm));
        (self.y_scale * table * fel * ft).max(FAILURE_STRAIN_FLOOR)
    }

    /// Damage increment at an explicit strain rate.
    ///
    /// Core of the P3C3 criterion: divides the plastic increment by the
    /// scaled failure strain, then applies (959) with parameter input or
    /// (960) with a damage curve. Non-positive increments yield zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    ///
    /// let curve = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
    /// let tab = FailTab::new(vec![0.0], vec![0.0], vec![curve]).unwrap();
    /// let step = tab.increment_at_rate(0.01, 0.05, 0.0, 0.0, 0.0, None, None);
    /// assert!((step - 0.02).abs() < 1e-15);
    /// ```
    pub fn increment_at_rate(
        &self,
        dep: f64,
        ep: f64,
        triaxiality: f64,
        lode: f64,
        rate: f64,
        size: Option<f64>,
        temperature: Option<(f64, f64, f64)>,
    ) -> f64 {
        if dep <= 0.0 {
            return 0.0;
        }
        let ef = self.failure_strain(triaxiality, rate, lode, size, temperature);
        let base = dep / ef;
        if let Some(curve) = &self.damage_curve {
            return base * curve.eval(ep.max(0.0));
        }
        let n = self.exponent;
        if n <= 0.0 {
            return base;
        }
        base * n * self.dp.max(0.0).powf(1.0 - 1.0 / n)
    }

    /// Damage increment from a stress update (P3C3 criterion).
    ///
    /// Reads the plastic increment from `update` and estimates the strain
    /// rate as `dep/dt`, the explicit-code convention shared with the
    /// Johnson-Cook rate update. Size and temperature scales stay neutral;
    /// use [`increment_at_rate`](Self::increment_at_rate) with explicit
    /// context when the deck defines those functions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::fail_tab::FailTab;
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    /// use sonium_fem::physics::solid::plasticity::update::StressUpdate;
    ///
    /// let curve = Funct1D::new(vec![-1.0, 1.0], vec![0.5, 0.5]).unwrap();
    /// let tab = FailTab::new(vec![0.0], vec![0.0], vec![curve]).unwrap();
    /// let update = StressUpdate {
    ///     stress: [300.0e6, 0.0, 0.0, 0.0, 0.0, 0.0],
    ///     tangent: [0.0; 36],
    ///     delta_equiv_plastic_strain: 0.01,
    ///     yielding: true,
    /// };
    /// let step = tab.increment(&update, 0.05, 1.0 / 3.0, 0.0, 1.0e-3);
    /// assert!((step - 0.02).abs() < 1e-15);
    /// ```
    pub fn increment(
        &self,
        update: &StressUpdate,
        ep: f64,
        triaxiality: f64,
        lode: f64,
        dt: f64,
    ) -> f64 {
        let dep = update.delta_equiv_plastic_strain;
        let rate = dep / dt.max(STEP_TIME_FLOOR);
        self.increment_at_rate(dep, ep, triaxiality, lode, rate.max(0.0), None, None)
    }

    /// Reports whether accumulated damage ruptures the point.
    ///
    /// Rupture trips on strict `damage > Dcrit` (Reference Guide (956)).
    pub fn ruptured(&self, damage: f64) -> bool {
        damage > self.dcrit
    }
}

/// Fading-exponent softening after diffuse necking (`Fad_exp`).
///
/// Mirrors the `/FAIL/TAB1` instability treatment (Reference Guide eq
/// 962 and `fail_tab_c.F`): past the necking onset the reported stress
/// scales by `1 - ((u - dd)/(1 - dd))^fad`, where `u` runs from the
/// onset fraction `dd` at necking to 1 at fracture. Only shells neck
/// in the vendor model; the native step applies wherever the caller
/// drives it and documents the scope.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::fracture::fail_tab::{FadeSoftening, FadeState};
///
/// let fade = FadeSoftening::new(5.0, 0.2);
/// let state = FadeState { neck: 1.0, soft: 0.5 };
/// let scale = fade.scale_only(&state);
/// assert!(scale < 1.0 && scale > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct FadeSoftening {
    /// Fading exponent (card `Fad_exp`; guide recommends 5-10).
    ///
    /// Zero disables softening (scale stays 1). A negative value
    /// reads the exponent from [`exponent_curve`](Self::exponent_curve)
    /// at the element-size ratio, else falls back to its magnitude.
    pub exponent: f64,
    /// Instability strain (card `inst_start`; default is `Dp`).
    ///
    /// Constant necking strain over triaxiality when
    /// [`neck_curve`](Self::neck_curve) is unset.
    pub neck_strain: f64,
    /// Softening-start fraction (card `D`, starter maps 1 to 0.999).
    pub onset_fraction: f64,
    /// Instability strain vs triaxiality (`table2_ID`), optional.
    pub neck_curve: Option<Funct1D>,
    /// Exponent vs size ratio (negative `Fad_exp`), optional.
    pub exponent_curve: Option<Funct1D>,
}

/// Post-instability progress (engine UVAR slots 3-4).
#[derive(Debug, Clone, Copy, Default)]
pub struct FadeState {
    /// Necking accumulation `sum w dep/ef_neck` (UVAR3).
    ///
    /// Softening starts once this reaches 1.
    pub neck: f64,
    /// Post-onset fraction `sum w dep/(ef - ef_neck)` (UVAR4).
    pub soft: f64,
}

impl FadeSoftening {
    /// Builds fading softening from exponent and necking strain.
    ///
    /// Sets the onset fraction to 0.999, the starter-mapped value of
    /// the card default `D = 1` (`hm_read_fail_tab1.F`), and leaves
    /// both optional curves unset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::fail_tab::FadeSoftening;
    ///
    /// let fade = FadeSoftening::new(5.0, 0.2);
    /// assert!((fade.onset_fraction - 0.999).abs() < 1e-15);
    /// ```
    pub fn new(exponent: f64, neck_strain: f64) -> Self {
        Self {
            exponent,
            neck_strain,
            onset_fraction: 0.999,
            neck_curve: None,
            exponent_curve: None,
        }
    }

    /// Resolves the fading exponent at a size ratio.
    ///
    /// Positive card values pass through; negative values evaluate
    /// the exponent curve at `lambda` (element size over reference),
    /// falling back to the magnitude without a curve.
    pub fn resolved_exponent(&self, lambda: f64) -> f64 {
        if self.exponent >= 0.0 {
            return self.exponent;
        }
        self.exponent_curve
            .as_ref()
            .map_or(-self.exponent, |curve| curve.eval(lambda))
    }

    /// Resolves the necking strain at a triaxiality.
    ///
    /// Evaluates the instability table when set, else the constant.
    pub fn necking_strain(&self, triaxiality: f64) -> f64 {
        self.neck_curve
            .as_ref()
            .map_or(self.neck_strain, |curve| curve.eval(triaxiality))
    }

    /// Stress scale for a post-onset state without stepping.
    ///
    /// Returns `1 - ((soft - dd)/(1 - dd))^fad` above the onset
    /// fraction and 1 below it, clamped to `[0, 1]`. Pure helper for
    /// closed-form checks; stepping lives in
    /// [`FailTab::fade_step`].
    pub fn scale_only(&self, state: &FadeState) -> f64 {
        fade_scale(self.exponent, self.onset_fraction, state.soft)
    }
}

/// Fading scale from exponent, onset fraction, progress.
///
/// Shared by [`FadeSoftening::scale_only`] and the stepper so the
/// closed form stays single-sourced.
fn fade_scale(exponent: f64, onset: f64, soft: f64) -> f64 {
    if exponent == 0.0 || soft <= onset {
        return 1.0;
    }
    let span = (1.0 - onset).max(FAILURE_STRAIN_FLOOR);
    let scale = 1.0 - ((soft - onset) / span).powf(exponent.max(0.0));
    scale.clamp(0.0, 1.0)
}

impl FailTab {
    /// Advances fading softening one step, returning the stress scale.
    ///
    /// Mirrors `fail_tab_c.F`: `d_damage` (this step's damage increment
    /// from [`increment_at_rate`](Self::increment_at_rate), either
    /// weight branch) rescales onto the necking strain while the mean
    /// stress stays tensile, then onto the neck-to-fracture span past
    /// onset. The returned `DMG_SCALE` multiplies the reported stress;
    /// law history commits unscaled, matching the engine order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::fail_tab::{
    ///     FadeSoftening, FadeState, FailTab,
    /// };
    /// use sonium_fem::physics::solid::plasticity::funct::Funct1D;
    ///
    /// let curve = Funct1D::new(vec![-1.0, 1.0], vec![0.36, 0.36]).unwrap();
    /// let tab = FailTab::new(vec![0.0], vec![0.0], vec![curve]).unwrap();
    /// let mut fade = FadeSoftening::new(2.0, 0.2);
    /// fade.onset_fraction = 0.0;
    /// let mut state = FadeState::default();
    /// // Damage 0.5 at ef 0.36 reaches necking (0.5 x 0.36/0.2 = 0.9).
    /// let scale = tab.fade_step(&fade, &mut state, 0.5, 0.36, 0.0, 1.0, 1.0);
    /// assert!((scale - 1.0).abs() < 1e-15);
    /// assert!((state.neck - 0.9).abs() < 1e-12);
    /// ```
    pub fn fade_step(
        &self,
        fade: &FadeSoftening,
        state: &mut FadeState,
        d_damage: f64,
        failure_strain: f64,
        triaxiality: f64,
        mean_stress: f64,
        size_lambda: f64,
    ) -> f64 {
        let _ = self;
        if d_damage > 0.0 {
            let ef_neck = fade.necking_strain(triaxiality);
            // Necking accumulates while tensile (engine UVAR3); the
            // onset check below runs in the same step, so softening
            // starts immediately once necking trips.
            if mean_stress >= 0.0 && ef_neck > 0.0 {
                state.neck += d_damage * failure_strain / ef_neck;
            }
            if state.neck >= 1.0 {
                let span = failure_strain - ef_neck;
                if span <= 0.0 {
                    // Necked past the fracture strain: fully soft.
                    state.soft = 1.0;
                } else {
                    state.soft += d_damage * failure_strain / span;
                }
            }
        }
        let exponent = fade.resolved_exponent(size_lambda);
        fade_scale(exponent, fade.onset_fraction, state.soft)
    }
}

/// Brackets an input on a validated grid with clamped ends.
///
/// Returns the lower and upper node indices plus the linear fraction;
/// out-of-range inputs clamp to the edge node with fraction 0 or 1.
fn bracket(grid: &[f64], x: f64) -> (usize, usize, f64) {
    if grid.len() == 1 || x <= grid[0] {
        return (0, 0, 0.0);
    }
    let last = grid.len() - 1;
    if x >= grid[last] {
        return (last, last, 0.0);
    }
    let upper = grid.partition_point(|node| *node <= x);
    let lo = upper - 1;
    (lo, upper, (x - grid[lo]) / (grid[upper] - grid[lo]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a two-rate shell table with flat triaxiality curves.
    fn flat_shell(low: f64, high: f64) -> FailTab {
        let mk = |v: f64| Funct1D::new(vec![-1.0, 1.0], vec![v, v]).unwrap();
        FailTab::new(vec![1.0e-5, 0.5], vec![0.0], vec![mk(low), mk(high)]).unwrap()
    }

    #[test]
    fn rate_blend_matches_linear_weights() {
        let tab = flat_shell(0.5, 1.0);
        assert!((tab.table_strain(0.0, 0.0, 0.0) - 0.5).abs() < 1e-15);
        assert!((tab.table_strain(0.0, 1.0e-5, 0.0) - 0.5).abs() < 1e-15);
        assert!((tab.table_strain(0.0, 0.5, 0.0) - 1.0).abs() < 1e-15);
        // Midpoint of the deck rate rows blends to 0.75.
        let mid = 0.5 * (1.0e-5 + 0.5);
        assert!((tab.table_strain(0.0, mid, 0.0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn above_last_rate_extrapolates_from_last_two_rows() {
        let tab = flat_shell(0.5, 1.0);
        // Slope (1 - 0.5)/(0.5 - 1e-5); value at 1.0 continues the line.
        let slope = 0.5 / (0.5 - 1.0e-5);
        let expected = 1.0 + slope * 0.5;
        assert!((tab.table_strain(0.0, 1.0, 0.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn single_rate_table_holds_everywhere() {
        let curve = Funct1D::new(vec![0.0, 1.0], vec![0.2, 0.4]).unwrap();
        let tab = FailTab::new(vec![0.1], vec![0.0], vec![curve]).unwrap();
        assert!((tab.table_strain(0.5, 0.0, 0.0) - 0.3).abs() < 1e-15);
        assert!((tab.table_strain(0.5, 1.0e6, 0.0) - 0.3).abs() < 1e-15);
    }

    #[test]
    fn lode_blend_is_bilinear_with_clamped_ends() {
        let mk = |v: f64| Funct1D::new(vec![-1.0, 1.0], vec![v, v]).unwrap();
        let tab = FailTab::new(vec![0.0], vec![-1.0, 1.0], vec![mk(0.2), mk(0.6)]).unwrap();
        assert!((tab.table_strain(0.0, 0.0, 0.0) - 0.4).abs() < 1e-15);
        assert!((tab.table_strain(0.0, 0.0, -3.0) - 0.2).abs() < 1e-15);
        assert!((tab.table_strain(0.0, 0.0, 3.0) - 0.6).abs() < 1e-15);
    }

    #[test]
    fn constructor_rejects_bad_grids() {
        let curve = Funct1D::new(vec![0.0, 1.0], vec![0.2, 0.2]).unwrap();
        assert!(FailTab::new(vec![], vec![0.0], vec![curve.clone()]).is_err());
        assert!(
            FailTab::new(
                vec![0.1, 0.1],
                vec![0.0],
                vec![curve.clone(), curve.clone()]
            )
            .is_err()
        );
        assert!(FailTab::new(vec![-1.0], vec![0.0], vec![curve.clone()]).is_err());
        assert!(FailTab::new(vec![0.1], vec![0.0], vec![]).is_err());
        assert!(FailTab::new(vec![0.1], vec![0.0], vec![curve.clone(), curve]).is_err());
    }

    #[test]
    fn damage_follows_equation_959() {
        let mut tab = flat_shell(0.36, 0.36);
        tab.dp = 2.0;
        tab.exponent = 2.0;
        // dD = (0.01/0.36) . 2 . 2^(1/2).
        let expected = (0.01 / 0.36) * 2.0 * 2.0f64.sqrt();
        let got = tab.increment_at_rate(0.01, 0.0, 0.0, 0.0, 0.5, None, None);
        assert!((got - expected).abs() < 1e-15, "got {got}, want {expected}");
    }

    #[test]
    fn damage_curve_selects_equation_960() {
        let mut tab = flat_shell(0.5, 0.5);
        tab.damage_curve = Some(Funct1D::new(vec![0.0, 1.0], vec![2.0, 2.0]).unwrap());
        let got = tab.increment_at_rate(0.01, 0.3, 0.0, 0.0, 0.0, None, None);
        assert!((got - 0.04).abs() < 1e-15);
    }

    #[test]
    fn size_factor_applies_only_inside_window() {
        let mut tab = flat_shell(0.5, 0.5);
        tab.size_curve = Some(Funct1D::new(vec![0.0, 2.0], vec![1.0, 3.0]).unwrap());
        tab.size_lo = -0.5;
        tab.size_hi = 0.5;
        assert!((tab.size_factor(0.0, 2.0) - 3.0).abs() < 1e-15);
        assert!((tab.size_factor(0.6, 2.0) - 1.0).abs() < 1e-15);
        assert!((tab.size_factor(-0.5, 2.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn temp_factor_normalizes_per_equation_955() {
        let mut tab = flat_shell(0.5, 0.5);
        tab.temp_curve = Some(Funct1D::new(vec![0.0, 1.0], vec![1.0, 0.5]).unwrap());
        // T* = (600 - 300)/(1200 - 300) = 1/3.
        let got = tab.temp_factor(600.0, 300.0, 1200.0);
        assert!((got - (1.0 - 0.5 / 3.0)).abs() < 1e-15);
        assert!((tab.temp_factor(600.0, 300.0, 300.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rupture_uses_strict_threshold() {
        let tab = flat_shell(0.5, 0.5);
        assert!(!tab.ruptured(0.999));
        assert!(tab.ruptured(1.0));
    }

    #[test]
    fn elem_samp_matches_deck_funct() {
        let tab = FailTab::elem_samp_steel();
        // Deck node (0.3333333, 0.360) at the fast rate row.
        let fast = tab.table_strain(0.3333333, 0.5, 0.0);
        assert!((fast - 0.360).abs() < 1e-9, "fast {fast}");
        // Slow row scales the same node by 0.5.
        let slow = tab.table_strain(0.3333333, 1.0e-5, 0.0);
        assert!((slow - 0.180).abs() < 1e-9, "slow {slow}");
        // Shear node (0, 0.300) and compression tail (-1, 3.088).
        assert!((tab.table_strain(0.0, 0.5, 0.0) - 0.300).abs() < 1e-12);
        assert!((tab.table_strain(-1.0, 0.5, 0.0) - 3.088).abs() < 1e-12);
        assert!((tab.dcrit - 1.1).abs() < 1e-15);
    }

    #[test]
    fn increment_reads_update_and_guards_degenerate_steps() {
        let tab = flat_shell(0.5, 0.5);
        let mk = |dep| StressUpdate {
            stress: [0.0; 6],
            tangent: [0.0; 36],
            delta_equiv_plastic_strain: dep,
            yielding: true,
        };
        assert_eq!(tab.increment(&mk(0.0), 0.0, 0.0, 0.0, 1.0e-3), 0.0);
        assert_eq!(tab.increment(&mk(-0.1), 0.0, 0.0, 0.0, 1.0e-3), 0.0);
        // Zero step time stays finite through the rate floor.
        let got = tab.increment(&mk(0.01), 0.0, 0.0, 0.0, 0.0);
        assert!(got.is_finite());
    }

    #[test]
    fn fade_scale_matches_closed_form() {
        let mut fade = FadeSoftening::new(2.0, 0.2);
        fade.onset_fraction = 0.25;
        // Below onset the scale is exactly one.
        let pre = FadeState {
            neck: 0.5,
            soft: 0.2,
        };
        assert!((fade.scale_only(&pre) - 1.0).abs() < 1e-15);
        // Midpoint: 1 - ((0.625-0.25)/0.75)^2 = 0.75.
        let mid = FadeState {
            neck: 1.5,
            soft: 0.625,
        };
        assert!((fade.scale_only(&mid) - 0.75).abs() < 1e-12);
        // Fracture maps to zero; overshoot clamps there.
        let end = FadeState {
            neck: 2.0,
            soft: 1.0,
        };
        assert!((fade.scale_only(&end)).abs() < 1e-15);
        let over = FadeState {
            neck: 3.0,
            soft: 1.4,
        };
        assert!((fade.scale_only(&over)).abs() < 1e-15);
    }

    #[test]
    fn fade_step_necks_then_softens() {
        let tab = flat_shell(0.5, 0.5);
        let mut fade = FadeSoftening::new(5.0, 0.2);
        fade.onset_fraction = 0.0;
        let mut state = FadeState::default();
        // Constant locus ef 0.5, necking strain 0.2: necking trips
        // once damage passes 0.4 (0.4 x 0.5/0.2 = 1).
        let s1 = tab.fade_step(&fade, &mut state, 0.3, 0.5, 0.0, 10.0, 1.0);
        assert!((s1 - 1.0).abs() < 1e-15);
        assert!((state.neck - 0.75).abs() < 1e-12);
        assert_eq!(state.soft, 0.0);
        let s2 = tab.fade_step(&fade, &mut state, 0.3, 0.5, 0.0, 10.0, 1.0);
        assert!(state.neck >= 1.0);
        // Same-step onset also books soft: 0.3 x 0.5/0.3 = 0.5.
        assert!((state.soft - 0.5).abs() < 1e-12);
        assert!((s2 - (1.0 - 0.5f64.powi(5))).abs() < 1e-12);
    }

    #[test]
    fn fade_step_gates_compression_and_degenerate_cards() {
        let tab = flat_shell(0.5, 0.5);
        let fade = FadeSoftening::new(5.0, 0.2);
        let mut state = FadeState::default();
        // Compressive mean stress freezes necking.
        let s = tab.fade_step(&fade, &mut state, 0.5, 0.5, 0.0, -3.0, 1.0);
        assert!((s - 1.0).abs() < 1e-15);
        assert_eq!(state.neck, 0.0);
        // Necking past the fracture strain fully softens.
        let mut past = FadeState {
            neck: 1.0,
            soft: 0.0,
        };
        let zero = tab.fade_step(&fade, &mut past, 0.1, 0.15, 0.0, 3.0, 1.0);
        assert!((zero).abs() < 1e-15);
        // Zero exponent disables softening regardless of progress.
        let off = FadeSoftening::new(0.0, 0.2);
        let mut late = FadeState {
            neck: 2.0,
            soft: 0.9,
        };
        let one = tab.fade_step(&off, &mut late, 0.1, 0.5, 0.0, 3.0, 1.0);
        assert!((one - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fade_negative_exponent_reads_size_curve() {
        let tab = flat_shell(0.5, 0.5);
        let mut fade = FadeSoftening::new(-1.0, 0.2);
        fade.onset_fraction = 0.0;
        fade.exponent_curve = Some(Funct1D::new(vec![0.5, 2.0], vec![2.0, 8.0]).unwrap());
        assert!((fade.resolved_exponent(0.5) - 2.0).abs() < 1e-15);
        assert!((fade.resolved_exponent(2.0) - 8.0).abs() < 1e-15);
        // Without a curve the magnitude applies.
        let bare = FadeSoftening::new(-4.0, 0.2);
        assert!((bare.resolved_exponent(1.0) - 4.0).abs() < 1e-15);
        let _ = tab;
    }
}
