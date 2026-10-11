//! Tabulated triaxiality failure with damage accumulation (`/FAIL/TAB1`).
//!
//! Implements the Radioss tabulated failure model: the plastic failure
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
//!   reader, the Radioss default.
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
}
