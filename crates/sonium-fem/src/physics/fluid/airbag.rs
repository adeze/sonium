//! Uniform-pressure airbag and monitored-volume gas physics.
//!
//! Lumped gas state, injector inflow, isentropic vent outflow, chamber
//! communication, and jetting overpressure for `/MONVOL/AIRBAG`,
//! `/MONVOL/COMMU`, and `/MONVOL/COMMU1` volumes. The equations are the
//! Theory Manual monitored-volume chapter (GAS/AIRBAG/COMMU1 types,
//! equations 1074-1133); the discrete update replicates the engine
//! `AIRBAGA`/`AIRBAGB` cycle so native replays match harvested time
//! histories sample by sample.
//!
//! # Model
//!
//! Each chamber holds a uniform-pressure ideal-gas mixture with
//! negligible kinetic energy. Per cycle the A-phase injects mass and
//! energy, applies lagged outflows, and solves the coupled
//! temperature update (manual equation 1112); the B-phase evaluates
//! vent and communication fluxes for the next cycle (equations
//! 1115-1122, 1132-1133). Venting and communication are isentropic
//! orifice flows with a shocked branch capped at the critical
//! pressure (equations 1121, 1093-1094) and cycle limiters carried
//! over from the engine.
//!
//! # Units
//!
//! All quantities use one consistent unit system fixed by the caller
//! (the miniqa decks use kilogram, millimeter, millisecond, kelvin).
//! Temperatures are kelvin; angles entering jetting tables are
//! degrees, matching the engine.
//!
//! # Scope
//!
//! Implemented: constant heat-capacity gas with linear and quadratic
//! temperature terms, tabulated injector mass and temperature laws,
//! pressure- and time-gated vents, directed chamber links, conical
//! and dihedral jetting, and starter-equivalent initial conditions.
//! Not implemented: Chemkin venting, fabric porosity and blockage,
//! `ITTF` time shifts, Python functions, the FVMBAG family, and the
//! mechanical side (pressure loads, viscous damping work, timestep).
//! Those live in the card layer or the structural solver; the gas
//! state here does not depend on them.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::fluid::airbag::{Gas, initial_mass};
//!
//! let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
//! let r = gas.specific_gas_constant(295.0);
//! let mass = initial_mass(1.0135e-4, 264009.0, 0.0, r, 295.0);
//! assert!((mass - 4.3571e-4).abs() < 1e-9);
//! ```

// Rust guideline compliant 2026-02-21

/// Default initial temperature in kelvin.
///
/// The starter replaces a zero `T0` with 295 K (`hm_read_monvol_type5`).
pub const DEFAULT_TEMPERATURE: f64 = 295.0;

/// Absolute floor guarding divisions by accumulated mass.
///
/// Mirrors engine `EM20`; only reached for empty chambers, where the
/// guarded quotients multiply a zero numerator anyway.
pub const TINY_MASS: f64 = 1e-20;

/// Absolute floor guarding function-slope divisions.
///
/// Mirrors engine `EM16` in `FINTER`; only reached for vertical table
/// segments, which carry no usable slope either way.
pub const TINY_SLOPE: f64 = 1e-16;

/// Absolute floor guarding jetting geometry normalizations.
///
/// Mirrors engine `EM30`; only reached for degenerate (zero-length)
/// jet or element-normal vectors.
pub const TINY_GEOMETRY: f64 = 1e-30;

/// Relative volume floor factor for flow gates.
///
/// Vents and links stay shut while `volume <= 1e-3 * area^1.5`
/// (engine `EM3` gate), which suppresses outflow from collapsed bags.
pub const MIN_VOLUME_FACTOR: f64 = 1e-3;

/// Relative floor factor for the initial volume.
///
/// The starter adds `1e-4 * area^1.5` when the initial volume is
/// smaller (Theory Manual initial conditions), avoiding a singular
/// gas law at first contact.
pub const MIN_INITIAL_VOLUME_FACTOR: f64 = 1e-4;

/// Caloric gas model with polynomial heat capacity.
///
/// The constant-pressure heat capacity per unit mass is
/// `cp(T) = a + b*T + c*T^2` (manual equation 1097); the constant
/// `a`, `b`, `c` triple is the deck `cpa/cpb/cpc` card data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gas {
    /// Ratio of specific heats at the reference temperature.
    pub gamma: f64,
    /// Heat-capacity coefficients `[a, b, c]`.
    pub cp_coeffs: [f64; 3],
}

impl Gas {
    /// Builds a gas from gamma and heat-capacity coefficients.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
    /// assert!((gas.cp_at(300.0) - 520.275).abs() < 1e-12);
    /// ```
    pub fn new(gamma: f64, cp_coeffs: [f64; 3]) -> Self {
        Self { gamma, cp_coeffs }
    }

    /// Specific heat at constant pressure at temperature `t`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.4, [1005.0, 0.1, 0.0]);
    /// assert!((gas.cp_at(300.0) - 1035.0).abs() < 1e-9);
    /// ```
    pub fn cp_at(&self, t: f64) -> f64 {
        let [a, b, c] = self.cp_coeffs;
        a + t * (b + c * t)
    }

    /// Specific gas constant derived like the starter.
    ///
    /// Evaluates `cp` at the reference temperature, takes
    /// `cv = cp / gamma` there, and returns `R = cv * (gamma - 1)`
    /// (the Mayer relation, manual equation 1098).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
    /// let r = gas.specific_gas_constant(295.0);
    /// assert!((r - 208.1724).abs() < 1e-3);
    /// ```
    pub fn specific_gas_constant(&self, t_ref: f64) -> f64 {
        let cp = self.cp_at(t_ref);
        cp / self.gamma * (self.gamma - 1.0)
    }

    /// Specific heat at constant volume at temperature `t`.
    ///
    /// Uses the fixed gas constant `r` from
    /// [`specific_gas_constant`](Self::specific_gas_constant), matching
    /// the engine evaluation `cv = cp(T) - R`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
    /// let r = gas.specific_gas_constant(295.0);
    /// assert!((gas.cp_at(400.0) - gas.cv_at(400.0, r) - r).abs() < 1e-9);
    /// ```
    pub fn cv_at(&self, t: f64, r: f64) -> f64 {
        self.cp_at(t) - r
    }

    /// Specific enthalpy at temperature `t`.
    ///
    /// Closed form of the `cp` integral (manual equation 1101):
    /// `h = T*(a + b*T/2 + c*T^2/3)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
    /// assert!((gas.enthalpy(295.0) - 295.0 * 520.275).abs() < 1e-6);
    /// ```
    pub fn enthalpy(&self, t: f64) -> f64 {
        let [a, b, c] = self.cp_coeffs;
        t * (a + 0.5 * b * t + c * t * t / 3.0)
    }

    /// Specific internal energy at temperature `t`.
    ///
    /// Follows the starter definition `e = h - R*T` with the fixed
    /// gas constant `r`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Gas;
    ///
    /// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
    /// let r = gas.specific_gas_constant(295.0);
    /// let e = gas.energy(295.0, r);
    /// assert!((e - (gas.enthalpy(295.0) - r * 295.0)).abs() < 1e-9);
    /// ```
    pub fn energy(&self, t: f64, r: f64) -> f64 {
        self.enthalpy(t) - r * t
    }
}

/// Mixture ratio of specific heats from mass fractions.
///
/// Mass-weighted `cp` over mass-weighted `cv` (manual venting
/// section), with each gas evaluated at the shared temperature `t`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::{Gas, mixture_gamma};
///
/// let gas = Gas::new(1.667, [520.275, 0.0, 0.0]);
/// let r = gas.specific_gas_constant(295.0);
/// let g = mixture_gamma(&[(1.0, gas, r), (2.0, gas, r)], 300.0);
/// assert!((g - 1.667).abs() < 1e-12);
/// ```
pub fn mixture_gamma(masses: &[(f64, Gas, f64)], t: f64) -> f64 {
    let mut cp = 0.0;
    let mut cv = 0.0;
    for (mass, gas, r) in masses {
        cp += mass * gas.cp_at(t);
        cv += mass * gas.cv_at(t, *r);
    }
    cp / cv
}

/// Piecewise-linear tabulated function with engine semantics.
///
/// Interpolation matches `FINTER`: linear between tabulated points,
/// linear extrapolation past the ends using the end slopes, and a
/// constant fast path for single-point tables.
#[derive(Debug, Clone, PartialEq)]
pub struct Tabulated {
    abscissae: Vec<f64>,
    ordinates: Vec<f64>,
}

impl Tabulated {
    /// Builds a table from abscissa and ordinate vectors.
    ///
    /// # Panics
    ///
    /// Panics when the vectors are empty, differ in length, or the
    /// abscissae are not strictly increasing; all three are caller
    /// contract violations, not runtime failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Tabulated;
    ///
    /// let f = Tabulated::new(vec![0.0, 1.0], vec![2.0, 4.0]);
    /// assert!((f.eval(0.25) - 2.5).abs() < 1e-12);
    /// ```
    pub fn new(abscissae: Vec<f64>, ordinates: Vec<f64>) -> Self {
        assert!(!abscissae.is_empty(), "tabulated function needs points");
        assert_eq!(
            abscissae.len(),
            ordinates.len(),
            "abscissae and ordinates must pair up"
        );
        assert!(
            abscissae.windows(2).all(|pair| pair[0] < pair[1]),
            "abscissae must be strictly increasing"
        );
        Self {
            abscissae,
            ordinates,
        }
    }

    /// Builds a constant table returning `value` everywhere.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Tabulated;
    ///
    /// let f = Tabulated::constant(307.61);
    /// assert!((f.eval(12.5) - 307.61).abs() < 1e-12);
    /// ```
    pub fn constant(value: f64) -> Self {
        Self {
            abscissae: vec![0.0],
            ordinates: vec![value],
        }
    }

    /// Evaluates the table at `x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Tabulated;
    ///
    /// let f = Tabulated::new(vec![1.0, 3.0], vec![1.0, 5.0]);
    /// assert!((f.eval(5.0) - 9.0).abs() < 1e-12);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.abscissae.len();
        if n == 1 {
            return self.ordinates[0];
        }
        let mut lo = 0;
        while lo + 1 < n && x > self.abscissae[lo + 1] {
            lo += 1;
        }
        let lo = lo.min(n - 2);
        let hi = lo + 1;
        let x0 = self.abscissae[lo];
        let x1 = self.abscissae[hi];
        let y0 = self.ordinates[lo];
        let y1 = self.ordinates[hi];
        let dx = x1 - x0;
        let slope = if dx.abs() < TINY_SLOPE {
            0.0
        } else {
            (y1 - y0) / dx
        };
        if x <= x0 {
            y0 + (x - x0) * slope
        } else {
            y1 - (x1 - x) * slope
        }
    }
}

/// Gas injector feeding one chamber.
///
/// The mass law gives cumulative injected mass versus time (or mass
/// flux when `flow_rate` is set); the temperature law gives the
/// inflow gas temperature. Both scale by the card factors.
#[derive(Debug, Clone)]
pub struct Injector {
    /// Injected gas caloric model.
    pub gas: Gas,
    /// Cumulative mass (or flux) law.
    pub mass_law: Tabulated,
    /// Inflow temperature law.
    pub temp_law: Tabulated,
    /// Mass-law scale factor.
    pub mass_scale: f64,
    /// Temperature-law scale factor.
    pub temp_scale: f64,
    /// Time scale applied to both laws.
    pub time_scale: f64,
    /// Set when the mass law is a flux, not a total.
    pub flow_rate: bool,
    /// Fire time; no injection before it.
    pub start: f64,
}

impl Injector {
    /// Cumulative injected mass at time `t`.
    ///
    /// Returns zero before [`start`](Self::start). Flux laws need the
    /// previous level and step in [`Chamber::step_a`](Chamber::step_a)
    /// instead; this helper covers total-mass laws.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Gas, Injector, Tabulated};
    ///
    /// let jet = Injector {
    ///     gas: Gas::new(1.667, [520.275, 0.0, 0.0]),
    ///     mass_law: Tabulated::new(vec![0.0, 1.0], vec![0.0, 2.0]),
    ///     temp_law: Tabulated::constant(307.61),
    ///     mass_scale: 1.0,
    ///     temp_scale: 1.0,
    ///     time_scale: 1.0,
    ///     flow_rate: false,
    ///     start: 0.0,
    /// };
    /// assert!((jet.mass_at(0.5) - 1.0).abs() < 1e-12);
    /// ```
    pub fn mass_at(&self, t: f64) -> f64 {
        if t < self.start {
            return 0.0;
        }
        self.mass_scale * self.mass_law.eval((t - self.start) * self.time_scale)
    }

    /// Inflow gas temperature at time `t`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Gas, Injector, Tabulated};
    ///
    /// let jet = Injector {
    ///     gas: Gas::new(1.667, [520.275, 0.0, 0.0]),
    ///     mass_law: Tabulated::constant(0.0),
    ///     temp_law: Tabulated::new(vec![0.0, 1.0], vec![300.0, 400.0]),
    ///     mass_scale: 1.0,
    ///     temp_scale: 1.0,
    ///     time_scale: 1.0,
    ///     flow_rate: false,
    ///     start: 0.0,
    /// };
    /// assert!((jet.temp_at(0.5) - 350.0).abs() < 1e-12);
    /// ```
    pub fn temp_at(&self, t: f64) -> f64 {
        if t < self.start {
            return self.temp_scale * self.temp_law.eval(0.0);
        }
        self.temp_scale * self.temp_law.eval((t - self.start) * self.time_scale)
    }

    /// Specific inflow enthalpy at time `t`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Gas, Injector, Tabulated};
    ///
    /// let jet = Injector {
    ///     gas: Gas::new(1.667, [520.275, 0.0, 0.0]),
    ///     mass_law: Tabulated::constant(0.0),
    ///     temp_law: Tabulated::constant(307.61),
    ///     mass_scale: 1.0,
    ///     temp_scale: 1.0,
    ///     time_scale: 1.0,
    ///     flow_rate: false,
    ///     start: 0.0,
    /// };
    /// let h = jet.inflow_enthalpy(2.0);
    /// assert!((h - 307.61 * 520.275).abs() < 1e-6);
    /// ```
    pub fn inflow_enthalpy(&self, t: f64) -> f64 {
        self.gas.enthalpy(self.temp_at(t))
    }
}

/// Isentropic orifice outflow state.
///
/// Bundles the exit velocity, downstream density, and mass and energy
/// rates of one vent or link evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrificeFlow {
    /// Exit velocity at the hole.
    pub velocity: f64,
    /// Downstream (exit-plane) density.
    pub exit_density: f64,
    /// Mass flow rate, positive outward.
    pub mass_rate: f64,
    /// Energy flow rate, positive outward.
    pub energy_rate: f64,
}

/// Critical pressure ratio for choked orifice flow.
///
/// Returns `(2 / (gamma + 1))^(gamma / (gamma - 1))` (manual equation
/// 1121); the vent pressure is the max of this times the chamber
/// pressure and the downstream pressure.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::critical_ratio;
///
/// let q = critical_ratio(1.4);
/// assert!((q - 0.5282818).abs() < 1e-6);
/// ```
pub fn critical_ratio(gamma: f64) -> f64 {
    (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0))
}

/// Isentropic orifice flow with shocked branch.
///
/// Evaluates the exit velocity (manual equation 1117), downstream
/// density, and mass and energy rates (equations 1120, 1119) for a
/// hole of `area` between chamber pressure `p` and downstream
/// pressure `downstream`. The `vmax` cap carries the engine cycle
/// limiter; pass `f64::INFINITY` for the pure theory value.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::isentropic_orifice;
///
/// let flow = isentropic_orifice(1.4, 2.0e-4, 1.0e-4, 1.2e-9, 840.0, 3.0e5, f64::INFINITY);
/// assert!(flow.velocity > 0.0);
/// assert!(flow.mass_rate > 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn isentropic_orifice(
    gamma: f64,
    p: f64,
    downstream: f64,
    density: f64,
    area: f64,
    specific_enthalpy: f64,
    vmax: f64,
) -> OrificeFlow {
    let vent = downstream.max(p * critical_ratio(gamma));
    let ratio = (vent / p).clamp(0.0, 1.0);
    let speed_sq =
        2.0 * gamma / (gamma - 1.0) * p / density * (1.0 - ratio.powf((gamma - 1.0) / gamma));
    let velocity = speed_sq.max(0.0).sqrt().min(vmax);
    let exit_density = density * ratio.powf(1.0 / gamma);
    let mass_rate = exit_density * area * velocity;
    OrificeFlow {
        velocity,
        exit_density,
        mass_rate,
        energy_rate: mass_rate * specific_enthalpy,
    }
}

/// Cycle cap for vent exit velocity.
///
/// Mirrors the engine `VMAX`: the energy-based bound plus the
/// geometric bound `volume / (2 * area * dt)`, both required to keep
/// explicit outflow stable.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::vent_velocity_cap;
///
/// let cap = vent_velocity_cap(1.667, 2.0e-4, 1.0e-4, 9.0e6, 1.6e5, 1.6e-9, 840.0, 0.01);
/// assert!(cap > 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn vent_velocity_cap(
    gamma: f64,
    p: f64,
    pext: f64,
    volume: f64,
    specific_enthalpy: f64,
    exit_density: f64,
    area: f64,
    dt: f64,
) -> f64 {
    let energy = 0.5 * (p - pext) * volume
        / (gamma - 1.0)
        / (specific_enthalpy * exit_density * area * dt).max(TINY_MASS);
    let geometric = 0.5 * volume / (area * dt).max(TINY_MASS);
    energy.min(geometric)
}

/// Vent hole with deflation gating.
///
/// Flow starts once the membrane deflates: either the overpressure
/// exceeds `pdef` for longer than `dtpdef`, or time passes `tvent`.
/// Flow stops at `tstop`. Areas are effective (discharge coefficient
/// included); time-varying area laws stay in the card layer.
#[derive(Debug, Clone)]
pub struct Vent {
    /// Effective hole area.
    pub area: f64,
    /// Overpressure that starts deflation.
    pub pdef: f64,
    /// Overpressure duration that starts deflation.
    pub dtpdef: f64,
    /// Time that starts venting.
    pub tvent: f64,
    /// Time that stops venting.
    pub tstop: f64,
    /// True once the membrane has deflated.
    pub deflated: bool,
    /// Accumulated overpressure time.
    pub deflated_time: f64,
}

impl Vent {
    /// Builds a shut vent with zero deflation state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Vent;
    ///
    /// let vent = Vent::new(840.0, 0.0, 0.0, 1e30, 1e20);
    /// assert!(!vent.deflated);
    /// ```
    pub fn new(area: f64, pdef: f64, dtpdef: f64, tvent: f64, tstop: f64) -> Self {
        Self {
            area,
            pdef,
            dtpdef,
            tvent,
            tstop,
            deflated: false,
            deflated_time: 0.0,
        }
    }

    /// Accumulates overpressure time for the deflation gate.
    ///
    /// Call once per cycle with the previous-cycle pressure before
    /// [`update`](Self::update).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Vent;
    ///
    /// let mut vent = Vent::new(840.0, 0.0, 0.0, 1e30, 1e20);
    /// vent.accumulate(1.1e-4, 1.0e-4, 0.01);
    /// assert!((vent.deflated_time - 0.01).abs() < 1e-12);
    /// ```
    pub fn accumulate(&mut self, p_old: f64, pext: f64, dt: f64) {
        if !self.deflated && p_old > self.pdef + pext {
            self.deflated_time += dt;
        }
    }

    /// Updates the deflation flag for the current cycle.
    ///
    /// Opens on pressure/time criteria and closes at `tstop`,
    /// matching the engine vent state machine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Vent;
    ///
    /// let mut vent = Vent::new(840.0, 0.0, 0.0, 1e30, 1e20);
    /// vent.deflated_time = 0.02;
    /// vent.update(1.1e-4, 1.0e-4, 264009.0, 294656.0, 0.05);
    /// assert!(vent.deflated);
    /// ```
    pub fn update(&mut self, p: f64, pext: f64, volume: f64, area: f64, t: f64) {
        if !self.deflated
            && (((p > self.pdef + pext && self.deflated_time > self.dtpdef) || t > self.tvent)
                && t < self.tstop
                && volume > MIN_VOLUME_FACTOR * area.powf(1.5))
        {
            self.deflated = true;
        }
        if self.deflated && t >= self.tstop {
            self.deflated = false;
        }
    }

    /// Effective open area for the current state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Vent;
    ///
    /// let mut vent = Vent::new(840.0, 0.0, 0.0, 1e30, 1e20);
    /// vent.deflated = true;
    /// assert!((vent.open_area(1.1e-4, 1.0e-4, 264009.0, 294656.0) - 840.0).abs() < 1e-12);
    /// ```
    pub fn open_area(&self, p: f64, pext: f64, volume: f64, area: f64) -> f64 {
        if self.deflated && p > pext && volume > MIN_VOLUME_FACTOR * area.powf(1.5) {
            self.area
        } else {
            0.0
        }
    }
}

/// Directed communication link between two chambers.
///
/// One link carries flow from its chamber to the partner only; a
/// two-way communication needs a mirrored pair (asymmetric areas and
/// gates model valves). The flux equations are the vent equations
/// with the partner pressure downstream (manual equations 1132-1133).
#[derive(Debug, Clone)]
pub struct Link {
    /// Partner chamber index in [`Chambers`].
    pub partner: usize,
    /// Effective communication area.
    pub area: f64,
    /// Overpressure over the partner that opens the link.
    pub pdef: f64,
    /// Overpressure duration that opens the link.
    pub dtpdef: f64,
    /// Time that opens the link.
    pub tvent: f64,
    /// True once the link has opened.
    pub open: bool,
    /// Accumulated overpressure time.
    pub open_time: f64,
}

impl Link {
    /// Builds a shut link with zero opening state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Link;
    ///
    /// let link = Link::new(0, 3958.5, 0.0, 0.0, 1e30);
    /// assert!(!link.open);
    /// ```
    pub fn new(partner: usize, area: f64, pdef: f64, dtpdef: f64, tvent: f64) -> Self {
        Self {
            partner,
            area,
            pdef,
            dtpdef,
            tvent,
            open: false,
            open_time: 0.0,
        }
    }

    /// Accumulates overpressure time for the opening gate.
    ///
    /// Call once per cycle with the previous-cycle pressure and the
    /// partner initial pressure before [`update`](Self::update).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Link;
    ///
    /// let mut link = Link::new(0, 3958.5, 0.0, 0.0, 1e30);
    /// link.accumulate(1.1e-4, 1.0e-4, 0.01);
    /// assert!((link.open_time - 0.01).abs() < 1e-12);
    /// ```
    pub fn accumulate(&mut self, p_old: f64, partner_pinit: f64, dt: f64) {
        if !self.open && p_old > self.pdef + partner_pinit {
            self.open_time += dt;
        }
    }

    /// Updates the opening flag for the current cycle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Link;
    ///
    /// let mut link = Link::new(0, 3958.5, 0.0, 0.0, 1e30);
    /// link.open_time = 0.02;
    /// link.update(1.1e-4, 1.0e-4, 264009.0, 294656.0, 0.05);
    /// assert!(link.open);
    /// ```
    pub fn update(&mut self, p: f64, partner_p: f64, volume: f64, area: f64, t: f64) {
        if !self.open
            && (p > self.pdef + partner_p && self.open_time > self.dtpdef || t > self.tvent)
            && volume > MIN_VOLUME_FACTOR * area.powf(1.5)
        {
            self.open = true;
        }
    }
}

/// Cycle cap for link velocity.
///
/// The geometric bound plus the pressure-difference bound that keeps
/// chamber-to-chamber exchange stable (engine `AIRBAGB` limiter).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::link_velocity_cap;
///
/// let cap = link_velocity_cap(1.667, 1.1e-4, 1.0e-4, 264009.0, 243096.0, 2.5e-4, 3958.5, 0.01);
/// assert!(cap > 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn link_velocity_cap(
    gamma: f64,
    p: f64,
    partner_p: f64,
    volume: f64,
    partner_volume: f64,
    energy_density: f64,
    area: f64,
    dt: f64,
) -> f64 {
    let geometric = 0.5 * volume / (area * dt).max(TINY_MASS);
    let pressure = (p - partner_p) * 0.5 * volume.min(partner_volume)
        / (gamma - 1.0)
        / energy_density
        / (area * dt).max(TINY_MASS);
    geometric.min(pressure)
}

/// Initial gas mass from the perfect gas law.
///
/// Returns `p * (volume + veps) / (r * t)` (starter equilibrium).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::initial_mass;
///
/// let m = initial_mass(1.0135e-4, 264009.0, 0.0, 208.1724, 295.0);
/// assert!((m - 4.3571e-4).abs() < 1e-9);
/// ```
pub fn initial_mass(p: f64, volume: f64, veps: f64, r: f64, t: f64) -> f64 {
    p * (volume + veps) / (r * t)
}

/// Small-volume allowance for a chamber surface.
///
/// Returns `max(0, 1e-4 * area^1.5 - volume)` (manual initial
/// conditions), the starter `VEPS`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::small_volume;
///
/// assert!((small_volume(264009.0, 294656.0) - 0.0).abs() < 1e-12);
/// assert!(small_volume(1.0, 294656.0) > 1.0e4);
/// ```
pub fn small_volume(volume: f64, area: f64) -> f64 {
    (MIN_INITIAL_VOLUME_FACTOR * area.powf(1.5) - volume).max(0.0)
}

/// Starter vent preset: deflated from the first cycle.
///
/// The starter pre-opens vents with zero pressure, time, and start
/// criteria (`hm_read_monvol_type5`).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::vent_starts_open;
///
/// assert!(vent_starts_open(0.0, 0.0, 0.0));
/// assert!(!vent_starts_open(0.0, 0.0, 1e30));
/// ```
pub fn vent_starts_open(pdef: f64, dtpdef: f64, tvent: f64) -> bool {
    pdef == 0.0 && dtpdef == 0.0 && tvent == 0.0
}

/// Starter link preset: open from the first cycle.
///
/// The starter pre-opens links with zero pressure criteria or a zero
/// start time (`hm_read_monvol_type5`).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::fluid::airbag::link_starts_open;
///
/// assert!(link_starts_open(0.0, 0.0, 1e30));
/// assert!(!link_starts_open(2e-7, 0.0, 1e30));
/// ```
pub fn link_starts_open(pdef: f64, dtpdef: f64, tvent: f64) -> bool {
    (pdef == 0.0 && dtpdef == 0.0) || tvent == 0.0
}

/// Per-injector running state: levels plus lagged outflow rates.
#[derive(Debug, Clone)]
pub struct InjectorState {
    /// Injector card data and laws.
    pub injector: Injector,
    /// Specific gas constant of the injected gas.
    pub rmw: f64,
    /// Last injector law level (total mass).
    pub law_level: f64,
    /// Injected gas mass currently in the chamber.
    pub stored: f64,
    /// Lagged outflow mass rate (B-phase output).
    pub out_mass_rate: f64,
    /// Lagged outflow energy rate (B-phase output).
    pub out_energy_rate: f64,
}

impl InjectorState {
    /// Builds zero running state for an injector.
    ///
    /// The gas constant derives at the chamber initial temperature,
    /// matching the starter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Gas, Injector, InjectorState, Tabulated};
    ///
    /// let jet = Injector {
    ///     gas: Gas::new(1.667, [520.275, 0.0, 0.0]),
    ///     mass_law: Tabulated::constant(0.0),
    ///     temp_law: Tabulated::constant(307.61),
    ///     mass_scale: 1.0,
    ///     temp_scale: 1.0,
    ///     time_scale: 1.0,
    ///     flow_rate: false,
    ///     start: 0.0,
    /// };
    /// let state = InjectorState::new(jet, 295.0);
    /// assert!((state.rmw - 208.1724).abs() < 1e-3);
    /// ```
    pub fn new(injector: Injector, t_init: f64) -> Self {
        let rmw = injector.gas.specific_gas_constant(t_init);
        Self {
            injector,
            rmw,
            law_level: 0.0,
            stored: 0.0,
            out_mass_rate: 0.0,
            out_energy_rate: 0.0,
        }
    }
}

/// Per-vent running state: totals plus lagged outflow rates.
#[derive(Debug, Clone)]
pub struct VentState {
    /// Vent card data and gate.
    pub vent: Vent,
    /// Cumulative vented mass.
    pub mass_out: f64,
    /// Cumulative vented energy.
    pub energy_out: f64,
    /// Lagged outflow mass rate (B-phase output).
    pub out_mass_rate: f64,
    /// Lagged outflow energy rate (B-phase output).
    pub out_energy_rate: f64,
    /// Open area at the current cycle (time-history output).
    pub open_area: f64,
    /// Exit velocity at the current cycle (time-history output).
    pub velocity: f64,
}

impl VentState {
    /// Builds zero running state for a vent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Vent, VentState};
    ///
    /// let state = VentState::new(Vent::new(840.0, 0.0, 0.0, 1e30, 1e20));
    /// assert!((state.mass_out - 0.0).abs() < 1e-12);
    /// ```
    pub fn new(vent: Vent) -> Self {
        Self {
            vent,
            mass_out: 0.0,
            energy_out: 0.0,
            out_mass_rate: 0.0,
            out_energy_rate: 0.0,
            open_area: 0.0,
            velocity: 0.0,
        }
    }
}

/// Chamber configuration: card data without running state.
#[derive(Debug, Clone, Default)]
pub struct ChamberConfig {
    /// Initial-gas caloric model.
    pub gas: Option<Gas>,
    /// External (ambient) pressure.
    pub pext: f64,
    /// Initial temperature; zero selects [`DEFAULT_TEMPERATURE`].
    pub t0: f64,
    /// Small-volume allowance; zero selects [`small_volume`].
    pub veps: f64,
    /// Injectors feeding the chamber.
    pub injectors: Vec<Injector>,
    /// Vent holes.
    pub vents: Vec<Vent>,
    /// Directed communication links.
    pub links: Vec<Link>,
}

/// Uniform-pressure chamber with full running state.
///
/// Holds the initial-gas bucket, per-injector buckets, lagged outflow
/// slots, gate flags, and vent totals. Volumes and areas are
/// prescribed by the caller (bag mechanics); [`Chambers::step`]
/// advances the gas state one cycle.
#[derive(Debug, Clone)]
pub struct Chamber {
    /// Initial-gas caloric model.
    pub gas: Gas,
    /// Specific gas constant of the initial gas.
    pub rmw: f64,
    /// External (ambient) pressure.
    pub pext: f64,
    /// Initial pressure (equals `pext` for AIRBAG/COMMU cards).
    pub pinit: f64,
    /// Small-volume allowance added to the geometric volume.
    pub veps: f64,
    /// Current geometric volume (without `veps`).
    pub volume: f64,
    /// Current surface area.
    pub area: f64,
    /// Gas temperature.
    pub temperature: f64,
    /// Gas pressure.
    pub pressure: f64,
    /// Initial-gas mass bucket.
    pub initial_mass: f64,
    /// Total gas mass.
    pub total_mass: f64,
    /// `R * n` mixture invariant.
    pub rnm: f64,
    /// Mixture ratio of specific heats.
    pub gamma: f64,
    /// Per-injector running state.
    pub injectors: Vec<InjectorState>,
    /// Per-vent running state.
    pub vents: Vec<VentState>,
    /// Directed communication links.
    pub links: Vec<Link>,
    /// Lagged initial-gas outflow mass rate (B-phase output).
    pub out_mass_rate: f64,
    /// Lagged initial-gas outflow energy rate (B-phase output).
    pub out_energy_rate: f64,
}

impl Chamber {
    /// Builds equilibrium running state from card data.
    ///
    /// Sets `T0` to [`DEFAULT_TEMPERATURE`] when zero, `VEPS` from
    /// [`small_volume`] when zero, the initial mass from
    /// [`initial_mass`], and vent/link starter presets. Panics when
    /// `gas` is missing (a caller contract violation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bag = Chamber::from_config(&config, 264009.0, 294656.0);
    /// assert!((bag.temperature - 295.0).abs() < 1e-12);
    /// assert!((bag.total_mass - 4.3571e-4).abs() < 1e-9);
    /// ```
    pub fn from_config(config: &ChamberConfig, volume: f64, area: f64) -> Self {
        let gas = config.gas.expect("chamber needs an initial gas");
        let t0 = if config.t0 == 0.0 {
            DEFAULT_TEMPERATURE
        } else {
            config.t0
        };
        let veps = if config.veps == 0.0 {
            small_volume(volume, area)
        } else {
            config.veps
        };
        let rmw = gas.specific_gas_constant(t0);
        let mass = initial_mass(config.pext, volume, veps, rmw, t0);
        let vents: Vec<VentState> = config
            .vents
            .iter()
            .map(|vent| {
                let mut state = VentState::new(vent.clone());
                state.vent.deflated = vent_starts_open(vent.pdef, vent.dtpdef, vent.tvent);
                state
            })
            .collect();
        let links: Vec<Link> = config
            .links
            .iter()
            .map(|link| {
                let mut opened = link.clone();
                opened.open = link_starts_open(link.pdef, link.dtpdef, link.tvent);
                opened
            })
            .collect();
        Self {
            gas,
            rmw,
            pext: config.pext,
            pinit: config.pext,
            veps,
            volume,
            area,
            temperature: t0,
            pressure: config.pext,
            initial_mass: mass,
            total_mass: mass,
            rnm: rmw * mass,
            gamma: gas.gamma,
            injectors: config
                .injectors
                .iter()
                .map(|jet| InjectorState::new(jet.clone(), t0))
                .collect(),
            vents,
            links,
            out_mass_rate: 0.0,
            out_energy_rate: 0.0,
        }
    }

    /// Working volume including the small-volume allowance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bag = Chamber::from_config(&config, 264009.0, 294656.0);
    /// assert!((bag.working_volume() - 264009.0).abs() < 1e-9);
    /// ```
    pub fn working_volume(&self) -> f64 {
        self.volume + self.veps
    }

    /// Mass-weighted specific enthalpy at the current temperature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bag = Chamber::from_config(&config, 264009.0, 294656.0);
    /// assert!((bag.specific_enthalpy() - 295.0 * 520.275).abs() < 1e-3);
    /// ```
    pub fn specific_enthalpy(&self) -> f64 {
        let mut h = self.initial_mass * self.gas.enthalpy(self.temperature);
        for jet in &self.injectors {
            h += jet.stored * jet.injector.gas.enthalpy(self.temperature);
        }
        h / self.total_mass.max(TINY_MASS)
    }

    /// Mixture ratio of specific heats at the current temperature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bag = Chamber::from_config(&config, 264009.0, 294656.0);
    /// assert!((bag.mixture_gamma() - 1.667).abs() < 1e-12);
    /// ```
    pub fn mixture_gamma(&self) -> f64 {
        let mut cp = self.initial_mass * self.gas.cp_at(self.temperature);
        let mut cv = self.initial_mass * self.gas.cv_at(self.temperature, self.rmw);
        for jet in &self.injectors {
            cp += jet.stored * jet.injector.gas.cp_at(self.temperature);
            cv += jet.stored * jet.injector.gas.cv_at(self.temperature, jet.rmw);
        }
        cp / cv
    }

    /// Clears lagged outflow slots before a B-phase evaluation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bag = Chamber::from_config(&config, 264009.0, 294656.0);
    /// bag.out_mass_rate = 1.0;
    /// bag.clear_rates();
    /// assert!((bag.out_mass_rate - 0.0).abs() < 1e-12);
    /// ```
    pub fn clear_rates(&mut self) {
        self.out_mass_rate = 0.0;
        self.out_energy_rate = 0.0;
        for jet in &mut self.injectors {
            jet.out_mass_rate = 0.0;
            jet.out_energy_rate = 0.0;
        }
    }
}

/// Time-history sample of one chamber after a cycle.
///
/// Mirrors the engine `FSAV(1..7)` monitored-volume outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThSample {
    /// Total gas mass.
    pub mass: f64,
    /// Working volume (geometric plus allowance).
    pub volume: f64,
    /// Gas pressure.
    pub pressure: f64,
    /// Surface area.
    pub area: f64,
    /// Gas temperature.
    pub temperature: f64,
    /// Total open vent area.
    pub vent_area: f64,
    /// Mean vent exit velocity.
    pub vent_velocity: f64,
}

/// Time-history sample of one vent after a cycle.
///
/// Mirrors the engine per-vent `RBAGHOL(16..20)` outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VentSample {
    /// Open area.
    pub area: f64,
    /// Exit velocity.
    pub velocity: f64,
    /// Cumulative vented mass.
    pub mass_out: f64,
    /// Cumulative vented energy.
    pub energy_out: f64,
}

/// Coupled uniform-pressure chambers stepped one cycle at a time.
///
/// Owns the chambers plus their link topology. Volumes and areas are
/// prescribed per cycle (bag mechanics); [`step`](Self::step) runs
/// the A-phase (injection, temperature solve) then the B-phase (vent
/// and link fluxes) for every chamber, matching the engine order.
#[derive(Debug, Clone)]
pub struct Chambers {
    chambers: Vec<Chamber>,
}

impl Chambers {
    /// Builds a system from chambers.
    ///
    /// # Panics
    ///
    /// Panics when a link partner index is out of range or a link
    /// target holds fewer injectors than the source (both are caller
    /// contract violations; the engine indexes partner jet buckets
    /// by source jet).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// assert_eq!(bags.len(), 1);
    /// ```
    pub fn new(chambers: Vec<Chamber>) -> Self {
        for (i, chamber) in chambers.iter().enumerate() {
            for link in &chamber.links {
                assert!(
                    link.partner < chambers.len(),
                    "chamber {i} links to missing partner {}",
                    link.partner
                );
                assert!(
                    chambers[link.partner].injectors.len() >= chamber.injectors.len(),
                    "chamber {} links to partner {} with fewer injectors",
                    i,
                    link.partner
                );
            }
        }
        Self { chambers }
    }

    /// Number of chambers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// assert!(bags.is_empty() || bags.len() == 1);
    /// ```
    pub fn len(&self) -> usize {
        self.chambers.len()
    }

    /// True when the system holds no chambers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::Chambers;
    ///
    /// assert!(Chambers::new(Vec::new()).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.chambers.is_empty()
    }

    /// Borrows chamber `index`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// assert!((bags.chamber(0).temperature - 295.0).abs() < 1e-12);
    /// ```
    pub fn chamber(&self, index: usize) -> &Chamber {
        &self.chambers[index]
    }

    /// Mutably borrows chamber `index`.
    ///
    /// Re-anchoring and test harnesses use this to set observed
    /// state; the stepping methods keep all other state consistent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// bags.chamber_mut(0).temperature = 300.0;
    /// assert!((bags.chamber(0).temperature - 300.0).abs() < 1e-12);
    /// ```
    pub fn chamber_mut(&mut self, index: usize) -> &mut Chamber {
        &mut self.chambers[index]
    }

    /// Clears lagged outflow slots on every chamber.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// bags.chamber_mut(0).out_mass_rate = 1.0;
    /// bags.clear_rates();
    /// assert!((bags.chamber(0).out_mass_rate - 0.0).abs() < 1e-12);
    /// ```
    pub fn clear_rates(&mut self) {
        for chamber in &mut self.chambers {
            chamber.clear_rates();
        }
    }

    /// Advances every chamber one cycle.
    ///
    /// Runs the A-phase (injection plus temperature solve) then the
    /// B-phase (vent and link fluxes) with prescribed `volumes` and
    /// `areas` at cycle end. Panics when the slices do not match the
    /// chamber count (a caller contract violation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// bags.step(&[264130.0], &[294658.0], 0.01, 0.01);
    /// assert!(bags.chamber(0).pressure > 0.0);
    /// ```
    pub fn step(&mut self, volumes: &[f64], areas: &[f64], dt: f64, t: f64) {
        self.step_a(volumes, areas, dt, t);
        self.step_b(dt, t);
    }

    /// A-phase: gate accumulation, vent totals, injection, temperature.
    ///
    /// Updates pressure, temperature, mass, and mixture state from the
    /// prescribed end-of-cycle volumes; fills no flux slots (those
    /// are B-phase outputs consumed next cycle).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// bags.step_a(&[264130.0], &[294658.0], 0.01, 0.01);
    /// assert!(bags.chamber(0).temperature > 0.0);
    /// ```
    pub fn step_a(&mut self, volumes: &[f64], areas: &[f64], dt: f64, t: f64) {
        assert_eq!(
            volumes.len(),
            self.chambers.len(),
            "volumes must match chambers"
        );
        assert_eq!(
            areas.len(),
            self.chambers.len(),
            "areas must match chambers"
        );
        let pinits: Vec<f64> = self.chambers.iter().map(|chamber| chamber.pinit).collect();
        for chamber in &mut self.chambers {
            for vent in &mut chamber.vents {
                vent.vent.accumulate(chamber.pressure, chamber.pext, dt);
            }
            for link in &mut chamber.links {
                link.accumulate(chamber.pressure, pinits[link.partner], dt);
            }
            for vent in &mut chamber.vents {
                vent.mass_out += vent.out_mass_rate * dt;
                vent.energy_out += vent.out_energy_rate * dt;
            }
        }
        for (index, chamber) in self.chambers.iter_mut().enumerate() {
            Self::chamber_step_a(chamber, volumes[index], areas[index], dt, t);
        }
    }

    /// B-phase: vent and link fluxes for the next cycle.
    ///
    /// Evaluates gates, orifice flows, per-gas splits, and
    /// chamber-to-chamber transfers; records time-history outputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// bags.step_a(&[264130.0], &[294658.0], 0.01, 0.01);
    /// bags.step_b(0.01, 0.01);
    /// assert!((bags.sample(0).mass - bags.chamber(0).total_mass).abs() < 1e-12);
    /// ```
    pub fn step_b(&mut self, dt: f64, t: f64) {
        for chamber in &mut self.chambers {
            Self::chamber_vents(chamber, dt, t);
        }
        let snaps: Vec<LinkSnap> = self.chambers.iter().map(LinkSnap::of).collect();
        let mut transfers: Vec<ChamberTransfer> = Vec::new();
        for (index, chamber) in self.chambers.iter_mut().enumerate() {
            let snap = &snaps[index];
            let fractions = chamber.gas_fractions();
            for link in &mut chamber.links {
                let partner = &snaps[link.partner];
                link.update(snap.pressure, partner.pressure, snap.volume, snap.area, t);
                let mut mass = 0.0;
                if link.open
                    && snap.pressure > partner.pressure
                    && snap.volume > MIN_VOLUME_FACTOR * snap.area.powf(1.5)
                {
                    let shocked = partner.pressure.max(snap.pcrit);
                    let energy_density =
                        snap.ro * (shocked / snap.pressure).powf(1.0 / snap.gamma) * snap.hspec;
                    let cap = link_velocity_cap(
                        snap.gamma,
                        snap.pressure,
                        partner.pressure,
                        snap.volume,
                        partner.volume,
                        energy_density,
                        link.area,
                        dt,
                    );
                    let flow = isentropic_orifice(
                        snap.gamma,
                        snap.pressure,
                        partner.pressure,
                        snap.ro,
                        link.area,
                        snap.hspec,
                        cap,
                    );
                    mass = flow.mass_rate;
                }
                if mass > 0.0 {
                    transfers.push(ChamberTransfer {
                        from: index,
                        to: link.partner,
                        mass,
                        fractions: fractions.clone(),
                    });
                }
            }
        }
        for transfer in &transfers {
            let enthalpy = self.chambers[transfer.from].split_enthalpies();
            self.chambers[transfer.from].accumulate_outflow(
                transfer.mass,
                &transfer.fractions,
                &enthalpy,
                1.0,
            );
            self.chambers[transfer.to].accumulate_outflow(
                transfer.mass,
                &transfer.fractions,
                &enthalpy,
                -1.0,
            );
        }
    }

    /// Time-history sample of chamber `index`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Chamber, ChamberConfig, Chambers, Gas};
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     ..Default::default()
    /// };
    /// let bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// assert!((bags.sample(0).pressure - 1.0135e-4).abs() < 1e-12);
    /// ```
    pub fn sample(&self, index: usize) -> ThSample {
        let chamber = &self.chambers[index];
        let vent_area: f64 = chamber.vents.iter().map(|vent| vent.open_area).sum();
        let flow: f64 = chamber
            .vents
            .iter()
            .map(|vent| vent.open_area * vent.velocity)
            .sum();
        ThSample {
            mass: chamber.total_mass,
            volume: chamber.working_volume(),
            pressure: chamber.pressure,
            area: chamber.area,
            temperature: chamber.temperature,
            vent_area,
            vent_velocity: flow / vent_area.max(TINY_MASS),
        }
    }

    /// Time-history sample of one vent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{
    ///     Chamber, ChamberConfig, Chambers, Gas, Vent,
    /// };
    ///
    /// let config = ChamberConfig {
    ///     gas: Some(Gas::new(1.667, [520.275, 0.0, 0.0])),
    ///     pext: 1.0135e-4,
    ///     vents: vec![Vent::new(840.0, 0.0, 0.0, 1e30, 1e20)],
    ///     ..Default::default()
    /// };
    /// let bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
    /// assert!((bags.vent_sample(0, 0).mass_out - 0.0).abs() < 1e-12);
    /// ```
    pub fn vent_sample(&self, index: usize, vent: usize) -> VentSample {
        let state = &self.chambers[index].vents[vent];
        VentSample {
            area: state.open_area,
            velocity: state.velocity,
            mass_out: state.mass_out,
            energy_out: state.energy_out,
        }
    }

    fn chamber_step_a(chamber: &mut Chamber, volume: f64, area: f64, dt: f64, t: f64) {
        let vol = volume + chamber.veps;
        let vold = chamber.working_volume();
        let dv = vol - vold;
        let t_old = chamber.temperature;
        let rnm_old = chamber.rnm;
        let mut left = 0.0;
        let mut right = 0.0;
        let mut total = chamber.total_mass;
        for jet in &mut chamber.injectors {
            let running = t >= jet.injector.start;
            let mut dmass = 0.0;
            if running {
                let inflow = jet.injector.inflow_enthalpy(t);
                let level = if jet.injector.flow_rate {
                    let rate = jet.injector.mass_scale
                        * jet
                            .injector
                            .mass_law
                            .eval((t - jet.injector.start) * jet.injector.time_scale);
                    rate * jet.injector.time_scale * dt + jet.law_level
                } else {
                    jet.injector.mass_at(t)
                };
                dmass = (level - jet.law_level).max(0.0);
                jet.law_level = level;
                right += dmass * inflow;
            } else {
                jet.law_level = 0.0;
            }
            let gas = jet.injector.gas;
            let rmw = jet.rmw;
            let dgmout = jet.out_mass_rate * dt;
            let stored = jet.stored + dmass - dgmout;
            let dgeout = jet.out_energy_rate * dt;
            right -= dgeout;
            right += (dgmout - dmass) * t_old * h_over_t_minus_r(&gas, t_old, rmw);
            let cvg = gas.cv_at(t_old, rmw);
            left += stored * cvg;
            right += stored * cvg * t_old;
            total += stored - jet.stored;
            jet.stored = stored;
        }
        let gas = chamber.gas;
        let rmw = chamber.rmw;
        let dgmout = chamber.out_mass_rate * dt;
        let initial = chamber.initial_mass - dgmout;
        let dgeout = chamber.out_energy_rate * dt;
        let cvi_old = gas.cv_at(t_old, rmw);
        right += dgmout * t_old * h_over_t_minus_r(&gas, t_old, rmw) - dgeout;
        left += initial * cvi_old;
        right += initial * cvi_old * t_old;
        total -= dgmout;
        chamber.initial_mass = initial;
        let mut rnm = initial * rmw;
        for jet in &chamber.injectors {
            rnm += jet.stored * jet.rmw;
        }
        left += 0.5 * rnm * dv / vol;
        right -= 0.5 * rnm_old * t_old * dv / vold;
        chamber.temperature = (right / left).max(0.0);
        chamber.pressure = rnm * chamber.temperature / vol;
        chamber.gamma = chamber.mixture_gamma();
        chamber.rnm = rnm;
        chamber.total_mass = total;
        chamber.volume = volume;
        chamber.area = area;
        chamber.clear_rates();
    }

    fn chamber_vents(chamber: &mut Chamber, dt: f64, t: f64) {
        let volume = chamber.working_volume();
        let area = chamber.area;
        let ro = chamber.total_mass / volume;
        let hspec = chamber.specific_enthalpy();
        for vent in &mut chamber.vents {
            vent.open_area = 0.0;
            vent.velocity = 0.0;
            vent.out_mass_rate = 0.0;
            vent.out_energy_rate = 0.0;
        }
        let mut open_total = 0.0;
        for vent in &mut chamber.vents {
            vent.vent
                .update(chamber.pressure, chamber.pext, volume, area, t);
            vent.open_area = vent
                .vent
                .open_area(chamber.pressure, chamber.pext, volume, area);
            open_total += vent.open_area;
        }
        if open_total <= 0.0 {
            return;
        }
        let roex = ro * (chamber.pext / chamber.pressure).powf(1.0 / chamber.gamma);
        let vmax = vent_velocity_cap(
            chamber.gamma,
            chamber.pressure,
            chamber.pext,
            volume,
            hspec,
            roex,
            open_total,
            dt,
        );
        let flow = isentropic_orifice(
            chamber.gamma,
            chamber.pressure,
            chamber.pext,
            ro,
            open_total,
            hspec,
            vmax,
        );
        for vent in &mut chamber.vents {
            vent.velocity = flow.velocity;
            vent.out_mass_rate = flow.exit_density * flow.velocity * vent.open_area;
            vent.out_energy_rate = vent.out_mass_rate * hspec;
        }
        let fractions = chamber.gas_fractions();
        let enthalpy = chamber.split_enthalpies();
        chamber.accumulate_outflow(flow.mass_rate, &fractions, &enthalpy, 1.0);
    }
}

/// Immutable B-phase view of one chamber for link evaluation.
#[derive(Debug, Clone, Copy)]
struct LinkSnap {
    pressure: f64,
    volume: f64,
    area: f64,
    gamma: f64,
    ro: f64,
    pcrit: f64,
    hspec: f64,
}

impl LinkSnap {
    fn of(chamber: &Chamber) -> Self {
        let volume = chamber.working_volume();
        Self {
            pressure: chamber.pressure,
            volume,
            area: chamber.area,
            gamma: chamber.gamma,
            ro: chamber.total_mass / volume,
            pcrit: chamber.pressure * critical_ratio(chamber.gamma),
            hspec: chamber.specific_enthalpy(),
        }
    }
}

/// Pending chamber-to-chamber mass-energy transfer.
#[derive(Debug, Clone)]
struct ChamberTransfer {
    from: usize,
    to: usize,
    mass: f64,
    fractions: Vec<f64>,
}

impl Chamber {
    /// Mole fractions of the initial bucket plus injector buckets.
    fn gas_fractions(&self) -> Vec<f64> {
        let denom = self.rnm.max(TINY_MASS);
        let mut fractions = Vec::with_capacity(self.injectors.len() + 1);
        fractions.push(self.initial_mass * self.rmw / denom);
        for jet in &self.injectors {
            fractions.push(jet.stored * jet.rmw / denom);
        }
        fractions
    }

    /// Bucket enthalpies matching [`gas_fractions`](Self::gas_fractions).
    fn split_enthalpies(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.injectors.len() + 1);
        out.push(self.gas.enthalpy(self.temperature));
        for jet in &self.injectors {
            out.push(jet.injector.gas.enthalpy(self.temperature));
        }
        out
    }

    /// Adds a signed outflow split to lagged slots.
    ///
    /// Positive `sign` records outflow from this chamber; negative
    /// records inflow into it (same bucket enthalpies, matching the
    /// engine transfer). Partner buckets past this chamber's
    /// injector count are dropped: the engine indexes the partner by
    /// the source jet, which valid decks keep aligned.
    fn accumulate_outflow(&mut self, mass: f64, fractions: &[f64], enthalpy: &[f64], sign: f64) {
        let part = fractions[0] * mass * sign;
        self.out_mass_rate += part;
        self.out_energy_rate += part * enthalpy[0];
        for ((jet, fraction), heat) in self
            .injectors
            .iter_mut()
            .zip(fractions.iter().skip(1))
            .zip(enthalpy.iter().skip(1))
        {
            let leg = fraction * mass * sign;
            jet.out_mass_rate += leg;
            jet.out_energy_rate += leg * heat;
        }
    }
}

/// `h(T)/T - R` expanded for the temperature solve.
///
/// Matches the engine grouping `cpa + cpb*T/2 + cpc*T^2/3 - R`.
fn h_over_t_minus_r(gas: &Gas, t: f64, r: f64) -> f64 {
    let [a, b, c] = gas.cp_coeffs;
    a + 0.5 * b * t + c * t * t / 3.0 - r
}

/// Jetting overpressure sample for one element.
///
/// Carries the geometric factors plus the element overpressure
/// `dP(t)*dP2(theta)*dP3(delta)*max(n.n1, 0)` (manual equation 1126).
/// Nodal load distribution (quarters for quads, thirds for
/// triangles) stays in the structural assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JetSample {
    /// Angle between jet axis and element offset, in degrees.
    pub theta_deg: f64,
    /// Distance from element center to jet segment.
    pub dist: f64,
    /// Facing factor `max(n.n1, 0)`.
    pub facing: f64,
    /// Element overpressure.
    pub overpressure: f64,
}

/// Injector jetting effect on bag elements.
///
/// Models the inflator jet as a directional overpressure shaped by
/// three laws (time, angle in degrees, distance) around the segment
/// `(n1, n3)` aimed by `n2`. A coincident `n3 == n1` reduces the
/// dihedral jet to a conical one, matching the starter mapping of a
/// zero third node.
#[derive(Debug, Clone)]
pub struct Jet {
    /// Jet segment start (zero disables the jet in the deck).
    pub n1: [f64; 3],
    /// Jet aim node.
    pub n2: [f64; 3],
    /// Jet segment end (`n1` selects the conical jet).
    pub n3: [f64; 3],
    /// Time law and scale.
    pub time_law: Tabulated,
    /// Time-law scale factor.
    pub pressure_scale: f64,
    /// Time argument scale.
    pub time_scale: f64,
    /// Angle law (argument in degrees) and scale.
    pub angle_law: Tabulated,
    /// Angle-law scale factor.
    pub angle_scale: f64,
    /// Angle argument scale.
    pub angle_arg_scale: f64,
    /// Distance law and scale.
    pub dist_law: Tabulated,
    /// Distance-law scale factor.
    pub dist_scale: f64,
    /// Distance argument scale.
    pub dist_arg_scale: f64,
    /// Fire time; the jet is quiet before it.
    pub start: f64,
    /// False when the deck leaves the first jet node zero.
    pub enabled: bool,
}

impl Jet {
    /// Samples the jetting overpressure for one element.
    ///
    /// Takes the element center and normal plus the current time and
    /// returns the geometric factors with the overpressure. Disabled
    /// jets and pre-fire times yield zero overpressure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::fluid::airbag::{Jet, Tabulated};
    ///
    /// let jet = Jet {
    ///     n1: [0.0, 0.0, 0.0],
    ///     n2: [0.0, 0.0, 10.0],
    ///     n3: [0.0, 0.0, 0.0],
    ///     time_law: Tabulated::constant(1.0),
    ///     pressure_scale: 1.0,
    ///     time_scale: 1.0,
    ///     angle_law: Tabulated::constant(0.01),
    ///     angle_scale: 1.0,
    ///     angle_arg_scale: 1.0,
    ///     dist_law: Tabulated::constant(0.01),
    ///     dist_scale: 1.0,
    ///     dist_arg_scale: 1.0,
    ///     start: 0.0,
    ///     enabled: true,
    /// };
    /// let sample = jet.sample([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 1.0);
    /// assert!((sample.theta_deg - 0.0).abs() < 1e-9);
    /// assert!((sample.overpressure - 1.0e-4).abs() < 1e-12);
    /// ```
    pub fn sample(&self, center: [f64; 3], normal: [f64; 3], t: f64) -> JetSample {
        if !self.enabled || t < self.start {
            return JetSample {
                theta_deg: 0.0,
                dist: 0.0,
                facing: 0.0,
                overpressure: 0.0,
            };
        }
        let midpoint = [
            0.5 * (self.n1[0] + self.n3[0]),
            0.5 * (self.n1[1] + self.n3[1]),
            0.5 * (self.n1[2] + self.n3[2]),
        ];
        let offset = sub(center, midpoint);
        let axis = sub(self.n2, midpoint);
        let span = sub(self.n3, midpoint);
        let ps1 = dot(offset, axis);
        let ps2 = dot(axis, span);
        let ps3 = dot(offset, span);
        let an2 = dot(axis, axis);
        let an3 = dot(span, span);
        let det = ps2 * ps2 - an2 * an3;
        let guard = det.abs().max(TINY_GEOMETRY);
        let denom = if det < 0.0 { -guard } else { guard };
        let mu = (ps2 * ps1 - ps3 * an2) / denom;
        let clamped = mu.clamp(-1.0, 1.0);
        let radial = sub(offset, scale(span, clamped));
        let an1 = dot(radial, radial).max(0.0);
        let len1 = an1.sqrt();
        let len_normal = dot(normal, normal).max(0.0).sqrt();
        let facing = (dot(radial, normal) / (len_normal * len1).max(TINY_GEOMETRY)).max(0.0);
        let cos_theta =
            (dot(radial, axis) / (an2.max(0.0).sqrt() * len1).max(TINY_GEOMETRY)).clamp(-1.0, 1.0);
        let theta_deg = cos_theta.acos() * 180.0 / std::f64::consts::PI;
        let overpressure = self.pressure_scale
            * self.time_law.eval((t - self.start) * self.time_scale)
            * facing
            * self.angle_scale
            * self.angle_law.eval(theta_deg * self.angle_arg_scale)
            * self.dist_scale
            * self.dist_law.eval(len1 * self.dist_arg_scale);
        JetSample {
            theta_deg,
            dist: len1,
            facing,
            overpressure,
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual {actual} differs from expected {expected} by more than {tol}"
        );
    }

    fn jetting41b_gas() -> Gas {
        Gas::new(1.667, [520.275, 0.0, 0.0])
    }

    fn tank_config() -> ChamberConfig {
        ChamberConfig {
            gas: Some(jetting41b_gas()),
            pext: 1.0135e-4,
            ..Default::default()
        }
    }

    #[test]
    fn gas_obeys_mayer_and_caloric_definitions() {
        let gas = jetting41b_gas();
        let r = gas.specific_gas_constant(295.0);
        close(r, 208.1724, 1e-3);
        close(gas.cp_at(400.0) - gas.cv_at(400.0, r), r, 1e-9);
        close(
            gas.enthalpy(307.61) - gas.energy(307.61, r),
            r * 307.61,
            1e-6,
        );
        let mix = mixture_gamma(&[(1.0, gas, r), (2.0, gas, r)], 300.0);
        close(mix, 1.667, 1e-12);
    }

    #[test]
    fn tabulated_matches_finter_edges() {
        let table = Tabulated::new(vec![1.0, 3.0], vec![1.0, 5.0]);
        close(table.eval(0.0), -1.0, 1e-12);
        close(table.eval(1.0), 1.0, 1e-12);
        close(table.eval(2.0), 3.0, 1e-12);
        close(table.eval(5.0), 9.0, 1e-12);
        close(Tabulated::constant(7.0).eval(-3.0), 7.0, 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn tabulated_rejects_unordered_points() {
        let _ = Tabulated::new(vec![2.0, 1.0], vec![0.0, 1.0]);
    }

    #[test]
    fn critical_ratio_matches_textbook_value() {
        close(critical_ratio(1.4), 0.5282818, 1e-6);
    }

    #[test]
    fn orifice_selects_shocked_branch_and_cap() {
        let gamma = 1.4;
        let unshocked =
            isentropic_orifice(gamma, 2.0e-4, 1.9e-4, 1.2e-9, 840.0, 3.0e5, f64::INFINITY);
        let shocked = isentropic_orifice(gamma, 2.0e-4, 0.0, 1.2e-9, 840.0, 3.0e5, f64::INFINITY);
        assert!(shocked.velocity > unshocked.velocity);
        close(
            shocked.exit_density,
            1.2e-9 * critical_ratio(gamma).powf(1.0 / gamma),
            1e-18,
        );
        let capped = isentropic_orifice(gamma, 2.0e-4, 0.0, 1.2e-9, 840.0, 3.0e5, 10.0);
        close(capped.velocity, 10.0, 1e-12);
        let shut = isentropic_orifice(gamma, 1.0e-4, 2.0e-4, 1.2e-9, 840.0, 3.0e5, f64::INFINITY);
        close(shut.velocity, 0.0, 1e-12);
        close(shut.mass_rate, 0.0, 1e-12);
        close(
            shocked.energy_rate,
            shocked.mass_rate * 3.0e5,
            shocked.mass_rate * 3.0e5 * 1e-12,
        );
    }

    #[test]
    fn vent_gate_opens_on_pressure_and_closes_at_tstop() {
        let mut vent = Vent::new(840.0, 1.0e-6, 0.005, 1e30, 2.0);
        vent.accumulate(1.0135e-4, 1.0135e-4, 0.01);
        close(vent.deflated_time, 0.0, 1e-12);
        vent.accumulate(1.03e-4, 1.0135e-4, 0.01);
        close(vent.deflated_time, 0.01, 1e-12);
        vent.update(1.03e-4, 1.0135e-4, 264009.0, 294656.0, 0.05);
        assert!(vent.deflated);
        close(
            vent.open_area(1.03e-4, 1.0135e-4, 264009.0, 294656.0),
            840.0,
            1e-12,
        );
        close(
            vent.open_area(1.0135e-4, 1.0135e-4, 264009.0, 294656.0),
            0.0,
            1e-12,
        );
        vent.update(1.03e-4, 1.0135e-4, 264009.0, 294656.0, 2.0);
        assert!(!vent.deflated);
    }

    #[test]
    fn link_gate_tracks_partner_pressure() {
        let mut link = Link::new(1, 3958.5, 0.0, 0.0, 1e30);
        link.accumulate(1.03e-4, 1.0135e-4, 0.01);
        link.update(1.03e-4, 1.0135e-4, 264009.0, 294656.0, 0.05);
        assert!(link.open);
    }

    #[test]
    fn starter_presets_match_reader_rules() {
        assert!(vent_starts_open(0.0, 0.0, 0.0));
        assert!(!vent_starts_open(0.0, 0.0, 1e30));
        assert!(link_starts_open(0.0, 0.0, 1e30));
        assert!(!link_starts_open(2e-7, 0.0, 1e30));
        close(small_volume(264009.0, 294656.0), 0.0, 1e-12);
        assert!(small_volume(1.0, 294656.0) > 1.0e4);
    }

    #[test]
    fn chamber_builds_starter_equivalent_state() {
        let bag = Chamber::from_config(&tank_config(), 264009.0, 294656.0);
        close(bag.temperature, 295.0, 1e-12);
        close(bag.pressure, 1.0135e-4, 1e-12);
        close(bag.total_mass, 4.3571e-4, 1e-9);
        close(
            bag.pressure,
            bag.rnm * bag.temperature / bag.working_volume(),
            1e-18,
        );
    }

    #[test]
    fn sealed_tank_holds_state_exactly() {
        let mut bags = Chambers::new(vec![Chamber::from_config(
            &tank_config(),
            264009.0,
            294656.0,
        )]);
        let mass_before = bags.chamber(0).total_mass;
        bags.step(&[264009.0], &[294656.0], 0.01, 0.01);
        let bag = bags.chamber(0);
        close(bag.temperature, 295.0, 1e-9);
        close(bag.pressure, 1.0135e-4, 1e-12);
        close(bag.total_mass, mass_before, 1e-15);
    }

    #[test]
    fn tank_injection_conserves_energy_exactly() {
        let gas = jetting41b_gas();
        let config = ChamberConfig {
            gas: Some(gas),
            pext: 1.0135e-4,
            injectors: vec![Injector {
                gas,
                mass_law: Tabulated::new(vec![0.0, 1.0], vec![0.0, 1.0e-3]),
                temp_law: Tabulated::constant(307.61),
                mass_scale: 1.0,
                temp_scale: 1.0,
                time_scale: 1.0,
                flow_rate: false,
                start: 0.0,
            }],
            ..Default::default()
        };
        let mut bags = Chambers::new(vec![Chamber::from_config(&config, 264009.0, 294656.0)]);
        let r = bags.chamber(0).rmw;
        let cv = gas.cv_at(295.0, r);
        let energy_before = bags.chamber(0).total_mass * cv * 295.0;
        bags.step(&[264009.0], &[294656.0], 0.01, 0.5);
        let bag = bags.chamber(0);
        let injected = 5.0e-4;
        let energy_after = bag.total_mass * cv * bag.temperature;
        close(
            energy_after - energy_before,
            injected * gas.enthalpy(307.61),
            1e-6,
        );
        close(bag.total_mass, 4.3571e-4 + injected, 1e-9);
    }

    #[test]
    fn slow_expansion_tracks_isentrope() {
        let mut bags = Chambers::new(vec![Chamber::from_config(&tank_config(), 1.0e6, 1.0e4)]);
        bags.step(&[1.0001e6], &[1.0e4], 0.01, 0.01);
        let bag = bags.chamber(0);
        let ratio: f64 = 1.0e6 / 1.0001e6;
        let expected = 295.0 * ratio.powf(1.667 - 1.0);
        close(bag.temperature, expected, 1e-6);
    }

    #[test]
    fn chambers_conserve_mass_across_vent_and_link() {
        let gas = jetting41b_gas();
        let injector = Injector {
            gas,
            mass_law: Tabulated::new(vec![0.0, 10.0], vec![0.0, 1.0e-2]),
            temp_law: Tabulated::constant(307.61),
            mass_scale: 1.0,
            temp_scale: 1.0,
            time_scale: 1.0,
            flow_rate: false,
            start: 0.0,
        };
        let config_a = ChamberConfig {
            gas: Some(gas),
            pext: 1.0135e-4,
            injectors: vec![injector],
            vents: vec![Vent::new(840.0, 0.0, 0.0, 1e30, 1e20)],
            links: vec![Link::new(1, 3958.5, 0.0, 0.0, 1e30)],
            ..Default::default()
        };
        let idle = Injector {
            gas,
            mass_law: Tabulated::constant(0.0),
            temp_law: Tabulated::constant(307.61),
            mass_scale: 0.0,
            temp_scale: 1.0,
            time_scale: 1.0,
            flow_rate: false,
            start: 0.0,
        };
        let config_b = ChamberConfig {
            gas: Some(gas),
            pext: 1.0135e-4,
            injectors: vec![idle],
            links: vec![Link::new(0, 3958.5, 0.0, 0.0, 1e30)],
            ..Default::default()
        };
        let bag_a = Chamber::from_config(&config_a, 264009.0, 294656.0);
        let bag_b = Chamber::from_config(&config_b, 243096.0, 269555.0);
        let mass_before = bag_a.total_mass + bag_b.total_mass;
        let mass_b_before = bag_b.total_mass;
        let mut bags = Chambers::new(vec![bag_a, bag_b]);
        for step in 1..=5 {
            let t = f64::from(step) * 0.01;
            bags.step(&[264009.0, 243096.0], &[294656.0, 269555.0], 0.01, t);
        }
        let stored = bags.chamber(0).total_mass + bags.chamber(1).total_mass;
        let vented = bags.chamber(0).vents[0].mass_out;
        let injected = bags.chamber(0).injectors[0].law_level;
        close(stored + vented, mass_before + injected, 1e-9);
        assert!(bags.chamber(1).total_mass > mass_b_before);
        assert!(bags.vent_sample(0, 0).mass_out > 0.0);
    }

    #[test]
    fn jetting_conical_geometry_matches_theory() {
        let jet = Jet {
            n1: [0.0, 0.0, 0.0],
            n2: [0.0, 0.0, 10.0],
            n3: [0.0, 0.0, 0.0],
            time_law: Tabulated::constant(1.0),
            pressure_scale: 1.0,
            time_scale: 1.0,
            angle_law: Tabulated::constant(0.01),
            angle_scale: 1.0,
            angle_arg_scale: 1.0,
            dist_law: Tabulated::constant(0.01),
            dist_scale: 1.0,
            dist_arg_scale: 1.0,
            start: 0.0,
            enabled: true,
        };
        let on_axis = jet.sample([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 1.0);
        close(on_axis.theta_deg, 0.0, 1e-9);
        close(on_axis.dist, 5.0, 1e-12);
        close(on_axis.facing, 1.0, 1e-12);
        close(on_axis.overpressure, 1.0e-4, 1e-12);
        let across = jet.sample([5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0);
        close(across.theta_deg, 90.0, 1e-9);
        let behind = jet.sample([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0);
        close(behind.facing, 0.0, 1e-12);
        close(behind.overpressure, 0.0, 1e-12);
    }

    #[test]
    fn jetting_dihedral_clamps_projection_to_segment() {
        let jet = Jet {
            n1: [-10.0, 0.0, 0.0],
            n2: [0.0, 0.0, 10.0],
            n3: [10.0, 0.0, 0.0],
            time_law: Tabulated::constant(2.0),
            pressure_scale: 1.0,
            time_scale: 1.0,
            angle_law: Tabulated::constant(1.0),
            angle_scale: 1.0,
            angle_arg_scale: 1.0,
            dist_law: Tabulated::constant(1.0),
            dist_scale: 1.0,
            dist_arg_scale: 1.0,
            start: 0.0,
            enabled: true,
        };
        let sample = jet.sample([30.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0);
        close(sample.dist, 20.0, 1e-9);
        close(sample.facing, 1.0, 1e-12);
        close(sample.overpressure, 2.0, 1e-12);
    }

    #[test]
    fn jetting_stays_quiet_when_disabled_or_early() {
        let jet = Jet {
            n1: [0.0, 0.0, 0.0],
            n2: [0.0, 0.0, 10.0],
            n3: [0.0, 0.0, 0.0],
            time_law: Tabulated::constant(1.0),
            pressure_scale: 1.0,
            time_scale: 1.0,
            angle_law: Tabulated::constant(1.0),
            angle_scale: 1.0,
            angle_arg_scale: 1.0,
            dist_law: Tabulated::constant(1.0),
            dist_scale: 1.0,
            dist_arg_scale: 1.0,
            start: 2.0,
            enabled: true,
        };
        close(
            jet.sample([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 1.0)
                .overpressure,
            0.0,
            1e-12,
        );
        let mut off = jet.clone();
        off.enabled = false;
        close(
            off.sample([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 3.0)
                .overpressure,
            0.0,
            1e-12,
        );
    }
}
