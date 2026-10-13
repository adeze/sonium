//! Central-difference crash driver over explicit-dynamics state.
//!
//! Steps [`ExplicitState`](sonium_compute::explicit_state::ExplicitState) with
//! the velocity-Verlet central-difference scheme from [`super::explicit`] (one
//! force evaluation per step, damping evaluated at the half-step velocity),
//! with adaptive step control driven by the Phase 1
//! [`critical_dt`](sonium_compute::explicit_dt::critical_dt) kernel and an
//! energy ledger (kinetic energy, internal energy, external work, drift).
//!
//! ## Adaptive step policy
//!
//! Each adaptive step proposes `dt = safety_factor * dt_crit`, clamps it into
//! `[dt_min, dt_max]`, then applies the growth limiter
//! `dt <= max_growth * dt_previous`. Shrinkage is never limited: a stiffness
//! jump collapses the step immediately, which is what keeps the conditionally
//! stable scheme alive through impact stiffening.
//!
//! ## Energy accounting
//!
//! Forces arrive through two closures in the [`super::explicit`] style —
//! `internal_force(u, v, t)` and `external_force(u, v, t)`, both over the
//! node-major state slices — so the ledger can split the work:
//!
//! ```text
//! dW_int = (F_int^n + F_int^{n+1})/2 . (u^{n+1} - u^n)   (trapezoidal)
//! dW_ext = (F_ext^n + F_ext^{n+1})/2 . (u^{n+1} - u^n)
//! IE    += -dW_int      (stored strain energy)
//! Wext  += dW_ext
//! drift  = (KE - KE_0) + IE - Wext
//! ```
//!
//! Viscous or damping forces placed in the internal closure dissipate into the
//! internal channel (there is no separate heat channel); the balance still
//! closes. The single-closure [`ExplicitStepDriver::step_total`] variant routes
//! all work through the external channel instead.
//!
//! ## Stability margin
//!
//! [`damped_stability_margin`] evaluates the closed-form spectral radius of the
//! damped velocity-Verlet recurrence. On the leapfrog state
//! `[u_n, dt * v_{n-1/2}]` the SDOF update (`Omega = omega*dt`,
//! `xi = c/(2*m*omega)`) is the 2x2 matrix
//!
//! ```text
//! A = [[1 - Omega^2, 1 - 2*xi*Omega], [-Omega^2, 1 - 2*xi*Omega]]
//! ```
//!
//! with trace `T = 2 - Omega^2 - 2*xi*Omega` and determinant
//! `D = 1 - 2*xi*Omega`. The margin is `1 - |A_max|` where `|A_max|` is the
//! largest-modulus eigenvalue: positive inside the stable region, zero on the
//! boundary `Omega = 2*(sqrt(1 + xi^2) - xi)`, negative above it. The W7
//! stability chart replays through this helper.
//!
//! ## Layer rule
//!
//! This module only touches `sonium-compute` state/kernels and force closures;
//! it never imports FEM types (P2C2).

// Rust guideline compliant 2026-02-21

use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_solvers_core::traits::SolverError;

/// Adaptive step-size policy for the crash driver.
///
/// The next step is `safety_factor * dt_crit` clamped into
/// `[dt_min, dt_max]` and limited to `max_growth` times the previous step.
/// See the module docs for the full policy.
///
/// # Examples
///
/// ```rust
/// use sonium_solvers_time_integration::AdaptiveDtConfig;
///
/// let config = AdaptiveDtConfig::new(0.9, 1e-9, 1e-3, 1.1).unwrap();
/// let dt = config.select_dt(1e-3, 1e-4).unwrap();
/// assert!((dt / 9e-5 - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveDtConfig {
    /// Safety factor on the critical step, in `(0, 1]`. Customary 0.8–0.95.
    pub safety_factor: f64,
    /// Hard floor for the step in seconds. Must be positive and finite.
    pub dt_min: f64,
    /// Hard ceiling for the step in seconds. Must cover `dt_min`.
    pub dt_max: f64,
    /// Growth limiter: `dt_next <= max_growth * dt_current`. Must be `>= 1`.
    pub max_growth: f64,
}

impl AdaptiveDtConfig {
    /// Build a validated adaptive step-size policy.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::OperationFailed`] when any field violates its
    /// contract (non-positive or oversized safety factor, non-positive
    /// `dt_min`, `dt_max < dt_min`, or `max_growth < 1`).
    pub fn new(
        safety_factor: f64,
        dt_min: f64,
        dt_max: f64,
        max_growth: f64,
    ) -> Result<Self, SolverError> {
        let config = Self {
            safety_factor,
            dt_min,
            dt_max,
            max_growth,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate the policy fields against their contracts.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::OperationFailed`] on the first violated field.
    pub fn validate(&self) -> Result<(), SolverError> {
        if !self.safety_factor.is_finite() || self.safety_factor <= 0.0 || self.safety_factor > 1.0
        {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt configuration",
                reason: format!(
                    "safety_factor must be finite and in (0, 1], got {}",
                    self.safety_factor
                ),
            });
        }
        if !self.dt_min.is_finite() || self.dt_min <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt configuration",
                reason: format!("dt_min must be finite and positive, got {}", self.dt_min),
            });
        }
        if !self.dt_max.is_finite() || self.dt_max < self.dt_min {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt configuration",
                reason: format!(
                    "dt_max must be finite and >= dt_min ({}), got {}",
                    self.dt_min, self.dt_max
                ),
            });
        }
        if !self.max_growth.is_finite() || self.max_growth < 1.0 {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt configuration",
                reason: format!(
                    "max_growth must be finite and >= 1, got {}",
                    self.max_growth
                ),
            });
        }
        Ok(())
    }

    /// Select the next step from the previous step and critical step.
    ///
    /// Returns `min(clamp(safety_factor * dt_crit, dt_min, dt_max),
    /// max_growth * dt_current)`: the growth limiter never blocks shrinkage,
    /// only regrowth after a stiff event.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::OperationFailed`] when the policy itself is
    /// invalid or either step argument is not finite and positive.
    pub fn select_dt(&self, dt_current: f64, dt_crit: f64) -> Result<f64, SolverError> {
        self.validate()?;
        if !dt_current.is_finite() || dt_current <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt selection",
                reason: format!("dt_current must be finite and positive, got {dt_current}"),
            });
        }
        if !dt_crit.is_finite() || dt_crit <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "adaptive dt selection",
                reason: format!("dt_crit must be finite and positive, got {dt_crit}"),
            });
        }
        let target = (self.safety_factor * dt_crit).clamp(self.dt_min, self.dt_max);
        Ok(target.min(self.max_growth * dt_current))
    }
}

/// Snapshot of the driver's energy ledger at the latest step.
///
/// `drift = (kinetic - kinetic_0) + internal - external_work` closes to zero
/// for an exactly integrated step; its growth bounds the integration error.
/// All quantities are in SI units (joules, seconds).
#[derive(Debug, Clone)]
pub struct EnergyBalance {
    /// Kinetic energy `sum 0.5*m*|v|^2` at the latest step, joules.
    pub kinetic: f64,
    /// Accumulated stored energy `-W_internal`, joules.
    pub internal: f64,
    /// Accumulated external work `W_external`, joules.
    pub external_work: f64,
    /// Balance residual `(KE - KE_0) + IE - Wext`, joules.
    pub drift: f64,
    /// Simulation time of the latest step, seconds.
    pub time: f64,
    /// Completed cycle count of the latest step.
    pub step: u64,
}

/// Damped central-difference stability margin `1 - |A_max|`.
///
/// `omega_dt` is the Courant-like number `omega*dt` of the highest mode and
/// `xi` its damping ratio. Positive means stable, zero marks the stability
/// boundary `Omega = 2*(sqrt(1 + xi^2) - xi)`, negative means the highest
/// mode grows geometrically. At `xi = 0` this reduces to `1 - |A|` of the
/// undamped closed form in
/// [`cd_amplification`](sonium_compute::explicit_dt::cd_amplification).
///
/// # Examples
///
/// ```rust
/// use sonium_solvers_time_integration::damped_stability_margin;
///
/// assert!(damped_stability_margin(1.0, 0.05) > 0.0);
/// assert!(damped_stability_margin(2.5, 0.05) < 0.0);
/// ```
///
/// # Panics
///
/// Panics when `omega_dt` is negative or non-finite, or `xi` is negative or
/// non-finite. Use [`try_damped_stability_margin`] for a `Result` instead.
pub fn damped_stability_margin(omega_dt: f64, xi: f64) -> f64 {
    try_damped_stability_margin(omega_dt, xi).expect("omega_dt/xi must be valid")
}

/// Fallible damped central-difference stability margin `1 - |A_max|`.
///
/// Closed form over the velocity-Verlet amplification matrix documented at the
/// top of this module: complex-conjugate roots take `|A_max| = sqrt(det)`,
/// real roots take the larger-modulus root.
///
/// # Errors
///
/// Returns [`SolverError::OperationFailed`] when `omega_dt` is negative or
/// non-finite, or `xi` is negative or non-finite.
pub fn try_damped_stability_margin(omega_dt: f64, xi: f64) -> Result<f64, SolverError> {
    if !omega_dt.is_finite() || omega_dt < 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "damped_stability_margin",
            reason: format!("omega_dt must be finite and non-negative, got {omega_dt}"),
        });
    }
    if !xi.is_finite() || xi < 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "damped_stability_margin",
            reason: format!("damping ratio must be finite and non-negative, got {xi}"),
        });
    }
    let trace = 2.0 - omega_dt * omega_dt - 2.0 * xi * omega_dt;
    let det = 1.0 - 2.0 * xi * omega_dt;
    // A non-positive discriminant implies det >= trace^2/4 >= 0, so the
    // square root below is always real.
    let discriminant = trace * trace - 4.0 * det;
    let radius = if discriminant <= 0.0 {
        det.sqrt()
    } else {
        let root = discriminant.sqrt();
        (0.5 * (trace + root))
            .abs()
            .max((0.5 * (trace - root)).abs())
    };
    Ok(1.0 - radius)
}

/// Crash driver stepping [`ExplicitState`] with force closures.
///
/// Owns the per-cycle state the scheme needs beyond [`ExplicitState`] itself
/// (current accelerations, previous force vectors for the trapezoidal work
/// terms, the energy ledger), so repeated [`step`](Self::step) calls march a
/// full run. The driver caches the degree-of-freedom count on the first step
/// and rejects any later state with a different layout.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_state::ExplicitState;
/// use sonium_solvers_time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
///
/// let config = AdaptiveDtConfig::new(0.9, 1e-9, 1e-3, 1.1).unwrap();
/// let mut driver = ExplicitStepDriver::new(config).unwrap();
/// let mut state = ExplicitState::new(1);
/// state.masses[0] = 1.0;
/// state.velocities[0] = 1.0;
/// let zero = |_: &[f64], _: &[f64], _: f64| vec![0.0; 3];
/// driver.step(&mut state, zero, zero, &[0.1], &[1000.0]).unwrap();
/// assert_eq!(state.step, 1);
/// assert!(driver.balance().drift.abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct ExplicitStepDriver {
    config: AdaptiveDtConfig,
    dt: f64,
    dofs: usize,
    accelerations: Vec<f64>,
    prev_internal: Vec<f64>,
    prev_external: Vec<f64>,
    kinetic0: f64,
    balance: EnergyBalance,
    initialized: bool,
}

impl ExplicitStepDriver {
    /// Create a driver from a validated adaptive policy.
    ///
    /// The first adaptive step starts from `dt_max`, so the growth limiter
    /// never clamps the opening step below the safety-scaled critical value.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::OperationFailed`] when the policy is invalid.
    pub fn new(config: AdaptiveDtConfig) -> Result<Self, SolverError> {
        config.validate()?;
        Ok(Self {
            dt: config.dt_max,
            config,
            dofs: 0,
            accelerations: Vec::new(),
            prev_internal: Vec::new(),
            prev_external: Vec::new(),
            kinetic0: 0.0,
            balance: EnergyBalance {
                kinetic: 0.0,
                internal: 0.0,
                external_work: 0.0,
                drift: 0.0,
                time: 0.0,
                step: 0,
            },
            initialized: false,
        })
    }

    /// Current step size in seconds.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Latest energy-ledger snapshot (zeros before the first step).
    #[must_use]
    pub fn balance(&self) -> &EnergyBalance {
        &self.balance
    }

    /// Advance one adaptive step with split force closures.
    ///
    /// Selects `dt` from `char_lengths`/`sound_speeds` via
    /// [`critical_dt`] and the [`AdaptiveDtConfig`] policy, then marches the
    /// velocity-Verlet update with `internal_force(u, v_half, t)` and
    /// `external_force(u, v_half, t)` evaluated at the new displacement and
    /// half-step velocity (initial forces use the full-step velocity). Returns
    /// the step size used.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the policy, state, mesh slices, closure
    /// outputs, or resulting state are invalid (see
    /// [`step_fixed`](Self::step_fixed) for the shared conditions).
    pub fn step<F, G>(
        &mut self,
        state: &mut ExplicitState,
        internal_force: F,
        external_force: G,
        char_lengths: &[f64],
        sound_speeds: &[f64],
    ) -> Result<f64, SolverError>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        self.config.validate()?;
        let dt_crit = checked_critical_dt(char_lengths, sound_speeds)?;
        let dt = self.config.select_dt(self.dt, dt_crit)?;
        self.step_with_dt(state, internal_force, external_force, dt)?;
        Ok(dt)
    }

    /// Advance one fixed-size step with split force closures.
    ///
    /// Same update as [`step`](Self::step) but uses the caller-supplied `dt`
    /// instead of the adaptive policy (still recorded for the next adaptive
    /// step's growth limiter).
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::DimensionMismatch`] when a closure returns the
    /// wrong length or the state layout changed since the first step, and
    /// [`SolverError::OperationFailed`] when the policy is invalid, `dt` is
    /// outside `[dt_min, dt_max]` or non-positive, the state is inconsistent
    /// or has non-positive masses, a force vector is non-finite, or the
    /// updated state is non-finite (an above-limit step blowing up).
    pub fn step_fixed<F, G>(
        &mut self,
        state: &mut ExplicitState,
        internal_force: F,
        external_force: G,
        dt: f64,
    ) -> Result<f64, SolverError>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        self.config.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "explicit step driver",
                reason: format!("dt must be finite and positive, got {dt}"),
            });
        }
        if dt < self.config.dt_min || dt > self.config.dt_max {
            return Err(SolverError::OperationFailed {
                operation: "explicit step driver",
                reason: format!(
                    "dt {dt} is outside [dt_min, dt_max] = [{}, {}]",
                    self.config.dt_min, self.config.dt_max
                ),
            });
        }
        self.step_with_dt(state, internal_force, external_force, dt)?;
        Ok(dt)
    }

    /// Advance one adaptive step with a single total-force closure.
    ///
    /// Routes all work through the external channel (`internal` stays zero);
    /// the drift identity still closes. Prefer [`step`](Self::step) when the
    /// caller can split internal and external forces.
    ///
    /// # Errors
    ///
    /// Same conditions as [`step`](Self::step).
    pub fn step_total<F>(
        &mut self,
        state: &mut ExplicitState,
        force: F,
        char_lengths: &[f64],
        sound_speeds: &[f64],
    ) -> Result<f64, SolverError>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let dofs = state.positions.len();
        let zero = |_: &[f64], _: &[f64], _: f64| vec![0.0; dofs];
        self.step(state, zero, force, char_lengths, sound_speeds)
    }

    /// Advance one fixed-size step with a single total-force closure.
    ///
    /// Fixed-`dt` variant of [`step_total`](Self::step_total).
    ///
    /// # Errors
    ///
    /// Same conditions as [`step_fixed`](Self::step_fixed).
    pub fn step_total_fixed<F>(
        &mut self,
        state: &mut ExplicitState,
        force: F,
        dt: f64,
    ) -> Result<f64, SolverError>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let dofs = state.positions.len();
        let zero = |_: &[f64], _: &[f64], _: f64| vec![0.0; dofs];
        self.step_fixed(state, zero, force, dt)
    }

    /// Velocity-Verlet update plus energy-ledger accounting at fixed `dt`.
    fn step_with_dt<F, G>(
        &mut self,
        state: &mut ExplicitState,
        internal_force: F,
        external_force: G,
        dt: f64,
    ) -> Result<(), SolverError>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        const OPERATION: &str = "explicit step driver";
        if !state.is_consistent() {
            return Err(SolverError::OperationFailed {
                operation: OPERATION,
                reason: "ExplicitState layout is inconsistent".to_string(),
            });
        }
        for (node, mass) in state.masses.iter().enumerate() {
            if !mass.is_finite() || *mass <= 0.0 {
                return Err(SolverError::OperationFailed {
                    operation: OPERATION,
                    reason: format!("mass of node {node} must be finite and positive, got {mass}"),
                });
            }
        }
        if !state.time.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: OPERATION,
                reason: format!("state time must be finite, got {}", state.time),
            });
        }
        let dofs = state.positions.len();
        if self.initialized && self.dofs != dofs {
            return Err(SolverError::DimensionMismatch {
                expected: self.dofs,
                got: dofs,
            });
        }
        let time_next = state.time + dt;
        if !time_next.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: OPERATION,
                reason: "time + dt overflowed to a non-finite value".to_string(),
            });
        }

        // Consistent initial acceleration plus the trapezoidal baseline.
        if !self.initialized {
            let internal0 = checked_force(
                &internal_force,
                &state.positions,
                &state.velocities,
                state.time,
                dofs,
                "internal force output",
            )?;
            let external0 = checked_force(
                &external_force,
                &state.positions,
                &state.velocities,
                state.time,
                dofs,
                "external force output",
            )?;
            let mut accelerations = vec![0.0; dofs];
            mass_solve_into(&internal0, &external0, &state.masses, &mut accelerations);
            self.kinetic0 = kinetic_energy(&state.velocities, &state.masses);
            self.balance.kinetic = self.kinetic0;
            self.balance.time = state.time;
            self.balance.step = state.step;
            self.accelerations = accelerations;
            self.prev_internal = internal0;
            self.prev_external = external0;
            self.dofs = dofs;
            self.initialized = true;
        }

        // u_{n+1} = u_n + dt*v_n + (dt^2/2)*a_n.
        let mut positions_next = vec![0.0; dofs];
        for ((next, (pos, vel)), acc) in positions_next
            .iter_mut()
            .zip(state.positions.iter().zip(state.velocities.iter()))
            .zip(self.accelerations.iter())
        {
            *next = *pos + dt * *vel + 0.5 * dt * dt * *acc;
        }
        // v_{n+1/2} = v_n + (dt/2)*a_n.
        let mut velocities_half = vec![0.0; dofs];
        for ((half, vel), acc) in velocities_half
            .iter_mut()
            .zip(state.velocities.iter())
            .zip(self.accelerations.iter())
        {
            *half = *vel + 0.5 * dt * *acc;
        }

        // a_{n+1} = M^{-1} F(u_{n+1}, v_{n+1/2}, t_{n+1}).
        let internal1 = checked_force(
            &internal_force,
            &positions_next,
            &velocities_half,
            time_next,
            dofs,
            "internal force output",
        )?;
        let external1 = checked_force(
            &external_force,
            &positions_next,
            &velocities_half,
            time_next,
            dofs,
            "external force output",
        )?;
        let mut accelerations_next = vec![0.0; dofs];
        mass_solve_into(
            &internal1,
            &external1,
            &state.masses,
            &mut accelerations_next,
        );

        // v_{n+1} = v_{n+1/2} + (dt/2)*a_{n+1}.
        for (half, acc) in velocities_half.iter_mut().zip(accelerations_next.iter()) {
            *half += 0.5 * dt * *acc;
        }
        if positions_next.iter().any(|x| !x.is_finite())
            || velocities_half.iter().any(|x| !x.is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: OPERATION,
                reason: "updated state is non-finite (dt above the stability limit?)".to_string(),
            });
        }

        // Trapezoidal work of each channel against the displacement increment.
        let mut work_internal = 0.0;
        let mut work_external = 0.0;
        let internal_pairs = self.prev_internal.iter().zip(internal1.iter());
        let external_pairs = self.prev_external.iter().zip(external1.iter());
        let steps = positions_next.iter().zip(state.positions.iter());
        for (((prev_i, next_i), (prev_e, next_e)), (next_pos, pos)) in
            internal_pairs.zip(external_pairs).zip(steps)
        {
            let displacement = next_pos - pos;
            work_internal += 0.5 * (prev_i + next_i) * displacement;
            work_external += 0.5 * (prev_e + next_e) * displacement;
        }
        let kinetic = kinetic_energy(&velocities_half, &state.masses);
        self.balance.internal += -work_internal;
        self.balance.external_work += work_external;
        self.balance.kinetic = kinetic;
        self.balance.drift =
            (kinetic - self.kinetic0) + self.balance.internal - self.balance.external_work;

        state.positions.copy_from_slice(&positions_next);
        state.velocities.copy_from_slice(&velocities_half);
        state.time = time_next;
        state.step += 1;
        state.dt = dt;
        self.balance.time = time_next;
        self.balance.step = state.step;
        self.accelerations = accelerations_next;
        self.prev_internal = internal1;
        self.prev_external = external1;
        self.dt = dt;
        Ok(())
    }
}

/// Total kinetic energy `sum 0.5*m*|v|^2` over node-major velocities.
fn kinetic_energy(velocities: &[f64], masses: &[f64]) -> f64 {
    let mut kinetic = 0.0;
    for (node, mass) in masses.iter().enumerate() {
        let base = 3 * node;
        kinetic += 0.5
            * mass
            * (velocities[base] * velocities[base]
                + velocities[base + 1] * velocities[base + 1]
                + velocities[base + 2] * velocities[base + 2]);
    }
    kinetic
}

/// Lumped-mass solve `a = M^{-1} (F_int + F_ext)` into a scratch vector.
fn mass_solve_into(internal: &[f64], external: &[f64], masses: &[f64], out: &mut [f64]) {
    for (node, mass) in masses.iter().enumerate() {
        let base = 3 * node;
        out[base] = (internal[base] + external[base]) / mass;
        out[base + 1] = (internal[base + 1] + external[base + 1]) / mass;
        out[base + 2] = (internal[base + 2] + external[base + 2]) / mass;
    }
}

/// Evaluate a force closure with length and finiteness checks.
fn checked_force<F>(
    force: &F,
    positions: &[f64],
    velocities: &[f64],
    time: f64,
    dofs: usize,
    role: &'static str,
) -> Result<Vec<f64>, SolverError>
where
    F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
{
    let values = force(positions, velocities, time);
    if values.len() != dofs {
        return Err(SolverError::DimensionMismatch {
            expected: dofs,
            got: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "explicit step driver",
            reason: format!("{role} is non-finite at index {index}"),
        });
    }
    Ok(values)
}

/// Validate mesh slices, then delegate to the Phase 1 kernel.
fn checked_critical_dt(char_lengths: &[f64], sound_speeds: &[f64]) -> Result<f64, SolverError> {
    const OPERATION: &str = "explicit step driver";
    if char_lengths.is_empty() {
        return Err(SolverError::OperationFailed {
            operation: OPERATION,
            reason: "adaptive stepping needs at least one element".to_string(),
        });
    }
    if char_lengths.len() != sound_speeds.len() {
        return Err(SolverError::DimensionMismatch {
            expected: char_lengths.len(),
            got: sound_speeds.len(),
        });
    }
    for (&length, &speed) in char_lengths.iter().zip(sound_speeds.iter()) {
        if !length.is_finite() || length <= 0.0 || !speed.is_finite() || speed <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: OPERATION,
                reason: format!(
                    "characteristic lengths and sound speeds must be finite and positive, got {length}/{speed}"
                ),
            });
        }
    }
    Ok(critical_dt(char_lengths, sound_speeds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::explicit_dt::cd_amplification;

    /// Damped stability boundary `Omega_crit = 2*(sqrt(1 + xi^2) - xi)`.
    fn omega_crit(xi: f64) -> f64 {
        2.0 * ((1.0 + xi * xi).sqrt() - xi)
    }

    /// Single-node state with unit mass, displaced in x, at rest.
    fn unit_oscillator(displacement: f64) -> ExplicitState {
        let mut state = ExplicitState::new(1);
        state.masses[0] = 1.0;
        state.positions[0] = displacement;
        state
    }

    /// Spring internal force `-k*u_x` on the x degree of freedom.
    fn spring_internal(stiffness: f64) -> impl Fn(&[f64], &[f64], f64) -> Vec<f64> + Copy {
        move |positions: &[f64], _: &[f64], _: f64| vec![-stiffness * positions[0], 0.0, 0.0]
    }

    fn zero_force() -> impl Fn(&[f64], &[f64], f64) -> Vec<f64> + Copy {
        |_: &[f64], _: &[f64], _: f64| vec![0.0; 3]
    }

    fn test_config() -> AdaptiveDtConfig {
        AdaptiveDtConfig::new(0.9, 1e-12, 1.0, 1.1).unwrap()
    }

    #[test]
    fn margin_matches_undamped_closed_form() {
        for omega in [0.0, 0.1, 0.5, 1.0, 1.99, 2.0, 2.01, 3.0] {
            let (abs, _) = cd_amplification(omega);
            let margin = try_damped_stability_margin(omega, 0.0).unwrap();
            assert!(
                (margin - (1.0 - abs)).abs() <= 1e-14 * (1.0 + abs),
                "omega {omega}: margin {margin} vs 1 - |A| = {}",
                1.0 - abs
            );
        }
    }

    #[test]
    fn margin_zero_on_damped_boundary() {
        for xi in [0.0, 0.02, 0.1, 0.5] {
            let critical = omega_crit(xi);
            let margin = damped_stability_margin(critical, xi);
            assert!(
                margin.abs() < 1e-12,
                "xi {xi}: boundary margin {margin} should vanish"
            );
            // Undamped interior is neutrally stable (|A| = 1 exactly);
            // damping makes it strictly contractive.
            let interior = damped_stability_margin(0.5 * critical, xi);
            if xi == 0.0 {
                assert!(
                    interior.abs() < 1e-15,
                    "undamped interior margin {interior} should vanish"
                );
            } else {
                assert!(interior > 0.0, "xi {xi}: interior must be stable");
            }
            assert!(
                damped_stability_margin(1.5 * critical, xi) < 0.0,
                "xi {xi}: exterior must be unstable"
            );
        }
    }

    #[test]
    fn margin_rejects_bad_inputs() {
        for (omega, xi) in [
            (-0.1, 0.0),
            (1.0, -0.1),
            (f64::NAN, 0.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(
                try_damped_stability_margin(omega, xi).is_err(),
                "omega {omega}, xi {xi} must be rejected"
            );
        }
    }

    #[test]
    #[should_panic(expected = "omega_dt/xi must be valid")]
    fn margin_panicking_wrapper_rejects_bad_inputs() {
        let _ = damped_stability_margin(-1.0, 0.0);
    }

    #[test]
    fn config_validation_rejects_bad_values() {
        assert!(AdaptiveDtConfig::new(0.0, 1e-9, 1e-3, 1.1).is_err());
        assert!(AdaptiveDtConfig::new(1.5, 1e-9, 1e-3, 1.1).is_err());
        assert!(AdaptiveDtConfig::new(0.9, 0.0, 1e-3, 1.1).is_err());
        assert!(AdaptiveDtConfig::new(0.9, 1e-3, 1e-9, 1.1).is_err());
        assert!(AdaptiveDtConfig::new(0.9, 1e-9, 1e-3, 0.5).is_err());
        assert!(AdaptiveDtConfig::new(f64::NAN, 1e-9, 1e-3, 1.1).is_err());
    }

    #[test]
    fn adaptive_dt_selects_safety_times_critical() {
        let config = AdaptiveDtConfig::new(0.9, 1e-9, 1e-3, 2.0).unwrap();
        // Well below the growth cap: exact safety-scaled critical step.
        let dt = config.select_dt(1e-3, 1e-4).unwrap();
        assert!((dt / 9e-5 - 1.0).abs() < 1e-12, "got {dt}");
        // Growth limiter binds on regrowth: min(9e-4, 2*1e-4).
        let dt = config.select_dt(1e-4, 1e-3).unwrap();
        assert!((dt / 2e-4 - 1.0).abs() < 1e-12, "got {dt}");
        // Clamps bind outside [dt_min, dt_max].
        assert_eq!(config.select_dt(1e-3, 1.0).unwrap(), 1e-3);
        assert_eq!(config.select_dt(1e-3, 1e-12).unwrap(), 1e-9);
    }

    #[test]
    fn adaptive_shrink_on_stiffness_jump() {
        let mut driver = ExplicitStepDriver::new(test_config()).unwrap();
        let mut state = unit_oscillator(1.0);
        let spring = spring_internal(100.0);
        let zero = zero_force();
        // Soft mesh: dt = 0.9 * 0.1/1000.
        let dt_soft = driver
            .step(&mut state, spring, zero, &[0.1], &[1000.0])
            .unwrap();
        assert!((dt_soft / 9e-5 - 1.0).abs() < 1e-12, "got {dt_soft}");
        for _ in 0..50 {
            driver
                .step(&mut state, spring, zero, &[0.1], &[1000.0])
                .unwrap();
        }
        // Tenfold stiffening: the step collapses tenfold at once (no lower
        // limiter), instead of ratcheting down 10% per step.
        let dt_stiff = driver
            .step(&mut state, spring, zero, &[0.1], &[10000.0])
            .unwrap();
        assert!((dt_stiff / 9e-6 - 1.0).abs() < 1e-12, "got {dt_stiff}");
        assert!(state.positions[0].is_finite());
        // Softening again: regrowth is capped by the limiter.
        let dt_regrow = driver
            .step(&mut state, spring, zero, &[0.1], &[1000.0])
            .unwrap();
        assert!(
            (dt_regrow / (1.1 * dt_stiff) - 1.0).abs() < 1e-12,
            "got {dt_regrow}"
        );
    }

    #[test]
    fn undamped_energy_conservation() {
        // Unit oscillator, k = 4 (omega = 2): 200 steps per period.
        let stiffness = 4.0;
        let period = std::f64::consts::PI;
        let dt = period / 200.0;
        let mut driver = ExplicitStepDriver::new(test_config()).unwrap();
        let mut state = unit_oscillator(1.0);
        let spring = spring_internal(stiffness);
        let zero = zero_force();
        // Initial total energy 0.5*k*u^2 = 2.
        let mut drifts = Vec::new();
        for _ in 0..1000 {
            driver.step_fixed(&mut state, spring, zero, dt).unwrap();
            drifts.push(driver.balance().drift);
        }
        let scale = 2.0;
        let max_drift = drifts.iter().map(|drift| drift.abs()).fold(0.0, f64::max);
        assert!(
            max_drift / scale < 1e-3,
            "relative drift {max_drift} exceeds the bound"
        );
        // No secular growth: the second half drifts no more than twice the
        // first half (symplectic bounded oscillation, not an energy leak).
        let first = drifts[..500]
            .iter()
            .map(|drift| drift.abs())
            .fold(0.0, f64::max);
        let second = drifts[500..]
            .iter()
            .map(|drift| drift.abs())
            .fold(0.0, f64::max);
        assert!(
            second <= 2.0 * first.max(1e-15),
            "drift grows secularly: first {first}, second {second}"
        );
        // Internal channel holds the strain-energy oscillation.
        assert!(driver.balance().internal.abs() > 0.0);
        assert_eq!(driver.balance().external_work, 0.0);
    }

    #[test]
    fn drift_accounting_with_external_work() {
        // Free unit mass under a constant 3 N load: velocity-Verlet is exact,
        // so the reported work and drift close to roundoff.
        let mut driver = ExplicitStepDriver::new(test_config()).unwrap();
        let mut state = ExplicitState::new(1);
        state.masses[0] = 2.0;
        let zero = zero_force();
        let load = |_: &[f64], _: &[f64], _: f64| vec![3.0, 0.0, 0.0];
        for _ in 0..100 {
            driver.step_fixed(&mut state, zero, load, 0.01).unwrap();
        }
        let balance = driver.balance().clone();
        let expected_work = 3.0 * state.positions[0];
        assert!(
            (balance.external_work / expected_work - 1.0).abs() < 1e-12,
            "work {} vs {}",
            balance.external_work,
            expected_work
        );
        assert_eq!(balance.internal, 0.0);
        assert!(
            balance.drift.abs() / expected_work.abs() < 1e-12,
            "drift {}",
            balance.drift
        );
    }

    #[test]
    fn single_closure_routes_work_externally() {
        let mut driver = ExplicitStepDriver::new(test_config()).unwrap();
        let mut state = unit_oscillator(1.0);
        let spring = spring_internal(4.0);
        for _ in 0..10 {
            driver.step_total_fixed(&mut state, spring, 0.01).unwrap();
        }
        assert_eq!(driver.balance().internal, 0.0);
        assert!(driver.balance().external_work.abs() > 0.0);
        // Ledger closes far below the energy scale (0.5*k*u^2 = 2).
        assert!(driver.balance().drift.abs() / 2.0 < 1e-4);
    }

    #[test]
    fn damped_scheme_matches_stability_prediction() {
        // The closed-form matrix must predict the actual velocity-Verlet
        // scheme in `super::explicit`: bounded below the damped limit,
        // geometric growth above it.
        use crate::explicit::{ExplicitConfig, critical_time_step_damped, explicit_integrate};
        use crate::newmark::NewmarkInitial;

        let omega = 2.0;
        let xi = 0.05;
        let (mass, stiffness, damping) = (1.0, omega * omega, 2.0 * xi * omega);
        let dt_crit = critical_time_step_damped(omega, xi);
        // The seed helper and the margin boundary must agree.
        assert!((omega_crit(xi) / omega - dt_crit).abs() / dt_crit < 1e-15);
        let initial = || NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![0.0],
        };
        let force = |u: &[f64], v: &[f64], _t: f64| vec![-damping * v[0] - stiffness * u[0]];
        let mass_solve = |rhs: &[f64]| vec![rhs[0] / mass];

        let stable = explicit_integrate(
            force,
            mass_solve,
            &initial(),
            &ExplicitConfig::new(0.9 * dt_crit, 2000),
        );
        assert!(
            stable.displacement[0].abs() < 2.0,
            "stable run diverged: {}",
            stable.displacement[0]
        );

        // Steps for a 1e9 blow-up at the predicted rate (asserted at 1e6,
        // so a small unstable-mode projection still trips the assertion).
        let growth = 1.0 - damped_stability_margin(1.1 * omega_crit(xi), xi);
        assert!(growth > 1.0);
        let steps = (1e9_f64.ln() / growth.ln()).ceil() as usize + 5;
        let unstable = explicit_integrate(
            force,
            mass_solve,
            &initial(),
            &ExplicitConfig::new(1.1 * dt_crit, steps),
        );
        assert!(
            unstable.displacement[0].abs() > 1e6,
            "unstable run stayed bounded: {}",
            unstable.displacement[0]
        );
    }

    #[test]
    fn step_rejects_bad_state_mesh_and_forces() {
        let mut driver = ExplicitStepDriver::new(test_config()).unwrap();
        let mut state = unit_oscillator(1.0);
        let spring = spring_internal(4.0);
        let zero = zero_force();

        // Empty and mismatched mesh slices.
        assert!(driver.step(&mut state, spring, zero, &[], &[]).is_err());
        assert!(
            driver
                .step(&mut state, spring, zero, &[0.1, 0.2], &[1000.0])
                .is_err()
        );
        assert!(
            driver
                .step(&mut state, spring, zero, &[0.1], &[-1000.0])
                .is_err()
        );
        // Non-positive mass.
        state.masses[0] = 0.0;
        assert!(
            driver
                .step(&mut state, spring, zero, &[0.1], &[1000.0])
                .is_err()
        );
        state.masses[0] = 1.0;
        // Wrong-length and non-finite closure outputs.
        let short = |_: &[f64], _: &[f64], _: f64| vec![0.0; 2];
        assert!(
            driver
                .step(&mut state, short, zero, &[0.1], &[1000.0])
                .is_err()
        );
        let infinite = |_: &[f64], _: &[f64], _: f64| vec![f64::INFINITY, 0.0, 0.0];
        assert!(
            driver
                .step(&mut state, infinite, zero, &[0.1], &[1000.0])
                .is_err()
        );
        // Fixed dt outside the clamps.
        assert!(driver.step_fixed(&mut state, spring, zero, 0.0).is_err());
        assert!(driver.step_fixed(&mut state, spring, zero, 2.0).is_err());
        // Layout change after the first step.
        driver.step_fixed(&mut state, spring, zero, 0.01).unwrap();
        state.velocities.pop();
        assert!(driver.step_fixed(&mut state, spring, zero, 0.01).is_err());
    }
}
