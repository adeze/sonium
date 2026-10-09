//! Multidimensional Cartesian DRP propagation of linearized Euler perturbations.
//!
//! This solver advances pressure and particle-velocity perturbations about
//! a uniform mean flow on two- and three-dimensional Cartesian grids:
//!
//! ```text
//! dp/dt + U·∇p + rho c^2 ∇·u = 0,
//! du/dt + (U·∇)u + (1/rho) ∇p = 0.
//! ```
//!
//! Spatial derivatives use the seven-point Tam--Webb DRP stencil along
//! each axis through ghosted grid lines
//! ([`DrpStencil::derivative_ghosted_line`]), so every face carries an
//! explicit boundary treatment: periodic wrap or outflow with a sponge
//! layer. Time integration is classical RK4 with preallocated scratch,
//! matching the one-dimensional reference in [`super::drp_euler`].
//!
//! Stability follows a Gershgorin-style rate bound: with `K` the stencil
//! modified-wavenumber bound (`2·Σ|a_j|`, a rigorous triangle-inequality
//! ceiling), the semi-discrete spectral radius obeys
//! `|λ| ≤ K·Σ_d (|U_d| + c) / dx_d`, and `dt = CFL / rate` with
//! `CFL ≤ 1` keeps `|λ·dt|` inside the RK4 imaginary-axis limit.
//!
//! # Reference
//!
//! C. K. W. Tam and J. C. Webb, "Dispersion-Relation-Preserving Finite
//! Difference Schemes for Computational Acoustics", *Journal of Computational
//! Physics* 107 (1993), 262--281.

// Rust guideline compliant 2026-02-21

use super::drp_euler::{AcousticBranch, DrpStencil, NumericalDispersion, TAM_WEBB_DRP_7};
use super::error::{AeroacousticsError, require_finite, require_positive};

/// Boundary treatment applied at one Cartesian face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StructuredFace {
    /// Periodic wrap into the ghost cells.
    Periodic,
    /// Zero-gradient exterior with sponge absorption ahead of the face.
    Outflow {
        /// Sponge thickness in grid points from the face.
        sponge_points: usize,
        /// Peak sponge rate `[1/s]` at the face.
        sponge_strength: f64,
    },
}

/// One Cartesian axis of a structured DRP grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredAxis {
    /// Interior node count (nodes span `[0, length)` with spacing `dx`).
    pub n_points: usize,
    /// Domain extent `[m]`.
    pub length: f64,
    /// Treatment at the lower face.
    pub lower: StructuredFace,
    /// Treatment at the upper face.
    pub upper: StructuredFace,
}

/// Configuration of a multidimensional Cartesian DRP LEE grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DrpStructuredConfig {
    /// Grid axes (two or three entries).
    pub axes: Vec<StructuredAxis>,
    /// Uniform mean density `[kg/m^3]`.
    pub density: f64,
    /// Uniform sound speed `[m/s]`.
    pub speed_of_sound: f64,
    /// Uniform mean-flow velocity per axis `[m/s]`.
    pub mean_velocity: Vec<f64>,
    /// Explicit RK4 step `[s]`.
    pub time_step: f64,
}

/// Semi-discrete dispersion of an oblique plane wave.
#[derive(Debug, Clone)]
pub struct PlaneWaveDispersion {
    /// Scalar dispersion along the propagation direction.
    pub dispersion: NumericalDispersion,
    /// Group-velocity vector `[m/s]`.
    pub group_velocity: Vec<f64>,
}

/// Rigorous ceiling of the stencil modified wavenumber `k*·dx`.
///
/// The triangle inequality bounds `|2·Σ a_j sin(j·θ)|` by `2·Σ|a_j|`,
/// evaluated here from the stencil coefficients so the stability bound
/// can never drift from the stencil it guards.
fn modified_wavenumber_bound() -> f64 {
    2.0 * TAM_WEBB_DRP_7
        .coefficients
        .iter()
        .map(|coefficient| coefficient.abs())
        .sum::<f64>()
}

impl DrpStructuredConfig {
    /// Construct a grid and select `dt` from a CFL number.
    ///
    /// The step is `dt = CFL / rate` with the Gershgorin-style
    /// characteristic rate, so `CFL ≤ 1` is stable by construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_structured::{
    ///     DrpStructuredConfig, StructuredAxis, StructuredFace,
    /// };
    ///
    /// let axis = StructuredAxis {
    ///     n_points: 32,
    ///     length: 1.0,
    ///     lower: StructuredFace::Periodic,
    ///     upper: StructuredFace::Periodic,
    /// };
    /// let config =
    ///     DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![10.0, 0.0], 0.5)
    ///         .unwrap();
    /// assert!(config.time_step > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for a dimension count outside
    /// `{2, 3}`, a short axis, a non-positive parameter, a CFL number
    /// outside `(0, 1]`, or a sponge layer covering the whole axis.
    pub fn with_cfl(
        axes: Vec<StructuredAxis>,
        density: f64,
        speed_of_sound: f64,
        mean_velocity: Vec<f64>,
        cfl: f64,
    ) -> Result<Self, AeroacousticsError> {
        let candidate = Self {
            axes,
            density,
            speed_of_sound,
            mean_velocity,
            time_step: f64::NAN,
        };
        candidate.validate_shape()?;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        let rate = candidate.characteristic_rate();
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "characteristic rate",
                value: rate,
            });
        }
        let mut configured = candidate;
        configured.time_step = cfl / rate;
        configured.validate()?;
        Ok(configured)
    }

    /// Grid spacing of one axis `[m]`.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is not a configured axis, which signals a
    /// caller-side indexing bug.
    pub fn dx(&self, axis: usize) -> f64 {
        self.axes[axis].length / self.axes[axis].n_points as f64
    }

    /// Interior node count over all axes.
    pub fn n_nodes(&self) -> usize {
        self.axes.iter().map(|axis| axis.n_points).product()
    }

    /// Gershgorin-style spectral-radius bound `[1/s]`.
    ///
    /// Exact plane waves satisfy `|λ| ≤ rate`, so `dt·rate ≤ 1` keeps
    /// every mode inside the RK4 stability domain.
    pub fn characteristic_rate(&self) -> f64 {
        let bound = modified_wavenumber_bound();
        (0..self.axes.len())
            .map(|axis| (self.mean_velocity[axis].abs() + self.speed_of_sound) / self.dx(axis))
            .sum::<f64>()
            * bound
    }

    /// Semi-discrete dispersion of an oblique plane wave.
    ///
    /// Each axis contributes its own modified wavenumber; the branch
    /// selects downstream (`+c|k*|`) or upstream (`-c|k*|`) propagation
    /// along the wavevector. A zero wavevector reports the acoustic
    /// speeds, matching the one-dimensional convention.
    ///
    /// # Panics
    ///
    /// Panics if `wavevector` does not match the grid dimension, which
    /// signals a caller-side layout bug.
    pub fn dispersion(&self, wavevector: &[f64], branch: AcousticBranch) -> PlaneWaveDispersion {
        assert_eq!(
            wavevector.len(),
            self.axes.len(),
            "wavevector must match the grid dimension"
        );
        let sign = match branch {
            AcousticBranch::Downstream => 1.0,
            AcousticBranch::Upstream => -1.0,
        };
        let modified: Vec<f64> = wavevector
            .iter()
            .enumerate()
            .map(|(axis, k)| TAM_WEBB_DRP_7.modified_wavenumber(*k, self.dx(axis)))
            .collect();
        let slopes: Vec<f64> = wavevector
            .iter()
            .enumerate()
            .map(|(axis, k)| TAM_WEBB_DRP_7.modified_wavenumber_slope(k * self.dx(axis)))
            .collect();
        let wavenumber = wavevector.iter().map(|k| k * k).sum::<f64>().sqrt();
        let modified_norm = modified.iter().map(|k| k * k).sum::<f64>().sqrt();
        let convection: f64 = self
            .mean_velocity
            .iter()
            .zip(modified.iter())
            .map(|(u, k)| u * k)
            .sum();
        let angular_frequency = convection + sign * self.speed_of_sound * modified_norm;
        // Chain rule through the modified wavenumber: convection uses the
        // same discrete derivative, so the mean flow carries the slope too.
        // At |k*| = 0 (zero or Nyquist wavevectors) |k*| has a cusp; the
        // convection part is the sensible principal value there.
        let group_velocity: Vec<f64> = self
            .mean_velocity
            .iter()
            .zip(modified.iter())
            .zip(slopes.iter())
            .map(|((u, k), slope)| {
                if modified_norm == 0.0 {
                    u * slope
                } else {
                    (u + sign * self.speed_of_sound * k / modified_norm) * slope
                }
            })
            .collect();
        let unit: Vec<f64> = if wavenumber == 0.0 {
            vec![0.0; wavevector.len()]
        } else {
            wavevector.iter().map(|k| k / wavenumber).collect()
        };
        let (phase_velocity, group_along) = if wavenumber == 0.0 {
            (sign * self.speed_of_sound, sign * self.speed_of_sound)
        } else {
            let group_along = group_velocity
                .iter()
                .zip(unit.iter())
                .map(|(g, u)| g * u)
                .sum();
            (angular_frequency / wavenumber, group_along)
        };
        PlaneWaveDispersion {
            dispersion: NumericalDispersion {
                wavenumber,
                modified_wavenumber: modified_norm,
                angular_frequency,
                phase_velocity,
                group_velocity: group_along,
            },
            group_velocity,
        }
    }

    /// Validate grid shape, flow parameters, and the CFL contract.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for the same conditions as
    /// [`Self::with_cfl`], plus a time step violating `dt·rate ≤ 1`.
    pub fn validate(&self) -> Result<(), AeroacousticsError> {
        self.validate_shape()?;
        require_positive("time_step", self.time_step)?;
        let rate = self.characteristic_rate();
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "characteristic rate",
                value: rate,
            });
        }
        let cfl = self.time_step * rate;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), AeroacousticsError> {
        if self.axes.len() != 2 && self.axes.len() != 3 {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "structured DRP grids need two or three axes",
            });
        }
        if self.mean_velocity.len() != self.axes.len() {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "mean_velocity must hold one entry per axis",
            });
        }
        require_positive("density", self.density)?;
        require_positive("speed_of_sound", self.speed_of_sound)?;
        for velocity in &self.mean_velocity {
            require_finite("mean_velocity", *velocity)?;
        }
        for axis in &self.axes {
            if axis.n_points < 7 {
                return Err(AeroacousticsError::GridTooSmall {
                    minimum: 7,
                    actual: axis.n_points,
                });
            }
            require_positive("axis length", axis.length)?;
            for face in [axis.lower, axis.upper] {
                if let StructuredFace::Outflow {
                    sponge_points,
                    sponge_strength,
                } = face
                {
                    if sponge_points == 0 || sponge_points >= axis.n_points {
                        return Err(AeroacousticsError::UnsupportedDiscretization {
                            detail: "sponge layer must cover 1..n_points-1 nodes",
                        });
                    }
                    require_finite("sponge_strength", sponge_strength)?;
                    if sponge_strength < 0.0 {
                        return Err(AeroacousticsError::NonPositiveParameter {
                            name: "sponge_strength",
                            value: sponge_strength,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Perturbation state on interior Cartesian nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct DrpStructuredState {
    /// Pressure perturbation `[Pa]`, row-major over the axes.
    pub pressure: Vec<f64>,
    /// Particle-velocity perturbation per axis `[m/s]`.
    pub velocity: Vec<Vec<f64>>,
}

impl DrpStructuredState {
    /// Construct a zero perturbation.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid, which signals a
    /// caller-side setup bug: admitted configs pass validation.
    pub fn zeros(config: &DrpStructuredConfig) -> Self {
        config.validate_shape().expect("valid DRP grid shape");
        Self {
            pressure: vec![0.0; config.n_nodes()],
            velocity: vec![vec![0.0; config.n_nodes()]; config.axes.len()],
        }
    }

    /// Construct an oblique plane wave with acoustic polarization.
    ///
    /// Pressure is `amplitude·cos(k·x)` and velocity follows the
    /// uniform-flow polarization `u = ±p·k_hat / (rho·c)` along the
    /// wavevector. A zero wavevector yields a uniform pressure pulse
    /// with zero velocity.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration, a
    /// wavevector of wrong length, or a non-finite parameter.
    pub fn plane_wave(
        config: &DrpStructuredConfig,
        wavevector: &[f64],
        pressure_amplitude: f64,
        branch: AcousticBranch,
    ) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        if wavevector.len() != config.axes.len() {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "wavevector must match the grid dimension",
            });
        }
        for k in wavevector {
            require_finite("wavevector", *k)?;
        }
        require_finite("pressure amplitude", pressure_amplitude)?;
        let sign = match branch {
            AcousticBranch::Downstream => 1.0,
            AcousticBranch::Upstream => -1.0,
        };
        let norm = wavevector.iter().map(|k| k * k).sum::<f64>().sqrt();
        let mut state = Self::zeros(config);
        let strides = interior_strides(config);
        for node in 0..config.n_nodes() {
            let mut phase = 0.0;
            for (axis, k) in wavevector.iter().enumerate() {
                let index = (node / strides[axis]) % config.axes[axis].n_points;
                phase += k * index as f64 * config.dx(axis);
            }
            let pressure = pressure_amplitude * phase.cos();
            state.pressure[node] = pressure;
            for (axis, velocity) in state.velocity.iter_mut().enumerate() {
                velocity[node] = if norm == 0.0 {
                    0.0
                } else {
                    sign * pressure * wavevector[axis]
                        / norm
                        / (config.density * config.speed_of_sound)
                };
            }
        }
        Ok(state)
    }

    /// Construct a Gaussian-modulated oblique wave packet.
    ///
    /// The envelope is isotropic with standard deviation
    /// `standard_deviation`, wrapped periodically along periodic axes
    /// and plain along outflow axes; polarization matches
    /// [`Self::plane_wave`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_euler::AcousticBranch;
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_structured::{
    ///     DrpStructuredConfig, DrpStructuredState, StructuredAxis, StructuredFace,
    /// };
    ///
    /// let axis = StructuredAxis {
    ///     n_points: 32,
    ///     length: 4.0,
    ///     lower: StructuredFace::Periodic,
    ///     upper: StructuredFace::Periodic,
    /// };
    /// let config =
    ///     DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![0.0, 0.0], 0.5)
    ///         .unwrap();
    /// let packet = DrpStructuredState::gaussian_packet(
    ///     &config,
    ///     &[2.0, 2.0],
    ///     0.3,
    ///     &[20.0, 0.0],
    ///     1.0,
    ///     AcousticBranch::Downstream,
    /// )
    /// .unwrap();
    /// assert!(packet.pressure.iter().any(|p| p.abs() > 0.5));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration, a
    /// center or wavevector of wrong length, or a non-finite or
    /// non-positive shape parameter.
    pub fn gaussian_packet(
        config: &DrpStructuredConfig,
        center: &[f64],
        standard_deviation: f64,
        wavevector: &[f64],
        pressure_amplitude: f64,
        branch: AcousticBranch,
    ) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        if center.len() != config.axes.len() || wavevector.len() != config.axes.len() {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "center and wavevector must match the grid dimension",
            });
        }
        for value in center.iter().chain(wavevector.iter()) {
            require_finite("packet parameter", *value)?;
        }
        require_positive("packet standard_deviation", standard_deviation)?;
        require_finite("pressure amplitude", pressure_amplitude)?;
        let sign = match branch {
            AcousticBranch::Downstream => 1.0,
            AcousticBranch::Upstream => -1.0,
        };
        let norm = wavevector.iter().map(|k| k * k).sum::<f64>().sqrt();
        let mut state = Self::zeros(config);
        let strides = interior_strides(config);
        for node in 0..config.n_nodes() {
            let mut squared = 0.0;
            let mut phase = 0.0;
            for (axis, spec) in config.axes.iter().enumerate() {
                let index = (node / strides[axis]) % spec.n_points;
                let x = index as f64 * config.dx(axis);
                let periodic = matches!(spec.lower, StructuredFace::Periodic)
                    && matches!(spec.upper, StructuredFace::Periodic);
                let delta = if periodic {
                    periodic_distance(x, center[axis], spec.length)
                } else {
                    x - center[axis]
                };
                squared += delta * delta;
                phase += wavevector[axis] * delta;
            }
            let envelope = (-0.5 * squared / standard_deviation.powi(2)).exp();
            let pressure = pressure_amplitude * envelope * phase.cos();
            state.pressure[node] = pressure;
            for (axis, velocity) in state.velocity.iter_mut().enumerate() {
                velocity[node] = if norm == 0.0 {
                    0.0
                } else {
                    sign * pressure * wavevector[axis]
                        / norm
                        / (config.density * config.speed_of_sound)
                };
            }
        }
        Ok(state)
    }
}

/// Cartesian DRP/RK4 solver for the linearized Euler equations.
#[derive(Debug, Clone)]
pub struct DrpStructuredEuler {
    /// Grid and mean-flow parameters.
    pub config: DrpStructuredConfig,
    /// Current perturbation state.
    pub state: DrpStructuredState,
    /// Simulated time `[s]`.
    pub time: f64,
    scratch: DrpStructuredScratch,
}

impl DrpStructuredEuler {
    /// Construct a solver from a validated state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::acoustics::aeroacoustics::drp_structured::{
    ///     DrpStructuredConfig, DrpStructuredEuler, DrpStructuredState, StructuredAxis,
    ///     StructuredFace,
    /// };
    ///
    /// let axis = StructuredAxis {
    ///     n_points: 16,
    ///     length: 1.0,
    ///     lower: StructuredFace::Periodic,
    ///     upper: StructuredFace::Periodic,
    /// };
    /// let config =
    ///     DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![0.0, 0.0], 0.5)
    ///         .unwrap();
    /// let state = DrpStructuredState::zeros(&config);
    /// let mut solver = DrpStructuredEuler::new(config, state).unwrap();
    /// solver.advance(4);
    /// assert!(solver.time > 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for an invalid configuration, a
    /// state whose arrays miss the grid length, or a non-finite state.
    pub fn new(
        config: DrpStructuredConfig,
        state: DrpStructuredState,
    ) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        let n = config.n_nodes();
        if state.pressure.len() != n {
            return Err(AeroacousticsError::StateLengthMismatch {
                expected: n,
                pressure: state.pressure.len(),
                velocity: n,
            });
        }
        if state.velocity.len() != config.axes.len() {
            return Err(AeroacousticsError::UnsupportedDiscretization {
                detail: "state must hold one velocity array per axis",
            });
        }
        for velocity in &state.velocity {
            if velocity.len() != n {
                return Err(AeroacousticsError::StateLengthMismatch {
                    expected: n,
                    pressure: n,
                    velocity: velocity.len(),
                });
            }
        }
        if state
            .pressure
            .iter()
            .chain(state.velocity.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField { name: "LEE state" });
        }
        let scratch = DrpStructuredScratch::zeros(&config);
        Ok(Self {
            config,
            state,
            time: 0.0,
            scratch,
        })
    }

    /// Advance one classical RK4 step.
    pub fn step(&mut self) {
        let dt = self.config.time_step;
        let scratch = &mut self.scratch;
        rhs(
            &self.config,
            &self.state,
            &mut scratch.k1,
            &mut scratch.work,
        );
        combine(
            &self.state,
            &scratch.k1,
            0.5 * dt,
            &mut scratch.intermediate,
        );
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k2,
            &mut scratch.work,
        );
        combine(
            &self.state,
            &scratch.k2,
            0.5 * dt,
            &mut scratch.intermediate,
        );
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k3,
            &mut scratch.work,
        );
        combine(&self.state, &scratch.k3, dt, &mut scratch.intermediate);
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k4,
            &mut scratch.work,
        );
        let n = self.config.n_nodes();
        for i in 0..n {
            self.state.pressure[i] += dt
                * (scratch.k1.pressure[i]
                    + 2.0 * scratch.k2.pressure[i]
                    + 2.0 * scratch.k3.pressure[i]
                    + scratch.k4.pressure[i])
                / 6.0;
        }
        for component in 0..self.config.axes.len() {
            for i in 0..n {
                self.state.velocity[component][i] += dt
                    * (scratch.k1.velocity[component][i]
                        + 2.0 * scratch.k2.velocity[component][i]
                        + 2.0 * scratch.k3.velocity[component][i]
                        + scratch.k4.velocity[component][i])
                    / 6.0;
            }
        }
        self.time += dt;
    }

    /// Advance an integer number of RK4 steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Total acoustic energy of a perturbation `[J]`.
///
/// Sums `p^2 / (2·rho·c^2) + rho·|u|^2 / 2` over the grid cells.
pub fn acoustic_energy(config: &DrpStructuredConfig, state: &DrpStructuredState) -> f64 {
    let cell_volume: f64 = (0..config.axes.len()).map(|axis| config.dx(axis)).product();
    let rho = config.density;
    let c = config.speed_of_sound;
    let mut total = 0.0;
    for node in 0..config.n_nodes() {
        total += state.pressure[node].powi(2) / (2.0 * rho * c * c);
        for velocity in &state.velocity {
            total += rho * velocity[node].powi(2) / 2.0;
        }
    }
    total * cell_volume
}

/// Transverse-integrated pressure energy along one axis.
///
/// Returns, per node of `axis`, the sum of `p^2` over all other axes:
/// the weight profile feeding [`centroid_1d`](super::wave_diagnostics::centroid_1d)
/// in group-velocity gates.
///
/// # Panics
///
/// Panics if `axis` is not a configured axis, which signals a
/// caller-side indexing bug.
pub fn transverse_energy_profile(
    config: &DrpStructuredConfig,
    state: &DrpStructuredState,
    axis: usize,
) -> Vec<f64> {
    let strides = interior_strides(config);
    let mut profile = vec![0.0; config.axes[axis].n_points];
    for node in 0..config.n_nodes() {
        let index = (node / strides[axis]) % config.axes[axis].n_points;
        profile[index] += state.pressure[node].powi(2);
    }
    profile
}

#[derive(Debug, Clone)]
struct DrpStructuredScratch {
    k1: DrpStructuredState,
    k2: DrpStructuredState,
    k3: DrpStructuredState,
    k4: DrpStructuredState,
    intermediate: DrpStructuredState,
    work: DrpStructuredWork,
}

#[derive(Debug, Clone)]
struct DrpStructuredWork {
    ghosted: Vec<f64>,
    line: Vec<f64>,
    pressure_gradient: Vec<Vec<f64>>,
    velocity_gradient: Vec<Vec<Vec<f64>>>,
    sponge: Vec<f64>,
    ghosted_strides: Vec<usize>,
}

impl DrpStructuredScratch {
    fn zeros(config: &DrpStructuredConfig) -> Self {
        Self {
            k1: DrpStructuredState::zeros(config),
            k2: DrpStructuredState::zeros(config),
            k3: DrpStructuredState::zeros(config),
            k4: DrpStructuredState::zeros(config),
            intermediate: DrpStructuredState::zeros(config),
            work: DrpStructuredWork::zeros(config),
        }
    }
}

impl DrpStructuredWork {
    fn zeros(config: &DrpStructuredConfig) -> Self {
        let n = config.n_nodes();
        let dims = config.axes.len();
        let ghosted_strides = ghosted_strides(config);
        let ghosted_total =
            ghosted_strides[dims - 1] * (config.axes[dims - 1].n_points + 2 * DrpStencil::RADIUS);
        let max_points = config
            .axes
            .iter()
            .map(|axis| axis.n_points)
            .max()
            .unwrap_or(0);
        Self {
            ghosted: vec![0.0; ghosted_total],
            line: vec![0.0; max_points],
            pressure_gradient: vec![vec![0.0; n]; dims],
            velocity_gradient: vec![vec![vec![0.0; n]; dims]; dims],
            sponge: sponge_profile(config),
            ghosted_strides,
        }
    }
}

fn interior_strides(config: &DrpStructuredConfig) -> Vec<usize> {
    let mut strides = Vec::with_capacity(config.axes.len());
    let mut stride = 1;
    for axis in &config.axes {
        strides.push(stride);
        stride *= axis.n_points;
    }
    strides
}

fn ghosted_strides(config: &DrpStructuredConfig) -> Vec<usize> {
    let mut strides = Vec::with_capacity(config.axes.len());
    let mut stride = 1;
    for axis in &config.axes {
        strides.push(stride);
        stride *= axis.n_points + 2 * DrpStencil::RADIUS;
    }
    strides
}

/// Quadratic sponge profile summed over all outflow faces.
fn sponge_profile(config: &DrpStructuredConfig) -> Vec<f64> {
    let strides = interior_strides(config);
    let mut sponge = vec![0.0; config.n_nodes()];
    for (node, slot) in sponge.iter_mut().enumerate() {
        let mut rate = 0.0;
        for (axis, spec) in config.axes.iter().enumerate() {
            let index = (node / strides[axis]) % spec.n_points;
            for (face, depth) in [(spec.lower, index), (spec.upper, spec.n_points - 1 - index)] {
                if let StructuredFace::Outflow {
                    sponge_points,
                    sponge_strength,
                } = face
                {
                    // Depth counts nodes from the face inward; the profile
                    // peaks at the face and vanishes at the sponge edge.
                    let inside = sponge_points - 1 - depth.min(sponge_points - 1);
                    let fraction = inside as f64 / sponge_points as f64;
                    rate += sponge_strength * fraction * fraction;
                }
            }
        }
        *slot = rate;
    }
    sponge
}

fn pack_interior(
    config: &DrpStructuredConfig,
    interior: &[f64],
    ghosted_strides: &[usize],
    ghosted: &mut [f64],
) {
    let radius = DrpStencil::RADIUS;
    let strides = interior_strides(config);
    for (node, value) in interior.iter().enumerate() {
        let mut target = 0;
        for (axis, spec) in config.axes.iter().enumerate() {
            let index = (node / strides[axis]) % spec.n_points;
            target += (index + radius) * ghosted_strides[axis];
        }
        ghosted[target] = *value;
    }
}

fn fill_ghosts_axis(
    config: &DrpStructuredConfig,
    axis: usize,
    ghosted_strides: &[usize],
    ghosted: &mut [f64],
) {
    let radius = DrpStencil::RADIUS;
    let n = config.axes[axis].n_points;
    let stride = ghosted_strides[axis];
    let spec = config.axes[axis];
    // Lines span interior coordinates of the transverse axes only: every
    // one-dimensional stencil is centered at an interior node, so corner
    // ghosts (ghost layers in two or more axes) are never read.
    let n_lines = config.n_nodes() / n;
    for line in 0..n_lines {
        let mut base = 0;
        let mut remainder = line;
        for (other, other_spec) in config.axes.iter().enumerate() {
            if other == axis {
                continue;
            }
            let coordinate = remainder % other_spec.n_points;
            remainder /= other_spec.n_points;
            base += (coordinate + radius) * ghosted_strides[other];
        }
        for layer in 0..radius {
            let lo_ghost = base + layer * stride;
            let hi_ghost = base + (radius + n + layer) * stride;
            ghosted[lo_ghost] = match spec.lower {
                StructuredFace::Periodic => ghosted[base + (radius + n - radius + layer) * stride],
                StructuredFace::Outflow { .. } => ghosted[base + radius * stride],
            };
            ghosted[hi_ghost] = match spec.upper {
                StructuredFace::Periodic => ghosted[base + (radius + layer) * stride],
                StructuredFace::Outflow { .. } => ghosted[base + (radius + n - 1) * stride],
            };
        }
    }
}

/// Differentiate one axis of a packed component into an interior buffer.
fn differentiate_axis(
    config: &DrpStructuredConfig,
    axis: usize,
    ghosted_strides: &[usize],
    ghosted: &[f64],
    line: &mut [f64],
    output: &mut [f64],
) {
    let radius = DrpStencil::RADIUS;
    let n = config.axes[axis].n_points;
    let stride = ghosted_strides[axis];
    let interior = interior_strides(config);
    let n_lines = config.n_nodes() / n;
    for grid_line in 0..n_lines {
        let mut base = 0;
        let mut interior_base = 0;
        let mut remainder = grid_line;
        for (other, other_spec) in config.axes.iter().enumerate() {
            if other == axis {
                continue;
            }
            let coordinate = remainder % other_spec.n_points;
            remainder /= other_spec.n_points;
            base += (coordinate + radius) * ghosted_strides[other];
            interior_base += coordinate * interior[other];
        }
        TAM_WEBB_DRP_7.derivative_ghosted_line(ghosted, base, stride, n, config.dx(axis), line);
        for k in 0..n {
            output[interior_base + k * interior[axis]] = line[k];
        }
    }
}

fn rhs(
    config: &DrpStructuredConfig,
    state: &DrpStructuredState,
    output: &mut DrpStructuredState,
    work: &mut DrpStructuredWork,
) {
    let dims = config.axes.len();
    for axis in 0..dims {
        pack_interior(
            config,
            &state.pressure,
            &work.ghosted_strides,
            &mut work.ghosted,
        );
        fill_ghosts_axis(config, axis, &work.ghosted_strides, &mut work.ghosted);
        differentiate_axis(
            config,
            axis,
            &work.ghosted_strides,
            &work.ghosted,
            &mut work.line,
            &mut work.pressure_gradient[axis],
        );
        for component in 0..dims {
            pack_interior(
                config,
                &state.velocity[component],
                &work.ghosted_strides,
                &mut work.ghosted,
            );
            fill_ghosts_axis(config, axis, &work.ghosted_strides, &mut work.ghosted);
            differentiate_axis(
                config,
                axis,
                &work.ghosted_strides,
                &work.ghosted,
                &mut work.line,
                &mut work.velocity_gradient[axis][component],
            );
        }
    }
    let rho = config.density;
    let c = config.speed_of_sound;
    for node in 0..config.n_nodes() {
        let mut pressure_rhs = -work.sponge[node] * state.pressure[node];
        for axis in 0..dims {
            pressure_rhs -= config.mean_velocity[axis] * work.pressure_gradient[axis][node]
                + rho * c * c * work.velocity_gradient[axis][axis][node];
        }
        output.pressure[node] = pressure_rhs;
        for component in 0..dims {
            let mut velocity_rhs = -work.sponge[node] * state.velocity[component][node]
                - work.pressure_gradient[component][node] / rho;
            for axis in 0..dims {
                velocity_rhs -=
                    config.mean_velocity[axis] * work.velocity_gradient[axis][component][node];
            }
            output.velocity[component][node] = velocity_rhs;
        }
    }
}

fn combine(
    base: &DrpStructuredState,
    increment: &DrpStructuredState,
    scale: f64,
    out: &mut DrpStructuredState,
) {
    for i in 0..base.pressure.len() {
        out.pressure[i] = base.pressure[i] + scale * increment.pressure[i];
    }
    for (out_velocity, (base_velocity, increment_velocity)) in out
        .velocity
        .iter_mut()
        .zip(base.velocity.iter().zip(increment.velocity.iter()))
    {
        for i in 0..base_velocity.len() {
            out_velocity[i] = base_velocity[i] + scale * increment_velocity[i];
        }
    }
}

fn periodic_distance(x: f64, center: f64, length: f64) -> f64 {
    (x - center + 0.5 * length).rem_euclid(length) - 0.5 * length
}

#[cfg(test)]
mod tests {
    use super::super::drp_euler::DrpEuler1dConfig;
    use super::super::wave_diagnostics::{centroid_1d, group_velocity_from_track};
    use super::*;
    use std::f64::consts::PI;

    /// One-dimensional configuration sharing the mean-flow parameters.
    ///
    /// Exists so oblique-wave gates can cross-check the multidimensional
    /// dispersion against the one-dimensional reference along each axis.
    fn axis_reference_1d(config: &DrpStructuredConfig, axis: usize) -> DrpEuler1dConfig {
        DrpEuler1dConfig {
            n_points: config.axes[axis].n_points,
            length: config.axes[axis].length,
            density: config.density,
            speed_of_sound: config.speed_of_sound,
            mean_velocity: config.mean_velocity[axis],
            time_step: config.time_step,
        }
    }

    fn periodic_axis(n_points: usize, length: f64) -> StructuredAxis {
        StructuredAxis {
            n_points,
            length,
            lower: StructuredFace::Periodic,
            upper: StructuredFace::Periodic,
        }
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        let axis = periodic_axis(16, 1.0);
        assert!(matches!(
            DrpStructuredConfig::with_cfl(vec![axis], 1.2, 343.0, vec![0.0], 0.5),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        let short = periodic_axis(6, 1.0);
        assert!(matches!(
            DrpStructuredConfig::with_cfl(vec![short, axis], 1.2, 343.0, vec![0.0, 0.0], 0.5),
            Err(AeroacousticsError::GridTooSmall { .. })
        ));
        assert!(matches!(
            DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![0.0], 0.5),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        assert!(matches!(
            DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![0.0, 0.0], 1.5),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
        let sponged = StructuredAxis {
            n_points: 16,
            length: 1.0,
            lower: StructuredFace::Periodic,
            upper: StructuredFace::Outflow {
                sponge_points: 16,
                sponge_strength: 10.0,
            },
        };
        assert!(matches!(
            DrpStructuredConfig::with_cfl(vec![sponged, axis], 1.2, 343.0, vec![0.0, 0.0], 0.5),
            Err(AeroacousticsError::UnsupportedDiscretization { .. })
        ));
        let config =
            DrpStructuredConfig::with_cfl(vec![axis, axis], 1.2, 343.0, vec![5.0, 0.0], 0.5)
                .unwrap();
        let mut over_cfl = config.clone();
        over_cfl.time_step *= 2.1;
        assert!(matches!(
            over_cfl.validate(),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
        let bad_state = DrpStructuredState {
            pressure: vec![0.0; 15],
            velocity: vec![vec![0.0; 256], vec![0.0; 256]],
        };
        assert!(matches!(
            DrpStructuredEuler::new(config.clone(), bad_state),
            Err(AeroacousticsError::StateLengthMismatch { .. })
        ));
        let mut non_finite = DrpStructuredState::zeros(&config);
        non_finite.velocity[1][7] = f64::NAN;
        assert!(matches!(
            DrpStructuredEuler::new(config, non_finite),
            Err(AeroacousticsError::NonFiniteField { .. })
        ));
    }

    #[test]
    fn oblique_dispersion_matches_axis_reference_on_axis_aligned_waves() {
        let axes = vec![periodic_axis(48, 3.0), periodic_axis(64, 4.0)];
        let config =
            DrpStructuredConfig::with_cfl(axes, 1.2, 343.0, vec![30.0, -10.0], 0.5).unwrap();
        for branch in [AcousticBranch::Downstream, AcousticBranch::Upstream] {
            let wavevector = [2.0 * PI / 3.0, 0.0];
            let oblique = config.dispersion(&wavevector, branch);
            let reference = axis_reference_1d(&config, 0).dispersion(wavevector[0], branch);
            assert!(
                (oblique.dispersion.angular_frequency - reference.angular_frequency).abs() < 1e-9
            );
            assert!((oblique.dispersion.phase_velocity - reference.phase_velocity).abs() < 1e-9);
            assert!((oblique.dispersion.group_velocity - reference.group_velocity).abs() < 1e-9);
            assert!((oblique.group_velocity[0] - reference.group_velocity).abs() < 1e-9);
            // Transverse group velocity is pure mean-flow convection.
            assert!((oblique.group_velocity[1] - config.mean_velocity[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn oblique_eigenmode_advances_at_discrete_frequency() {
        let axes = vec![periodic_axis(64, 2.0 * PI), periodic_axis(64, 2.0 * PI)];
        let config = DrpStructuredConfig::with_cfl(axes, 1.3, 2.0, vec![0.3, -0.2], 0.1).unwrap();
        let wavevector = [3.0, 4.0];
        let state =
            DrpStructuredState::plane_wave(&config, &wavevector, 1.0, AcousticBranch::Downstream)
                .unwrap();
        let mut solver = DrpStructuredEuler::new(config.clone(), state).unwrap();
        solver.advance(60);
        let omega = config
            .dispersion(&wavevector, AcousticBranch::Downstream)
            .dispersion
            .angular_frequency;
        let strides = interior_strides(&config);
        let mut squared = 0.0;
        for node in 0..config.n_nodes() {
            let mut phase = 0.0;
            for (axis, k) in wavevector.iter().enumerate() {
                let index = (node / strides[axis]) % config.axes[axis].n_points;
                phase += k * index as f64 * config.dx(axis);
            }
            let expected = (phase - omega * solver.time).cos();
            squared += (solver.state.pressure[node] - expected).powi(2);
        }
        let relative = squared.sqrt() / (config.n_nodes() as f64 / 2.0).sqrt();
        assert!(relative < 2e-5, "relative error={relative:e}");
    }

    #[test]
    fn packet_group_velocity_matches_semidiscrete_prediction() {
        let axes = vec![periodic_axis(160, 8.0), periodic_axis(32, 2.0)];
        let config = DrpStructuredConfig::with_cfl(axes, 1.2, 343.0, vec![20.0, 0.0], 0.3).unwrap();
        let wavevector = [15.0, 0.0];
        // A plane packet (uniform transverse profile): transverse sidebands
        // would steal axial group velocity, blurring the carrier prediction.
        let mut state =
            DrpStructuredState::plane_wave(&config, &wavevector, 1.0, AcousticBranch::Downstream)
                .unwrap();
        let strides = interior_strides(&config);
        for node in 0..config.n_nodes() {
            let index = (node / strides[0]) % config.axes[0].n_points;
            let x = index as f64 * config.dx(0);
            let delta = periodic_distance(x, 2.5, config.axes[0].length);
            let envelope = (-0.5 * (delta / 0.7).powi(2)).exp();
            state.pressure[node] *= envelope;
            for velocity in state.velocity.iter_mut() {
                velocity[node] *= envelope;
            }
        }
        let mut solver = DrpStructuredEuler::new(config.clone(), state).unwrap();
        let prediction = config
            .dispersion(&wavevector, AcousticBranch::Downstream)
            .dispersion
            .group_velocity;
        let coordinates: Vec<f64> = (0..config.axes[0].n_points)
            .map(|i| i as f64 * config.dx(0))
            .collect();
        let mut times = Vec::new();
        let mut centroids = Vec::new();
        // The packet starts at x = 2 and drifts downstream; sample while it
        // stays clear of the periodic wrap (group speed ~363 m/s).
        for _ in 0..12 {
            solver.advance(25);
            let profile = transverse_energy_profile(&config, &solver.state, 0);
            let centroid = centroid_1d(&coordinates, &profile).unwrap();
            if centroid > 6.0 {
                break;
            }
            times.push(solver.time);
            centroids.push(centroid);
        }
        assert!(times.len() >= 6, "track too short: {}", times.len());
        let measured = group_velocity_from_track(&times, &centroids).unwrap();
        let analytic = config.mean_velocity[0] + config.speed_of_sound;
        // At k·dx = 0.75 the stencil carries ~1% real discrete dispersion,
        // so the analytic comparison is loose; the tight gate is the
        // semi-discrete prediction below, which must capture that shift.
        assert!(
            (measured - analytic).abs() / analytic < 2e-2,
            "measured={measured}, analytic={analytic}"
        );
        // The track measures band-averaged group velocity while the
        // prediction is the carrier value; slope curvature over the
        // Gaussian sidebands accounts for the residual gap.
        assert!(
            (measured - prediction).abs() / prediction < 1e-3,
            "measured={measured}, predicted={prediction}"
        );
    }

    #[test]
    fn periodic_energy_stays_bounded_over_long_runs() {
        let axes = vec![periodic_axis(48, 3.0), periodic_axis(48, 3.0)];
        let config = DrpStructuredConfig::with_cfl(axes, 1.2, 343.0, vec![30.0, 0.0], 0.5).unwrap();
        let state = DrpStructuredState::gaussian_packet(
            &config,
            &[1.5, 1.5],
            0.2,
            &[30.0, 10.0],
            1.0,
            AcousticBranch::Downstream,
        )
        .unwrap();
        let mut solver = DrpStructuredEuler::new(config.clone(), state).unwrap();
        let initial = acoustic_energy(&config, &solver.state);
        solver.advance(400);
        let drift = (acoustic_energy(&config, &solver.state) - initial).abs() / initial;
        // RK4 amplitude error accumulates over hundreds of steps; the gate
        // proves boundedness (no modal growth), not exact conservation.
        assert!(drift < 2e-3, "energy drift={drift:e}");
        assert!(
            solver
                .state
                .pressure
                .iter()
                .chain(solver.state.velocity.iter().flatten())
                .all(|x| x.is_finite())
        );
    }

    #[test]
    fn sponge_absorbs_an_outgoing_packet() {
        let outflow = StructuredAxis {
            n_points: 160,
            length: 8.0,
            lower: StructuredFace::Periodic,
            upper: StructuredFace::Outflow {
                sponge_points: 24,
                sponge_strength: 2600.0,
            },
        };
        let config = DrpStructuredConfig::with_cfl(
            vec![outflow, periodic_axis(32, 2.0)],
            1.2,
            343.0,
            vec![20.0, 0.0],
            0.3,
        )
        .unwrap();
        let state = DrpStructuredState::gaussian_packet(
            &config,
            &[2.0, 1.0],
            0.3,
            &[15.0, 0.0],
            1.0,
            AcousticBranch::Downstream,
        )
        .unwrap();
        let mut solver = DrpStructuredEuler::new(config.clone(), state).unwrap();
        let initial = acoustic_energy(&config, &solver.state);
        // Group speed ~363 m/s crosses 6 m in ~17 ms; run well past exit.
        let steps = (0.03 / config.time_step).ceil() as usize;
        solver.advance(steps);
        let remaining = acoustic_energy(&config, &solver.state) / initial;
        assert!(remaining < 0.02, "reflected fraction={remaining:e}");
    }

    #[test]
    fn three_dimensional_eigenmode_matches_discrete_frequency() {
        let axes = vec![
            periodic_axis(16, 2.0 * PI),
            periodic_axis(16, 2.0 * PI),
            periodic_axis(16, 2.0 * PI),
        ];
        let config =
            DrpStructuredConfig::with_cfl(axes, 1.0, 2.0, vec![0.2, 0.1, 0.0], 0.1).unwrap();
        let wavevector = [2.0, 1.0, 1.0];
        let state =
            DrpStructuredState::plane_wave(&config, &wavevector, 1.0, AcousticBranch::Downstream)
                .unwrap();
        let mut solver = DrpStructuredEuler::new(config.clone(), state).unwrap();
        solver.advance(20);
        let omega = config
            .dispersion(&wavevector, AcousticBranch::Downstream)
            .dispersion
            .angular_frequency;
        let strides = interior_strides(&config);
        let mut squared = 0.0;
        for node in 0..config.n_nodes() {
            let mut phase = 0.0;
            for (axis, k) in wavevector.iter().enumerate() {
                let index = (node / strides[axis]) % config.axes[axis].n_points;
                phase += k * index as f64 * config.dx(axis);
            }
            let expected = (phase - omega * solver.time).cos();
            squared += (solver.state.pressure[node] - expected).powi(2);
        }
        let relative = squared.sqrt() / (config.n_nodes() as f64 / 2.0).sqrt();
        assert!(relative < 1e-4, "relative error={relative:e}");
    }
}
