//! One-dimensional linearized Euler propagation with a DRP spatial stencil.
//!
//! The solver advances pressure and particle-velocity perturbations about a
//! uniform mean flow on a periodic grid:
//!
//! ```text
//! dp/dt + U dp/dx + rho c^2 du/dx = 0,
//! du/dt + U du/dx + (1/rho) dp/dx = 0.
//! ```
//!
//! Spatial derivatives use the optimized seven-point Tam--Webb
//! dispersion-relation-preserving (DRP) stencil. Time integration is classical
//! RK4. The discrete dispersion relation and its derivative expose numerical
//! phase and group velocities for both acoustic branches.
//!
//! This is a uniform-flow, linear, inviscid, periodic reference solver. It is
//! complementary to the frequency-domain mixed FEM LEE assembler in
//! [`super::linearized_euler`], not a replacement for nonuniform-flow FEM.
//!
//! # Reference
//!
//! C. K. W. Tam and J. C. Webb, "Dispersion-Relation-Preserving Finite
//! Difference Schemes for Computational Acoustics", *Journal of Computational
//! Physics* 107 (1993), 262--281.

use super::error::{AeroacousticsError, require_finite, require_positive};

/// Optimized seven-point antisymmetric first-derivative stencil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrpStencil {
    /// Positive-offset coefficients `a_j` in
    /// `df_i/dx = sum_j a_j (f_{i+j} - f_{i-j}) / dx`.
    pub coefficients: [f64; 3],
}

/// Tam--Webb seven-point DRP stencil.
pub const TAM_WEBB_DRP_7: DrpStencil = DrpStencil {
    coefficients: [
        0.770_882_380_518_225_7,
        -0.166_705_904_414_58,
        0.020_843_142_770_311,
    ],
};

impl DrpStencil {
    /// Modified wavenumber `k*` for physical wavenumber `k` and spacing `dx`.
    pub fn modified_wavenumber(&self, wavenumber: f64, dx: f64) -> f64 {
        let theta = wavenumber * dx;
        2.0 * self
            .coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| coefficient * ((index + 1) as f64 * theta).sin())
            .sum::<f64>()
            / dx
    }

    /// Dimensionless slope `d k* / d k`, used by numerical group velocity.
    pub fn modified_wavenumber_slope(&self, nondimensional_wavenumber: f64) -> f64 {
        2.0 * self
            .coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| {
                let offset = (index + 1) as f64;
                offset * coefficient * (offset * nondimensional_wavenumber).cos()
            })
            .sum::<f64>()
    }

    fn derivative_periodic(&self, field: &[f64], dx: f64, output: &mut [f64]) {
        let n = field.len();
        for (i, derivative) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (offset_index, coefficient) in self.coefficients.iter().enumerate() {
                let offset = offset_index + 1;
                let plus = field[(i + offset) % n];
                let minus = field[(i + n - offset) % n];
                sum += coefficient * (plus - minus);
            }
            *derivative = sum / dx;
        }
    }
}

/// Downstream (`U+c`) or upstream (`U-c`) acoustic characteristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticBranch {
    /// Pressure and velocity are in phase: `p = rho c u`.
    Downstream,
    /// Pressure and velocity have opposite signs: `p = -rho c u`.
    Upstream,
}

impl AcousticBranch {
    fn characteristic_speed(self, mean_velocity: f64, speed_of_sound: f64) -> f64 {
        match self {
            Self::Downstream => mean_velocity + speed_of_sound,
            Self::Upstream => mean_velocity - speed_of_sound,
        }
    }

    fn impedance_sign(self) -> f64 {
        match self {
            Self::Downstream => 1.0,
            Self::Upstream => -1.0,
        }
    }
}

/// Discrete dispersion data for one Fourier mode and acoustic branch.
#[derive(Debug, Clone, Copy)]
pub struct NumericalDispersion {
    /// Physical wavenumber `k` `[rad/m]`.
    pub wavenumber: f64,
    /// DRP modified wavenumber `k*` `[rad/m]`.
    pub modified_wavenumber: f64,
    /// Semi-discrete angular frequency `(U +/- c) k*` `[rad/s]`.
    pub angular_frequency: f64,
    /// Numerical phase velocity `omega/k` `[m/s]`.
    pub phase_velocity: f64,
    /// Numerical group velocity `d omega/d k` `[m/s]`.
    pub group_velocity: f64,
}

/// Configuration of a periodic one-dimensional DRP LEE grid.
#[derive(Debug, Clone, Copy)]
pub struct DrpEuler1dConfig {
    /// Number of periodic grid points.
    pub n_points: usize,
    /// Periodic domain length `[m]`.
    pub length: f64,
    /// Uniform mean density `[kg/m^3]`.
    pub density: f64,
    /// Uniform sound speed `[m/s]`.
    pub speed_of_sound: f64,
    /// Uniform mean-flow velocity `[m/s]`.
    pub mean_velocity: f64,
    /// Explicit RK4 step `[s]`.
    pub time_step: f64,
}

impl DrpEuler1dConfig {
    /// Construct a grid and select `dt = CFL dx / (|U| + c)`.
    pub fn with_cfl(
        n_points: usize,
        length: f64,
        density: f64,
        speed_of_sound: f64,
        mean_velocity: f64,
        cfl: f64,
    ) -> Result<Self, AeroacousticsError> {
        if n_points < 7 {
            return Err(AeroacousticsError::GridTooSmall {
                minimum: 7,
                actual: n_points,
            });
        }
        require_positive("length", length)?;
        require_positive("density", density)?;
        require_positive("speed_of_sound", speed_of_sound)?;
        require_finite("mean_velocity", mean_velocity)?;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        let dx = length / n_points as f64;
        let max_characteristic_speed = mean_velocity.abs() + speed_of_sound;
        if !max_characteristic_speed.is_finite() {
            return Err(AeroacousticsError::NonFiniteParameter {
                name: "maximum characteristic speed",
                value: max_characteristic_speed,
            });
        }
        let time_step = cfl * dx / max_characteristic_speed;
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "time_step",
                value: time_step,
            });
        }
        Ok(Self {
            n_points,
            length,
            density,
            speed_of_sound,
            mean_velocity,
            time_step,
        })
    }

    /// Grid spacing `[m]`.
    pub fn dx(&self) -> f64 {
        self.length / self.n_points as f64
    }

    /// Validate a manually constructed configuration.
    ///
    /// The fields are public to keep the configuration easy to serialize and
    /// inspect, so [`DrpEuler1d::new`] repeats the same admission checks that
    /// [`Self::with_cfl`] performs. This prevents a zero/NaN spacing or an
    /// over-CFL time step from reaching the periodic stencil.
    pub fn validate(&self) -> Result<(), AeroacousticsError> {
        if self.n_points < 7 {
            return Err(AeroacousticsError::GridTooSmall {
                minimum: 7,
                actual: self.n_points,
            });
        }
        require_positive("length", self.length)?;
        require_positive("density", self.density)?;
        require_positive("speed_of_sound", self.speed_of_sound)?;
        require_finite("mean_velocity", self.mean_velocity)?;
        require_positive("time_step", self.time_step)?;

        let dx = self.dx();
        if !dx.is_finite() || dx <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "grid spacing",
                value: dx,
            });
        }
        let cfl = self.time_step * (self.mean_velocity.abs() + self.speed_of_sound) / dx;
        if !cfl.is_finite() || !(0.0 < cfl && cfl <= 1.0) {
            return Err(AeroacousticsError::InvalidCfl(cfl));
        }
        Ok(())
    }

    /// Semi-discrete dispersion relation of a Fourier mode.
    pub fn dispersion(&self, wavenumber: f64, branch: AcousticBranch) -> NumericalDispersion {
        let dx = self.dx();
        let modified_wavenumber = TAM_WEBB_DRP_7.modified_wavenumber(wavenumber, dx);
        let characteristic_speed =
            branch.characteristic_speed(self.mean_velocity, self.speed_of_sound);
        let angular_frequency = characteristic_speed * modified_wavenumber;
        let phase_velocity = if wavenumber.abs() <= f64::EPSILON {
            characteristic_speed
        } else {
            angular_frequency / wavenumber
        };
        let group_velocity =
            characteristic_speed * TAM_WEBB_DRP_7.modified_wavenumber_slope(wavenumber * dx);
        NumericalDispersion {
            wavenumber,
            modified_wavenumber,
            angular_frequency,
            phase_velocity,
            group_velocity,
        }
    }
}

/// Pressure/velocity perturbation state on a periodic grid.
#[derive(Debug, Clone)]
pub struct DrpWaveState {
    /// Pressure perturbation `[Pa]`.
    pub pressure: Vec<f64>,
    /// Particle-velocity perturbation `[m/s]`.
    pub velocity: Vec<f64>,
}

impl DrpWaveState {
    /// Construct a zero perturbation.
    pub fn zeros(n_points: usize) -> Self {
        Self {
            pressure: vec![0.0; n_points],
            velocity: vec![0.0; n_points],
        }
    }

    /// Construct a right- or left-running Gaussian-modulated wave packet.
    pub fn gaussian_packet(
        config: &DrpEuler1dConfig,
        center: f64,
        standard_deviation: f64,
        wavenumber: f64,
        pressure_amplitude: f64,
        branch: AcousticBranch,
    ) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        require_positive("packet standard_deviation", standard_deviation)?;
        require_finite("packet center", center)?;
        require_finite("packet wavenumber", wavenumber)?;
        require_finite("pressure amplitude", pressure_amplitude)?;
        let mut state = Self::zeros(config.n_points);
        let impedance = config.density * config.speed_of_sound * branch.impedance_sign();
        for i in 0..config.n_points {
            let x = i as f64 * config.dx();
            let distance = periodic_distance(x, center, config.length);
            let envelope = (-0.5 * (distance / standard_deviation).powi(2)).exp();
            let pressure = pressure_amplitude * envelope * (wavenumber * distance).cos();
            state.pressure[i] = pressure;
            state.velocity[i] = pressure / impedance;
        }
        Ok(state)
    }
}

/// Periodic DRP/RK4 solver for the one-dimensional linearized Euler equations.
#[derive(Debug, Clone)]
pub struct DrpEuler1d {
    /// Grid and mean-flow parameters.
    pub config: DrpEuler1dConfig,
    /// Current perturbation state.
    pub state: DrpWaveState,
    /// Simulated time `[s]`.
    pub time: f64,
    scratch: DrpEulerScratch,
}

impl DrpEuler1d {
    /// Construct a solver from a validated state.
    pub fn new(config: DrpEuler1dConfig, state: DrpWaveState) -> Result<Self, AeroacousticsError> {
        config.validate()?;
        if state.pressure.len() != config.n_points || state.velocity.len() != config.n_points {
            return Err(AeroacousticsError::StateLengthMismatch {
                expected: config.n_points,
                pressure: state.pressure.len(),
                velocity: state.velocity.len(),
            });
        }
        if state
            .pressure
            .iter()
            .chain(&state.velocity)
            .any(|x| !x.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField { name: "LEE state" });
        }
        Ok(Self {
            config,
            state,
            time: 0.0,
            scratch: DrpEulerScratch::zeros(config.n_points),
        })
    }

    /// Advance one classical RK4 step.
    pub fn step(&mut self) {
        let n = self.config.n_points;
        let dt = self.config.time_step;
        let scratch = &mut self.scratch;

        rhs(
            &self.config,
            &self.state,
            &mut scratch.k1,
            &mut scratch.pressure_gradient,
            &mut scratch.velocity_gradient,
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
            &mut scratch.pressure_gradient,
            &mut scratch.velocity_gradient,
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
            &mut scratch.pressure_gradient,
            &mut scratch.velocity_gradient,
        );
        combine(&self.state, &scratch.k3, dt, &mut scratch.intermediate);
        rhs(
            &self.config,
            &scratch.intermediate,
            &mut scratch.k4,
            &mut scratch.pressure_gradient,
            &mut scratch.velocity_gradient,
        );

        for i in 0..n {
            self.state.pressure[i] += dt
                * (scratch.k1.pressure[i]
                    + 2.0 * scratch.k2.pressure[i]
                    + 2.0 * scratch.k3.pressure[i]
                    + scratch.k4.pressure[i])
                / 6.0;
            self.state.velocity[i] += dt
                * (scratch.k1.velocity[i]
                    + 2.0 * scratch.k2.velocity[i]
                    + 2.0 * scratch.k3.velocity[i]
                    + scratch.k4.velocity[i])
                / 6.0;
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

#[derive(Debug, Clone)]
struct DrpEulerScratch {
    k1: DrpWaveState,
    k2: DrpWaveState,
    k3: DrpWaveState,
    k4: DrpWaveState,
    intermediate: DrpWaveState,
    pressure_gradient: Vec<f64>,
    velocity_gradient: Vec<f64>,
}

impl DrpEulerScratch {
    fn zeros(n_points: usize) -> Self {
        Self {
            k1: DrpWaveState::zeros(n_points),
            k2: DrpWaveState::zeros(n_points),
            k3: DrpWaveState::zeros(n_points),
            k4: DrpWaveState::zeros(n_points),
            intermediate: DrpWaveState::zeros(n_points),
            pressure_gradient: vec![0.0; n_points],
            velocity_gradient: vec![0.0; n_points],
        }
    }
}

fn rhs(
    config: &DrpEuler1dConfig,
    state: &DrpWaveState,
    output: &mut DrpWaveState,
    pressure_gradient: &mut [f64],
    velocity_gradient: &mut [f64],
) {
    TAM_WEBB_DRP_7.derivative_periodic(&state.pressure, config.dx(), pressure_gradient);
    TAM_WEBB_DRP_7.derivative_periodic(&state.velocity, config.dx(), velocity_gradient);
    let rho = config.density;
    let c = config.speed_of_sound;
    let mean_velocity = config.mean_velocity;
    for i in 0..config.n_points {
        output.pressure[i] =
            -mean_velocity * pressure_gradient[i] - rho * c * c * velocity_gradient[i];
        output.velocity[i] = -mean_velocity * velocity_gradient[i] - pressure_gradient[i] / rho;
    }
}

fn combine(base: &DrpWaveState, increment: &DrpWaveState, scale: f64, out: &mut DrpWaveState) {
    for i in 0..base.pressure.len() {
        out.pressure[i] = base.pressure[i] + scale * increment.pressure[i];
        out.velocity[i] = base.velocity[i] + scale * increment.velocity[i];
    }
}

fn periodic_distance(x: f64, center: f64, length: f64) -> f64 {
    (x - center + 0.5 * length).rem_euclid(length) - 0.5 * length
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn drp_stencil_is_consistent_at_zero_wavenumber() {
        assert!((TAM_WEBB_DRP_7.modified_wavenumber_slope(0.0) - 1.0).abs() < 5e-15);
    }

    #[test]
    fn drp_stencil_is_odd_and_conservative_on_a_periodic_grid() {
        let theta = 0.73;
        let positive = TAM_WEBB_DRP_7.modified_wavenumber(theta, 1.0);
        let negative = TAM_WEBB_DRP_7.modified_wavenumber(-theta, 1.0);
        assert!((positive + negative).abs() < 1e-15);
        assert!(
            (TAM_WEBB_DRP_7.modified_wavenumber_slope(theta)
                - TAM_WEBB_DRP_7.modified_wavenumber_slope(-theta))
            .abs()
                < 1e-15
        );

        let n = 32;
        let mut field = vec![0.0; n];
        let mut derivative = vec![0.0; n];
        for (i, value) in field.iter_mut().enumerate() {
            *value = (2.0 * PI * i as f64 / n as f64).sin();
        }
        TAM_WEBB_DRP_7.derivative_periodic(&field, 1.0, &mut derivative);
        assert!(derivative.iter().sum::<f64>().abs() < 1e-14);

        let mut constant_derivative = vec![0.0; n];
        TAM_WEBB_DRP_7.derivative_periodic(&vec![3.5; n], 1.0, &mut constant_derivative);
        assert!(constant_derivative.iter().all(|value| value.abs() < 1e-15));
    }

    #[test]
    fn drp_has_lower_midband_phase_error_than_centered_second_order() {
        let theta = 1.2;
        let drp_ratio = TAM_WEBB_DRP_7.modified_wavenumber(theta, 1.0) / theta;
        let centered_ratio = theta.sin() / theta;
        assert!((drp_ratio - 1.0).abs() < (centered_ratio - 1.0).abs());
    }

    #[test]
    fn modified_wavenumber_phase_error_is_fourth_order_under_refinement() {
        let wavenumber = 1.3;
        let dx_values = [0.4, 0.2, 0.1];
        let errors: Vec<f64> = dx_values
            .iter()
            .map(|&dx| {
                let ratio = TAM_WEBB_DRP_7.modified_wavenumber(wavenumber, dx) / wavenumber;
                (ratio - 1.0).abs()
            })
            .collect();
        assert!(errors[1] < errors[0]);
        assert!(errors[2] < errors[1]);
        let first_rate = (errors[0] / errors[1]).log2();
        let second_rate = (errors[1] / errors[2]).log2();
        assert!(first_rate > 3.5, "first DRP rate={first_rate}");
        assert!(second_rate > 3.5, "second DRP rate={second_rate}");
    }

    #[test]
    fn config_and_state_validation_rejects_invalid_inputs() {
        assert!(matches!(
            DrpEuler1dConfig::with_cfl(6, 1.0, 1.0, 1.0, 0.0, 0.5),
            Err(AeroacousticsError::GridTooSmall { .. })
        ));
        assert!(matches!(
            DrpEuler1dConfig::with_cfl(16, 1.0, 1.0, 1.0, 0.0, 0.0),
            Err(AeroacousticsError::InvalidCfl(0.0))
        ));
        assert!(matches!(
            DrpEuler1dConfig::with_cfl(16, 1.0, 1.0, 1.0, 0.0, 1.1),
            Err(AeroacousticsError::InvalidCfl(1.1))
        ));
        let unit_cfl = DrpEuler1dConfig::with_cfl(16, 1.0, 1.0, 1.0, 0.0, 1.0).unwrap();
        assert!(unit_cfl.validate().is_ok());

        let config = DrpEuler1dConfig::with_cfl(16, 1.0, 1.0, 1.0, 0.0, 0.5).unwrap();
        let short_state = DrpWaveState {
            pressure: vec![0.0; 15],
            velocity: vec![0.0; 16],
        };
        assert!(matches!(
            DrpEuler1d::new(config, short_state),
            Err(AeroacousticsError::StateLengthMismatch { .. })
        ));

        let mut non_finite_state = DrpWaveState::zeros(config.n_points);
        non_finite_state.pressure[3] = f64::NAN;
        assert!(matches!(
            DrpEuler1d::new(config, non_finite_state),
            Err(AeroacousticsError::NonFiniteField { name: "LEE state" })
        ));

        let mut over_cfl = config;
        over_cfl.time_step *= 2.1;
        assert!(matches!(
            DrpEuler1d::new(over_cfl, DrpWaveState::zeros(over_cfl.n_points)),
            Err(AeroacousticsError::InvalidCfl(_))
        ));
    }

    #[test]
    fn dispersion_reports_characteristic_phase_and_group_speeds() {
        let config = DrpEuler1dConfig::with_cfl(64, 2.0 * PI, 1.2, 3.0, 0.4, 0.5).unwrap();
        let downstream = config.dispersion(1.0e-9, AcousticBranch::Downstream);
        let upstream = config.dispersion(1.0e-9, AcousticBranch::Upstream);
        assert!((downstream.phase_velocity - 3.4).abs() < 1e-7);
        assert!((downstream.group_velocity - 3.4).abs() < 1e-7);
        assert!((upstream.phase_velocity + 2.6).abs() < 1e-7);
        assert!((upstream.group_velocity + 2.6).abs() < 1e-7);

        let finite_mode = config.dispersion(7.0, AcousticBranch::Downstream);
        assert!(finite_mode.modified_wavenumber.is_finite());
        assert!(finite_mode.angular_frequency.is_finite());
        assert!(finite_mode.phase_velocity.is_finite());
        assert!(finite_mode.group_velocity.is_finite());
    }

    #[test]
    fn rk4_advances_a_fourier_eigenmode_at_discrete_frequency() {
        let config = DrpEuler1dConfig::with_cfl(128, 2.0 * PI, 1.3, 2.0, 0.3, 0.08).unwrap();
        let mode = 7.0;
        let mut state = DrpWaveState::zeros(config.n_points);
        for i in 0..config.n_points {
            let x = i as f64 * config.dx();
            state.pressure[i] = (mode * x).cos();
            state.velocity[i] = state.pressure[i] / (config.density * config.speed_of_sound);
        }
        let mut solver = DrpEuler1d::new(config, state).unwrap();
        solver.advance(80);
        let omega = config
            .dispersion(mode, AcousticBranch::Downstream)
            .angular_frequency;
        let relative_error = solver
            .state
            .pressure
            .iter()
            .enumerate()
            .map(|(i, actual)| {
                let x = i as f64 * config.dx();
                let expected = (mode * x - omega * solver.time).cos();
                (actual - expected).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / (config.n_points as f64 / 2.0).sqrt();
        assert!(relative_error < 2e-5, "relative error={relative_error:e}");
    }

    #[test]
    fn gaussian_packet_preserves_branch_impedance_and_periodicity() {
        let config = DrpEuler1dConfig::with_cfl(32, 4.0, 1.4, 2.0, 0.3, 0.2).unwrap();
        let packet =
            DrpWaveState::gaussian_packet(&config, 3.9, 0.2, 2.0, 0.7, AcousticBranch::Upstream)
                .unwrap();
        let impedance = config.density * config.speed_of_sound;
        for (&pressure, &velocity) in packet.pressure.iter().zip(&packet.velocity) {
            if pressure.abs() > 1e-12 {
                assert!((pressure + impedance * velocity).abs() < 1e-12);
            }
            assert!(pressure.is_finite() && velocity.is_finite());
        }
        assert!(packet.pressure[0].abs() > 0.0);
        assert!(packet.pressure[config.n_points - 1].abs() > 0.0);
        assert!(
            DrpWaveState::gaussian_packet(&config, 0.0, 0.0, 1.0, 1.0, AcousticBranch::Downstream,)
                .is_err()
        );
    }
}
