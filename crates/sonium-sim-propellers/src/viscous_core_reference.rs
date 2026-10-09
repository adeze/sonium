//! Bounded Lamb--Oseen viscous vortex-core evolution reference.
//!
//! The reference diffuses a prescribed signed circulation with the analytic
//! Gaussian-core relation `a²(t) = a₀² + 4νt` and evaluates the corresponding
//! tangential velocity at a finite probe radius. It is a viscous-core evidence
//! contract for wake model coupling, not a resolved viscous wake, turbulence,
//! dynamic-stall, CFD, or FW-H solver.

use std::fmt;

/// Controls bounded viscous diffusion of one prescribed vortex core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViscousCoreOptions {
    /// Initial Gaussian core radius [m].
    pub initial_core_radius_m: f64,
    /// Kinematic viscosity [m²/s].
    pub kinematic_viscosity_m2_s: f64,
    /// Explicit sampling interval for the reported evolution [s].
    pub time_step_s: f64,
    /// Number of reported evolution steps [-].
    pub steps: usize,
    /// Maximum permitted requested steps [-].
    pub maximum_steps: usize,
    /// Radius at which the Lamb--Oseen tangential velocity is evaluated [m].
    pub probe_radius_m: f64,
    /// Maximum admitted diffused core radius [m].
    pub maximum_core_radius_m: f64,
}

impl Default for ViscousCoreOptions {
    fn default() -> Self {
        Self {
            initial_core_radius_m: 1.0e-3,
            kinematic_viscosity_m2_s: 1.5e-5,
            time_step_s: 1.0e-3,
            steps: 64,
            maximum_steps: 1024,
            probe_radius_m: 2.0e-3,
            maximum_core_radius_m: 0.1,
        }
    }
}

/// One deterministic viscous-core evolution sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViscousCoreStep {
    /// Zero-based evolution index.
    pub step_index: usize,
    /// Vortex age at this sample [s].
    pub age_s: f64,
    /// Diffused Gaussian core radius [m].
    pub core_radius_m: f64,
    /// Signed Lamb--Oseen tangential velocity at the probe [m/s].
    pub tangential_velocity_m_s: f64,
    /// Prescribed signed circulation retained by the model [m²/s].
    pub circulation_m2_s: f64,
}

/// Result of the bounded viscous-core evolution reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ViscousCoreResult {
    /// Prescribed signed circulation [m²/s].
    pub circulation_m2_s: f64,
    /// Initial core radius [m].
    pub initial_core_radius_m: f64,
    /// Evolution samples, including one sample per requested step.
    pub steps: Vec<ViscousCoreStep>,
    /// Final diffused core radius [m].
    pub final_core_radius_m: f64,
    /// Final signed tangential velocity at the probe [m/s].
    pub final_tangential_velocity_m_s: f64,
}

impl ViscousCoreResult {
    /// Whether the result is finite and internally ordered.
    pub fn is_finite(&self) -> bool {
        self.circulation_m2_s.is_finite()
            && self.initial_core_radius_m.is_finite()
            && self.initial_core_radius_m > 0.0
            && self.final_core_radius_m.is_finite()
            && self.final_core_radius_m >= self.initial_core_radius_m
            && self.final_tangential_velocity_m_s.is_finite()
            && !self.steps.is_empty()
            && self
                .steps
                .windows(2)
                .all(|pair| pair[1].age_s > pair[0].age_s)
            && self.steps.iter().all(|step| {
                step.age_s.is_finite()
                    && step.age_s > 0.0
                    && step.core_radius_m.is_finite()
                    && step.core_radius_m >= self.initial_core_radius_m
                    && step.tangential_velocity_m_s.is_finite()
                    && step.circulation_m2_s == self.circulation_m2_s
            })
    }
}

/// Failure-closed errors from the viscous-core reference.
#[derive(Debug, Clone, PartialEq)]
pub enum ViscousCoreError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The prescribed circulation is not finite.
    InvalidCirculation,
    /// Diffusion exceeded the declared core-radius bound.
    CoreRadiusExceeded {
        /// Zero-based evolution index.
        step: usize,
        /// Computed radius [m].
        radius_m: f64,
        /// Maximum admitted radius [m].
        maximum_m: f64,
    },
    /// A derived quantity became non-finite.
    NonFiniteResult {
        /// Derived field name.
        field: &'static str,
    },
}

impl fmt::Display for ViscousCoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid viscous-core option {field}: {requirement}"
                )
            }
            Self::InvalidCirculation => write!(formatter, "viscous-core circulation is non-finite"),
            Self::CoreRadiusExceeded {
                step,
                radius_m,
                maximum_m,
            } => write!(
                formatter,
                "viscous-core radius {radius_m} m at step {step} exceeds {maximum_m} m"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite viscous-core result field {field}")
            }
        }
    }
}

impl std::error::Error for ViscousCoreError {}

/// Evaluate bounded analytic Lamb--Oseen core diffusion for prescribed circulation.
pub fn evaluate_viscous_core_evolution(
    circulation_m2_s: f64,
    options: ViscousCoreOptions,
) -> Result<ViscousCoreResult, ViscousCoreError> {
    validate_options(options)?;
    if !circulation_m2_s.is_finite() {
        return Err(ViscousCoreError::InvalidCirculation);
    }

    let mut steps = Vec::with_capacity(options.steps);
    for step_index in 0..options.steps {
        let age_s = (step_index + 1) as f64 * options.time_step_s;
        let core_radius_m = (options.initial_core_radius_m.powi(2)
            + 4.0 * options.kinematic_viscosity_m2_s * age_s)
            .sqrt();
        if !core_radius_m.is_finite() {
            return Err(ViscousCoreError::NonFiniteResult {
                field: "core_radius_m",
            });
        }
        if core_radius_m > options.maximum_core_radius_m {
            return Err(ViscousCoreError::CoreRadiusExceeded {
                step: step_index,
                radius_m: core_radius_m,
                maximum_m: options.maximum_core_radius_m,
            });
        }

        let ratio = options.probe_radius_m.powi(2) / core_radius_m.powi(2);
        let vortex_fraction = -(-ratio).exp_m1();
        let tangential_velocity_m_s = circulation_m2_s
            / (2.0 * std::f64::consts::PI * options.probe_radius_m)
            * vortex_fraction;
        if !age_s.is_finite() || !tangential_velocity_m_s.is_finite() {
            return Err(ViscousCoreError::NonFiniteResult {
                field: "evolution_sample",
            });
        }
        steps.push(ViscousCoreStep {
            step_index,
            age_s,
            core_radius_m,
            tangential_velocity_m_s,
            circulation_m2_s,
        });
    }

    let final_step = steps
        .last()
        .copied()
        .ok_or(ViscousCoreError::NonFiniteResult { field: "steps" })?;
    let result = ViscousCoreResult {
        circulation_m2_s,
        initial_core_radius_m: options.initial_core_radius_m,
        steps,
        final_core_radius_m: final_step.core_radius_m,
        final_tangential_velocity_m_s: final_step.tangential_velocity_m_s,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ViscousCoreError::NonFiniteResult {
            field: "viscous_core_result",
        })
    }
}

fn validate_options(options: ViscousCoreOptions) -> Result<(), ViscousCoreError> {
    for (field, value, requirement) in [
        (
            "initial_core_radius_m",
            options.initial_core_radius_m,
            "finite and positive",
        ),
        (
            "kinematic_viscosity_m2_s",
            options.kinematic_viscosity_m2_s,
            "finite and non-negative",
        ),
        ("time_step_s", options.time_step_s, "finite and positive"),
        (
            "probe_radius_m",
            options.probe_radius_m,
            "finite and positive",
        ),
        (
            "maximum_core_radius_m",
            options.maximum_core_radius_m,
            "finite and at least the initial radius",
        ),
    ] {
        let valid = if field == "kinematic_viscosity_m2_s" {
            value.is_finite() && value >= 0.0
        } else if field == "maximum_core_radius_m" {
            value.is_finite() && value >= options.initial_core_radius_m
        } else {
            value.is_finite() && value > 0.0
        };
        if !valid {
            return Err(ViscousCoreError::InvalidOptions { field, requirement });
        }
    }
    if options.steps == 0 || options.steps > options.maximum_steps {
        return Err(ViscousCoreError::InvalidOptions {
            field: "steps",
            requirement: "positive and no greater than maximum_steps",
        });
    }
    if options.maximum_steps == 0 {
        return Err(ViscousCoreError::InvalidOptions {
            field: "maximum_steps",
            requirement: "positive",
        });
    }
    Ok(())
}
