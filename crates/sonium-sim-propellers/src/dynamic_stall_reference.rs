//! Bounded first-order dynamic-stall state reference.
//!
//! This module evolves a phenomenological separated-flow state over a supplied
//! angle-of-attack history. Attached lift, separation onset, delayed stall,
//! and recovery are explicit and deterministic; the contract is useful for
//! exercising an unsteady blade-element coupling seam. It is not a
//! Beddoes--Leishman implementation, a CFD/URANS solver, or measured dynamic
//! stall validation.

use std::fmt;

/// One prescribed angle-of-attack history sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicStallInput {
    /// Sample time [s].
    pub time_s: f64,
    /// Section angle of attack [deg].
    pub angle_of_attack_deg: f64,
}

/// Controls for the bounded first-order dynamic-stall state model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicStallOptions {
    /// Linear attached lift slope per radian [-].
    pub lift_slope_per_rad: f64,
    /// Zero-lift angle [deg].
    pub zero_lift_angle_deg: f64,
    /// Absolute angle where separation begins [deg].
    pub separation_onset_angle_deg: f64,
    /// Width of the smooth separation transition [deg].
    pub separation_transition_width_deg: f64,
    /// Fraction of attached lift retained when fully separated [-].
    pub separated_lift_fraction: f64,
    /// Maximum magnitude of attached lift coefficient [-].
    pub maximum_attached_lift_coefficient: f64,
    /// Attached-lift state time constant [s].
    pub lift_time_constant_s: f64,
    /// Separation-state time constant while stall develops [s].
    pub separation_time_constant_s: f64,
    /// Separation-state time constant during recovery [s].
    pub recovery_time_constant_s: f64,
    /// Maximum admitted input step [s].
    pub maximum_step_s: f64,
    /// Maximum admitted history length [-].
    pub maximum_samples: usize,
}

impl Default for DynamicStallOptions {
    fn default() -> Self {
        Self {
            lift_slope_per_rad: 5.7,
            zero_lift_angle_deg: 0.0,
            separation_onset_angle_deg: 12.0,
            separation_transition_width_deg: 4.0,
            separated_lift_fraction: 0.35,
            maximum_attached_lift_coefficient: 1.6,
            lift_time_constant_s: 0.015,
            separation_time_constant_s: 0.04,
            recovery_time_constant_s: 0.12,
            maximum_step_s: 0.1,
            maximum_samples: 4096,
        }
    }
}

/// One evaluated dynamic-stall state sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicStallSample {
    /// Sample time [s].
    pub time_s: f64,
    /// Input angle of attack [deg].
    pub angle_of_attack_deg: f64,
    /// Instantaneous attached-flow lift coefficient before separation loss [-].
    pub attached_lift_coefficient: f64,
    /// Quasi-steady separation target in [0, 1] [-].
    pub separation_target: f64,
    /// Lagged separation state in [0, 1] [-].
    pub separation_state: f64,
    /// Lift coefficient after delayed separation and lift-state lag [-].
    pub dynamic_lift_coefficient: f64,
}

/// Result of the bounded dynamic-stall state evolution.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicStallResult {
    /// Evaluated state samples in input order.
    pub samples: Vec<DynamicStallSample>,
    /// Maximum absolute dynamic lift coefficient [-].
    pub maximum_abs_dynamic_lift_coefficient: f64,
    /// Maximum lagged separation state [-].
    pub maximum_separation_state: f64,
}

impl DynamicStallResult {
    /// Whether all states are finite, ordered, and bounded.
    pub fn is_finite(&self) -> bool {
        self.maximum_abs_dynamic_lift_coefficient.is_finite()
            && self.maximum_abs_dynamic_lift_coefficient >= 0.0
            && self.maximum_separation_state.is_finite()
            && (0.0..=1.0).contains(&self.maximum_separation_state)
            && !self.samples.is_empty()
            && self
                .samples
                .windows(2)
                .all(|pair| pair[1].time_s > pair[0].time_s)
            && self.samples.iter().all(|sample| {
                sample.time_s.is_finite()
                    && sample.angle_of_attack_deg.is_finite()
                    && sample.attached_lift_coefficient.is_finite()
                    && sample.separation_target.is_finite()
                    && (0.0..=1.0).contains(&sample.separation_target)
                    && sample.separation_state.is_finite()
                    && (0.0..=1.0).contains(&sample.separation_state)
                    && sample.dynamic_lift_coefficient.is_finite()
            })
    }
}

/// Failure-closed errors from the dynamic-stall reference.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicStallError {
    /// An option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No history samples were supplied.
    EmptyHistory,
    /// A history sample is malformed.
    InvalidInput {
        /// Zero-based sample index.
        index: usize,
        /// Input field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Input time is not strictly increasing.
    NonMonotonicTime {
        /// Zero-based sample index.
        index: usize,
    },
    /// An input step exceeds the declared integration bound.
    StepTooLarge {
        /// Zero-based sample index.
        index: usize,
        /// Requested step [s].
        step_s: f64,
        /// Maximum admitted step [s].
        maximum_s: f64,
    },
    /// A derived state became non-finite.
    NonFiniteResult {
        /// Derived field name.
        field: &'static str,
    },
}

impl fmt::Display for DynamicStallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid dynamic-stall option {field}: {requirement}"
            ),
            Self::EmptyHistory => write!(formatter, "dynamic-stall history must not be empty"),
            Self::InvalidInput {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid dynamic-stall input {index} field {field}: {requirement}"
            ),
            Self::NonMonotonicTime { index } => {
                write!(
                    formatter,
                    "dynamic-stall input time is not increasing at {index}"
                )
            }
            Self::StepTooLarge {
                index,
                step_s,
                maximum_s,
            } => write!(
                formatter,
                "dynamic-stall step {step_s} s at {index} exceeds {maximum_s} s"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite dynamic-stall result field {field}")
            }
        }
    }
}

impl std::error::Error for DynamicStallError {}

/// Evaluate a prescribed angle-of-attack history with delayed separation.
pub fn evaluate_dynamic_stall_reference(
    history: &[DynamicStallInput],
    options: DynamicStallOptions,
) -> Result<DynamicStallResult, DynamicStallError> {
    validate_options(options)?;
    if history.is_empty() {
        return Err(DynamicStallError::EmptyHistory);
    }
    if history.len() > options.maximum_samples {
        return Err(DynamicStallError::InvalidInput {
            index: history.len(),
            field: "history",
            requirement: "no greater than maximum_samples",
        });
    }

    let mut samples = Vec::with_capacity(history.len());
    let mut separation_state = 0.0;
    let mut dynamic_lift_coefficient = 0.0;
    let mut maximum_abs_dynamic_lift_coefficient: f64 = 0.0;
    let mut maximum_separation_state: f64 = 0.0;

    for (index, &input) in history.iter().enumerate() {
        validate_input(index, input)?;
        let previous_time_s = if index == 0 {
            input.time_s
        } else {
            history[index - 1].time_s
        };
        let step_s = input.time_s - previous_time_s;
        if index > 0 && step_s <= 0.0 {
            return Err(DynamicStallError::NonMonotonicTime { index });
        }
        if index > 0 && step_s > options.maximum_step_s {
            return Err(DynamicStallError::StepTooLarge {
                index,
                step_s,
                maximum_s: options.maximum_step_s,
            });
        }

        let attached_lift_coefficient = attached_lift(input.angle_of_attack_deg, options);
        let separation_target = separation_target(input.angle_of_attack_deg, options);
        if index == 0 {
            separation_state = separation_target;
            dynamic_lift_coefficient =
                separated_lift_target(attached_lift_coefficient, separation_state, options);
        } else {
            let separation_time_constant = if separation_target > separation_state {
                options.separation_time_constant_s
            } else {
                options.recovery_time_constant_s
            };
            separation_state = relax(
                separation_state,
                separation_target,
                step_s,
                separation_time_constant,
            );
            let lift_target =
                separated_lift_target(attached_lift_coefficient, separation_state, options);
            dynamic_lift_coefficient = relax(
                dynamic_lift_coefficient,
                lift_target,
                step_s,
                options.lift_time_constant_s,
            );
        }

        if !separation_state.is_finite() || !dynamic_lift_coefficient.is_finite() {
            return Err(DynamicStallError::NonFiniteResult { field: "state" });
        }
        maximum_abs_dynamic_lift_coefficient =
            maximum_abs_dynamic_lift_coefficient.max(dynamic_lift_coefficient.abs());
        maximum_separation_state = maximum_separation_state.max(separation_state);
        samples.push(DynamicStallSample {
            time_s: input.time_s,
            angle_of_attack_deg: input.angle_of_attack_deg,
            attached_lift_coefficient,
            separation_target,
            separation_state,
            dynamic_lift_coefficient,
        });
    }

    let result = DynamicStallResult {
        samples,
        maximum_abs_dynamic_lift_coefficient,
        maximum_separation_state,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DynamicStallError::NonFiniteResult {
            field: "dynamic_stall_result",
        })
    }
}

fn attached_lift(angle_of_attack_deg: f64, options: DynamicStallOptions) -> f64 {
    let angle_rad = (angle_of_attack_deg - options.zero_lift_angle_deg).to_radians();
    (options.lift_slope_per_rad * angle_rad).clamp(
        -options.maximum_attached_lift_coefficient,
        options.maximum_attached_lift_coefficient,
    )
}

fn separation_target(angle_of_attack_deg: f64, options: DynamicStallOptions) -> f64 {
    let normalized = (angle_of_attack_deg.abs() - options.separation_onset_angle_deg)
        / options.separation_transition_width_deg;
    if normalized <= 0.0 {
        0.0
    } else if normalized >= 1.0 {
        1.0
    } else {
        normalized * normalized * (3.0 - 2.0 * normalized)
    }
}

fn separated_lift_target(
    attached_lift_coefficient: f64,
    separation_state: f64,
    options: DynamicStallOptions,
) -> f64 {
    attached_lift_coefficient * (1.0 - separation_state * (1.0 - options.separated_lift_fraction))
}

fn relax(current: f64, target: f64, step_s: f64, time_constant_s: f64) -> f64 {
    let gain = 1.0 - (-step_s / time_constant_s).exp();
    current + (target - current) * gain
}

fn validate_options(options: DynamicStallOptions) -> Result<(), DynamicStallError> {
    for (field, value, requirement) in [
        (
            "lift_slope_per_rad",
            options.lift_slope_per_rad,
            "finite and positive",
        ),
        (
            "separation_onset_angle_deg",
            options.separation_onset_angle_deg,
            "finite and non-negative",
        ),
        (
            "separation_transition_width_deg",
            options.separation_transition_width_deg,
            "finite and positive",
        ),
        (
            "maximum_attached_lift_coefficient",
            options.maximum_attached_lift_coefficient,
            "finite and positive",
        ),
        (
            "lift_time_constant_s",
            options.lift_time_constant_s,
            "finite and positive",
        ),
        (
            "separation_time_constant_s",
            options.separation_time_constant_s,
            "finite and positive",
        ),
        (
            "recovery_time_constant_s",
            options.recovery_time_constant_s,
            "finite and positive",
        ),
        (
            "maximum_step_s",
            options.maximum_step_s,
            "finite and positive",
        ),
    ] {
        if !value.is_finite()
            || (field == "separation_onset_angle_deg" && value < 0.0)
            || (field != "separation_onset_angle_deg" && value <= 0.0)
        {
            return Err(DynamicStallError::InvalidOptions { field, requirement });
        }
    }
    if !options.zero_lift_angle_deg.is_finite() {
        return Err(DynamicStallError::InvalidOptions {
            field: "zero_lift_angle_deg",
            requirement: "finite",
        });
    }
    if !options.separated_lift_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.separated_lift_fraction)
    {
        return Err(DynamicStallError::InvalidOptions {
            field: "separated_lift_fraction",
            requirement: "finite and in [0, 1]",
        });
    }
    if options.maximum_samples == 0 {
        return Err(DynamicStallError::InvalidOptions {
            field: "maximum_samples",
            requirement: "positive",
        });
    }
    Ok(())
}

fn validate_input(index: usize, input: DynamicStallInput) -> Result<(), DynamicStallError> {
    if !input.time_s.is_finite() || input.time_s < 0.0 {
        return Err(DynamicStallError::InvalidInput {
            index,
            field: "time_s",
            requirement: "finite and non-negative",
        });
    }
    if !input.angle_of_attack_deg.is_finite() {
        return Err(DynamicStallError::InvalidInput {
            index,
            field: "angle_of_attack_deg",
            requirement: "finite",
        });
    }
    Ok(())
}
