//! Bounded uniform-mean-flow moving-surface FW-H reference.
//!
//! This module extends the reduced moving-surface source contract with a
//! spatially uniform, subsonic mean flow.  The propagation time is obtained
//! from the convected acoustic-ray identity
//! `|x_observer - x_source - U*tau| = c*tau`; source derivatives use the
//! relative source velocity and the corresponding retarded-time Jacobian.
//! It is a quiescent/constant-flow far-field reference, not a full
//! Farassat FW-H, mean-flow CFD, or turbulence/acoustic-mode solver.

use std::fmt;

use crate::{MovingFwhElementHistory, MovingFwhSample, Point3};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// Controls for the uniform-mean-flow moving-surface reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanFlowFwhOptions {
    /// Requested observer time [s].
    pub observer_time_s: f64,
    /// Observer position [m].
    pub observer_position: Point3,
    /// Uniform mean-flow velocity in the laboratory frame [m/s].
    pub mean_flow_m_s: Point3,
    /// Medium density [kg/m³].
    pub density_kg_m3: f64,
    /// Sound speed relative to the mean flow [m/s].
    pub sound_speed_m_s: f64,
    /// Minimum admitted Euclidean source-observer distance [m].
    pub minimum_observer_distance_m: f64,
    /// Maximum admitted mean-flow Mach number [−].
    pub maximum_mean_flow_mach: f64,
    /// Maximum admitted source Mach number relative to the mean flow [−].
    pub maximum_relative_source_mach: f64,
    /// Minimum admitted retarded-time Jacobian `1 - M_relative,r` [−].
    pub minimum_doppler_factor: f64,
    /// Absolute retarded-time root tolerance [s].
    pub retarded_time_tolerance_s: f64,
    /// Maximum bisection iterations per retarded-time root.
    pub maximum_retarded_iterations: usize,
    /// Maximum number of surface elements.
    pub maximum_elements: usize,
    /// Maximum samples in one element history.
    pub maximum_samples_per_element: usize,
    /// Maximum samples across all histories.
    pub maximum_total_samples: usize,
}

impl Default for MeanFlowFwhOptions {
    fn default() -> Self {
        Self {
            observer_time_s: 1.0,
            observer_position: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            mean_flow_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            density_kg_m3: 1.2,
            sound_speed_m_s: 343.0,
            minimum_observer_distance_m: 0.1,
            maximum_mean_flow_mach: 0.5,
            maximum_relative_source_mach: 0.5,
            minimum_doppler_factor: 0.25,
            retarded_time_tolerance_s: 1.0e-10,
            maximum_retarded_iterations: 64,
            maximum_elements: 16_384,
            maximum_samples_per_element: 4_096,
            maximum_total_samples: 1_000_000,
        }
    }
}

/// One element's convected retarded-time contribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanFlowFwhElementResult {
    /// Source time selected by the convected retarded-time equation [s].
    pub retarded_time_s: f64,
    /// Observer-to-source-event propagation time [s].
    pub propagation_time_s: f64,
    /// Convected acoustic-ray distance `c*tau` [m].
    pub convected_distance_m: f64,
    /// Unit propagation direction in the moving medium frame [-].
    pub propagation_direction: Point3,
    /// Mean-flow velocity projected onto the propagation direction [m/s].
    pub mean_flow_projection_m_s: f64,
    /// Relative source Mach number `|v_source-U|/c` [-].
    pub relative_source_mach: f64,
    /// Retarded-time Jacobian `1 - (v_source-U)·n/c` [-].
    pub doppler_factor: f64,
    /// Thickness/monopole pressure contribution [Pa].
    pub thickness_pressure_pa: f64,
    /// Loading/dipole pressure contribution [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of thickness and loading pressure [Pa].
    pub total_pressure_pa: f64,
}

/// Uniform-mean-flow moving-surface FW-H result.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanFlowFwhResult {
    /// Uniform mean flow used by the evaluation [m/s].
    pub mean_flow_m_s: Point3,
    /// Mean-flow Mach number [-].
    pub mean_flow_mach: f64,
    /// Sum of thickness terms [Pa].
    pub thickness_pressure_pa: f64,
    /// Sum of loading terms [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of all element contributions [Pa].
    pub total_pressure_pa: f64,
    /// Per-element retarded-time and convected-ray diagnostics.
    pub elements: Vec<MeanFlowFwhElementResult>,
    /// Minimum Euclidean source-observer distance [m].
    pub minimum_source_observer_distance_m: f64,
    /// Maximum relative source Mach number [-].
    pub maximum_relative_source_mach: f64,
}

impl MeanFlowFwhResult {
    /// Returns whether all contributions and diagnostics are finite.
    pub fn is_finite(&self) -> bool {
        self.mean_flow_m_s.finite()
            && self.mean_flow_mach.is_finite()
            && self.mean_flow_mach >= 0.0
            && self.thickness_pressure_pa.is_finite()
            && self.loading_pressure_pa.is_finite()
            && self.total_pressure_pa.is_finite()
            && self.elements.iter().all(|element| {
                element.retarded_time_s.is_finite()
                    && element.propagation_time_s.is_finite()
                    && element.propagation_time_s > 0.0
                    && element.convected_distance_m.is_finite()
                    && element.convected_distance_m > 0.0
                    && element.propagation_direction.finite()
                    && (element.propagation_direction.norm() - 1.0).abs() < 1.0e-8
                    && element.mean_flow_projection_m_s.is_finite()
                    && element.relative_source_mach.is_finite()
                    && element.relative_source_mach >= 0.0
                    && element.doppler_factor.is_finite()
                    && element.doppler_factor > 0.0
                    && element.thickness_pressure_pa.is_finite()
                    && element.loading_pressure_pa.is_finite()
                    && element.total_pressure_pa.is_finite()
            })
            && self.minimum_source_observer_distance_m.is_finite()
            && self.minimum_source_observer_distance_m > 0.0
            && self.maximum_relative_source_mach.is_finite()
            && self.maximum_relative_source_mach >= 0.0
    }
}

/// Failure-closed errors for the uniform-mean-flow reference.
#[derive(Debug, Clone, PartialEq)]
pub enum MeanFlowFwhError {
    /// An option outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No surface histories were supplied.
    EmptySurface,
    /// Element count exceeds the declared budget.
    TooManyElements { count: usize, maximum: usize },
    /// One history exceeds its sample budget.
    TooManySamples {
        /// Zero-based element index.
        element_index: usize,
        /// Supplied sample count.
        count: usize,
        /// Declared maximum.
        maximum: usize,
    },
    /// Total sample count exceeds the declared budget.
    TooManyTotalSamples { count: usize, maximum: usize },
    /// A history has too few samples to differentiate source strengths.
    InsufficientSamples {
        /// Zero-based element index.
        element_index: usize,
        /// Required minimum.
        minimum: usize,
    },
    /// A sample field is malformed.
    InvalidSample {
        /// Zero-based element index.
        element_index: usize,
        /// Zero-based sample index.
        sample_index: usize,
        /// Field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Time samples are not strictly increasing.
    NonIncreasingTime {
        /// Zero-based element index.
        element_index: usize,
        /// Previous sample time [s].
        previous_time_s: f64,
        /// Current sample time [s].
        current_time_s: f64,
    },
    /// The requested observer time is outside a history's arrival window.
    RetardedTimeOutsideHistory {
        /// Zero-based element index.
        element_index: usize,
        /// Earliest possible arrival [s].
        earliest_arrival_s: f64,
        /// Latest possible arrival [s].
        latest_arrival_s: f64,
        /// Requested observer time [s].
        observer_time_s: f64,
    },
    /// Bisection did not meet the declared retarded-time tolerance.
    RetardedTimeNoConvergence {
        /// Zero-based element index.
        element_index: usize,
        /// Final absolute arrival-time residual [s].
        residual_s: f64,
    },
    /// The source-observer distance is below the far-field admission bound.
    ObserverTooClose {
        /// Zero-based element index.
        element_index: usize,
        /// Euclidean distance [m].
        distance_m: f64,
        /// Minimum admitted distance [m].
        minimum_m: f64,
    },
    /// The mean flow is too close to or above sonic.
    MeanFlowMachLimit {
        /// Mean-flow Mach number [-].
        mach: f64,
        /// Maximum admitted Mach number [-].
        maximum_mach: f64,
    },
    /// The relative source speed exceeds the admitted subsonic budget.
    RelativeSourceMachLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Relative source Mach number [-].
        mach: f64,
        /// Maximum admitted Mach number [-].
        maximum_mach: f64,
    },
    /// The retarded source is too close to sonic in the ray direction.
    DopplerLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Measured Jacobian [-].
        doppler_factor: f64,
        /// Minimum admitted factor [-].
        minimum_factor: f64,
    },
    /// A derived result became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for MeanFlowFwhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid mean-flow FW-H option {field}: {requirement}"
                )
            }
            Self::EmptySurface => formatter.write_str("mean-flow FW-H surface is empty"),
            Self::TooManyElements { count, maximum } => write!(
                formatter,
                "mean-flow FW-H has {count} elements, maximum is {maximum}"
            ),
            Self::TooManySamples {
                element_index,
                count,
                maximum,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} has {count} samples, maximum is {maximum}"
            ),
            Self::TooManyTotalSamples { count, maximum } => write!(
                formatter,
                "mean-flow FW-H has {count} total samples, maximum is {maximum}"
            ),
            Self::InsufficientSamples {
                element_index,
                minimum,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} needs at least {minimum} samples"
            ),
            Self::InvalidSample {
                element_index,
                sample_index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid mean-flow FW-H sample {element_index}/{sample_index} {field}: {requirement}"
            ),
            Self::NonIncreasingTime {
                element_index,
                previous_time_s,
                current_time_s,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} times are not increasing: {previous_time_s} then {current_time_s}"
            ),
            Self::RetardedTimeOutsideHistory {
                element_index,
                earliest_arrival_s,
                latest_arrival_s,
                observer_time_s,
            } => write!(
                formatter,
                "mean-flow FW-H observer time {observer_time_s} is outside element {element_index} arrival window [{earliest_arrival_s}, {latest_arrival_s}]"
            ),
            Self::RetardedTimeNoConvergence {
                element_index,
                residual_s,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} retarded-time residual {residual_s} did not converge"
            ),
            Self::ObserverTooClose {
                element_index,
                distance_m,
                minimum_m,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} distance {distance_m} is below {minimum_m}"
            ),
            Self::MeanFlowMachLimit { mach, maximum_mach } => write!(
                formatter,
                "mean-flow Mach {mach} exceeds the admitted maximum {maximum_mach}"
            ),
            Self::RelativeSourceMachLimit {
                element_index,
                mach,
                maximum_mach,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} relative source Mach {mach} exceeds {maximum_mach}"
            ),
            Self::DopplerLimit {
                element_index,
                doppler_factor,
                minimum_factor,
            } => write!(
                formatter,
                "mean-flow FW-H element {element_index} Doppler factor {doppler_factor} is below {minimum_factor}"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite mean-flow FW-H result field {field}")
            }
        }
    }
}

impl std::error::Error for MeanFlowFwhError {}

/// Evaluate reduced far-field thickness/loading terms in a uniform mean flow.
pub fn evaluate_mean_flow_fwh_reference(
    histories: &[MovingFwhElementHistory],
    options: MeanFlowFwhOptions,
) -> Result<MeanFlowFwhResult, MeanFlowFwhError> {
    validate_options(options)?;
    if histories.is_empty() {
        return Err(MeanFlowFwhError::EmptySurface);
    }
    if histories.len() > options.maximum_elements {
        return Err(MeanFlowFwhError::TooManyElements {
            count: histories.len(),
            maximum: options.maximum_elements,
        });
    }

    let mut total_samples = 0usize;
    for (element_index, history) in histories.iter().enumerate() {
        validate_history(element_index, history, options)?;
        total_samples = total_samples.checked_add(history.samples.len()).ok_or(
            MeanFlowFwhError::TooManyTotalSamples {
                count: usize::MAX,
                maximum: options.maximum_total_samples,
            },
        )?;
    }
    if total_samples > options.maximum_total_samples {
        return Err(MeanFlowFwhError::TooManyTotalSamples {
            count: total_samples,
            maximum: options.maximum_total_samples,
        });
    }

    let mean_flow_speed_m_s = norm(options.mean_flow_m_s);
    let mean_flow_mach = mean_flow_speed_m_s / options.sound_speed_m_s;
    if mean_flow_mach > options.maximum_mean_flow_mach {
        return Err(MeanFlowFwhError::MeanFlowMachLimit {
            mach: mean_flow_mach,
            maximum_mach: options.maximum_mean_flow_mach,
        });
    }

    let mut thickness_pressure_pa = 0.0;
    let mut loading_pressure_pa = 0.0;
    let mut elements = Vec::with_capacity(histories.len());
    let mut minimum_source_observer_distance_m = f64::INFINITY;
    let mut maximum_relative_source_mach: f64 = 0.0;

    for (element_index, history) in histories.iter().enumerate() {
        let retarded_time_s = solve_retarded_time(element_index, history, options)?;
        let bracket = bracket_index(history, retarded_time_s);
        let source = interpolate_sample(history, bracket, retarded_time_s);
        let left = history.samples[bracket];
        let right = history.samples[bracket + 1];
        let delta_time_s = right.time_s - left.time_s;
        let source_velocity = scale(subtract(right.position, left.position), 1.0 / delta_time_s);
        let relative_source_velocity = subtract(source_velocity, options.mean_flow_m_s);
        let relative_source_mach = norm(relative_source_velocity) / options.sound_speed_m_s;
        if relative_source_mach > options.maximum_relative_source_mach {
            return Err(MeanFlowFwhError::RelativeSourceMachLimit {
                element_index,
                mach: relative_source_mach,
                maximum_mach: options.maximum_relative_source_mach,
            });
        }

        let propagation_time_s = options.observer_time_s - retarded_time_s;
        let path = subtract(
            subtract(options.observer_position, source.position),
            scale(options.mean_flow_m_s, propagation_time_s),
        );
        let convected_distance_m = norm(path);
        if !convected_distance_m.is_finite() || convected_distance_m <= 0.0 {
            return Err(MeanFlowFwhError::NonFiniteResult {
                field: "convected_distance_m",
            });
        }
        let direction = scale(path, 1.0 / convected_distance_m);
        let euclidean_path = subtract(options.observer_position, source.position);
        let distance_m = norm(euclidean_path);
        if distance_m < options.minimum_observer_distance_m {
            return Err(MeanFlowFwhError::ObserverTooClose {
                element_index,
                distance_m,
                minimum_m: options.minimum_observer_distance_m,
            });
        }
        let relative_radial_speed_m_s = dot(relative_source_velocity, direction);
        let doppler_factor = 1.0 - relative_radial_speed_m_s / options.sound_speed_m_s;
        if doppler_factor < options.minimum_doppler_factor {
            return Err(MeanFlowFwhError::DopplerLimit {
                element_index,
                doppler_factor,
                minimum_factor: options.minimum_doppler_factor,
            });
        }

        let q = options.density_kg_m3 * source.normal_velocity_m_s * source.area_m2;
        let q_left = options.density_kg_m3 * left.normal_velocity_m_s * left.area_m2;
        let q_right = options.density_kg_m3 * right.normal_velocity_m_s * right.area_m2;
        let q_dot = (q_right - q_left) / delta_time_s;

        let normal = normalize(source.normal);
        let loading_source = source.pressure_pa * source.area_m2 * dot(normal, direction);
        let left_loading = left.pressure_pa
            * left.area_m2
            * dot(
                normalize(left.normal),
                direction_at(options.observer_position, left.position, options)?,
            );
        let right_loading = right.pressure_pa
            * right.area_m2
            * dot(
                normalize(right.normal),
                direction_at(options.observer_position, right.position, options)?,
            );
        let loading_source_dot = (right_loading - left_loading) / delta_time_s;

        // The convected path and relative radial speed retain the reduced
        // far-field source identities while the Doppler factor is now the
        // uniform-flow retarded-time Jacobian.
        let thickness = (q_dot / (convected_distance_m * doppler_factor)
            + q * relative_radial_speed_m_s
                / (convected_distance_m * convected_distance_m * doppler_factor))
            / FOUR_PI;
        let loading = (loading_source_dot / (convected_distance_m * doppler_factor)
            + loading_source * relative_radial_speed_m_s
                / (convected_distance_m * convected_distance_m * doppler_factor))
            / (FOUR_PI * options.sound_speed_m_s);
        let total = thickness + loading;
        let contribution = MeanFlowFwhElementResult {
            retarded_time_s,
            propagation_time_s,
            convected_distance_m,
            propagation_direction: direction,
            mean_flow_projection_m_s: dot(options.mean_flow_m_s, direction),
            relative_source_mach,
            doppler_factor,
            thickness_pressure_pa: thickness,
            loading_pressure_pa: loading,
            total_pressure_pa: total,
        };
        if !contribution.retarded_time_s.is_finite()
            || !contribution.propagation_time_s.is_finite()
            || !contribution.convected_distance_m.is_finite()
            || !contribution.propagation_direction.finite()
            || !contribution.mean_flow_projection_m_s.is_finite()
            || !contribution.relative_source_mach.is_finite()
            || !contribution.doppler_factor.is_finite()
            || !contribution.thickness_pressure_pa.is_finite()
            || !contribution.loading_pressure_pa.is_finite()
            || !contribution.total_pressure_pa.is_finite()
        {
            return Err(MeanFlowFwhError::NonFiniteResult {
                field: "mean_flow_fwh_element",
            });
        }
        thickness_pressure_pa += thickness;
        loading_pressure_pa += loading;
        elements.push(contribution);
        minimum_source_observer_distance_m = minimum_source_observer_distance_m.min(distance_m);
        maximum_relative_source_mach = maximum_relative_source_mach.max(relative_source_mach);
    }

    let result = MeanFlowFwhResult {
        mean_flow_m_s: options.mean_flow_m_s,
        mean_flow_mach,
        thickness_pressure_pa,
        loading_pressure_pa,
        total_pressure_pa: thickness_pressure_pa + loading_pressure_pa,
        elements,
        minimum_source_observer_distance_m,
        maximum_relative_source_mach,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(MeanFlowFwhError::NonFiniteResult {
            field: "mean_flow_fwh_result",
        })
    }
}

fn validate_options(options: MeanFlowFwhOptions) -> Result<(), MeanFlowFwhError> {
    if !options.observer_time_s.is_finite() {
        return Err(invalid_option("observer_time_s", "finite"));
    }
    if !options.observer_position.finite() {
        return Err(invalid_option("observer_position", "finite"));
    }
    if !options.mean_flow_m_s.finite() {
        return Err(invalid_option("mean_flow_m_s", "finite"));
    }
    for (field, value, requirement) in [
        ("density_kg_m3", options.density_kg_m3, "finite and > 0"),
        ("sound_speed_m_s", options.sound_speed_m_s, "finite and > 0"),
        (
            "minimum_observer_distance_m",
            options.minimum_observer_distance_m,
            "finite and > 0",
        ),
        (
            "maximum_mean_flow_mach",
            options.maximum_mean_flow_mach,
            "finite and in [0, 1)",
        ),
        (
            "maximum_relative_source_mach",
            options.maximum_relative_source_mach,
            "finite and in [0, 1)",
        ),
        (
            "minimum_doppler_factor",
            options.minimum_doppler_factor,
            "finite and in (0, 1]",
        ),
        (
            "retarded_time_tolerance_s",
            options.retarded_time_tolerance_s,
            "finite and > 0",
        ),
    ] {
        let valid = match field {
            "maximum_mean_flow_mach" | "maximum_relative_source_mach" => {
                value.is_finite() && (0.0..1.0).contains(&value)
            }
            "minimum_doppler_factor" => value.is_finite() && value > 0.0 && value <= 1.0,
            _ => value.is_finite() && value > 0.0,
        };
        if !valid {
            return Err(invalid_option(field, requirement));
        }
    }
    if options.maximum_retarded_iterations == 0 {
        return Err(invalid_option(
            "maximum_retarded_iterations",
            "greater than zero",
        ));
    }
    if options.maximum_elements == 0 {
        return Err(invalid_option("maximum_elements", "greater than zero"));
    }
    if options.maximum_samples_per_element < 2 {
        return Err(invalid_option(
            "maximum_samples_per_element",
            "at least two",
        ));
    }
    if options.maximum_total_samples < 2 {
        return Err(invalid_option("maximum_total_samples", "at least two"));
    }
    Ok(())
}

fn invalid_option(field: &'static str, requirement: &'static str) -> MeanFlowFwhError {
    MeanFlowFwhError::InvalidOptions { field, requirement }
}

fn validate_history(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: MeanFlowFwhOptions,
) -> Result<(), MeanFlowFwhError> {
    if history.samples.len() < 2 {
        return Err(MeanFlowFwhError::InsufficientSamples {
            element_index,
            minimum: 2,
        });
    }
    if history.samples.len() > options.maximum_samples_per_element {
        return Err(MeanFlowFwhError::TooManySamples {
            element_index,
            count: history.samples.len(),
            maximum: options.maximum_samples_per_element,
        });
    }
    for (sample_index, sample) in history.samples.iter().enumerate() {
        validate_sample(element_index, sample_index, sample)?;
        if sample_index > 0 {
            let previous = history.samples[sample_index - 1].time_s;
            if sample.time_s <= previous {
                return Err(MeanFlowFwhError::NonIncreasingTime {
                    element_index,
                    previous_time_s: previous,
                    current_time_s: sample.time_s,
                });
            }
        }
    }
    Ok(())
}

fn validate_sample(
    element_index: usize,
    sample_index: usize,
    sample: &MovingFwhSample,
) -> Result<(), MeanFlowFwhError> {
    for (field, value) in [
        ("time_s", sample.time_s),
        ("area_m2", sample.area_m2),
        ("pressure_pa", sample.pressure_pa),
        ("normal_velocity_m_s", sample.normal_velocity_m_s),
    ] {
        if !value.is_finite() || (field == "area_m2" && value <= 0.0) {
            return Err(MeanFlowFwhError::InvalidSample {
                element_index,
                sample_index,
                field,
                requirement: if field == "area_m2" {
                    "finite and > 0"
                } else {
                    "finite"
                },
            });
        }
    }
    if !sample.position.finite() {
        return Err(invalid_sample(
            element_index,
            sample_index,
            "position",
            "finite",
        ));
    }
    if !sample.normal.finite() || norm(sample.normal) <= 0.0 {
        return Err(invalid_sample(
            element_index,
            sample_index,
            "normal",
            "finite and nonzero",
        ));
    }
    Ok(())
}

fn invalid_sample(
    element_index: usize,
    sample_index: usize,
    field: &'static str,
    requirement: &'static str,
) -> MeanFlowFwhError {
    MeanFlowFwhError::InvalidSample {
        element_index,
        sample_index,
        field,
        requirement,
    }
}

fn solve_retarded_time(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: MeanFlowFwhOptions,
) -> Result<f64, MeanFlowFwhError> {
    let first = history.samples[0];
    let last = history.samples[history.samples.len() - 1];
    let earliest_arrival_s = arrival_time(element_index, first.time_s, first.position, options)?;
    let latest_arrival_s = arrival_time(element_index, last.time_s, last.position, options)?;
    if options.observer_time_s < earliest_arrival_s || options.observer_time_s > latest_arrival_s {
        return Err(MeanFlowFwhError::RetardedTimeOutsideHistory {
            element_index,
            earliest_arrival_s,
            latest_arrival_s,
            observer_time_s: options.observer_time_s,
        });
    }

    let mut lower = first.time_s;
    let mut upper = last.time_s;
    for _ in 0..options.maximum_retarded_iterations {
        let midpoint = 0.5 * (lower + upper);
        let bracket = bracket_index(history, midpoint);
        let position = interpolate_position(history, bracket, midpoint);
        let arrival_s = arrival_time(element_index, midpoint, position, options)?;
        let residual_s = arrival_s - options.observer_time_s;
        if residual_s.abs() <= options.retarded_time_tolerance_s {
            return Ok(midpoint);
        }
        if residual_s < 0.0 {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    let midpoint = 0.5 * (lower + upper);
    let bracket = bracket_index(history, midpoint);
    let position = interpolate_position(history, bracket, midpoint);
    let residual_s =
        arrival_time(element_index, midpoint, position, options)? - options.observer_time_s;
    Err(MeanFlowFwhError::RetardedTimeNoConvergence {
        element_index,
        residual_s: residual_s.abs(),
    })
}

fn arrival_time(
    element_index: usize,
    source_time_s: f64,
    source_position: Point3,
    options: MeanFlowFwhOptions,
) -> Result<f64, MeanFlowFwhError> {
    let euclidean_path = subtract(options.observer_position, source_position);
    let distance_m = norm(euclidean_path);
    if distance_m < options.minimum_observer_distance_m {
        return Err(MeanFlowFwhError::ObserverTooClose {
            element_index,
            distance_m,
            minimum_m: options.minimum_observer_distance_m,
        });
    }
    let mean_flow_speed_squared = dot(options.mean_flow_m_s, options.mean_flow_m_s);
    let sound_speed_squared = options.sound_speed_m_s * options.sound_speed_m_s;
    let denominator = sound_speed_squared - mean_flow_speed_squared;
    let linear_term = dot(euclidean_path, options.mean_flow_m_s);
    let discriminant = linear_term * linear_term + denominator * distance_m * distance_m;
    if !discriminant.is_finite() || discriminant <= 0.0 {
        return Err(MeanFlowFwhError::NonFiniteResult {
            field: "convected_travel_time",
        });
    }
    let propagation_time_s = (-linear_term + discriminant.sqrt()) / denominator;
    if !propagation_time_s.is_finite() || propagation_time_s <= 0.0 {
        return Err(MeanFlowFwhError::NonFiniteResult {
            field: "propagation_time_s",
        });
    }
    Ok(source_time_s + propagation_time_s)
}

fn interpolate_sample(
    history: &MovingFwhElementHistory,
    bracket: usize,
    time_s: f64,
) -> MovingFwhSample {
    let left = history.samples[bracket];
    let right = history.samples[bracket + 1];
    let fraction = (time_s - left.time_s) / (right.time_s - left.time_s);
    MovingFwhSample {
        time_s,
        position: lerp(left.position, right.position, fraction),
        normal: lerp(left.normal, right.normal, fraction),
        area_m2: left.area_m2 + fraction * (right.area_m2 - left.area_m2),
        pressure_pa: left.pressure_pa + fraction * (right.pressure_pa - left.pressure_pa),
        normal_velocity_m_s: left.normal_velocity_m_s
            + fraction * (right.normal_velocity_m_s - left.normal_velocity_m_s),
    }
}

fn interpolate_position(history: &MovingFwhElementHistory, bracket: usize, time_s: f64) -> Point3 {
    let left = history.samples[bracket];
    let right = history.samples[bracket + 1];
    let fraction = (time_s - left.time_s) / (right.time_s - left.time_s);
    lerp(left.position, right.position, fraction)
}

fn bracket_index(history: &MovingFwhElementHistory, time_s: f64) -> usize {
    history
        .samples
        .windows(2)
        .position(|samples| time_s <= samples[1].time_s)
        .unwrap_or(history.samples.len() - 2)
}

fn direction_at(
    observer: Point3,
    source: Point3,
    options: MeanFlowFwhOptions,
) -> Result<Point3, MeanFlowFwhError> {
    let propagation_time_s = arrival_time(0, 0.0, source, options)?;
    let path = subtract(
        subtract(observer, source),
        scale(options.mean_flow_m_s, propagation_time_s),
    );
    Ok(scale(path, 1.0 / norm(path)))
}

fn lerp(left: Point3, right: Point3, fraction: f64) -> Point3 {
    Point3 {
        x: left.x + fraction * (right.x - left.x),
        y: left.y + fraction * (right.y - left.y),
        z: left.z + fraction * (right.z - left.z),
    }
}

fn subtract(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn scale(point: Point3, factor: f64) -> Point3 {
    Point3 {
        x: point.x * factor,
        y: point.y * factor,
        z: point.z * factor,
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn norm(point: Point3) -> f64 {
    dot(point, point).sqrt()
}

fn normalize(point: Point3) -> Point3 {
    scale(point, 1.0 / norm(point))
}

trait FinitePoint {
    fn finite(self) -> bool;
    fn norm(self) -> f64;
}

impl FinitePoint for Point3 {
    fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    fn norm(self) -> f64 {
        norm(self)
    }
}
