//! Bounded Farassat 1A Ffowcs Williams--Hawkings surface reference.
//!
//! The implementation evaluates the time-domain Farassat 1A thickness and
//! loading integrands for caller-supplied, piecewise-linear moving surface
//! histories in a quiescent homogeneous medium.  It keeps source-time
//! derivatives explicit, solves the retarded-time equation per element, and
//! fails closed on supersonic/Doppler, far-field, history, and work-budget
//! violations.
//!
//! This is a complete surface-integral reference under its stated assumptions;
//! it is not a CFD solver, a turbulence model, or a mean-flow correction.

use std::fmt;

use crate::{MovingFwhElementHistory, MovingFwhSample, Point3};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// Controls for the bounded Farassat 1A surface evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarassatFwhOptions {
    /// Requested observer time [s].
    pub observer_time_s: f64,
    /// Observer position [m].
    pub observer_position: Point3,
    /// Quiescent-medium density [kg/m³].
    pub density_kg_m3: f64,
    /// Quiescent-medium sound speed [m/s].
    pub sound_speed_m_s: f64,
    /// Minimum admitted retarded source-observer distance [m].
    pub minimum_observer_distance_m: f64,
    /// Maximum admitted source Mach number [-].
    pub maximum_source_mach: f64,
    /// Minimum admitted retarded-time Doppler factor `1 - M_r` [-].
    pub minimum_doppler_factor: f64,
    /// Retarded-time bisection tolerance [s].
    pub retarded_time_tolerance_s: f64,
    /// Maximum retarded-time bisection iterations per element.
    pub maximum_retarded_iterations: usize,
    /// Maximum number of surface elements.
    pub maximum_elements: usize,
    /// Maximum samples in one element history.
    pub maximum_samples_per_element: usize,
    /// Maximum samples across all histories.
    pub maximum_total_samples: usize,
}

impl Default for FarassatFwhOptions {
    fn default() -> Self {
        Self {
            observer_time_s: 1.0,
            observer_position: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            density_kg_m3: 1.2,
            sound_speed_m_s: 343.0,
            minimum_observer_distance_m: 0.1,
            maximum_source_mach: 0.5,
            minimum_doppler_factor: 0.25,
            retarded_time_tolerance_s: 1.0e-10,
            maximum_retarded_iterations: 64,
            maximum_elements: 16_384,
            maximum_samples_per_element: 4_096,
            maximum_total_samples: 1_000_000,
        }
    }
}

/// Retarded-time Farassat 1A contribution from one surface element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarassatFwhElementResult {
    /// Source time selected by the retarded-time equation [s].
    pub retarded_time_s: f64,
    /// Retarded source-observer distance [m].
    pub source_observer_distance_m: f64,
    /// Retarded source Mach number [-].
    pub source_mach: f64,
    /// Retarded-time Doppler factor `1 - M_r` [-].
    pub doppler_factor: f64,
    /// Thickness/monopole pressure contribution [Pa].
    pub thickness_pressure_pa: f64,
    /// Loading/dipole pressure contribution [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of thickness and loading contributions [Pa].
    pub total_pressure_pa: f64,
}

/// Farassat 1A surface-integral result.
#[derive(Debug, Clone, PartialEq)]
pub struct FarassatFwhResult {
    /// Sum of thickness terms [Pa].
    pub thickness_pressure_pa: f64,
    /// Sum of loading terms [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of all element contributions [Pa].
    pub total_pressure_pa: f64,
    /// Per-element retarded-time diagnostics.
    pub elements: Vec<FarassatFwhElementResult>,
    /// Minimum retarded source-observer distance [m].
    pub minimum_source_observer_distance_m: f64,
    /// Maximum source Mach number encountered [-].
    pub maximum_source_mach: f64,
}

impl FarassatFwhResult {
    /// Whether all result fields satisfy the finite, positive diagnostics.
    pub fn is_finite(&self) -> bool {
        self.thickness_pressure_pa.is_finite()
            && self.loading_pressure_pa.is_finite()
            && self.total_pressure_pa.is_finite()
            && self.minimum_source_observer_distance_m.is_finite()
            && self.minimum_source_observer_distance_m > 0.0
            && self.maximum_source_mach.is_finite()
            && self.maximum_source_mach >= 0.0
            && self.elements.iter().all(|element| {
                element.retarded_time_s.is_finite()
                    && element.source_observer_distance_m.is_finite()
                    && element.source_observer_distance_m > 0.0
                    && element.source_mach.is_finite()
                    && element.source_mach >= 0.0
                    && element.doppler_factor.is_finite()
                    && element.doppler_factor > 0.0
                    && element.thickness_pressure_pa.is_finite()
                    && element.loading_pressure_pa.is_finite()
                    && element.total_pressure_pa.is_finite()
            })
    }
}

/// Failure-closed Farassat 1A evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum FarassatFwhError {
    /// An option lies outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No surface histories were supplied.
    EmptySurface,
    /// Element count exceeds its configured budget.
    TooManyElements { count: usize, maximum: usize },
    /// One element history exceeds its configured budget.
    TooManySamples {
        /// Zero-based element index.
        element_index: usize,
        /// Supplied sample count.
        count: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// Total history sample count exceeds its configured budget.
    TooManyTotalSamples { count: usize, maximum: usize },
    /// A history cannot supply a source-time derivative.
    InsufficientSamples {
        /// Zero-based element index.
        element_index: usize,
        /// Minimum required sample count.
        minimum: usize,
    },
    /// A sample contains malformed data.
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
    /// Source times are not strictly increasing.
    NonIncreasingTime {
        /// Zero-based element index.
        element_index: usize,
        /// Previous source time [s].
        previous_time_s: f64,
        /// Current source time [s].
        current_time_s: f64,
    },
    /// A source segment exceeds the admitted Mach budget.
    SourceMachLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Segment Mach number [-].
        source_mach: f64,
        /// Configured maximum Mach number [-].
        maximum_mach: f64,
    },
    /// Observer time is outside the history arrival window.
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
    /// Retarded-time bisection did not meet its tolerance.
    RetardedTimeNoConvergence {
        /// Zero-based element index.
        element_index: usize,
        /// Final absolute arrival-time residual [s].
        residual_s: f64,
    },
    /// Retarded source-observer distance is too small for the far-field form.
    ObserverTooClose {
        /// Zero-based element index.
        element_index: usize,
        /// Retarded distance [m].
        distance_m: f64,
        /// Configured minimum distance [m].
        minimum_m: f64,
    },
    /// Retarded source is too close to sonic in the observer direction.
    DopplerLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Retarded `1 - M_r` factor [-].
        doppler_factor: f64,
        /// Configured minimum factor [-].
        minimum_factor: f64,
    },
    /// A derived result became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for FarassatFwhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "Farassat FW-H option {field} must be {requirement}"
            ),
            Self::EmptySurface => formatter.write_str("Farassat FW-H surface is empty"),
            Self::TooManyElements { count, maximum } => write!(
                formatter,
                "Farassat FW-H surface has {count} elements, maximum is {maximum}"
            ),
            Self::TooManySamples {
                element_index,
                count,
                maximum,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} has {count} samples, maximum is {maximum}"
            ),
            Self::TooManyTotalSamples { count, maximum } => write!(
                formatter,
                "Farassat FW-H history has {count} samples, maximum is {maximum}"
            ),
            Self::InsufficientSamples {
                element_index,
                minimum,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} needs at least {minimum} samples"
            ),
            Self::InvalidSample {
                element_index,
                sample_index,
                field,
                requirement,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} sample {sample_index} has invalid {field}: {requirement}"
            ),
            Self::NonIncreasingTime {
                element_index,
                previous_time_s,
                current_time_s,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} time is not increasing: {previous_time_s} then {current_time_s}"
            ),
            Self::SourceMachLimit {
                element_index,
                source_mach,
                maximum_mach,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} source Mach {source_mach} exceeds {maximum_mach}"
            ),
            Self::RetardedTimeOutsideHistory {
                element_index,
                earliest_arrival_s,
                latest_arrival_s,
                observer_time_s,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} observer time {observer_time_s} is outside arrival window [{earliest_arrival_s}, {latest_arrival_s}]"
            ),
            Self::RetardedTimeNoConvergence {
                element_index,
                residual_s,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} residual {residual_s} did not converge"
            ),
            Self::ObserverTooClose {
                element_index,
                distance_m,
                minimum_m,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} distance {distance_m} is below {minimum_m}"
            ),
            Self::DopplerLimit {
                element_index,
                doppler_factor,
                minimum_factor,
            } => write!(
                formatter,
                "Farassat FW-H element {element_index} Doppler factor {doppler_factor} is below {minimum_factor}"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite Farassat FW-H result field {field}")
            }
        }
    }
}

impl std::error::Error for FarassatFwhError {}

/// Evaluate the quiescent-medium Farassat 1A surface integral.
pub fn evaluate_farassat_1a_reference(
    histories: &[MovingFwhElementHistory],
    options: FarassatFwhOptions,
) -> Result<FarassatFwhResult, FarassatFwhError> {
    validate_options(options)?;
    if histories.is_empty() {
        return Err(FarassatFwhError::EmptySurface);
    }
    if histories.len() > options.maximum_elements {
        return Err(FarassatFwhError::TooManyElements {
            count: histories.len(),
            maximum: options.maximum_elements,
        });
    }

    let mut total_samples = 0usize;
    for (element_index, history) in histories.iter().enumerate() {
        validate_history(element_index, history, options)?;
        total_samples = total_samples.checked_add(history.samples.len()).ok_or(
            FarassatFwhError::TooManyTotalSamples {
                count: usize::MAX,
                maximum: options.maximum_total_samples,
            },
        )?;
        if total_samples > options.maximum_total_samples {
            return Err(FarassatFwhError::TooManyTotalSamples {
                count: total_samples,
                maximum: options.maximum_total_samples,
            });
        }
    }

    let mut thickness_pressure_pa = 0.0;
    let mut loading_pressure_pa = 0.0;
    let mut minimum_source_observer_distance_m = f64::INFINITY;
    let mut maximum_source_mach = 0.0_f64;
    let mut elements = Vec::with_capacity(histories.len());

    for (element_index, history) in histories.iter().enumerate() {
        let segment = locate_retarded_segment(element_index, history, options)?;
        let contribution = evaluate_element(element_index, history, segment, options)?;
        if !contribution.retarded_time_s.is_finite()
            || !contribution.source_observer_distance_m.is_finite()
            || !contribution.source_mach.is_finite()
            || !contribution.doppler_factor.is_finite()
            || !contribution.thickness_pressure_pa.is_finite()
            || !contribution.loading_pressure_pa.is_finite()
            || !contribution.total_pressure_pa.is_finite()
        {
            return Err(FarassatFwhError::NonFiniteResult {
                field: "element_contribution",
            });
        }
        thickness_pressure_pa += contribution.thickness_pressure_pa;
        loading_pressure_pa += contribution.loading_pressure_pa;
        minimum_source_observer_distance_m =
            minimum_source_observer_distance_m.min(contribution.source_observer_distance_m);
        maximum_source_mach = maximum_source_mach.max(contribution.source_mach);
        elements.push(contribution);
    }

    let total_pressure_pa = thickness_pressure_pa + loading_pressure_pa;
    let result = FarassatFwhResult {
        thickness_pressure_pa,
        loading_pressure_pa,
        total_pressure_pa,
        elements,
        minimum_source_observer_distance_m,
        maximum_source_mach,
    };
    if !result.is_finite() {
        return Err(FarassatFwhError::NonFiniteResult {
            field: "surface_integral",
        });
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct RetardedSegment {
    index: usize,
    alpha: f64,
    source_time_s: f64,
}

fn validate_options(options: FarassatFwhOptions) -> Result<(), FarassatFwhError> {
    if !options.observer_time_s.is_finite() {
        return Err(invalid_option("observer_time_s", "finite"));
    }
    if !point_is_finite(options.observer_position) {
        return Err(invalid_option("observer_position", "finite"));
    }
    for (field, value) in [
        ("density_kg_m3", options.density_kg_m3),
        ("sound_speed_m_s", options.sound_speed_m_s),
        (
            "minimum_observer_distance_m",
            options.minimum_observer_distance_m,
        ),
        (
            "retarded_time_tolerance_s",
            options.retarded_time_tolerance_s,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_option(field, "finite and positive"));
        }
    }
    if !options.maximum_source_mach.is_finite()
        || !(0.0..1.0).contains(&options.maximum_source_mach)
    {
        return Err(invalid_option(
            "maximum_source_mach",
            "finite and within [0, 1)",
        ));
    }
    if !options.minimum_doppler_factor.is_finite()
        || !(0.0..=1.0).contains(&options.minimum_doppler_factor)
        || options.minimum_doppler_factor == 0.0
    {
        return Err(invalid_option(
            "minimum_doppler_factor",
            "finite and within (0, 1]",
        ));
    }
    if options.maximum_retarded_iterations == 0 {
        return Err(invalid_option("maximum_retarded_iterations", "positive"));
    }
    if options.maximum_elements == 0 {
        return Err(invalid_option("maximum_elements", "positive"));
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

fn invalid_option(field: &'static str, requirement: &'static str) -> FarassatFwhError {
    FarassatFwhError::InvalidOptions { field, requirement }
}

fn validate_history(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: FarassatFwhOptions,
) -> Result<(), FarassatFwhError> {
    if history.samples.len() < 2 {
        return Err(FarassatFwhError::InsufficientSamples {
            element_index,
            minimum: 2,
        });
    }
    if history.samples.len() > options.maximum_samples_per_element {
        return Err(FarassatFwhError::TooManySamples {
            element_index,
            count: history.samples.len(),
            maximum: options.maximum_samples_per_element,
        });
    }
    for (sample_index, sample) in history.samples.iter().enumerate() {
        if !sample.time_s.is_finite() {
            return Err(invalid_sample(
                element_index,
                sample_index,
                "time_s",
                "finite",
            ));
        }
        if !point_is_finite(sample.position) {
            return Err(invalid_sample(
                element_index,
                sample_index,
                "position",
                "finite",
            ));
        }
        if !point_is_finite(sample.normal) || norm(sample.normal) <= f64::EPSILON {
            return Err(invalid_sample(
                element_index,
                sample_index,
                "normal",
                "finite and non-zero",
            ));
        }
        for (field, value) in [
            ("area_m2", sample.area_m2),
            ("pressure_pa", sample.pressure_pa),
            ("normal_velocity_m_s", sample.normal_velocity_m_s),
        ] {
            if !value.is_finite() || (field == "area_m2" && value <= 0.0) {
                return Err(invalid_sample(
                    element_index,
                    sample_index,
                    field,
                    if field == "area_m2" {
                        "finite and positive"
                    } else {
                        "finite"
                    },
                ));
            }
        }
        if let Some(previous) = history.samples.get(sample_index.wrapping_sub(1))
            && sample.time_s <= previous.time_s
        {
            return Err(FarassatFwhError::NonIncreasingTime {
                element_index,
                previous_time_s: previous.time_s,
                current_time_s: sample.time_s,
            });
        }
    }
    for (left, right) in history.samples.iter().zip(history.samples.iter().skip(1)) {
        let dt = right.time_s - left.time_s;
        let velocity = scale(subtract(right.position, left.position), 1.0 / dt);
        let source_mach = norm(velocity) / options.sound_speed_m_s;
        if !source_mach.is_finite() || source_mach > options.maximum_source_mach {
            return Err(FarassatFwhError::SourceMachLimit {
                element_index,
                source_mach,
                maximum_mach: options.maximum_source_mach,
            });
        }
    }
    Ok(())
}

fn invalid_sample(
    element_index: usize,
    sample_index: usize,
    field: &'static str,
    requirement: &'static str,
) -> FarassatFwhError {
    FarassatFwhError::InvalidSample {
        element_index,
        sample_index,
        field,
        requirement,
    }
}

fn locate_retarded_segment(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: FarassatFwhOptions,
) -> Result<RetardedSegment, FarassatFwhError> {
    let arrival = |index: usize, alpha: f64| {
        let left = history.samples[index];
        let right = history.samples[index + 1];
        let position = lerp_point(left.position, right.position, alpha);
        let source_time = left.time_s + alpha * (right.time_s - left.time_s);
        source_time + norm(subtract(options.observer_position, position)) / options.sound_speed_m_s
    };
    let earliest_arrival_s = history.samples[0].time_s
        + norm(subtract(
            options.observer_position,
            history.samples[0].position,
        )) / options.sound_speed_m_s;
    let last = history.samples.len() - 1;
    let latest_arrival_s = history.samples[last].time_s
        + norm(subtract(
            options.observer_position,
            history.samples[last].position,
        )) / options.sound_speed_m_s;
    if options.observer_time_s < earliest_arrival_s - options.retarded_time_tolerance_s
        || options.observer_time_s > latest_arrival_s + options.retarded_time_tolerance_s
    {
        return Err(FarassatFwhError::RetardedTimeOutsideHistory {
            element_index,
            earliest_arrival_s,
            latest_arrival_s,
            observer_time_s: options.observer_time_s,
        });
    }

    for index in 0..last {
        let left_arrival = arrival(index, 0.0);
        let right_arrival = arrival(index, 1.0);
        if options.observer_time_s < left_arrival - options.retarded_time_tolerance_s
            || options.observer_time_s > right_arrival + options.retarded_time_tolerance_s
        {
            continue;
        }
        let mut low = 0.0;
        let mut high = 1.0;
        let mut best_alpha = 0.0;
        let mut best_residual = f64::INFINITY;
        for _ in 0..options.maximum_retarded_iterations {
            let alpha = 0.5 * (low + high);
            let residual = arrival(index, alpha) - options.observer_time_s;
            if residual.abs() < best_residual {
                best_alpha = alpha;
                best_residual = residual.abs();
            }
            if residual.abs() <= options.retarded_time_tolerance_s {
                return Ok(RetardedSegment {
                    index,
                    alpha,
                    source_time_s: history.samples[index].time_s
                        + alpha
                            * (history.samples[index + 1].time_s - history.samples[index].time_s),
                });
            }
            if residual > 0.0 {
                high = alpha;
            } else {
                low = alpha;
            }
        }
        if best_residual <= options.retarded_time_tolerance_s {
            return Ok(RetardedSegment {
                index,
                alpha: best_alpha,
                source_time_s: history.samples[index].time_s
                    + best_alpha
                        * (history.samples[index + 1].time_s - history.samples[index].time_s),
            });
        }
        return Err(FarassatFwhError::RetardedTimeNoConvergence {
            element_index,
            residual_s: best_residual,
        });
    }

    Err(FarassatFwhError::RetardedTimeOutsideHistory {
        element_index,
        earliest_arrival_s,
        latest_arrival_s,
        observer_time_s: options.observer_time_s,
    })
}

fn evaluate_element(
    element_index: usize,
    history: &MovingFwhElementHistory,
    segment: RetardedSegment,
    options: FarassatFwhOptions,
) -> Result<FarassatFwhElementResult, FarassatFwhError> {
    let left = history.samples[segment.index];
    let right = history.samples[segment.index + 1];
    let dt = right.time_s - left.time_s;
    let source = interpolate_sample(left, right, segment.alpha);
    let velocity_m_s = scale(subtract(right.position, left.position), 1.0 / dt);
    let source_mach = norm(velocity_m_s) / options.sound_speed_m_s;
    let path = subtract(options.observer_position, source.position);
    let distance_m = norm(path);
    if !distance_m.is_finite() {
        return Err(FarassatFwhError::NonFiniteResult {
            field: "source_observer_distance_m",
        });
    }
    if distance_m < options.minimum_observer_distance_m {
        return Err(FarassatFwhError::ObserverTooClose {
            element_index,
            distance_m,
            minimum_m: options.minimum_observer_distance_m,
        });
    }

    let direction = scale(path, 1.0 / distance_m);
    let mach = scale(velocity_m_s, 1.0 / options.sound_speed_m_s);
    let m_r = dot(mach, direction);
    let m_squared = dot(mach, mach);
    let doppler_factor = 1.0 - m_r;
    if doppler_factor < options.minimum_doppler_factor {
        return Err(FarassatFwhError::DopplerLimit {
            element_index,
            doppler_factor,
            minimum_factor: options.minimum_doppler_factor,
        });
    }

    let direction_rate = scale(
        add(
            scale(direction, dot(velocity_m_s, direction)),
            scale(velocity_m_s, -1.0),
        ),
        1.0 / distance_m,
    );
    let radial_mach_rate = dot(mach, direction_rate);
    let left_normal = normalize(left.normal);
    let right_normal = normalize(right.normal);
    let left_load = scale(left_normal, left.pressure_pa * left.area_m2);
    let right_load = scale(right_normal, right.pressure_pa * right.area_m2);
    let load = lerp_point(left_load, right_load, segment.alpha);
    let load_rate = scale(subtract(right_load, left_load), 1.0 / dt);
    let q = options.density_kg_m3 * source.normal_velocity_m_s * source.area_m2;
    let q_rate = options.density_kg_m3
        * (right.normal_velocity_m_s * right.area_m2 - left.normal_velocity_m_s * left.area_m2)
        / dt;
    let l_r = dot(load, direction);
    let l_m = dot(load, mach);
    let l_r_rate = dot(load_rate, direction) + dot(load, direction_rate);
    let common_motion = distance_m * radial_mach_rate + options.sound_speed_m_s * (m_r - m_squared);
    let doppler_squared = doppler_factor * doppler_factor;
    let doppler_cubed = doppler_squared * doppler_factor;
    let thickness_integrand = q_rate / (distance_m * doppler_squared)
        + q * common_motion / (distance_m * distance_m * doppler_cubed);
    let loading_integrand = l_r_rate / (distance_m * doppler_squared)
        + (l_r - l_m) / (distance_m * distance_m * doppler_squared)
        + l_r * common_motion / (distance_m * distance_m * doppler_cubed);
    let thickness_pressure_pa = thickness_integrand / (FOUR_PI * options.sound_speed_m_s);
    let loading_pressure_pa = loading_integrand / (FOUR_PI * options.sound_speed_m_s);
    let total_pressure_pa = thickness_pressure_pa + loading_pressure_pa;
    if !thickness_pressure_pa.is_finite()
        || !loading_pressure_pa.is_finite()
        || !total_pressure_pa.is_finite()
    {
        return Err(FarassatFwhError::NonFiniteResult {
            field: "pressure_contribution",
        });
    }
    Ok(FarassatFwhElementResult {
        retarded_time_s: segment.source_time_s,
        source_observer_distance_m: distance_m,
        source_mach,
        doppler_factor,
        thickness_pressure_pa,
        loading_pressure_pa,
        total_pressure_pa,
    })
}

fn interpolate_sample(
    left: MovingFwhSample,
    right: MovingFwhSample,
    alpha: f64,
) -> MovingFwhSample {
    MovingFwhSample {
        time_s: left.time_s + alpha * (right.time_s - left.time_s),
        position: lerp_point(left.position, right.position, alpha),
        normal: lerp_point(left.normal, right.normal, alpha),
        area_m2: left.area_m2 + alpha * (right.area_m2 - left.area_m2),
        pressure_pa: left.pressure_pa + alpha * (right.pressure_pa - left.pressure_pa),
        normal_velocity_m_s: left.normal_velocity_m_s
            + alpha * (right.normal_velocity_m_s - left.normal_velocity_m_s),
    }
}

fn point_is_finite(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn add(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
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

fn lerp_point(left: Point3, right: Point3, alpha: f64) -> Point3 {
    add(left, scale(subtract(right, left), alpha))
}
