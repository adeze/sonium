//! Bounded retarded-time moving-surface FW-H reference.
//!
//! The reference consumes caller-supplied piecewise-linear surface histories
//! and evaluates the reduced far-field thickness/loading terms in a quiescent,
//! homogeneous medium.  It solves the retarded-time equation for every
//! element, includes the subsonic Doppler factor, and differentiates the
//! prescribed source strengths on their local time segment.  This is a
//! moving-surface acoustic reference, not a mean-flow, CFD, or full Farassat
//! implementation.

use std::fmt;

use crate::Point3;

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// One time sample of a prescribed moving FW-H surface element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingFwhSample {
    /// Source time [s].
    pub time_s: f64,
    /// Element centroid [m].
    pub position: Point3,
    /// Outward element normal (normalized internally) [-].
    pub normal: Point3,
    /// Element area [m²].
    pub area_m2: f64,
    /// Surface pressure [Pa].
    pub pressure_pa: f64,
    /// Surface-normal velocity [m/s].
    pub normal_velocity_m_s: f64,
}

/// Ordered history for one moving FW-H surface element.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingFwhElementHistory {
    /// Strictly increasing source-time samples.
    pub samples: Vec<MovingFwhSample>,
}

/// Controls for the retarded-time moving-surface reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingFwhOptions {
    /// Requested observer time [s].
    pub observer_time_s: f64,
    /// Observer position [m].
    pub observer_position: Point3,
    /// Quiescent medium density [kg/m³].
    pub density_kg_m3: f64,
    /// Quiescent medium sound speed [m/s].
    pub sound_speed_m_s: f64,
    /// Minimum admitted retarded source-observer distance [m].
    pub minimum_observer_distance_m: f64,
    /// Maximum source Mach number admitted by the history [−].
    pub maximum_source_mach: f64,
    /// Minimum admitted `1 - M_r` Doppler factor [−].
    pub minimum_doppler_factor: f64,
    /// Absolute retarded-time root tolerance [s].
    pub retarded_time_tolerance_s: f64,
    /// Maximum bisection iterations for each retarded-time root.
    pub maximum_retarded_iterations: usize,
    /// Maximum number of surface elements.
    pub maximum_elements: usize,
    /// Maximum samples in one element history.
    pub maximum_samples_per_element: usize,
    /// Maximum samples across all histories.
    pub maximum_total_samples: usize,
}

impl Default for MovingFwhOptions {
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

/// One element's retarded-time contribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingFwhElementResult {
    /// Source time selected by the retarded-time equation [s].
    pub retarded_time_s: f64,
    /// Retarded source-observer distance [m].
    pub source_observer_distance_m: f64,
    /// `1 - M_r` at the retarded source state [−].
    pub doppler_factor: f64,
    /// Thickness/monopole pressure contribution [Pa].
    pub thickness_pressure_pa: f64,
    /// Loading/dipole pressure contribution [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of thickness and loading pressure [Pa].
    pub total_pressure_pa: f64,
}

/// Retarded-time moving-surface FW-H result.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingFwhResult {
    /// Sum of thickness terms [Pa].
    pub thickness_pressure_pa: f64,
    /// Sum of loading terms [Pa].
    pub loading_pressure_pa: f64,
    /// Sum of all element contributions [Pa].
    pub total_pressure_pa: f64,
    /// Per-element retarded-time diagnostics.
    pub elements: Vec<MovingFwhElementResult>,
    /// Minimum retarded source-observer distance [m].
    pub minimum_source_observer_distance_m: f64,
    /// Maximum source Mach number encountered [−].
    pub maximum_source_mach: f64,
}

impl MovingFwhResult {
    /// Returns whether all contributions and diagnostics are finite.
    pub fn is_finite(&self) -> bool {
        self.thickness_pressure_pa.is_finite()
            && self.loading_pressure_pa.is_finite()
            && self.total_pressure_pa.is_finite()
            && self.elements.iter().all(|element| {
                element.retarded_time_s.is_finite()
                    && element.source_observer_distance_m.is_finite()
                    && element.source_observer_distance_m > 0.0
                    && element.doppler_factor.is_finite()
                    && element.doppler_factor > 0.0
                    && element.thickness_pressure_pa.is_finite()
                    && element.loading_pressure_pa.is_finite()
                    && element.total_pressure_pa.is_finite()
            })
            && self.minimum_source_observer_distance_m.is_finite()
            && self.minimum_source_observer_distance_m > 0.0
            && self.maximum_source_mach.is_finite()
            && self.maximum_source_mach >= 0.0
    }
}

/// Failure-closed errors for the moving-surface reference.
#[derive(Debug, Clone, PartialEq)]
pub enum MovingFwhError {
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
    /// A source segment exceeds the admitted Mach budget.
    SourceMachLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Measured segment Mach number [−].
        source_mach: f64,
        /// Maximum admitted Mach number [−].
        maximum_mach: f64,
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
        /// Retarded distance [m].
        distance_m: f64,
        /// Minimum admitted distance [m].
        minimum_m: f64,
    },
    /// The retarded source is too close to sonic in the observer direction.
    DopplerLimit {
        /// Zero-based element index.
        element_index: usize,
        /// Measured `1 - M_r` [−].
        doppler_factor: f64,
        /// Minimum admitted factor [−].
        minimum_factor: f64,
    },
    /// A derived result became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for MovingFwhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid moving FW-H option {field}: {requirement}"
                )
            }
            Self::EmptySurface => formatter.write_str("moving FW-H surface is empty"),
            Self::TooManyElements { count, maximum } => {
                write!(
                    formatter,
                    "moving FW-H has {count} elements, maximum is {maximum}"
                )
            }
            Self::TooManySamples {
                element_index,
                count,
                maximum,
            } => write!(
                formatter,
                "moving FW-H element {element_index} has {count} samples, maximum is {maximum}"
            ),
            Self::TooManyTotalSamples { count, maximum } => write!(
                formatter,
                "moving FW-H has {count} total samples, maximum is {maximum}"
            ),
            Self::InsufficientSamples {
                element_index,
                minimum,
            } => write!(
                formatter,
                "moving FW-H element {element_index} needs at least {minimum} samples"
            ),
            Self::InvalidSample {
                element_index,
                sample_index,
                field,
                requirement,
            } => write!(
                formatter,
                "moving FW-H element {element_index} sample {sample_index} has invalid {field}: {requirement}"
            ),
            Self::NonIncreasingTime {
                element_index,
                previous_time_s,
                current_time_s,
            } => write!(
                formatter,
                "moving FW-H element {element_index} time is not increasing: {previous_time_s} then {current_time_s}"
            ),
            Self::SourceMachLimit {
                element_index,
                source_mach,
                maximum_mach,
            } => write!(
                formatter,
                "moving FW-H element {element_index} source Mach {source_mach} exceeds {maximum_mach}"
            ),
            Self::RetardedTimeOutsideHistory {
                element_index,
                earliest_arrival_s,
                latest_arrival_s,
                observer_time_s,
            } => write!(
                formatter,
                "moving FW-H element {element_index} observer time {observer_time_s} is outside arrival window [{earliest_arrival_s}, {latest_arrival_s}]"
            ),
            Self::RetardedTimeNoConvergence {
                element_index,
                residual_s,
            } => write!(
                formatter,
                "moving FW-H element {element_index} retarded-time residual {residual_s} did not converge"
            ),
            Self::ObserverTooClose {
                element_index,
                distance_m,
                minimum_m,
            } => write!(
                formatter,
                "moving FW-H element {element_index} distance {distance_m} is below {minimum_m}"
            ),
            Self::DopplerLimit {
                element_index,
                doppler_factor,
                minimum_factor,
            } => write!(
                formatter,
                "moving FW-H element {element_index} Doppler factor {doppler_factor} is below {minimum_factor}"
            ),
            Self::NonFiniteResult { field } => {
                write!(formatter, "non-finite moving FW-H result field {field}")
            }
        }
    }
}

impl std::error::Error for MovingFwhError {}

/// Evaluate reduced far-field thickness/loading terms at one observer time.
pub fn evaluate_moving_fwh_reference(
    histories: &[MovingFwhElementHistory],
    options: MovingFwhOptions,
) -> Result<MovingFwhResult, MovingFwhError> {
    validate_options(options)?;
    if histories.is_empty() {
        return Err(MovingFwhError::EmptySurface);
    }
    if histories.len() > options.maximum_elements {
        return Err(MovingFwhError::TooManyElements {
            count: histories.len(),
            maximum: options.maximum_elements,
        });
    }

    let mut total_samples = 0usize;
    for (element_index, history) in histories.iter().enumerate() {
        validate_history(element_index, history, options)?;
        total_samples = total_samples.checked_add(history.samples.len()).ok_or(
            MovingFwhError::TooManyTotalSamples {
                count: usize::MAX,
                maximum: options.maximum_total_samples,
            },
        )?;
    }
    if total_samples > options.maximum_total_samples {
        return Err(MovingFwhError::TooManyTotalSamples {
            count: total_samples,
            maximum: options.maximum_total_samples,
        });
    }

    let mut thickness_pressure_pa = 0.0;
    let mut loading_pressure_pa = 0.0;
    let mut elements = Vec::with_capacity(histories.len());
    let mut minimum_source_observer_distance_m = f64::INFINITY;
    let mut maximum_source_mach: f64 = 0.0;

    for (element_index, history) in histories.iter().enumerate() {
        let retarded_time_s = solve_retarded_time(element_index, history, options)?;
        let bracket = bracket_index(history, retarded_time_s);
        let source = interpolate_sample(history, bracket, retarded_time_s, element_index)?;
        let left = history.samples[bracket];
        let right = history.samples[bracket + 1];
        let delta_time_s = right.time_s - left.time_s;
        let source_velocity = scale(subtract(right.position, left.position), 1.0 / delta_time_s);
        let source_speed_m_s = norm(source_velocity);
        let source_mach = source_speed_m_s / options.sound_speed_m_s;

        let path = subtract(options.observer_position, source.position);
        let distance_m = norm(path);
        if !distance_m.is_finite() {
            return Err(MovingFwhError::NonFiniteResult {
                field: "source_observer_distance_m",
            });
        }
        if distance_m < options.minimum_observer_distance_m {
            return Err(MovingFwhError::ObserverTooClose {
                element_index,
                distance_m,
                minimum_m: options.minimum_observer_distance_m,
            });
        }
        let direction = scale(path, 1.0 / distance_m);
        let radial_speed_m_s = dot(source_velocity, direction);
        let doppler_factor = 1.0 - radial_speed_m_s / options.sound_speed_m_s;
        if doppler_factor < options.minimum_doppler_factor {
            return Err(MovingFwhError::DopplerLimit {
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
                direction_at(options.observer_position, left.position),
            );
        let right_loading = right.pressure_pa
            * right.area_m2
            * dot(
                normalize(right.normal),
                direction_at(options.observer_position, right.position),
            );
        let loading_source_dot = (right_loading - left_loading) / delta_time_s;

        // These are the retarded derivatives of the reduced source strengths
        // q/R and (F·r_hat)/R.  The 1/(1-M_r) factor is the source-time to
        // observer-time Jacobian; the 1/R² terms retain motion of the path.
        let thickness = (q_dot / (distance_m * doppler_factor)
            + q * radial_speed_m_s / (distance_m * distance_m * doppler_factor))
            / FOUR_PI;
        let loading = (loading_source_dot / (distance_m * doppler_factor)
            + loading_source * radial_speed_m_s / (distance_m * distance_m * doppler_factor))
            / (FOUR_PI * options.sound_speed_m_s);
        let total = thickness + loading;
        let contribution = MovingFwhElementResult {
            retarded_time_s,
            source_observer_distance_m: distance_m,
            doppler_factor,
            thickness_pressure_pa: thickness,
            loading_pressure_pa: loading,
            total_pressure_pa: total,
        };
        if !contribution.retarded_time_s.is_finite()
            || !contribution.source_observer_distance_m.is_finite()
            || !contribution.doppler_factor.is_finite()
            || !contribution.thickness_pressure_pa.is_finite()
            || !contribution.loading_pressure_pa.is_finite()
            || !contribution.total_pressure_pa.is_finite()
        {
            return Err(MovingFwhError::NonFiniteResult {
                field: "moving_fwh_element",
            });
        }
        thickness_pressure_pa += thickness;
        loading_pressure_pa += loading;
        elements.push(contribution);
        minimum_source_observer_distance_m = minimum_source_observer_distance_m.min(distance_m);
        maximum_source_mach = maximum_source_mach.max(source_mach);
    }

    let result = MovingFwhResult {
        thickness_pressure_pa,
        loading_pressure_pa,
        total_pressure_pa: thickness_pressure_pa + loading_pressure_pa,
        elements,
        minimum_source_observer_distance_m,
        maximum_source_mach,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(MovingFwhError::NonFiniteResult {
            field: "moving_fwh_result",
        })
    }
}

fn validate_options(options: MovingFwhOptions) -> Result<(), MovingFwhError> {
    if !options.observer_time_s.is_finite() {
        return Err(MovingFwhError::InvalidOptions {
            field: "observer_time_s",
            requirement: "finite",
        });
    }
    for (field, value, requirement) in [
        (
            "density_kg_m3",
            options.density_kg_m3,
            "finite and positive",
        ),
        (
            "sound_speed_m_s",
            options.sound_speed_m_s,
            "finite and positive",
        ),
        (
            "minimum_observer_distance_m",
            options.minimum_observer_distance_m,
            "finite and positive",
        ),
        (
            "maximum_source_mach",
            options.maximum_source_mach,
            "finite and positive and below one",
        ),
        (
            "minimum_doppler_factor",
            options.minimum_doppler_factor,
            "finite and positive and at most one",
        ),
        (
            "retarded_time_tolerance_s",
            options.retarded_time_tolerance_s,
            "finite and positive",
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MovingFwhError::InvalidOptions { field, requirement });
        }
    }
    if options.maximum_source_mach >= 1.0 {
        return Err(MovingFwhError::InvalidOptions {
            field: "maximum_source_mach",
            requirement: "below one",
        });
    }
    if options.minimum_doppler_factor > 1.0 {
        return Err(MovingFwhError::InvalidOptions {
            field: "minimum_doppler_factor",
            requirement: "at most one",
        });
    }
    if !point_is_finite(options.observer_position) {
        return Err(MovingFwhError::InvalidOptions {
            field: "observer_position",
            requirement: "finite",
        });
    }
    for (field, value, requirement) in [
        (
            "maximum_retarded_iterations",
            options.maximum_retarded_iterations,
            "positive",
        ),
        ("maximum_elements", options.maximum_elements, "positive"),
        (
            "maximum_samples_per_element",
            options.maximum_samples_per_element,
            "at least two",
        ),
        (
            "maximum_total_samples",
            options.maximum_total_samples,
            "at least two",
        ),
    ] {
        if value == 0 {
            return Err(MovingFwhError::InvalidOptions { field, requirement });
        }
    }
    if options.maximum_samples_per_element < 2 || options.maximum_total_samples < 2 {
        return Err(MovingFwhError::InvalidOptions {
            field: "maximum_samples_per_element",
            requirement: "at least two",
        });
    }
    Ok(())
}

fn validate_history(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: MovingFwhOptions,
) -> Result<(), MovingFwhError> {
    if history.samples.len() < 2 {
        return Err(MovingFwhError::InsufficientSamples {
            element_index,
            minimum: 2,
        });
    }
    if history.samples.len() > options.maximum_samples_per_element {
        return Err(MovingFwhError::TooManySamples {
            element_index,
            count: history.samples.len(),
            maximum: options.maximum_samples_per_element,
        });
    }
    for (sample_index, sample) in history.samples.iter().enumerate() {
        if !sample.time_s.is_finite() {
            return Err(MovingFwhError::InvalidSample {
                element_index,
                sample_index,
                field: "time_s",
                requirement: "finite",
            });
        }
        if !point_is_finite(sample.position) {
            return Err(MovingFwhError::InvalidSample {
                element_index,
                sample_index,
                field: "position",
                requirement: "finite",
            });
        }
        if !point_is_finite(sample.normal) || norm(sample.normal) <= 0.0 {
            return Err(MovingFwhError::InvalidSample {
                element_index,
                sample_index,
                field: "normal",
                requirement: "finite and non-zero",
            });
        }
        if !sample.area_m2.is_finite() || sample.area_m2 <= 0.0 {
            return Err(MovingFwhError::InvalidSample {
                element_index,
                sample_index,
                field: "area_m2",
                requirement: "finite and positive",
            });
        }
        for (field, value) in [
            ("pressure_pa", sample.pressure_pa),
            ("normal_velocity_m_s", sample.normal_velocity_m_s),
        ] {
            if !value.is_finite() {
                return Err(MovingFwhError::InvalidSample {
                    element_index,
                    sample_index,
                    field,
                    requirement: "finite",
                });
            }
        }
        if sample_index > 0 {
            let previous = history.samples[sample_index - 1].time_s;
            if sample.time_s <= previous {
                return Err(MovingFwhError::NonIncreasingTime {
                    element_index,
                    previous_time_s: previous,
                    current_time_s: sample.time_s,
                });
            }
            let delta_time_s = sample.time_s - previous;
            let source_speed_m_s = norm(scale(
                subtract(sample.position, history.samples[sample_index - 1].position),
                1.0 / delta_time_s,
            ));
            let source_mach = source_speed_m_s / options.sound_speed_m_s;
            if !source_mach.is_finite() || source_mach > options.maximum_source_mach {
                return Err(MovingFwhError::SourceMachLimit {
                    element_index,
                    source_mach,
                    maximum_mach: options.maximum_source_mach,
                });
            }
        }
    }
    Ok(())
}

fn solve_retarded_time(
    element_index: usize,
    history: &MovingFwhElementHistory,
    options: MovingFwhOptions,
) -> Result<f64, MovingFwhError> {
    let first = history.samples[0];
    let last = history.samples[history.samples.len() - 1];
    let earliest_arrival_s = first.time_s
        + norm(subtract(options.observer_position, first.position)) / options.sound_speed_m_s;
    let latest_arrival_s = last.time_s
        + norm(subtract(options.observer_position, last.position)) / options.sound_speed_m_s;
    if options.observer_time_s < earliest_arrival_s - options.retarded_time_tolerance_s
        || options.observer_time_s > latest_arrival_s + options.retarded_time_tolerance_s
    {
        return Err(MovingFwhError::RetardedTimeOutsideHistory {
            element_index,
            earliest_arrival_s,
            latest_arrival_s,
            observer_time_s: options.observer_time_s,
        });
    }

    let mut lower = first.time_s;
    let mut upper = last.time_s;
    let mut root = lower;
    let mut residual_s = f64::INFINITY;
    for _ in 0..options.maximum_retarded_iterations {
        root = 0.5 * (lower + upper);
        let distance_m = norm(subtract(
            options.observer_position,
            interpolate_position(history, root),
        ));
        let arrival_s = root + distance_m / options.sound_speed_m_s;
        residual_s = (arrival_s - options.observer_time_s).abs();
        if residual_s <= options.retarded_time_tolerance_s {
            return Ok(root);
        }
        if arrival_s < options.observer_time_s {
            lower = root;
        } else {
            upper = root;
        }
    }
    if residual_s <= options.retarded_time_tolerance_s {
        Ok(root)
    } else {
        Err(MovingFwhError::RetardedTimeNoConvergence {
            element_index,
            residual_s,
        })
    }
}

fn bracket_index(history: &MovingFwhElementHistory, time_s: f64) -> usize {
    if time_s <= history.samples[0].time_s {
        return 0;
    }
    for index in 0..history.samples.len() - 1 {
        if time_s <= history.samples[index + 1].time_s {
            return index;
        }
    }
    history.samples.len() - 2
}

fn interpolate_sample(
    history: &MovingFwhElementHistory,
    bracket: usize,
    time_s: f64,
    element_index: usize,
) -> Result<MovingFwhSample, MovingFwhError> {
    let left = history.samples[bracket];
    let right = history.samples[bracket + 1];
    let fraction = (time_s - left.time_s) / (right.time_s - left.time_s);
    let normal = lerp_point(left.normal, right.normal, fraction);
    if norm(normal) <= 0.0 || !point_is_finite(normal) {
        return Err(MovingFwhError::InvalidSample {
            element_index,
            sample_index: bracket,
            field: "interpolated_normal",
            requirement: "finite and non-zero",
        });
    }
    Ok(MovingFwhSample {
        time_s,
        position: lerp_point(left.position, right.position, fraction),
        normal,
        area_m2: lerp(left.area_m2, right.area_m2, fraction),
        pressure_pa: lerp(left.pressure_pa, right.pressure_pa, fraction),
        normal_velocity_m_s: lerp(
            left.normal_velocity_m_s,
            right.normal_velocity_m_s,
            fraction,
        ),
    })
}

fn interpolate_position(history: &MovingFwhElementHistory, time_s: f64) -> Point3 {
    let bracket = bracket_index(history, time_s);
    let left = history.samples[bracket];
    let right = history.samples[bracket + 1];
    let fraction = (time_s - left.time_s) / (right.time_s - left.time_s);
    lerp_point(left.position, right.position, fraction)
}

fn direction_at(observer: Point3, source: Point3) -> Point3 {
    normalize(subtract(observer, source))
}

fn normalize(point: Point3) -> Point3 {
    scale(point, 1.0 / norm(point))
}

fn point_is_finite(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn lerp(left: f64, right: f64, fraction: f64) -> f64 {
    left + fraction * (right - left)
}

fn lerp_point(left: Point3, right: Point3, fraction: f64) -> Point3 {
    Point3 {
        x: lerp(left.x, right.x, fraction),
        y: lerp(left.y, right.y, fraction),
        z: lerp(left.z, right.z, fraction),
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
    point.x.hypot(point.y).hypot(point.z)
}
