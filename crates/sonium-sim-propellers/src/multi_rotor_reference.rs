//! Bounded multi-rotor actuator-disk wake-interference reference.
//!
//! This module estimates one-way interaction between prescribed positive-thrust
//! rotors. Each rotor is represented by its momentum-theory induced velocity;
//! downstream wake velocity is spread with a finite-core Gaussian and averaged
//! over the receiving disk. The result is useful for screening geometry and
//! sign conventions while keeping pairwise evidence and evaluation budgets
//! visible.
//!
//! It is intentionally not a body-fitted ducted or shrouded-flow solver, a
//! load-coupled rotor model, a viscous wake solver, or an FW-H/CFD result.

use std::fmt;

const MIN_AXIS_NORM: f64 = 1.0e-12;

/// Prescribed rotor used by the bounded interaction screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRotorReferenceRotor {
    /// Rotor centre in a common Cartesian frame [m].
    pub center_m: [f64; 3],
    /// Thrust direction. The evaluator normalizes this vector.
    pub axis: [f64; 3],
    /// Rotor disk radius [m].
    pub radius_m: f64,
    /// Prescribed positive thrust [N].
    pub thrust_n: f64,
}

/// Controls finite disk quadrature and wake spreading for the interaction
/// screen.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRotorReferenceOptions {
    /// Air density [kg/m³].
    pub density_kg_m3: f64,
    /// Uniform freestream velocity in the common Cartesian frame [m/s].
    pub freestream_m_s: [f64; 3],
    /// Number of midpoint radial rings used for receiving-disk averaging.
    pub radial_samples: usize,
    /// Number of azimuthal samples used on every radial ring.
    pub azimuthal_samples: usize,
    /// Linear wake-radius growth per downstream rotor radius.
    pub wake_radius_growth: f64,
    /// Maximum number of rotors admitted by this bounded reference.
    pub max_rotors: usize,
    /// Maximum ordered source/receiver pair evaluations.
    pub max_pair_evaluations: usize,
}

impl Default for MultiRotorReferenceOptions {
    fn default() -> Self {
        Self {
            density_kg_m3: 1.225,
            freestream_m_s: [0.0, 0.0, 0.0],
            radial_samples: 3,
            azimuthal_samples: 16,
            wake_radius_growth: 0.25,
            max_rotors: 16,
            max_pair_evaluations: 256,
        }
    }
}

/// Per-rotor result from the one-way interaction screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRotorReferenceRotorResult {
    /// Rotor centre, copied from the admitted input [m].
    pub center_m: [f64; 3],
    /// Normalized rotor axis.
    pub axis: [f64; 3],
    /// Isolated momentum-theory induced velocity at the disk [m/s].
    pub isolated_induced_velocity_m_s: f64,
    /// Freestream component along this rotor axis [m/s].
    pub axial_freestream_m_s: f64,
    /// Signed induced velocity contributed by all other rotor wakes [m/s].
    pub external_induced_velocity_m_s: f64,
    /// Isolated plus external induced velocity [m/s].
    pub net_induced_velocity_m_s: f64,
    /// Axial freestream plus net induced velocity [m/s].
    pub effective_axial_velocity_m_s: f64,
    /// External induced velocity divided by isolated induced velocity.
    pub interference_ratio: f64,
}

/// Complete evidence packet for a bounded multi-rotor interaction evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRotorReferenceResult {
    /// Per-rotor summaries in input order.
    pub rotors: Vec<MultiRotorReferenceRotorResult>,
    /// Signed velocity induced on each receiving rotor by each source rotor
    /// [receiver][source] [m/s]. Diagonal entries are zero.
    pub pairwise_induced_velocity_m_s: Vec<Vec<f64>>,
    /// Sum of prescribed rotor thrusts [N].
    pub total_thrust_n: f64,
    /// Number of ordered source/receiver pairs evaluated.
    pub pair_evaluation_count: usize,
    /// Number of receiving-disk points evaluated for each ordered pair.
    pub disk_sample_count: usize,
}

impl MultiRotorReferenceResult {
    /// Returns whether all retained values satisfy the result contract.
    pub fn is_finite(&self) -> bool {
        let rotor_count = self.rotors.len();
        self.total_thrust_n.is_finite()
            && self.total_thrust_n > 0.0
            && self.pair_evaluation_count
                == rotor_count.saturating_mul(rotor_count.saturating_sub(1))
            && self.disk_sample_count > 0
            && self.pairwise_induced_velocity_m_s.len() == rotor_count
            && self
                .pairwise_induced_velocity_m_s
                .iter()
                .enumerate()
                .all(|(receiver, row)| {
                    row.len() == rotor_count
                        && row.iter().enumerate().all(|(source, value)| {
                            value.is_finite() && (receiver != source || *value == 0.0)
                        })
                })
            && self.rotors.iter().all(|rotor| {
                rotor.center_m.iter().all(|value| value.is_finite())
                    && rotor.axis.iter().all(|value| value.is_finite())
                    && rotor.isolated_induced_velocity_m_s.is_finite()
                    && rotor.isolated_induced_velocity_m_s > 0.0
                    && rotor.axial_freestream_m_s.is_finite()
                    && rotor.external_induced_velocity_m_s.is_finite()
                    && rotor.net_induced_velocity_m_s.is_finite()
                    && rotor.effective_axial_velocity_m_s.is_finite()
                    && rotor.interference_ratio.is_finite()
            })
    }
}

/// Failure-closed errors for the bounded multi-rotor reference.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiRotorReferenceError {
    /// No rotors were supplied.
    EmptyRotorSet,
    /// An options field is outside its admitted domain.
    InvalidOptions {
        /// Name of the invalid options field.
        field: &'static str,
        /// Required finite/positive domain.
        requirement: &'static str,
    },
    /// A rotor field is outside its admitted domain.
    InvalidRotor {
        /// Input rotor index.
        rotor_index: usize,
        /// Name of the invalid rotor field.
        field: &'static str,
        /// Required finite/positive domain.
        requirement: &'static str,
    },
    /// The caller supplied more rotors than the bounded screen permits.
    TooManyRotors {
        /// Supplied rotor count.
        count: usize,
        /// Admitted rotor count.
        maximum: usize,
    },
    /// The pairwise work budget would be exceeded.
    PairEvaluationBudgetExceeded {
        /// Required ordered pair count.
        required: usize,
        /// Admitted ordered pair count.
        maximum: usize,
    },
}

impl fmt::Display for MultiRotorReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRotorSet => write!(formatter, "at least one rotor is required"),
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid multi-rotor option {field}: {requirement}"
            ),
            Self::InvalidRotor {
                rotor_index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid rotor {rotor_index} field {field}: {requirement}"
            ),
            Self::TooManyRotors { count, maximum } => {
                write!(formatter, "rotor count {count} exceeds maximum {maximum}")
            }
            Self::PairEvaluationBudgetExceeded { required, maximum } => write!(
                formatter,
                "pair evaluation count {required} exceeds maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for MultiRotorReferenceError {}

/// Evaluate one-way finite-core wake interaction between prescribed rotors.
pub fn evaluate_multi_rotor_reference(
    rotors: &[MultiRotorReferenceRotor],
    options: MultiRotorReferenceOptions,
) -> Result<MultiRotorReferenceResult, MultiRotorReferenceError> {
    validate_options(&options)?;
    if rotors.is_empty() {
        return Err(MultiRotorReferenceError::EmptyRotorSet);
    }
    if rotors.len() > options.max_rotors {
        return Err(MultiRotorReferenceError::TooManyRotors {
            count: rotors.len(),
            maximum: options.max_rotors,
        });
    }
    let pair_count = rotors
        .len()
        .checked_mul(rotors.len().saturating_sub(1))
        .ok_or(MultiRotorReferenceError::PairEvaluationBudgetExceeded {
            required: usize::MAX,
            maximum: options.max_pair_evaluations,
        })?;
    if pair_count > options.max_pair_evaluations {
        return Err(MultiRotorReferenceError::PairEvaluationBudgetExceeded {
            required: pair_count,
            maximum: options.max_pair_evaluations,
        });
    }

    let normalized_rotors = rotors
        .iter()
        .enumerate()
        .map(|(rotor_index, rotor)| validate_and_normalize_rotor(rotor_index, *rotor))
        .collect::<Result<Vec<_>, _>>()?;
    let isolated_induced_velocities = normalized_rotors
        .iter()
        .map(|rotor| {
            momentum_induced_velocity(
                rotor.thrust_n,
                rotor.radius_m,
                options.density_kg_m3,
                dot(options.freestream_m_s, rotor.axis),
            )
        })
        .collect::<Vec<_>>();

    let disk_sample_count = options
        .radial_samples
        .checked_mul(options.azimuthal_samples)
        .ok_or(MultiRotorReferenceError::InvalidOptions {
            field: "radial_samples",
            requirement: "radial_samples * azimuthal_samples must fit usize",
        })?;
    let mut pairwise = vec![vec![0.0; normalized_rotors.len()]; normalized_rotors.len()];
    for (receiver_index, receiver) in normalized_rotors.iter().enumerate() {
        for (source_index, source) in normalized_rotors.iter().enumerate() {
            if receiver_index != source_index {
                pairwise[receiver_index][source_index] = average_wake_on_disk(
                    receiver,
                    source,
                    isolated_induced_velocities[source_index],
                    &options,
                );
            }
        }
    }

    let rotor_results = normalized_rotors
        .iter()
        .enumerate()
        .map(|(receiver_index, rotor)| {
            let axial_freestream_m_s = dot(options.freestream_m_s, rotor.axis);
            let external_induced_velocity_m_s = pairwise[receiver_index].iter().sum::<f64>();
            let isolated_induced_velocity_m_s = isolated_induced_velocities[receiver_index];
            let net_induced_velocity_m_s =
                isolated_induced_velocity_m_s + external_induced_velocity_m_s;
            MultiRotorReferenceRotorResult {
                center_m: rotor.center_m,
                axis: rotor.axis,
                isolated_induced_velocity_m_s,
                axial_freestream_m_s,
                external_induced_velocity_m_s,
                net_induced_velocity_m_s,
                effective_axial_velocity_m_s: axial_freestream_m_s + net_induced_velocity_m_s,
                interference_ratio: external_induced_velocity_m_s / isolated_induced_velocity_m_s,
            }
        })
        .collect();

    let result = MultiRotorReferenceResult {
        rotors: rotor_results,
        pairwise_induced_velocity_m_s: pairwise,
        total_thrust_n: normalized_rotors.iter().map(|rotor| rotor.thrust_n).sum(),
        pair_evaluation_count: pair_count,
        disk_sample_count,
    };
    debug_assert!(result.is_finite());
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct NormalizedRotor {
    center_m: [f64; 3],
    axis: [f64; 3],
    radius_m: f64,
    thrust_n: f64,
}

fn validate_options(options: &MultiRotorReferenceOptions) -> Result<(), MultiRotorReferenceError> {
    if !options.density_kg_m3.is_finite() || options.density_kg_m3 <= 0.0 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "density_kg_m3",
            requirement: "finite and > 0",
        });
    }
    if options
        .freestream_m_s
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "freestream_m_s",
            requirement: "all components finite",
        });
    }
    if options.radial_samples == 0 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "radial_samples",
            requirement: "> 0",
        });
    }
    if options.azimuthal_samples < 4 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "azimuthal_samples",
            requirement: ">= 4",
        });
    }
    if !options.wake_radius_growth.is_finite() || options.wake_radius_growth < 0.0 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "wake_radius_growth",
            requirement: "finite and >= 0",
        });
    }
    if options.max_rotors == 0 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "max_rotors",
            requirement: "> 0",
        });
    }
    if options.max_pair_evaluations == 0 {
        return Err(MultiRotorReferenceError::InvalidOptions {
            field: "max_pair_evaluations",
            requirement: "> 0",
        });
    }
    Ok(())
}

fn validate_and_normalize_rotor(
    rotor_index: usize,
    rotor: MultiRotorReferenceRotor,
) -> Result<NormalizedRotor, MultiRotorReferenceError> {
    if rotor.center_m.iter().any(|value| !value.is_finite()) {
        return Err(MultiRotorReferenceError::InvalidRotor {
            rotor_index,
            field: "center_m",
            requirement: "all components finite",
        });
    }
    if rotor.axis.iter().any(|value| !value.is_finite()) {
        return Err(MultiRotorReferenceError::InvalidRotor {
            rotor_index,
            field: "axis",
            requirement: "all components finite",
        });
    }
    let axis_norm = norm(rotor.axis);
    if !axis_norm.is_finite() || axis_norm <= MIN_AXIS_NORM {
        return Err(MultiRotorReferenceError::InvalidRotor {
            rotor_index,
            field: "axis",
            requirement: "finite norm > 1e-12",
        });
    }
    if !rotor.radius_m.is_finite() || rotor.radius_m <= 0.0 {
        return Err(MultiRotorReferenceError::InvalidRotor {
            rotor_index,
            field: "radius_m",
            requirement: "finite and > 0",
        });
    }
    if !rotor.thrust_n.is_finite() || rotor.thrust_n <= 0.0 {
        return Err(MultiRotorReferenceError::InvalidRotor {
            rotor_index,
            field: "thrust_n",
            requirement: "finite and > 0",
        });
    }
    Ok(NormalizedRotor {
        center_m: rotor.center_m,
        axis: scale(rotor.axis, 1.0 / axis_norm),
        radius_m: rotor.radius_m,
        thrust_n: rotor.thrust_n,
    })
}

fn momentum_induced_velocity(
    thrust_n: f64,
    radius_m: f64,
    density_kg_m3: f64,
    axial_freestream_m_s: f64,
) -> f64 {
    let disk_area_m2 = std::f64::consts::PI * radius_m * radius_m;
    let loading_m2_s2 = thrust_n / (density_kg_m3 * disk_area_m2);
    let discriminant = axial_freestream_m_s.mul_add(axial_freestream_m_s, 2.0 * loading_m2_s2);
    let root = discriminant.sqrt();
    let numerator = root - axial_freestream_m_s;
    if numerator > 0.0 {
        0.5 * numerator
    } else {
        loading_m2_s2 / (root + axial_freestream_m_s)
    }
}

fn average_wake_on_disk(
    receiver: &NormalizedRotor,
    source: &NormalizedRotor,
    source_induced_velocity_m_s: f64,
    options: &MultiRotorReferenceOptions,
) -> f64 {
    let (tangent, bitangent) = orthonormal_basis(receiver.axis);
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for radial_index in 0..options.radial_samples {
        let radial_fraction = (radial_index as f64 + 0.5) / options.radial_samples as f64;
        let weight = 2.0 * radial_fraction / options.radial_samples as f64;
        for azimuth_index in 0..options.azimuthal_samples {
            let theta = 2.0 * std::f64::consts::PI * azimuth_index as f64
                / options.azimuthal_samples as f64;
            let radial_offset = scale(
                add(scale(tangent, theta.cos()), scale(bitangent, theta.sin())),
                receiver.radius_m * radial_fraction,
            );
            let observer = add(receiver.center_m, radial_offset);
            let source_to_observer = sub(observer, source.center_m);
            let downstream_distance_m = dot(source_to_observer, source.axis);
            if downstream_distance_m <= 0.0 {
                weight_sum += weight;
                continue;
            }
            let radial_component = sub(
                source_to_observer,
                scale(source.axis, downstream_distance_m),
            );
            let wake_radius_m = source.radius_m
                * (1.0 + options.wake_radius_growth * downstream_distance_m / source.radius_m);
            let gaussian =
                (-dot(radial_component, radial_component) / (wake_radius_m * wake_radius_m)).exp();
            let induced_velocity_m_s = 2.0 * source_induced_velocity_m_s * gaussian;
            weighted_sum += weight * induced_velocity_m_s * dot(source.axis, receiver.axis);
            weight_sum += weight;
        }
    }
    weighted_sum / weight_sum
}

fn orthonormal_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let reference = if axis[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = normalize(cross(axis, reference));
    (tangent, cross(axis, tangent))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    scale(vector, 1.0 / norm(vector))
}
