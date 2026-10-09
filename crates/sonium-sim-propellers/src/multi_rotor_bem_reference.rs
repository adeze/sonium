//! Bounded load-coupled multi-rotor BEM interaction reference.
//!
//! Each rotor owns a checked steady blade-element model. A finite-core
//! downstream wake estimate is averaged over each receiving disk, injected as
//! a station-uniform external axial inflow, and the BEM loads are re-evaluated
//! until the thrust/torque residual meets the caller's tolerance.
//!
//! This is a steady, one-way-per-iteration coupling with prescribed rotor
//! geometry. It is not an unsteady wake, viscous-core, dynamic-stall,
//! aeroelastic, body-fitted duct/shroud, CFD, or FW-H solver.

use std::fmt;

use crate::{
    BladeElementBemOptions, BladeElementBemResult, BladeElementError, BladeElementExternalInflow,
    BladeElementModel, BladeElementOperatingPoint,
};

const MIN_AXIS_NORM: f64 = 1.0e-12;

/// A rotor participating in the checked load-coupled interaction reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRotorBemRotor {
    /// Rotor centre in the common Cartesian frame [m].
    pub center_m: [f64; 3],
    /// Rotor thrust direction. The evaluator normalizes this vector.
    pub axis: [f64; 3],
    /// Checked steady blade-element model for this rotor.
    pub model: BladeElementModel,
    /// Positive-freestream operating point used by the BEM model.
    pub operating_point: BladeElementOperatingPoint,
}

/// Controls BEM coupling, disk quadrature, and finite-core wake spreading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRotorBemOptions {
    /// BEM station solver options applied to every rotor.
    pub bem_options: BladeElementBemOptions,
    /// Maximum outer wake/load coupling iterations.
    pub max_iterations: usize,
    /// Maximum relative thrust/torque update admitted as converged.
    pub convergence_tolerance: f64,
    /// Under-relaxation factor in `(0, 1]` for external wake velocity.
    pub relaxation: f64,
    /// Midpoint radial rings used to average a source wake over a receiving disk.
    pub radial_samples: usize,
    /// Azimuthal points per receiving-disk ring.
    pub azimuthal_samples: usize,
    /// Linear wake-radius growth per downstream source-rotor radius.
    pub wake_radius_growth: f64,
    /// Maximum rotor count admitted by this bounded reference.
    pub max_rotors: usize,
    /// Maximum ordered source/receiver disk evaluations per outer iteration.
    pub max_pair_evaluations: usize,
    /// Maximum total BEM evaluations, including the initial uncoupled solve.
    pub max_bem_evaluations: usize,
}

impl Default for MultiRotorBemOptions {
    fn default() -> Self {
        Self {
            bem_options: BladeElementBemOptions::default(),
            max_iterations: 24,
            convergence_tolerance: 1.0e-4,
            relaxation: 0.5,
            radial_samples: 3,
            azimuthal_samples: 16,
            wake_radius_growth: 0.25,
            max_rotors: 8,
            max_pair_evaluations: 64,
            max_bem_evaluations: 256,
        }
    }
}

/// Per-rotor result from the converged load-coupled reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRotorBemRotorResult {
    /// Rotor centre in the common frame [m].
    pub center_m: [f64; 3],
    /// Normalized rotor axis.
    pub axis: [f64; 3],
    /// Final checked BEM state.
    pub bem_result: BladeElementBemResult,
    /// External induced velocity projected onto this rotor axis [m/s].
    pub external_induced_velocity_m_s: f64,
    /// Isolated momentum-theory induced velocity for the final BEM thrust [m/s].
    pub isolated_induced_velocity_m_s: f64,
    /// Effective axial velocity used for the final coupling evidence [m/s].
    pub effective_axial_velocity_m_s: f64,
    /// Relative thrust/torque residual at the final outer iteration.
    pub relative_load_change: f64,
}

/// Converged evidence packet for load-coupled multi-rotor BEM interaction.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRotorBemResult {
    /// Per-rotor results in input order.
    pub rotors: Vec<MultiRotorBemRotorResult>,
    /// Signed induced velocity on each receiving rotor by each source rotor
    /// [receiver][source] [m/s]. Diagonal entries are zero.
    pub pairwise_induced_velocity_m_s: Vec<Vec<f64>>,
    /// Sum of final BEM thrusts [N].
    pub total_thrust_n: f64,
    /// Number of outer iterations used.
    pub iterations: usize,
    /// Whether the outer load residual met the requested tolerance.
    pub converged: bool,
    /// Maximum final relative thrust/torque change.
    pub maximum_relative_load_change: f64,
    /// Number of ordered source/receiver disk evaluations performed.
    pub pair_evaluation_count: usize,
    /// Number of BEM evaluations performed, including initial states.
    pub bem_evaluation_count: usize,
    /// Receiving-disk sample count per ordered pair evaluation.
    pub disk_sample_count_per_pair: usize,
}

impl MultiRotorBemResult {
    /// Returns whether the packet satisfies its finite/convergence contract.
    pub fn is_finite(&self) -> bool {
        let count = self.rotors.len();
        self.total_thrust_n.is_finite()
            && self.total_thrust_n > 0.0
            && self.iterations > 0
            && self.converged
            && self.maximum_relative_load_change.is_finite()
            && self.maximum_relative_load_change >= 0.0
            && (count < 2 || self.pair_evaluation_count > 0)
            && self.bem_evaluation_count >= count
            && self.disk_sample_count_per_pair > 0
            && self.pairwise_induced_velocity_m_s.len() == count
            && self
                .pairwise_induced_velocity_m_s
                .iter()
                .enumerate()
                .all(|(receiver, row)| {
                    row.len() == count
                        && row.iter().enumerate().all(|(source, value)| {
                            value.is_finite() && (receiver != source || *value == 0.0)
                        })
                })
            && self.rotors.iter().all(|rotor| {
                rotor.center_m.iter().all(|value| value.is_finite())
                    && rotor.axis.iter().all(|value| value.is_finite())
                    && rotor.bem_result.loads.thrust_n.is_finite()
                    && rotor.bem_result.loads.thrust_n > 0.0
                    && rotor.bem_result.loads.torque_nm.is_finite()
                    && rotor.bem_result.loads.torque_nm > 0.0
                    && rotor.external_induced_velocity_m_s.is_finite()
                    && rotor.isolated_induced_velocity_m_s.is_finite()
                    && rotor.isolated_induced_velocity_m_s > 0.0
                    && rotor.effective_axial_velocity_m_s.is_finite()
                    && rotor.relative_load_change.is_finite()
                    && rotor.relative_load_change >= 0.0
                    && rotor.bem_result.strips.iter().all(|strip| {
                        strip.axial_induction.is_finite()
                            && strip.tangential_induction.is_finite()
                            && strip.residual.is_finite()
                    })
            })
    }
}

/// Failure-closed errors for load-coupled multi-rotor BEM evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiRotorBemError {
    /// No rotors were supplied.
    EmptyRotorSet,
    /// An options field is outside its admitted domain.
    InvalidOptions {
        /// Options field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A rotor geometry or operating-point field is outside the interaction
    /// model's admitted domain.
    InvalidRotor {
        /// Input rotor index.
        rotor_index: usize,
        /// Rotor field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A rotor count exceeds the bounded reference budget.
    TooManyRotors {
        /// Supplied count.
        count: usize,
        /// Maximum admitted count.
        maximum: usize,
    },
    /// Ordered pair work exceeds the bounded reference budget.
    PairEvaluationBudgetExceeded {
        /// Required ordered pair count per iteration.
        required: usize,
        /// Maximum admitted ordered pair count per iteration.
        maximum: usize,
    },
    /// BEM evaluation work exceeds the bounded reference budget.
    BemEvaluationBudgetExceeded {
        /// Required total BEM evaluations.
        required: usize,
        /// Maximum admitted total evaluations.
        maximum: usize,
    },
    /// A nested BEM solve failed for one rotor.
    BemEvaluation {
        /// Input rotor index.
        rotor_index: usize,
        /// Nested error.
        error: BladeElementError,
    },
    /// The nested BEM result violated the positive-load interaction contract.
    InvalidBemResult {
        /// Input rotor index.
        rotor_index: usize,
        /// Result field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The final axial velocity was not physical for the forward-flight BEM
    /// contract.
    InvalidEffectiveAxialVelocity {
        /// Input rotor index.
        rotor_index: usize,
        /// Computed axial velocity [m/s].
        velocity_m_s: f64,
    },
    /// The outer fixed-point iteration did not meet the requested tolerance.
    NonConverged {
        /// Iterations performed.
        iterations: usize,
        /// Final relative load change.
        relative_load_change: f64,
        /// Requested tolerance.
        tolerance: f64,
    },
}

impl fmt::Display for MultiRotorBemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRotorSet => write!(formatter, "at least one rotor is required"),
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid multi-rotor BEM option {field}: {requirement}"
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
            Self::BemEvaluationBudgetExceeded { required, maximum } => write!(
                formatter,
                "BEM evaluation count {required} exceeds maximum {maximum}"
            ),
            Self::BemEvaluation { rotor_index, error } => {
                write!(
                    formatter,
                    "BEM evaluation failed for rotor {rotor_index}: {error}"
                )
            }
            Self::InvalidBemResult {
                rotor_index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid BEM result for rotor {rotor_index} field {field}: {requirement}"
            ),
            Self::InvalidEffectiveAxialVelocity {
                rotor_index,
                velocity_m_s,
            } => write!(
                formatter,
                "invalid effective axial velocity for rotor {rotor_index}: {velocity_m_s} m/s"
            ),
            Self::NonConverged {
                iterations,
                relative_load_change,
                tolerance,
            } => write!(
                formatter,
                "multi-rotor BEM did not converge in {iterations} iterations: residual {relative_load_change} > {tolerance}"
            ),
        }
    }
}

impl std::error::Error for MultiRotorBemError {}

/// Evaluate the steady load-coupled multi-rotor BEM reference.
pub fn evaluate_multi_rotor_bem(
    rotors: &[MultiRotorBemRotor],
    options: MultiRotorBemOptions,
) -> Result<MultiRotorBemResult, MultiRotorBemError> {
    validate_options(&options)?;
    if rotors.is_empty() {
        return Err(MultiRotorBemError::EmptyRotorSet);
    }
    if rotors.len() > options.max_rotors {
        return Err(MultiRotorBemError::TooManyRotors {
            count: rotors.len(),
            maximum: options.max_rotors,
        });
    }
    let pair_count = rotors
        .len()
        .checked_mul(rotors.len().saturating_sub(1))
        .ok_or(MultiRotorBemError::PairEvaluationBudgetExceeded {
            required: usize::MAX,
            maximum: options.max_pair_evaluations,
        })?;
    if pair_count > options.max_pair_evaluations {
        return Err(MultiRotorBemError::PairEvaluationBudgetExceeded {
            required: pair_count,
            maximum: options.max_pair_evaluations,
        });
    }

    let normalized_rotors = rotors
        .iter()
        .enumerate()
        .map(|(rotor_index, rotor)| normalize_rotor(rotor_index, rotor))
        .collect::<Result<Vec<_>, _>>()?;
    let disk_sample_count_per_pair = options
        .radial_samples
        .checked_mul(options.azimuthal_samples)
        .ok_or(MultiRotorBemError::InvalidOptions {
            field: "radial_samples",
            requirement: "radial_samples * azimuthal_samples must fit usize",
        })?;

    let initial_required = rotors.len();
    if initial_required > options.max_bem_evaluations {
        return Err(MultiRotorBemError::BemEvaluationBudgetExceeded {
            required: initial_required,
            maximum: options.max_bem_evaluations,
        });
    }
    let mut bem_evaluations: usize = 0;
    let mut states = Vec::with_capacity(rotors.len());
    for (rotor_index, rotor) in normalized_rotors.iter().enumerate() {
        states.push(
            rotor
                .model
                .evaluate_bem(rotor.operating_point, options.bem_options)
                .map_err(|error| MultiRotorBemError::BemEvaluation { rotor_index, error })?,
        );
        bem_evaluations += 1;
    }
    validate_states(&states)?;

    let mut external_velocity_m_s = vec![0.0; rotors.len()];
    let mut final_relative_load_change = f64::INFINITY;
    let mut iterations = 0;
    let mut pair_evaluation_count: usize = 0;

    for iteration in 1..=options.max_iterations {
        iterations = iteration;
        let pairwise = pairwise_induced_velocity(&normalized_rotors, &states, &options);
        pair_evaluation_count = pair_evaluation_count.checked_add(pair_count).ok_or(
            MultiRotorBemError::PairEvaluationBudgetExceeded {
                required: usize::MAX,
                maximum: options.max_pair_evaluations,
            },
        )?;
        let candidate_external = pairwise
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .collect::<Vec<_>>();
        let blended_external = external_velocity_m_s
            .iter()
            .zip(candidate_external.iter())
            .map(|(previous, candidate)| previous + options.relaxation * (candidate - previous))
            .collect::<Vec<_>>();

        let required = bem_evaluations.checked_add(rotors.len()).ok_or(
            MultiRotorBemError::BemEvaluationBudgetExceeded {
                required: usize::MAX,
                maximum: options.max_bem_evaluations,
            },
        )?;
        if required > options.max_bem_evaluations {
            return Err(MultiRotorBemError::BemEvaluationBudgetExceeded {
                required,
                maximum: options.max_bem_evaluations,
            });
        }
        let mut candidate_states = Vec::with_capacity(rotors.len());
        for (rotor_index, rotor) in normalized_rotors.iter().enumerate() {
            let external = blended_external[rotor_index];
            if !external.is_finite() {
                return Err(MultiRotorBemError::InvalidEffectiveAxialVelocity {
                    rotor_index,
                    velocity_m_s: external,
                });
            }
            let effective_axial = rotor.operating_point.freestream_m_s + external;
            if !effective_axial.is_finite() || effective_axial <= 0.0 {
                return Err(MultiRotorBemError::InvalidEffectiveAxialVelocity {
                    rotor_index,
                    velocity_m_s: effective_axial,
                });
            }
            let inflows = vec![
                BladeElementExternalInflow {
                    axial_velocity_m_s: external,
                    tangential_velocity_m_s: 0.0,
                };
                rotor.model.stations().len()
            ];
            candidate_states.push(
                rotor
                    .model
                    .evaluate_bem_with_external_inflow(
                        rotor.operating_point,
                        options.bem_options,
                        &inflows,
                    )
                    .map_err(|error| MultiRotorBemError::BemEvaluation { rotor_index, error })?,
            );
        }
        bem_evaluations = required;
        validate_states(&candidate_states)?;
        final_relative_load_change = states
            .iter()
            .zip(candidate_states.iter())
            .map(|(previous, candidate)| {
                relative_change(previous.loads.thrust_n, candidate.loads.thrust_n).max(
                    relative_change(previous.loads.torque_nm, candidate.loads.torque_nm),
                )
            })
            .fold(0.0, f64::max);
        states = candidate_states;
        external_velocity_m_s = blended_external;
        if final_relative_load_change <= options.convergence_tolerance {
            let final_pairwise = pairwise_induced_velocity(&normalized_rotors, &states, &options);
            let results = build_result(
                &normalized_rotors,
                &states,
                CouplingEvidence {
                    pairwise: &final_pairwise,
                    relative_load_change: final_relative_load_change,
                    iterations,
                    pair_evaluation_count,
                    bem_evaluation_count: bem_evaluations,
                    disk_sample_count_per_pair,
                },
            );
            debug_assert!(results.is_finite());
            return Ok(results);
        }
    }

    Err(MultiRotorBemError::NonConverged {
        iterations,
        relative_load_change: final_relative_load_change,
        tolerance: options.convergence_tolerance,
    })
}

#[derive(Debug, Clone)]
struct NormalizedRotor {
    center_m: [f64; 3],
    axis: [f64; 3],
    radius_m: f64,
    model: BladeElementModel,
    operating_point: BladeElementOperatingPoint,
}

fn validate_options(options: &MultiRotorBemOptions) -> Result<(), MultiRotorBemError> {
    if options.max_iterations == 0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "max_iterations",
            requirement: "> 0",
        });
    }
    if !options.convergence_tolerance.is_finite() || options.convergence_tolerance <= 0.0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "convergence_tolerance",
            requirement: "finite and > 0",
        });
    }
    if !options.relaxation.is_finite() || options.relaxation <= 0.0 || options.relaxation > 1.0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "relaxation",
            requirement: "finite and in (0, 1]",
        });
    }
    if options.radial_samples == 0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "radial_samples",
            requirement: "> 0",
        });
    }
    if options.azimuthal_samples < 4 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "azimuthal_samples",
            requirement: ">= 4",
        });
    }
    if !options.wake_radius_growth.is_finite() || options.wake_radius_growth < 0.0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "wake_radius_growth",
            requirement: "finite and >= 0",
        });
    }
    if options.max_rotors == 0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "max_rotors",
            requirement: "> 0",
        });
    }
    if options.max_pair_evaluations == 0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "max_pair_evaluations",
            requirement: "> 0",
        });
    }
    if options.max_bem_evaluations == 0 {
        return Err(MultiRotorBemError::InvalidOptions {
            field: "max_bem_evaluations",
            requirement: "> 0",
        });
    }
    Ok(())
}

fn normalize_rotor(
    rotor_index: usize,
    rotor: &MultiRotorBemRotor,
) -> Result<NormalizedRotor, MultiRotorBemError> {
    if rotor.center_m.iter().any(|value| !value.is_finite()) {
        return Err(MultiRotorBemError::InvalidRotor {
            rotor_index,
            field: "center_m",
            requirement: "all components finite",
        });
    }
    if rotor.axis.iter().any(|value| !value.is_finite()) {
        return Err(MultiRotorBemError::InvalidRotor {
            rotor_index,
            field: "axis",
            requirement: "all components finite",
        });
    }
    let axis_norm = norm(rotor.axis);
    if !axis_norm.is_finite() || axis_norm <= MIN_AXIS_NORM {
        return Err(MultiRotorBemError::InvalidRotor {
            rotor_index,
            field: "axis",
            requirement: "finite norm > 1e-12",
        });
    }
    if !rotor.operating_point.freestream_m_s.is_finite()
        || rotor.operating_point.freestream_m_s <= 0.0
    {
        return Err(MultiRotorBemError::InvalidRotor {
            rotor_index,
            field: "operating_point.freestream_m_s",
            requirement: "finite and > 0 for forward-flight BEM",
        });
    }
    let radius_m = rotor.model.tip_radius_m();
    if !radius_m.is_finite() || radius_m <= 0.0 {
        return Err(MultiRotorBemError::InvalidRotor {
            rotor_index,
            field: "model.tip_radius_m",
            requirement: "finite and > 0",
        });
    }
    Ok(NormalizedRotor {
        center_m: rotor.center_m,
        axis: scale(rotor.axis, 1.0 / axis_norm),
        radius_m,
        model: rotor.model.clone(),
        operating_point: rotor.operating_point,
    })
}

fn validate_states(states: &[BladeElementBemResult]) -> Result<(), MultiRotorBemError> {
    for (rotor_index, state) in states.iter().enumerate() {
        if !state.loads.thrust_n.is_finite() || state.loads.thrust_n <= 0.0 {
            return Err(MultiRotorBemError::InvalidBemResult {
                rotor_index,
                field: "loads.thrust_n",
                requirement: "finite and > 0",
            });
        }
        if !state.loads.torque_nm.is_finite() || state.loads.torque_nm <= 0.0 {
            return Err(MultiRotorBemError::InvalidBemResult {
                rotor_index,
                field: "loads.torque_nm",
                requirement: "finite and > 0",
            });
        }
        if !state.loads.shaft_power_w.is_finite() || state.loads.shaft_power_w <= 0.0 {
            return Err(MultiRotorBemError::InvalidBemResult {
                rotor_index,
                field: "loads.shaft_power_w",
                requirement: "finite and > 0",
            });
        }
        if state.strips.iter().any(|strip| {
            !strip.axial_induction.is_finite()
                || !strip.tangential_induction.is_finite()
                || !strip.residual.is_finite()
        }) {
            return Err(MultiRotorBemError::InvalidBemResult {
                rotor_index,
                field: "strips",
                requirement: "finite induction diagnostics",
            });
        }
    }
    Ok(())
}

struct CouplingEvidence<'a> {
    pairwise: &'a [Vec<f64>],
    relative_load_change: f64,
    iterations: usize,
    pair_evaluation_count: usize,
    bem_evaluation_count: usize,
    disk_sample_count_per_pair: usize,
}

fn build_result(
    rotors: &[NormalizedRotor],
    states: &[BladeElementBemResult],
    evidence: CouplingEvidence<'_>,
) -> MultiRotorBemResult {
    let rotor_results = rotors
        .iter()
        .enumerate()
        .map(|(index, rotor)| {
            let external_induced_velocity_m_s = evidence.pairwise[index].iter().sum::<f64>();
            let isolated_induced_velocity_m_s = momentum_induced_velocity(
                states[index].loads.thrust_n,
                rotor.radius_m,
                rotor.operating_point.density_kg_m3,
                rotor.operating_point.freestream_m_s,
            );
            MultiRotorBemRotorResult {
                center_m: rotor.center_m,
                axis: rotor.axis,
                bem_result: states[index].clone(),
                external_induced_velocity_m_s,
                isolated_induced_velocity_m_s,
                effective_axial_velocity_m_s: rotor.operating_point.freestream_m_s
                    + external_induced_velocity_m_s,
                relative_load_change: evidence.relative_load_change,
            }
        })
        .collect::<Vec<_>>();
    MultiRotorBemResult {
        total_thrust_n: rotor_results
            .iter()
            .map(|rotor| rotor.bem_result.loads.thrust_n)
            .sum(),
        rotors: rotor_results,
        pairwise_induced_velocity_m_s: evidence.pairwise.to_vec(),
        iterations: evidence.iterations,
        converged: true,
        maximum_relative_load_change: evidence.relative_load_change,
        pair_evaluation_count: evidence.pair_evaluation_count,
        bem_evaluation_count: evidence.bem_evaluation_count,
        disk_sample_count_per_pair: evidence.disk_sample_count_per_pair,
    }
}

fn pairwise_induced_velocity(
    rotors: &[NormalizedRotor],
    states: &[BladeElementBemResult],
    options: &MultiRotorBemOptions,
) -> Vec<Vec<f64>> {
    let mut pairwise = vec![vec![0.0; rotors.len()]; rotors.len()];
    for (receiver_index, receiver) in rotors.iter().enumerate() {
        for (source_index, source) in rotors.iter().enumerate() {
            if receiver_index != source_index {
                let source_induced_velocity_m_s = momentum_induced_velocity(
                    states[source_index].loads.thrust_n,
                    source.radius_m,
                    source.operating_point.density_kg_m3,
                    source.operating_point.freestream_m_s,
                );
                pairwise[receiver_index][source_index] =
                    average_wake_on_disk(receiver, source, source_induced_velocity_m_s, options);
            }
        }
    }
    pairwise
}

fn momentum_induced_velocity(
    thrust_n: f64,
    radius_m: f64,
    density_kg_m3: f64,
    freestream_m_s: f64,
) -> f64 {
    let disk_area_m2 = std::f64::consts::PI * radius_m * radius_m;
    let loading_m2_s2 = thrust_n / (density_kg_m3 * disk_area_m2);
    let root = freestream_m_s
        .mul_add(freestream_m_s, 2.0 * loading_m2_s2)
        .sqrt();
    let numerator = root - freestream_m_s;
    if numerator > 0.0 {
        0.5 * numerator
    } else {
        loading_m2_s2 / (root + freestream_m_s)
    }
}

fn average_wake_on_disk(
    receiver: &NormalizedRotor,
    source: &NormalizedRotor,
    source_induced_velocity_m_s: f64,
    options: &MultiRotorBemOptions,
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
            weighted_sum += weight
                * 2.0
                * source_induced_velocity_m_s
                * gaussian
                * dot(source.axis, receiver.axis);
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

fn relative_change(previous: f64, current: f64) -> f64 {
    (current - previous).abs() / previous.abs().max(current.abs()).max(1.0)
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
