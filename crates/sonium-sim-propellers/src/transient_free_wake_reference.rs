//! Bounded transient prescribed-circulation free-wake reference.
//!
//! This module advances a converged fixed-circulation wake with explicit Euler
//! steps. Each step recomputes regularized finite-segment Biot--Savart
//! induction for every wake node and convects nodes with the declared
//! background plus induced velocity. Topology and circulation remain
//! prescribed, so this is a transient kinematic reference rather than a
//! shedding/load-coupled, viscous, CFD, or FW-H solver.

use std::fmt;

use crate::{Point3, SteadyFreeWakeNode, SteadyFreeWakeResult};

const MIN_CORE_RADIUS_M: f64 = 1.0e-12;

/// Controls bounded transient prescribed-wake evolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientFreeWakeOptions {
    /// Explicit time step [s].
    pub time_step_s: f64,
    /// Number of transient steps.
    pub time_steps: usize,
    /// Declared uniform background convection velocity [m/s].
    pub background_velocity_m_s: Point3,
    /// Multiplier applied to recomputed induced velocity [-].
    pub induced_velocity_scale: f64,
    /// Linear finite-core growth rate [m/s].
    pub core_growth_m_s: f64,
    /// Maximum total nodes admitted.
    pub max_node_count: usize,
    /// Maximum node/segment Biot--Savart evaluations per step.
    pub max_biot_savart_evaluations_per_step: usize,
    /// Maximum node displacement in one explicit step [m].
    pub max_node_displacement_m: f64,
}

impl Default for TransientFreeWakeOptions {
    fn default() -> Self {
        Self {
            time_step_s: 1.0e-4,
            time_steps: 8,
            background_velocity_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            induced_velocity_scale: 1.0,
            core_growth_m_s: 0.0,
            max_node_count: 20_000,
            max_biot_savart_evaluations_per_step: 5_000_000,
            max_node_displacement_m: 5.0e-3,
        }
    }
}

/// Per-step transient wake diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientFreeWakeStep {
    /// One-based transient step index.
    pub step: usize,
    /// Elapsed time [s].
    pub time_s: f64,
    /// Maximum induced speed across wake nodes [m/s].
    pub maximum_induced_speed_m_s: f64,
    /// Maximum node displacement during this step [m].
    pub maximum_node_displacement_m: f64,
    /// Maximum node age after this step [s].
    pub maximum_age_s: f64,
    /// Biot--Savart evaluations performed during this step.
    pub biot_savart_evaluation_count: usize,
}

/// Final transient wake state and per-step evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientFreeWakeResult {
    /// Operating point inherited from the initial steady wake.
    pub operating_point: crate::BladeElementOperatingPoint,
    /// Blade multiplicity inherited from the initial steady wake.
    pub blade_count: usize,
    /// Final prescribed-circulation wake filaments.
    pub filaments: Vec<crate::SteadyFreeWakeFilament>,
    /// Per-step transient diagnostics.
    pub steps: Vec<TransientFreeWakeStep>,
    /// Final finite-core radius used by the induction kernel [m].
    pub final_core_radius_m: f64,
    /// Total Biot--Savart evaluations over all steps.
    pub total_biot_savart_evaluation_count: usize,
    /// Whether every requested step completed.
    pub completed: bool,
}

impl TransientFreeWakeResult {
    /// Returns whether the final state and diagnostics are finite and complete.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s > 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.blade_count > 0
            && !self.filaments.is_empty()
            && !self.steps.is_empty()
            && self.final_core_radius_m.is_finite()
            && self.final_core_radius_m > 0.0
            && self.completed
            && self.filaments.iter().all(|filament| {
                filament.blade_index < self.blade_count
                    && filament.base_azimuth_rad.is_finite()
                    && filament.radius_m.is_finite()
                    && filament.radius_m > 0.0
                    && filament.circulation_m2_s.is_finite()
                    && filament.axial_convection_speed_m_s.is_finite()
                    && filament.axial_convection_speed_m_s > 0.0
                    && !filament.nodes.is_empty()
                    && filament.nodes.iter().all(|node| {
                        finite_point(node.position)
                            && finite_point(node.induced_velocity_m_s)
                            && node.age_s.is_finite()
                            && node.age_s >= 0.0
                    })
            })
            && self.steps.iter().all(|step| {
                step.step > 0
                    && step.time_s.is_finite()
                    && step.time_s > 0.0
                    && step.maximum_induced_speed_m_s.is_finite()
                    && step.maximum_induced_speed_m_s >= 0.0
                    && step.maximum_node_displacement_m.is_finite()
                    && step.maximum_node_displacement_m >= 0.0
                    && step.maximum_age_s.is_finite()
                    && step.maximum_age_s >= 0.0
            })
    }
}

/// Failure-closed errors for transient wake evolution.
#[derive(Debug, Clone, PartialEq)]
pub enum TransientFreeWakeError {
    /// Options field outside its admitted domain.
    InvalidOptions {
        /// Options field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Initial steady wake is not a finite converged state.
    InvalidInitialWake {
        /// Initial wake field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Initial node count exceeds the state budget.
    NodeBudgetExceeded {
        /// Initial node count.
        count: usize,
        /// Maximum admitted nodes.
        maximum: usize,
    },
    /// Per-step Biot--Savart work exceeds the state budget.
    BiotSavartBudgetExceeded {
        /// One-based step index.
        step: usize,
        /// Required evaluations.
        required: usize,
        /// Maximum admitted evaluations.
        maximum: usize,
    },
    /// The node/segment evaluation count or accumulated count overflowed.
    EvaluationCountOverflow {
        /// One-based transient step index.
        step: usize,
    },
    /// Explicit Euler motion exceeds the declared stability screen.
    NodeDisplacementExceeded {
        /// One-based step index.
        step: usize,
        /// Node index in flattened filament order.
        node_index: usize,
        /// Computed displacement [m].
        displacement_m: f64,
        /// Maximum admitted displacement [m].
        maximum_m: f64,
    },
    /// A finite-segment induction or convection calculation produced a
    /// non-finite value.
    NonFiniteInduction {
        /// One-based step index.
        step: usize,
        /// Node index in flattened filament order.
        node_index: usize,
    },
}

impl fmt::Display for TransientFreeWakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid transient wake option {field}: {requirement}"
            ),
            Self::InvalidInitialWake { field, requirement } => write!(
                formatter,
                "invalid initial wake field {field}: {requirement}"
            ),
            Self::NodeBudgetExceeded { count, maximum } => {
                write!(
                    formatter,
                    "wake node count {count} exceeds maximum {maximum}"
                )
            }
            Self::BiotSavartBudgetExceeded {
                step,
                required,
                maximum,
            } => write!(
                formatter,
                "step {step} requires {required} Biot--Savart evaluations, maximum {maximum}"
            ),
            Self::EvaluationCountOverflow { step } => write!(
                formatter,
                "step {step} Biot--Savart evaluation count overflowed"
            ),
            Self::NodeDisplacementExceeded {
                step,
                node_index,
                displacement_m,
                maximum_m,
            } => write!(
                formatter,
                "step {step} node {node_index} displacement {displacement_m} m exceeds {maximum_m} m"
            ),
            Self::NonFiniteInduction { step, node_index } => write!(
                formatter,
                "step {step} node {node_index} produced non-finite induction or convection"
            ),
        }
    }
}

impl std::error::Error for TransientFreeWakeError {}

/// Advance a converged fixed-circulation wake through bounded transient steps.
pub fn advance_prescribed_free_wake(
    initial: &SteadyFreeWakeResult,
    options: TransientFreeWakeOptions,
) -> Result<TransientFreeWakeResult, TransientFreeWakeError> {
    validate_options(&options)?;

    if !initial.is_finite() {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "state",
            requirement: "finite and converged",
        });
    }
    if initial.filaments.is_empty() {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "filaments",
            requirement: "non-empty",
        });
    }

    let mut node_count = 0usize;
    let mut segment_count = 0usize;
    for filament in &initial.filaments {
        node_count = node_count.checked_add(filament.nodes.len()).ok_or(
            TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.nodes",
                requirement: "total node count fits usize",
            },
        )?;
        segment_count = segment_count
            .checked_add(filament.nodes.len().saturating_sub(1))
            .ok_or(TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.nodes",
                requirement: "total segment count fits usize",
            })?;

        if filament.blade_index >= initial.blade_count {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.blade_index",
                requirement: "less than blade_count",
            });
        }
        if filament.nodes.is_empty() {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.nodes",
                requirement: "each filament has at least one node",
            });
        }
    }
    if node_count == 0 {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "filaments.nodes",
            requirement: "at least one node",
        });
    }
    if node_count != initial.node_count {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "node_count",
            requirement: "matches the filament node total",
        });
    }
    if node_count > options.max_node_count {
        return Err(TransientFreeWakeError::NodeBudgetExceeded {
            count: node_count,
            maximum: options.max_node_count,
        });
    }

    let evaluations_per_step = node_count
        .checked_mul(segment_count)
        .ok_or(TransientFreeWakeError::EvaluationCountOverflow { step: 1 })?;
    if evaluations_per_step > options.max_biot_savart_evaluations_per_step {
        return Err(TransientFreeWakeError::BiotSavartBudgetExceeded {
            step: 1,
            required: evaluations_per_step,
            maximum: options.max_biot_savart_evaluations_per_step,
        });
    }
    if evaluations_per_step
        .checked_mul(options.time_steps)
        .is_none()
    {
        return Err(TransientFreeWakeError::EvaluationCountOverflow {
            step: options.time_steps,
        });
    }

    let mut filaments = initial.filaments.clone();
    let mut steps = Vec::new();
    let mut total_evaluations = 0usize;

    for step in 1..=options.time_steps {
        let time_s = step as f64 * options.time_step_s;
        let core_radius_m = initial.core_radius_m + options.core_growth_m_s * time_s;
        if !time_s.is_finite() || !core_radius_m.is_finite() || core_radius_m <= 0.0 {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "core_radius_m",
                requirement: "finite and positive throughout the requested duration",
            });
        }

        let induced = induced_velocities(&filaments, core_radius_m);
        let mut induced_index = 0usize;
        let mut maximum_induced_speed_m_s: f64 = 0.0;
        let mut maximum_node_displacement_m: f64 = 0.0;
        let mut maximum_age_s: f64 = 0.0;

        for filament in &mut filaments {
            for node in &mut filament.nodes {
                let induced_velocity = induced[induced_index];
                let velocity = add(
                    options.background_velocity_m_s,
                    scale(induced_velocity, options.induced_velocity_scale),
                );
                let displacement = scale(velocity, options.time_step_s);
                let displacement_m = norm(displacement);
                if !finite_point(induced_velocity) || !finite_point(velocity) {
                    return Err(TransientFreeWakeError::NonFiniteInduction {
                        step,
                        node_index: induced_index,
                    });
                }
                if !displacement_m.is_finite() {
                    return Err(TransientFreeWakeError::NonFiniteInduction {
                        step,
                        node_index: induced_index,
                    });
                }
                if displacement_m > options.max_node_displacement_m {
                    return Err(TransientFreeWakeError::NodeDisplacementExceeded {
                        step,
                        node_index: induced_index,
                        displacement_m,
                        maximum_m: options.max_node_displacement_m,
                    });
                }

                node.position = add(node.position, displacement);
                node.induced_velocity_m_s = induced_velocity;
                node.age_s += options.time_step_s;
                if !finite_point(node.position) || !node.age_s.is_finite() {
                    return Err(TransientFreeWakeError::NonFiniteInduction {
                        step,
                        node_index: induced_index,
                    });
                }

                maximum_induced_speed_m_s = maximum_induced_speed_m_s.max(norm(induced_velocity));
                maximum_node_displacement_m = maximum_node_displacement_m.max(displacement_m);
                maximum_age_s = maximum_age_s.max(node.age_s);
                induced_index += 1;
            }
        }

        total_evaluations = total_evaluations
            .checked_add(evaluations_per_step)
            .ok_or(TransientFreeWakeError::EvaluationCountOverflow { step })?;
        steps.push(TransientFreeWakeStep {
            step,
            time_s,
            maximum_induced_speed_m_s,
            maximum_node_displacement_m,
            maximum_age_s,
            biot_savart_evaluation_count: evaluations_per_step,
        });
    }

    let final_core_radius_m = initial.core_radius_m
        + options.core_growth_m_s * (options.time_steps as f64 * options.time_step_s);
    Ok(TransientFreeWakeResult {
        operating_point: initial.operating_point,
        blade_count: initial.blade_count,
        filaments,
        steps,
        final_core_radius_m,
        total_biot_savart_evaluation_count: total_evaluations,
        completed: true,
    })
}

/// Advance a prescribed-circulation wake while shedding one node per
/// filament at every explicit step.
///
/// Node zero in each filament is the anchored rotor-side shedding point and
/// remains fixed. Existing downstream nodes are convected, aged, and
/// regularized; a new age-zero node is inserted immediately downstream of
/// the anchor. Circulation and topology are still prescribed, so this is a
/// bounded shedding kinematics reference rather than a load-coupled,
/// viscous, CFD, or FW-H solver.
pub fn advance_prescribed_free_wake_with_shedding(
    initial: &SteadyFreeWakeResult,
    options: TransientFreeWakeOptions,
) -> Result<TransientFreeWakeResult, TransientFreeWakeError> {
    validate_options(&options)?;
    if !initial.is_finite() {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "state",
            requirement: "finite and converged",
        });
    }
    if initial.filaments.is_empty() {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "filaments",
            requirement: "non-empty",
        });
    }

    let mut node_count = 0usize;
    for filament in &initial.filaments {
        if filament.nodes.is_empty() {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.nodes",
                requirement: "each filament has at least one node",
            });
        }
        if filament.blade_index >= initial.blade_count {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.blade_index",
                requirement: "less than blade_count",
            });
        }
        node_count = node_count.checked_add(filament.nodes.len()).ok_or(
            TransientFreeWakeError::InvalidInitialWake {
                field: "filaments.nodes",
                requirement: "total node count fits usize",
            },
        )?;
    }
    if node_count == 0 {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "filaments.nodes",
            requirement: "at least one node",
        });
    }
    if node_count != initial.node_count {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "node_count",
            requirement: "matches the filament node total",
        });
    }
    if node_count > options.max_node_count {
        return Err(TransientFreeWakeError::NodeBudgetExceeded {
            count: node_count,
            maximum: options.max_node_count,
        });
    }

    let mut filaments = initial.filaments.clone();
    let mut steps = Vec::with_capacity(options.time_steps);
    let mut total_evaluations = 0usize;
    let mut final_core_radius_m = initial.core_radius_m;

    for step in 1..=options.time_steps {
        let time_s = step as f64 * options.time_step_s;
        let core_radius_m = initial.core_radius_m + options.core_growth_m_s * time_s;
        if !time_s.is_finite() || !core_radius_m.is_finite() || core_radius_m <= 0.0 {
            return Err(TransientFreeWakeError::InvalidInitialWake {
                field: "core_radius_m",
                requirement: "finite and positive throughout the requested duration",
            });
        }

        let segment_count = filaments.iter().try_fold(0usize, |count, filament| {
            count.checked_add(filament.nodes.len().saturating_sub(1))
        });
        let segment_count =
            segment_count.ok_or(TransientFreeWakeError::EvaluationCountOverflow { step })?;
        let evaluations = node_count
            .checked_mul(segment_count)
            .ok_or(TransientFreeWakeError::EvaluationCountOverflow { step })?;
        if evaluations > options.max_biot_savart_evaluations_per_step {
            return Err(TransientFreeWakeError::BiotSavartBudgetExceeded {
                step,
                required: evaluations,
                maximum: options.max_biot_savart_evaluations_per_step,
            });
        }
        let next_node_count = node_count.checked_add(filaments.len()).ok_or(
            TransientFreeWakeError::NodeBudgetExceeded {
                count: usize::MAX,
                maximum: options.max_node_count,
            },
        )?;
        if next_node_count > options.max_node_count {
            return Err(TransientFreeWakeError::NodeBudgetExceeded {
                count: next_node_count,
                maximum: options.max_node_count,
            });
        }

        let induced = induced_velocities(&filaments, core_radius_m);
        let anchors = filaments
            .iter()
            .map(|filament| filament.nodes[0].position)
            .collect::<Vec<_>>();
        let mut anchor_indices = Vec::with_capacity(filaments.len());
        let mut anchor_index = 0usize;
        for filament in &filaments {
            anchor_indices.push(anchor_index);
            anchor_index += filament.nodes.len();
        }
        let mut induced_index = 0usize;
        let mut maximum_induced_speed_m_s: f64 = 0.0;
        let mut maximum_node_displacement_m: f64 = 0.0;
        let mut maximum_age_s: f64 = 0.0;
        for (filament_index, filament) in filaments.iter_mut().enumerate() {
            for (node_index, node) in filament.nodes.iter_mut().enumerate() {
                let induced_velocity = induced[induced_index];
                if !finite_point(induced_velocity) {
                    return Err(TransientFreeWakeError::NonFiniteInduction {
                        step,
                        node_index: induced_index,
                    });
                }
                node.induced_velocity_m_s = induced_velocity;
                maximum_induced_speed_m_s = maximum_induced_speed_m_s.max(norm(induced_velocity));
                if node_index > 0 {
                    let velocity = add(
                        options.background_velocity_m_s,
                        scale(induced_velocity, options.induced_velocity_scale),
                    );
                    let displacement = scale(velocity, options.time_step_s);
                    let displacement_m = norm(displacement);
                    if !finite_point(velocity) || !displacement_m.is_finite() {
                        return Err(TransientFreeWakeError::NonFiniteInduction {
                            step,
                            node_index: induced_index,
                        });
                    }
                    if displacement_m > options.max_node_displacement_m {
                        return Err(TransientFreeWakeError::NodeDisplacementExceeded {
                            step,
                            node_index: induced_index,
                            displacement_m,
                            maximum_m: options.max_node_displacement_m,
                        });
                    }
                    node.position = add(node.position, displacement);
                    node.age_s += options.time_step_s;
                    if !finite_point(node.position) || !node.age_s.is_finite() {
                        return Err(TransientFreeWakeError::NonFiniteInduction {
                            step,
                            node_index: induced_index,
                        });
                    }
                    maximum_node_displacement_m = maximum_node_displacement_m.max(displacement_m);
                }
                maximum_age_s = maximum_age_s.max(node.age_s);
                induced_index += 1;
            }
            filament.nodes.insert(
                1,
                SteadyFreeWakeNode {
                    position: anchors[filament_index],
                    induced_velocity_m_s: induced[anchor_indices[filament_index]],
                    age_s: 0.0,
                },
            );
        }

        total_evaluations = total_evaluations
            .checked_add(evaluations)
            .ok_or(TransientFreeWakeError::EvaluationCountOverflow { step })?;
        steps.push(TransientFreeWakeStep {
            step,
            time_s,
            maximum_induced_speed_m_s,
            maximum_node_displacement_m,
            maximum_age_s,
            biot_savart_evaluation_count: evaluations,
        });
        node_count = next_node_count;
        final_core_radius_m = core_radius_m;
    }

    let result = TransientFreeWakeResult {
        operating_point: initial.operating_point,
        blade_count: initial.blade_count,
        filaments,
        steps,
        final_core_radius_m,
        total_biot_savart_evaluation_count: total_evaluations,
        completed: true,
    };
    if !result.is_finite() {
        return Err(TransientFreeWakeError::InvalidInitialWake {
            field: "result",
            requirement: "finite and complete",
        });
    }
    Ok(result)
}

fn validate_options(options: &TransientFreeWakeOptions) -> Result<(), TransientFreeWakeError> {
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "time_step_s",
            requirement: "finite and > 0",
        });
    }
    if options.time_steps == 0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "time_steps",
            requirement: "> 0",
        });
    }
    if !finite_point(options.background_velocity_m_s) {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "background_velocity_m_s",
            requirement: "all components finite",
        });
    }
    if !options.induced_velocity_scale.is_finite() {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "induced_velocity_scale",
            requirement: "finite",
        });
    }
    if !options.core_growth_m_s.is_finite() || options.core_growth_m_s < 0.0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "core_growth_m_s",
            requirement: "finite and >= 0",
        });
    }
    if options.max_node_count == 0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "max_node_count",
            requirement: "> 0",
        });
    }
    if options.max_biot_savart_evaluations_per_step == 0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "max_biot_savart_evaluations_per_step",
            requirement: "> 0",
        });
    }
    if !options.max_node_displacement_m.is_finite() || options.max_node_displacement_m <= 0.0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "max_node_displacement_m",
            requirement: "finite and > 0",
        });
    }

    let duration_s = options.time_steps as f64 * options.time_step_s;
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "time_steps",
            requirement: "time_steps * time_step_s is finite",
        });
    }
    if !options.core_growth_m_s.mul_add(duration_s, 0.0).is_finite() {
        return Err(TransientFreeWakeError::InvalidOptions {
            field: "core_growth_m_s",
            requirement: "growth over the requested duration is finite",
        });
    }
    Ok(())
}

fn induced_velocities(
    filaments: &[crate::SteadyFreeWakeFilament],
    core_radius_m: f64,
) -> Vec<Point3> {
    filaments
        .iter()
        .flat_map(|receiver_filament| {
            receiver_filament.nodes.iter().map(move |receiver| {
                filaments.iter().fold(zero(), |induced, source_filament| {
                    source_filament
                        .nodes
                        .windows(2)
                        .fold(induced, |sum, nodes| {
                            add(
                                sum,
                                finite_vortex_segment_velocity(
                                    receiver.position,
                                    nodes[0].position,
                                    nodes[1].position,
                                    source_filament.circulation_m2_s,
                                    core_radius_m,
                                ),
                            )
                        })
                })
            })
        })
        .collect()
}

fn finite_vortex_segment_velocity(
    receiver: Point3,
    start: Point3,
    end: Point3,
    circulation_m2_s: f64,
    core_radius_m: f64,
) -> Point3 {
    let r1 = sub(receiver, start);
    let r2 = sub(receiver, end);
    let segment = sub(end, start);
    let r1_norm = norm(r1);
    let r2_norm = norm(r2);
    let cross_product = cross(r1, r2);
    let denominator = dot(cross_product, cross_product) + core_radius_m * core_radius_m;
    if r1_norm <= MIN_CORE_RADIUS_M
        || r2_norm <= MIN_CORE_RADIUS_M
        || norm(segment) <= MIN_CORE_RADIUS_M
        || !denominator.is_finite()
        || denominator <= 0.0
    {
        return zero();
    }

    let direction_difference = sub(scale(r1, 1.0 / r1_norm), scale(r2, 1.0 / r2_norm));
    let axial_factor = dot(segment, direction_difference);
    scale(
        cross_product,
        circulation_m2_s * axial_factor / (4.0 * std::f64::consts::PI * denominator),
    )
}

fn finite_point(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn zero() -> Point3 {
    Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn add(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
    }
}

fn sub(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn cross(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn norm(vector: Point3) -> f64 {
    dot(vector, vector).sqrt()
}

fn scale(vector: Point3, factor: f64) -> Point3 {
    Point3 {
        x: vector.x * factor,
        y: vector.y * factor,
        z: vector.z * factor,
    }
}
