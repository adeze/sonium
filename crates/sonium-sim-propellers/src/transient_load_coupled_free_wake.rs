//! Bounded quasi-unsteady load-coupled BEM/free-wake reference.
//!
//! This module deliberately implements a bounded lagged coupling reference:
//! each step samples the current prescribed wake, evaluates checked BEM,
//! updates filament circulation, and explicitly convects the wake. It does
//! not claim dynamic stall, viscous wake physics, aeroelastic deformation,
//! body-fitted duct flow, CFD, or FW-H acoustics.

use std::fmt;

use crate::{
    BladeElementBemOptions, BladeElementBemResult, BladeElementError, BladeElementExternalInflow,
    BladeElementModel, BladeElementOperatingPoint, Point3, SteadyFreeWakeError,
    SteadyFreeWakeOptions, SteadyFreeWakeResult, TransientFreeWakeError, TransientFreeWakeOptions,
    advance_prescribed_free_wake_with_shedding, evaluate_steady_free_wake,
};

/// Controls bounded lagged load-coupled transient wake evolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientLoadCoupledFreeWakeOptions {
    /// Explicit time step [s].
    pub time_step_s: f64,
    /// Number of load-coupled transient steps.
    pub time_steps: usize,
    /// Uniform background convection velocity [m/s].
    pub background_velocity_m_s: Point3,
    /// Multiplier applied to wake-induced convection [-].
    pub induced_velocity_scale: f64,
    /// Linear finite-core growth rate [m/s].
    pub core_growth_m_s: f64,
    /// Under-relaxation for sampled station inflow in the interval (0, 1].
    pub inflow_relaxation: f64,
    /// Maximum total BEM evaluations, including the initial solve.
    pub max_bem_evaluations: usize,
    /// Maximum total transient wake Biot--Savart evaluations.
    pub max_total_wake_evaluations: usize,
    /// Maximum node displacement in one explicit step [m].
    pub max_node_displacement_m: f64,
    /// Maximum node/segment Biot--Savart evaluations per transient step.
    pub max_biot_savart_evaluations_per_step: usize,
    /// Bounded fixed-circulation wake construction options.
    pub wake_options: SteadyFreeWakeOptions,
}

impl Default for TransientLoadCoupledFreeWakeOptions {
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
            inflow_relaxation: 0.4,
            max_bem_evaluations: 64,
            max_total_wake_evaluations: 20_000_000,
            max_node_displacement_m: 5.0e-3,
            max_biot_savart_evaluations_per_step: 5_000_000,
            wake_options: SteadyFreeWakeOptions::default(),
        }
    }
}

/// Evidence from one lagged load-coupled transient step.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientLoadCoupledFreeWakeStep {
    /// One-based transient step index.
    pub step: usize,
    /// Elapsed time [s].
    pub time_s: f64,
    /// BEM result evaluated with this step's relaxed station inflow.
    pub bem: BladeElementBemResult,
    /// Relaxed external inflow supplied to the BEM solve.
    pub station_external_inflow: Vec<BladeElementExternalInflow>,
    /// Maximum station inflow update from the previous step [m/s].
    pub maximum_inflow_update_m_s: f64,
    /// Maximum filament circulation update [m²/s].
    pub maximum_circulation_update_m2_s: f64,
    /// Maximum node displacement during this step [m].
    pub maximum_node_displacement_m: f64,
    /// Maximum node age after this step [s].
    pub maximum_age_s: f64,
    /// Biot--Savart evaluations performed during this step.
    pub biot_savart_evaluation_count: usize,
}

/// Final state and evidence from bounded load-coupled transient evolution.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientLoadCoupledFreeWakeResult {
    /// Operating point inherited from the BEM evaluations.
    pub operating_point: BladeElementOperatingPoint,
    /// Number of blades represented by the wake.
    pub blade_count: usize,
    /// Final load-updated wake filaments.
    pub filaments: Vec<crate::SteadyFreeWakeFilament>,
    /// Per-step BEM, inflow, motion, and work diagnostics.
    pub steps: Vec<TransientLoadCoupledFreeWakeStep>,
    /// Total BEM evaluations, including the initial zero-inflow solve.
    pub total_bem_evaluation_count: usize,
    /// Total transient and initial wake Biot--Savart evaluations.
    pub total_wake_evaluation_count: usize,
    /// Whether every requested step completed.
    pub completed: bool,
}

impl TransientLoadCoupledFreeWakeResult {
    /// Returns whether the result is finite, complete, and internally valid.
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
            && self.total_bem_evaluation_count > 0
            && self.completed
            && finite_filaments(&self.filaments, self.blade_count)
            && self.steps.iter().all(|step| {
                step.step > 0
                    && step.time_s.is_finite()
                    && step.time_s > 0.0
                    && finite_bem(&step.bem)
                    && finite_inflow(&step.station_external_inflow)
                    && step.maximum_inflow_update_m_s.is_finite()
                    && step.maximum_inflow_update_m_s >= 0.0
                    && step.maximum_circulation_update_m2_s.is_finite()
                    && step.maximum_circulation_update_m2_s >= 0.0
                    && step.maximum_node_displacement_m.is_finite()
                    && step.maximum_node_displacement_m >= 0.0
                    && step.maximum_age_s.is_finite()
                    && step.maximum_age_s >= 0.0
            })
    }
}

/// Failure-closed errors from the bounded transient coupling.
#[derive(Debug, Clone, PartialEq)]
pub enum TransientLoadCoupledFreeWakeError {
    /// An option is outside its admitted domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Initial zero-inflow BEM evaluation failed.
    InitialBem {
        /// Underlying checked BEM error.
        error: Box<BladeElementError>,
    },
    /// A transient-step BEM evaluation failed.
    StepBem {
        /// One-based step index.
        step: usize,
        /// Underlying checked BEM error.
        error: Box<BladeElementError>,
    },
    /// Initial fixed-circulation wake construction failed.
    InitialWake {
        /// Underlying steady-wake error.
        error: SteadyFreeWakeError,
    },
    /// A transient wake step failed.
    StepWake {
        /// One-based step index.
        step: usize,
        /// Underlying transient-wake error.
        error: TransientFreeWakeError,
    },
    /// A wake sample at one rotor station failed.
    StationInflow {
        /// One-based step index; zero identifies initial setup.
        step: usize,
        /// Station index.
        station: usize,
        /// Underlying steady-wake error.
        error: SteadyFreeWakeError,
    },
    /// A derived load/topology state violates the coupling contract.
    InvalidState {
        /// State field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A total work budget would be exceeded.
    BudgetExceeded {
        /// Budget name.
        field: &'static str,
        /// Required work.
        required: usize,
        /// Maximum admitted work.
        maximum: usize,
    },
}

impl fmt::Display for TransientLoadCoupledFreeWakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "invalid transient load-coupled option {field}: {requirement}"
            ),
            Self::InitialBem { error } => {
                write!(
                    formatter,
                    "initial transient load-coupled BEM failed: {error}"
                )
            }
            Self::StepBem { step, error } => {
                write!(
                    formatter,
                    "transient load-coupled BEM failed at step {step}: {error}"
                )
            }
            Self::InitialWake { error } => {
                write!(
                    formatter,
                    "initial transient load-coupled wake failed: {error}"
                )
            }
            Self::StepWake { step, error } => {
                write!(formatter, "transient wake failed at step {step}: {error}")
            }
            Self::StationInflow {
                step,
                station,
                error,
            } => write!(
                formatter,
                "wake inflow sample failed at step {step}, station {station}: {error}"
            ),
            Self::InvalidState { field, requirement } => write!(
                formatter,
                "invalid transient load-coupled state {field}: {requirement}"
            ),
            Self::BudgetExceeded {
                field,
                required,
                maximum,
            } => write!(
                formatter,
                "transient load-coupled {field} requires {required}, maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for TransientLoadCoupledFreeWakeError {}

/// Evaluate a bounded lagged load-coupled BEM/free-wake transient reference.
///
/// The initial BEM solve is performed with zero external inflow. At each
/// subsequent step, the current wake is sampled, the sampled station inflow
/// is relaxed, BEM is re-evaluated, the filament circulation is updated from
/// the new sectional lift, and the updated wake is explicitly convected.
pub fn evaluate_transient_load_coupled_free_wake(
    model: &BladeElementModel,
    operating_point: BladeElementOperatingPoint,
    bem_options: BladeElementBemOptions,
    options: TransientLoadCoupledFreeWakeOptions,
) -> Result<TransientLoadCoupledFreeWakeResult, TransientLoadCoupledFreeWakeError> {
    validate_options(&options)?;

    let zero_inflow = vec![
        BladeElementExternalInflow {
            axial_velocity_m_s: 0.0,
            tangential_velocity_m_s: 0.0,
        };
        model.stations().len()
    ];
    let initial_bem = model
        .evaluate_bem_with_external_inflow(operating_point, bem_options, &zero_inflow)
        .map_err(|error| TransientLoadCoupledFreeWakeError::InitialBem {
            error: Box::new(error),
        })?;
    if !finite_bem(&initial_bem) {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "initial_bem",
            requirement: "finite and converged",
        });
    }

    let mut wake = evaluate_steady_free_wake(model, &initial_bem, options.wake_options)
        .map_err(|error| TransientLoadCoupledFreeWakeError::InitialWake { error })?;
    if !wake.is_finite() {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "initial_wake",
            requirement: "finite and converged",
        });
    }
    validate_wake_topology(model, &wake)?;
    let mut external_inflow = zero_inflow;
    let mut steps = Vec::with_capacity(options.time_steps);
    let mut total_wake_evaluations = wake.biot_savart_evaluation_count;
    if total_wake_evaluations > options.max_total_wake_evaluations {
        return Err(TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "total_wake_evaluations",
            required: total_wake_evaluations,
            maximum: options.max_total_wake_evaluations,
        });
    }

    for step in 1..=options.time_steps {
        let time_s = step as f64 * options.time_step_s;
        let sampled_inflow = sample_station_inflow(model, &wake, step)?;
        let maximum_inflow_update_m_s = max_inflow_change(&external_inflow, &sampled_inflow);
        external_inflow =
            relax_inflow(&external_inflow, &sampled_inflow, options.inflow_relaxation)?;

        let bem = model
            .evaluate_bem_with_external_inflow(operating_point, bem_options, &external_inflow)
            .map_err(|error| TransientLoadCoupledFreeWakeError::StepBem {
                step,
                error: Box::new(error),
            })?;
        if !finite_bem(&bem) {
            return Err(TransientLoadCoupledFreeWakeError::InvalidState {
                field: "step_bem",
                requirement: "finite and converged",
            });
        }

        let previous_circulations = wake
            .filaments
            .iter()
            .map(|filament| filament.circulation_m2_s)
            .collect::<Vec<_>>();
        update_filament_loading(model, &bem, &mut wake.filaments, options.wake_options)?;
        let maximum_circulation_update_m2_s = previous_circulations
            .iter()
            .zip(&wake.filaments)
            .map(|(previous, filament)| (filament.circulation_m2_s - previous).abs())
            .fold(0.0, f64::max);

        let transient_options = TransientFreeWakeOptions {
            time_step_s: options.time_step_s,
            time_steps: 1,
            background_velocity_m_s: options.background_velocity_m_s,
            induced_velocity_scale: options.induced_velocity_scale,
            core_growth_m_s: options.core_growth_m_s,
            max_node_count: options.wake_options.max_node_count,
            max_biot_savart_evaluations_per_step: options.max_biot_savart_evaluations_per_step,
            max_node_displacement_m: options.max_node_displacement_m,
        };
        let advanced = advance_prescribed_free_wake_with_shedding(&wake, transient_options)
            .map_err(|error| TransientLoadCoupledFreeWakeError::StepWake { step, error })?;
        let diagnostic =
            advanced
                .steps
                .first()
                .ok_or(TransientLoadCoupledFreeWakeError::InvalidState {
                    field: "step_diagnostics",
                    requirement: "one completed diagnostic",
                })?;
        total_wake_evaluations = total_wake_evaluations
            .checked_add(diagnostic.biot_savart_evaluation_count)
            .ok_or(TransientLoadCoupledFreeWakeError::BudgetExceeded {
                field: "total_wake_evaluations",
                required: usize::MAX,
                maximum: options.max_total_wake_evaluations,
            })?;
        if total_wake_evaluations > options.max_total_wake_evaluations {
            return Err(TransientLoadCoupledFreeWakeError::BudgetExceeded {
                field: "total_wake_evaluations",
                required: total_wake_evaluations,
                maximum: options.max_total_wake_evaluations,
            });
        }

        wake.filaments = advanced.filaments;
        wake.node_count = wake
            .filaments
            .iter()
            .try_fold(0usize, |count, filament| {
                count.checked_add(filament.nodes.len())
            })
            .ok_or(TransientLoadCoupledFreeWakeError::InvalidState {
                field: "wake.node_count",
                requirement: "fits usize",
            })?;
        wake.biot_savart_evaluation_count = diagnostic.biot_savart_evaluation_count.max(1);
        wake.core_radius_m = advanced.final_core_radius_m;
        wake.operating_point = bem.loads.operating_point;
        steps.push(TransientLoadCoupledFreeWakeStep {
            step,
            time_s,
            bem,
            station_external_inflow: external_inflow.clone(),
            maximum_inflow_update_m_s,
            maximum_circulation_update_m2_s,
            maximum_node_displacement_m: diagnostic.maximum_node_displacement_m,
            maximum_age_s: diagnostic.maximum_age_s,
            biot_savart_evaluation_count: diagnostic.biot_savart_evaluation_count,
        });
    }

    let total_bem_evaluation_count = options.time_steps.checked_add(1).ok_or(
        TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "bem_evaluations",
            required: usize::MAX,
            maximum: options.max_bem_evaluations,
        },
    )?;
    if total_bem_evaluation_count > options.max_bem_evaluations {
        return Err(TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "bem_evaluations",
            required: total_bem_evaluation_count,
            maximum: options.max_bem_evaluations,
        });
    }
    let result = TransientLoadCoupledFreeWakeResult {
        operating_point: wake.operating_point,
        blade_count: wake.blade_count,
        filaments: wake.filaments,
        steps,
        total_bem_evaluation_count,
        total_wake_evaluation_count: total_wake_evaluations,
        completed: true,
    };
    if !result.is_finite() {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "result",
            requirement: "finite and complete",
        });
    }
    Ok(result)
}

fn validate_options(
    options: &TransientLoadCoupledFreeWakeOptions,
) -> Result<(), TransientLoadCoupledFreeWakeError> {
    if !options.time_step_s.is_finite() || options.time_step_s <= 0.0 {
        return Err(invalid_option("time_step_s", "finite and > 0"));
    }
    if options.time_steps == 0 {
        return Err(invalid_option("time_steps", "> 0"));
    }
    if !finite_point(options.background_velocity_m_s) {
        return Err(invalid_option(
            "background_velocity_m_s",
            "all components finite",
        ));
    }
    if !options.induced_velocity_scale.is_finite() {
        return Err(invalid_option("induced_velocity_scale", "finite"));
    }
    if !options.core_growth_m_s.is_finite() || options.core_growth_m_s < 0.0 {
        return Err(invalid_option("core_growth_m_s", "finite and >= 0"));
    }
    if !options.inflow_relaxation.is_finite()
        || options.inflow_relaxation <= 0.0
        || options.inflow_relaxation > 1.0
    {
        return Err(invalid_option("inflow_relaxation", "finite and in (0, 1]"));
    }
    if options.max_bem_evaluations == 0 {
        return Err(invalid_option("max_bem_evaluations", "> 0"));
    }
    if options.max_total_wake_evaluations == 0 {
        return Err(invalid_option("max_total_wake_evaluations", "> 0"));
    }
    if options.max_biot_savart_evaluations_per_step == 0 {
        return Err(invalid_option(
            "max_biot_savart_evaluations_per_step",
            "> 0",
        ));
    }
    if !options.max_node_displacement_m.is_finite() || options.max_node_displacement_m <= 0.0 {
        return Err(invalid_option("max_node_displacement_m", "finite and > 0"));
    }
    let duration_s = options.time_steps as f64 * options.time_step_s;
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(invalid_option(
            "time_steps",
            "time_steps * time_step_s is finite",
        ));
    }
    if !options.core_growth_m_s.mul_add(duration_s, 0.0).is_finite() {
        return Err(invalid_option(
            "core_growth_m_s",
            "growth over the requested duration is finite",
        ));
    }
    let required_bem = options.time_steps.checked_add(1).ok_or(
        TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "bem_evaluations",
            required: usize::MAX,
            maximum: options.max_bem_evaluations,
        },
    )?;
    if required_bem > options.max_bem_evaluations {
        return Err(TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "bem_evaluations",
            required: required_bem,
            maximum: options.max_bem_evaluations,
        });
    }
    Ok(())
}

fn invalid_option(
    field: &'static str,
    requirement: &'static str,
) -> TransientLoadCoupledFreeWakeError {
    TransientLoadCoupledFreeWakeError::InvalidOptions { field, requirement }
}

fn validate_wake_topology(
    model: &BladeElementModel,
    wake: &SteadyFreeWakeResult,
) -> Result<(), TransientLoadCoupledFreeWakeError> {
    let seed_count = model.stations().len().checked_add(1).ok_or(
        TransientLoadCoupledFreeWakeError::InvalidState {
            field: "wake.filaments",
            requirement: "station seed count fits usize",
        },
    )?;
    let expected_filament_count = seed_count.checked_mul(model.n_blades()).ok_or(
        TransientLoadCoupledFreeWakeError::InvalidState {
            field: "wake.filaments",
            requirement: "blade seed count fits usize",
        },
    )?;
    if wake.filaments.len() != expected_filament_count {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "wake.filaments",
            requirement: "one station/tip seed per blade",
        });
    }
    Ok(())
}

fn sample_station_inflow(
    model: &BladeElementModel,
    wake: &SteadyFreeWakeResult,
    step: usize,
) -> Result<Vec<BladeElementExternalInflow>, TransientLoadCoupledFreeWakeError> {
    model
        .stations()
        .iter()
        .enumerate()
        .map(|(station, geometry)| {
            let induced = wake
                .induced_velocity_at(Point3 {
                    x: geometry.radius_m,
                    y: 0.0,
                    z: 0.0,
                })
                .map_err(|error| TransientLoadCoupledFreeWakeError::StationInflow {
                    step,
                    station,
                    error,
                })?;
            if !finite_point(induced) {
                return Err(TransientLoadCoupledFreeWakeError::InvalidState {
                    field: "station_inflow",
                    requirement: "finite",
                });
            }
            Ok(BladeElementExternalInflow {
                axial_velocity_m_s: induced.z,
                tangential_velocity_m_s: induced.y,
            })
        })
        .collect()
}

fn relax_inflow(
    previous: &[BladeElementExternalInflow],
    sampled: &[BladeElementExternalInflow],
    relaxation: f64,
) -> Result<Vec<BladeElementExternalInflow>, TransientLoadCoupledFreeWakeError> {
    if previous.len() != sampled.len() {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "station_external_inflow",
            requirement: "stable station count",
        });
    }
    previous
        .iter()
        .zip(sampled)
        .map(|(previous, sampled)| {
            let inflow = BladeElementExternalInflow {
                axial_velocity_m_s: previous.axial_velocity_m_s
                    + relaxation * (sampled.axial_velocity_m_s - previous.axial_velocity_m_s),
                tangential_velocity_m_s: previous.tangential_velocity_m_s
                    + relaxation
                        * (sampled.tangential_velocity_m_s - previous.tangential_velocity_m_s),
            };
            if !inflow.axial_velocity_m_s.is_finite() || !inflow.tangential_velocity_m_s.is_finite()
            {
                return Err(TransientLoadCoupledFreeWakeError::InvalidState {
                    field: "station_external_inflow",
                    requirement: "finite after relaxation",
                });
            }
            Ok(inflow)
        })
        .collect()
}

fn update_filament_loading(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    filaments: &mut [crate::SteadyFreeWakeFilament],
    wake_options: SteadyFreeWakeOptions,
) -> Result<(), TransientLoadCoupledFreeWakeError> {
    let station_count = model.stations().len();
    let seed_count =
        station_count
            .checked_add(1)
            .ok_or(TransientLoadCoupledFreeWakeError::InvalidState {
                field: "load_topology",
                requirement: "station seed count fits usize",
            })?;
    let expected_filament_count = seed_count.checked_mul(model.n_blades()).ok_or(
        TransientLoadCoupledFreeWakeError::InvalidState {
            field: "load_topology",
            requirement: "blade seed count fits usize",
        },
    )?;
    if bem.loads.strips.len() != station_count
        || bem.strips.len() != station_count
        || filaments.len() != expected_filament_count
    {
        return Err(TransientLoadCoupledFreeWakeError::InvalidState {
            field: "load_topology",
            requirement: "BEM station and wake filament counts agree",
        });
    }
    let mut last_axial_speed = 0.0;
    let mut last_circulation = 0.0;
    let mut loading = Vec::with_capacity(station_count);
    for (index, (station, load)) in model.stations().iter().zip(&bem.loads.strips).enumerate() {
        let circulation_m2_s =
            0.5 * station.chord_m * load.relative_speed_m_s * load.lift_coefficient;
        let axial_speed_m_s = bem.loads.operating_point.freestream_m_s
            * (1.0 + bem.strips[index].axial_induction)
            * wake_options.axial_convection_factor;
        if !circulation_m2_s.is_finite() || !axial_speed_m_s.is_finite() || axial_speed_m_s <= 0.0 {
            return Err(TransientLoadCoupledFreeWakeError::InvalidState {
                field: "filament_loading",
                requirement: "finite circulation and positive axial convection",
            });
        }
        loading.push((circulation_m2_s, axial_speed_m_s));
        last_circulation = circulation_m2_s;
        last_axial_speed = axial_speed_m_s;
    }
    for blade_filaments in filaments.chunks_exact_mut(seed_count) {
        for (index, (circulation_m2_s, axial_speed_m_s)) in loading.iter().copied().enumerate() {
            blade_filaments[index].circulation_m2_s = circulation_m2_s;
            blade_filaments[index].axial_convection_speed_m_s = axial_speed_m_s;
        }
        let tip =
            blade_filaments
                .last_mut()
                .ok_or(TransientLoadCoupledFreeWakeError::InvalidState {
                    field: "filaments",
                    requirement: "tip closure filament exists",
                })?;
        tip.circulation_m2_s = -last_circulation;
        tip.axial_convection_speed_m_s = last_axial_speed;
    }
    Ok(())
}

fn max_inflow_change(
    previous: &[BladeElementExternalInflow],
    next: &[BladeElementExternalInflow],
) -> f64 {
    previous
        .iter()
        .zip(next)
        .map(|(previous, next)| {
            (next.axial_velocity_m_s - previous.axial_velocity_m_s)
                .abs()
                .max((next.tangential_velocity_m_s - previous.tangential_velocity_m_s).abs())
        })
        .fold(0.0, f64::max)
}

fn finite_bem(bem: &BladeElementBemResult) -> bool {
    let loads = &bem.loads;
    bem.converged
        && bem.max_iterations_used > 0
        && loads.operating_point.rpm.is_finite()
        && loads.operating_point.rpm > 0.0
        && loads.operating_point.freestream_m_s.is_finite()
        && loads.operating_point.freestream_m_s > 0.0
        && loads.operating_point.density_kg_m3.is_finite()
        && loads.operating_point.density_kg_m3 > 0.0
        && loads.omega_rad_s.is_finite()
        && loads.omega_rad_s > 0.0
        && loads.advance_ratio.is_finite()
        && loads.tip_speed_m_s.is_finite()
        && loads.thrust_n.is_finite()
        && loads.torque_nm.is_finite()
        && loads.shaft_power_w.is_finite()
        && loads.useful_power_w.is_finite()
        && loads.propulsive_efficiency.is_finite()
        && loads.strips.len() == bem.strips.len()
        && loads.strips.iter().all(|strip| {
            strip.radius_m.is_finite()
                && strip.width_m.is_finite()
                && strip.chord_m.is_finite()
                && strip.pitch_deg.is_finite()
                && strip.inflow_angle_deg.is_finite()
                && strip.angle_of_attack_deg.is_finite()
                && strip.relative_speed_m_s.is_finite()
                && strip.lift_coefficient.is_finite()
                && strip.drag_coefficient.is_finite()
                && strip.axial_force_per_m_n.is_finite()
                && strip.tangential_force_per_m_n.is_finite()
                && strip.thrust_n.is_finite()
                && strip.torque_nm.is_finite()
        })
        && bem.strips.iter().all(|strip| {
            strip.axial_induction.is_finite()
                && strip.tangential_induction.is_finite()
                && strip.loss_factor.is_finite()
                && strip.iterations > 0
                && strip.residual.is_finite()
        })
}

fn finite_inflow(inflow: &[BladeElementExternalInflow]) -> bool {
    inflow
        .iter()
        .all(|item| item.axial_velocity_m_s.is_finite() && item.tangential_velocity_m_s.is_finite())
}

fn finite_filaments(filaments: &[crate::SteadyFreeWakeFilament], blade_count: usize) -> bool {
    filaments.iter().all(|filament| {
        filament.blade_index < blade_count
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
}

fn finite_point(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}
