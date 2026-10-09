//! Checked independence evidence for resolved PC-fan CFD studies.
//!
//! The steady FEM solver reports one converged field at a time.  Resolved
//! transient or rotating-frame runners can use this module to publish their
//! mesh, time-step, revolution, and nonlinear-iteration ladders through one
//! fail-closed contract without pretending that a single field is an
//! independence study.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::cfd::{
    PcFanCfdTransientError, solve_pc_fan_cfd_rotating_wall_transient,
    try_solve_pc_fan_cfd_mesh_convergence,
};
#[cfg(test)]
use crate::model::{PcFanCfdBackend, PcFanFlowModel};
use crate::model::{
    PcFanCfdConfig, PcFanCfdMeshConvergenceConfig, PcFanCfdMeshConvergenceConfigError, PcFanConfig,
    PcFanDesignVariables,
};

/// Independent resolution axis represented by a CFD evidence ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanCfdIndependenceAxis {
    /// Increasing cells per axis in an isotropic refinement.
    Mesh,
    /// Decreasing physical time step in seconds.
    TimeStep,
    /// Increasing simulated rotor revolutions.
    RotorRevolutions,
    /// Increasing nonlinear/Picard iteration budget.
    NonlinearIterations,
}

impl PcFanCfdIndependenceAxis {
    fn label(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::TimeStep => "time step",
            Self::RotorRevolutions => "rotor revolutions",
            Self::NonlinearIterations => "nonlinear iterations",
        }
    }

    fn levels_increase(self) -> bool {
        !matches!(self, Self::TimeStep)
    }

    fn requires_integer_level(self) -> bool {
        matches!(self, Self::Mesh | Self::NonlinearIterations)
    }
}

/// One solved observation in an independence ladder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdIndependenceSample {
    /// Resolution value. Units are defined by [`PcFanCfdIndependenceAxis`].
    pub level: f64,
    pub airflow_cfm: f64,
    pub pressure_drop_pa: f64,
    /// Axial recovered actuator/wall loading on the fluid.
    pub axial_loading_n: f64,
    pub residual: f64,
    pub relative_mass_imbalance: f64,
    pub relative_momentum_imbalance: f64,
    pub relative_pressure_closure: f64,
    pub relative_energy_imbalance: f64,
    pub converged: bool,
}

/// Input observations for one independent CFD ladder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdIndependenceStudy {
    pub axis: PcFanCfdIndependenceAxis,
    pub samples: Vec<PcFanCfdIndependenceSample>,
}

/// Checked tolerances and required axes for an independence report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdIndependenceConfig {
    pub relative_tolerance: f64,
    pub min_levels: usize,
    /// Maximum residual admitted for the finest level of each study.
    ///
    /// Observable agreement alone is not sufficient evidence of an
    /// independent CFD result: two under-converged levels can agree while
    /// both still carry a large equation residual. This absolute bound is
    /// separate from the relative observable-change tolerance because
    /// residuals and observables have different units and scales.
    #[serde(default = "default_max_final_residual")]
    pub max_final_residual: f64,
    /// Maximum relative control-volume mass-flow imbalance per sample.
    #[serde(default = "default_max_relative_mass_imbalance")]
    pub max_relative_mass_imbalance: f64,
    /// Maximum relative control-volume axial-momentum imbalance per sample.
    #[serde(default = "default_max_relative_momentum_imbalance")]
    pub max_relative_momentum_imbalance: f64,
    /// Maximum relative pressure-force closure error per sample.
    #[serde(default = "default_max_relative_pressure_closure")]
    pub max_relative_pressure_closure: f64,
    /// Maximum relative pressure/shaft-power closure error per sample.
    #[serde(default = "default_max_relative_energy_imbalance")]
    pub max_relative_energy_imbalance: f64,
    /// Defaults to all four axes required for resolved-flow promotion.
    pub required_axes: Vec<PcFanCfdIndependenceAxis>,
}

fn default_max_final_residual() -> f64 {
    1.0e-3
}

fn default_max_relative_mass_imbalance() -> f64 {
    5.0e-2
}

fn default_max_relative_momentum_imbalance() -> f64 {
    2.0
}

fn default_max_relative_pressure_closure() -> f64 {
    2.0
}

fn default_max_relative_energy_imbalance() -> f64 {
    2.0
}

impl Default for PcFanCfdIndependenceConfig {
    fn default() -> Self {
        Self {
            relative_tolerance: 0.05,
            min_levels: 2,
            max_final_residual: default_max_final_residual(),
            max_relative_mass_imbalance: default_max_relative_mass_imbalance(),
            max_relative_momentum_imbalance: default_max_relative_momentum_imbalance(),
            max_relative_pressure_closure: default_max_relative_pressure_closure(),
            max_relative_energy_imbalance: default_max_relative_energy_imbalance(),
            required_axes: vec![
                PcFanCfdIndependenceAxis::Mesh,
                PcFanCfdIndependenceAxis::TimeStep,
                PcFanCfdIndependenceAxis::RotorRevolutions,
                PcFanCfdIndependenceAxis::NonlinearIterations,
            ],
        }
    }
}

/// Input/configuration errors found before physical comparison begins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcFanCfdIndependenceError {
    NoRequiredAxes,
    DuplicateRequiredAxis(PcFanCfdIndependenceAxis),
    DuplicateStudyAxis(PcFanCfdIndependenceAxis),
    UnexpectedStudyAxis(PcFanCfdIndependenceAxis),
    MissingStudyAxis(PcFanCfdIndependenceAxis),
    InvalidRelativeTolerance,
    InvalidFinalResidualTolerance,
    InvalidConservationTolerance {
        metric: &'static str,
    },
    MinLevelsTooSmall,
    TooFewSamples {
        axis: PcFanCfdIndependenceAxis,
        actual: usize,
        required: usize,
    },
    NonFiniteSample {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
    },
    InvalidSampleLevel {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        level: f64,
    },
    NonMonotonicLevels {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        previous: f64,
        current: f64,
    },
    ConservationViolation {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        metric: &'static str,
        value: f64,
        tolerance: f64,
    },
}

/// Errors raised while adapting a production PC-fan mesh sweep into the
/// generic independence evidence contract.
#[derive(Debug, PartialEq)]
pub enum PcFanCfdProductionIndependenceError {
    MeshConvergence(PcFanCfdMeshConvergenceConfigError),
    Transient(PcFanCfdTransientError),
    Independence(PcFanCfdIndependenceError),
    UnsupportedAxis(PcFanCfdIndependenceAxis),
    /// A production transient ladder must represent its requested interval
    /// exactly; the transient runner otherwise ceil-rounds the step count.
    NonIntegralTimeGrid {
        time_step_s: f64,
        rotor_revolutions: f64,
        step_count: f64,
    },
    TooFewTimeSteps {
        time_step_s: f64,
        rotor_revolutions: f64,
        step_count: f64,
    },
    /// Time-step levels must reach the same physical final time.
    TimeStepFinalTimeMismatch {
        expected_final_time_s: f64,
        actual_final_time_s: f64,
    },
}

impl fmt::Display for PcFanCfdProductionIndependenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeshConvergence(error) => write!(formatter, "mesh convergence: {error}"),
            Self::Transient(error) => write!(formatter, "transient CFD: {error}"),
            Self::Independence(error) => write!(formatter, "independence contract: {error}"),
            Self::UnsupportedAxis(axis) => write!(
                formatter,
                "production steady PC-fan runner does not provide the {} axis",
                axis.label()
            ),
            Self::NonIntegralTimeGrid {
                time_step_s,
                rotor_revolutions,
                step_count,
            } => write!(
                formatter,
                "transient time step {time_step_s} does not exactly represent {rotor_revolutions} rotor revolutions ({step_count} steps)"
            ),
            Self::TooFewTimeSteps {
                time_step_s,
                rotor_revolutions,
                step_count,
            } => write!(
                formatter,
                "transient time step {time_step_s} gives only {step_count} steps for {rotor_revolutions} rotor revolutions"
            ),
            Self::TimeStepFinalTimeMismatch {
                expected_final_time_s,
                actual_final_time_s,
            } => write!(
                formatter,
                "transient time-step levels end at different times: expected {expected_final_time_s}, got {actual_final_time_s}"
            ),
        }
    }
}

impl std::error::Error for PcFanCfdProductionIndependenceError {}

impl From<PcFanCfdIndependenceError> for PcFanCfdProductionIndependenceError {
    fn from(error: PcFanCfdIndependenceError) -> Self {
        Self::Independence(error)
    }
}

impl From<PcFanCfdTransientError> for PcFanCfdProductionIndependenceError {
    fn from(error: PcFanCfdTransientError) -> Self {
        Self::Transient(error)
    }
}

impl fmt::Display for PcFanCfdIndependenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRequiredAxes => write!(f, "PC-fan CFD independence requires at least one axis"),
            Self::DuplicateRequiredAxis(axis) => {
                write!(
                    f,
                    "PC-fan CFD independence repeats required {} axis",
                    axis.label()
                )
            }
            Self::DuplicateStudyAxis(axis) => {
                write!(f, "PC-fan CFD independence repeats {} study", axis.label())
            }
            Self::UnexpectedStudyAxis(axis) => {
                write!(
                    f,
                    "PC-fan CFD independence received unexpected {} study",
                    axis.label()
                )
            }
            Self::MissingStudyAxis(axis) => {
                write!(
                    f,
                    "PC-fan CFD independence is missing {} study",
                    axis.label()
                )
            }
            Self::InvalidRelativeTolerance => write!(
                f,
                "PC-fan CFD independence relative_tolerance must be finite and non-negative"
            ),
            Self::InvalidFinalResidualTolerance => write!(
                f,
                "PC-fan CFD independence max_final_residual must be finite and non-negative"
            ),
            Self::InvalidConservationTolerance { metric } => write!(
                f,
                "PC-fan CFD independence {metric} tolerance must be finite non-negative"
            ),
            Self::MinLevelsTooSmall => {
                write!(f, "PC-fan CFD independence min_levels must be at least two")
            }
            Self::TooFewSamples {
                axis,
                actual,
                required,
            } => write!(
                f,
                "PC-fan CFD independence {} study has {actual} samples; requires {required}",
                axis.label()
            ),
            Self::NonFiniteSample { axis, index } => write!(
                f,
                "PC-fan CFD independence {} sample {index} contains a non-finite value",
                axis.label()
            ),
            Self::InvalidSampleLevel { axis, index, level } => write!(
                f,
                "PC-fan CFD independence {} sample {index} has invalid level {level}",
                axis.label()
            ),
            Self::ConservationViolation {
                axis,
                index,
                metric,
                value,
                tolerance,
            } => write!(
                f,
                "PC-fan CFD independence {} sample {index} {metric} {value:.3e} exceeds tolerance {tolerance:.3e}",
                axis.label()
            ),
            Self::NonMonotonicLevels {
                axis,
                index,
                previous,
                current,
            } => write!(
                f,
                "PC-fan CFD independence {} levels are not monotonic at {index}: {previous} -> {current}",
                axis.label()
            ),
        }
    }
}

impl std::error::Error for PcFanCfdIndependenceError {}

impl PcFanCfdIndependenceConfig {
    fn with_cfd_conservation_budgets(&self, cfd: &PcFanCfdConfig) -> Self {
        let mut config = self.clone();
        config.max_relative_mass_imbalance = cfd.max_relative_mass_imbalance;
        config.max_relative_momentum_imbalance = cfd.max_relative_momentum_imbalance;
        config.max_relative_pressure_closure = cfd.max_relative_pressure_closure;
        config.max_relative_energy_imbalance = cfd.max_relative_energy_imbalance;
        config
    }

    fn validate(&self) -> Result<(), PcFanCfdIndependenceError> {
        if self.required_axes.is_empty() {
            return Err(PcFanCfdIndependenceError::NoRequiredAxes);
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(PcFanCfdIndependenceError::InvalidRelativeTolerance);
        }
        if !self.max_final_residual.is_finite() || self.max_final_residual < 0.0 {
            return Err(PcFanCfdIndependenceError::InvalidFinalResidualTolerance);
        }
        for (metric, tolerance) in self.conservation_tolerances() {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(PcFanCfdIndependenceError::InvalidConservationTolerance { metric });
            }
        }
        if self.min_levels < 2 {
            return Err(PcFanCfdIndependenceError::MinLevelsTooSmall);
        }
        for (index, axis) in self.required_axes.iter().copied().enumerate() {
            if self.required_axes[..index].contains(&axis) {
                return Err(PcFanCfdIndependenceError::DuplicateRequiredAxis(axis));
            }
        }
        Ok(())
    }

    fn conservation_tolerances(&self) -> [(&'static str, f64); 4] {
        [
            ("mass imbalance", self.max_relative_mass_imbalance),
            ("momentum imbalance", self.max_relative_momentum_imbalance),
            ("pressure closure", self.max_relative_pressure_closure),
            ("energy imbalance", self.max_relative_energy_imbalance),
        ]
    }
}

/// Per-axis comparison generated by [`evaluate_pc_fan_cfd_independence`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdIndependenceStudyReport {
    pub axis: PcFanCfdIndependenceAxis,
    pub sample_count: usize,
    pub first_level: f64,
    pub final_level: f64,
    pub airflow_relative_change: f64,
    pub pressure_relative_change: f64,
    pub axial_loading_relative_change: f64,
    pub final_residual: f64,
    pub all_samples_converged: bool,
    pub converged: bool,
}

/// Complete, promotion-ready independence report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdIndependenceReport {
    pub studies: Vec<PcFanCfdIndependenceStudyReport>,
    pub required_axes: Vec<PcFanCfdIndependenceAxis>,
    pub all_samples_finite: bool,
    pub all_samples_converged: bool,
    pub converged: bool,
}

impl PcFanCfdIndependenceReport {
    /// Whether all required axes passed their observable-agreement gates.
    pub fn is_valid(&self) -> bool {
        !self.studies.is_empty()
            && self.studies.len() == self.required_axes.len()
            && self.all_samples_finite
            && self.all_samples_converged
            && self.converged
    }
}

/// Validate and compare independent CFD ladders.
///
/// This function intentionally consumes solved observations rather than
/// inventing transient data from a steady field.  A resolved-flow runner can
/// serialize each ladder into these types, and promotion then requires all
/// configured axes to agree in airflow, pressure drop, and recovered loading.
pub fn evaluate_pc_fan_cfd_independence(
    config: &PcFanCfdIndependenceConfig,
    studies: &[PcFanCfdIndependenceStudy],
) -> Result<PcFanCfdIndependenceReport, PcFanCfdIndependenceError> {
    config.validate()?;

    for (index, study) in studies.iter().enumerate() {
        if !config.required_axes.contains(&study.axis) {
            return Err(PcFanCfdIndependenceError::UnexpectedStudyAxis(study.axis));
        }
        if studies[..index]
            .iter()
            .any(|previous| previous.axis == study.axis)
        {
            return Err(PcFanCfdIndependenceError::DuplicateStudyAxis(study.axis));
        }
    }

    let mut reports = Vec::with_capacity(config.required_axes.len());
    for axis in config.required_axes.iter().copied() {
        let study = studies
            .iter()
            .find(|study| study.axis == axis)
            .ok_or(PcFanCfdIndependenceError::MissingStudyAxis(axis))?;
        reports.push(compare_study(config, study)?);
    }

    let all_samples_finite = reports.iter().all(|report| {
        report.airflow_relative_change.is_finite()
            && report.pressure_relative_change.is_finite()
            && report.axial_loading_relative_change.is_finite()
            && report.final_residual.is_finite()
    });
    let all_samples_converged = reports.iter().all(|report| report.all_samples_converged);
    let converged = reports.iter().all(|report| report.converged);

    Ok(PcFanCfdIndependenceReport {
        studies: reports,
        required_axes: config.required_axes.clone(),
        all_samples_finite,
        all_samples_converged,
        converged,
    })
}

/// Run the production steady mesh ladder and adapt each solved level into
/// the checked independence contract.
///
/// The current PC-fan runner is steady and has no physical time-step or rotor
/// revolution integrator.  Those axes therefore fail closed instead of being
/// synthesized from repeated steady solves.  Callers can still combine this
/// report with independently produced transient studies through
/// [`evaluate_pc_fan_cfd_independence`].
pub fn evaluate_pc_fan_cfd_mesh_independence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    mesh_config: PcFanCfdMeshConvergenceConfig,
    independence_config: &PcFanCfdIndependenceConfig,
) -> Result<PcFanCfdIndependenceReport, PcFanCfdProductionIndependenceError> {
    let independence_config = independence_config.with_cfd_conservation_budgets(&config.cfd);
    independence_config.validate()?;
    if let Some(axis) = independence_config
        .required_axes
        .iter()
        .copied()
        .find(|axis| *axis != PcFanCfdIndependenceAxis::Mesh)
    {
        return Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(axis));
    }

    let convergence = try_solve_pc_fan_cfd_mesh_convergence(config, design, mesh_config)
        .map_err(PcFanCfdProductionIndependenceError::MeshConvergence)?;
    let study = PcFanCfdIndependenceStudy {
        axis: PcFanCfdIndependenceAxis::Mesh,
        samples: convergence
            .levels
            .iter()
            .map(|level| PcFanCfdIndependenceSample {
                level: level.cells_x as f64,
                airflow_cfm: level.solution.airflow_cfm,
                pressure_drop_pa: level.solution.pressure_drop_pa,
                axial_loading_n: level.solution.conservation.body_force_n,
                residual: level.solution.residual,
                relative_mass_imbalance: level.solution.conservation.relative_mass_imbalance,
                relative_momentum_imbalance: level
                    .solution
                    .conservation
                    .relative_momentum_imbalance,
                relative_pressure_closure: level.solution.conservation.relative_pressure_closure,
                relative_energy_imbalance: level.solution.conservation.relative_energy_imbalance,
                converged: level.solution.converged,
            })
            .collect(),
    };
    evaluate_pc_fan_cfd_independence(&independence_config, &[study]).map_err(Into::into)
}

/// Run a bounded production nonlinear-iteration ladder and adapt it into the
/// checked independence contract.  This varies only the Picard iteration
/// budget; it does not synthesize temporal or rotor-revolution evidence.
pub fn evaluate_pc_fan_cfd_nonlinear_independence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    iteration_levels: &[usize],
    independence_config: &PcFanCfdIndependenceConfig,
) -> Result<PcFanCfdIndependenceReport, PcFanCfdProductionIndependenceError> {
    let independence_config = independence_config.with_cfd_conservation_budgets(&config.cfd);
    if let Some(axis) = independence_config
        .required_axes
        .iter()
        .copied()
        .find(|axis| *axis != PcFanCfdIndependenceAxis::NonlinearIterations)
    {
        return Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(axis));
    }
    if let Some((index, &level)) = iteration_levels
        .iter()
        .enumerate()
        .find(|(_, level)| **level == 0)
    {
        return Err(PcFanCfdProductionIndependenceError::Independence(
            PcFanCfdIndependenceError::InvalidSampleLevel {
                axis: PcFanCfdIndependenceAxis::NonlinearIterations,
                index,
                level: level as f64,
            },
        ));
    }

    let mut samples = Vec::with_capacity(iteration_levels.len());
    for &iteration_limit in iteration_levels {
        let mut level_config = config.clone();
        level_config.cfd.max_picard_iterations = iteration_limit;
        let solution = crate::cfd::solve_pc_fan_cfd_mean_flow(&level_config, design);
        samples.push(PcFanCfdIndependenceSample {
            level: iteration_limit as f64,
            airflow_cfm: solution.airflow_cfm,
            pressure_drop_pa: solution.pressure_drop_pa,
            axial_loading_n: solution.conservation.body_force_n,
            residual: solution.residual,
            relative_mass_imbalance: solution.conservation.relative_mass_imbalance,
            relative_momentum_imbalance: solution.conservation.relative_momentum_imbalance,
            relative_pressure_closure: solution.conservation.relative_pressure_closure,
            relative_energy_imbalance: solution.conservation.relative_energy_imbalance,
            converged: solution.converged,
        });
    }
    evaluate_pc_fan_cfd_independence(
        &independence_config,
        &[PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::NonlinearIterations,
            samples,
        }],
    )
    .map_err(Into::into)
}

/// Run production resolved rotating-wall transient ladders for the requested
/// time-step and rotor-revolution axes.  Each observation is generated by the
/// BDF runner and averaged over its final revolution; no steady field is
/// duplicated to manufacture transient evidence.
pub fn evaluate_pc_fan_cfd_transient_independence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_levels_s: &[f64],
    rotor_revolution_levels: &[f64],
    independence_config: &PcFanCfdIndependenceConfig,
) -> Result<PcFanCfdIndependenceReport, PcFanCfdProductionIndependenceError> {
    let independence_config = independence_config.with_cfd_conservation_budgets(&config.cfd);
    independence_config.validate()?;
    if let Some(axis) = independence_config
        .required_axes
        .iter()
        .copied()
        .find(|axis| {
            !matches!(
                axis,
                PcFanCfdIndependenceAxis::TimeStep | PcFanCfdIndependenceAxis::RotorRevolutions
            )
        })
    {
        return Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(axis));
    }

    // Reference levels must come from the caller: silently substituting 1.0
    // revolution / 0.01 s when a ladder is empty manufactures a reference
    // instead of failing closed like the rest of this module.
    if independence_config
        .required_axes
        .contains(&PcFanCfdIndependenceAxis::TimeStep)
        && rotor_revolution_levels.is_empty()
    {
        return Err(PcFanCfdProductionIndependenceError::Independence(
            PcFanCfdIndependenceError::TooFewSamples {
                axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                actual: 0,
                required: 1,
            },
        ));
    }
    if independence_config
        .required_axes
        .contains(&PcFanCfdIndependenceAxis::RotorRevolutions)
        && time_step_levels_s.is_empty()
    {
        return Err(PcFanCfdProductionIndependenceError::Independence(
            PcFanCfdIndependenceError::TooFewSamples {
                axis: PcFanCfdIndependenceAxis::TimeStep,
                actual: 0,
                required: 1,
            },
        ));
    }
    let mut studies = Vec::with_capacity(independence_config.required_axes.len());
    if independence_config
        .required_axes
        .contains(&PcFanCfdIndependenceAxis::TimeStep)
    {
        if time_step_levels_s.is_empty() {
            return Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::TimeStep,
                    actual: 0,
                    required: independence_config.min_levels,
                },
            ));
        }
        validate_level_ladder(
            &independence_config,
            PcFanCfdIndependenceAxis::TimeStep,
            time_step_levels_s,
        )?;
        // The complementary-ladder guard above guarantees a caller-supplied
        // reference; re-derive it here so no manufactured fallback can reach
        // the solver if the guards are ever reordered.
        let reference_revolutions = rotor_revolution_levels.last().copied().ok_or(
            PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                    actual: 0,
                    required: 1,
                },
            ),
        )?;
        for &time_step_s in time_step_levels_s {
            validate_transient_time_grid(config, time_step_s, reference_revolutions)?;
        }
        let mut samples = Vec::with_capacity(time_step_levels_s.len());
        let mut expected_final_time_s: Option<f64> = None;
        for &time_step_s in time_step_levels_s {
            let observation = solve_pc_fan_cfd_rotating_wall_transient(
                config,
                design,
                time_step_s,
                reference_revolutions,
            )?;
            if let Some(expected) = expected_final_time_s {
                if (observation.final_time_s - expected).abs()
                    > 1.0e-10 * observation.final_time_s.abs().max(expected.abs()).max(1.0)
                {
                    return Err(
                        PcFanCfdProductionIndependenceError::TimeStepFinalTimeMismatch {
                            expected_final_time_s: expected,
                            actual_final_time_s: observation.final_time_s,
                        },
                    );
                }
            } else {
                expected_final_time_s = Some(observation.final_time_s);
            }
            samples.push(PcFanCfdIndependenceSample {
                level: time_step_s,
                airflow_cfm: observation.airflow_cfm,
                pressure_drop_pa: observation.pressure_drop_pa,
                axial_loading_n: observation.axial_loading_n,
                residual: observation.residual,
                relative_mass_imbalance: observation.conservation.relative_mass_imbalance,
                relative_momentum_imbalance: observation.conservation.relative_momentum_imbalance,
                relative_pressure_closure: observation.conservation.relative_pressure_closure,
                relative_energy_imbalance: observation.conservation.relative_energy_imbalance,
                converged: observation.converged,
            });
        }
        studies.push(PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::TimeStep,
            samples,
        });
    }
    if independence_config
        .required_axes
        .contains(&PcFanCfdIndependenceAxis::RotorRevolutions)
    {
        if rotor_revolution_levels.is_empty() {
            return Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                    actual: 0,
                    required: independence_config.min_levels,
                },
            ));
        }
        validate_level_ladder(
            &independence_config,
            PcFanCfdIndependenceAxis::RotorRevolutions,
            rotor_revolution_levels,
        )?;
        // Caller-supplied reference only: never substitute a manufactured
        // time step when the ladder is empty.
        let reference_time_step = time_step_levels_s.last().copied().ok_or(
            PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::TimeStep,
                    actual: 0,
                    required: 1,
                },
            ),
        )?;
        for &rotor_revolutions in rotor_revolution_levels {
            validate_transient_time_grid(config, reference_time_step, rotor_revolutions)?;
        }
        let mut samples = Vec::with_capacity(rotor_revolution_levels.len());
        for &rotor_revolutions in rotor_revolution_levels {
            let observation = solve_pc_fan_cfd_rotating_wall_transient(
                config,
                design,
                reference_time_step,
                rotor_revolutions,
            )?;
            samples.push(PcFanCfdIndependenceSample {
                level: rotor_revolutions,
                airflow_cfm: observation.airflow_cfm,
                pressure_drop_pa: observation.pressure_drop_pa,
                axial_loading_n: observation.axial_loading_n,
                residual: observation.residual,
                relative_mass_imbalance: observation.conservation.relative_mass_imbalance,
                relative_momentum_imbalance: observation.conservation.relative_momentum_imbalance,
                relative_pressure_closure: observation.conservation.relative_pressure_closure,
                relative_energy_imbalance: observation.conservation.relative_energy_imbalance,
                converged: observation.converged,
            });
        }
        studies.push(PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::RotorRevolutions,
            samples,
        });
    }

    evaluate_pc_fan_cfd_independence(&independence_config, &studies).map_err(Into::into)
}

fn validate_transient_time_grid(
    config: &PcFanConfig,
    time_step_s: f64,
    rotor_revolutions: f64,
) -> Result<(), PcFanCfdProductionIndependenceError> {
    let angular_speed = std::f64::consts::TAU * config.operating_point.rpm / 60.0;
    if !angular_speed.is_finite() || angular_speed.abs() <= f64::EPSILON {
        return Err(PcFanCfdProductionIndependenceError::Transient(
            PcFanCfdTransientError::InvalidRotorRevolutions,
        ));
    }
    let duration_s = rotor_revolutions * std::f64::consts::TAU / angular_speed.abs();
    let step_count = duration_s / time_step_s;
    let rounded_step_count = step_count.round();
    if !step_count.is_finite()
        || (step_count - rounded_step_count).abs() > 1.0e-9 * step_count.abs().max(1.0)
    {
        return Err(PcFanCfdProductionIndependenceError::NonIntegralTimeGrid {
            time_step_s,
            rotor_revolutions,
            step_count,
        });
    }
    if rounded_step_count < 2.0 {
        return Err(PcFanCfdProductionIndependenceError::TooFewTimeSteps {
            time_step_s,
            rotor_revolutions,
            step_count,
        });
    }
    Ok(())
}

fn validate_level_ladder(
    config: &PcFanCfdIndependenceConfig,
    axis: PcFanCfdIndependenceAxis,
    levels: &[f64],
) -> Result<(), PcFanCfdIndependenceError> {
    if levels.len() < config.min_levels {
        return Err(PcFanCfdIndependenceError::TooFewSamples {
            axis,
            actual: levels.len(),
            required: config.min_levels,
        });
    }
    for (index, &level) in levels.iter().enumerate() {
        if !level.is_finite()
            || level <= 0.0
            || (axis.requires_integer_level() && level.fract() != 0.0)
        {
            return Err(PcFanCfdIndependenceError::InvalidSampleLevel { axis, index, level });
        }
        if index > 0 {
            let previous = levels[index - 1];
            let monotonic = if axis.levels_increase() {
                level > previous
            } else {
                level < previous
            };
            if !monotonic {
                return Err(PcFanCfdIndependenceError::NonMonotonicLevels {
                    axis,
                    index,
                    previous,
                    current: level,
                });
            }
        }
    }
    Ok(())
}

fn compare_study(
    config: &PcFanCfdIndependenceConfig,
    study: &PcFanCfdIndependenceStudy,
) -> Result<PcFanCfdIndependenceStudyReport, PcFanCfdIndependenceError> {
    if study.samples.len() < config.min_levels {
        return Err(PcFanCfdIndependenceError::TooFewSamples {
            axis: study.axis,
            actual: study.samples.len(),
            required: config.min_levels,
        });
    }

    for (index, sample) in study.samples.iter().enumerate() {
        if [
            sample.level,
            sample.airflow_cfm,
            sample.pressure_drop_pa,
            sample.axial_loading_n,
            sample.residual,
            sample.relative_mass_imbalance,
            sample.relative_momentum_imbalance,
            sample.relative_pressure_closure,
            sample.relative_energy_imbalance,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(PcFanCfdIndependenceError::NonFiniteSample {
                axis: study.axis,
                index,
            });
        }
        for (metric, value, tolerance) in [
            (
                "mass imbalance",
                sample.relative_mass_imbalance,
                config.max_relative_mass_imbalance,
            ),
            (
                "momentum imbalance",
                sample.relative_momentum_imbalance,
                config.max_relative_momentum_imbalance,
            ),
            (
                "pressure closure",
                sample.relative_pressure_closure,
                config.max_relative_pressure_closure,
            ),
            (
                "energy imbalance",
                sample.relative_energy_imbalance,
                config.max_relative_energy_imbalance,
            ),
        ] {
            if value < 0.0 || value > tolerance {
                return Err(PcFanCfdIndependenceError::ConservationViolation {
                    axis: study.axis,
                    index,
                    metric,
                    value,
                    tolerance,
                });
            }
        }
        if sample.level <= 0.0
            || (study.axis.requires_integer_level() && sample.level.fract() != 0.0)
        {
            return Err(PcFanCfdIndependenceError::InvalidSampleLevel {
                axis: study.axis,
                index,
                level: sample.level,
            });
        }
        if index > 0 {
            let previous = &study.samples[index - 1];
            let monotonic = if study.axis.levels_increase() {
                sample.level > previous.level
            } else {
                sample.level < previous.level
            };
            if !monotonic {
                return Err(PcFanCfdIndependenceError::NonMonotonicLevels {
                    axis: study.axis,
                    index,
                    previous: previous.level,
                    current: sample.level,
                });
            }
        }
    }

    let previous = &study.samples[study.samples.len() - 2];
    let final_sample = study
        .samples
        .last()
        .expect("minimum sample count validated");
    let airflow_relative_change = relative_change(final_sample.airflow_cfm, previous.airflow_cfm);
    let pressure_relative_change = relative_change(
        final_sample.pressure_drop_pa.abs(),
        previous.pressure_drop_pa.abs(),
    );
    let axial_loading_relative_change = relative_change(
        final_sample.axial_loading_n.abs(),
        previous.axial_loading_n.abs(),
    );
    let all_samples_converged = study.samples.iter().all(|sample| sample.converged);
    let converged = all_samples_converged
        && final_sample.residual <= config.max_final_residual
        && airflow_relative_change <= config.relative_tolerance
        && pressure_relative_change <= config.relative_tolerance
        && axial_loading_relative_change <= config.relative_tolerance;

    Ok(PcFanCfdIndependenceStudyReport {
        axis: study.axis,
        sample_count: study.samples.len(),
        first_level: study.samples[0].level,
        final_level: final_sample.level,
        airflow_relative_change,
        pressure_relative_change,
        axial_loading_relative_change,
        final_residual: final_sample.residual,
        all_samples_converged,
        converged,
    })
}

fn relative_change(current: f64, previous: f64) -> f64 {
    (current - previous).abs() / current.abs().max(previous.abs()).max(f64::MIN_POSITIVE)
}

/// One checked adjoint result in a mesh, time-step, revolution, or nonlinear
/// iteration independence ladder.
///
/// The gradient must be produced by the analytic resolved-flow adjoint path.
/// Diagnostic finite differences are intentionally not admissible evidence for
/// this contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdAdjointIndependenceSample {
    /// Resolution value in the units of [`PcFanCfdIndependenceAxis`].
    pub level: f64,
    pub design_gradient: Vec<f64>,
    pub state_residual_norm: f64,
    pub adjoint_residual_norm: f64,
    pub provenance: String,
    pub converged: bool,
}

/// Input observations for one adjoint independence ladder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdAdjointIndependenceStudy {
    pub axis: PcFanCfdIndependenceAxis,
    pub samples: Vec<PcFanCfdAdjointIndependenceSample>,
}

/// Checked tolerances and required axes for adjoint-independence evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdAdjointIndependenceConfig {
    pub relative_gradient_tolerance: f64,
    pub min_levels: usize,
    #[serde(default = "default_max_adjoint_state_residual")]
    pub max_state_residual_norm: f64,
    #[serde(default = "default_max_adjoint_residual")]
    pub max_adjoint_residual_norm: f64,
    /// Minimum cosine similarity between adjacent analytic gradients.
    #[serde(default = "default_min_gradient_cosine")]
    pub min_gradient_cosine: f64,
    /// Defaults to the four independent axes used by resolved-flow promotion.
    pub required_axes: Vec<PcFanCfdIndependenceAxis>,
}

fn default_max_adjoint_state_residual() -> f64 {
    1.0e-3
}

fn default_max_adjoint_residual() -> f64 {
    1.0e-3
}

fn default_min_gradient_cosine() -> f64 {
    0.99
}

impl Default for PcFanCfdAdjointIndependenceConfig {
    fn default() -> Self {
        Self {
            relative_gradient_tolerance: 0.05,
            min_levels: 2,
            max_state_residual_norm: default_max_adjoint_state_residual(),
            max_adjoint_residual_norm: default_max_adjoint_residual(),
            min_gradient_cosine: default_min_gradient_cosine(),
            required_axes: vec![
                PcFanCfdIndependenceAxis::Mesh,
                PcFanCfdIndependenceAxis::TimeStep,
                PcFanCfdIndependenceAxis::RotorRevolutions,
                PcFanCfdIndependenceAxis::NonlinearIterations,
            ],
        }
    }
}

/// Typed input and evidence errors for adjoint-independence evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcFanCfdAdjointIndependenceError {
    NoRequiredAxes,
    DuplicateRequiredAxis(PcFanCfdIndependenceAxis),
    DuplicateStudyAxis(PcFanCfdIndependenceAxis),
    UnexpectedStudyAxis(PcFanCfdIndependenceAxis),
    MissingStudyAxis(PcFanCfdIndependenceAxis),
    InvalidGradientTolerance,
    InvalidStateResidualTolerance,
    InvalidAdjointResidualTolerance,
    InvalidGradientCosineThreshold,
    MinLevelsTooSmall,
    TooFewSamples {
        axis: PcFanCfdIndependenceAxis,
        actual: usize,
        required: usize,
    },
    NonFiniteSample {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
    },
    EmptyGradient {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
    },
    ZeroGradient {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
    },
    GradientDimensionMismatch {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        expected: usize,
        actual: usize,
    },
    NonAnalyticProvenance {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
    },
    InvalidResidualNorm {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        metric: &'static str,
        value: f64,
    },
    InvalidSampleLevel {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        level: f64,
    },
    NonMonotonicLevels {
        axis: PcFanCfdIndependenceAxis,
        index: usize,
        previous: f64,
        current: f64,
    },
}

impl fmt::Display for PcFanCfdAdjointIndependenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRequiredAxes => {
                write!(
                    formatter,
                    "PC-fan adjoint independence requires at least one axis"
                )
            }
            Self::DuplicateRequiredAxis(axis) => write!(
                formatter,
                "PC-fan adjoint independence repeats required {} axis",
                axis.label()
            ),
            Self::DuplicateStudyAxis(axis) => write!(
                formatter,
                "PC-fan adjoint independence repeats {} study",
                axis.label()
            ),
            Self::UnexpectedStudyAxis(axis) => write!(
                formatter,
                "PC-fan adjoint independence received unexpected {} study",
                axis.label()
            ),
            Self::MissingStudyAxis(axis) => write!(
                formatter,
                "PC-fan adjoint independence is missing {} study",
                axis.label()
            ),
            Self::InvalidGradientTolerance => write!(
                formatter,
                "PC-fan adjoint independence relative_gradient_tolerance must be finite and non-negative"
            ),
            Self::InvalidStateResidualTolerance => write!(
                formatter,
                "PC-fan adjoint independence max_state_residual_norm must be finite and non-negative"
            ),
            Self::InvalidAdjointResidualTolerance => write!(
                formatter,
                "PC-fan adjoint independence max_adjoint_residual_norm must be finite and non-negative"
            ),
            Self::InvalidGradientCosineThreshold => write!(
                formatter,
                "PC-fan adjoint independence min_gradient_cosine must be finite and in [-1, 1]"
            ),
            Self::MinLevelsTooSmall => write!(
                formatter,
                "PC-fan adjoint independence min_levels must be at least two"
            ),
            Self::TooFewSamples {
                axis,
                actual,
                required,
            } => write!(
                formatter,
                "PC-fan adjoint independence {} study has {actual} samples; requires {required}",
                axis.label()
            ),
            Self::NonFiniteSample { axis, index } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} contains a non-finite value",
                axis.label()
            ),
            Self::EmptyGradient { axis, index } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} has an empty gradient",
                axis.label()
            ),
            Self::ZeroGradient { axis, index } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} has a zero gradient",
                axis.label()
            ),
            Self::GradientDimensionMismatch {
                axis,
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} has gradient dimension {actual}; expected {expected}",
                axis.label()
            ),
            Self::NonAnalyticProvenance { axis, index } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} lacks analytic adjoint provenance",
                axis.label()
            ),
            Self::InvalidResidualNorm {
                axis,
                index,
                metric,
                value,
            } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} has invalid {metric} {value}",
                axis.label()
            ),
            Self::InvalidSampleLevel { axis, index, level } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} has invalid level {level}",
                axis.label()
            ),
            Self::NonMonotonicLevels {
                axis,
                index,
                previous,
                current,
            } => write!(
                formatter,
                "PC-fan adjoint independence {} sample {index} is not monotonic ({previous} -> {current})",
                axis.label()
            ),
        }
    }
}

impl std::error::Error for PcFanCfdAdjointIndependenceError {}

impl PcFanCfdAdjointIndependenceConfig {
    fn validate(&self) -> Result<(), PcFanCfdAdjointIndependenceError> {
        if self.required_axes.is_empty() {
            return Err(PcFanCfdAdjointIndependenceError::NoRequiredAxes);
        }
        if !self.relative_gradient_tolerance.is_finite() || self.relative_gradient_tolerance < 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::InvalidGradientTolerance);
        }
        if !self.max_state_residual_norm.is_finite() || self.max_state_residual_norm < 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::InvalidStateResidualTolerance);
        }
        if !self.max_adjoint_residual_norm.is_finite() || self.max_adjoint_residual_norm < 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::InvalidAdjointResidualTolerance);
        }
        if !self.min_gradient_cosine.is_finite()
            || !(-1.0..=1.0).contains(&self.min_gradient_cosine)
        {
            return Err(PcFanCfdAdjointIndependenceError::InvalidGradientCosineThreshold);
        }
        if self.min_levels < 2 {
            return Err(PcFanCfdAdjointIndependenceError::MinLevelsTooSmall);
        }
        for (index, axis) in self.required_axes.iter().copied().enumerate() {
            if self.required_axes[..index].contains(&axis) {
                return Err(PcFanCfdAdjointIndependenceError::DuplicateRequiredAxis(
                    axis,
                ));
            }
        }
        Ok(())
    }
}

/// Per-axis comparison generated by [`evaluate_pc_fan_cfd_adjoint_independence`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdAdjointIndependenceStudyReport {
    pub axis: PcFanCfdIndependenceAxis,
    pub sample_count: usize,
    pub gradient_dimension: usize,
    pub first_level: f64,
    pub final_level: f64,
    pub max_adjacent_gradient_relative_change: f64,
    pub min_adjacent_gradient_cosine: f64,
    pub final_state_residual_norm: f64,
    pub final_adjoint_residual_norm: f64,
    pub all_samples_within_residual_budgets: bool,
    pub all_samples_converged: bool,
    pub converged: bool,
}

/// Complete adjoint-independence evidence report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdAdjointIndependenceReport {
    pub studies: Vec<PcFanCfdAdjointIndependenceStudyReport>,
    pub required_axes: Vec<PcFanCfdIndependenceAxis>,
    pub all_samples_finite: bool,
    pub all_samples_within_residual_budgets: bool,
    pub all_samples_converged: bool,
    pub converged: bool,
}

impl PcFanCfdAdjointIndependenceReport {
    pub fn is_valid(&self) -> bool {
        !self.studies.is_empty()
            && self.studies.len() == self.required_axes.len()
            && self.all_samples_finite
            && self.all_samples_within_residual_budgets
            && self.all_samples_converged
            && self.converged
    }
}

/// Validate analytic resolved-CFD adjoint gradients across independent ladders.
pub fn evaluate_pc_fan_cfd_adjoint_independence(
    config: &PcFanCfdAdjointIndependenceConfig,
    studies: &[PcFanCfdAdjointIndependenceStudy],
) -> Result<PcFanCfdAdjointIndependenceReport, PcFanCfdAdjointIndependenceError> {
    config.validate()?;

    for (index, study) in studies.iter().enumerate() {
        if !config.required_axes.contains(&study.axis) {
            return Err(PcFanCfdAdjointIndependenceError::UnexpectedStudyAxis(
                study.axis,
            ));
        }
        if studies[..index]
            .iter()
            .any(|previous| previous.axis == study.axis)
        {
            return Err(PcFanCfdAdjointIndependenceError::DuplicateStudyAxis(
                study.axis,
            ));
        }
    }

    let mut reports = Vec::with_capacity(config.required_axes.len());
    for axis in config.required_axes.iter().copied() {
        let study = studies
            .iter()
            .find(|study| study.axis == axis)
            .ok_or(PcFanCfdAdjointIndependenceError::MissingStudyAxis(axis))?;
        reports.push(compare_adjoint_study(config, study)?);
    }

    let all_samples_finite = true;
    let all_samples_within_residual_budgets = reports
        .iter()
        .all(|report| report.all_samples_within_residual_budgets);
    let all_samples_converged = reports.iter().all(|report| report.all_samples_converged);
    let converged = reports.iter().all(|report| report.converged);
    Ok(PcFanCfdAdjointIndependenceReport {
        studies: reports,
        required_axes: config.required_axes.clone(),
        all_samples_finite,
        all_samples_within_residual_budgets,
        all_samples_converged,
        converged,
    })
}

fn compare_adjoint_study(
    config: &PcFanCfdAdjointIndependenceConfig,
    study: &PcFanCfdAdjointIndependenceStudy,
) -> Result<PcFanCfdAdjointIndependenceStudyReport, PcFanCfdAdjointIndependenceError> {
    if study.samples.len() < config.min_levels {
        return Err(PcFanCfdAdjointIndependenceError::TooFewSamples {
            axis: study.axis,
            actual: study.samples.len(),
            required: config.min_levels,
        });
    }

    let expected_dimension = study.samples[0].design_gradient.len();
    let mut all_samples_converged = true;
    for (index, sample) in study.samples.iter().enumerate() {
        if !sample.level.is_finite()
            || !sample.state_residual_norm.is_finite()
            || !sample.adjoint_residual_norm.is_finite()
            || sample
                .design_gradient
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(PcFanCfdAdjointIndependenceError::NonFiniteSample {
                axis: study.axis,
                index,
            });
        }
        if sample.level <= 0.0
            || (axis_requires_integer_level(study.axis) && sample.level.fract() != 0.0)
        {
            return Err(PcFanCfdAdjointIndependenceError::InvalidSampleLevel {
                axis: study.axis,
                index,
                level: sample.level,
            });
        }
        if sample.design_gradient.is_empty() {
            return Err(PcFanCfdAdjointIndependenceError::EmptyGradient {
                axis: study.axis,
                index,
            });
        }
        if sample.design_gradient.len() != expected_dimension {
            return Err(
                PcFanCfdAdjointIndependenceError::GradientDimensionMismatch {
                    axis: study.axis,
                    index,
                    expected: expected_dimension,
                    actual: sample.design_gradient.len(),
                },
            );
        }
        let gradient_norm = stable_gradient_norm(&sample.design_gradient);
        if !gradient_norm.is_finite() {
            return Err(PcFanCfdAdjointIndependenceError::NonFiniteSample {
                axis: study.axis,
                index,
            });
        }
        if gradient_norm == 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::ZeroGradient {
                axis: study.axis,
                index,
            });
        }
        if !has_analytic_adjoint_provenance(&sample.provenance) {
            return Err(PcFanCfdAdjointIndependenceError::NonAnalyticProvenance {
                axis: study.axis,
                index,
            });
        }
        if sample.state_residual_norm < 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::InvalidResidualNorm {
                axis: study.axis,
                index,
                metric: "state residual norm",
                value: sample.state_residual_norm,
            });
        }
        if sample.adjoint_residual_norm < 0.0 {
            return Err(PcFanCfdAdjointIndependenceError::InvalidResidualNorm {
                axis: study.axis,
                index,
                metric: "adjoint residual norm",
                value: sample.adjoint_residual_norm,
            });
        }
        if index > 0 {
            let previous = &study.samples[index - 1];
            let monotonic = if study.axis.levels_increase() {
                sample.level > previous.level
            } else {
                sample.level < previous.level
            };
            if !monotonic {
                return Err(PcFanCfdAdjointIndependenceError::NonMonotonicLevels {
                    axis: study.axis,
                    index,
                    previous: previous.level,
                    current: sample.level,
                });
            }
        }
        all_samples_converged &= sample.converged;
    }

    let mut max_relative_change = 0.0_f64;
    let mut min_cosine = 1.0_f64;
    for adjacent in study.samples.windows(2) {
        let relative_change = stable_gradient_relative_change(
            &adjacent[1].design_gradient,
            &adjacent[0].design_gradient,
        );
        let cosine =
            stable_gradient_cosine(&adjacent[1].design_gradient, &adjacent[0].design_gradient);
        max_relative_change = max_relative_change.max(relative_change);
        min_cosine = min_cosine.min(cosine);
    }

    let final_sample = study
        .samples
        .last()
        .expect("minimum sample count validated");
    let all_samples_within_residual_budgets = study.samples.iter().all(|sample| {
        sample.state_residual_norm <= config.max_state_residual_norm
            && sample.adjoint_residual_norm <= config.max_adjoint_residual_norm
    });
    let converged = all_samples_converged
        && all_samples_within_residual_budgets
        && max_relative_change <= config.relative_gradient_tolerance
        && min_cosine >= config.min_gradient_cosine;
    Ok(PcFanCfdAdjointIndependenceStudyReport {
        axis: study.axis,
        sample_count: study.samples.len(),
        gradient_dimension: expected_dimension,
        first_level: study.samples[0].level,
        final_level: final_sample.level,
        max_adjacent_gradient_relative_change: max_relative_change,
        min_adjacent_gradient_cosine: min_cosine,
        final_state_residual_norm: final_sample.state_residual_norm,
        final_adjoint_residual_norm: final_sample.adjoint_residual_norm,
        all_samples_within_residual_budgets,
        all_samples_converged,
        converged,
    })
}

fn axis_requires_integer_level(axis: PcFanCfdIndependenceAxis) -> bool {
    axis.requires_integer_level()
}

fn has_analytic_adjoint_provenance(provenance: &str) -> bool {
    let normalized = provenance.trim().to_ascii_lowercase();
    !normalized.is_empty()
        && normalized.contains("analytic")
        && ![
            "finite-difference",
            "finite_difference",
            "finite difference",
            "black-box",
            "black_box",
            "black box",
            "diagnostic",
            "probe",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn stable_gradient_norm(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let normalized_sum = values
        .iter()
        .map(|value| (value / scale).powi(2))
        .sum::<f64>();
    scale * normalized_sum.sqrt()
}

fn stable_gradient_relative_change(current: &[f64], previous: &[f64]) -> f64 {
    let scale = current
        .iter()
        .chain(previous)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let difference = current
        .iter()
        .zip(previous)
        .map(|(current, previous)| (current / scale - previous / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    let current_norm = stable_gradient_norm(current) / scale;
    let previous_norm = stable_gradient_norm(previous) / scale;
    difference / (0.5 * (current_norm + previous_norm)).max(f64::MIN_POSITIVE)
}

fn stable_gradient_cosine(left: &[f64], right: &[f64]) -> f64 {
    let left_scale = left.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let right_scale = right
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let left_norm = left
        .iter()
        .map(|value| (value / left_scale).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| (value / right_scale).powi(2))
        .sum::<f64>()
        .sqrt();
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left / left_scale) * (right / right_scale))
        .sum::<f64>();
    (dot / (left_norm * right_norm)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(level: f64, scale: f64, converged: bool) -> PcFanCfdIndependenceSample {
        PcFanCfdIndependenceSample {
            level,
            airflow_cfm: 20.0 * scale,
            pressure_drop_pa: 80.0 * scale,
            axial_loading_n: 1.5 * scale,
            residual: 1.0e-5,
            relative_mass_imbalance: 0.0,
            relative_momentum_imbalance: 0.0,
            relative_pressure_closure: 0.0,
            relative_energy_imbalance: 0.0,
            converged,
        }
    }

    fn passing_studies() -> Vec<PcFanCfdIndependenceStudy> {
        vec![
            PcFanCfdIndependenceStudy {
                axis: PcFanCfdIndependenceAxis::Mesh,
                samples: vec![sample(2.0, 0.99, true), sample(3.0, 1.0, true)],
            },
            PcFanCfdIndependenceStudy {
                axis: PcFanCfdIndependenceAxis::TimeStep,
                samples: vec![sample(0.002, 0.99, true), sample(0.001, 1.0, true)],
            },
            PcFanCfdIndependenceStudy {
                axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                samples: vec![sample(3.0, 0.99, true), sample(6.0, 1.0, true)],
            },
            PcFanCfdIndependenceStudy {
                axis: PcFanCfdIndependenceAxis::NonlinearIterations,
                samples: vec![sample(20.0, 0.99, true), sample(40.0, 1.0, true)],
            },
        ]
    }

    #[test]
    fn independence_report_requires_all_axes_and_passes_finite_ladders() {
        let report = evaluate_pc_fan_cfd_independence(
            &PcFanCfdIndependenceConfig::default(),
            &passing_studies(),
        )
        .expect("valid independence ladders");

        assert!(report.is_valid());
        assert_eq!(report.studies.len(), 4);
        assert!(report.studies.iter().all(|study| study.sample_count == 2));
        assert!(
            report
                .studies
                .iter()
                .all(|study| study.airflow_relative_change < 0.02)
        );
    }

    #[test]
    fn independence_rejects_missing_duplicate_and_unexpected_axes() {
        let mut studies = passing_studies();
        studies.pop();
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::MissingStudyAxis(
                PcFanCfdIndependenceAxis::NonlinearIterations
            ))
        );

        let mut studies = passing_studies();
        studies.push(studies[0].clone());
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::DuplicateStudyAxis(
                PcFanCfdIndependenceAxis::Mesh
            ))
        );

        let mut studies = passing_studies();
        studies[0].axis = PcFanCfdIndependenceAxis::TimeStep;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::DuplicateStudyAxis(
                PcFanCfdIndependenceAxis::TimeStep
            ))
        );
    }

    #[test]
    fn independence_rejects_invalid_configuration_and_samples() {
        let mut config = PcFanCfdIndependenceConfig::default();
        config.relative_tolerance = f64::NAN;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&config, &passing_studies()),
            Err(PcFanCfdIndependenceError::InvalidRelativeTolerance)
        );

        let mut studies = passing_studies();
        studies[0].samples[1].level = 2.5;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::InvalidSampleLevel {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 1,
                level: 2.5,
            })
        );

        let mut studies = passing_studies();
        studies[1].samples[1].level = 0.003;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::NonMonotonicLevels {
                axis: PcFanCfdIndependenceAxis::TimeStep,
                index: 1,
                previous: 0.002,
                current: 0.003,
            })
        );

        let mut studies = passing_studies();
        studies[2].samples[0].airflow_cfm = f64::INFINITY;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::NonFiniteSample {
                axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                index: 0,
            })
        );
    }

    #[test]
    fn independence_reports_failed_observable_and_solver_gates() {
        let mut studies = passing_studies();
        studies[0].samples[1].airflow_cfm *= 1.20;
        studies[1].samples[1].converged = false;
        let report =
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies)
                .expect("failed evidence is reportable");

        assert!(!report.is_valid());
        assert!(!report.converged);
        assert!(!report.all_samples_converged);
        assert!(!report.studies[0].converged);
        assert!(!report.studies[1].converged);
    }

    #[test]
    fn independence_rejects_nonmonotonic_and_short_ladders() {
        let mut studies = passing_studies();
        studies[3].samples[1].level = 10.0;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies),
            Err(PcFanCfdIndependenceError::NonMonotonicLevels {
                axis: PcFanCfdIndependenceAxis::NonlinearIterations,
                index: 1,
                previous: 20.0,
                current: 10.0,
            })
        );

        let mut config = PcFanCfdIndependenceConfig::default();
        config.min_levels = 3;
        assert_eq!(
            evaluate_pc_fan_cfd_independence(&config, &passing_studies()),
            Err(PcFanCfdIndependenceError::TooFewSamples {
                axis: PcFanCfdIndependenceAxis::Mesh,
                actual: 2,
                required: 3,
            })
        );
    }

    #[test]
    fn independence_report_rejects_agreeing_levels_with_large_final_residual() {
        let mut config = PcFanCfdIndependenceConfig::default();
        config.max_final_residual = 1.0e-6;

        let report = evaluate_pc_fan_cfd_independence(&config, &passing_studies())
            .expect("finite independence samples should still produce a report");

        assert!(report.all_samples_finite);
        assert!(report.all_samples_converged);
        assert!(!report.converged);
        assert!(!report.is_valid());
        assert!(
            report
                .studies
                .iter()
                .all(|study| study.final_residual == 1.0e-5)
        );
    }

    #[test]
    fn independence_config_rejects_invalid_final_residual_budget() {
        let mut config = PcFanCfdIndependenceConfig::default();
        config.max_final_residual = f64::NAN;

        assert_eq!(
            evaluate_pc_fan_cfd_independence(&config, &passing_studies()),
            Err(PcFanCfdIndependenceError::InvalidFinalResidualTolerance)
        );
    }

    #[test]
    fn independence_config_rejects_invalid_conservation_budget() {
        let mut config = PcFanCfdIndependenceConfig::default();
        config.max_relative_mass_imbalance = f64::NAN;

        assert_eq!(
            evaluate_pc_fan_cfd_independence(&config, &passing_studies()),
            Err(PcFanCfdIndependenceError::InvalidConservationTolerance {
                metric: "mass imbalance",
            })
        );
    }

    #[test]
    fn independence_report_rejects_sample_over_conservation_budget() {
        let mut studies = passing_studies();
        studies[0].samples[1].relative_mass_imbalance =
            PcFanCfdIndependenceConfig::default().max_relative_mass_imbalance + 1.0;

        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies,),
            Err(PcFanCfdIndependenceError::ConservationViolation {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 1,
                metric: "mass imbalance",
                value: 1.05,
                tolerance: 0.05,
            })
        );
    }

    #[test]
    fn independence_report_rejects_nonfinite_conservation_metric() {
        let mut studies = passing_studies();
        studies[0].samples[0].relative_energy_imbalance = f64::NAN;

        assert_eq!(
            evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies,),
            Err(PcFanCfdIndependenceError::NonFiniteSample {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 0,
            })
        );
    }

    #[test]
    fn production_mesh_independence_rejects_unavailable_transient_axes() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::Mesh];
        independence
            .required_axes
            .push(PcFanCfdIndependenceAxis::TimeStep);

        assert_eq!(
            evaluate_pc_fan_cfd_mesh_independence(
                &config,
                &design,
                PcFanCfdMeshConvergenceConfig::default(),
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(
                PcFanCfdIndependenceAxis::TimeStep
            ))
        );
    }

    #[test]
    fn production_mesh_independence_validates_mesh_budget_before_solving() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::Mesh];
        let mut mesh = PcFanCfdMeshConvergenceConfig::default();
        mesh.start_cells = 0;

        assert_eq!(
            evaluate_pc_fan_cfd_mesh_independence(&config, &design, mesh, &independence),
            Err(PcFanCfdProductionIndependenceError::MeshConvergence(
                PcFanCfdMeshConvergenceConfigError::StartCellsMustBePositive
            ))
        );
    }

    #[test]
    fn production_nonlinear_independence_rejects_wrong_axis_and_zero_level() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::Mesh];
        assert_eq!(
            evaluate_pc_fan_cfd_nonlinear_independence(&config, &design, &[1, 2], &independence,),
            Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(
                PcFanCfdIndependenceAxis::Mesh
            ))
        );

        independence.required_axes = vec![PcFanCfdIndependenceAxis::NonlinearIterations];
        assert_eq!(
            evaluate_pc_fan_cfd_nonlinear_independence(&config, &design, &[0, 2], &independence,),
            Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::InvalidSampleLevel {
                    axis: PcFanCfdIndependenceAxis::NonlinearIterations,
                    index: 0,
                    level: 0.0,
                }
            ))
        );
    }

    #[test]
    fn transient_independence_rejects_reduced_order_without_solving() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::TimeStep];
        assert_eq!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[0.02, 0.01],
                &[1.0],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::Transient(
                PcFanCfdTransientError::UnsupportedFlowModel(PcFanFlowModel::ReducedOrder)
            ))
        );
    }

    #[test]
    fn transient_independence_rejects_non_integral_time_grid_before_solving() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::TimeStep];

        assert!(matches!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[3.0e-4, 2.0e-4],
                &[0.02],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::NonIntegralTimeGrid { .. })
        ));

        assert!(matches!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[2.0e-3, 1.0e-3],
                &[0.02],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::TooFewTimeSteps { .. })
        ));
    }

    #[test]
    fn transient_independence_rejects_mesh_axis_and_empty_required_ladder() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        independence.required_axes = vec![PcFanCfdIndependenceAxis::Mesh];
        assert_eq!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[0.02, 0.01],
                &[1.0],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::UnsupportedAxis(
                PcFanCfdIndependenceAxis::Mesh
            ))
        );
        independence.required_axes = vec![PcFanCfdIndependenceAxis::RotorRevolutions];
        assert_eq!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[0.02],
                &[],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                    actual: 0,
                    required: 2,
                }
            ))
        );
    }

    #[test]
    fn transient_independence_rejects_empty_complementary_ladder() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let design = config.default_design_variables();
        let mut independence = PcFanCfdIndependenceConfig::default();
        // A TimeStep study with no rotor-revolution reference must fail
        // closed instead of solving against a manufactured 1.0 revolution.
        independence.required_axes = vec![PcFanCfdIndependenceAxis::TimeStep];
        assert_eq!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[0.02, 0.01],
                &[],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::RotorRevolutions,
                    actual: 0,
                    required: 1,
                }
            ))
        );
        // Symmetrically, a RotorRevolutions study with no time-step
        // reference must fail closed instead of using 0.01 s.
        independence.required_axes = vec![PcFanCfdIndependenceAxis::RotorRevolutions];
        assert_eq!(
            evaluate_pc_fan_cfd_transient_independence(
                &config,
                &design,
                &[],
                &[1.0, 0.5],
                &independence,
            ),
            Err(PcFanCfdProductionIndependenceError::Independence(
                PcFanCfdIndependenceError::TooFewSamples {
                    axis: PcFanCfdIndependenceAxis::TimeStep,
                    actual: 0,
                    required: 1,
                }
            ))
        );
    }

    #[test]
    fn transient_runner_rejects_invalid_controls_before_mesh_construction() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let design = config.default_design_variables();
        config.cfd.nonlinear_tolerance = 0.0;
        assert_eq!(
            crate::cfd::solve_pc_fan_cfd_rotating_wall_transient(&config, &design, 0.01, 0.01),
            Err(crate::cfd::PcFanCfdTransientError::InvalidNonlinearTolerance { tolerance: 0.0 })
        );
        config.cfd.nonlinear_tolerance = 1.0e-4;
        config.cfd.max_relative_energy_imbalance = f64::NAN;
        assert!(matches!(
            crate::cfd::solve_pc_fan_cfd_rotating_wall_transient(&config, &design, 0.01, 0.01),
            Err(
                crate::cfd::PcFanCfdTransientError::InvalidConservationTolerance {
                    metric: "max_relative_energy_imbalance",
                    ..
                }
            )
        ));
        config.cfd.max_relative_energy_imbalance = 10.0;
        assert_eq!(
            crate::cfd::solve_pc_fan_cfd_rotating_wall_transient(&config, &design, 0.0, 1.0),
            Err(crate::cfd::PcFanCfdTransientError::InvalidTimeStep)
        );
        assert_eq!(
            crate::cfd::solve_pc_fan_cfd_rotating_wall_transient(&config, &design, 0.01, 0.0),
            Err(crate::cfd::PcFanCfdTransientError::InvalidRotorRevolutions)
        );
    }

    #[test]
    fn transient_runner_rejects_window_residual_above_budget() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.linear_solver_max_iterations = 100;
        config.cfd.linear_solver_tolerance = 1.0e-3;
        config.cfd.nonlinear_tolerance = 1.0e-16;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        let design = config.default_design_variables();

        assert!(matches!(
            crate::cfd::solve_pc_fan_cfd_rotating_wall_transient(&config, &design, 0.01, 0.02),
            Err(crate::cfd::PcFanCfdTransientError::NonConvergedObservation { .. })
        ));
    }

    #[test]
    fn transient_runner_publishes_finite_rotating_wall_window() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cuda;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.linear_solver_max_iterations = 100;
        config.cfd.linear_solver_tolerance = 1.0e-3;
        // The 2x2x2 transient fixture is deliberately coarse; keep its
        // conservation budgets explicit so this test isolates finite-window
        // residual admission rather than mesh conservation quality.
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        config.cfd.nonlinear_tolerance = 10.0;
        let design = config.default_design_variables();
        let trajectory = crate::cfd::solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
            &config, &design, 0.01, 0.02,
        )
        .expect("bounded resolved transient smoke should solve");
        let observation = trajectory.observation;
        assert!(trajectory.steps.len() >= 2);
        assert!(
            trajectory
                .steps
                .iter()
                .all(|step| step.state_jacobian.dimension == step.state.len())
        );
        assert!(
            trajectory
                .steps
                .iter()
                .all(|step| step.actual_backend == PcFanCfdBackend::Cpu)
        );
        assert!(
            trajectory
                .steps
                .iter()
                .all(|step| step.residual_norm.is_finite())
        );
        assert!(
            trajectory.steps[1]
                .history_jacobians
                .iter()
                .any(|history| history.previous_step == 0)
        );
        let mut objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();
        let last = objective_gradients.last_mut().unwrap();
        last[0] = 1.0;
        let adjoint_config = crate::resolved_adjoint::PcFanResolvedAdjointConfig {
            // This 2x2x2 fixture checks finite trajectory/adjoint wiring. Its
            // coarse transported residual is intentionally bounded rather than
            // representative of production convergence quality.
            max_state_residual: 1.0,
            ..crate::resolved_adjoint::PcFanResolvedAdjointConfig::default()
        };
        let adjoint = crate::cfd::solve_pc_fan_cfd_resolved_transient_state_adjoint(
            &adjoint_config,
            &config,
            &trajectory,
            &objective_gradients,
        )
        .expect("production transient state adjoint should solve");
        assert_eq!(adjoint.adjoints.len(), trajectory.steps.len());
        assert!(
            adjoint
                .adjoints
                .iter()
                .flat_map(|values| values.iter())
                .all(|value| value.is_finite())
        );
        assert!(adjoint.design_gradient.is_empty());
        assert!(observation.final_time_s >= 0.002);
        assert!(observation.airflow_cfm.is_finite());
        assert!(observation.pressure_drop_pa.is_finite());
        assert!(observation.axial_loading_n.is_finite());
        assert!(observation.residual.is_finite());
        assert!(observation.converged);
    }
}

#[cfg(test)]
mod adjoint_independence_tests {
    use super::*;

    fn sample(
        level: f64,
        gradient: &[f64],
        state_residual_norm: f64,
        adjoint_residual_norm: f64,
        provenance: &str,
        converged: bool,
    ) -> PcFanCfdAdjointIndependenceSample {
        PcFanCfdAdjointIndependenceSample {
            level,
            design_gradient: gradient.to_vec(),
            state_residual_norm,
            adjoint_residual_norm,
            provenance: provenance.to_owned(),
            converged,
        }
    }

    fn study(axis: PcFanCfdIndependenceAxis, levels: &[f64]) -> PcFanCfdAdjointIndependenceStudy {
        PcFanCfdAdjointIndependenceStudy {
            axis,
            samples: levels
                .iter()
                .enumerate()
                .map(|(index, &level)| {
                    sample(
                        level,
                        &[1.0 + index as f64 * 0.01, -0.5],
                        1.0e-5,
                        2.0e-5,
                        "production:analytic-adjoint:v1",
                        true,
                    )
                })
                .collect(),
        }
    }

    fn config_for(axis: PcFanCfdIndependenceAxis) -> PcFanCfdAdjointIndependenceConfig {
        PcFanCfdAdjointIndependenceConfig {
            required_axes: vec![axis],
            ..PcFanCfdAdjointIndependenceConfig::default()
        }
    }

    #[test]
    fn adjoint_independence_accepts_finite_analytic_ladders() {
        let config = PcFanCfdAdjointIndependenceConfig {
            required_axes: vec![
                PcFanCfdIndependenceAxis::Mesh,
                PcFanCfdIndependenceAxis::TimeStep,
                PcFanCfdIndependenceAxis::NonlinearIterations,
            ],
            ..PcFanCfdAdjointIndependenceConfig::default()
        };
        let report = evaluate_pc_fan_cfd_adjoint_independence(
            &config,
            &[
                study(PcFanCfdIndependenceAxis::Mesh, &[1.0, 2.0]),
                study(PcFanCfdIndependenceAxis::TimeStep, &[0.02, 0.01]),
                study(PcFanCfdIndependenceAxis::NonlinearIterations, &[2.0, 4.0]),
            ],
        )
        .expect("valid analytic adjoint ladders");

        assert!(report.is_valid());
        assert_eq!(report.studies[0].gradient_dimension, 2);
        assert!(report.studies[0].min_adjacent_gradient_cosine > 0.99);
        assert!(
            report.studies[0]
                .max_adjacent_gradient_relative_change
                .is_finite()
        );
    }

    #[test]
    fn adjoint_independence_rejects_gradient_thresholds_without_promoting_report() {
        let mut config = config_for(PcFanCfdIndependenceAxis::Mesh);
        config.relative_gradient_tolerance = 0.01;
        config.min_gradient_cosine = 0.99;
        let study = PcFanCfdAdjointIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::Mesh,
            samples: vec![
                sample(
                    1.0,
                    &[1.0, 0.0],
                    1.0e-5,
                    1.0e-5,
                    "production:analytic-adjoint:v1",
                    true,
                ),
                sample(
                    2.0,
                    &[0.0, 1.0],
                    1.0e-5,
                    1.0e-5,
                    "production:analytic-adjoint:v1",
                    true,
                ),
            ],
        };

        let report = evaluate_pc_fan_cfd_adjoint_independence(&config, &[study])
            .expect("threshold rejection is reported, not malformed input");
        assert!(!report.is_valid());
        assert!(report.studies[0].max_adjacent_gradient_relative_change > 1.0);
        assert!(report.studies[0].min_adjacent_gradient_cosine.abs() < 1.0e-12);
    }

    #[test]
    fn adjoint_independence_rejects_malformed_gradient_dimensions_and_zero_gradients() {
        let config = config_for(PcFanCfdIndependenceAxis::Mesh);
        let malformed = PcFanCfdAdjointIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::Mesh,
            samples: vec![
                sample(
                    1.0,
                    &[1.0, 0.0],
                    0.0,
                    0.0,
                    "production:analytic-adjoint:v1",
                    true,
                ),
                sample(
                    2.0,
                    &[1.0],
                    0.0,
                    0.0,
                    "production:analytic-adjoint:v1",
                    true,
                ),
            ],
        };
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(&config, &[malformed]),
            Err(
                PcFanCfdAdjointIndependenceError::GradientDimensionMismatch {
                    axis: PcFanCfdIndependenceAxis::Mesh,
                    index: 1,
                    expected: 2,
                    actual: 1,
                }
            )
        );

        let zero = PcFanCfdAdjointIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::Mesh,
            samples: vec![
                sample(
                    1.0,
                    &[0.0, 0.0],
                    0.0,
                    0.0,
                    "production:analytic-adjoint:v1",
                    true,
                ),
                sample(
                    2.0,
                    &[1.0, 0.0],
                    0.0,
                    0.0,
                    "production:analytic-adjoint:v1",
                    true,
                ),
            ],
        };
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(&config, &[zero]),
            Err(PcFanCfdAdjointIndependenceError::ZeroGradient {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 0,
            })
        );
    }

    #[test]
    fn adjoint_independence_rejects_nonanalytic_provenance_and_invalid_levels() {
        let config = config_for(PcFanCfdIndependenceAxis::TimeStep);
        let nonanalytic = PcFanCfdAdjointIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::TimeStep,
            samples: vec![
                sample(
                    0.02,
                    &[1.0, 0.0],
                    0.0,
                    0.0,
                    "diagnostic:finite-difference-gradient",
                    true,
                ),
                sample(
                    0.01,
                    &[1.0, 0.0],
                    0.0,
                    0.0,
                    "diagnostic:finite-difference-gradient",
                    true,
                ),
            ],
        };
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(&config, &[nonanalytic]),
            Err(PcFanCfdAdjointIndependenceError::NonAnalyticProvenance {
                axis: PcFanCfdIndependenceAxis::TimeStep,
                index: 0,
            })
        );

        let mut invalid_level = study(PcFanCfdIndependenceAxis::Mesh, &[1.5, 2.0]);
        invalid_level.axis = PcFanCfdIndependenceAxis::Mesh;
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(
                &config_for(PcFanCfdIndependenceAxis::Mesh),
                &[invalid_level]
            ),
            Err(PcFanCfdAdjointIndependenceError::InvalidSampleLevel {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 0,
                level: 1.5,
            })
        );
    }

    #[test]
    fn adjoint_independence_rejects_invalid_configuration_and_nonconverged_residuals() {
        let mut config = config_for(PcFanCfdIndependenceAxis::Mesh);
        config.min_gradient_cosine = 1.1;
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(
                &config,
                &[study(PcFanCfdIndependenceAxis::Mesh, &[1.0, 2.0])]
            ),
            Err(PcFanCfdAdjointIndependenceError::InvalidGradientCosineThreshold)
        );

        let mut rejected = study(PcFanCfdIndependenceAxis::Mesh, &[1.0, 2.0]);
        rejected.samples[0].state_residual_norm = 1.0;
        rejected.samples[0].adjoint_residual_norm = 1.0;
        rejected.samples[0].converged = false;
        let report = evaluate_pc_fan_cfd_adjoint_independence(
            &config_for(PcFanCfdIndependenceAxis::Mesh),
            &[rejected],
        )
        .expect("residual/convergence budgets are report gates");
        assert!(!report.is_valid());
        assert!(!report.all_samples_within_residual_budgets);
        assert!(!report.all_samples_converged);

        let mut malformed = study(PcFanCfdIndependenceAxis::Mesh, &[1.0, 2.0]);
        malformed.samples[0].state_residual_norm = -1.0;
        assert_eq!(
            evaluate_pc_fan_cfd_adjoint_independence(
                &config_for(PcFanCfdIndependenceAxis::Mesh),
                &[malformed]
            ),
            Err(PcFanCfdAdjointIndependenceError::InvalidResidualNorm {
                axis: PcFanCfdIndependenceAxis::Mesh,
                index: 0,
                metric: "state residual norm",
                value: -1.0,
            })
        );
    }
}
