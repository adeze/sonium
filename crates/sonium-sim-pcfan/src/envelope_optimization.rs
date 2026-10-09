//! Robust reduced-order operating-envelope objective and optimizer.
//!
//! This module promotes the existing deterministic RPM/manufacturing sweep
//! into an optimization contract. It intentionally accepts only the
//! reduced-order model: a resolved-flow envelope optimizer still needs a
//! pressure/load boundary sweep and independent convergence evidence.

use crate::evaluation::{
    PcFanManufacturingOperatingEnvelopeError, evaluate_pc_fan,
    evaluate_pc_fan_manufacturing_operating_envelope,
};
use crate::model::{
    PcFanConfig, PcFanDesignVariables, PcFanEvaluation, PcFanFlowModel,
    PcFanManufacturingOperatingEnvelope,
};
use serde::{Deserialize, Serialize};
use sonium_solvers_optimization::OptimStatus;
use sonium_solvers_optimization::unconstrained::de::{
    DEConfigBuilder, LShadeConfig, Strategy, differential_evolution,
};
use std::cmp::Ordering;
use std::fmt;

/// Target and robustness policy for reduced-order RPM-envelope optimization.
///
/// Target vectors are indexed by `rpms` and are compared against every
/// manufacturing realization at that RPM. This makes the score a true
/// cross-product objective rather than a nominal-only point objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanOperatingEnvelopeObjectiveConfig {
    /// Strictly increasing positive RPM samples.
    pub rpms: Vec<f64>,
    pub target_airflow_cfm: Vec<f64>,
    pub target_pressure_rise_pa: Vec<f64>,
    #[serde(default)]
    pub target_acoustic_level_db: Option<Vec<f64>>,
    pub airflow_weight: f64,
    pub pressure_weight: f64,
    #[serde(default)]
    pub acoustic_weight: f64,
    /// Require at least one non-zero manufacturing tolerance axis.
    #[serde(default = "default_require_manufacturing_coverage")]
    pub require_manufacturing_coverage: bool,
}

fn default_require_manufacturing_coverage() -> bool {
    true
}

/// Objective values for one robust operating-envelope candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanOperatingEnvelopeObjective {
    pub score: f64,
    pub airflow_rmse_cfm: f64,
    pub pressure_rmse_pa: f64,
    pub acoustic_rmse_db: Option<f64>,
    pub worst_airflow_error_cfm: f64,
    pub worst_pressure_error_pa: f64,
    pub worst_acoustic_error_db: Option<f64>,
    pub envelope: PcFanManufacturingOperatingEnvelope,
    /// False means the candidate must not be promoted even when the score is
    /// finite. Reasons include non-robust tolerance coverage, a non-monotonic
    /// envelope, or a hard feasibility failure at any corner.
    pub feasible: bool,
}

impl PcFanOperatingEnvelopeObjective {
    pub fn is_valid(&self) -> bool {
        self.score.is_finite()
            && self.airflow_rmse_cfm.is_finite()
            && self.pressure_rmse_pa.is_finite()
            && self.acoustic_rmse_db.is_none_or(f64::is_finite)
            && self.worst_airflow_error_cfm.is_finite()
            && self.worst_pressure_error_pa.is_finite()
            && self.worst_acoustic_error_db.is_none_or(f64::is_finite)
            && self.feasible
            && self.envelope.is_valid()
    }
}

/// Errors found before or during robust-envelope objective evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanOperatingEnvelopeOptimizationError {
    UnsupportedFlowModel(PcFanFlowModel),
    NotEnoughRpms {
        actual: usize,
    },
    InvalidRpm {
        index: usize,
        value: f64,
    },
    NonIncreasingRpms {
        index: usize,
        previous: f64,
        value: f64,
    },
    TargetLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidTarget {
        field: &'static str,
        index: usize,
        value: f64,
    },
    InvalidWeight {
        field: &'static str,
        value: f64,
    },
    NoPositiveWeights,
    ManufacturingEnvelope(PcFanManufacturingOperatingEnvelopeError),
    OptimizerFailure(String),
}

impl fmt::Display for PcFanOperatingEnvelopeOptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                f,
                "PC-fan operating-envelope optimization supports reduced-order flow only, got {model:?}"
            ),
            Self::NotEnoughRpms { actual } => {
                write!(
                    f,
                    "PC-fan operating envelope requires at least two RPMs, got {actual}"
                )
            }
            Self::InvalidRpm { index, value } => {
                write!(
                    f,
                    "PC-fan operating-envelope RPM {index} is invalid: {value}"
                )
            }
            Self::NonIncreasingRpms {
                index,
                previous,
                value,
            } => write!(
                f,
                "PC-fan operating-envelope RPM {index} is not strictly increasing: {value} <= {previous}"
            ),
            Self::TargetLengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "PC-fan operating-envelope target {field} has length {actual}; expected {expected}"
            ),
            Self::InvalidTarget {
                field,
                index,
                value,
            } => write!(
                f,
                "PC-fan operating-envelope target {field}[{index}] is invalid: {value}"
            ),
            Self::InvalidWeight { field, value } => {
                write!(
                    f,
                    "PC-fan operating-envelope weight {field} is invalid: {value}"
                )
            }
            Self::NoPositiveWeights => {
                write!(
                    f,
                    "PC-fan operating-envelope requires a positive objective weight"
                )
            }
            Self::ManufacturingEnvelope(error) => write!(f, "{error}"),
            Self::OptimizerFailure(error) => {
                write!(f, "PC-fan operating-envelope optimizer failed: {error}")
            }
        }
    }
}

impl std::error::Error for PcFanOperatingEnvelopeOptimizationError {}

impl From<PcFanManufacturingOperatingEnvelopeError> for PcFanOperatingEnvelopeOptimizationError {
    fn from(error: PcFanManufacturingOperatingEnvelopeError) -> Self {
        Self::ManufacturingEnvelope(error)
    }
}

/// Summary of the deterministic reduced-order envelope optimization run.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanOperatingEnvelopeOptimizationSummary {
    pub baseline_score: f64,
    pub best_score: f64,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub feasible: bool,
    pub converged: bool,
    pub status: OptimStatus,
}

/// Result of robust reduced-order operating-envelope optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizedPcFanOperatingEnvelope {
    pub config: PcFanConfig,
    pub design_variables: PcFanDesignVariables,
    pub geometry: crate::model::PcFanGeometry,
    pub evaluation: PcFanEvaluation,
    pub objective: PcFanOperatingEnvelopeObjective,
    pub optimization: PcFanOperatingEnvelopeOptimizationSummary,
}

impl OptimizedPcFanOperatingEnvelope {
    /// Whether the selected candidate satisfies both the envelope contract
    /// and the nominal hard feasibility contract.
    pub fn is_feasible(&self) -> bool {
        self.objective.feasible && self.evaluation.is_feasible(&self.config)
    }

    pub fn manufacturing_operating_envelope(&self) -> &PcFanManufacturingOperatingEnvelope {
        &self.objective.envelope
    }
}

fn validate_objective_config(
    objective: &PcFanOperatingEnvelopeObjectiveConfig,
) -> Result<(), PcFanOperatingEnvelopeOptimizationError> {
    if objective.rpms.len() < 2 {
        return Err(PcFanOperatingEnvelopeOptimizationError::NotEnoughRpms {
            actual: objective.rpms.len(),
        });
    }
    for (index, &rpm) in objective.rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanOperatingEnvelopeOptimizationError::InvalidRpm { index, value: rpm });
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| objective.rpms.get(i))
            && rpm.total_cmp(previous) != Ordering::Greater
        {
            return Err(PcFanOperatingEnvelopeOptimizationError::NonIncreasingRpms {
                index,
                previous: *previous,
                value: rpm,
            });
        }
    }

    for (field, values) in [
        (
            "target_airflow_cfm",
            objective.target_airflow_cfm.as_slice(),
        ),
        (
            "target_pressure_rise_pa",
            objective.target_pressure_rise_pa.as_slice(),
        ),
    ] {
        if values.len() != objective.rpms.len() {
            return Err(
                PcFanOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field,
                    expected: objective.rpms.len(),
                    actual: values.len(),
                },
            );
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PcFanOperatingEnvelopeOptimizationError::InvalidTarget {
                    field,
                    index,
                    value,
                });
            }
        }
    }

    if let Some(values) = objective.target_acoustic_level_db.as_deref() {
        if values.len() != objective.rpms.len() {
            return Err(
                PcFanOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field: "target_acoustic_level_db",
                    expected: objective.rpms.len(),
                    actual: values.len(),
                },
            );
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PcFanOperatingEnvelopeOptimizationError::InvalidTarget {
                    field: "target_acoustic_level_db",
                    index,
                    value,
                });
            }
        }
    }

    let weights = [
        ("airflow_weight", objective.airflow_weight),
        ("pressure_weight", objective.pressure_weight),
        ("acoustic_weight", objective.acoustic_weight),
    ];
    for (field, value) in weights {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanOperatingEnvelopeOptimizationError::InvalidWeight { field, value });
        }
    }
    if weights.iter().all(|(_, value)| *value == 0.0) {
        return Err(PcFanOperatingEnvelopeOptimizationError::NoPositiveWeights);
    }
    Ok(())
}

fn target_scale(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(1.0, f64::max)
}

fn rpm_index(rpms: &[f64], rpm: f64) -> usize {
    rpms.iter()
        .position(|candidate| candidate.total_cmp(&rpm) == Ordering::Equal)
        .expect("validated operating-envelope point must match a target RPM")
}

/// Score one design over every requested RPM and manufacturing realization.
pub fn evaluate_pc_fan_operating_envelope_objective(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    objective: &PcFanOperatingEnvelopeObjectiveConfig,
) -> Result<PcFanOperatingEnvelopeObjective, PcFanOperatingEnvelopeOptimizationError> {
    validate_objective_config(objective)?;
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(
            PcFanOperatingEnvelopeOptimizationError::UnsupportedFlowModel(config.cfd.flow_model),
        );
    }

    let mut envelope_config = config.clone();
    envelope_config.operating_point_spread.rpms = objective.rpms.clone();
    let envelope = evaluate_pc_fan_manufacturing_operating_envelope(&envelope_config, design)?;
    let sample_count = envelope.points.len() as f64;
    let airflow_scale = target_scale(&objective.target_airflow_cfm);
    let pressure_scale = target_scale(&objective.target_pressure_rise_pa);
    let acoustic_scale = objective
        .target_acoustic_level_db
        .as_deref()
        .map(target_scale);
    let mut airflow_squared = 0.0;
    let mut pressure_squared = 0.0;
    let mut acoustic_squared = 0.0;
    let mut worst_airflow_error_cfm: f64 = 0.0;
    let mut worst_pressure_error_pa: f64 = 0.0;
    let mut worst_acoustic_error_db: f64 = 0.0;

    for point in &envelope.points {
        let index = rpm_index(&objective.rpms, point.rpm);
        let airflow_error = point.point.airflow_cfm - objective.target_airflow_cfm[index];
        let pressure_error =
            point.point.pressure_rise_pa - objective.target_pressure_rise_pa[index];
        airflow_squared += airflow_error * airflow_error;
        pressure_squared += pressure_error * pressure_error;
        worst_airflow_error_cfm = worst_airflow_error_cfm.max(airflow_error.abs());
        worst_pressure_error_pa = worst_pressure_error_pa.max(pressure_error.abs());
        if let Some(targets) = objective.target_acoustic_level_db.as_deref() {
            let acoustic_error = point.point.acoustic_level_db - targets[index];
            acoustic_squared += acoustic_error * acoustic_error;
            worst_acoustic_error_db = worst_acoustic_error_db.max(acoustic_error.abs());
        }
    }

    let airflow_rmse_cfm = (airflow_squared / sample_count).sqrt();
    let pressure_rmse_pa = (pressure_squared / sample_count).sqrt();
    let acoustic_rmse_db = objective
        .target_acoustic_level_db
        .as_ref()
        .map(|_| (acoustic_squared / sample_count).sqrt());
    let score = objective.airflow_weight * (airflow_rmse_cfm / airflow_scale).powi(2)
        + objective.pressure_weight * (pressure_rmse_pa / pressure_scale).powi(2)
        + objective.acoustic_weight * acoustic_rmse_db.unwrap_or(0.0).powi(2)
            / acoustic_scale.unwrap_or(1.0).powi(2);
    let coverage_ok =
        !objective.require_manufacturing_coverage || envelope.has_tolerance_coverage();
    let feasible = envelope.is_valid() && coverage_ok && score.is_finite();

    Ok(PcFanOperatingEnvelopeObjective {
        score: if feasible { score } else { f64::INFINITY },
        airflow_rmse_cfm,
        pressure_rmse_pa,
        acoustic_rmse_db,
        worst_airflow_error_cfm,
        worst_pressure_error_pa,
        worst_acoustic_error_db: objective
            .target_acoustic_level_db
            .as_ref()
            .map(|_| worst_acoustic_error_db),
        envelope,
        feasible,
    })
}

fn candidate_is_better(
    candidate: &PcFanOperatingEnvelopeObjective,
    incumbent: &PcFanOperatingEnvelopeObjective,
) -> bool {
    match (candidate.feasible, incumbent.feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => candidate.score + 1.0e-12 < incumbent.score,
    }
}

/// Optimize a reduced-order PC-fan design against a robust RPM/manufacturing
/// operating envelope using deterministic seeded L-SHADE.
pub fn optimize_pc_fan_operating_envelope(
    mut config: PcFanConfig,
    objective: PcFanOperatingEnvelopeObjectiveConfig,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
) -> Result<OptimizedPcFanOperatingEnvelope, PcFanOperatingEnvelopeOptimizationError> {
    validate_objective_config(&objective)?;
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(
            PcFanOperatingEnvelopeOptimizationError::UnsupportedFlowModel(config.cfd.flow_model),
        );
    }
    config.operating_point_spread.rpms = objective.rpms.clone();
    let baseline_design = config.default_design_variables();
    let baseline_objective =
        evaluate_pc_fan_operating_envelope_objective(&config, &baseline_design, &objective)?;
    let bounds = crate::optimizer::design_bounds(&config);
    let de_bounds: Vec<(f64, f64)> = bounds
        .iter()
        .map(|bound| {
            (
                bound.lower.unwrap_or(f64::NEG_INFINITY),
                bound.upper.unwrap_or(f64::INFINITY),
            )
        })
        .collect();
    let lshade = LShadeConfig {
        np_init: popsize_factor.max(5),
        np_final: 4,
        p: 0.11,
        arc_rate: 2.1,
        memory_size: 6,
    };
    let de_config = DEConfigBuilder::new()
        .maxiter(maxiter)
        .popsize(popsize_factor.max(5))
        .seed(seed)
        .strategy(Strategy::LShadeBin)
        .lshade(lshade)
        .build()
        .map_err(|error| {
            PcFanOperatingEnvelopeOptimizationError::OptimizerFailure(format!("{error:?}"))
        })?;
    let report = differential_evolution(
        &|x: &[f64]| {
            let design = crate::optimizer::vector_to_design(&config, x);
            evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective)
                .map(|result| result.score)
                .unwrap_or(f64::INFINITY)
        },
        &de_bounds,
        de_config,
    )
    .map_err(|error| {
        PcFanOperatingEnvelopeOptimizationError::OptimizerFailure(format!("{error:?}"))
    })?;
    let candidate_slice = report.x.as_slice();
    let candidate_design = crate::optimizer::vector_to_design(&config, candidate_slice);
    let candidate_objective =
        evaluate_pc_fan_operating_envelope_objective(&config, &candidate_design, &objective)?;
    let baseline_score = baseline_objective.score;
    let (selected_design, selected_objective) =
        if candidate_is_better(&candidate_objective, &baseline_objective) {
            (candidate_design, candidate_objective)
        } else {
            (baseline_design, baseline_objective)
        };
    let geometry = config.build_geometry(&selected_design);
    let evaluation = evaluate_pc_fan(&config, &selected_design);
    let feasible = selected_objective.feasible && evaluation.is_feasible(&config);
    let status = if report.success {
        OptimStatus::Converged
    } else {
        OptimStatus::MaxIterationsReached
    };
    Ok(OptimizedPcFanOperatingEnvelope {
        config,
        design_variables: selected_design,
        geometry,
        evaluation,
        optimization: PcFanOperatingEnvelopeOptimizationSummary {
            baseline_score,
            best_score: selected_objective.score,
            iterations: report.nit,
            function_evaluations: report.nfev,
            feasible,
            converged: report.success,
            status,
        },
        objective: selected_objective,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::evaluate_pc_fan_manufacturing_operating_envelope;
    use crate::model::{PcFanManufacturingTolerance, PcFanManufacturingVariation};

    fn robust_config() -> PcFanConfig {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 2.0;
        config.structural.min_modal_separation_hz = 0.0;
        config.structural.max_imbalance_g_mm = 100.0;
        config.manufacturing.tolerance = PcFanManufacturingTolerance {
            blade_thickness_fraction: 0.02,
            blade_chord_fraction: 0.02,
            twist_deg: 1.0,
            shroud_offset_m: 0.00025,
        };
        config
    }

    fn objective_from_baseline(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
    ) -> PcFanOperatingEnvelopeObjectiveConfig {
        let mut config = config.clone();
        config.operating_point_spread.rpms = vec![400.0, 600.0, 900.0, 1200.0];
        let envelope = evaluate_pc_fan_manufacturing_operating_envelope(&config, design).unwrap();
        let nominal = envelope
            .points
            .iter()
            .filter(|point| point.variation == PcFanManufacturingVariation::nominal())
            .collect::<Vec<_>>();
        PcFanOperatingEnvelopeObjectiveConfig {
            rpms: nominal.iter().map(|point| point.rpm).collect(),
            target_airflow_cfm: nominal
                .iter()
                .map(|point| point.point.airflow_cfm)
                .collect(),
            target_pressure_rise_pa: nominal
                .iter()
                .map(|point| point.point.pressure_rise_pa)
                .collect(),
            target_acoustic_level_db: Some(
                nominal
                    .iter()
                    .map(|point| point.point.acoustic_level_db)
                    .collect(),
            ),
            airflow_weight: 1.0,
            pressure_weight: 1.0,
            acoustic_weight: 0.25,
            require_manufacturing_coverage: true,
        }
    }

    #[test]
    fn objective_pairs_targets_by_rpm_and_accepts_robust_baseline() {
        let config = robust_config();
        let design = config.default_design_variables();
        let objective_config = objective_from_baseline(&config, &design);
        let result =
            evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective_config)
                .expect("baseline envelope objective must evaluate");
        assert!(result.feasible, "unexpected envelope rejection");
        assert!(result.is_valid());
        // The targets describe the nominal realization while the robust score
        // also prices deterministic tolerance-corner drift.
        assert!(result.score < 1.0e-2, "unexpected robust baseline score");
        assert_eq!(result.envelope.variation_count, 17);
        assert_eq!(result.envelope.rpm_count, 4);
        assert_eq!(result.envelope.points.len(), 68);
    }

    #[test]
    fn objective_rejects_malformed_rpm_and_target_contracts() {
        let config = robust_config();
        let design = config.default_design_variables();
        let mut objective = objective_from_baseline(&config, &design);
        objective.rpms[1] = objective.rpms[0];
        assert!(matches!(
            evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective),
            Err(PcFanOperatingEnvelopeOptimizationError::NonIncreasingRpms { .. })
        ));

        let mut objective = objective_from_baseline(&config, &design);
        objective.target_pressure_rise_pa.pop();
        assert!(matches!(
            evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective),
            Err(
                PcFanOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field: "target_pressure_rise_pa",
                    ..
                }
            )
        ));

        let mut objective = objective_from_baseline(&config, &design);
        objective.acoustic_weight = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective),
            Err(PcFanOperatingEnvelopeOptimizationError::InvalidWeight {
                field: "acoustic_weight",
                ..
            })
        ));
    }

    #[test]
    fn objective_fails_closed_without_manufacturing_coverage() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 2.0;
        config.structural.min_modal_separation_hz = 0.0;
        config.structural.max_imbalance_g_mm = 100.0;
        let design = config.default_design_variables();
        let mut objective = objective_from_baseline(&robust_config(), &design);
        objective.require_manufacturing_coverage = true;
        let result = evaluate_pc_fan_operating_envelope_objective(&config, &design, &objective)
            .expect("nominal envelope should return a diagnostic objective");
        assert!(!result.feasible);
        assert!(!result.is_valid());
        assert!(result.score.is_infinite());
    }

    #[test]
    fn optimizer_refuses_resolved_flow_before_de_search() {
        let mut config = robust_config();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        let design = config.default_design_variables();
        let objective = objective_from_baseline(&robust_config(), &design);
        assert!(matches!(
            optimize_pc_fan_operating_envelope(config, objective, 1, 5, 7),
            Err(
                PcFanOperatingEnvelopeOptimizationError::UnsupportedFlowModel(
                    PcFanFlowModel::IncompressibleCfdFull
                )
            )
        ));
    }

    #[test]
    fn optimizer_evaluates_and_returns_the_robust_baseline_when_budget_is_small() {
        let config = robust_config();
        let design = config.default_design_variables();
        let objective = objective_from_baseline(&config, &design);

        let result = optimize_pc_fan_operating_envelope(config, objective, 1, 5, 11)
            .expect("one reduced-order DE iteration should produce a result");

        assert!(result.optimization.function_evaluations > 0);
        assert!(result.objective.score.is_finite());
        assert!(result.is_feasible());
    }
}
