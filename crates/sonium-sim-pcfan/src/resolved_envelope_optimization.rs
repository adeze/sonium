//! Derivative-free optimization over a checked resolved-flow envelope.
//!
//! The optimizer deliberately consumes [`crate::resolved_envelope`] reports
//! rather than raw CFD fields. A candidate is promotable only when every RPM
//! and manufacturing corner has passed mesh convergence, conservation
//! admission, feasibility, and monotonicity gates. This is a resolved-flow
//! optimizer, but it does not pretend to provide design derivatives or a
//! resolved-flow adjoint.

use crate::model::{PcFanConfig, PcFanDesignVariables, PcFanFlowModel};
use crate::optimizer::{design_bounds, vector_to_design};
use crate::resolved_envelope::{
    PcFanResolvedOperatingEnvelope, PcFanResolvedOperatingEnvelopeConfig,
    PcFanResolvedOperatingEnvelopeError, evaluate_pc_fan_resolved_operating_envelope,
};
use sonium_solvers_optimization::OptimStatus;
use sonium_solvers_optimization::unconstrained::de::{
    DEConfigBuilder, LShadeConfig, Strategy, differential_evolution,
};
use std::cmp::Ordering;
use std::fmt;

/// Target and admission policy for resolved-flow envelope optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanResolvedOperatingEnvelopeObjectiveConfig {
    pub envelope: PcFanResolvedOperatingEnvelopeConfig,
    pub target_airflow_cfm: Vec<f64>,
    pub target_pressure_rise_pa: Vec<f64>,
    pub target_acoustic_level_db: Option<Vec<f64>>,
    pub airflow_weight: f64,
    pub pressure_weight: f64,
    pub acoustic_weight: f64,
}

/// Objective and checked envelope for one resolved-flow candidate.
#[derive(Debug, PartialEq)]
pub struct PcFanResolvedOperatingEnvelopeObjective {
    pub score: f64,
    pub airflow_rmse_cfm: f64,
    pub pressure_rmse_pa: f64,
    pub acoustic_rmse_db: Option<f64>,
    pub worst_airflow_error_cfm: f64,
    pub worst_pressure_error_pa: f64,
    pub worst_acoustic_error_db: Option<f64>,
    pub envelope: PcFanResolvedOperatingEnvelope,
    pub feasible: bool,
}

impl PcFanResolvedOperatingEnvelopeObjective {
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

/// Summary of one deterministic resolved-flow optimization run.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanResolvedOperatingEnvelopeOptimizationSummary {
    pub baseline_score: f64,
    pub best_score: f64,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub feasible: bool,
    pub converged: bool,
    pub status: OptimStatus,
}

/// Selected design and checked resolved-flow envelope.
#[derive(Debug, PartialEq)]
pub struct OptimizedPcFanResolvedOperatingEnvelope {
    pub config: PcFanConfig,
    pub design_variables: PcFanDesignVariables,
    pub objective: PcFanResolvedOperatingEnvelopeObjective,
    pub optimization: PcFanResolvedOperatingEnvelopeOptimizationSummary,
}

impl OptimizedPcFanResolvedOperatingEnvelope {
    pub fn is_feasible(&self) -> bool {
        self.objective.feasible
    }

    pub fn envelope(&self) -> &PcFanResolvedOperatingEnvelope {
        &self.objective.envelope
    }
}

/// Errors raised before or during resolved-flow envelope optimization.
#[derive(Debug, PartialEq)]
pub enum PcFanResolvedOperatingEnvelopeOptimizationError {
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
    EmptyEnvelope,
    RpmNotInEnvelope {
        rpm: f64,
    },
    Envelope(PcFanResolvedOperatingEnvelopeError),
    OptimizerFailure(String),
}

impl fmt::Display for PcFanResolvedOperatingEnvelopeOptimizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                formatter,
                "resolved PC-fan envelope optimization requires CFD flow, got {model:?}"
            ),
            Self::NotEnoughRpms { actual } => write!(
                formatter,
                "resolved PC-fan envelope optimization requires at least two RPMs, got {actual}"
            ),
            Self::InvalidRpm { index, value } => {
                write!(formatter, "resolved PC-fan RPM {index} is invalid: {value}")
            }
            Self::NonIncreasingRpms {
                index,
                previous,
                value,
            } => write!(
                formatter,
                "resolved PC-fan RPM {index} is not strictly increasing: {value} <= {previous}"
            ),
            Self::TargetLengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "resolved PC-fan target {field} has length {actual}; expected {expected}"
            ),
            Self::InvalidTarget {
                field,
                index,
                value,
            } => write!(
                formatter,
                "resolved PC-fan target {field}[{index}] is invalid: {value}"
            ),
            Self::InvalidWeight { field, value } => {
                write!(
                    formatter,
                    "resolved PC-fan weight {field} is invalid: {value}"
                )
            }
            Self::NoPositiveWeights => write!(
                formatter,
                "resolved PC-fan envelope optimization requires a positive objective weight"
            ),
            Self::EmptyEnvelope => write!(formatter, "resolved PC-fan envelope has no samples"),
            Self::RpmNotInEnvelope { rpm } => {
                write!(
                    formatter,
                    "resolved PC-fan envelope contains unknown RPM {rpm}"
                )
            }
            Self::Envelope(error) => write!(formatter, "{error}"),
            Self::OptimizerFailure(error) => {
                write!(
                    formatter,
                    "resolved PC-fan envelope optimizer failed: {error}"
                )
            }
        }
    }
}

impl std::error::Error for PcFanResolvedOperatingEnvelopeOptimizationError {}

impl From<PcFanResolvedOperatingEnvelopeError> for PcFanResolvedOperatingEnvelopeOptimizationError {
    fn from(error: PcFanResolvedOperatingEnvelopeError) -> Self {
        Self::Envelope(error)
    }
}

fn validate_objective_config(
    objective: &PcFanResolvedOperatingEnvelopeObjectiveConfig,
) -> Result<(), PcFanResolvedOperatingEnvelopeOptimizationError> {
    if objective.envelope.rpms.len() < 2 {
        return Err(
            PcFanResolvedOperatingEnvelopeOptimizationError::NotEnoughRpms {
                actual: objective.envelope.rpms.len(),
            },
        );
    }
    for (index, &rpm) in objective.envelope.rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::InvalidRpm { index, value: rpm },
            );
        }
        if let Some(&previous) = index
            .checked_sub(1)
            .and_then(|previous| objective.envelope.rpms.get(previous))
            && rpm.total_cmp(&previous) != Ordering::Greater
        {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::NonIncreasingRpms {
                    index,
                    previous,
                    value: rpm,
                },
            );
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
        if values.len() != objective.envelope.rpms.len() {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field,
                    expected: objective.envelope.rpms.len(),
                    actual: values.len(),
                },
            );
        }
        if let Some((index, &value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::InvalidTarget {
                    field,
                    index,
                    value,
                },
            );
        }
    }
    if let Some(values) = objective.target_acoustic_level_db.as_deref() {
        if values.len() != objective.envelope.rpms.len() {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field: "target_acoustic_level_db",
                    expected: objective.envelope.rpms.len(),
                    actual: values.len(),
                },
            );
        }
        if let Some((index, &value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::InvalidTarget {
                    field: "target_acoustic_level_db",
                    index,
                    value,
                },
            );
        }
    }

    let weights = [
        ("airflow_weight", objective.airflow_weight),
        ("pressure_weight", objective.pressure_weight),
        ("acoustic_weight", objective.acoustic_weight),
    ];
    for (field, value) in weights {
        if !value.is_finite() || value < 0.0 {
            return Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::InvalidWeight { field, value },
            );
        }
    }
    if objective.airflow_weight == 0.0
        && objective.pressure_weight == 0.0
        && objective.acoustic_weight == 0.0
    {
        return Err(PcFanResolvedOperatingEnvelopeOptimizationError::NoPositiveWeights);
    }
    Ok(())
}

fn target_scale(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(1.0, f64::max)
}

#[derive(Debug, Clone, Copy)]
struct EnvelopeMetricSample {
    rpm: f64,
    airflow_cfm: f64,
    pressure_rise_pa: f64,
    acoustic_level_db: f64,
}

#[derive(Debug, Clone, Copy)]
struct ScoredEnvelopeMetrics {
    score: f64,
    airflow_rmse_cfm: f64,
    pressure_rmse_pa: f64,
    acoustic_rmse_db: Option<f64>,
    worst_airflow_error_cfm: f64,
    worst_pressure_error_pa: f64,
    worst_acoustic_error_db: Option<f64>,
}

fn score_metric_samples(
    objective: &PcFanResolvedOperatingEnvelopeObjectiveConfig,
    samples: &[EnvelopeMetricSample],
    admitted: bool,
) -> Result<ScoredEnvelopeMetrics, PcFanResolvedOperatingEnvelopeOptimizationError> {
    if samples.is_empty() {
        return Err(PcFanResolvedOperatingEnvelopeOptimizationError::EmptyEnvelope);
    }
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

    for sample in samples {
        let index = objective
            .envelope
            .rpms
            .iter()
            .position(|rpm| rpm.total_cmp(&sample.rpm) == Ordering::Equal)
            .ok_or(
                PcFanResolvedOperatingEnvelopeOptimizationError::RpmNotInEnvelope {
                    rpm: sample.rpm,
                },
            )?;
        let airflow_error = sample.airflow_cfm - objective.target_airflow_cfm[index];
        let pressure_error = sample.pressure_rise_pa - objective.target_pressure_rise_pa[index];
        airflow_squared += airflow_error * airflow_error;
        pressure_squared += pressure_error * pressure_error;
        worst_airflow_error_cfm = worst_airflow_error_cfm.max(airflow_error.abs());
        worst_pressure_error_pa = worst_pressure_error_pa.max(pressure_error.abs());
        if let Some(targets) = objective.target_acoustic_level_db.as_deref() {
            let acoustic_error = sample.acoustic_level_db - targets[index];
            acoustic_squared += acoustic_error * acoustic_error;
            worst_acoustic_error_db = worst_acoustic_error_db.max(acoustic_error.abs());
        }
    }

    let sample_count = samples.len() as f64;
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
    let finite = score.is_finite()
        && airflow_rmse_cfm.is_finite()
        && pressure_rmse_pa.is_finite()
        && acoustic_rmse_db.is_none_or(f64::is_finite);
    Ok(ScoredEnvelopeMetrics {
        score: if admitted && finite {
            score
        } else {
            f64::INFINITY
        },
        airflow_rmse_cfm,
        pressure_rmse_pa,
        acoustic_rmse_db,
        worst_airflow_error_cfm,
        worst_pressure_error_pa,
        worst_acoustic_error_db: objective
            .target_acoustic_level_db
            .as_ref()
            .map(|_| worst_acoustic_error_db),
    })
}

/// Evaluate a resolved-flow design against an RPM/manufacturing target.
pub fn evaluate_pc_fan_resolved_operating_envelope_objective(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    objective: &PcFanResolvedOperatingEnvelopeObjectiveConfig,
) -> Result<PcFanResolvedOperatingEnvelopeObjective, PcFanResolvedOperatingEnvelopeOptimizationError>
{
    validate_objective_config(objective)?;
    if !config.cfd.flow_model.is_cfd() {
        return Err(
            PcFanResolvedOperatingEnvelopeOptimizationError::UnsupportedFlowModel(
                config.cfd.flow_model,
            ),
        );
    }
    let envelope =
        evaluate_pc_fan_resolved_operating_envelope(config, design, &objective.envelope)?;
    let samples = envelope
        .points
        .iter()
        .map(|point| EnvelopeMetricSample {
            rpm: point.rpm,
            airflow_cfm: point.evaluation.airflow_cfm,
            pressure_rise_pa: point.evaluation.flow.pressure_rise_pa,
            acoustic_level_db: point.evaluation.acoustic_level_db,
        })
        .collect::<Vec<_>>();
    let admitted = envelope.is_valid()
        && (!objective.envelope.require_manufacturing_coverage || envelope.has_tolerance_coverage);
    let metrics = score_metric_samples(objective, &samples, admitted)?;
    let feasible = admitted && metrics.score.is_finite();
    Ok(PcFanResolvedOperatingEnvelopeObjective {
        score: metrics.score,
        airflow_rmse_cfm: metrics.airflow_rmse_cfm,
        pressure_rmse_pa: metrics.pressure_rmse_pa,
        acoustic_rmse_db: metrics.acoustic_rmse_db,
        worst_airflow_error_cfm: metrics.worst_airflow_error_cfm,
        worst_pressure_error_pa: metrics.worst_pressure_error_pa,
        worst_acoustic_error_db: metrics.worst_acoustic_error_db,
        envelope,
        feasible,
    })
}

fn candidate_is_better(
    candidate: &PcFanResolvedOperatingEnvelopeObjective,
    incumbent: &PcFanResolvedOperatingEnvelopeObjective,
) -> bool {
    match (candidate.feasible, incumbent.feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => candidate.score + 1.0e-12 < incumbent.score,
    }
}

fn run_design_de<F>(
    config: &PcFanConfig,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
    evaluator: F,
) -> Result<(Vec<f64>, usize, usize, bool), PcFanResolvedOperatingEnvelopeOptimizationError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let de_bounds = design_bounds(config)
        .iter()
        .map(|bound| {
            (
                bound.lower.unwrap_or(f64::NEG_INFINITY),
                bound.upper.unwrap_or(f64::INFINITY),
            )
        })
        .collect::<Vec<_>>();
    let population = popsize_factor.max(5);
    let de_config = DEConfigBuilder::new()
        .maxiter(maxiter)
        .popsize(population)
        .seed(seed)
        .strategy(Strategy::LShadeBin)
        .lshade(LShadeConfig {
            np_init: population,
            np_final: 4,
            p: 0.11,
            arc_rate: 2.1,
            memory_size: 6,
        })
        .build()
        .map_err(|error| {
            PcFanResolvedOperatingEnvelopeOptimizationError::OptimizerFailure(format!("{error:?}"))
        })?;
    let report = differential_evolution(&|x: &[f64]| evaluator(x), &de_bounds, de_config).map_err(
        |error| {
            PcFanResolvedOperatingEnvelopeOptimizationError::OptimizerFailure(format!("{error:?}"))
        },
    )?;
    let x = report.x.clone();
    Ok((x, report.nit, report.nfev, report.success))
}

/// Optimize a resolved-flow design with deterministic derivative-free search.
///
/// Every objective callback runs the checked resolved-flow envelope path. A
/// failed candidate receives an infinite score and can never be promoted;
/// the final candidate is re-evaluated through the same checked path.
pub fn optimize_pc_fan_resolved_operating_envelope(
    config: PcFanConfig,
    objective: PcFanResolvedOperatingEnvelopeObjectiveConfig,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
) -> Result<OptimizedPcFanResolvedOperatingEnvelope, PcFanResolvedOperatingEnvelopeOptimizationError>
{
    validate_objective_config(&objective)?;
    if !config.cfd.flow_model.is_cfd() {
        return Err(
            PcFanResolvedOperatingEnvelopeOptimizationError::UnsupportedFlowModel(
                config.cfd.flow_model,
            ),
        );
    }
    let baseline_design = config.default_design_variables();
    let baseline_objective = evaluate_pc_fan_resolved_operating_envelope_objective(
        &config,
        &baseline_design,
        &objective,
    )?;
    let (candidate_vector, iterations, function_evaluations, de_success) =
        run_design_de(&config, maxiter, popsize_factor, seed, |values| {
            let design = vector_to_design(&config, values);
            evaluate_pc_fan_resolved_operating_envelope_objective(&config, &design, &objective)
                .map(|result| result.score)
                .unwrap_or(f64::INFINITY)
        })?;
    let candidate_design = vector_to_design(&config, &candidate_vector);
    let candidate_objective = evaluate_pc_fan_resolved_operating_envelope_objective(
        &config,
        &candidate_design,
        &objective,
    )?;
    let baseline_score = baseline_objective.score;
    let (selected_design, selected_objective) =
        if candidate_is_better(&candidate_objective, &baseline_objective) {
            (candidate_design, candidate_objective)
        } else {
            (baseline_design, baseline_objective)
        };
    let selected_score = selected_objective.score;
    let feasible = selected_objective.feasible;
    let status = if !feasible {
        OptimStatus::RestorationFailed
    } else if de_success {
        OptimStatus::Converged
    } else {
        OptimStatus::MaxIterationsReached
    };
    Ok(OptimizedPcFanResolvedOperatingEnvelope {
        config,
        design_variables: selected_design,
        objective: selected_objective,
        optimization: PcFanResolvedOperatingEnvelopeOptimizationSummary {
            baseline_score,
            best_score: selected_score,
            iterations,
            function_evaluations,
            feasible,
            converged: de_success && feasible,
            status,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective_config() -> PcFanResolvedOperatingEnvelopeObjectiveConfig {
        PcFanResolvedOperatingEnvelopeObjectiveConfig {
            envelope: PcFanResolvedOperatingEnvelopeConfig {
                rpms: vec![500.0, 700.0],
                ..PcFanResolvedOperatingEnvelopeConfig::default()
            },
            target_airflow_cfm: vec![10.0, 14.0],
            target_pressure_rise_pa: vec![80.0, 120.0],
            target_acoustic_level_db: Some(vec![30.0, 34.0]),
            airflow_weight: 1.0,
            pressure_weight: 2.0,
            acoustic_weight: 0.5,
        }
    }

    #[test]
    fn metric_scoring_is_zero_for_exact_rpm_aligned_samples() {
        let objective = objective_config();
        let samples = [
            EnvelopeMetricSample {
                rpm: 700.0,
                airflow_cfm: 14.0,
                pressure_rise_pa: 120.0,
                acoustic_level_db: 34.0,
            },
            EnvelopeMetricSample {
                rpm: 500.0,
                airflow_cfm: 10.0,
                pressure_rise_pa: 80.0,
                acoustic_level_db: 30.0,
            },
        ];
        let metrics = score_metric_samples(&objective, &samples, true).unwrap();
        assert_eq!(metrics.score, 0.0);
        assert_eq!(metrics.airflow_rmse_cfm, 0.0);
        assert_eq!(metrics.pressure_rmse_pa, 0.0);
        assert_eq!(metrics.acoustic_rmse_db, Some(0.0));
    }

    #[test]
    fn metric_scoring_fails_closed_for_unknown_rpm_or_unadmitted_samples() {
        let objective = objective_config();
        let unknown = [EnvelopeMetricSample {
            rpm: 600.0,
            airflow_cfm: 12.0,
            pressure_rise_pa: 100.0,
            acoustic_level_db: 32.0,
        }];
        assert!(matches!(
            score_metric_samples(&objective, &unknown, true),
            Err(PcFanResolvedOperatingEnvelopeOptimizationError::RpmNotInEnvelope { rpm: 600.0 })
        ));

        let samples = [EnvelopeMetricSample {
            rpm: 500.0,
            airflow_cfm: 10.0,
            pressure_rise_pa: 80.0,
            acoustic_level_db: 30.0,
        }];
        let metrics = score_metric_samples(&objective, &samples, false).unwrap();
        assert!(metrics.score.is_infinite());
    }

    #[test]
    fn objective_validation_rejects_topology_targets_and_weights() {
        let mut objective = objective_config();
        objective.envelope.rpms = vec![700.0];
        assert!(matches!(
            validate_objective_config(&objective),
            Err(PcFanResolvedOperatingEnvelopeOptimizationError::NotEnoughRpms { actual: 1 })
        ));

        let mut objective = objective_config();
        objective.envelope.rpms[1] = objective.envelope.rpms[0];
        assert!(matches!(
            validate_objective_config(&objective),
            Err(PcFanResolvedOperatingEnvelopeOptimizationError::NonIncreasingRpms { .. })
        ));

        let mut objective = objective_config();
        objective.target_pressure_rise_pa.pop();
        assert!(matches!(
            validate_objective_config(&objective),
            Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::TargetLengthMismatch {
                    field: "target_pressure_rise_pa",
                    ..
                }
            )
        ));

        let mut objective = objective_config();
        objective.airflow_weight = f64::NAN;
        assert!(matches!(
            validate_objective_config(&objective),
            Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::InvalidWeight {
                    field: "airflow_weight",
                    ..
                }
            )
        ));

        let mut objective = objective_config();
        objective.airflow_weight = 0.0;
        objective.pressure_weight = 0.0;
        objective.acoustic_weight = 0.0;
        assert!(matches!(
            validate_objective_config(&objective),
            Err(PcFanResolvedOperatingEnvelopeOptimizationError::NoPositiveWeights)
        ));
    }

    #[test]
    fn resolved_objective_refuses_reduced_flow_before_cfd_evaluation() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        assert!(matches!(
            evaluate_pc_fan_resolved_operating_envelope_objective(
                &config,
                &design,
                &objective_config(),
            ),
            Err(
                PcFanResolvedOperatingEnvelopeOptimizationError::UnsupportedFlowModel(
                    PcFanFlowModel::ReducedOrder
                )
            )
        ));
    }

    #[test]
    fn derivative_free_design_search_has_bounded_deterministic_contract() {
        let config = PcFanConfig::bench_120mm();
        let (x, iterations, evaluations, _) = run_design_de(&config, 1, 5, 17, |values| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let target = if index == 0 { 0.001 } else { 0.0 };
                    (value - target).powi(2)
                })
                .sum()
        })
        .unwrap();
        assert_eq!(x.len(), crate::model::DESIGN_VARIABLE_COUNT);
        assert!(iterations <= 1);
        assert!(evaluations > 0);
        assert!(x.iter().all(|value| value.is_finite()));
    }
}
