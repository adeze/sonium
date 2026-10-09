//! Measured-data correlation contract for PC-fan predictions.
//!
//! The contract keeps measured, synthetic, and solved data provenance
//! explicit.  It computes only comparison statistics; passing a synthetic
//! fixture must never be described as measured-product validation.

use crate::evaluation::{
    evaluate_pc_fan, evaluate_pc_fan_from_cfd_solution, try_evaluate_pc_fan_from_cfd_solution,
};
use crate::model::{PcFanConfig, PcFanDesignVariables, PcFanFlowModel};
use serde::{Deserialize, Serialize};
use std::fmt;

/// One operating point shared by measured and predicted datasets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanCorrelationObservation {
    pub rpm: f64,
    pub airflow_cfm: f64,
    pub pressure_rise_pa: f64,
    pub acoustic_level_db: Option<f64>,
}

/// A provenance-labelled measured or predicted dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCorrelationDataset {
    /// Source URI, fixture identifier, instrument record, or solver run ID.
    pub provenance: String,
    pub observations: Vec<PcFanCorrelationObservation>,
}

/// Acceptance budgets for measured-data correlation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredCorrelationConfig {
    pub rpm_tolerance: f64,
    pub max_airflow_rmse_cfm: f64,
    pub max_pressure_rmse_pa: f64,
    pub max_acoustic_rmse_db: f64,
    pub min_points: usize,
    pub require_acoustic: bool,
}

impl Default for PcFanMeasuredCorrelationConfig {
    fn default() -> Self {
        Self {
            rpm_tolerance: 1.0e-6,
            max_airflow_rmse_cfm: 2.0,
            max_pressure_rmse_pa: 20.0,
            max_acoustic_rmse_db: 3.0,
            min_points: 3,
            require_acoustic: true,
        }
    }
}

/// Malformed correlation input rejected before any metric is published.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcFanMeasuredCorrelationError {
    InvalidTolerance(&'static str, f64),
    InvalidMinimumPoints,
    MissingProvenance(&'static str),
    TooFewPoints {
        dataset: &'static str,
        actual: usize,
        required: usize,
    },
    NonFiniteObservation {
        dataset: &'static str,
        index: usize,
    },
    InvalidRpm {
        dataset: &'static str,
        index: usize,
        rpm: f64,
    },
    DuplicateRpm {
        dataset: &'static str,
        index: usize,
    },
    RpmTopologyMismatch {
        index: usize,
        measured_rpm: f64,
        predicted_rpm: f64,
    },
    MissingAcoustic(&'static str),
    UnsupportedFlowModel(PcFanFlowModel),
    UnsupportedResolvedFlowModel(PcFanFlowModel),
    ResolvedFlowSolutionCountMismatch {
        expected: usize,
        actual: usize,
    },
    ResolvedFlowEvaluationRejected {
        index: usize,
    },
    InvalidPredictionRpm {
        index: usize,
        rpm: f64,
    },
    NonIncreasingPredictionRpms {
        index: usize,
        previous: f64,
        rpm: f64,
    },
}

impl fmt::Display for PcFanMeasuredCorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance(name, value) => {
                write!(
                    f,
                    "PC-fan correlation {name} must be finite and non-negative: {value}"
                )
            }
            Self::InvalidMinimumPoints => {
                write!(f, "PC-fan correlation min_points must be at least one")
            }
            Self::MissingProvenance(dataset) => {
                write!(
                    f,
                    "PC-fan correlation {dataset} provenance must not be empty"
                )
            }
            Self::TooFewPoints {
                dataset,
                actual,
                required,
            } => write!(
                f,
                "PC-fan correlation {dataset} dataset has {actual} points; requires {required}"
            ),
            Self::NonFiniteObservation { dataset, index } => write!(
                f,
                "PC-fan correlation {dataset} observation {index} contains a non-finite value"
            ),
            Self::InvalidRpm {
                dataset,
                index,
                rpm,
            } => write!(
                f,
                "PC-fan correlation {dataset} observation {index} has invalid RPM {rpm}"
            ),
            Self::DuplicateRpm { dataset, index } => write!(
                f,
                "PC-fan correlation {dataset} has duplicate RPM near sorted index {index}"
            ),
            Self::RpmTopologyMismatch {
                index,
                measured_rpm,
                predicted_rpm,
            } => write!(
                f,
                "PC-fan correlation RPM mismatch at {index}: measured {measured_rpm}, predicted {predicted_rpm}"
            ),
            Self::MissingAcoustic(dataset) => write!(
                f,
                "PC-fan correlation {dataset} acoustic observations are incomplete"
            ),
            Self::UnsupportedFlowModel(model) => write!(
                f,
                "PC-fan measured-correlation prediction supports reduced-order flow only, got {model:?}"
            ),
            Self::UnsupportedResolvedFlowModel(model) => write!(
                f,
                "PC-fan resolved-correlation prediction requires CFD flow, got {model:?}"
            ),
            Self::ResolvedFlowSolutionCountMismatch { expected, actual } => write!(
                f,
                "PC-fan resolved-correlation has {actual} solutions for {expected} RPM points"
            ),
            Self::ResolvedFlowEvaluationRejected { index } => write!(
                f,
                "PC-fan resolved-correlation solution {index} failed checked CFD admission"
            ),
            Self::InvalidPredictionRpm { index, rpm } => {
                write!(
                    f,
                    "PC-fan predicted correlation RPM {index} is invalid: {rpm}"
                )
            }
            Self::NonIncreasingPredictionRpms {
                index,
                previous,
                rpm,
            } => write!(
                f,
                "PC-fan predicted correlation RPM {index} is not strictly increasing: {rpm} <= {previous}"
            ),
        }
    }
}

impl std::error::Error for PcFanMeasuredCorrelationError {}

/// RMSE and worst-case error report for a measured/predicted pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredCorrelationReport {
    pub measured_provenance: String,
    pub predicted_provenance: String,
    pub sample_count: usize,
    pub airflow_rmse_cfm: f64,
    pub pressure_rmse_pa: f64,
    pub acoustic_rmse_db: Option<f64>,
    pub airflow_max_abs_error_cfm: f64,
    pub pressure_max_abs_error_pa: f64,
    pub acoustic_max_abs_error_db: Option<f64>,
    pub passed: bool,
}

impl PcFanMeasuredCorrelationReport {
    pub fn is_valid(&self) -> bool {
        self.sample_count > 0
            && !self.measured_provenance.trim().is_empty()
            && !self.predicted_provenance.trim().is_empty()
            && self.airflow_rmse_cfm.is_finite()
            && self.pressure_rmse_pa.is_finite()
            && self.airflow_max_abs_error_cfm.is_finite()
            && self.pressure_max_abs_error_pa.is_finite()
            && self.acoustic_rmse_db.is_none_or(|value| value.is_finite())
            && self
                .acoustic_max_abs_error_db
                .is_none_or(|value| value.is_finite())
            && self.passed
    }
}

/// Compare measured and predicted fan-curve/acoustic observations.
pub fn evaluate_pc_fan_measured_correlation(
    config: &PcFanMeasuredCorrelationConfig,
    measured: &PcFanCorrelationDataset,
    predicted: &PcFanCorrelationDataset,
) -> Result<PcFanMeasuredCorrelationReport, PcFanMeasuredCorrelationError> {
    validate_config(config)?;
    validate_dataset(
        "measured",
        measured,
        config.min_points,
        config.rpm_tolerance,
    )?;
    validate_dataset(
        "predicted",
        predicted,
        config.min_points,
        config.rpm_tolerance,
    )?;

    let measured_acoustic_complete = measured
        .observations
        .iter()
        .all(|observation| observation.acoustic_level_db.is_some());
    let predicted_acoustic_complete = predicted
        .observations
        .iter()
        .all(|observation| observation.acoustic_level_db.is_some());
    let measured_acoustic_partial = measured
        .observations
        .iter()
        .any(|observation| observation.acoustic_level_db.is_some())
        && !measured_acoustic_complete;
    let predicted_acoustic_partial = predicted
        .observations
        .iter()
        .any(|observation| observation.acoustic_level_db.is_some())
        && !predicted_acoustic_complete;
    if measured_acoustic_partial {
        return Err(PcFanMeasuredCorrelationError::MissingAcoustic("measured"));
    }
    if predicted_acoustic_partial {
        return Err(PcFanMeasuredCorrelationError::MissingAcoustic("predicted"));
    }
    if config.require_acoustic && !measured_acoustic_complete {
        return Err(PcFanMeasuredCorrelationError::MissingAcoustic("measured"));
    }
    if measured_acoustic_complete != predicted_acoustic_complete {
        let missing_dataset = if measured_acoustic_complete {
            "predicted"
        } else {
            "measured"
        };
        return Err(PcFanMeasuredCorrelationError::MissingAcoustic(
            missing_dataset,
        ));
    }
    if measured.observations.len() != predicted.observations.len() {
        return Err(PcFanMeasuredCorrelationError::RpmTopologyMismatch {
            index: measured
                .observations
                .len()
                .min(predicted.observations.len()),
            measured_rpm: measured
                .observations
                .last()
                .map_or(0.0, |observation| observation.rpm),
            predicted_rpm: predicted
                .observations
                .last()
                .map_or(0.0, |observation| observation.rpm),
        });
    }

    let mut measured_sorted = measured.observations.clone();
    let mut predicted_sorted = predicted.observations.clone();
    measured_sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));
    predicted_sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));

    let mut airflow_squared_error = 0.0;
    let mut pressure_squared_error = 0.0;
    let mut acoustic_squared_error = 0.0;
    let mut airflow_max_abs_error = 0.0_f64;
    let mut pressure_max_abs_error = 0.0_f64;
    let mut acoustic_max_abs_error = 0.0_f64;
    for (index, (measured, predicted)) in measured_sorted
        .iter()
        .zip(predicted_sorted.iter())
        .enumerate()
    {
        if (measured.rpm - predicted.rpm).abs() > config.rpm_tolerance {
            return Err(PcFanMeasuredCorrelationError::RpmTopologyMismatch {
                index,
                measured_rpm: measured.rpm,
                predicted_rpm: predicted.rpm,
            });
        }
        let airflow_error = measured.airflow_cfm - predicted.airflow_cfm;
        let pressure_error = measured.pressure_rise_pa - predicted.pressure_rise_pa;
        airflow_squared_error += airflow_error * airflow_error;
        pressure_squared_error += pressure_error * pressure_error;
        airflow_max_abs_error = airflow_max_abs_error.max(airflow_error.abs());
        pressure_max_abs_error = pressure_max_abs_error.max(pressure_error.abs());
        if let (Some(measured_acoustic), Some(predicted_acoustic)) =
            (measured.acoustic_level_db, predicted.acoustic_level_db)
        {
            let acoustic_error = measured_acoustic - predicted_acoustic;
            acoustic_squared_error += acoustic_error * acoustic_error;
            acoustic_max_abs_error = acoustic_max_abs_error.max(acoustic_error.abs());
        }
    }

    let sample_count = measured_sorted.len();
    let sample_count_f64 = sample_count as f64;
    let airflow_rmse_cfm = (airflow_squared_error / sample_count_f64).sqrt();
    let pressure_rmse_pa = (pressure_squared_error / sample_count_f64).sqrt();
    let acoustic_rmse_db =
        measured_acoustic_complete.then(|| (acoustic_squared_error / sample_count_f64).sqrt());
    let acoustic_max_abs_error_db = measured_acoustic_complete.then_some(acoustic_max_abs_error);
    let passed = airflow_rmse_cfm <= config.max_airflow_rmse_cfm
        && pressure_rmse_pa <= config.max_pressure_rmse_pa
        && acoustic_rmse_db.is_none_or(|value| value <= config.max_acoustic_rmse_db);

    Ok(PcFanMeasuredCorrelationReport {
        measured_provenance: measured.provenance.clone(),
        predicted_provenance: predicted.provenance.clone(),
        sample_count,
        airflow_rmse_cfm,
        pressure_rmse_pa,
        acoustic_rmse_db,
        airflow_max_abs_error_cfm: airflow_max_abs_error,
        pressure_max_abs_error_pa: pressure_max_abs_error,
        acoustic_max_abs_error_db,
        passed,
    })
}

/// Generate a provenance-labelled reduced-order prediction dataset for a
/// measured-correlation comparison.
///
/// This adapter owns only the solver side of the correlation workflow. It
/// rejects resolved-flow configurations before evaluation and preserves the
/// caller's RPM topology so the existing measured-data contract can compare
/// it with an instrument dataset. The returned provenance must identify the
/// solver run, commit, or artifact rather than being described as measured
/// data.
pub fn predict_pc_fan_correlation_dataset(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    rpms: &[f64],
    provenance: impl Into<String>,
) -> Result<PcFanCorrelationDataset, PcFanMeasuredCorrelationError> {
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(PcFanMeasuredCorrelationError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }

    let provenance = provenance.into();
    if provenance.trim().is_empty() {
        return Err(PcFanMeasuredCorrelationError::MissingProvenance(
            "predicted",
        ));
    }
    if rpms.is_empty() {
        return Err(PcFanMeasuredCorrelationError::TooFewPoints {
            dataset: "predicted",
            actual: 0,
            required: 1,
        });
    }
    for (index, &rpm) in rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanMeasuredCorrelationError::InvalidPredictionRpm { index, rpm });
        }
        if let Some(&previous) = index.checked_sub(1).and_then(|i| rpms.get(i))
            && rpm.total_cmp(&previous) != std::cmp::Ordering::Greater
        {
            return Err(PcFanMeasuredCorrelationError::NonIncreasingPredictionRpms {
                index,
                previous,
                rpm,
            });
        }
    }

    let mut observations = Vec::with_capacity(rpms.len());
    for (index, &rpm) in rpms.iter().enumerate() {
        let mut point_config = config.clone();
        point_config.operating_point.rpm = rpm;
        // A prediction point must not recursively evaluate a separate spread
        // envelope; the caller supplies the complete comparison RPM set.
        point_config.weights.operating_spread_weight = 0.0;
        let evaluation = evaluate_pc_fan(&point_config, design);
        let observation = PcFanCorrelationObservation {
            rpm,
            airflow_cfm: evaluation.airflow_cfm,
            pressure_rise_pa: evaluation.flow.pressure_rise_pa,
            acoustic_level_db: Some(evaluation.acoustic_level_db),
        };
        if !observation.rpm.is_finite()
            || !observation.airflow_cfm.is_finite()
            || !observation.pressure_rise_pa.is_finite()
            || !observation.acoustic_level_db.is_none_or(f64::is_finite)
        {
            return Err(PcFanMeasuredCorrelationError::NonFiniteObservation {
                dataset: "predicted",
                index,
            });
        }
        observations.push(observation);
    }

    Ok(PcFanCorrelationDataset {
        provenance,
        observations,
    })
}

/// Build a provenance-labelled prediction dataset from already checked
/// resolved-flow solutions, one solution for each requested RPM.
///
/// This adapter does not launch a CFD solve and does not infer RPM from a
/// solution that lacks that metadata. Callers must provide the complete
/// RPM-aligned series; every solution is re-admitted through the checked
/// CFD-to-objective boundary before its airflow, pressure, and acoustic
/// observables are published.
pub fn predict_pc_fan_resolved_correlation_dataset(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    rpms: &[f64],
    solutions: &[crate::model::PcFanCfdFlowSolution],
    provenance: impl Into<String>,
) -> Result<PcFanCorrelationDataset, PcFanMeasuredCorrelationError> {
    if !config.cfd.flow_model.is_cfd() {
        return Err(PcFanMeasuredCorrelationError::UnsupportedResolvedFlowModel(
            config.cfd.flow_model,
        ));
    }
    let provenance = provenance.into();
    if provenance.trim().is_empty() {
        return Err(PcFanMeasuredCorrelationError::MissingProvenance(
            "predicted-resolved",
        ));
    }
    validate_prediction_rpms(rpms)?;
    if solutions.len() != rpms.len() {
        return Err(
            PcFanMeasuredCorrelationError::ResolvedFlowSolutionCountMismatch {
                expected: rpms.len(),
                actual: solutions.len(),
            },
        );
    }

    let mut observations = Vec::with_capacity(rpms.len());
    for (index, (&rpm, solution)) in rpms.iter().zip(solutions).enumerate() {
        let mut point_config = config.clone();
        point_config.operating_point.rpm = rpm;
        point_config.weights.operating_spread_weight = 0.0;
        try_evaluate_pc_fan_from_cfd_solution(&point_config, design, solution)
            .map_err(|_| PcFanMeasuredCorrelationError::ResolvedFlowEvaluationRejected { index })?;
        let evaluation = evaluate_pc_fan_from_cfd_solution(&point_config, design, solution);
        let observation = PcFanCorrelationObservation {
            rpm,
            airflow_cfm: evaluation.airflow_cfm,
            pressure_rise_pa: evaluation.flow.pressure_rise_pa,
            acoustic_level_db: Some(evaluation.acoustic_level_db),
        };
        if !observation.rpm.is_finite()
            || !observation.airflow_cfm.is_finite()
            || !observation.pressure_rise_pa.is_finite()
            || !observation.acoustic_level_db.is_none_or(f64::is_finite)
        {
            return Err(PcFanMeasuredCorrelationError::NonFiniteObservation {
                dataset: "predicted-resolved",
                index,
            });
        }
        observations.push(observation);
    }

    Ok(PcFanCorrelationDataset {
        provenance,
        observations,
    })
}

fn validate_prediction_rpms(rpms: &[f64]) -> Result<(), PcFanMeasuredCorrelationError> {
    if rpms.is_empty() {
        return Err(PcFanMeasuredCorrelationError::TooFewPoints {
            dataset: "predicted",
            actual: 0,
            required: 1,
        });
    }
    for (index, &rpm) in rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanMeasuredCorrelationError::InvalidPredictionRpm { index, rpm });
        }
        if let Some(&previous) = index.checked_sub(1).and_then(|i| rpms.get(i))
            && rpm.total_cmp(&previous) != std::cmp::Ordering::Greater
        {
            return Err(PcFanMeasuredCorrelationError::NonIncreasingPredictionRpms {
                index,
                previous,
                rpm,
            });
        }
    }
    Ok(())
}

fn validate_config(
    config: &PcFanMeasuredCorrelationConfig,
) -> Result<(), PcFanMeasuredCorrelationError> {
    for (name, value) in [
        ("rpm_tolerance", config.rpm_tolerance),
        ("max_airflow_rmse_cfm", config.max_airflow_rmse_cfm),
        ("max_pressure_rmse_pa", config.max_pressure_rmse_pa),
        ("max_acoustic_rmse_db", config.max_acoustic_rmse_db),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanMeasuredCorrelationError::InvalidTolerance(name, value));
        }
    }
    if config.min_points == 0 {
        return Err(PcFanMeasuredCorrelationError::InvalidMinimumPoints);
    }
    Ok(())
}

fn validate_dataset(
    name: &'static str,
    dataset: &PcFanCorrelationDataset,
    min_points: usize,
    rpm_tolerance: f64,
) -> Result<(), PcFanMeasuredCorrelationError> {
    if dataset.provenance.trim().is_empty() {
        return Err(PcFanMeasuredCorrelationError::MissingProvenance(name));
    }
    if dataset.observations.len() < min_points {
        return Err(PcFanMeasuredCorrelationError::TooFewPoints {
            dataset: name,
            actual: dataset.observations.len(),
            required: min_points,
        });
    }
    for (index, observation) in dataset.observations.iter().enumerate() {
        if !observation.rpm.is_finite()
            || !observation.airflow_cfm.is_finite()
            || !observation.pressure_rise_pa.is_finite()
            || observation
                .acoustic_level_db
                .is_some_and(|value| !value.is_finite())
        {
            return Err(PcFanMeasuredCorrelationError::NonFiniteObservation {
                dataset: name,
                index,
            });
        }
        if observation.rpm <= 0.0 {
            return Err(PcFanMeasuredCorrelationError::InvalidRpm {
                dataset: name,
                index,
                rpm: observation.rpm,
            });
        }
    }
    let mut sorted = dataset.observations.clone();
    sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));
    for index in 1..sorted.len() {
        if (sorted[index].rpm - sorted[index - 1].rpm).abs() <= rpm_tolerance {
            return Err(PcFanMeasuredCorrelationError::DuplicateRpm {
                dataset: name,
                index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_adapter_produces_a_self_consistent_provenance_dataset() {
        let mut config = PcFanConfig::default_120mm();
        config.mesh = crate::model::PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        let design = config.default_design_variables();
        let predicted = predict_pc_fan_correlation_dataset(
            &config,
            &design,
            &[600.0, 900.0, 1200.0],
            "solver:pcfan:reduced-order:test",
        )
        .expect("reduced-order prediction adapter should evaluate");
        let mut measured = predicted.clone();
        measured.provenance = "measurement:synthetic-parity:test".into();
        let report = evaluate_pc_fan_measured_correlation(
            &PcFanMeasuredCorrelationConfig::default(),
            &measured,
            &predicted,
        )
        .expect("self-consistent prediction should correlate");
        assert!(report.is_valid());
        assert_eq!(report.sample_count, 3);
        assert_eq!(
            report.predicted_provenance,
            "solver:pcfan:reduced-order:test"
        );
    }

    #[test]
    fn prediction_adapter_rejects_invalid_rpm_topology_and_resolved_flow() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        assert_eq!(
            predict_pc_fan_correlation_dataset(&config, &design, &[], "solver:test"),
            Err(PcFanMeasuredCorrelationError::TooFewPoints {
                dataset: "predicted",
                actual: 0,
                required: 1,
            })
        );
        assert_eq!(
            predict_pc_fan_correlation_dataset(&config, &design, &[600.0], " "),
            Err(PcFanMeasuredCorrelationError::MissingProvenance(
                "predicted"
            ))
        );
        assert!(matches!(
            predict_pc_fan_correlation_dataset(
                &config,
                &design,
                &[f64::NAN, 900.0],
                "solver:test",
            ),
            Err(PcFanMeasuredCorrelationError::InvalidPredictionRpm { index: 0, rpm })
                if rpm.is_nan()
        ));
        assert!(matches!(
            predict_pc_fan_correlation_dataset(&config, &design, &[600.0, 600.0], "solver:test",),
            Err(PcFanMeasuredCorrelationError::NonIncreasingPredictionRpms { index: 1, .. })
        ));

        let mut resolved = config;
        resolved.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        assert!(matches!(
            predict_pc_fan_correlation_dataset(&resolved, &design, &[600.0, 900.0], "solver:test",),
            Err(PcFanMeasuredCorrelationError::UnsupportedFlowModel(
                PcFanFlowModel::IncompressibleCfdFull
            ))
        ));
    }

    #[test]
    fn resolved_prediction_adapter_requires_checked_cfd_series() {
        let mut config = PcFanConfig::bench_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 1.0e-4;
        let design = config.default_design_variables();
        let mut solution = crate::solve_pc_fan_cfd_mean_flow(&config, &design);
        solution.converged = true;
        solution.residual = 0.0;

        let predicted = predict_pc_fan_resolved_correlation_dataset(
            &config,
            &design,
            &[600.0],
            std::slice::from_ref(&solution),
            "solver:pcfan:resolved-test",
        )
        .expect("checked resolved solution should adapt");
        assert_eq!(predicted.observations.len(), 1);
        assert_eq!(predicted.provenance, "solver:pcfan:resolved-test");

        let mut malformed = solution.clone();
        malformed.pressure_pa.pop();
        assert!(matches!(
            predict_pc_fan_resolved_correlation_dataset(
                &config,
                &design,
                &[600.0],
                std::slice::from_ref(&malformed),
                "solver:test",
            ),
            Err(PcFanMeasuredCorrelationError::ResolvedFlowEvaluationRejected { index: 0 })
        ));

        let reduced = PcFanConfig::default_120mm();
        assert!(matches!(
            predict_pc_fan_resolved_correlation_dataset(
                &reduced,
                &reduced.default_design_variables(),
                &[600.0],
                &[],
                "solver:test",
            ),
            Err(PcFanMeasuredCorrelationError::UnsupportedResolvedFlowModel(
                PcFanFlowModel::ReducedOrder
            ))
        ));
    }

    fn dataset(provenance: &str, acoustic: bool) -> PcFanCorrelationDataset {
        PcFanCorrelationDataset {
            provenance: provenance.into(),
            observations: [600.0, 900.0, 1200.0]
                .into_iter()
                .enumerate()
                .map(|(index, rpm)| PcFanCorrelationObservation {
                    rpm,
                    airflow_cfm: 20.0 + index as f64 * 5.0,
                    pressure_rise_pa: 100.0 + index as f64 * 20.0,
                    acoustic_level_db: acoustic.then_some(40.0 + index as f64),
                })
                .collect(),
        }
    }

    #[test]
    fn measured_correlation_accepts_reordered_exact_fixture() {
        let measured = dataset("measurement:fan-a:bench-2026-08", true);
        let mut predicted = dataset("solver:pcfan:commit-abc", true);
        predicted.observations.swap(0, 2);
        predicted.observations[0].airflow_cfm += 0.5;
        predicted.observations[1].pressure_rise_pa -= 2.0;
        let report = evaluate_pc_fan_measured_correlation(
            &PcFanMeasuredCorrelationConfig::default(),
            &measured,
            &predicted,
        )
        .expect("valid reordered correlation");

        assert!(report.is_valid());
        assert_eq!(report.sample_count, 3);
        assert!(report.acoustic_rmse_db.is_some());
        assert!(report.airflow_max_abs_error_cfm >= 0.5);
        assert_eq!(
            report.measured_provenance,
            "measurement:fan-a:bench-2026-08"
        );
    }

    #[test]
    fn measured_correlation_rejects_bad_config_provenance_and_values() {
        let measured = dataset("measured", true);
        let predicted = dataset("predicted", true);
        let mut config = PcFanMeasuredCorrelationConfig::default();
        config.max_pressure_rmse_pa = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_measured_correlation(&config, &measured, &predicted),
            Err(PcFanMeasuredCorrelationError::InvalidTolerance(
                "max_pressure_rmse_pa",
                value
            )) if value.is_nan()
        ));

        let mut missing_provenance = measured.clone();
        missing_provenance.provenance.clear();
        assert_eq!(
            evaluate_pc_fan_measured_correlation(
                &PcFanMeasuredCorrelationConfig::default(),
                &missing_provenance,
                &predicted,
            ),
            Err(PcFanMeasuredCorrelationError::MissingProvenance("measured"))
        );

        let mut nonfinite = predicted.clone();
        nonfinite.observations[1].pressure_rise_pa = f64::INFINITY;
        assert_eq!(
            evaluate_pc_fan_measured_correlation(
                &PcFanMeasuredCorrelationConfig::default(),
                &measured,
                &nonfinite,
            ),
            Err(PcFanMeasuredCorrelationError::NonFiniteObservation {
                dataset: "predicted",
                index: 1,
            })
        );
    }

    #[test]
    fn measured_correlation_rejects_duplicate_and_mismatched_rpm_topology() {
        let measured = dataset("measured", true);
        let mut duplicate = dataset("predicted", true);
        duplicate.observations[1].rpm = duplicate.observations[0].rpm;
        assert!(matches!(
            evaluate_pc_fan_measured_correlation(
                &PcFanMeasuredCorrelationConfig::default(),
                &measured,
                &duplicate,
            ),
            Err(PcFanMeasuredCorrelationError::DuplicateRpm {
                dataset: "predicted",
                ..
            })
        ));

        let mut mismatch = dataset("predicted", true);
        mismatch.observations[2].rpm = 1300.0;
        assert!(matches!(
            evaluate_pc_fan_measured_correlation(
                &PcFanMeasuredCorrelationConfig::default(),
                &measured,
                &mismatch,
            ),
            Err(PcFanMeasuredCorrelationError::RpmTopologyMismatch { index: 2, .. })
        ));
    }

    #[test]
    fn measured_correlation_requires_complete_acoustics_when_configured() {
        let measured = dataset("measured", false);
        let predicted = dataset("predicted", false);
        assert_eq!(
            evaluate_pc_fan_measured_correlation(
                &PcFanMeasuredCorrelationConfig::default(),
                &measured,
                &predicted,
            ),
            Err(PcFanMeasuredCorrelationError::MissingAcoustic("measured"))
        );

        let mut config = PcFanMeasuredCorrelationConfig::default();
        config.require_acoustic = false;
        let report = evaluate_pc_fan_measured_correlation(&config, &measured, &predicted)
            .expect("acoustic optional");
        assert!(report.is_valid());
        assert!(report.acoustic_rmse_db.is_none());

        assert_eq!(
            evaluate_pc_fan_measured_correlation(&config, &measured, &dataset("predicted", true)),
            Err(PcFanMeasuredCorrelationError::MissingAcoustic("measured"))
        );

        let mut partial = dataset("measured", true);
        partial.observations[1].acoustic_level_db = None;
        assert_eq!(
            evaluate_pc_fan_measured_correlation(&config, &partial, &predicted),
            Err(PcFanMeasuredCorrelationError::MissingAcoustic("measured"))
        );
    }

    #[test]
    fn measured_correlation_reports_budget_failure_without_claiming_validity() {
        let measured = dataset("measured", true);
        let mut predicted = dataset("predicted", true);
        predicted.observations[0].airflow_cfm += 10.0;
        let report = evaluate_pc_fan_measured_correlation(
            &PcFanMeasuredCorrelationConfig::default(),
            &measured,
            &predicted,
        )
        .expect("well-formed but inaccurate data is reportable");

        assert!(!report.passed);
        assert!(!report.is_valid());
        assert!(report.airflow_rmse_cfm > 2.0);
    }
}
