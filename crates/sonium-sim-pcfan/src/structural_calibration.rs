//! Provenance-labelled PC-fan thermo-structural calibration contract.
//!
//! The reduced-order evaluator already exposes thermal, stress, growth, and
//! fatigue observables. This module provides the independent comparison
//! boundary needed before those observables can be used as product-calibration
//! evidence. It does not acquire measurements or claim that a synthetic
//! fixture is a measured product.

use crate::evaluation::try_evaluate_pc_fan;
use crate::model::{PcFanConfig, PcFanDesignVariables, PcFanFlowModel};
use serde::{Deserialize, Serialize};
use std::fmt;

const METRIC_COUNT: usize = 4;

/// One RPM-aligned thermal/structural observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanStructuralCalibrationObservation {
    pub rpm: f64,
    pub winding_temperature_c: f64,
    pub maximum_stress_pa: f64,
    pub thermal_expansion_m: f64,
    pub fatigue_alternating_stress_pa: f64,
}

/// A provenance-labelled measured or predicted structural dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanStructuralCalibrationDataset {
    /// Measurement record, fixture identifier, or solver run identity.
    pub provenance: String,
    pub observations: Vec<PcFanStructuralCalibrationObservation>,
}

/// Acceptance budgets for independent thermo-structural calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanStructuralCalibrationConfig {
    pub rpm_tolerance: f64,
    pub max_winding_temperature_rmse_c: f64,
    pub max_stress_rmse_pa: f64,
    pub max_thermal_expansion_rmse_m: f64,
    pub max_fatigue_stress_rmse_pa: f64,
    pub max_relative_error: f64,
    pub min_points: usize,
}

impl Default for PcFanStructuralCalibrationConfig {
    fn default() -> Self {
        Self {
            rpm_tolerance: 1.0e-6,
            max_winding_temperature_rmse_c: 5.0,
            max_stress_rmse_pa: 2.0e6,
            max_thermal_expansion_rmse_m: 1.0e-4,
            max_fatigue_stress_rmse_pa: 2.0e6,
            max_relative_error: 0.25,
            min_points: 3,
        }
    }
}

/// Rejection reasons for malformed calibration data or failed admission.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanStructuralCalibrationError {
    InvalidTolerance {
        name: &'static str,
        value: f64,
    },
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
    InvalidPhysicalObservation {
        dataset: &'static str,
        index: usize,
        field: &'static str,
        value: f64,
    },
    DuplicateRpm {
        dataset: &'static str,
        index: usize,
    },
    NonIncreasingPredictionRpms {
        index: usize,
        previous: f64,
        rpm: f64,
    },
    RpmTopologyMismatch {
        index: usize,
        measured_rpm: f64,
        predicted_rpm: f64,
    },
    UnsupportedFlowModel(PcFanFlowModel),
    EvaluationRejected(String),
}

impl fmt::Display for PcFanStructuralCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance { name, value } => write!(
                formatter,
                "PC-fan structural-calibration tolerance {name} must be finite and non-negative, got {value}"
            ),
            Self::InvalidMinimumPoints => {
                write!(
                    formatter,
                    "PC-fan structural calibration min_points must be at least one"
                )
            }
            Self::MissingProvenance(dataset) => write!(
                formatter,
                "PC-fan structural-calibration {dataset} provenance must not be empty"
            ),
            Self::TooFewPoints {
                dataset,
                actual,
                required,
            } => write!(
                formatter,
                "PC-fan structural-calibration {dataset} has {actual} points; requires {required}"
            ),
            Self::NonFiniteObservation { dataset, index } => write!(
                formatter,
                "PC-fan structural-calibration {dataset} observation {index} contains a non-finite value"
            ),
            Self::InvalidRpm {
                dataset,
                index,
                rpm,
            } => write!(
                formatter,
                "PC-fan structural-calibration {dataset} observation {index} has invalid RPM {rpm}"
            ),
            Self::InvalidPhysicalObservation {
                dataset,
                index,
                field,
                value,
            } => write!(
                formatter,
                "PC-fan structural-calibration {dataset} observation {index} has invalid {field} {value}"
            ),
            Self::DuplicateRpm { dataset, index } => write!(
                formatter,
                "PC-fan structural-calibration {dataset} has duplicate RPM near sorted index {index}"
            ),
            Self::NonIncreasingPredictionRpms {
                index,
                previous,
                rpm,
            } => write!(
                formatter,
                "PC-fan structural-calibration predicted RPM {index} is not strictly increasing: {rpm} <= {previous}"
            ),
            Self::RpmTopologyMismatch {
                index,
                measured_rpm,
                predicted_rpm,
            } => write!(
                formatter,
                "PC-fan structural-calibration RPM mismatch at {index}: measured {measured_rpm}, predicted {predicted_rpm}"
            ),
            Self::UnsupportedFlowModel(model) => write!(
                formatter,
                "PC-fan structural calibration prediction supports reduced-order flow only, got {model:?}"
            ),
            Self::EvaluationRejected(error) => write!(
                formatter,
                "PC-fan structural calibration prediction evaluation was rejected: {error}"
            ),
        }
    }
}

impl std::error::Error for PcFanStructuralCalibrationError {}

/// RMSE and worst-relative-error report for a measured/predicted pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanStructuralCalibrationReport {
    pub measured_provenance: String,
    pub predicted_provenance: String,
    pub sample_count: usize,
    pub winding_temperature_rmse_c: f64,
    pub maximum_stress_rmse_pa: f64,
    pub thermal_expansion_rmse_m: f64,
    pub fatigue_alternating_stress_rmse_pa: f64,
    pub winding_temperature_max_relative_error: f64,
    pub maximum_stress_max_relative_error: f64,
    pub thermal_expansion_max_relative_error: f64,
    pub fatigue_alternating_stress_max_relative_error: f64,
    pub passed: bool,
}

impl PcFanStructuralCalibrationReport {
    /// Whether the report is internally complete and passes every configured
    /// observable gate.
    pub fn is_valid(&self) -> bool {
        self.sample_count > 0
            && !self.measured_provenance.trim().is_empty()
            && !self.predicted_provenance.trim().is_empty()
            && [
                self.winding_temperature_rmse_c,
                self.maximum_stress_rmse_pa,
                self.thermal_expansion_rmse_m,
                self.fatigue_alternating_stress_rmse_pa,
                self.winding_temperature_max_relative_error,
                self.maximum_stress_max_relative_error,
                self.thermal_expansion_max_relative_error,
                self.fatigue_alternating_stress_max_relative_error,
            ]
            .into_iter()
            .all(f64::is_finite)
            && self.passed
    }
}

/// Compare RPM-aligned structural measurements with model predictions.
pub fn evaluate_pc_fan_structural_calibration(
    config: &PcFanStructuralCalibrationConfig,
    measured: &PcFanStructuralCalibrationDataset,
    predicted: &PcFanStructuralCalibrationDataset,
) -> Result<PcFanStructuralCalibrationReport, PcFanStructuralCalibrationError> {
    validate_config(config)?;
    validate_dataset(config, "measured", measured)?;
    validate_dataset(config, "predicted", predicted)?;

    if measured.observations.len() != predicted.observations.len() {
        return Err(PcFanStructuralCalibrationError::RpmTopologyMismatch {
            index: measured
                .observations
                .len()
                .min(predicted.observations.len()),
            measured_rpm: measured.observations.last().map_or(0.0, |point| point.rpm),
            predicted_rpm: predicted.observations.last().map_or(0.0, |point| point.rpm),
        });
    }

    let mut measured_sorted = measured.observations.clone();
    let mut predicted_sorted = predicted.observations.clone();
    measured_sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));
    predicted_sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));

    let mut squared_errors = [0.0_f64; METRIC_COUNT];
    let mut maximum_relative_errors = [0.0_f64; METRIC_COUNT];
    for (index, (measured_point, predicted_point)) in measured_sorted
        .iter()
        .zip(predicted_sorted.iter())
        .enumerate()
    {
        if (measured_point.rpm - predicted_point.rpm).abs() > config.rpm_tolerance {
            return Err(PcFanStructuralCalibrationError::RpmTopologyMismatch {
                index,
                measured_rpm: measured_point.rpm,
                predicted_rpm: predicted_point.rpm,
            });
        }

        for (metric_index, (measured_value, predicted_value)) in
            metric_values(measured_point, predicted_point)
                .into_iter()
                .enumerate()
        {
            let error = measured_value - predicted_value;
            squared_errors[metric_index] += error * error;
            let scale = measured_value.abs().max(predicted_value.abs());
            let relative_error = if scale <= f64::EPSILON {
                if error.abs() <= f64::EPSILON {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                error.abs() / scale
            };
            maximum_relative_errors[metric_index] =
                maximum_relative_errors[metric_index].max(relative_error);
        }
    }

    let sample_count = measured_sorted.len() as f64;
    let rmses = squared_errors.map(|value| (value / sample_count).sqrt());
    let passed = rmses[0] <= config.max_winding_temperature_rmse_c
        && rmses[1] <= config.max_stress_rmse_pa
        && rmses[2] <= config.max_thermal_expansion_rmse_m
        && rmses[3] <= config.max_fatigue_stress_rmse_pa
        && maximum_relative_errors
            .into_iter()
            .all(|value| value <= config.max_relative_error);

    Ok(PcFanStructuralCalibrationReport {
        measured_provenance: measured.provenance.clone(),
        predicted_provenance: predicted.provenance.clone(),
        sample_count: measured_sorted.len(),
        winding_temperature_rmse_c: rmses[0],
        maximum_stress_rmse_pa: rmses[1],
        thermal_expansion_rmse_m: rmses[2],
        fatigue_alternating_stress_rmse_pa: rmses[3],
        winding_temperature_max_relative_error: maximum_relative_errors[0],
        maximum_stress_max_relative_error: maximum_relative_errors[1],
        thermal_expansion_max_relative_error: maximum_relative_errors[2],
        fatigue_alternating_stress_max_relative_error: maximum_relative_errors[3],
        passed,
    })
}

/// Generate a provenance-labelled reduced-order prediction dataset.
pub fn predict_pc_fan_structural_calibration_dataset(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    rpms: &[f64],
    provenance: impl Into<String>,
) -> Result<PcFanStructuralCalibrationDataset, PcFanStructuralCalibrationError> {
    let provenance = provenance.into();
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(PcFanStructuralCalibrationError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if provenance.trim().is_empty() {
        return Err(PcFanStructuralCalibrationError::MissingProvenance(
            "predicted",
        ));
    }
    if rpms.is_empty() {
        return Err(PcFanStructuralCalibrationError::TooFewPoints {
            dataset: "predicted",
            actual: 0,
            required: 1,
        });
    }

    for (index, &rpm) in rpms.iter().enumerate() {
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(PcFanStructuralCalibrationError::InvalidRpm {
                dataset: "predicted",
                index,
                rpm,
            });
        }
        if let Some(&previous) = index.checked_sub(1).and_then(|position| rpms.get(position))
            && rpm.total_cmp(&previous) != std::cmp::Ordering::Greater
        {
            return Err(
                PcFanStructuralCalibrationError::NonIncreasingPredictionRpms {
                    index,
                    previous,
                    rpm,
                },
            );
        }
    }

    let mut observations = Vec::with_capacity(rpms.len());
    for (index, &rpm) in rpms.iter().enumerate() {
        let mut point_config = config.clone();
        point_config.operating_point.rpm = rpm;
        point_config.operating_point_spread.rpms.clear();
        point_config.weights.operating_spread_weight = 0.0;
        let evaluation = try_evaluate_pc_fan(&point_config, design).map_err(|error| {
            PcFanStructuralCalibrationError::EvaluationRejected(error.to_string())
        })?;
        let observation = PcFanStructuralCalibrationObservation {
            rpm,
            winding_temperature_c: evaluation.thermal.winding_temperature_c,
            maximum_stress_pa: evaluation.structural.maximum_stress_pa,
            thermal_expansion_m: evaluation.structural.thermal_expansion_m,
            fatigue_alternating_stress_pa: evaluation.structural.fatigue_alternating_stress_pa,
        };
        validate_observation("predicted", index, &observation)?;
        observations.push(observation);
    }

    Ok(PcFanStructuralCalibrationDataset {
        provenance,
        observations,
    })
}

fn metric_values(
    measured: &PcFanStructuralCalibrationObservation,
    predicted: &PcFanStructuralCalibrationObservation,
) -> [(f64, f64); METRIC_COUNT] {
    [
        (
            measured.winding_temperature_c,
            predicted.winding_temperature_c,
        ),
        (measured.maximum_stress_pa, predicted.maximum_stress_pa),
        (measured.thermal_expansion_m, predicted.thermal_expansion_m),
        (
            measured.fatigue_alternating_stress_pa,
            predicted.fatigue_alternating_stress_pa,
        ),
    ]
}

fn validate_config(
    config: &PcFanStructuralCalibrationConfig,
) -> Result<(), PcFanStructuralCalibrationError> {
    for (name, value) in [
        ("rpm_tolerance", config.rpm_tolerance),
        (
            "max_winding_temperature_rmse_c",
            config.max_winding_temperature_rmse_c,
        ),
        ("max_stress_rmse_pa", config.max_stress_rmse_pa),
        (
            "max_thermal_expansion_rmse_m",
            config.max_thermal_expansion_rmse_m,
        ),
        (
            "max_fatigue_stress_rmse_pa",
            config.max_fatigue_stress_rmse_pa,
        ),
        ("max_relative_error", config.max_relative_error),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanStructuralCalibrationError::InvalidTolerance { name, value });
        }
    }
    if config.min_points == 0 {
        return Err(PcFanStructuralCalibrationError::InvalidMinimumPoints);
    }
    Ok(())
}

fn validate_dataset(
    config: &PcFanStructuralCalibrationConfig,
    dataset_name: &'static str,
    dataset: &PcFanStructuralCalibrationDataset,
) -> Result<(), PcFanStructuralCalibrationError> {
    if dataset.provenance.trim().is_empty() {
        return Err(PcFanStructuralCalibrationError::MissingProvenance(
            dataset_name,
        ));
    }
    if dataset.observations.len() < config.min_points {
        return Err(PcFanStructuralCalibrationError::TooFewPoints {
            dataset: dataset_name,
            actual: dataset.observations.len(),
            required: config.min_points,
        });
    }
    for (index, observation) in dataset.observations.iter().enumerate() {
        validate_observation(dataset_name, index, observation)?;
    }

    let mut sorted = dataset.observations.clone();
    sorted.sort_by(|left, right| left.rpm.total_cmp(&right.rpm));
    for (index, pair) in sorted.windows(2).enumerate() {
        if (pair[1].rpm - pair[0].rpm).abs() <= config.rpm_tolerance {
            return Err(PcFanStructuralCalibrationError::DuplicateRpm {
                dataset: dataset_name,
                index,
            });
        }
    }
    Ok(())
}

fn validate_observation(
    dataset: &'static str,
    index: usize,
    observation: &PcFanStructuralCalibrationObservation,
) -> Result<(), PcFanStructuralCalibrationError> {
    if ![
        observation.rpm,
        observation.winding_temperature_c,
        observation.maximum_stress_pa,
        observation.thermal_expansion_m,
        observation.fatigue_alternating_stress_pa,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        return Err(PcFanStructuralCalibrationError::NonFiniteObservation { dataset, index });
    }
    if observation.rpm <= 0.0 {
        return Err(PcFanStructuralCalibrationError::InvalidRpm {
            dataset,
            index,
            rpm: observation.rpm,
        });
    }
    for (field, value) in [
        ("winding_temperature_c", observation.winding_temperature_c),
        ("maximum_stress_pa", observation.maximum_stress_pa),
        ("thermal_expansion_m", observation.thermal_expansion_m),
        (
            "fatigue_alternating_stress_pa",
            observation.fatigue_alternating_stress_pa,
        ),
    ] {
        let valid = if field == "winding_temperature_c" {
            value > -273.15
        } else {
            value >= 0.0
        };
        if !valid {
            return Err(
                PcFanStructuralCalibrationError::InvalidPhysicalObservation {
                    dataset,
                    index,
                    field,
                    value,
                },
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction() -> PcFanStructuralCalibrationDataset {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        predict_pc_fan_structural_calibration_dataset(
            &config,
            &design,
            &[600.0, 900.0, 1200.0],
            "solver:pcfan:reduced-order:test",
        )
        .unwrap()
    }

    #[test]
    fn self_consistent_structural_calibration_passes_and_reports_all_metrics() {
        let predicted = prediction();
        let mut measured = predicted.clone();
        measured.provenance = "measurement:synthetic-parity:test".into();
        for (index, observation) in measured.observations.iter_mut().enumerate() {
            observation.winding_temperature_c += 0.05 * index as f64;
            observation.maximum_stress_pa += 1.0;
            observation.thermal_expansion_m += 1.0e-9;
            observation.fatigue_alternating_stress_pa += 1.0;
        }

        let report = evaluate_pc_fan_structural_calibration(
            &PcFanStructuralCalibrationConfig::default(),
            &measured,
            &predicted,
        )
        .unwrap();
        assert!(report.is_valid());
        assert_eq!(report.sample_count, 3);
        assert!(report.winding_temperature_rmse_c > 0.0);
        assert!(report.maximum_stress_rmse_pa > 0.0);
        assert!(report.thermal_expansion_rmse_m > 0.0);
        assert!(report.fatigue_alternating_stress_rmse_pa > 0.0);
    }

    #[test]
    fn calibration_is_order_independent_but_rpm_alignment_is_checked() {
        let predicted = prediction();
        let mut measured = predicted.clone();
        measured.provenance = "measurement:synthetic-order:test".into();
        measured.observations.reverse();
        assert!(
            evaluate_pc_fan_structural_calibration(
                &PcFanStructuralCalibrationConfig::default(),
                &measured,
                &predicted,
            )
            .unwrap()
            .is_valid()
        );

        measured.observations[0].rpm += 25.0;
        assert!(matches!(
            evaluate_pc_fan_structural_calibration(
                &PcFanStructuralCalibrationConfig::default(),
                &measured,
                &predicted,
            ),
            Err(PcFanStructuralCalibrationError::RpmTopologyMismatch { .. })
        ));
    }

    #[test]
    fn calibration_rejects_bad_config_provenance_points_and_physics() {
        let predicted = prediction();
        let mut config = PcFanStructuralCalibrationConfig::default();
        config.max_relative_error = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_structural_calibration(&config, &predicted, &predicted),
            Err(PcFanStructuralCalibrationError::InvalidTolerance {
                name: "max_relative_error",
                ..
            })
        ));

        let mut missing_provenance = predicted.clone();
        missing_provenance.provenance = " ".into();
        assert!(matches!(
            evaluate_pc_fan_structural_calibration(
                &PcFanStructuralCalibrationConfig::default(),
                &missing_provenance,
                &predicted,
            ),
            Err(PcFanStructuralCalibrationError::MissingProvenance(
                "measured"
            ))
        ));

        let mut invalid = predicted.clone();
        invalid.observations[0].maximum_stress_pa = -1.0;
        assert!(matches!(
            evaluate_pc_fan_structural_calibration(
                &PcFanStructuralCalibrationConfig::default(),
                &invalid,
                &predicted,
            ),
            Err(
                PcFanStructuralCalibrationError::InvalidPhysicalObservation {
                    field: "maximum_stress_pa",
                    ..
                }
            )
        ));
    }

    #[test]
    fn calibration_fails_when_an_observable_exceeds_absolute_or_relative_budget() {
        let predicted = prediction();
        let mut measured = predicted.clone();
        measured.provenance = "measurement:synthetic-failure:test".into();
        measured.observations[1].maximum_stress_pa *= 2.0;
        let mut config = PcFanStructuralCalibrationConfig::default();
        config.max_stress_rmse_pa = 0.0;
        config.max_relative_error = 0.0;
        let report =
            evaluate_pc_fan_structural_calibration(&config, &measured, &predicted).unwrap();
        assert!(!report.passed);
        assert!(!report.is_valid());
        assert!(report.maximum_stress_max_relative_error > 0.0);
    }

    #[test]
    fn prediction_adapter_rejects_invalid_rpm_provenance_and_resolved_flow() {
        let config = PcFanConfig::bench_120mm();
        let design = config.default_design_variables();
        assert!(matches!(
            predict_pc_fan_structural_calibration_dataset(&config, &design, &[], "solver:test"),
            Err(PcFanStructuralCalibrationError::TooFewPoints {
                dataset: "predicted",
                actual: 0,
                required: 1,
            })
        ));
        assert!(matches!(
            predict_pc_fan_structural_calibration_dataset(&config, &design, &[600.0], " ",),
            Err(PcFanStructuralCalibrationError::MissingProvenance(
                "predicted"
            ))
        ));
        assert!(matches!(
            predict_pc_fan_structural_calibration_dataset(
                &config,
                &design,
                &[600.0, 600.0],
                "solver:test",
            ),
            Err(PcFanStructuralCalibrationError::NonIncreasingPredictionRpms { index: 1, .. })
        ));
        let mut resolved = config;
        resolved.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        assert!(matches!(
            predict_pc_fan_structural_calibration_dataset(
                &resolved,
                &design,
                &[600.0, 900.0, 1200.0],
                "solver:test",
            ),
            Err(PcFanStructuralCalibrationError::UnsupportedFlowModel(
                PcFanFlowModel::IncompressibleCfdFull
            ))
        ));
    }

    #[test]
    fn calibration_dataset_and_report_serde_round_trip() {
        let predicted = prediction();
        let dataset_json = serde_json::to_vec(&predicted).unwrap();
        let round_trip =
            serde_json::from_slice::<PcFanStructuralCalibrationDataset>(&dataset_json).unwrap();
        assert_eq!(round_trip.provenance, predicted.provenance);
        assert_eq!(round_trip.observations.len(), predicted.observations.len());
        for (actual, expected) in round_trip.observations.iter().zip(&predicted.observations) {
            assert_eq!(actual.rpm, expected.rpm);
            assert!((actual.winding_temperature_c - expected.winding_temperature_c).abs() < 1e-12);
            assert!((actual.maximum_stress_pa - expected.maximum_stress_pa).abs() < 1e-12);
            assert!((actual.thermal_expansion_m - expected.thermal_expansion_m).abs() < 1e-12);
            assert!(
                (actual.fatigue_alternating_stress_pa - expected.fatigue_alternating_stress_pa)
                    .abs()
                    < 1e-12
            );
        }
        let report = evaluate_pc_fan_structural_calibration(
            &PcFanStructuralCalibrationConfig::default(),
            &predicted,
            &predicted,
        )
        .unwrap();
        let report_json = serde_json::to_vec(&report).unwrap();
        assert_eq!(
            serde_json::from_slice::<PcFanStructuralCalibrationReport>(&report_json).unwrap(),
            report
        );
    }
}
