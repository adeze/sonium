//! Independent measured-product validation for PC-fan predictions.
//!
//! The ingestion contract in [`crate::measured_product`] proves that a data
//! record has traceable metadata and a checksum. This module keeps the
//! comparison boundary equally strict: predictions must have their own
//! provenance, must use the same RPM topology, and are compared in the
//! declared physical units with explicit absolute RMSE budgets.

use crate::measured_product::{
    PcFanMeasuredProductDataset, PcFanMeasuredProductError, PcFanMeasuredProductObservation,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Absolute RMSE budgets for an independent measured-product comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredProductValidationConfig {
    /// Minimum number of matched operating points.
    pub min_points: usize,
    /// Airflow RMSE budget in CFM.
    pub max_airflow_rmse_cfm: f64,
    /// Pressure-drop RMSE budget in pascals.
    pub max_pressure_drop_rmse_pa: f64,
    /// Acoustic RMSE budget in dB. Acoustic values are required when true.
    pub max_acoustic_rmse_db: f64,
    /// Winding-temperature RMSE budget in degrees Celsius.
    pub max_winding_temperature_rmse_c: f64,
    /// Maximum-stress RMSE budget in pascals.
    pub max_stress_rmse_pa: f64,
    /// Thermal-expansion RMSE budget in metres.
    pub max_thermal_expansion_rmse_m: f64,
    /// Alternating-fatigue-stress RMSE budget in pascals.
    pub max_fatigue_alternating_stress_rmse_pa: f64,
    /// Require an acoustic measurement and prediction at every point.
    pub require_acoustic: bool,
}

impl Default for PcFanMeasuredProductValidationConfig {
    fn default() -> Self {
        Self {
            min_points: 3,
            max_airflow_rmse_cfm: 2.0,
            max_pressure_drop_rmse_pa: 5.0,
            max_acoustic_rmse_db: 3.0,
            max_winding_temperature_rmse_c: 3.0,
            max_stress_rmse_pa: 100_000.0,
            max_thermal_expansion_rmse_m: 5.0e-6,
            max_fatigue_alternating_stress_rmse_pa: 50_000.0,
            require_acoustic: true,
        }
    }
}

/// Failure while admitting or comparing independent measured-product data.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanMeasuredProductValidationError {
    /// A tolerance or minimum-point setting is invalid.
    InvalidConfig {
        /// Configuration field name.
        field: &'static str,
        /// Invalid value, represented as a scalar for diagnostics.
        value: f64,
    },
    /// The measured dataset failed its evidence or physical validation.
    MeasuredDataset(PcFanMeasuredProductError),
    /// The prediction provenance is absent or pretends to be measured data.
    InvalidPredictionProvenance(String),
    /// The two records do not contain the same number of operating points.
    PredictionCount { expected: usize, actual: usize },
    /// An operating point does not match the measured RPM topology.
    RpmMismatch {
        /// Zero-based row index.
        index: usize,
        /// Measured RPM.
        expected: f64,
        /// Predicted RPM.
        actual: f64,
    },
    /// A required acoustic value is absent from one side of the comparison.
    MissingAcoustic { index: usize },
    /// A prediction contains a non-finite value.
    NonFinitePrediction { index: usize, field: &'static str },
    /// Subtracting two finite observations produced a non-finite residual.
    NonFiniteComparison { index: usize, field: &'static str },
    /// A prediction contains a physically invalid non-negative quantity.
    InvalidPrediction {
        /// Zero-based row index.
        index: usize,
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
}

impl fmt::Display for PcFanMeasuredProductValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, value } => {
                write!(
                    formatter,
                    "invalid measured-product validation {field}={value}"
                )
            }
            Self::MeasuredDataset(error) => write!(formatter, "measured dataset rejected: {error}"),
            Self::InvalidPredictionProvenance(provenance) => write!(
                formatter,
                "prediction provenance must be independent of measured-product evidence: {provenance:?}"
            ),
            Self::PredictionCount { expected, actual } => write!(
                formatter,
                "prediction has {actual} points; measured dataset has {expected}"
            ),
            Self::RpmMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "prediction RPM mismatch at row {index}: expected {expected}, got {actual}"
            ),
            Self::MissingAcoustic { index } => {
                write!(
                    formatter,
                    "missing acoustic value at comparison row {index}"
                )
            }
            Self::NonFinitePrediction { index, field } => {
                write!(formatter, "prediction row {index} has non-finite {field}")
            }
            Self::NonFiniteComparison { index, field } => write!(
                formatter,
                "comparison row {index} has non-finite residual for {field}"
            ),
            Self::InvalidPrediction {
                index,
                field,
                value,
            } => write!(
                formatter,
                "prediction row {index} has invalid {field}={value}"
            ),
        }
    }
}

impl std::error::Error for PcFanMeasuredProductValidationError {}

impl From<PcFanMeasuredProductError> for PcFanMeasuredProductValidationError {
    fn from(error: PcFanMeasuredProductError) -> Self {
        Self::MeasuredDataset(error)
    }
}

/// RMSE report for a measured-product comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredProductValidationReport {
    /// Provenance assembled from the independently ingested measured record.
    pub measured_provenance: String,
    /// Provenance supplied by the simulation or prediction producer.
    pub prediction_provenance: String,
    /// Number of matched operating points.
    pub sample_count: usize,
    /// Airflow RMSE in CFM.
    pub airflow_rmse_cfm: f64,
    /// Pressure-drop RMSE in pascals.
    pub pressure_drop_rmse_pa: f64,
    /// Acoustic RMSE in dB, if acoustic validation was enabled.
    pub acoustic_rmse_db: Option<f64>,
    /// Winding-temperature RMSE in degrees Celsius.
    pub winding_temperature_rmse_c: f64,
    /// Maximum-stress RMSE in pascals.
    pub stress_rmse_pa: f64,
    /// Thermal-expansion RMSE in metres.
    pub thermal_expansion_rmse_m: f64,
    /// Alternating-fatigue-stress RMSE in pascals.
    pub fatigue_alternating_stress_rmse_pa: f64,
    /// True only when every configured budget is satisfied.
    pub passed: bool,
}

impl PcFanMeasuredProductValidationReport {
    /// Return the gate result using a readable predicate.
    pub fn is_valid(&self) -> bool {
        self.passed
    }
}

/// Compare independently ingested measured data with simulation predictions.
///
/// The prediction rows use the same fixed-unit observation type as the
/// measured adapter, but carry no measured evidence themselves. The separate
/// provenance argument is mandatory and rejects measured/synthetic labels so
/// a solver fixture cannot be accidentally promoted to product evidence.
pub fn validate_pc_fan_measured_product_predictions(
    measured: &PcFanMeasuredProductDataset,
    predictions: &[PcFanMeasuredProductObservation],
    prediction_provenance: &str,
    config: &PcFanMeasuredProductValidationConfig,
) -> Result<PcFanMeasuredProductValidationReport, PcFanMeasuredProductValidationError> {
    validate_config(config)?;
    measured.validate(config.min_points)?;
    validate_prediction_provenance(prediction_provenance)?;

    if predictions.len() != measured.observations.len() {
        return Err(PcFanMeasuredProductValidationError::PredictionCount {
            expected: measured.observations.len(),
            actual: predictions.len(),
        });
    }

    let mut airflow = Vec::with_capacity(predictions.len());
    let mut pressure_drop = Vec::with_capacity(predictions.len());
    let mut acoustic = Vec::with_capacity(predictions.len());
    let mut winding_temperature = Vec::with_capacity(predictions.len());
    let mut stress = Vec::with_capacity(predictions.len());
    let mut thermal_expansion = Vec::with_capacity(predictions.len());
    let mut fatigue = Vec::with_capacity(predictions.len());

    for (index, (measured, prediction)) in measured.observations.iter().zip(predictions).enumerate()
    {
        validate_prediction_observation(index, prediction)?;
        if measured.rpm.total_cmp(&prediction.rpm) != std::cmp::Ordering::Equal {
            return Err(PcFanMeasuredProductValidationError::RpmMismatch {
                index,
                expected: measured.rpm,
                actual: prediction.rpm,
            });
        }

        airflow.push(comparison_residual(
            index,
            "airflow_cfm",
            prediction.airflow_cfm,
            measured.airflow_cfm,
        )?);
        pressure_drop.push(comparison_residual(
            index,
            "pressure_drop_pa",
            prediction.pressure_drop_pa,
            measured.pressure_drop_pa,
        )?);
        winding_temperature.push(comparison_residual(
            index,
            "winding_temperature_c",
            prediction.winding_temperature_c,
            measured.winding_temperature_c,
        )?);
        stress.push(comparison_residual(
            index,
            "maximum_stress_pa",
            prediction.maximum_stress_pa,
            measured.maximum_stress_pa,
        )?);
        thermal_expansion.push(comparison_residual(
            index,
            "thermal_expansion_m",
            prediction.thermal_expansion_m,
            measured.thermal_expansion_m,
        )?);
        fatigue.push(comparison_residual(
            index,
            "fatigue_alternating_stress_pa",
            prediction.fatigue_alternating_stress_pa,
            measured.fatigue_alternating_stress_pa,
        )?);

        if config.require_acoustic {
            let (Some(measured), Some(prediction)) =
                (measured.acoustic_level_db, prediction.acoustic_level_db)
            else {
                return Err(PcFanMeasuredProductValidationError::MissingAcoustic { index });
            };
            acoustic.push(comparison_residual(
                index,
                "acoustic_level_db",
                prediction,
                measured,
            )?);
        }
    }

    let acoustic_rmse_db = config.require_acoustic.then(|| root_mean_square(&acoustic));
    let report = PcFanMeasuredProductValidationReport {
        measured_provenance: measured.evidence.provenance(),
        prediction_provenance: prediction_provenance.to_owned(),
        sample_count: predictions.len(),
        airflow_rmse_cfm: root_mean_square(&airflow),
        pressure_drop_rmse_pa: root_mean_square(&pressure_drop),
        acoustic_rmse_db,
        winding_temperature_rmse_c: root_mean_square(&winding_temperature),
        stress_rmse_pa: root_mean_square(&stress),
        thermal_expansion_rmse_m: root_mean_square(&thermal_expansion),
        fatigue_alternating_stress_rmse_pa: root_mean_square(&fatigue),
        passed: false,
    };

    let passed = report.airflow_rmse_cfm <= config.max_airflow_rmse_cfm
        && report.pressure_drop_rmse_pa <= config.max_pressure_drop_rmse_pa
        && report
            .acoustic_rmse_db
            .is_none_or(|value| value <= config.max_acoustic_rmse_db)
        && report.winding_temperature_rmse_c <= config.max_winding_temperature_rmse_c
        && report.stress_rmse_pa <= config.max_stress_rmse_pa
        && report.thermal_expansion_rmse_m <= config.max_thermal_expansion_rmse_m
        && report.fatigue_alternating_stress_rmse_pa
            <= config.max_fatigue_alternating_stress_rmse_pa;

    Ok(PcFanMeasuredProductValidationReport { passed, ..report })
}

fn validate_config(
    config: &PcFanMeasuredProductValidationConfig,
) -> Result<(), PcFanMeasuredProductValidationError> {
    if config.min_points == 0 {
        return Err(PcFanMeasuredProductValidationError::InvalidConfig {
            field: "min_points",
            value: 0.0,
        });
    }
    for (field, value) in [
        ("max_airflow_rmse_cfm", config.max_airflow_rmse_cfm),
        (
            "max_pressure_drop_rmse_pa",
            config.max_pressure_drop_rmse_pa,
        ),
        ("max_acoustic_rmse_db", config.max_acoustic_rmse_db),
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
            "max_fatigue_alternating_stress_rmse_pa",
            config.max_fatigue_alternating_stress_rmse_pa,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanMeasuredProductValidationError::InvalidConfig { field, value });
        }
    }
    Ok(())
}

fn validate_prediction_provenance(
    provenance: &str,
) -> Result<(), PcFanMeasuredProductValidationError> {
    let normalized = provenance.trim().to_ascii_lowercase();
    // Reserved provenance markers are rejected anywhere in the producer
    // label. A wrapper such as `solver:synthetic:fixture` is no more
    // independent than the shorter `synthetic:fixture` form.
    if normalized.is_empty()
        || normalized.contains("measured-product")
        || normalized.contains("synthetic")
    {
        return Err(
            PcFanMeasuredProductValidationError::InvalidPredictionProvenance(provenance.to_owned()),
        );
    }
    Ok(())
}

fn validate_prediction_observation(
    index: usize,
    observation: &PcFanMeasuredProductObservation,
) -> Result<(), PcFanMeasuredProductValidationError> {
    let finite_values = [
        ("rpm", observation.rpm),
        ("airflow_cfm", observation.airflow_cfm),
        ("pressure_drop_pa", observation.pressure_drop_pa),
        ("winding_temperature_c", observation.winding_temperature_c),
        ("maximum_stress_pa", observation.maximum_stress_pa),
        ("thermal_expansion_m", observation.thermal_expansion_m),
        (
            "fatigue_alternating_stress_pa",
            observation.fatigue_alternating_stress_pa,
        ),
    ];
    for (field, value) in finite_values {
        if !value.is_finite() {
            return Err(PcFanMeasuredProductValidationError::NonFinitePrediction { index, field });
        }
    }
    if let Some(value) = observation.acoustic_level_db
        && !value.is_finite()
    {
        return Err(PcFanMeasuredProductValidationError::NonFinitePrediction {
            index,
            field: "acoustic_level_db",
        });
    }
    for (field, value) in [
        ("rpm", observation.rpm),
        ("airflow_cfm", observation.airflow_cfm),
        ("winding_temperature_c", observation.winding_temperature_c),
        ("maximum_stress_pa", observation.maximum_stress_pa),
        ("thermal_expansion_m", observation.thermal_expansion_m),
        (
            "fatigue_alternating_stress_pa",
            observation.fatigue_alternating_stress_pa,
        ),
    ] {
        if value < 0.0 || (field == "rpm" && value <= 0.0) {
            return Err(PcFanMeasuredProductValidationError::InvalidPrediction {
                index,
                field,
                value,
            });
        }
    }
    Ok(())
}

fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
        / (values.len() as f64).sqrt()
}

fn comparison_residual(
    index: usize,
    field: &'static str,
    prediction: f64,
    measured: f64,
) -> Result<f64, PcFanMeasuredProductValidationError> {
    let residual = prediction - measured;
    if residual.is_finite() {
        Ok(residual)
    } else {
        Err(PcFanMeasuredProductValidationError::NonFiniteComparison { index, field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measured_product::PcFanMeasuredProductEvidence;

    fn measured_dataset() -> PcFanMeasuredProductDataset {
        PcFanMeasuredProductDataset {
            evidence: PcFanMeasuredProductEvidence {
                product_id: "fan-120mm".into(),
                serial_number: "SN-42".into(),
                instrument_id: "daq-7".into(),
                calibration_certificate: "cal-2026".into(),
                acquisition_utc: "2026-08-24T12:00:00Z".into(),
                source_uri: "https://example.invalid/fan.csv".into(),
                raw_data_sha256: "0".repeat(64),
            },
            observations: vec![
                observation(600.0, 42.0, 18.0, Some(31.0)),
                observation(900.0, 48.0, 25.0, Some(36.0)),
                observation(1200.0, 52.0, 31.0, Some(42.0)),
            ],
        }
    }

    fn observation(
        rpm: f64,
        airflow_cfm: f64,
        pressure_drop_pa: f64,
        acoustic_level_db: Option<f64>,
    ) -> PcFanMeasuredProductObservation {
        PcFanMeasuredProductObservation {
            rpm,
            airflow_cfm,
            pressure_drop_pa,
            acoustic_level_db,
            winding_temperature_c: 35.0 + rpm / 300.0,
            maximum_stress_pa: 1.0e6 + rpm * 100.0,
            thermal_expansion_m: rpm * 1.0e-8,
            fatigue_alternating_stress_pa: 400_000.0 + rpm * 100.0,
        }
    }

    fn config() -> PcFanMeasuredProductValidationConfig {
        PcFanMeasuredProductValidationConfig {
            max_airflow_rmse_cfm: 0.1,
            max_pressure_drop_rmse_pa: 0.1,
            max_acoustic_rmse_db: 0.1,
            max_winding_temperature_rmse_c: 0.1,
            max_stress_rmse_pa: 0.1,
            max_thermal_expansion_rmse_m: 0.1,
            max_fatigue_alternating_stress_rmse_pa: 0.1,
            ..Default::default()
        }
    }

    #[test]
    fn exact_independent_prediction_passes_and_reports_provenance() {
        let measured = measured_dataset();
        let predictions = measured.observations.clone();
        let report = validate_pc_fan_measured_product_predictions(
            &measured,
            &predictions,
            "resolved-cfd:cpu:v1",
            &config(),
        )
        .unwrap();
        assert!(report.is_valid());
        assert_eq!(report.sample_count, 3);
        assert_eq!(report.airflow_rmse_cfm, 0.0);
        assert!(report.measured_provenance.starts_with("measured-product:"));
        assert_eq!(report.prediction_provenance, "resolved-cfd:cpu:v1");
    }

    #[test]
    fn bounded_prediction_reports_rmse_and_fails_only_exceeded_budget() {
        let measured = measured_dataset();
        let mut predictions = measured.observations.clone();
        predictions[0].airflow_cfm += 0.3;
        let report = validate_pc_fan_measured_product_predictions(
            &measured,
            &predictions,
            "resolved-cfd:cpu:v1",
            &config(),
        )
        .unwrap();
        assert!(!report.is_valid());
        assert!((report.airflow_rmse_cfm - 0.3 / 3.0_f64.sqrt()).abs() < 1.0e-12);
        assert_eq!(report.pressure_drop_rmse_pa, 0.0);
    }

    #[test]
    fn validation_rejects_provenance_that_could_be_circular() {
        let measured = measured_dataset();
        let predictions = measured.observations.clone();
        for provenance in [
            "",
            "measured-product:copy",
            "synthetic:fixture",
            "solver:measured-product:copy",
            "solver:synthetic:fixture",
        ] {
            assert!(matches!(
                validate_pc_fan_measured_product_predictions(
                    &measured,
                    &predictions,
                    provenance,
                    &config(),
                ),
                Err(PcFanMeasuredProductValidationError::InvalidPredictionProvenance(_))
            ));
        }
    }

    #[test]
    fn validation_rejects_count_rpm_and_nonfinite_prediction_errors() {
        let measured = measured_dataset();
        let mut predictions = measured.observations[..2].to_vec();
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured,
                &predictions,
                "resolved-cfd:cpu:v1",
                &config(),
            ),
            Err(PcFanMeasuredProductValidationError::PredictionCount { .. })
        ));
        predictions = measured.observations.clone();
        predictions[1].rpm = 901.0;
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured,
                &predictions,
                "resolved-cfd:cpu:v1",
                &config(),
            ),
            Err(PcFanMeasuredProductValidationError::RpmMismatch { index: 1, .. })
        ));
        predictions[1].rpm = 900.0;
        predictions[2].airflow_cfm = f64::NAN;
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured,
                &predictions,
                "resolved-cfd:cpu:v1",
                &config(),
            ),
            Err(PcFanMeasuredProductValidationError::NonFinitePrediction {
                index: 2,
                field: "airflow_cfm"
            })
        ));
    }

    #[test]
    fn validation_rejects_missing_required_acoustic_and_invalid_config() {
        let mut measured = measured_dataset();
        measured.observations[1].acoustic_level_db = None;
        let predictions = measured.observations.clone();
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured,
                &predictions,
                "resolved-cfd:cpu:v1",
                &config(),
            ),
            Err(PcFanMeasuredProductValidationError::MissingAcoustic { index: 1 })
        ));

        let mut invalid = config();
        invalid.max_airflow_rmse_cfm = f64::NAN;
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured_dataset(),
                &measured_dataset().observations,
                "resolved-cfd:cpu:v1",
                &invalid,
            ),
            Err(PcFanMeasuredProductValidationError::InvalidConfig {
                field: "max_airflow_rmse_cfm",
                ..
            })
        ));
    }

    #[test]
    fn validation_rejects_finite_inputs_with_nonfinite_subtraction() {
        let mut measured = measured_dataset();
        measured.observations[0].pressure_drop_pa = -f64::MAX;
        let mut predictions = measured.observations.clone();
        predictions[0].pressure_drop_pa = f64::MAX;
        assert!(matches!(
            validate_pc_fan_measured_product_predictions(
                &measured,
                &predictions,
                "resolved-cfd:cpu:v1",
                &config(),
            ),
            Err(PcFanMeasuredProductValidationError::NonFiniteComparison {
                index: 0,
                field: "pressure_drop_pa"
            })
        ));
    }

    #[test]
    fn report_serde_round_trip_preserves_gate_and_provenance() {
        let measured = measured_dataset();
        let report = validate_pc_fan_measured_product_predictions(
            &measured,
            &measured.observations,
            "resolved-cfd:cpu:v1",
            &config(),
        )
        .unwrap();
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: PcFanMeasuredProductValidationReport =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, report);
    }
}
