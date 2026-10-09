//! Bounded calibration fit for caller-supplied propeller measurements.
//!
//! This module deliberately owns the numerical calibration contract, not a
//! dataset or an instrument claim. It fits a scalar affine correction to
//! caller-supplied operating-point samples and reports enough residual
//! evidence for a later fidelity-promotion decision.

use serde::{Deserialize, Serialize};
use std::fmt;

/// One predicted/measured scalar at a declared operating point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCalibrationSample {
    /// Caller-defined operating coordinate, for example advance ratio or RPM.
    pub operating_point: f64,
    /// Uncalibrated model output in the declared channel units.
    pub predicted_value: f64,
    /// Measured output in the same channel units.
    pub measured_value: f64,
}

/// Admission limits for [`calibrate_measured_response`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredCalibrationOptions {
    /// Maximum number of caller-supplied samples.
    pub max_samples: usize,
    /// Minimum finite predictor range required to identify an affine fit.
    pub minimum_predictor_range: f64,
    /// Maximum admitted absolute affine gain.
    pub maximum_absolute_gain: f64,
    /// Maximum admitted fitted RMSE.
    pub maximum_rmse: f64,
    /// Maximum admitted fitted absolute residual.
    pub maximum_absolute_error: f64,
}

impl Default for MeasuredCalibrationOptions {
    fn default() -> Self {
        Self {
            max_samples: 1_024,
            minimum_predictor_range: 1.0e-12,
            maximum_absolute_gain: 100.0,
            maximum_rmse: 1.0,
            maximum_absolute_error: 1.0,
        }
    }
}

/// Per-sample result from the affine calibration fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredCalibrationPoint {
    /// Original operating coordinate.
    pub operating_point: f64,
    /// Original model output.
    pub predicted_value: f64,
    /// Caller-supplied measured output.
    pub measured_value: f64,
    /// Affine-corrected model output.
    pub calibrated_value: f64,
    /// `calibrated_value - measured_value`.
    pub error: f64,
    /// Absolute value of [`Self::error`].
    pub absolute_error: f64,
}

/// Checked affine calibration result.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredCalibrationResult {
    /// Fitted multiplier in `measured ≈ gain * predicted + offset`.
    pub gain: f64,
    /// Fitted additive correction in `measured ≈ gain * predicted + offset`.
    pub offset: f64,
    /// Predictor range used by the fit.
    pub predictor_range: f64,
    /// Per-sample fitted residuals.
    pub points: Vec<MeasuredCalibrationPoint>,
    /// Root-mean-square fitted residual.
    pub rmse: f64,
    /// Largest fitted absolute residual.
    pub maximum_absolute_error: f64,
    /// Coefficient of determination when the measured values have variance.
    pub r_squared: Option<f64>,
    /// Whether configured residual and coefficient gates passed.
    pub passed: bool,
}

impl MeasuredCalibrationResult {
    /// Returns whether all result fields are finite and structurally valid.
    pub fn is_finite(&self) -> bool {
        self.gain.is_finite()
            && self.offset.is_finite()
            && self.predictor_range.is_finite()
            && self.predictor_range > 0.0
            && !self.points.is_empty()
            && self.rmse.is_finite()
            && self.rmse >= 0.0
            && self.maximum_absolute_error.is_finite()
            && self.maximum_absolute_error >= 0.0
            && self
                .r_squared
                .map(|value| value.is_finite())
                .unwrap_or(true)
            && self.points.iter().all(|point| {
                point.operating_point.is_finite()
                    && point.predicted_value.is_finite()
                    && point.measured_value.is_finite()
                    && point.calibrated_value.is_finite()
                    && point.error.is_finite()
                    && point.absolute_error.is_finite()
                    && point.absolute_error >= 0.0
            })
    }
}

/// Failure-closed errors from the calibration contract.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasuredCalibrationError {
    /// Calibration options are malformed.
    InvalidOptions {
        /// Invalid option field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No samples were supplied.
    EmptySamples,
    /// Too few samples to identify gain and offset.
    InsufficientSamples { count: usize, minimum: usize },
    /// More samples were supplied than the declared work budget.
    TooManySamples { count: usize, maximum: usize },
    /// One sample contains an invalid field.
    InvalidSample {
        /// Sample index in caller order.
        index: usize,
        /// Invalid field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Predictor values do not identify a two-parameter affine fit.
    DegeneratePredictor { range: f64 },
    /// The fitted gain exceeds the declared physical/admission bound.
    GainOutOfRange {
        gain: f64,
        maximum_absolute_gain: f64,
    },
    /// A finite input produced non-finite fit arithmetic or output.
    NonFiniteResult,
}

impl fmt::Display for MeasuredCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid measured calibration option {field}: {requirement}"
                )
            }
            Self::EmptySamples => {
                formatter.write_str("measured calibration samples must not be empty")
            }
            Self::InsufficientSamples { count, minimum } => write!(
                formatter,
                "measured calibration has {count} samples, but needs at least {minimum}"
            ),
            Self::TooManySamples { count, maximum } => write!(
                formatter,
                "measured calibration sample count {count} exceeds budget {maximum}"
            ),
            Self::InvalidSample {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid measured calibration sample {index} field {field}: {requirement}"
            ),
            Self::DegeneratePredictor { range } => {
                write!(
                    formatter,
                    "predictor range {range} cannot identify affine calibration"
                )
            }
            Self::GainOutOfRange {
                gain,
                maximum_absolute_gain,
            } => write!(
                formatter,
                "calibration gain {gain} exceeds absolute bound {maximum_absolute_gain}"
            ),
            Self::NonFiniteResult => {
                formatter.write_str("measured calibration result is non-finite")
            }
        }
    }
}

impl std::error::Error for MeasuredCalibrationError {}

/// Fit `measured ≈ gain * predicted + offset` over checked samples.
pub fn calibrate_measured_response(
    samples: &[MeasuredCalibrationSample],
    options: MeasuredCalibrationOptions,
) -> Result<MeasuredCalibrationResult, MeasuredCalibrationError> {
    validate_options(options)?;
    if samples.is_empty() {
        return Err(MeasuredCalibrationError::EmptySamples);
    }
    if samples.len() < 2 {
        return Err(MeasuredCalibrationError::InsufficientSamples {
            count: samples.len(),
            minimum: 2,
        });
    }
    if samples.len() > options.max_samples {
        return Err(MeasuredCalibrationError::TooManySamples {
            count: samples.len(),
            maximum: options.max_samples,
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        validate_sample(index, *sample)?;
    }

    let (minimum, maximum) = samples.iter().map(|sample| sample.predicted_value).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    );
    let predictor_range = maximum - minimum;
    if !predictor_range.is_finite() {
        return Err(MeasuredCalibrationError::NonFiniteResult);
    }
    if predictor_range < options.minimum_predictor_range {
        return Err(MeasuredCalibrationError::DegeneratePredictor {
            range: predictor_range,
        });
    }

    let count = samples.len() as f64;
    let sum_x = samples
        .iter()
        .map(|sample| sample.predicted_value)
        .sum::<f64>();
    let sum_y = samples
        .iter()
        .map(|sample| sample.measured_value)
        .sum::<f64>();
    let sum_xx = samples
        .iter()
        .map(|sample| sample.predicted_value * sample.predicted_value)
        .sum::<f64>();
    let sum_xy = samples
        .iter()
        .map(|sample| sample.predicted_value * sample.measured_value)
        .sum::<f64>();
    if !sum_x.is_finite() || !sum_y.is_finite() || !sum_xx.is_finite() || !sum_xy.is_finite() {
        return Err(MeasuredCalibrationError::NonFiniteResult);
    }
    let denominator = count * sum_xx - sum_x * sum_x;
    let numerator = count * sum_xy - sum_x * sum_y;
    if !denominator.is_finite() || !numerator.is_finite() || denominator <= 0.0 {
        return Err(MeasuredCalibrationError::DegeneratePredictor {
            range: predictor_range,
        });
    }
    let gain = numerator / denominator;
    let offset = (sum_y - gain * sum_x) / count;
    if !gain.is_finite() || !offset.is_finite() {
        return Err(MeasuredCalibrationError::NonFiniteResult);
    }
    if gain.abs() > options.maximum_absolute_gain {
        return Err(MeasuredCalibrationError::GainOutOfRange {
            gain,
            maximum_absolute_gain: options.maximum_absolute_gain,
        });
    }

    let mut points = Vec::with_capacity(samples.len());
    for sample in samples {
        let calibrated_value = gain * sample.predicted_value + offset;
        let error = calibrated_value - sample.measured_value;
        if !calibrated_value.is_finite() || !error.is_finite() {
            return Err(MeasuredCalibrationError::NonFiniteResult);
        }
        points.push(MeasuredCalibrationPoint {
            operating_point: sample.operating_point,
            predicted_value: sample.predicted_value,
            measured_value: sample.measured_value,
            calibrated_value,
            error,
            absolute_error: error.abs(),
        });
    }

    let sum_squared_error = points
        .iter()
        .map(|point| point.error * point.error)
        .sum::<f64>();
    let rmse = (sum_squared_error / count).sqrt();
    let maximum_absolute_error = points
        .iter()
        .map(|point| point.absolute_error)
        .fold(0.0, f64::max);
    let measured_mean = sum_y / count;
    let total_sum_squares = samples
        .iter()
        .map(|sample| {
            let difference = sample.measured_value - measured_mean;
            difference * difference
        })
        .sum::<f64>();
    let r_squared = if total_sum_squares > f64::MIN_POSITIVE {
        Some((1.0 - sum_squared_error / total_sum_squares).clamp(-1.0, 1.0))
    } else {
        None
    };
    let passed =
        rmse <= options.maximum_rmse && maximum_absolute_error <= options.maximum_absolute_error;
    let result = MeasuredCalibrationResult {
        gain,
        offset,
        predictor_range,
        points,
        rmse,
        maximum_absolute_error,
        r_squared,
        passed,
    };
    if !result.is_finite() {
        return Err(MeasuredCalibrationError::NonFiniteResult);
    }
    Ok(result)
}

fn validate_options(options: MeasuredCalibrationOptions) -> Result<(), MeasuredCalibrationError> {
    if options.max_samples < 2 {
        return Err(MeasuredCalibrationError::InvalidOptions {
            field: "max_samples",
            requirement: "at least 2",
        });
    }
    if !options.minimum_predictor_range.is_finite() || options.minimum_predictor_range <= 0.0 {
        return Err(MeasuredCalibrationError::InvalidOptions {
            field: "minimum_predictor_range",
            requirement: "finite and > 0",
        });
    }
    if !options.maximum_absolute_gain.is_finite() || options.maximum_absolute_gain <= 0.0 {
        return Err(MeasuredCalibrationError::InvalidOptions {
            field: "maximum_absolute_gain",
            requirement: "finite and > 0",
        });
    }
    if !options.maximum_rmse.is_finite() || options.maximum_rmse < 0.0 {
        return Err(MeasuredCalibrationError::InvalidOptions {
            field: "maximum_rmse",
            requirement: "finite and >= 0",
        });
    }
    if !options.maximum_absolute_error.is_finite() || options.maximum_absolute_error < 0.0 {
        return Err(MeasuredCalibrationError::InvalidOptions {
            field: "maximum_absolute_error",
            requirement: "finite and >= 0",
        });
    }
    Ok(())
}

fn validate_sample(
    index: usize,
    sample: MeasuredCalibrationSample,
) -> Result<(), MeasuredCalibrationError> {
    if !sample.operating_point.is_finite() {
        return Err(MeasuredCalibrationError::InvalidSample {
            index,
            field: "operating_point",
            requirement: "finite",
        });
    }
    if !sample.predicted_value.is_finite() {
        return Err(MeasuredCalibrationError::InvalidSample {
            index,
            field: "predicted_value",
            requirement: "finite",
        });
    }
    if !sample.measured_value.is_finite() {
        return Err(MeasuredCalibrationError::InvalidSample {
            index,
            field: "measured_value",
            requirement: "finite",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_samples() -> [MeasuredCalibrationSample; 4] {
        [
            MeasuredCalibrationSample {
                operating_point: 0.2,
                predicted_value: 1.0,
                measured_value: 2.5,
            },
            MeasuredCalibrationSample {
                operating_point: 0.4,
                predicted_value: 2.0,
                measured_value: 4.5,
            },
            MeasuredCalibrationSample {
                operating_point: 0.6,
                predicted_value: 3.0,
                measured_value: 6.5,
            },
            MeasuredCalibrationSample {
                operating_point: 0.8,
                predicted_value: 4.0,
                measured_value: 8.5,
            },
        ]
    }

    #[test]
    fn recovers_exact_affine_calibration() {
        let result =
            calibrate_measured_response(&exact_samples(), MeasuredCalibrationOptions::default())
                .expect("exact affine samples should fit");
        assert!(result.is_finite());
        assert!((result.gain - 2.0).abs() < 1.0e-12);
        assert!((result.offset - 0.5).abs() < 1.0e-12);
        assert_eq!(result.rmse, 0.0);
        assert_eq!(result.maximum_absolute_error, 0.0);
        assert_eq!(result.r_squared, Some(1.0));
        assert!(result.passed);
    }

    #[test]
    fn reports_residuals_and_admission_failure() {
        let mut samples = exact_samples();
        samples[2].measured_value += 0.4;
        let result = calibrate_measured_response(
            &samples,
            MeasuredCalibrationOptions {
                maximum_rmse: 0.01,
                maximum_absolute_error: 0.01,
                ..MeasuredCalibrationOptions::default()
            },
        )
        .expect("finite residual fit should return evidence");
        assert!(result.is_finite());
        assert!(result.rmse > 0.0);
        assert!(!result.passed);
        assert!(result.points.iter().any(|point| point.absolute_error > 0.0));
    }

    #[test]
    fn malformed_and_budgeted_inputs_fail_closed() {
        assert!(matches!(
            calibrate_measured_response(&[], MeasuredCalibrationOptions::default()),
            Err(MeasuredCalibrationError::EmptySamples)
        ));
        assert!(matches!(
            calibrate_measured_response(
                &[exact_samples()[0]],
                MeasuredCalibrationOptions::default()
            ),
            Err(MeasuredCalibrationError::InsufficientSamples { .. })
        ));
        assert!(matches!(
            calibrate_measured_response(
                &exact_samples(),
                MeasuredCalibrationOptions {
                    max_samples: 2,
                    ..MeasuredCalibrationOptions::default()
                }
            ),
            Err(MeasuredCalibrationError::TooManySamples { .. })
        ));
        let mut malformed = exact_samples();
        malformed[1].predicted_value = f64::NAN;
        assert!(matches!(
            calibrate_measured_response(&malformed, MeasuredCalibrationOptions::default()),
            Err(MeasuredCalibrationError::InvalidSample {
                field: "predicted_value",
                ..
            })
        ));
    }

    #[test]
    fn degenerate_predictor_and_gain_bounds_fail_closed() {
        let samples = [
            MeasuredCalibrationSample {
                operating_point: 0.0,
                predicted_value: 1.0,
                measured_value: 1.0,
            },
            MeasuredCalibrationSample {
                operating_point: 1.0,
                predicted_value: 1.0,
                measured_value: 2.0,
            },
        ];
        assert!(matches!(
            calibrate_measured_response(&samples, MeasuredCalibrationOptions::default()),
            Err(MeasuredCalibrationError::DegeneratePredictor { .. })
        ));
        assert!(matches!(
            calibrate_measured_response(
                &exact_samples(),
                MeasuredCalibrationOptions {
                    maximum_absolute_gain: 1.0,
                    ..MeasuredCalibrationOptions::default()
                }
            ),
            Err(MeasuredCalibrationError::GainOutOfRange { .. })
        ));
    }
}
