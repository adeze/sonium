//! Bounded uncertainty propagation for caller-supplied measured calibration.
//!
//! The contract treats prediction uncertainty, measurement uncertainty, and
//! the affine calibration residual as independent first-order Gaussian terms.
//! It reports per-point combined standard deviations and standardized errors,
//! then applies explicit coverage and reduced-chi-squared admission gates.
//! Correlated errors, Bayesian posteriors, instrument traceability, and truth
//! of a public dataset remain outside this reference.

use std::fmt;

use crate::measured_calibration_reference::{
    MeasuredCalibrationError, MeasuredCalibrationOptions, MeasuredCalibrationResult,
    MeasuredCalibrationSample, calibrate_measured_response,
};

/// One measured/model point with independent one-sigma uncertainties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredUncertaintySample {
    /// Caller-defined operating coordinate.
    pub operating_point: f64,
    /// Model prediction in the declared channel unit.
    pub predicted_value: f64,
    /// Measured value in the same declared channel unit.
    pub measured_value: f64,
    /// One-sigma model prediction uncertainty [channel unit].
    pub predicted_standard_deviation: f64,
    /// One-sigma measurement uncertainty [channel unit].
    pub measured_standard_deviation: f64,
}

/// Admission gates for uncertainty propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredUncertaintyOptions {
    /// Underlying affine calibration fit options.
    pub calibration: MeasuredCalibrationOptions,
    /// Maximum admitted per-point absolute z-score.
    pub maximum_z_score: f64,
    /// Minimum fraction of points inside the z-score gate.
    pub minimum_coverage_fraction: f64,
    /// Maximum admitted reduced chi-squared value.
    pub maximum_reduced_chi_squared: f64,
}

impl Default for MeasuredUncertaintyOptions {
    fn default() -> Self {
        Self {
            calibration: MeasuredCalibrationOptions::default(),
            maximum_z_score: 3.0,
            minimum_coverage_fraction: 0.95,
            maximum_reduced_chi_squared: 4.0,
        }
    }
}

/// One uncertainty-propagated calibrated point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredUncertaintyPoint {
    /// Original input sample.
    pub sample: MeasuredUncertaintySample,
    /// Affine-calibrated model value [channel unit].
    pub calibrated_value: f64,
    /// Calibrated-model minus measured value [channel unit].
    pub error: f64,
    /// Combined independent one-sigma uncertainty [channel unit].
    pub combined_standard_deviation: f64,
    /// Absolute standardized error `|error|/sigma`.
    pub z_score: f64,
    /// Whether this point satisfies `z_score <= maximum_z_score`.
    pub within_limit: bool,
}

/// Checked uncertainty-propagation result.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredUncertaintyResult {
    /// Affine fit used to transform model predictions.
    pub calibration: MeasuredCalibrationResult,
    /// Per-point uncertainty envelope and standardized residuals.
    pub points: Vec<MeasuredUncertaintyPoint>,
    /// Maximum absolute z-score across points.
    pub maximum_z_score: f64,
    /// Fraction of points inside the per-point z-score gate.
    pub coverage_fraction: f64,
    /// `sum(z²) / max(n - 2, 1)`.
    pub reduced_chi_squared: f64,
    /// Whether all configured uncertainty gates passed.
    pub passed: bool,
}

impl MeasuredUncertaintyResult {
    /// Returns whether all result fields are finite and structurally valid.
    pub fn is_finite(&self) -> bool {
        self.calibration.is_finite()
            && !self.points.is_empty()
            && self.points.iter().all(|point| {
                point.sample.operating_point.is_finite()
                    && point.sample.predicted_value.is_finite()
                    && point.sample.measured_value.is_finite()
                    && point.sample.predicted_standard_deviation.is_finite()
                    && point.sample.predicted_standard_deviation > 0.0
                    && point.sample.measured_standard_deviation.is_finite()
                    && point.sample.measured_standard_deviation > 0.0
                    && point.calibrated_value.is_finite()
                    && point.error.is_finite()
                    && point.combined_standard_deviation.is_finite()
                    && point.combined_standard_deviation > 0.0
                    && point.z_score.is_finite()
                    && point.z_score >= 0.0
            })
            && self.maximum_z_score.is_finite()
            && self.maximum_z_score >= 0.0
            && self.coverage_fraction.is_finite()
            && (0.0..=1.0).contains(&self.coverage_fraction)
            && self.reduced_chi_squared.is_finite()
            && self.reduced_chi_squared >= 0.0
    }
}

/// Failure-closed uncertainty propagation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasuredUncertaintyError {
    /// An uncertainty option is outside its declared domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No samples were supplied.
    EmptySamples,
    /// Sample count exceeds the calibration work budget.
    TooManySamples { count: usize, maximum: usize },
    /// A sample contains an invalid value.
    InvalidSample {
        /// Zero-based sample index.
        index: usize,
        /// Field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The underlying affine calibration failed.
    Calibration(MeasuredCalibrationError),
    /// A derived uncertainty field became non-finite.
    NonFiniteResult { field: &'static str },
}

impl fmt::Display for MeasuredUncertaintyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid uncertainty option {field}: {requirement}"
                )
            }
            Self::EmptySamples => formatter.write_str("uncertainty samples must not be empty"),
            Self::TooManySamples { count, maximum } => write!(
                formatter,
                "uncertainty sample count {count} exceeds calibration budget {maximum}"
            ),
            Self::InvalidSample {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "uncertainty sample {index} has invalid {field}: {requirement}"
            ),
            Self::Calibration(error) => {
                write!(formatter, "uncertainty calibration failed: {error}")
            }
            Self::NonFiniteResult { field } => {
                write!(formatter, "uncertainty result field {field} is non-finite")
            }
        }
    }
}

impl std::error::Error for MeasuredUncertaintyError {}

impl From<MeasuredCalibrationError> for MeasuredUncertaintyError {
    fn from(error: MeasuredCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Propagate independent uncertainty through the checked affine calibration.
pub fn evaluate_measured_uncertainty(
    samples: &[MeasuredUncertaintySample],
    options: MeasuredUncertaintyOptions,
) -> Result<MeasuredUncertaintyResult, MeasuredUncertaintyError> {
    validate_options(options)?;
    if samples.is_empty() {
        return Err(MeasuredUncertaintyError::EmptySamples);
    }
    if samples.len() > options.calibration.max_samples {
        return Err(MeasuredUncertaintyError::TooManySamples {
            count: samples.len(),
            maximum: options.calibration.max_samples,
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        validate_sample(index, *sample)?;
    }

    let calibration_samples: Vec<MeasuredCalibrationSample> = samples
        .iter()
        .map(|sample| MeasuredCalibrationSample {
            operating_point: sample.operating_point,
            predicted_value: sample.predicted_value,
            measured_value: sample.measured_value,
        })
        .collect();
    let calibration = calibrate_measured_response(&calibration_samples, options.calibration)?;
    let mut points = Vec::with_capacity(samples.len());
    let mut sum_squared_z = 0.0;
    let mut maximum_z_score: f64 = 0.0;
    let mut within_limit_count = 0usize;

    for (sample, calibration_point) in samples.iter().zip(calibration.points.iter()) {
        let prediction_variance = (calibration.gain * sample.predicted_standard_deviation).powi(2);
        let measurement_variance = sample.measured_standard_deviation.powi(2);
        let residual_variance = calibration.rmse.powi(2);
        let combined_variance = prediction_variance + measurement_variance + residual_variance;
        if !combined_variance.is_finite() || combined_variance <= 0.0 {
            return Err(MeasuredUncertaintyError::NonFiniteResult {
                field: "combined_variance",
            });
        }
        let combined_standard_deviation = combined_variance.sqrt();
        let z_score = calibration_point.absolute_error / combined_standard_deviation;
        let within_limit = z_score <= options.maximum_z_score;
        if within_limit {
            within_limit_count += 1;
        }
        sum_squared_z += z_score * z_score;
        maximum_z_score = maximum_z_score.max(z_score);
        points.push(MeasuredUncertaintyPoint {
            sample: *sample,
            calibrated_value: calibration_point.calibrated_value,
            error: calibration_point.error,
            combined_standard_deviation,
            z_score,
            within_limit,
        });
    }

    let sample_count = points.len() as f64;
    let coverage_fraction = within_limit_count as f64 / sample_count;
    let degrees_of_freedom = (points.len() as f64 - 2.0).max(1.0);
    let reduced_chi_squared = sum_squared_z / degrees_of_freedom;
    let passed = maximum_z_score <= options.maximum_z_score
        && coverage_fraction >= options.minimum_coverage_fraction
        && reduced_chi_squared <= options.maximum_reduced_chi_squared;
    let result = MeasuredUncertaintyResult {
        calibration,
        points,
        maximum_z_score,
        coverage_fraction,
        reduced_chi_squared,
        passed,
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(MeasuredUncertaintyError::NonFiniteResult {
            field: "measured_uncertainty_result",
        })
    }
}

fn validate_options(options: MeasuredUncertaintyOptions) -> Result<(), MeasuredUncertaintyError> {
    for (field, value, requirement) in [
        ("maximum_z_score", options.maximum_z_score, "finite and > 0"),
        (
            "minimum_coverage_fraction",
            options.minimum_coverage_fraction,
            "finite and in [0, 1]",
        ),
        (
            "maximum_reduced_chi_squared",
            options.maximum_reduced_chi_squared,
            "finite and > 0",
        ),
    ] {
        let valid = match field {
            "minimum_coverage_fraction" => value.is_finite() && (0.0..=1.0).contains(&value),
            _ => value.is_finite() && value > 0.0,
        };
        if !valid {
            return Err(MeasuredUncertaintyError::InvalidOptions { field, requirement });
        }
    }
    Ok(())
}

fn validate_sample(
    index: usize,
    sample: MeasuredUncertaintySample,
) -> Result<(), MeasuredUncertaintyError> {
    for (field, value) in [
        ("operating_point", sample.operating_point),
        ("predicted_value", sample.predicted_value),
        ("measured_value", sample.measured_value),
    ] {
        if !value.is_finite() {
            return Err(MeasuredUncertaintyError::InvalidSample {
                index,
                field,
                requirement: "finite",
            });
        }
    }
    for (field, value) in [
        (
            "predicted_standard_deviation",
            sample.predicted_standard_deviation,
        ),
        (
            "measured_standard_deviation",
            sample.measured_standard_deviation,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MeasuredUncertaintyError::InvalidSample {
                index,
                field,
                requirement: "finite and > 0",
            });
        }
    }
    Ok(())
}
