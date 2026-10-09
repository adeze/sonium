//! Checked measured-harmonic correlation over the published spectrum contract.
//!
//! The API compares externally supplied peak-pressure measurements at a
//! declared reference distance with the existing magnitude-only spectrum
//! convention. It provides deterministic matching and error metrics, but does
//! not ship measured data or claim product validation, observer phase,
//! propagation, wake, or resolved CFD/FW-H fidelity.

use std::fmt;

use crate::{HarmonicNoise, PropellerSpectrum};

/// One externally supplied measured harmonic in the spectrum convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredHarmonicPoint {
    /// Harmonic number, where one is the blade-passing-frequency component.
    pub harmonic: usize,
    /// Measured harmonic frequency [Hz].
    pub frequency_hz: f64,
    /// Measured positive peak pressure at the declared reference distance [Pa].
    pub pressure_pa: f64,
}

/// Admission and matching tolerances for measured harmonic correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredHarmonicCorrelationOptions {
    /// Absolute frequency matching tolerance [Hz].
    pub frequency_tolerance_hz: f64,
    /// Maximum admitted pressure RMSE [Pa].
    pub max_rmse_pa: f64,
    /// Maximum admitted absolute pressure error [Pa].
    pub max_abs_error_pa: f64,
    /// Require one measured point for every predicted harmonic.
    pub require_complete_match: bool,
}

impl Default for MeasuredHarmonicCorrelationOptions {
    fn default() -> Self {
        Self {
            frequency_tolerance_hz: 1.0e-9,
            max_rmse_pa: 1.0,
            max_abs_error_pa: 1.0,
            require_complete_match: true,
        }
    }
}

/// One matched predicted/measured harmonic and its pressure errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredHarmonicCorrelationPoint {
    /// Harmonic number used for matching.
    pub harmonic: usize,
    /// Predicted frequency [Hz].
    pub predicted_frequency_hz: f64,
    /// Measured frequency [Hz].
    pub measured_frequency_hz: f64,
    /// Predicted peak pressure [Pa].
    pub predicted_pressure_pa: f64,
    /// Measured peak pressure [Pa].
    pub measured_pressure_pa: f64,
    /// Signed predicted-minus-measured pressure error [Pa].
    pub error_pa: f64,
    /// Absolute pressure error [Pa].
    pub absolute_error_pa: f64,
    /// Relative signed error, absent when measured pressure is zero.
    pub relative_error: Option<f64>,
}

/// Aggregate evidence from checked measured harmonic correlation.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredHarmonicCorrelationResult {
    /// Matched points in caller-provided measurement order.
    pub points: Vec<MeasuredHarmonicCorrelationPoint>,
    /// Root-mean-square pressure error [Pa].
    pub rmse_pa: f64,
    /// Maximum absolute pressure error [Pa].
    pub max_abs_error_pa: f64,
    /// Mean signed pressure error [Pa].
    pub mean_error_pa: f64,
    /// RMSE divided by measured pressure RMS, absent for an all-zero sample.
    pub relative_rmse: Option<f64>,
    /// Whether configured RMSE and maximum-error limits were met.
    pub passed: bool,
}

impl MeasuredHarmonicCorrelationResult {
    /// Returns whether the correlation evidence is finite and structurally valid.
    pub fn is_finite(&self) -> bool {
        !self.points.is_empty()
            && self.rmse_pa.is_finite()
            && self.max_abs_error_pa.is_finite()
            && self.mean_error_pa.is_finite()
            && self
                .relative_rmse
                .map(|value| value.is_finite())
                .unwrap_or(true)
            && self.points.iter().all(|point| {
                point.harmonic > 0
                    && point.predicted_frequency_hz.is_finite()
                    && point.predicted_frequency_hz > 0.0
                    && point.measured_frequency_hz.is_finite()
                    && point.measured_frequency_hz > 0.0
                    && point.predicted_pressure_pa.is_finite()
                    && point.predicted_pressure_pa >= 0.0
                    && point.measured_pressure_pa.is_finite()
                    && point.measured_pressure_pa >= 0.0
                    && point.error_pa.is_finite()
                    && point.absolute_error_pa.is_finite()
                    && point.absolute_error_pa >= 0.0
                    && point
                        .relative_error
                        .map(|value| value.is_finite())
                        .unwrap_or(true)
            })
    }
}

/// Failure-closed errors from measured harmonic matching.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasuredHarmonicCorrelationError {
    /// Correlation options are invalid.
    InvalidOptions {
        /// Option field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The measured data set is empty.
    EmptyMeasurements,
    /// A measured point is malformed.
    InvalidMeasuredPoint {
        /// Point index in caller order.
        index: usize,
        /// Invalid field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The predicted spectrum contains a malformed harmonic.
    InvalidPredictedHarmonic {
        /// Harmonic number, if available.
        harmonic: usize,
    },
    /// A harmonic number is duplicated.
    DuplicateHarmonic {
        /// Duplicated harmonic number.
        harmonic: usize,
        /// Whether the duplicate came from prediction or measurements.
        source: &'static str,
    },
    /// A measurement has no predicted harmonic with the same number.
    MissingPrediction {
        /// Missing harmonic number.
        harmonic: usize,
    },
    /// Complete matching requires a measurement for this prediction.
    MissingMeasurement {
        /// Missing harmonic number.
        harmonic: usize,
    },
    /// Harmonic numbers match but frequencies do not.
    FrequencyMismatch {
        /// Harmonic number.
        harmonic: usize,
        /// Predicted frequency [Hz].
        predicted_hz: f64,
        /// Measured frequency [Hz].
        measured_hz: f64,
        /// Admitted absolute tolerance [Hz].
        tolerance_hz: f64,
    },
    /// The aggregate result is non-finite.
    NonFiniteResult,
}

impl fmt::Display for MeasuredHarmonicCorrelationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => write!(
                formatter,
                "measured correlation option {field}: {requirement}"
            ),
            Self::EmptyMeasurements => write!(formatter, "measured correlation requires data"),
            Self::InvalidMeasuredPoint {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "measured point {index} field {field} must be {requirement}"
            ),
            Self::InvalidPredictedHarmonic { harmonic } => {
                write!(formatter, "predicted harmonic {harmonic} is malformed")
            }
            Self::DuplicateHarmonic { harmonic, source } => {
                write!(formatter, "harmonic {harmonic} is duplicated in {source}")
            }
            Self::MissingPrediction { harmonic } => {
                write!(
                    formatter,
                    "measurement harmonic {harmonic} has no prediction"
                )
            }
            Self::MissingMeasurement { harmonic } => {
                write!(
                    formatter,
                    "prediction harmonic {harmonic} has no measurement"
                )
            }
            Self::FrequencyMismatch {
                harmonic,
                predicted_hz,
                measured_hz,
                tolerance_hz,
            } => write!(
                formatter,
                "harmonic {harmonic} frequency mismatch: predicted {predicted_hz} Hz, measured {measured_hz} Hz, tolerance {tolerance_hz} Hz"
            ),
            Self::NonFiniteResult => write!(formatter, "measured correlation result is non-finite"),
        }
    }
}

impl std::error::Error for MeasuredHarmonicCorrelationError {}

/// Correlate externally supplied measured harmonics with a predicted spectrum.
pub fn correlate_measured_harmonics(
    spectrum: &PropellerSpectrum,
    measurements: &[MeasuredHarmonicPoint],
    options: MeasuredHarmonicCorrelationOptions,
) -> Result<MeasuredHarmonicCorrelationResult, MeasuredHarmonicCorrelationError> {
    validate_options(options)?;
    if measurements.is_empty() {
        return Err(MeasuredHarmonicCorrelationError::EmptyMeasurements);
    }

    validate_predicted_harmonics(&spectrum.harmonics)?;
    validate_measured_harmonics(measurements)?;
    if options.require_complete_match && measurements.len() != spectrum.harmonics.len() {
        let missing = spectrum
            .harmonics
            .iter()
            .find(|prediction| {
                !measurements
                    .iter()
                    .any(|measurement| measurement.harmonic == prediction.harmonic)
            })
            .map(|prediction| prediction.harmonic)
            .unwrap_or(0);
        return Err(MeasuredHarmonicCorrelationError::MissingMeasurement { harmonic: missing });
    }

    let mut points = Vec::with_capacity(measurements.len());
    for measurement in measurements {
        let prediction = spectrum
            .harmonics
            .iter()
            .find(|prediction| prediction.harmonic == measurement.harmonic)
            .ok_or(MeasuredHarmonicCorrelationError::MissingPrediction {
                harmonic: measurement.harmonic,
            })?;
        if (prediction.frequency_hz - measurement.frequency_hz).abs()
            > options.frequency_tolerance_hz
        {
            return Err(MeasuredHarmonicCorrelationError::FrequencyMismatch {
                harmonic: measurement.harmonic,
                predicted_hz: prediction.frequency_hz,
                measured_hz: measurement.frequency_hz,
                tolerance_hz: options.frequency_tolerance_hz,
            });
        }
        let error_pa = prediction.pressure_pa - measurement.pressure_pa;
        points.push(MeasuredHarmonicCorrelationPoint {
            harmonic: measurement.harmonic,
            predicted_frequency_hz: prediction.frequency_hz,
            measured_frequency_hz: measurement.frequency_hz,
            predicted_pressure_pa: prediction.pressure_pa,
            measured_pressure_pa: measurement.pressure_pa,
            error_pa,
            absolute_error_pa: error_pa.abs(),
            relative_error: (measurement.pressure_pa > 0.0)
                .then_some(error_pa / measurement.pressure_pa),
        });
    }

    let count = points.len() as f64;
    let squared_error_sum = points
        .iter()
        .map(|point| point.error_pa * point.error_pa)
        .sum::<f64>();
    let measured_squared_sum = points
        .iter()
        .map(|point| point.measured_pressure_pa * point.measured_pressure_pa)
        .sum::<f64>();
    let rmse_pa = (squared_error_sum / count).sqrt();
    let max_abs_error_pa = points
        .iter()
        .map(|point| point.absolute_error_pa)
        .fold(0.0, f64::max);
    let mean_error_pa = points.iter().map(|point| point.error_pa).sum::<f64>() / count;
    let measured_rms_pa = (measured_squared_sum / count).sqrt();
    let relative_rmse = if measured_rms_pa > 0.0 {
        Some(rmse_pa / measured_rms_pa)
    } else {
        None
    };
    let result = MeasuredHarmonicCorrelationResult {
        points,
        rmse_pa,
        max_abs_error_pa,
        mean_error_pa,
        relative_rmse,
        passed: rmse_pa <= options.max_rmse_pa && max_abs_error_pa <= options.max_abs_error_pa,
    };
    if !result.is_finite() {
        return Err(MeasuredHarmonicCorrelationError::NonFiniteResult);
    }
    Ok(result)
}

fn validate_options(
    options: MeasuredHarmonicCorrelationOptions,
) -> Result<(), MeasuredHarmonicCorrelationError> {
    if !options.frequency_tolerance_hz.is_finite() || options.frequency_tolerance_hz < 0.0 {
        return Err(MeasuredHarmonicCorrelationError::InvalidOptions {
            field: "frequency_tolerance_hz",
            requirement: "finite and >= 0",
        });
    }
    if !options.max_rmse_pa.is_finite() || options.max_rmse_pa < 0.0 {
        return Err(MeasuredHarmonicCorrelationError::InvalidOptions {
            field: "max_rmse_pa",
            requirement: "finite and >= 0",
        });
    }
    if !options.max_abs_error_pa.is_finite() || options.max_abs_error_pa < 0.0 {
        return Err(MeasuredHarmonicCorrelationError::InvalidOptions {
            field: "max_abs_error_pa",
            requirement: "finite and >= 0",
        });
    }
    Ok(())
}

fn validate_predicted_harmonics(
    harmonics: &[HarmonicNoise],
) -> Result<(), MeasuredHarmonicCorrelationError> {
    for (index, harmonic) in harmonics.iter().enumerate() {
        if harmonic.harmonic == 0
            || !harmonic.frequency_hz.is_finite()
            || harmonic.frequency_hz <= 0.0
            || !harmonic.pressure_pa.is_finite()
            || harmonic.pressure_pa < 0.0
        {
            return Err(MeasuredHarmonicCorrelationError::InvalidPredictedHarmonic {
                harmonic: harmonic.harmonic.max(index + 1),
            });
        }
        if harmonics[..index]
            .iter()
            .any(|previous| previous.harmonic == harmonic.harmonic)
        {
            return Err(MeasuredHarmonicCorrelationError::DuplicateHarmonic {
                harmonic: harmonic.harmonic,
                source: "prediction",
            });
        }
    }
    Ok(())
}

fn validate_measured_harmonics(
    measurements: &[MeasuredHarmonicPoint],
) -> Result<(), MeasuredHarmonicCorrelationError> {
    for (index, measurement) in measurements.iter().enumerate() {
        if measurement.harmonic == 0 {
            return Err(MeasuredHarmonicCorrelationError::InvalidMeasuredPoint {
                index,
                field: "harmonic",
                requirement: "non-zero",
            });
        }
        if !measurement.frequency_hz.is_finite() || measurement.frequency_hz <= 0.0 {
            return Err(MeasuredHarmonicCorrelationError::InvalidMeasuredPoint {
                index,
                field: "frequency_hz",
                requirement: "finite and positive",
            });
        }
        if !measurement.pressure_pa.is_finite() || measurement.pressure_pa < 0.0 {
            return Err(MeasuredHarmonicCorrelationError::InvalidMeasuredPoint {
                index,
                field: "pressure_pa",
                requirement: "finite and non-negative",
            });
        }
        if measurements[..index]
            .iter()
            .any(|previous| previous.harmonic == measurement.harmonic)
        {
            return Err(MeasuredHarmonicCorrelationError::DuplicateHarmonic {
                harmonic: measurement.harmonic,
                source: "measurements",
            });
        }
    }
    Ok(())
}
