//! Independent complex-directivity reference correlation.
//!
//! The exported directivity grid is normalized by the complex on-axis
//! pressure.  This module keeps that normalization and the speaker-frame
//! angle convention explicit when a measured or published reference is used
//! for promotion.  Reference frequencies must remain inside the simulated
//! grid; [`ComplexDirectivityGrid::try_interpolate`] otherwise holds the
//! nearest frequency edge, which would silently turn extrapolation into a
//! passing comparison.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_xem::ComplexDirectivityGrid;
use std::collections::BTreeSet;
use std::fmt;

/// Current serialized schema for independent directivity references.
pub const DIRECTIVITY_REFERENCE_SCHEMA_VERSION: u32 = 1;
const NORMALIZED_ON_AXIS_TOLERANCE: f64 = 1.0e-6;

/// One normalized complex far-field directivity observation.
///
/// Angles follow the exported speaker-frame convention: vertical angle is
/// polar angle from `+up` in `[0, 180]` degrees, and horizontal angle is
/// azimuth from `+right` toward `+forward` in `[0, 360)` degrees.  The
/// pressure is dimensionless because both sides of the comparison are
/// normalized by the complex on-axis pressure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectivityReferencePoint {
    pub frequency_hz: f64,
    pub vertical_angle_deg: f64,
    pub horizontal_angle_deg: f64,
    pub normalized_pressure: Complex64,
}

/// Versioned independent directivity observations and their provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectivityReference {
    pub schema_version: u32,
    pub provenance: String,
    pub points: Vec<DirectivityReferencePoint>,
}

/// Error thresholds for complex directivity correlation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectivityReferenceTolerances {
    /// RMSE of complex error normalized by the reference magnitude floor.
    pub max_rmse_relative_error: f64,
    /// Maximum pointwise relative error using the same floor.
    pub max_relative_error: f64,
    /// Maximum absolute complex error in normalized-pressure units.
    pub max_absolute_error: f64,
    /// Non-zero scale for angular nulls and other near-zero references.
    pub absolute_floor: f64,
}

impl Default for DirectivityReferenceTolerances {
    fn default() -> Self {
        Self {
            max_rmse_relative_error: 5.0e-2,
            max_relative_error: 1.5e-1,
            max_absolute_error: 1.0e-1,
            absolute_floor: 1.0e-6,
        }
    }
}

/// Correlation statistics retained for promotion and audit reporting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectivityCorrelationReport {
    pub samples: usize,
    pub rmse_relative_error: f64,
    pub maximum_relative_error: f64,
    pub maximum_absolute_error: f64,
    pub rmse_tolerance: f64,
    pub maximum_relative_tolerance: f64,
    pub maximum_absolute_tolerance: f64,
    pub passed: bool,
}

/// Invalid reference data or an invalid correlation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectivityCorrelationError(pub String);

impl fmt::Display for DirectivityCorrelationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DirectivityCorrelationError {}

impl DirectivityReference {
    /// Validate schema, frame coordinates, finite complex data, and anchors.
    pub fn validate(&self) -> Result<(), DirectivityCorrelationError> {
        if self.schema_version != DIRECTIVITY_REFERENCE_SCHEMA_VERSION {
            return Err(DirectivityCorrelationError(format!(
                "directivity reference schema must be v{}",
                DIRECTIVITY_REFERENCE_SCHEMA_VERSION
            )));
        }
        if self.provenance.trim().is_empty() || self.points.is_empty() {
            return Err(DirectivityCorrelationError(
                "directivity reference needs provenance and at least one point".into(),
            ));
        }

        let mut identities = BTreeSet::new();
        let mut frequencies = BTreeSet::new();
        let mut on_axis_frequencies = BTreeSet::new();
        for point in &self.points {
            if !point.frequency_hz.is_finite()
                || point.frequency_hz <= 0.0
                || !point.vertical_angle_deg.is_finite()
                || !(0.0..=180.0).contains(&point.vertical_angle_deg)
                || !point.horizontal_angle_deg.is_finite()
                || !(0.0..360.0).contains(&point.horizontal_angle_deg)
                || !point.normalized_pressure.re.is_finite()
                || !point.normalized_pressure.im.is_finite()
            {
                return Err(DirectivityCorrelationError(
                    "directivity reference point has invalid frequency, angle, or complex value"
                        .into(),
                ));
            }
            let identity = (
                canonical_bits(point.frequency_hz),
                canonical_bits(point.vertical_angle_deg),
                canonical_bits(point.horizontal_angle_deg),
            );
            if !identities.insert(identity) {
                return Err(DirectivityCorrelationError(
                    "directivity reference contains duplicate frequency/angle points".into(),
                ));
            }
            let frequency_bits = canonical_bits(point.frequency_hz);
            frequencies.insert(frequency_bits);
            let is_on_axis = (point.vertical_angle_deg - 90.0).abs() <= 1.0e-9
                && (point.horizontal_angle_deg - 90.0).abs() <= 1.0e-9;
            if is_on_axis {
                if (point.normalized_pressure - Complex64::new(1.0, 0.0)).norm()
                    > NORMALIZED_ON_AXIS_TOLERANCE
                {
                    return Err(DirectivityCorrelationError(
                        "directivity on-axis anchors must be normalized to 1+0i".into(),
                    ));
                }
                on_axis_frequencies.insert(frequency_bits);
            }
        }
        if frequencies != on_axis_frequencies {
            return Err(DirectivityCorrelationError(
                "directivity reference needs one on-axis anchor per frequency".into(),
            ));
        }
        Ok(())
    }
}

impl DirectivityReferenceTolerances {
    fn validate(self) -> Result<(), DirectivityCorrelationError> {
        if !self.absolute_floor.is_finite()
            || self.absolute_floor <= 0.0
            || [
                self.max_rmse_relative_error,
                self.max_relative_error,
                self.max_absolute_error,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(DirectivityCorrelationError(
                "directivity tolerances must be finite, non-negative, and use a positive floor"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Correlate a normalized complex directivity grid against independent data.
///
/// The returned report is `passed == false` when finite data exceeds a gate;
/// malformed data, duplicate observations, missing on-axis anchors, and
/// out-of-support frequencies are hard errors.  This distinction lets a
/// caller retain a failed physical comparison without treating it as an
/// invalid artifact.
pub fn correlate_directivity_reference(
    grid: &ComplexDirectivityGrid,
    reference: &DirectivityReference,
    tolerances: DirectivityReferenceTolerances,
) -> Result<DirectivityCorrelationReport, DirectivityCorrelationError> {
    reference.validate()?;
    tolerances.validate()?;

    let frequencies = grid.frequencies();
    let minimum_frequency = *frequencies.first().ok_or_else(|| {
        DirectivityCorrelationError("directivity grid has no frequency support".into())
    })?;
    let maximum_frequency = *frequencies.last().ok_or_else(|| {
        DirectivityCorrelationError("directivity grid has no frequency support".into())
    })?;

    let mut squared_relative_error = 0.0;
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f64;
    for point in &reference.points {
        if point.frequency_hz < minimum_frequency || point.frequency_hz > maximum_frequency {
            return Err(DirectivityCorrelationError(format!(
                "directivity reference frequency {} is outside grid support [{}, {}]",
                point.frequency_hz, minimum_frequency, maximum_frequency
            )));
        }
        let predicted = grid
            .try_interpolate(
                point.vertical_angle_deg.to_radians(),
                point.horizontal_angle_deg.to_radians(),
                point.frequency_hz,
            )
            .map_err(|error| DirectivityCorrelationError(error.to_string()))?;
        let absolute_error = (predicted - point.normalized_pressure).norm();
        let relative_error = absolute_error
            / point
                .normalized_pressure
                .norm()
                .max(tolerances.absolute_floor);
        squared_relative_error += relative_error * relative_error;
        maximum_relative_error = maximum_relative_error.max(relative_error);
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
    }

    let samples = reference.points.len();
    let rmse_relative_error = (squared_relative_error / samples as f64).sqrt();
    Ok(DirectivityCorrelationReport {
        samples,
        rmse_relative_error,
        maximum_relative_error,
        maximum_absolute_error,
        rmse_tolerance: tolerances.max_rmse_relative_error,
        maximum_relative_tolerance: tolerances.max_relative_error,
        maximum_absolute_tolerance: tolerances.max_absolute_error,
        passed: rmse_relative_error <= tolerances.max_rmse_relative_error
            && maximum_relative_error <= tolerances.max_relative_error
            && maximum_absolute_error <= tolerances.max_absolute_error,
    })
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::HostMatrix;

    fn grid() -> ComplexDirectivityGrid {
        let pattern = |phase: f64| {
            HostMatrix::from_row_major_vec(
                3,
                4,
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(0.7, phase),
                    Complex64::new(1.0, 0.0),
                    Complex64::from_polar(0.4, phase),
                    Complex64::from_polar(0.7, phase),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
        };
        ComplexDirectivityGrid::try_new(
            vec![0.0, 90.0, 180.0, 270.0],
            vec![0.0, 90.0, 180.0],
            vec![100.0, 200.0],
            vec![pattern(0.0), pattern(0.25)],
        )
        .expect("valid directivity grid")
    }

    fn reference() -> DirectivityReference {
        DirectivityReference {
            schema_version: DIRECTIVITY_REFERENCE_SCHEMA_VERSION,
            provenance: "independent-directivity-fixture.v1".into(),
            points: vec![
                DirectivityReferencePoint {
                    frequency_hz: 100.0,
                    vertical_angle_deg: 90.0,
                    horizontal_angle_deg: 90.0,
                    normalized_pressure: Complex64::new(1.0, 0.0),
                },
                DirectivityReferencePoint {
                    frequency_hz: 100.0,
                    vertical_angle_deg: 90.0,
                    horizontal_angle_deg: 0.0,
                    normalized_pressure: Complex64::new(0.7, 0.0),
                },
                DirectivityReferencePoint {
                    frequency_hz: 200.0,
                    vertical_angle_deg: 90.0,
                    horizontal_angle_deg: 90.0,
                    normalized_pressure: Complex64::new(1.0, 0.0),
                },
                DirectivityReferencePoint {
                    frequency_hz: 200.0,
                    vertical_angle_deg: 90.0,
                    horizontal_angle_deg: 0.0,
                    normalized_pressure: Complex64::from_polar(0.7, 0.25),
                },
            ],
        }
    }

    #[test]
    fn directivity_reference_matches_complex_grid_and_round_trips() {
        let reference = reference();
        reference.validate().expect("valid reference");
        let decoded: DirectivityReference =
            serde_json::from_value(serde_json::to_value(&reference).expect("serialize reference"))
                .expect("deserialize reference");
        assert_eq!(decoded, reference);

        let report = correlate_directivity_reference(
            &grid(),
            &decoded,
            DirectivityReferenceTolerances {
                max_rmse_relative_error: 1.0e-12,
                max_relative_error: 1.0e-12,
                max_absolute_error: 1.0e-12,
                absolute_floor: 1.0e-6,
            },
        )
        .expect("reference correlation");
        assert_eq!(report.samples, 4);
        assert!(report.passed);
        assert!(report.rmse_relative_error < 1.0e-12);
    }

    #[test]
    fn directivity_reference_is_phase_sensitive_and_reports_gate_failure() {
        let mut reference = reference();
        reference.points[1].normalized_pressure = Complex64::new(0.0, 1.0);
        let report = correlate_directivity_reference(
            &grid(),
            &reference,
            DirectivityReferenceTolerances {
                max_rmse_relative_error: 1.0e-3,
                max_relative_error: 1.0e-3,
                max_absolute_error: 1.0e-3,
                absolute_floor: 1.0e-6,
            },
        )
        .expect("finite mismatched reference remains reportable");
        assert!(!report.passed);
        assert!(report.maximum_absolute_error > 1.0);
        assert!(report.maximum_relative_error > 1.0);
    }

    #[test]
    fn directivity_reference_rejects_duplicates_missing_anchors_and_extrapolation() {
        let mut duplicate = reference();
        duplicate.points.push(duplicate.points[0]);
        assert!(duplicate.validate().is_err());

        let mut missing_anchor = reference();
        missing_anchor
            .points
            .retain(|point| !(point.frequency_hz == 200.0 && point.horizontal_angle_deg == 90.0));
        assert!(missing_anchor.validate().is_err());

        let mut outside_support = reference();
        outside_support.points[0].frequency_hz = 50.0;
        outside_support.points[1].frequency_hz = 50.0;
        assert!(matches!(
            correlate_directivity_reference(
                &grid(),
                &outside_support,
                DirectivityReferenceTolerances::default(),
            ),
            Err(DirectivityCorrelationError(message))
                if message.contains("outside grid support")
        ));
    }

    #[test]
    fn directivity_reference_rejects_invalid_tolerances_and_points() {
        let mut invalid = reference();
        invalid.points[0].horizontal_angle_deg = 360.0;
        assert!(invalid.validate().is_err());

        let mut invalid_anchor = reference();
        invalid_anchor.points[0].normalized_pressure = Complex64::new(0.9, 0.0);
        assert!(matches!(
            invalid_anchor.validate(),
            Err(DirectivityCorrelationError(message))
                if message.contains("normalized to 1+0i")
        ));

        let mut invalid_tolerance = DirectivityReferenceTolerances::default();
        invalid_tolerance.absolute_floor = 0.0;
        assert!(matches!(
            correlate_directivity_reference(&grid(), &reference(), invalid_tolerance),
            Err(DirectivityCorrelationError(message))
                if message.contains("positive floor")
        ));

        let mut invalid_schema = reference();
        invalid_schema.schema_version += 1;
        assert!(invalid_schema.validate().is_err());
    }
}
