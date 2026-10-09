//! Bounded caller-supplied measured displacement validation.
//!
//! This module compares a predicted displacement curve against an explicitly
//! supplied measured curve. It reports numerical agreement separately from
//! data provenance: passing this contract does not authenticate an instrument,
//! calibration certificate, or published benchmark.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for the measured validation result.
pub const MEASURED_VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const MEASURED_VALIDATION_DIGEST_ALGORITHM: &str = "sha256";

/// One coordinate-aligned predicted/measured displacement observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredDisplacementPoint {
    pub coordinate_m: f64,
    pub predicted_m: f64,
    pub measured_m: f64,
}

/// Comparison controls and caller-provided data provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredValidationStudy {
    pub case_id: String,
    pub points: Vec<MeasuredDisplacementPoint>,
    pub absolute_tolerance_m: f64,
    pub relative_tolerance: f64,
    pub source_id: String,
    /// Optional digest binding the case, controls, and supplied observations.
    /// A digest protects input integrity but does not authenticate its origin.
    #[serde(default)]
    pub input_sha256: Option<String>,
}

impl MeasuredValidationStudy {
    /// Deterministic curve with a small bounded measurement perturbation.
    pub fn introductory() -> Self {
        let coordinates: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        let points = coordinates
            .into_iter()
            .map(|coordinate_m| {
                let predicted_m = 0.002 * coordinate_m.powi(2);
                MeasuredDisplacementPoint {
                    coordinate_m,
                    predicted_m,
                    measured_m: predicted_m + 2.0e-6 * coordinate_m,
                }
            })
            .collect();
        Self {
            case_id: "fixture:static-displacement-curve".to_string(),
            points,
            absolute_tolerance_m: 5.0e-6,
            relative_tolerance: 0.01,
            source_id: "fixture:caller-supplied".to_string(),
            input_sha256: None,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.case_id.trim().is_empty() || self.source_id.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "measured_validation.provenance",
                value: "case_id and source_id must be non-empty".to_string(),
            });
        }
        if let Some(input_sha256) = &self.input_sha256 {
            let expected = measured_validation_input_sha256(self);
            if input_sha256.len() != 64
                || !input_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !input_sha256.eq_ignore_ascii_case(&expected)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_validation.input_sha256",
                    value: "digest does not match the supplied study inputs".to_string(),
                });
            }
        }
        if self.points.len() < 2 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "measured_validation.points",
                value: self.points.len().to_string(),
            });
        }
        if !self.absolute_tolerance_m.is_finite()
            || self.absolute_tolerance_m <= 0.0
            || !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "measured_validation.tolerances",
                value: format!(
                    "({}, {})",
                    self.absolute_tolerance_m, self.relative_tolerance
                ),
            });
        }
        let mut previous_coordinate = -f64::EPSILON;
        for point in &self.points {
            if !point.coordinate_m.is_finite()
                || !point.predicted_m.is_finite()
                || !point.measured_m.is_finite()
                || point.coordinate_m <= previous_coordinate
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_validation.points",
                    value: "finite, strictly increasing coordinates required".to_string(),
                });
            }
            previous_coordinate = point.coordinate_m;
        }
        Ok(())
    }
}

/// Versioned measured displacement comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredValidationResult {
    pub schema_version: u32,
    pub study: MeasuredValidationStudy,
    pub point_count: usize,
    pub rmse_m: f64,
    pub max_abs_error_m: f64,
    pub max_relative_error: f64,
    pub absolute_gate_passed: bool,
    pub relative_gate_passed: bool,
    pub accepted: bool,
    pub evidence_source: MeasuredValidationEvidenceSource,
    pub execution: StructuralExecutionProvenance,
    #[serde(default)]
    pub integrity: MeasuredValidationIntegrity,
}

/// Integrity state for caller-supplied measured-validation inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeasuredValidationIntegrity {
    #[default]
    Unbound,
    Sha256Validated,
}

/// Provenance class for the supplied measured samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasuredValidationEvidenceSource {
    CallerSupplied,
}

/// Compute the canonical SHA-256 binding for a study's controls and samples.
///
/// The digest is an integrity check over the values consumed by the comparison;
/// it is not a signature and cannot establish instrument or dataset provenance.
pub fn measured_validation_input_sha256(study: &MeasuredValidationStudy) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MEASURED_VALIDATION_SCHEMA_VERSION.to_le_bytes());
    update_digest_string(&mut hasher, &study.case_id);
    update_digest_string(&mut hasher, &study.source_id);
    hasher.update(study.absolute_tolerance_m.to_bits().to_le_bytes());
    hasher.update(study.relative_tolerance.to_bits().to_le_bytes());
    hasher.update((study.points.len() as u64).to_le_bytes());
    for point in &study.points {
        hasher.update(point.coordinate_m.to_bits().to_le_bytes());
        hasher.update(point.predicted_m.to_bits().to_le_bytes());
        hasher.update(point.measured_m.to_bits().to_le_bytes());
    }
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn update_digest_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Compare a predicted displacement curve with caller-supplied observations.
pub fn compare_measured_displacements(
    study: &MeasuredValidationStudy,
) -> Result<MeasuredValidationResult, StructuralMechanicsError> {
    study.validate()?;
    let mut squared_error = 0.0;
    let mut max_abs_error_m: f64 = 0.0;
    let mut max_relative_error: f64 = 0.0;
    for point in &study.points {
        let error_m = point.predicted_m - point.measured_m;
        squared_error += error_m * error_m;
        max_abs_error_m = max_abs_error_m.max(error_m.abs());
        max_relative_error = max_relative_error
            .max(error_m.abs() / point.measured_m.abs().max(study.absolute_tolerance_m));
    }
    let rmse_m = (squared_error / study.points.len() as f64).sqrt();
    let absolute_gate_passed = max_abs_error_m <= study.absolute_tolerance_m;
    let relative_gate_passed = max_relative_error <= study.relative_tolerance;
    let accepted = absolute_gate_passed && relative_gate_passed;

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.measured-displacement-validation.v1".to_string();
    execution.limitations.extend([
        "An optional SHA-256 digest binds the supplied study inputs but does not authenticate instrument origin, calibration, or dataset custody.".to_string(),
        "Comparison consumes caller-supplied measurements and does not authenticate instrument calibration, uncertainty budgets, or published-reference provenance.".to_string(),
        "The contract compares coordinate-aligned scalar displacement only; it is not full-field, modal, stress, dynamic, thermal, or product-scale validation.".to_string(),
    ]);

    Ok(MeasuredValidationResult {
        schema_version: MEASURED_VALIDATION_SCHEMA_VERSION,
        study: study.clone(),
        point_count: study.points.len(),
        rmse_m,
        max_abs_error_m,
        max_relative_error,
        absolute_gate_passed,
        relative_gate_passed,
        accepted,
        evidence_source: MeasuredValidationEvidenceSource::CallerSupplied,
        execution,
        integrity: if study.input_sha256.is_some() {
            MeasuredValidationIntegrity::Sha256Validated
        } else {
            MeasuredValidationIntegrity::Unbound
        },
    })
}
