//! Bounded caller-supplied multi-channel structural validation.
//!
//! This module extends the scalar displacement comparison with independently
//! gated aligned channels for field, stress, and modal reference data.  It is
//! deliberately a validation boundary rather than an instrument adapter:
//! callers must supply the samples, and passing the numerical gates does not
//! authenticate their origin, calibration, or custody.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Schema version for the multi-channel measured validation result.
pub const MEASURED_FIELD_VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const MEASURED_FIELD_VALIDATION_DIGEST_ALGORITHM: &str = "sha256";

/// Physical quantity represented by one aligned measured channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasuredFieldQuantity {
    DisplacementComponent { axis: u8 },
    VonMisesStress,
    ModalFrequencyHz,
    ModalShapeComponent { mode: u32, axis: u8 },
}

impl MeasuredFieldQuantity {
    fn validate(self) -> Result<(), StructuralMechanicsError> {
        match self {
            Self::DisplacementComponent { axis } | Self::ModalShapeComponent { axis, .. }
                if axis > 2 =>
            {
                Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.quantity.axis",
                    value: axis.to_string(),
                })
            }
            Self::ModalShapeComponent { mode, .. } if mode == 0 => {
                Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.quantity.mode",
                    value: mode.to_string(),
                })
            }
            _ => Ok(()),
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::DisplacementComponent { .. } => 0,
            Self::VonMisesStress => 1,
            Self::ModalFrequencyHz => 2,
            Self::ModalShapeComponent { .. } => 3,
        }
    }
}

/// One spatially aligned predicted/measured observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFieldPoint {
    pub coordinate_m: [f64; 3],
    pub predicted: f64,
    pub measured: f64,
}

/// Controls and samples for one independently gated quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFieldChannel {
    pub channel_id: String,
    pub quantity: MeasuredFieldQuantity,
    pub points: Vec<MeasuredFieldPoint>,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

/// Caller-supplied multi-channel structural validation input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFieldValidationStudy {
    pub case_id: String,
    pub channels: Vec<MeasuredFieldChannel>,
    pub source_id: String,
    /// Optional digest binding every value consumed by the comparison.
    /// This is an integrity check, not a signature or source authentication.
    #[serde(default)]
    pub input_sha256: Option<String>,
}

/// Per-channel numerical comparison result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFieldChannelResult {
    pub channel_id: String,
    pub quantity: MeasuredFieldQuantity,
    pub point_count: usize,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub max_relative_error: f64,
    pub absolute_gate_passed: bool,
    pub relative_gate_passed: bool,
    pub accepted: bool,
}

/// Versioned multi-channel comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFieldValidationResult {
    pub schema_version: u32,
    pub study: MeasuredFieldValidationStudy,
    pub channels: Vec<MeasuredFieldChannelResult>,
    pub max_abs_error: f64,
    pub max_relative_error: f64,
    pub accepted: bool,
    pub evidence_source: MeasuredFieldEvidenceSource,
    pub execution: StructuralExecutionProvenance,
    #[serde(default)]
    pub integrity: MeasuredFieldValidationIntegrity,
}

/// Provenance class for the supplied channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasuredFieldEvidenceSource {
    CallerSupplied,
}

/// Integrity state for the caller-supplied input packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeasuredFieldValidationIntegrity {
    #[default]
    Unbound,
    Sha256Validated,
}

impl MeasuredFieldValidationStudy {
    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.case_id.trim().is_empty() || self.source_id.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "measured_field.provenance",
                value: "case_id and source_id must be non-empty".to_string(),
            });
        }
        if self.channels.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "measured_field.channels",
                value: "at least one channel is required".to_string(),
            });
        }
        if let Some(input_sha256) = &self.input_sha256 {
            let expected = measured_field_input_sha256(self);
            if input_sha256.len() != 64
                || !input_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !input_sha256.eq_ignore_ascii_case(&expected)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.input_sha256",
                    value: "digest does not match the supplied channels".to_string(),
                });
            }
        }

        let mut channel_ids = HashSet::with_capacity(self.channels.len());
        for channel in &self.channels {
            if channel.channel_id.trim().is_empty() || !channel_ids.insert(&channel.channel_id) {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.channel_id",
                    value: "channel IDs must be non-empty and unique".to_string(),
                });
            }
            channel.quantity.validate()?;
            let minimum_points = match channel.quantity {
                MeasuredFieldQuantity::ModalFrequencyHz => 1,
                _ => 2,
            };
            if channel.points.len() < minimum_points
                || (channel.quantity == MeasuredFieldQuantity::ModalFrequencyHz
                    && channel.points.len() != 1)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.points",
                    value: format!(
                        "channel {} requires {} aligned point(s)",
                        channel.channel_id, minimum_points
                    ),
                });
            }
            if !channel.absolute_tolerance.is_finite()
                || channel.absolute_tolerance <= 0.0
                || !channel.relative_tolerance.is_finite()
                || channel.relative_tolerance <= 0.0
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "measured_field.tolerances",
                    value: channel.channel_id.clone(),
                });
            }
            for (index, point) in channel.points.iter().enumerate() {
                if point.coordinate_m.iter().any(|value| !value.is_finite())
                    || !point.predicted.is_finite()
                    || !point.measured.is_finite()
                {
                    return Err(StructuralMechanicsError::InvalidConfiguration {
                        field: "measured_field.points",
                        value: format!("non-finite sample in {} at {}", channel.channel_id, index),
                    });
                }
                if channel.points[..index]
                    .iter()
                    .any(|previous| previous.coordinate_m == point.coordinate_m)
                {
                    return Err(StructuralMechanicsError::InvalidConfiguration {
                        field: "measured_field.points",
                        value: format!("duplicate coordinate in {}", channel.channel_id),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Compute a canonical SHA-256 binding over the complete input packet.
pub fn measured_field_input_sha256(study: &MeasuredFieldValidationStudy) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MEASURED_FIELD_VALIDATION_SCHEMA_VERSION.to_le_bytes());
    update_digest_string(&mut hasher, &study.case_id);
    update_digest_string(&mut hasher, &study.source_id);
    hasher.update((study.channels.len() as u64).to_le_bytes());
    for channel in &study.channels {
        update_digest_string(&mut hasher, &channel.channel_id);
        hasher.update([channel.quantity.tag()]);
        match channel.quantity {
            MeasuredFieldQuantity::DisplacementComponent { axis }
            | MeasuredFieldQuantity::ModalShapeComponent { axis, .. } => {
                hasher.update([axis]);
                if let MeasuredFieldQuantity::ModalShapeComponent { mode, .. } = channel.quantity {
                    hasher.update(mode.to_le_bytes());
                }
            }
            MeasuredFieldQuantity::VonMisesStress | MeasuredFieldQuantity::ModalFrequencyHz => {}
        }
        hasher.update(channel.absolute_tolerance.to_bits().to_le_bytes());
        hasher.update(channel.relative_tolerance.to_bits().to_le_bytes());
        hasher.update((channel.points.len() as u64).to_le_bytes());
        for point in &channel.points {
            for coordinate in point.coordinate_m {
                hasher.update(coordinate.to_bits().to_le_bytes());
            }
            hasher.update(point.predicted.to_bits().to_le_bytes());
            hasher.update(point.measured.to_bits().to_le_bytes());
        }
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

/// Compare all caller-supplied channels and require every channel gate.
pub fn compare_measured_field_validation(
    study: &MeasuredFieldValidationStudy,
) -> Result<MeasuredFieldValidationResult, StructuralMechanicsError> {
    study.validate()?;
    let mut channel_results = Vec::with_capacity(study.channels.len());
    let mut max_abs_error: f64 = 0.0;
    let mut max_relative_error: f64 = 0.0;

    for channel in &study.channels {
        let mut squared_error = 0.0;
        let mut channel_max_abs_error: f64 = 0.0;
        let mut channel_max_relative_error: f64 = 0.0;
        for point in &channel.points {
            let error = point.predicted - point.measured;
            squared_error += error * error;
            channel_max_abs_error = channel_max_abs_error.max(error.abs());
            channel_max_relative_error = channel_max_relative_error
                .max(error.abs() / point.measured.abs().max(channel.absolute_tolerance));
        }
        let rmse = (squared_error / channel.points.len() as f64).sqrt();
        let absolute_gate_passed = channel_max_abs_error <= channel.absolute_tolerance;
        let relative_gate_passed = channel_max_relative_error <= channel.relative_tolerance;
        let accepted = absolute_gate_passed && relative_gate_passed;
        max_abs_error = max_abs_error.max(channel_max_abs_error);
        max_relative_error = max_relative_error.max(channel_max_relative_error);
        channel_results.push(MeasuredFieldChannelResult {
            channel_id: channel.channel_id.clone(),
            quantity: channel.quantity,
            point_count: channel.points.len(),
            rmse,
            max_abs_error: channel_max_abs_error,
            max_relative_error: channel_max_relative_error,
            absolute_gate_passed,
            relative_gate_passed,
            accepted,
        });
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.measured-field-validation.v1".to_string();
    execution.limitations.extend([
        "An optional SHA-256 digest binds supplied channels but does not authenticate instrument origin, calibration, or custody.".to_string(),
        "Channels are caller-supplied and aligned by coordinate; modal-shape values must already use a compatible sign and normalization.".to_string(),
        "This contract is not production solver coupling, uncertainty quantification, image correlation, or product-scale validation.".to_string(),
    ]);

    let accepted = channel_results.iter().all(|result| result.accepted);
    Ok(MeasuredFieldValidationResult {
        schema_version: MEASURED_FIELD_VALIDATION_SCHEMA_VERSION,
        study: study.clone(),
        channels: channel_results,
        max_abs_error,
        max_relative_error,
        accepted,
        evidence_source: MeasuredFieldEvidenceSource::CallerSupplied,
        execution,
        integrity: if study.input_sha256.is_some() {
            MeasuredFieldValidationIntegrity::Sha256Validated
        } else {
            MeasuredFieldValidationIntegrity::Unbound
        },
    })
}
