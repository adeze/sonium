//! Versioned measured-calibration dataset provenance and integrity envelope.
//!
//! The envelope binds caller-supplied scalar operating-point samples to a
//! dataset identity, source/license metadata, acquisition instrument, and
//! calibration reference. It authenticates content integrity, not the truth
//! of an instrument, certificate, public source, or measured result.

use sha2::{Digest, Sha256};
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::measured_calibration_reference::{
    MeasuredCalibrationError, MeasuredCalibrationOptions, MeasuredCalibrationResult,
    MeasuredCalibrationSample, calibrate_measured_response,
};

/// Schema identifier for the scalar propeller calibration dataset envelope.
pub const MEASURED_CALIBRATION_DATASET_SCHEMA_V1: &str =
    "propeller.measured-calibration-dataset.v1";

/// Provenance and integrity metadata bound to measured calibration samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCalibrationDatasetMetadata {
    /// Stable identifier assigned by the dataset owner.
    pub dataset_id: String,
    /// URI, DOI, or local evidence reference for the source dataset.
    pub source_uri: String,
    /// Declared data license or usage terms.
    pub license: String,
    /// Instrument or acquisition-chain identifier.
    pub instrument_id: String,
    /// Calibration certificate, run, or reference identifier.
    pub calibration_reference: String,
    /// Physical quantity represented by the scalar values, e.g. `thrust`.
    pub channel: String,
    /// Unit shared by predicted and measured values, e.g. `N`.
    pub unit: String,
    /// Lowercase SHA-256 digest over samples and all metadata except itself.
    pub content_sha256: String,
}

/// Measured calibration samples with a checked provenance envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredCalibrationDataset {
    /// Ordered samples used by the calibration fit.
    pub samples: Vec<MeasuredCalibrationSample>,
    /// Identity, source, calibration, and content-integrity metadata.
    pub metadata: MeasuredCalibrationDatasetMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasuredCalibrationDatasetDocument {
    schema_version: String,
    samples: Vec<MeasuredCalibrationSample>,
    dataset_id: String,
    source_uri: String,
    license: String,
    instrument_id: String,
    calibration_reference: String,
    channel: String,
    unit: String,
    content_sha256: String,
}

/// Errors emitted by the dataset provenance/integrity boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasuredCalibrationDatasetError {
    /// Metadata or dataset structure is malformed.
    InvalidMetadata {
        /// Invalid metadata field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Dataset has too few samples for affine calibration.
    InsufficientSamples { count: usize, minimum: usize },
    /// A sample has a non-finite field.
    InvalidSample {
        /// Sample index in file/order.
        index: usize,
        /// Invalid field.
        field: &'static str,
    },
    /// Digest has invalid shape or characters.
    InvalidDigest,
    /// Supplied digest does not match canonical content.
    DigestMismatch { expected: String },
    /// JSON could not be decoded or had the wrong schema.
    InvalidDocument(String),
    /// Nested calibration returned a typed failure.
    Calibration(MeasuredCalibrationError),
}

impl std::fmt::Display for MeasuredCalibrationDatasetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata { field, requirement } => {
                write!(
                    formatter,
                    "invalid measured dataset metadata {field}: {requirement}"
                )
            }
            Self::InsufficientSamples { count, minimum } => write!(
                formatter,
                "measured calibration dataset has {count} samples, needs at least {minimum}"
            ),
            Self::InvalidSample { index, field } => {
                write!(
                    formatter,
                    "measured calibration dataset sample {index} has invalid {field}"
                )
            }
            Self::InvalidDigest => formatter.write_str(
                "measured calibration dataset digest must be 64 lowercase hexadecimal characters",
            ),
            Self::DigestMismatch { expected } => {
                write!(
                    formatter,
                    "measured calibration dataset digest mismatch: expected {expected}"
                )
            }
            Self::InvalidDocument(error) => {
                write!(formatter, "invalid measured calibration dataset: {error}")
            }
            Self::Calibration(error) => write!(formatter, "measured calibration failed: {error}"),
        }
    }
}

impl std::error::Error for MeasuredCalibrationDatasetError {}

impl From<MeasuredCalibrationError> for MeasuredCalibrationDatasetError {
    fn from(error: MeasuredCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

impl MeasuredCalibrationDataset {
    /// Build a dataset and compute its canonical content digest.
    pub fn from_parts(
        samples: Vec<MeasuredCalibrationSample>,
        mut metadata: MeasuredCalibrationDatasetMetadata,
    ) -> Result<Self, MeasuredCalibrationDatasetError> {
        let dataset = Self {
            samples,
            metadata: {
                metadata.content_sha256.clear();
                metadata
            },
        };
        dataset.validate_without_digest()?;
        let mut dataset = dataset;
        dataset.metadata.content_sha256 =
            canonical_content_sha256(&dataset.samples, &dataset.metadata);
        dataset.validate()?;
        Ok(dataset)
    }

    /// Validate sample shape, metadata, and the supplied content digest.
    pub fn validate(&self) -> Result<(), MeasuredCalibrationDatasetError> {
        self.validate_without_digest()?;
        if !is_canonical_sha256(&self.metadata.content_sha256) {
            return Err(MeasuredCalibrationDatasetError::InvalidDigest);
        }
        let expected = canonical_content_sha256(&self.samples, &self.metadata);
        if self.metadata.content_sha256 != expected {
            return Err(MeasuredCalibrationDatasetError::DigestMismatch { expected });
        }
        Ok(())
    }

    /// Fit the checked samples through the existing bounded calibration API.
    pub fn calibrate(
        &self,
        options: MeasuredCalibrationOptions,
    ) -> Result<MeasuredCalibrationResult, MeasuredCalibrationDatasetError> {
        self.validate()?;
        calibrate_measured_response(&self.samples, options).map_err(Into::into)
    }

    /// Serialize the validated dataset envelope.
    pub fn to_json(&self) -> Result<String, MeasuredCalibrationDatasetError> {
        self.validate()?;
        let document = MeasuredCalibrationDatasetDocument {
            schema_version: MEASURED_CALIBRATION_DATASET_SCHEMA_V1.into(),
            samples: self.samples.clone(),
            dataset_id: self.metadata.dataset_id.clone(),
            source_uri: self.metadata.source_uri.clone(),
            license: self.metadata.license.clone(),
            instrument_id: self.metadata.instrument_id.clone(),
            calibration_reference: self.metadata.calibration_reference.clone(),
            channel: self.metadata.channel.clone(),
            unit: self.metadata.unit.clone(),
            content_sha256: self.metadata.content_sha256.clone(),
        };
        serde_json::to_string_pretty(&document)
            .map_err(|error| MeasuredCalibrationDatasetError::InvalidDocument(error.to_string()))
    }

    /// Parse and validate a versioned dataset envelope without recomputing or
    /// overwriting its supplied digest.
    pub fn from_json(document: &str) -> Result<Self, MeasuredCalibrationDatasetError> {
        let document: MeasuredCalibrationDatasetDocument = serde_json::from_str(document)
            .map_err(|error| MeasuredCalibrationDatasetError::InvalidDocument(error.to_string()))?;
        if document.schema_version != MEASURED_CALIBRATION_DATASET_SCHEMA_V1 {
            return Err(MeasuredCalibrationDatasetError::InvalidDocument(format!(
                "unsupported schema {:?}; expected {:?}",
                document.schema_version, MEASURED_CALIBRATION_DATASET_SCHEMA_V1
            )));
        }
        let dataset = Self {
            samples: document.samples,
            metadata: MeasuredCalibrationDatasetMetadata {
                dataset_id: document.dataset_id,
                source_uri: document.source_uri,
                license: document.license,
                instrument_id: document.instrument_id,
                calibration_reference: document.calibration_reference,
                channel: document.channel,
                unit: document.unit,
                content_sha256: document.content_sha256,
            },
        };
        dataset.validate()?;
        Ok(dataset)
    }

    fn validate_without_digest(&self) -> Result<(), MeasuredCalibrationDatasetError> {
        if self.samples.len() < 2 {
            return Err(MeasuredCalibrationDatasetError::InsufficientSamples {
                count: self.samples.len(),
                minimum: 2,
            });
        }
        for (index, sample) in self.samples.iter().enumerate() {
            if !sample.operating_point.is_finite() {
                return Err(MeasuredCalibrationDatasetError::InvalidSample {
                    index,
                    field: "operating_point",
                });
            }
            if !sample.predicted_value.is_finite() {
                return Err(MeasuredCalibrationDatasetError::InvalidSample {
                    index,
                    field: "predicted_value",
                });
            }
            if !sample.measured_value.is_finite() {
                return Err(MeasuredCalibrationDatasetError::InvalidSample {
                    index,
                    field: "measured_value",
                });
            }
        }
        for (field, value) in [
            ("dataset_id", self.metadata.dataset_id.as_str()),
            ("source_uri", self.metadata.source_uri.as_str()),
            ("license", self.metadata.license.as_str()),
            ("instrument_id", self.metadata.instrument_id.as_str()),
            (
                "calibration_reference",
                self.metadata.calibration_reference.as_str(),
            ),
            ("channel", self.metadata.channel.as_str()),
            ("unit", self.metadata.unit.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MeasuredCalibrationDatasetError::InvalidMetadata {
                    field,
                    requirement: "non-empty",
                });
            }
        }
        if !(self.metadata.source_uri.starts_with("https://")
            || self.metadata.source_uri.starts_with("http://")
            || self.metadata.source_uri.starts_with("doi:")
            || self.metadata.source_uri.starts_with("file:"))
        {
            return Err(MeasuredCalibrationDatasetError::InvalidMetadata {
                field: "source_uri",
                requirement: "an http(s), doi, or file URI",
            });
        }
        let normalized_ids = [
            self.metadata.dataset_id.to_ascii_lowercase(),
            self.metadata.instrument_id.to_ascii_lowercase(),
            self.metadata.calibration_reference.to_ascii_lowercase(),
        ];
        if normalized_ids.iter().any(|value| value.contains('\n')) {
            return Err(MeasuredCalibrationDatasetError::InvalidMetadata {
                field: "dataset_id/instrument_id/calibration_reference",
                requirement: "must not contain newlines",
            });
        }
        if self
            .samples
            .iter()
            .map(|sample| sample.operating_point.to_bits())
            .collect::<HashSet<_>>()
            .len()
            != self.samples.len()
        {
            return Err(MeasuredCalibrationDatasetError::InvalidMetadata {
                field: "operating_point",
                requirement: "unique for each sample",
            });
        }
        Ok(())
    }
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn update_len(hasher: &mut Sha256, length: usize) {
    hasher.update((length as u64).to_le_bytes());
}

fn update_str(hasher: &mut Sha256, value: &str) {
    update_len(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

fn canonical_content_sha256(
    samples: &[MeasuredCalibrationSample],
    metadata: &MeasuredCalibrationDatasetMetadata,
) -> String {
    let mut hasher = Sha256::new();
    update_str(&mut hasher, MEASURED_CALIBRATION_DATASET_SCHEMA_V1);
    update_len(&mut hasher, samples.len());
    for sample in samples {
        update_f64(&mut hasher, sample.operating_point);
        update_f64(&mut hasher, sample.predicted_value);
        update_f64(&mut hasher, sample.measured_value);
    }
    update_str(&mut hasher, &metadata.dataset_id);
    update_str(&mut hasher, &metadata.source_uri);
    update_str(&mut hasher, &metadata.license);
    update_str(&mut hasher, &metadata.instrument_id);
    update_str(&mut hasher, &metadata.calibration_reference);
    update_str(&mut hasher, &metadata.channel);
    update_str(&mut hasher, &metadata.unit);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> MeasuredCalibrationDatasetMetadata {
        MeasuredCalibrationDatasetMetadata {
            dataset_id: "public-fixture-v1".into(),
            source_uri: "https://example.invalid/propeller/public-fixture-v1".into(),
            license: "CC-BY-4.0".into(),
            instrument_id: "rig-fixture-01".into(),
            calibration_reference: "certificate-fixture-v1".into(),
            channel: "thrust".into(),
            unit: "N".into(),
            content_sha256: String::new(),
        }
    }

    fn samples() -> Vec<MeasuredCalibrationSample> {
        vec![
            MeasuredCalibrationSample {
                operating_point: 0.2,
                predicted_value: 1.0,
                measured_value: 1.2,
            },
            MeasuredCalibrationSample {
                operating_point: 0.4,
                predicted_value: 2.0,
                measured_value: 2.4,
            },
        ]
    }

    #[test]
    fn from_parts_binds_digest_and_supports_calibration() {
        let dataset = MeasuredCalibrationDataset::from_parts(samples(), metadata())
            .expect("metadata and samples should be admitted");
        assert!(is_canonical_sha256(&dataset.metadata.content_sha256));
        let result = dataset
            .calibrate(MeasuredCalibrationOptions::default())
            .expect("bound samples should calibrate");
        assert!(result.is_finite());
        assert!(result.passed);
    }

    #[test]
    fn json_round_trip_preserves_bound_content() {
        let dataset =
            MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset");
        let restored = MeasuredCalibrationDataset::from_json(&dataset.to_json().expect("json"))
            .expect("round trip");
        assert_eq!(restored, dataset);
    }

    #[test]
    fn tampered_sample_metadata_or_digest_fails_closed() {
        let dataset =
            MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset");
        let json = dataset.to_json().expect("json");
        let tampered_sample = json.replace("\"measured_value\": 1.2", "\"measured_value\": 1.3");
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&tampered_sample),
            Err(MeasuredCalibrationDatasetError::DigestMismatch { .. })
        ));
        let tampered_metadata = json.replace("public-fixture-v1", "other-fixture-v1");
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&tampered_metadata),
            Err(MeasuredCalibrationDatasetError::DigestMismatch { .. })
        ));
        let tampered_digest = json.replace(&dataset.metadata.content_sha256, &"0".repeat(64));
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&tampered_digest),
            Err(MeasuredCalibrationDatasetError::DigestMismatch { .. })
                | Err(MeasuredCalibrationDatasetError::InvalidDigest)
        ));
    }

    #[test]
    fn malformed_metadata_and_samples_fail_closed() {
        let mut invalid_metadata = metadata();
        invalid_metadata.source_uri = "not-a-uri".into();
        assert!(matches!(
            MeasuredCalibrationDataset::from_parts(samples(), invalid_metadata),
            Err(MeasuredCalibrationDatasetError::InvalidMetadata {
                field: "source_uri",
                ..
            })
        ));
        assert!(matches!(
            MeasuredCalibrationDataset::from_parts(vec![], metadata()),
            Err(MeasuredCalibrationDatasetError::InsufficientSamples { .. })
        ));
        let mut invalid_sample = samples();
        invalid_sample[0].operating_point = f64::NAN;
        assert!(matches!(
            MeasuredCalibrationDataset::from_parts(invalid_sample, metadata()),
            Err(MeasuredCalibrationDatasetError::InvalidSample {
                field: "operating_point",
                ..
            })
        ));
    }

    #[test]
    fn unknown_schema_and_fields_are_rejected() {
        let dataset =
            MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset");
        let json = dataset.to_json().expect("json");
        let schema = json.replace(
            MEASURED_CALIBRATION_DATASET_SCHEMA_V1,
            "propeller.measured-calibration-dataset.v0",
        );
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&schema),
            Err(MeasuredCalibrationDatasetError::InvalidDocument(_))
        ));
        let unknown = json.replace("\"samples\":", "\"unknown\": 1, \"samples\":");
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&unknown),
            Err(MeasuredCalibrationDatasetError::InvalidDocument(_))
        ));
    }
}
