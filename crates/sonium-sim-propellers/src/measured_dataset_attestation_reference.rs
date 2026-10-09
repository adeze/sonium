//! Authenticated measured-dataset attestation envelope.
//!
//! [`crate::measured_dataset_reference`] binds dataset content with SHA-256.
//! This module adds caller-supplied Ed25519 signatures over a canonical,
//! domain-separated dataset payload.  It verifies an attestation; it does not
//! discover public keys, fetch instruments or certificates, or establish that
//! a measurement is physically truthful.

use ed25519_zebra::{Signature, VerificationKey, VerificationKeyBytes};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::measured_dataset_reference::{
    MeasuredCalibrationDataset, MeasuredCalibrationDatasetError,
};

/// Versioned identity for signed measured-dataset attestations.
pub const MEASURED_DATASET_ATTESTATION_SCHEMA_V1: &str =
    "propeller.measured-dataset-attestation.v1";

const SIGNING_DOMAIN: &[u8] = b"sonium-propeller-measured-dataset-attestation-v1\0";

/// A canonical measured dataset and its detached Ed25519 signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetAttestation {
    /// Attestation schema.
    pub schema_version: String,
    /// Canonical JSON produced by `MeasuredCalibrationDataset::to_json`.
    pub dataset_json: String,
    /// Signature algorithm identifier.
    pub algorithm: String,
    /// Stable key identifier supplied by the dataset trust policy.
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key.
    pub public_key: String,
    /// Lowercase hexadecimal detached Ed25519 signature.
    pub signature: String,
}

/// A host-owned Ed25519 trust anchor for measured-dataset attestations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetTrustAnchor {
    /// Stable key identifier expected in an attestation.
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key.
    pub public_key: String,
}

/// An explicit allow-list for authenticated measured-dataset attestations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetTrustPolicy {
    /// Accepted key identifiers and their pinned public keys.
    pub trusted_keys: Vec<MeasuredDatasetTrustAnchor>,
}

impl MeasuredDatasetTrustPolicy {
    /// Construct and validate a host-owned trust policy.
    pub fn new(
        trusted_keys: Vec<MeasuredDatasetTrustAnchor>,
    ) -> Result<Self, MeasuredDatasetAttestationError> {
        let policy = Self { trusted_keys };
        policy.validate()?;
        Ok(policy)
    }

    /// Validate key identity uniqueness and public-key encoding.
    pub fn validate(&self) -> Result<(), MeasuredDatasetAttestationError> {
        if self.trusted_keys.is_empty() {
            return Err(MeasuredDatasetAttestationError::InvalidField {
                field: "trusted_keys",
                requirement: "at least one trust anchor",
            });
        }

        let mut key_ids = HashSet::with_capacity(self.trusted_keys.len());
        for anchor in &self.trusted_keys {
            validate_key_id(&anchor.key_id)?;
            if !key_ids.insert(&anchor.key_id) {
                return Err(MeasuredDatasetAttestationError::InvalidField {
                    field: "trusted_keys",
                    requirement: "unique key IDs",
                });
            }
            decode_hex::<32>(&anchor.public_key, "trusted_keys.public_key")?;
        }
        Ok(())
    }

    /// Verify an attestation against the pinned key identified by its key ID.
    pub fn verify(
        &self,
        attestation: &MeasuredDatasetAttestation,
    ) -> Result<MeasuredCalibrationDataset, MeasuredDatasetAttestationError> {
        self.validate()?;
        let anchor = self
            .trusted_keys
            .iter()
            .find(|anchor| anchor.key_id == attestation.key_id)
            .ok_or(MeasuredDatasetAttestationError::InvalidField {
                field: "key_id",
                requirement: "present in the trusted key policy",
            })?;
        if anchor.public_key != attestation.public_key {
            return Err(MeasuredDatasetAttestationError::InvalidField {
                field: "public_key",
                requirement: "matches the pinned trusted public key",
            });
        }
        attestation.verify()
    }
}

/// Failure-closed errors for signed measured-dataset attestations.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasuredDatasetAttestationError {
    /// The nested dataset failed validation or decoding.
    Dataset(MeasuredCalibrationDatasetError),
    /// The attestation JSON was malformed.
    InvalidDocument(String),
    /// A scalar envelope field is invalid.
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A hexadecimal key or signature has invalid length or characters.
    InvalidHex {
        /// Field name.
        field: &'static str,
    },
    /// Dataset JSON differs from the canonical validated representation.
    NonCanonicalDataset,
    /// Signature verification failed.
    SignatureMismatch,
}

impl fmt::Display for MeasuredDatasetAttestationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(error) => write!(formatter, "measured dataset rejected: {error}"),
            Self::InvalidDocument(error) => {
                write!(formatter, "invalid measured attestation document: {error}")
            }
            Self::InvalidField { field, requirement } => {
                write!(
                    formatter,
                    "invalid measured attestation {field}: {requirement}"
                )
            }
            Self::InvalidHex { field } => {
                write!(
                    formatter,
                    "measured attestation {field} must be lowercase hexadecimal"
                )
            }
            Self::NonCanonicalDataset => {
                formatter.write_str("measured attestation dataset JSON is not canonical")
            }
            Self::SignatureMismatch => {
                formatter.write_str("measured attestation signature rejected")
            }
        }
    }
}

impl std::error::Error for MeasuredDatasetAttestationError {}

impl From<MeasuredCalibrationDatasetError> for MeasuredDatasetAttestationError {
    fn from(error: MeasuredCalibrationDatasetError) -> Self {
        Self::Dataset(error)
    }
}

impl MeasuredDatasetAttestation {
    /// Construct and immediately verify an attestation from binary signature
    /// material supplied by a caller-owned trust policy.
    pub fn from_signed_parts(
        dataset: &MeasuredCalibrationDataset,
        key_id: impl Into<String>,
        public_key: [u8; 32],
        signature: [u8; 64],
    ) -> Result<Self, MeasuredDatasetAttestationError> {
        let dataset_json = dataset.to_json()?;
        let attestation = Self {
            schema_version: MEASURED_DATASET_ATTESTATION_SCHEMA_V1.to_owned(),
            dataset_json,
            algorithm: "ed25519".to_owned(),
            key_id: key_id.into(),
            public_key: encode_hex(&public_key),
            signature: encode_hex(&signature),
        };
        attestation.verify()?;
        Ok(attestation)
    }

    /// Decode, validate, and verify a signed attestation document.
    pub fn from_json(json: &str) -> Result<Self, MeasuredDatasetAttestationError> {
        let attestation = serde_json::from_str::<Self>(json)
            .map_err(|error| MeasuredDatasetAttestationError::InvalidDocument(error.to_string()))?;
        attestation.verify()?;
        Ok(attestation)
    }

    /// Serialize a verified attestation document.
    pub fn to_json(&self) -> Result<String, MeasuredDatasetAttestationError> {
        self.verify()?;
        serde_json::to_string(self)
            .map_err(|error| MeasuredDatasetAttestationError::InvalidDocument(error.to_string()))
    }

    /// Verify the nested dataset and detached signature, returning the
    /// validated dataset for downstream calibration/correlation.
    pub fn verify(&self) -> Result<MeasuredCalibrationDataset, MeasuredDatasetAttestationError> {
        if self.schema_version != MEASURED_DATASET_ATTESTATION_SCHEMA_V1 {
            return Err(MeasuredDatasetAttestationError::InvalidField {
                field: "schema_version",
                requirement: "supported v1 schema",
            });
        }
        if self.algorithm != "ed25519" {
            return Err(MeasuredDatasetAttestationError::InvalidField {
                field: "algorithm",
                requirement: "ed25519",
            });
        }
        if self.key_id.trim().is_empty() || self.key_id.chars().any(char::is_control) {
            return Err(MeasuredDatasetAttestationError::InvalidField {
                field: "key_id",
                requirement: "non-empty and without control characters",
            });
        }
        let public_key = decode_hex::<32>(&self.public_key, "public_key")?;
        let signature = decode_hex::<64>(&self.signature, "signature")?;
        let dataset = MeasuredCalibrationDataset::from_json(&self.dataset_json)?;
        let canonical_dataset_json = dataset.to_json()?;
        if self.dataset_json != canonical_dataset_json {
            return Err(MeasuredDatasetAttestationError::NonCanonicalDataset);
        }
        let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
            .map_err(|_| MeasuredDatasetAttestationError::InvalidHex {
                field: "public_key",
            })?;
        let detached = Signature::from_bytes(&signature);
        verification_key
            .verify(
                &detached,
                &signing_payload_bytes(&self.dataset_json, &self.key_id),
            )
            .map_err(|_| MeasuredDatasetAttestationError::SignatureMismatch)?;
        Ok(dataset)
    }

    /// Verify the attestation against a caller-owned, pinned trust policy.
    pub fn verify_with_policy(
        &self,
        policy: &MeasuredDatasetTrustPolicy,
    ) -> Result<MeasuredCalibrationDataset, MeasuredDatasetAttestationError> {
        policy.verify(self)
    }

    /// Return the exact bytes that must be signed for this dataset/key ID.
    pub fn signing_payload(&self) -> Result<Vec<u8>, MeasuredDatasetAttestationError> {
        let dataset = MeasuredCalibrationDataset::from_json(&self.dataset_json)?;
        let canonical = dataset.to_json()?;
        if canonical != self.dataset_json {
            return Err(MeasuredDatasetAttestationError::NonCanonicalDataset);
        }
        validate_key_id(&self.key_id)?;
        Ok(signing_payload_bytes(&self.dataset_json, &self.key_id))
    }
}

/// Return the domain-separated canonical payload a trust authority signs.
pub fn measured_dataset_attestation_payload(
    dataset: &MeasuredCalibrationDataset,
    key_id: &str,
) -> Result<Vec<u8>, MeasuredDatasetAttestationError> {
    validate_key_id(key_id)?;
    let dataset_json = dataset.to_json()?;
    Ok(signing_payload_bytes(&dataset_json, key_id))
}

fn validate_key_id(key_id: &str) -> Result<(), MeasuredDatasetAttestationError> {
    if key_id.trim().is_empty() || key_id.chars().any(char::is_control) {
        return Err(MeasuredDatasetAttestationError::InvalidField {
            field: "key_id",
            requirement: "non-empty and without control characters",
        });
    }
    Ok(())
}

fn signing_payload_bytes(dataset_json: &str, key_id: &str) -> Vec<u8> {
    let mut payload =
        Vec::with_capacity(SIGNING_DOMAIN.len() + key_id.len() + 1 + dataset_json.len());
    payload.extend_from_slice(SIGNING_DOMAIN);
    payload.extend_from_slice(key_id.as_bytes());
    payload.push(0);
    payload.extend_from_slice(dataset_json.as_bytes());
    payload
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], MeasuredDatasetAttestationError> {
    if value.len() != 2 * N
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err(MeasuredDatasetAttestationError::InvalidHex { field });
    }
    let mut bytes = [0_u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let high = hex_nibble(value.as_bytes()[2 * index]);
        let low = hex_nibble(value.as_bytes()[2 * index + 1]);
        *byte = (high << 4) | low;
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("hex validated before conversion"),
    }
}
