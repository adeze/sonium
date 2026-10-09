//! Uncertainty-bound measured-room acceptance for solved RoomSim responses.
//!
//! [`crate::measured_io`] authenticates the shape and content identity of an
//! imported response dataset, while [`crate::measured_validation`] computes
//! unweighted complex-response errors.  This module adds the next, deliberately
//! separate contract: caller-supplied per-sample magnitude and phase
//! uncertainties are bound to the admitted dataset digest and can be used in
//! an explicit acceptance budget.
//!
//! The uncertainty values are evidence supplied by the measurement workflow;
//! the certificate path verifies cryptographic binding and configured issuer
//! trust, but cannot establish physical calibration, chain of custody, or the
//! statistical method used to produce the uncertainty values.

use crate::SimulationOutput;
use crate::measured_io::{MeasuredResponseDataset, compare_solved_output_to_measured_dataset};
use crate::measured_validation::MeasuredComparisonReport;
use ed25519_zebra::{Signature, VerificationKey, VerificationKeyBytes};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::path::Path;

/// Versioned JSON envelope for uncertainty data bound to a measured dataset.
pub const MEASURED_UNCERTAINTY_SCHEMA_V1: &str = "roomsim.measured-response-uncertainty.v1";

/// Versioned signed certificate binding an instrument/calibration claim to a
/// measured response dataset digest.
pub const MEASURED_INSTRUMENT_CERTIFICATE_SCHEMA_V1: &str =
    "roomsim.measured-instrument-certificate.v1";
const MEASURED_INSTRUMENT_CERTIFICATE_DOMAIN: &str =
    "sonium-sim-room-measured-instrument-certificate-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredInstrumentCertificate {
    pub schema: String,
    pub certificate_id: String,
    pub instrument_id: String,
    pub calibration_reference: String,
    pub dataset_content_sha256: String,
    pub issuer_key_id: String,
    pub issuer_public_key: String,
    pub issued_at_unix_s: i64,
    pub expires_at_unix_s: i64,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredInstrumentTrustAnchor {
    pub key_id: String,
    pub public_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredInstrumentTrustPolicy {
    pub trusted_keys: Vec<MeasuredInstrumentTrustAnchor>,
}

impl MeasuredInstrumentTrustPolicy {
    pub fn new(trusted_keys: Vec<MeasuredInstrumentTrustAnchor>) -> Result<Self, String> {
        if trusted_keys.is_empty() {
            return Err("instrument trust policy must contain a trusted key".into());
        }
        let mut key_ids = HashSet::with_capacity(trusted_keys.len());
        for anchor in &trusted_keys {
            if anchor.key_id.trim().is_empty() || !key_ids.insert(anchor.key_id.clone()) {
                return Err("instrument trust policy has an empty or duplicate key ID".into());
            }
            decode_certificate_hex::<32>(&anchor.public_key, "trusted instrument public key")?;
        }
        Ok(Self { trusted_keys })
    }

    /// Serialize a validated host-owned issuer allowlist.
    pub fn to_json(&self) -> Result<String, String> {
        Self::new(self.trusted_keys.clone())?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("failed to serialize instrument trust policy: {error}"))
    }

    /// Parse and validate a host-owned issuer allowlist.
    pub fn from_json(document: &str) -> Result<Self, String> {
        let policy: Self = serde_json::from_str(document)
            .map_err(|error| format!("invalid instrument trust policy JSON: {error}"))?;
        Self::new(policy.trusted_keys)
    }
}

impl MeasuredInstrumentCertificate {
    fn validate_fields(&self) -> Result<(), String> {
        if self.schema != MEASURED_INSTRUMENT_CERTIFICATE_SCHEMA_V1 {
            return Err(format!(
                "unsupported instrument certificate schema {:?}",
                self.schema
            ));
        }
        for (field, value) in [
            ("certificate_id", &self.certificate_id),
            ("instrument_id", &self.instrument_id),
            ("calibration_reference", &self.calibration_reference),
            ("issuer_key_id", &self.issuer_key_id),
        ] {
            if value.trim().is_empty() {
                return Err(format!("instrument certificate {field} must not be empty"));
            }
        }
        if !is_canonical_sha256(&self.dataset_content_sha256) {
            return Err("instrument certificate dataset digest must be SHA-256".into());
        }
        if self.issued_at_unix_s < 0 || self.expires_at_unix_s <= self.issued_at_unix_s {
            return Err("instrument certificate validity interval is invalid".into());
        }
        decode_certificate_hex::<32>(&self.issuer_public_key, "instrument issuer public key")?;
        decode_certificate_hex::<64>(&self.signature, "instrument certificate signature")?;
        Ok(())
    }

    /// Construct an unsigned certificate envelope for a signing workflow.
    pub fn unsigned(
        certificate_id: impl Into<String>,
        dataset: &MeasuredResponseDataset,
        issuer_key_id: impl Into<String>,
        issuer_public_key: impl Into<String>,
        issued_at_unix_s: i64,
        expires_at_unix_s: i64,
    ) -> Result<Self, String> {
        dataset.validate()?;
        let certificate = Self {
            schema: MEASURED_INSTRUMENT_CERTIFICATE_SCHEMA_V1.into(),
            certificate_id: certificate_id.into(),
            instrument_id: dataset.metadata.instrument_id.clone(),
            calibration_reference: dataset.metadata.calibration_reference.clone(),
            dataset_content_sha256: dataset.metadata.content_sha256.clone(),
            issuer_key_id: issuer_key_id.into(),
            issuer_public_key: issuer_public_key.into(),
            issued_at_unix_s,
            expires_at_unix_s,
            signature: String::new(),
        };
        certificate.validate_unsigned_fields()?;
        Ok(certificate)
    }

    fn validate_unsigned_fields(&self) -> Result<(), String> {
        if self.signature.is_empty() {
            let mut unsigned = self.clone();
            unsigned.signature = "00".repeat(64);
            unsigned.validate_fields()
        } else {
            self.validate_fields()
        }
    }

    pub fn signature_payload(&self) -> Result<Vec<u8>, String> {
        self.validate_unsigned_fields()?;
        let mut hasher = Sha256::new();
        update_certificate_string(&mut hasher, MEASURED_INSTRUMENT_CERTIFICATE_DOMAIN);
        update_certificate_string(&mut hasher, &self.schema);
        update_certificate_string(&mut hasher, &self.certificate_id);
        update_certificate_string(&mut hasher, &self.instrument_id);
        update_certificate_string(&mut hasher, &self.calibration_reference);
        update_certificate_string(&mut hasher, &self.dataset_content_sha256);
        update_certificate_string(&mut hasher, &self.issuer_key_id);
        update_certificate_string(&mut hasher, &self.issuer_public_key);
        hasher.update(self.issued_at_unix_s.to_le_bytes());
        hasher.update(self.expires_at_unix_s.to_le_bytes());
        Ok(hasher.finalize().to_vec())
    }

    /// Verify certificate identity, dataset binding, trusted issuer, signature,
    /// and validity at a caller-supplied clock value.
    pub fn validate_against(
        &self,
        dataset: &MeasuredResponseDataset,
        now_unix_s: i64,
        policy: &MeasuredInstrumentTrustPolicy,
    ) -> Result<(), String> {
        dataset.validate()?;
        self.validate_fields()?;
        if self.instrument_id != dataset.metadata.instrument_id {
            return Err("instrument certificate instrument ID does not match dataset".into());
        }
        if self.calibration_reference != dataset.metadata.calibration_reference {
            return Err(
                "instrument certificate calibration reference does not match dataset".into(),
            );
        }
        if self.dataset_content_sha256 != dataset.metadata.content_sha256 {
            return Err("instrument certificate dataset digest does not match dataset".into());
        }
        if now_unix_s < self.issued_at_unix_s || now_unix_s > self.expires_at_unix_s {
            return Err("instrument certificate is outside its validity interval".into());
        }
        let anchor = policy
            .trusted_keys
            .iter()
            .find(|anchor| anchor.key_id == self.issuer_key_id)
            .ok_or_else(|| {
                format!(
                    "instrument issuer key ID {:?} is not trusted",
                    self.issuer_key_id
                )
            })?;
        if anchor.public_key != self.issuer_public_key {
            return Err("instrument certificate issuer public key is not trusted".into());
        }
        let public_key =
            decode_certificate_hex::<32>(&self.issuer_public_key, "instrument issuer public key")?;
        let signature =
            decode_certificate_hex::<64>(&self.signature, "instrument certificate signature")?;
        let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
            .map_err(|_| "invalid instrument issuer public key".to_string())?;
        verification_key
            .verify(
                &Signature::from_bytes(&signature),
                &self.signature_payload()?,
            )
            .map_err(|_| "instrument certificate signature verification failed".to_string())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate_fields()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("failed to serialize instrument certificate: {error}"))
    }

    pub fn from_json(document: &str) -> Result<Self, String> {
        let certificate: Self = serde_json::from_str(document)
            .map_err(|error| format!("invalid instrument certificate JSON: {error}"))?;
        certificate.validate_fields()?;
        Ok(certificate)
    }

    /// Load and validate an instrument certificate from disk.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let document = std::fs::read_to_string(path).map_err(|error| {
            format!("failed to read instrument certificate {:?}: {error}", path)
        })?;
        Self::from_json(&document)
    }
}

/// Measured response admitted only after instrument certificate verification.
#[derive(Debug, PartialEq)]
pub struct AuthenticatedMeasuredRoomEvidence {
    dataset: MeasuredResponseDataset,
    certificate: MeasuredInstrumentCertificate,
}

impl AuthenticatedMeasuredRoomEvidence {
    pub fn from_parts(
        dataset: MeasuredResponseDataset,
        certificate: MeasuredInstrumentCertificate,
        now_unix_s: i64,
        policy: &MeasuredInstrumentTrustPolicy,
    ) -> Result<Self, String> {
        certificate.validate_against(&dataset, now_unix_s, policy)?;
        Ok(Self {
            dataset,
            certificate,
        })
    }

    /// Parse and admit a certificate-backed measured dataset.
    pub fn from_json(
        dataset: MeasuredResponseDataset,
        certificate_document: &str,
        now_unix_s: i64,
        policy: &MeasuredInstrumentTrustPolicy,
    ) -> Result<Self, String> {
        let certificate = MeasuredInstrumentCertificate::from_json(certificate_document)?;
        Self::from_parts(dataset, certificate, now_unix_s, policy)
    }

    pub fn dataset(&self) -> &MeasuredResponseDataset {
        &self.dataset
    }

    pub fn certificate(&self) -> &MeasuredInstrumentCertificate {
        &self.certificate
    }

    /// Compare a solved output against this certificate-admitted dataset.
    pub fn compare_solved_output(
        &self,
        output: &SimulationOutput,
        uncertainty: &MeasuredUncertaintyEnvelope,
        listener_names: &[String],
        phase_floor_db: f64,
    ) -> Result<MeasuredAcceptanceReport, String> {
        compare_solved_output_to_measured_with_uncertainty(
            output,
            &self.dataset,
            uncertainty,
            listener_names,
            phase_floor_db,
        )
    }

    /// Admit a solved output only when its uncertainty-normalized errors fit
    /// the caller-supplied acceptance budget.
    pub fn accept_solved_output(
        &self,
        output: &SimulationOutput,
        uncertainty: &MeasuredUncertaintyEnvelope,
        listener_names: &[String],
        phase_floor_db: f64,
        budget: MeasuredAcceptanceBudget,
    ) -> Result<MeasuredAcceptanceReport, String> {
        let report =
            self.compare_solved_output(output, uncertainty, listener_names, phase_floor_db)?;
        report.require_within(budget)?;
        Ok(report)
    }
}

fn update_certificate_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn decode_certificate_hex<const N: usize>(value: &str, label: &str) -> Result<[u8; N], String> {
    if value.len() != N * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be exactly {} lowercase hex bytes", N));
    }
    let mut bytes = [0_u8; N];
    for (index, chunk) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let high = (chunk[0] as char)
            .to_digit(16)
            .ok_or_else(|| format!("invalid {label} hex"))?;
        let low = (chunk[1] as char)
            .to_digit(16)
            .ok_or_else(|| format!("invalid {label} hex"))?;
        bytes[index] = ((high << 4) | low) as u8;
    }
    Ok(bytes)
}

/// Per-sample uncertainty bounds in the measured response tensor's axis order.
///
/// Both tensors are indexed `[source][listener][frequency]`.  Values are
/// positive one-sigma-or-budget-like bounds in the units named by the field;
/// their statistical interpretation is intentionally owned by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredUncertaintyEnvelope {
    /// Digest of the v2 measured dataset this envelope belongs to.
    pub dataset_content_sha256: String,
    /// Positive magnitude uncertainty in dB for every response sample.
    pub magnitude_uncertainty_db: Vec<Vec<Vec<f64>>>,
    /// Positive phase uncertainty in radians for every response sample.
    pub phase_uncertainty_rad: Vec<Vec<Vec<f64>>>,
    /// Digest of this envelope's canonical tensor and dataset binding.
    pub content_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasuredUncertaintyDocument {
    schema_version: String,
    dataset_content_sha256: String,
    magnitude_uncertainty_db: Vec<Vec<Vec<f64>>>,
    phase_uncertainty_rad: Vec<Vec<Vec<f64>>>,
    content_sha256: String,
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn update_len(hasher: &mut Sha256, len: usize) {
    hasher.update((len as u64).to_le_bytes());
}

fn update_str(hasher: &mut Sha256, value: &str) {
    update_len(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn update_tensor(hasher: &mut Sha256, tensor: &[Vec<Vec<f64>>]) {
    update_len(hasher, tensor.len());
    for source in tensor {
        update_len(hasher, source.len());
        for listener in source {
            update_len(hasher, listener.len());
            for value in listener {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
}

fn canonical_content_sha256(
    dataset_content_sha256: &str,
    magnitude_uncertainty_db: &[Vec<Vec<f64>>],
    phase_uncertainty_rad: &[Vec<Vec<f64>>],
) -> String {
    let mut hasher = Sha256::new();
    update_str(&mut hasher, MEASURED_UNCERTAINTY_SCHEMA_V1);
    update_str(&mut hasher, dataset_content_sha256);
    update_tensor(&mut hasher, magnitude_uncertainty_db);
    update_tensor(&mut hasher, phase_uncertainty_rad);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl MeasuredUncertaintyEnvelope {
    /// Build an uncertainty envelope bound to an admitted dataset.
    pub fn from_parts(
        dataset: &MeasuredResponseDataset,
        magnitude_uncertainty_db: Vec<Vec<Vec<f64>>>,
        phase_uncertainty_rad: Vec<Vec<Vec<f64>>>,
    ) -> Result<Self, String> {
        dataset.validate()?;
        let mut envelope = Self {
            dataset_content_sha256: dataset.metadata.content_sha256.clone(),
            magnitude_uncertainty_db,
            phase_uncertainty_rad,
            content_sha256: String::new(),
        };
        envelope.content_sha256 = canonical_content_sha256(
            &envelope.dataset_content_sha256,
            &envelope.magnitude_uncertainty_db,
            &envelope.phase_uncertainty_rad,
        );
        envelope.validate_against(dataset)?;
        Ok(envelope)
    }

    /// Validate identity, tensor topology, finite values, and the digest.
    pub fn validate_against(&self, dataset: &MeasuredResponseDataset) -> Result<(), String> {
        dataset.validate()?;
        if !is_canonical_sha256(&self.dataset_content_sha256)
            || self.dataset_content_sha256 != dataset.metadata.content_sha256
        {
            return Err(
                "measured uncertainty dataset digest does not match admitted response".into(),
            );
        }
        if self.magnitude_uncertainty_db.len() != dataset.response.pressures.len()
            || self.phase_uncertainty_rad.len() != dataset.response.pressures.len()
        {
            return Err("measured uncertainty source axis does not match response".into());
        }
        for source_index in 0..dataset.response.pressures.len() {
            let measured_source = &dataset.response.pressures[source_index];
            let magnitude_source = &self.magnitude_uncertainty_db[source_index];
            let phase_source = &self.phase_uncertainty_rad[source_index];
            if magnitude_source.len() != measured_source.len()
                || phase_source.len() != measured_source.len()
            {
                return Err(format!(
                    "measured uncertainty listener axis does not match source {source_index}"
                ));
            }
            for listener_index in 0..measured_source.len() {
                let expected_len = measured_source[listener_index].len();
                let magnitude = &magnitude_source[listener_index];
                let phase = &phase_source[listener_index];
                if magnitude.len() != expected_len || phase.len() != expected_len {
                    return Err(format!(
                        "measured uncertainty frequency axis does not match source {source_index}, listener {listener_index}"
                    ));
                }
                if magnitude
                    .iter()
                    .chain(phase)
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(format!(
                        "measured uncertainty at source {source_index}, listener {listener_index} must be finite and positive"
                    ));
                }
            }
        }
        if !is_canonical_sha256(&self.content_sha256) {
            return Err(
                "measured uncertainty content_sha256 must be 64 lowercase hexadecimal characters"
                    .into(),
            );
        }
        let expected = canonical_content_sha256(
            &self.dataset_content_sha256,
            &self.magnitude_uncertainty_db,
            &self.phase_uncertainty_rad,
        );
        if self.content_sha256 != expected {
            return Err(format!(
                "measured uncertainty content_sha256 mismatch: expected {expected}"
            ));
        }
        Ok(())
    }

    /// Serialize the validated uncertainty envelope.
    pub fn to_json(&self, dataset: &MeasuredResponseDataset) -> Result<String, String> {
        self.validate_against(dataset)?;
        serde_json::to_string_pretty(&MeasuredUncertaintyDocument {
            schema_version: MEASURED_UNCERTAINTY_SCHEMA_V1.into(),
            dataset_content_sha256: self.dataset_content_sha256.clone(),
            magnitude_uncertainty_db: self.magnitude_uncertainty_db.clone(),
            phase_uncertainty_rad: self.phase_uncertainty_rad.clone(),
            content_sha256: self.content_sha256.clone(),
        })
        .map_err(|error| format!("failed to serialize measured uncertainty: {error}"))
    }

    /// Parse and validate an uncertainty envelope against its measured data.
    pub fn from_json(document: &str, dataset: &MeasuredResponseDataset) -> Result<Self, String> {
        let document: MeasuredUncertaintyDocument = serde_json::from_str(document)
            .map_err(|error| format!("invalid measured uncertainty JSON: {error}"))?;
        if document.schema_version != MEASURED_UNCERTAINTY_SCHEMA_V1 {
            return Err(format!(
                "unsupported measured uncertainty schema {:?}; expected {:?}",
                document.schema_version, MEASURED_UNCERTAINTY_SCHEMA_V1
            ));
        }
        let envelope = Self {
            dataset_content_sha256: document.dataset_content_sha256,
            magnitude_uncertainty_db: document.magnitude_uncertainty_db,
            phase_uncertainty_rad: document.phase_uncertainty_rad,
            content_sha256: document.content_sha256,
        };
        envelope.validate_against(dataset)?;
        Ok(envelope)
    }

    /// Load and validate an uncertainty envelope from disk.
    pub fn load_json(
        path: impl AsRef<Path>,
        dataset: &MeasuredResponseDataset,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let document = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read measured uncertainty {:?}: {error}", path))?;
        Self::from_json(&document, dataset)
    }
}

/// Explicit absolute and uncertainty-normalized acceptance budgets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredAcceptanceBudget {
    /// Maximum absolute SPL error at any sample, in dB.
    pub max_abs_magnitude_error_db: f64,
    /// Maximum absolute wrapped phase error at any non-zero sample, in rad.
    pub max_abs_phase_error_rad: f64,
    /// Maximum magnitude error divided by its supplied uncertainty.
    pub max_normalized_magnitude_error: f64,
    /// Maximum phase error divided by its supplied uncertainty.
    pub max_normalized_phase_error: f64,
    /// Minimum number of non-zero samples required for phase acceptance.
    pub minimum_phase_samples: usize,
}

impl MeasuredAcceptanceBudget {
    fn validate(self) -> Result<(), String> {
        if !self.max_abs_magnitude_error_db.is_finite()
            || self.max_abs_magnitude_error_db < 0.0
            || !self.max_abs_phase_error_rad.is_finite()
            || self.max_abs_phase_error_rad < 0.0
            || !self.max_normalized_magnitude_error.is_finite()
            || self.max_normalized_magnitude_error < 0.0
            || !self.max_normalized_phase_error.is_finite()
            || self.max_normalized_phase_error < 0.0
            || self.minimum_phase_samples == 0
        {
            return Err("measured acceptance budget is invalid".into());
        }
        Ok(())
    }
}

/// Absolute and uncertainty-normalized measured-room acceptance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MeasuredAcceptanceReport {
    pub comparison: MeasuredComparisonReport,
    pub magnitude_sample_count: usize,
    pub phase_sample_count: usize,
    pub rms_normalized_magnitude_error: f64,
    pub max_normalized_magnitude_error: f64,
    pub rms_normalized_phase_error: f64,
    pub max_normalized_phase_error: f64,
}

impl MeasuredAcceptanceReport {
    /// Require both absolute and uncertainty-normalized error budgets.
    pub fn require_within(&self, budget: MeasuredAcceptanceBudget) -> Result<(), String> {
        budget.validate()?;
        if self.magnitude_sample_count == 0
            || self.phase_sample_count < budget.minimum_phase_samples
            || self.phase_sample_count > self.magnitude_sample_count
            || [
                self.rms_normalized_magnitude_error,
                self.max_normalized_magnitude_error,
                self.rms_normalized_phase_error,
                self.max_normalized_phase_error,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("measured acceptance report is invalid or lacks phase coverage".into());
        }
        self.comparison.require_within(
            budget.max_abs_magnitude_error_db,
            budget.max_abs_phase_error_rad,
        )?;
        if self.max_normalized_magnitude_error > budget.max_normalized_magnitude_error {
            return Err(format!(
                "normalized measured magnitude error {:.6} exceeds {:.6} budget",
                self.max_normalized_magnitude_error, budget.max_normalized_magnitude_error
            ));
        }
        if self.max_normalized_phase_error > budget.max_normalized_phase_error {
            return Err(format!(
                "normalized measured phase error {:.6} exceeds {:.6} budget",
                self.max_normalized_phase_error, budget.max_normalized_phase_error
            ));
        }
        Ok(())
    }
}

/// Compare a solved output against a digest-bound uncertainty envelope.
pub fn compare_solved_output_to_measured_with_uncertainty(
    output: &SimulationOutput,
    dataset: &MeasuredResponseDataset,
    uncertainty: &MeasuredUncertaintyEnvelope,
    model_listener_names: &[String],
    frequency_tolerance_hz: f64,
) -> Result<MeasuredAcceptanceReport, String> {
    uncertainty.validate_against(dataset)?;
    let comparison = compare_solved_output_to_measured_dataset(
        output,
        dataset,
        model_listener_names,
        frequency_tolerance_hz,
    )?;

    let mut magnitude_squared = 0.0;
    let mut max_normalized_magnitude_error: f64 = 0.0;
    let mut phase_squared = 0.0;
    let mut max_normalized_phase_error: f64 = 0.0;
    let mut magnitude_sample_count = 0;
    let mut phase_sample_count = 0;

    for (source_index, source) in output.pressures.iter().enumerate() {
        for (listener_index, response) in source.iter().enumerate() {
            for (frequency_index, model_pressure) in response.iter().enumerate() {
                let measured_pressure =
                    dataset.response.pressures[source_index][listener_index][frequency_index];
                let magnitude_error_db = crate::room_metrics::peak_pressure_to_spl(*model_pressure)
                    - crate::room_metrics::peak_pressure_to_spl(measured_pressure);
                let magnitude_uncertainty = uncertainty.magnitude_uncertainty_db[source_index]
                    [listener_index][frequency_index];
                let normalized_magnitude_error = magnitude_error_db.abs() / magnitude_uncertainty;
                magnitude_squared += normalized_magnitude_error * normalized_magnitude_error;
                max_normalized_magnitude_error =
                    max_normalized_magnitude_error.max(normalized_magnitude_error);
                magnitude_sample_count += 1;

                if model_pressure.norm() > 0.0 && measured_pressure.norm() > 0.0 {
                    let phase_uncertainty = uncertainty.phase_uncertainty_rad[source_index]
                        [listener_index][frequency_index];
                    let phase_error = wrap_phase(model_pressure.arg() - measured_pressure.arg());
                    let normalized_phase_error = phase_error.abs() / phase_uncertainty;
                    phase_squared += normalized_phase_error * normalized_phase_error;
                    max_normalized_phase_error =
                        max_normalized_phase_error.max(normalized_phase_error);
                    phase_sample_count += 1;
                }
            }
        }
    }

    let report = MeasuredAcceptanceReport {
        comparison,
        magnitude_sample_count,
        phase_sample_count,
        rms_normalized_magnitude_error: (magnitude_squared / magnitude_sample_count as f64).sqrt(),
        max_normalized_magnitude_error,
        rms_normalized_phase_error: if phase_sample_count == 0 {
            0.0
        } else {
            (phase_squared / phase_sample_count as f64).sqrt()
        },
        max_normalized_phase_error,
    };
    if [
        report.rms_normalized_magnitude_error,
        report.max_normalized_magnitude_error,
        report.rms_normalized_phase_error,
        report.max_normalized_phase_error,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("measured acceptance produced non-finite normalized errors".into());
    }
    Ok(report)
}

fn wrap_phase(phase: f64) -> f64 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrequencyProvenance;
    use crate::measured_io::{MeasuredDatasetMetadata, MeasuredResponseDataset};
    use crate::measured_validation::MeasuredResponseSet;
    use ed25519_zebra::{SigningKey, VerificationKeyBytes};
    use num_complex::Complex64;

    fn dataset() -> MeasuredResponseDataset {
        MeasuredResponseDataset::from_parts(
            MeasuredResponseSet::new(
                vec![100.0, 200.0],
                vec!["source".into()],
                vec![vec![vec![
                    Complex64::from_polar(1.0, 0.2),
                    Complex64::from_polar(2.0, -0.7),
                ]]],
                "fixture-measurement",
            )
            .unwrap(),
            MeasuredDatasetMetadata {
                dataset_id: "room-a".into(),
                instrument_id: "mic-chain-1".into(),
                calibration_reference: "cal-1".into(),
                pressure_unit: "Pa".into(),
                sample_convention: "peak_complex".into(),
                phase_reference: "trigger".into(),
                listener_names: vec!["seat-a".into()],
                content_sha256: String::new(),
            },
        )
        .unwrap()
    }

    fn uncertainty_for(dataset: &MeasuredResponseDataset) -> MeasuredUncertaintyEnvelope {
        MeasuredUncertaintyEnvelope::from_parts(
            dataset,
            vec![vec![vec![0.25, 0.25]]],
            vec![vec![vec![0.1, 0.1]]],
        )
        .unwrap()
    }

    fn solved_output(dataset: &MeasuredResponseDataset) -> SimulationOutput {
        SimulationOutput::solved(
            dataset.response.frequencies.clone(),
            dataset.response.pressures.clone(),
            dataset.response.source_names.clone(),
        )
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn certificate_for(
        dataset: &MeasuredResponseDataset,
    ) -> (MeasuredInstrumentCertificate, MeasuredInstrumentTrustPolicy) {
        let signing_key = SigningKey::from([9_u8; 32]);
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let mut certificate = MeasuredInstrumentCertificate::unsigned(
            "certificate-1",
            dataset,
            "lab-issuer-v1",
            hex(&public_key),
            1_000,
            2_000,
        )
        .unwrap();
        let signature = signing_key.sign(&certificate.signature_payload().unwrap());
        certificate.signature = hex(&signature.to_bytes());
        let policy = MeasuredInstrumentTrustPolicy::new(vec![MeasuredInstrumentTrustAnchor {
            key_id: "lab-issuer-v1".into(),
            public_key: hex(&public_key),
        }])
        .unwrap();
        (certificate, policy)
    }

    #[test]
    fn exact_response_round_trips_uncertainty_identity_and_accepts() {
        let dataset = dataset();
        let uncertainty = uncertainty_for(&dataset);
        let encoded = uncertainty.to_json(&dataset).unwrap();
        let decoded = MeasuredUncertaintyEnvelope::from_json(&encoded, &dataset).unwrap();
        assert_eq!(decoded, uncertainty);

        let report = compare_solved_output_to_measured_with_uncertainty(
            &solved_output(&dataset),
            &dataset,
            &decoded,
            &["seat-a".into()],
            0.0,
        )
        .unwrap();
        assert_eq!(report.magnitude_sample_count, 2);
        assert_eq!(report.phase_sample_count, 2);
        report
            .require_within(MeasuredAcceptanceBudget {
                max_abs_magnitude_error_db: 0.01,
                max_abs_phase_error_rad: 0.01,
                max_normalized_magnitude_error: 0.01,
                max_normalized_phase_error: 0.01,
                minimum_phase_samples: 2,
            })
            .unwrap();
    }

    #[test]
    fn normalized_error_rejects_sample_outside_supplied_uncertainty() {
        let dataset = dataset();
        let uncertainty = uncertainty_for(&dataset);
        let mut output = solved_output(&dataset);
        output.pressures[0][0][0] *= 1.2;
        let report = compare_solved_output_to_measured_with_uncertainty(
            &output,
            &dataset,
            &uncertainty,
            &["seat-a".into()],
            0.0,
        )
        .unwrap();
        assert!(report.max_normalized_magnitude_error > 1.0);
        assert!(
            report
                .require_within(MeasuredAcceptanceBudget {
                    max_abs_magnitude_error_db: 2.0,
                    max_abs_phase_error_rad: 0.01,
                    max_normalized_magnitude_error: 1.0,
                    max_normalized_phase_error: 1.0,
                    minimum_phase_samples: 2,
                })
                .is_err()
        );
    }

    #[test]
    fn uncertainty_binding_and_non_solved_output_fail_closed() {
        let dataset = dataset();
        let mut uncertainty = uncertainty_for(&dataset);
        uncertainty.dataset_content_sha256.replace_range(..1, "0");
        assert!(uncertainty.validate_against(&dataset).is_err());

        let uncertainty = uncertainty_for(&dataset);
        let reduced = SimulationOutput::with_uniform_provenance(
            dataset.response.frequencies.clone(),
            dataset.response.pressures.clone(),
            dataset.response.source_names.clone(),
            FrequencyProvenance::ReducedOrder,
        );
        let error = compare_solved_output_to_measured_with_uncertainty(
            &reduced,
            &dataset,
            &uncertainty,
            &["seat-a".into()],
            0.0,
        )
        .unwrap_err();
        assert!(error.contains("requires solved samples"), "{error}");
    }

    #[test]
    fn malformed_uncertainty_shape_and_budget_are_rejected() {
        let dataset = dataset();
        assert!(
            MeasuredUncertaintyEnvelope::from_parts(
                &dataset,
                vec![vec![vec![0.25]]],
                vec![vec![vec![0.1, 0.1]]],
            )
            .is_err()
        );
        assert!(
            MeasuredAcceptanceBudget {
                max_abs_magnitude_error_db: 0.0,
                max_abs_phase_error_rad: 0.0,
                max_normalized_magnitude_error: 0.0,
                max_normalized_phase_error: 0.0,
                minimum_phase_samples: 0,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn signed_instrument_certificate_binds_dataset_and_admits_evidence() {
        let dataset = dataset();
        let (certificate, policy) = certificate_for(&dataset);
        certificate
            .validate_against(&dataset, 1_500, &policy)
            .unwrap();
        let encoded = certificate.to_json().unwrap();
        let decoded = MeasuredInstrumentCertificate::from_json(&encoded).unwrap();
        let evidence =
            AuthenticatedMeasuredRoomEvidence::from_parts(dataset, decoded, 1_500, &policy)
                .unwrap();
        assert_eq!(evidence.certificate().certificate_id, "certificate-1");
        assert_eq!(evidence.dataset().metadata.instrument_id, "mic-chain-1");
    }

    #[test]
    fn authenticated_evidence_roundtrips_policy_and_accepts_budgeted_output() {
        let dataset = dataset();
        let (certificate, policy) = certificate_for(&dataset);
        let certificate_document = certificate.to_json().unwrap();
        let policy_document = policy.to_json().unwrap();
        let decoded_policy = MeasuredInstrumentTrustPolicy::from_json(&policy_document).unwrap();
        let evidence = AuthenticatedMeasuredRoomEvidence::from_json(
            dataset,
            &certificate_document,
            1_500,
            &decoded_policy,
        )
        .unwrap();
        let uncertainty = uncertainty_for(evidence.dataset());
        let output = solved_output(evidence.dataset());
        let report = evidence
            .accept_solved_output(
                &output,
                &uncertainty,
                &["seat-a".into()],
                0.0,
                MeasuredAcceptanceBudget {
                    max_abs_magnitude_error_db: 0.01,
                    max_abs_phase_error_rad: 0.01,
                    max_normalized_magnitude_error: 0.01,
                    max_normalized_phase_error: 0.01,
                    minimum_phase_samples: 2,
                },
            )
            .unwrap();
        assert_eq!(report.magnitude_sample_count, 2);
        assert_eq!(report.phase_sample_count, 2);
        assert_eq!(evidence.certificate().issuer_key_id, "lab-issuer-v1");
    }

    #[test]
    fn authenticated_evidence_rejects_budget_violation_after_certificate_admission() {
        let dataset = dataset();
        let (certificate, policy) = certificate_for(&dataset);
        let evidence =
            AuthenticatedMeasuredRoomEvidence::from_parts(dataset, certificate, 1_500, &policy)
                .unwrap();
        let uncertainty = uncertainty_for(evidence.dataset());
        let mut output = solved_output(evidence.dataset());
        output.pressures[0][0][0] *= 1.2;
        let error = evidence
            .accept_solved_output(
                &output,
                &uncertainty,
                &["seat-a".into()],
                0.0,
                MeasuredAcceptanceBudget {
                    max_abs_magnitude_error_db: 0.1,
                    max_abs_phase_error_rad: 0.1,
                    max_normalized_magnitude_error: 0.1,
                    max_normalized_phase_error: 0.1,
                    minimum_phase_samples: 2,
                },
            )
            .unwrap_err();
        assert!(
            error.contains("magnitude") || error.contains("normalized"),
            "{error}"
        );
    }

    #[test]
    fn instrument_trust_policy_rejects_empty_duplicate_and_unknown_fields() {
        assert!(MeasuredInstrumentTrustPolicy::new(Vec::new()).is_err());
        let anchor = MeasuredInstrumentTrustAnchor {
            key_id: "issuer".into(),
            public_key: "01".repeat(32),
        };
        assert!(MeasuredInstrumentTrustPolicy::new(vec![anchor.clone(), anchor]).is_err());
        let policy = MeasuredInstrumentTrustPolicy::new(vec![MeasuredInstrumentTrustAnchor {
            key_id: "issuer".into(),
            public_key: "01".repeat(32),
        }])
        .unwrap();
        let malformed = policy
            .to_json()
            .unwrap()
            .replacen('{', "{\"unexpected\":true,", 1);
        assert!(MeasuredInstrumentTrustPolicy::from_json(&malformed).is_err());
    }

    #[test]
    fn instrument_certificate_rejects_tampering_expiry_and_unknown_fields() {
        let dataset = dataset();
        let (certificate, policy) = certificate_for(&dataset);

        let mut tampered = certificate.clone();
        tampered.instrument_id = "other-mic".into();
        assert!(tampered.validate_against(&dataset, 1_500, &policy).is_err());

        let mut bad_signature = certificate.clone();
        bad_signature.signature.replace_range(..2, "00");
        assert!(
            bad_signature
                .validate_against(&dataset, 1_500, &policy)
                .unwrap_err()
                .contains("signature verification")
        );

        assert!(
            certificate
                .validate_against(&dataset, 2_001, &policy)
                .unwrap_err()
                .contains("validity interval")
        );

        let malformed = certificate
            .to_json()
            .unwrap()
            .replacen('{', "{\"unexpected\":true,", 1);
        assert!(MeasuredInstrumentCertificate::from_json(&malformed).is_err());

        let mut uppercase_key = certificate;
        uppercase_key.issuer_public_key = uppercase_key.issuer_public_key.to_uppercase();
        assert!(
            uppercase_key
                .validate_against(&dataset, 1_500, &policy)
                .unwrap_err()
                .contains("lowercase")
        );
    }

    #[test]
    fn instrument_certificate_rejects_untrusted_or_mismatched_issuer() {
        let dataset = dataset();
        let (certificate, policy) = certificate_for(&dataset);
        let other_policy =
            MeasuredInstrumentTrustPolicy::new(vec![MeasuredInstrumentTrustAnchor {
                key_id: "other-issuer".into(),
                public_key: policy.trusted_keys[0].public_key.clone(),
            }])
            .unwrap();
        assert!(
            certificate
                .validate_against(&dataset, 1_500, &other_policy)
                .unwrap_err()
                .contains("not trusted")
        );

        let mut mismatched = certificate;
        mismatched.issuer_public_key = "00".repeat(32);
        assert!(
            mismatched
                .validate_against(&dataset, 1_500, &policy)
                .unwrap_err()
                .contains("not trusted")
        );
    }
}
