//! Versioned import boundary for externally supplied measured responses.
//!
//! The importer is deliberately narrow: it accepts a checked JSON document
//! containing peak-pressure complex samples in pascals and converts it to the
//! validated [`crate::measured_validation::MeasuredResponseSet`] tensor. It
//! does not claim that the file came from a calibrated instrument or that the
//! dataset is independent of the model being compared.

use crate::SimulationOutput;
use crate::measured_validation::{
    MeasuredComparisonReport, MeasuredResponseSet, compare_solved_output_to_measured,
};
use ed25519_zebra::{Signature, VerificationKey, VerificationKeyBytes};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;

/// Schema identifier for the checked measured-response JSON document.
pub const MEASURED_RESPONSE_SCHEMA_V1: &str = "roomsim.measured-response.v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasuredResponseDocument {
    schema_version: String,
    frequencies_hz: Vec<f64>,
    source_names: Vec<String>,
    /// Samples indexed `[source][listener][frequency]`.
    pressures: Vec<Vec<Vec<ComplexPressure>>>,
    provenance_label: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ComplexPressure {
    real_pa: f64,
    imag_pa: f64,
}

/// Parse a versioned measured-response JSON document.
pub fn parse_measured_response_json(document: &str) -> Result<MeasuredResponseSet, String> {
    let document: MeasuredResponseDocument = serde_json::from_str(document)
        .map_err(|error| format!("invalid measured-response JSON: {error}"))?;

    if document.schema_version != MEASURED_RESPONSE_SCHEMA_V1 {
        return Err(format!(
            "unsupported measured-response schema {:?}; expected {:?}",
            document.schema_version, MEASURED_RESPONSE_SCHEMA_V1
        ));
    }

    let pressures = document
        .pressures
        .into_iter()
        .map(|source| {
            source
                .into_iter()
                .map(|listener| {
                    listener
                        .into_iter()
                        .map(|pressure| Complex64::new(pressure.real_pa, pressure.imag_pa))
                        .collect()
                })
                .collect()
        })
        .collect();

    MeasuredResponseSet::new(
        document.frequencies_hz,
        document.source_names,
        pressures,
        document.provenance_label,
    )
}

/// Load and parse a versioned measured-response JSON document from disk.
pub fn load_measured_response_json(path: impl AsRef<Path>) -> Result<MeasuredResponseSet, String> {
    let path = path.as_ref();
    let document = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read measured-response {:?}: {error}", path))?;
    parse_measured_response_json(&document)
}

/// Schema identifier for an imported dataset with explicit calibration,
/// listener metadata, and a canonical content digest.
pub const MEASURED_DATASET_SCHEMA_V2: &str = "roomsim.measured-response-dataset.v2";

/// Metadata required before an imported response can enter a calibrated-data
/// comparison workflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MeasuredDatasetMetadata {
    /// Stable identifier supplied by the measurement dataset owner.
    pub dataset_id: String,
    /// Instrument or acquisition chain identifier.
    pub instrument_id: String,
    /// Calibration certificate, run, or reference identifier.
    pub calibration_reference: String,
    /// Unit of the complex pressure samples. The v1 contract is peak Pa.
    pub pressure_unit: String,
    /// Representation of each complex sample. The v1 contract is peak complex.
    pub sample_convention: String,
    /// Phase-reference description, for example a calibrated source trigger.
    pub phase_reference: String,
    /// Stable listener names in the response tensor's listener-axis order.
    pub listener_names: Vec<String>,
    /// Lowercase SHA-256 digest binding the metadata and response tensor.
    pub content_sha256: String,
}

/// Algorithm identifier for detached authenticated measured-data signatures.
pub const MEASURED_DATASET_SIGNATURE_ALGORITHM: &str = "ed25519";
const MEASURED_DATASET_SIGNATURE_DOMAIN: &str = "sonium-sim-room-measured-response-dataset-v1";

/// Detached Ed25519 signature over a canonical measured dataset digest and
/// admission metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetSignature {
    pub algorithm: String,
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key (32 bytes).
    pub public_key: String,
    /// Lowercase hexadecimal Ed25519 signature (64 bytes).
    pub signature: String,
}

/// One host-owned trusted key for authenticated measured datasets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetTrustAnchor {
    pub key_id: String,
    /// Lowercase hexadecimal Ed25519 public key (32 bytes).
    pub public_key: String,
}

/// Explicit trusted-key allowlist for authenticated measured datasets.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredDatasetTrustPolicy {
    pub trusted_keys: Vec<MeasuredDatasetTrustAnchor>,
}

/// A dataset whose detached signature has been verified against a trust
/// policy. The private fields make the authenticated type impossible to build
/// without going through [`Self::from_parts`] or the authenticated loader.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthenticatedMeasuredResponseDataset {
    dataset: MeasuredResponseDataset,
    signature: MeasuredDatasetSignature,
}

/// A measured response tensor with an explicit dataset-admission envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredResponseDataset {
    pub response: MeasuredResponseSet,
    pub metadata: MeasuredDatasetMetadata,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasuredDatasetDocument {
    schema_version: String,
    frequencies_hz: Vec<f64>,
    source_names: Vec<String>,
    /// Samples indexed `[source][listener][frequency]`.
    pressures: Vec<Vec<Vec<ComplexPressure>>>,
    provenance_label: String,
    dataset_id: String,
    instrument_id: String,
    calibration_reference: String,
    pressure_unit: String,
    sample_convention: String,
    phase_reference: String,
    listener_names: Vec<String>,
    /// SHA-256 of the canonical response and metadata payload below.
    content_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signature: Option<MeasuredDatasetSignature>,
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_lower_hex<const N: usize>(value: &str, field: &str) -> Result<[u8; N], String> {
    if value.len() != N * 2 {
        return Err(format!(
            "measured dataset {field} must contain exactly {} hexadecimal characters",
            N * 2
        ));
    }
    let mut bytes = [0_u8; N];
    for (index, chunk) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let high = match chunk[0] {
            b'0'..=b'9' => chunk[0] - b'0',
            b'a'..=b'f' => chunk[0] - b'a' + 10,
            _ => {
                return Err(format!(
                    "measured dataset {field} must use lowercase hexadecimal"
                ));
            }
        };
        let low = match chunk[1] {
            b'0'..=b'9' => chunk[1] - b'0',
            b'a'..=b'f' => chunk[1] - b'a' + 10,
            _ => {
                return Err(format!(
                    "measured dataset {field} must use lowercase hexadecimal"
                ));
            }
        };
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

fn update_payload_str(payload: &mut Vec<u8>, value: &str) {
    payload.extend_from_slice(&(value.len() as u64).to_le_bytes());
    payload.extend_from_slice(value.as_bytes());
}

fn update_len(hasher: &mut Sha256, len: usize) {
    hasher.update((len as u64).to_le_bytes());
}

fn update_str(hasher: &mut Sha256, value: &str) {
    update_len(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn update_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

/// Hash all response and admission metadata fields except the digest itself.
/// Length-prefixing and IEEE-754 bit encoding make the result independent of
/// JSON whitespace, object-key order, and locale formatting.
fn canonical_content_sha256(
    response: &MeasuredResponseSet,
    metadata: &MeasuredDatasetMetadata,
) -> String {
    let mut hasher = Sha256::new();
    update_str(&mut hasher, MEASURED_DATASET_SCHEMA_V2);
    update_str(&mut hasher, &response.provenance_label);

    update_len(&mut hasher, response.frequencies.len());
    for frequency in &response.frequencies {
        update_f64(&mut hasher, *frequency);
    }
    update_len(&mut hasher, response.source_names.len());
    for source_name in &response.source_names {
        update_str(&mut hasher, source_name);
    }
    update_len(&mut hasher, response.pressures.len());
    for source in &response.pressures {
        update_len(&mut hasher, source.len());
        for listener in source {
            update_len(&mut hasher, listener.len());
            for pressure in listener {
                update_f64(&mut hasher, pressure.re);
                update_f64(&mut hasher, pressure.im);
            }
        }
    }

    update_str(&mut hasher, &metadata.dataset_id);
    update_str(&mut hasher, &metadata.instrument_id);
    update_str(&mut hasher, &metadata.calibration_reference);
    update_str(&mut hasher, &metadata.pressure_unit);
    update_str(&mut hasher, &metadata.sample_convention);
    update_str(&mut hasher, &metadata.phase_reference);
    update_len(&mut hasher, metadata.listener_names.len());
    for listener_name in &metadata.listener_names {
        update_str(&mut hasher, listener_name);
    }

    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl MeasuredDatasetTrustPolicy {
    /// Construct and validate a host-owned trusted-key allowlist.
    pub fn new(trusted_keys: Vec<MeasuredDatasetTrustAnchor>) -> Result<Self, String> {
        let policy = Self { trusted_keys };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), String> {
        if self.trusted_keys.is_empty() {
            return Err("measured dataset trust policy must contain a trusted key".into());
        }
        let mut key_ids = HashSet::with_capacity(self.trusted_keys.len());
        for anchor in &self.trusted_keys {
            if anchor.key_id.trim().is_empty() || !key_ids.insert(anchor.key_id.clone()) {
                return Err("measured dataset trust key IDs must be non-empty and unique".into());
            }
            decode_lower_hex::<32>(&anchor.public_key, "trusted public_key")?;
        }
        Ok(())
    }
}

/// Build the domain-separated bytes signed for an authenticated dataset.
///
/// The canonical v2 content digest already covers every response and
/// admission field. Repeating the important admission metadata here makes the
/// detached signature payload self-describing and prevents it from being
/// confused with a signature from another product or envelope version.
pub fn measured_dataset_signature_payload(
    dataset: &MeasuredResponseDataset,
    key_id: &str,
) -> Result<Vec<u8>, String> {
    dataset.validate()?;
    if key_id.trim().is_empty() {
        return Err("measured dataset signature key ID must not be empty".into());
    }

    let metadata = &dataset.metadata;
    let mut payload = Vec::new();
    update_payload_str(&mut payload, MEASURED_DATASET_SIGNATURE_DOMAIN);
    update_payload_str(&mut payload, MEASURED_DATASET_SCHEMA_V2);
    update_payload_str(&mut payload, key_id);
    update_payload_str(&mut payload, &metadata.content_sha256);
    update_payload_str(&mut payload, &metadata.dataset_id);
    update_payload_str(&mut payload, &metadata.instrument_id);
    update_payload_str(&mut payload, &metadata.calibration_reference);
    update_payload_str(&mut payload, &metadata.pressure_unit);
    update_payload_str(&mut payload, &metadata.sample_convention);
    update_payload_str(&mut payload, &metadata.phase_reference);
    payload.extend_from_slice(&(metadata.listener_names.len() as u64).to_le_bytes());
    for listener_name in &metadata.listener_names {
        update_payload_str(&mut payload, listener_name);
    }
    Ok(payload)
}

/// Verify a detached dataset signature against a host-owned key allowlist.
pub fn verify_measured_dataset_signature(
    dataset: &MeasuredResponseDataset,
    signature: &MeasuredDatasetSignature,
    policy: &MeasuredDatasetTrustPolicy,
) -> Result<(), String> {
    dataset.validate()?;
    policy.validate()?;
    if signature.algorithm != MEASURED_DATASET_SIGNATURE_ALGORITHM {
        return Err(format!(
            "measured dataset signature algorithm must be {}",
            MEASURED_DATASET_SIGNATURE_ALGORITHM
        ));
    }
    if signature.key_id.trim().is_empty() {
        return Err("measured dataset signature key ID must not be empty".into());
    }
    let anchor = policy
        .trusted_keys
        .iter()
        .find(|anchor| anchor.key_id == signature.key_id)
        .ok_or_else(|| {
            format!(
                "measured dataset signature key ID {:?} is not trusted",
                signature.key_id
            )
        })?;
    if anchor.public_key != signature.public_key {
        return Err(format!(
            "measured dataset signature public key does not match trusted key {:?}",
            signature.key_id
        ));
    }

    let public_key = decode_lower_hex::<32>(&signature.public_key, "public_key")?;
    let signature_bytes = decode_lower_hex::<64>(&signature.signature, "signature")?;
    let verification_key = VerificationKey::try_from(VerificationKeyBytes::from(public_key))
        .map_err(|error| format!("invalid measured dataset public key: {error:?}"))?;
    let payload = measured_dataset_signature_payload(dataset, &signature.key_id)?;
    verification_key
        .verify(&Signature::from_bytes(&signature_bytes), &payload)
        .map_err(|_| "measured dataset signature verification failed".to_string())
}

impl AuthenticatedMeasuredResponseDataset {
    /// Build an authenticated dataset only after trust-policy verification.
    pub fn from_parts(
        dataset: MeasuredResponseDataset,
        signature: MeasuredDatasetSignature,
        policy: &MeasuredDatasetTrustPolicy,
    ) -> Result<Self, String> {
        verify_measured_dataset_signature(&dataset, &signature, policy)?;
        Ok(Self { dataset, signature })
    }

    /// Access the dataset whose content and admission metadata were signed.
    pub fn dataset(&self) -> &MeasuredResponseDataset {
        &self.dataset
    }

    /// Access the verified detached signature metadata.
    pub fn signature(&self) -> &MeasuredDatasetSignature {
        &self.signature
    }

    /// Recheck the authenticated dataset's digest and signature.
    pub fn validate(&self, policy: &MeasuredDatasetTrustPolicy) -> Result<(), String> {
        verify_measured_dataset_signature(&self.dataset, &self.signature, policy)
    }

    /// Serialize the authenticated v2 envelope, retaining its detached
    /// signature instead of downgrading it to an unsigned dataset.
    pub fn to_json(&self) -> Result<String, String> {
        serialize_measured_dataset_json(&self.dataset, Some(&self.signature))
    }
}

impl MeasuredDatasetMetadata {
    /// Validate metadata and its relationship to the imported tensor.
    pub fn validate_for(&self, response: &MeasuredResponseSet) -> Result<(), String> {
        if self.dataset_id.trim().is_empty() {
            return Err("measured dataset requires a dataset identifier".into());
        }
        if self.instrument_id.trim().is_empty() {
            return Err("measured dataset requires an instrument identifier".into());
        }
        if self.calibration_reference.trim().is_empty() {
            return Err("measured dataset requires a calibration reference".into());
        }
        if self.pressure_unit != "Pa" {
            return Err("measured dataset pressure unit must be Pa".into());
        }
        if self.sample_convention != "peak_complex" {
            return Err("measured dataset sample convention must be peak_complex".into());
        }
        if self.phase_reference.trim().is_empty() {
            return Err("measured dataset requires a phase reference".into());
        }
        if self.listener_names.is_empty()
            || self
                .listener_names
                .iter()
                .any(|name| name.trim().is_empty())
            || self
                .listener_names
                .iter()
                .map(|name| name.to_ascii_lowercase())
                .collect::<HashSet<_>>()
                .len()
                != self.listener_names.len()
        {
            return Err("measured dataset listener names must be non-empty and unique".into());
        }
        if response
            .pressures
            .iter()
            .any(|source| source.len() != self.listener_names.len())
        {
            return Err("measured dataset listener metadata does not match tensor".into());
        }
        if !is_canonical_sha256(&self.content_sha256) {
            return Err(
                "measured dataset content_sha256 must be 64 lowercase hexadecimal characters"
                    .into(),
            );
        }
        let expected = canonical_content_sha256(response, self);
        if self.content_sha256 != expected {
            return Err(format!(
                "measured dataset content_sha256 mismatch: expected {expected}"
            ));
        }
        Ok(())
    }
}

impl MeasuredResponseDataset {
    /// Build an admitted dataset and compute its canonical content digest.
    ///
    /// Use this for in-process data sources. JSON imports must carry and
    /// verify their supplied digest through [`parse_measured_response_dataset_json`].
    pub fn from_parts(
        response: MeasuredResponseSet,
        mut metadata: MeasuredDatasetMetadata,
    ) -> Result<Self, String> {
        response.validate_layout()?;
        metadata.content_sha256 = canonical_content_sha256(&response, &metadata);
        let dataset = Self { response, metadata };
        dataset.validate()?;
        Ok(dataset)
    }

    /// Validate the response and calibrated-data admission envelope.
    pub fn validate(&self) -> Result<(), String> {
        self.response.validate_layout()?;
        self.metadata.validate_for(&self.response)
    }

    /// Serialize a validated dataset using the checked admission envelope.
    pub fn to_json(&self) -> Result<String, String> {
        serialize_measured_dataset_json(self, None)
    }
}

fn serialize_measured_dataset_json(
    dataset: &MeasuredResponseDataset,
    signature: Option<&MeasuredDatasetSignature>,
) -> Result<String, String> {
    dataset.validate()?;
    let pressures = dataset
        .response
        .pressures
        .iter()
        .map(|source| {
            source
                .iter()
                .map(|listener| {
                    listener
                        .iter()
                        .map(|pressure| ComplexPressure {
                            real_pa: pressure.re,
                            imag_pa: pressure.im,
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let document = MeasuredDatasetDocument {
        schema_version: MEASURED_DATASET_SCHEMA_V2.into(),
        frequencies_hz: dataset.response.frequencies.clone(),
        source_names: dataset.response.source_names.clone(),
        pressures,
        provenance_label: dataset.response.provenance_label.clone(),
        dataset_id: dataset.metadata.dataset_id.clone(),
        instrument_id: dataset.metadata.instrument_id.clone(),
        calibration_reference: dataset.metadata.calibration_reference.clone(),
        pressure_unit: dataset.metadata.pressure_unit.clone(),
        sample_convention: dataset.metadata.sample_convention.clone(),
        phase_reference: dataset.metadata.phase_reference.clone(),
        listener_names: dataset.metadata.listener_names.clone(),
        content_sha256: dataset.metadata.content_sha256.clone(),
        signature: signature.cloned(),
    };
    serde_json::to_string_pretty(&document)
        .map_err(|error| format!("failed to serialize measured-response dataset: {error}"))
}

/// Parse a measured-response document that includes calibration and listener
/// metadata. This is an admission contract only; it does not independently
/// verify an instrument, certificate, or dataset's provenance.
pub fn parse_measured_response_dataset_json(
    document: &str,
) -> Result<MeasuredResponseDataset, String> {
    parse_measured_response_dataset_document(document).map(|(dataset, _)| dataset)
}

fn parse_measured_response_dataset_document(
    document: &str,
) -> Result<(MeasuredResponseDataset, Option<MeasuredDatasetSignature>), String> {
    let document: MeasuredDatasetDocument = serde_json::from_str(document)
        .map_err(|error| format!("invalid measured-response dataset JSON: {error}"))?;
    if document.schema_version != MEASURED_DATASET_SCHEMA_V2 {
        return Err(format!(
            "unsupported measured-response dataset schema {:?}; expected {:?}",
            document.schema_version, MEASURED_DATASET_SCHEMA_V2
        ));
    }

    let pressures = document
        .pressures
        .into_iter()
        .map(|source| {
            source
                .into_iter()
                .map(|listener| {
                    listener
                        .into_iter()
                        .map(|pressure| Complex64::new(pressure.real_pa, pressure.imag_pa))
                        .collect()
                })
                .collect()
        })
        .collect();
    let response = MeasuredResponseSet::new(
        document.frequencies_hz,
        document.source_names,
        pressures,
        document.provenance_label,
    )?;
    let dataset = MeasuredResponseDataset {
        response,
        metadata: MeasuredDatasetMetadata {
            dataset_id: document.dataset_id,
            instrument_id: document.instrument_id,
            calibration_reference: document.calibration_reference,
            pressure_unit: document.pressure_unit,
            sample_convention: document.sample_convention,
            phase_reference: document.phase_reference,
            listener_names: document.listener_names,
            content_sha256: document.content_sha256,
        },
    };
    dataset.validate()?;
    Ok((dataset, document.signature))
}

/// Parse and authenticate a signed measured-response dataset using a trusted
/// key allowlist. Unsigned v2 documents remain accepted by
/// [`parse_measured_response_dataset_json`] but cannot produce this type.
pub fn parse_authenticated_measured_response_dataset_json(
    document: &str,
    policy: &MeasuredDatasetTrustPolicy,
) -> Result<AuthenticatedMeasuredResponseDataset, String> {
    let (dataset, signature) = parse_measured_response_dataset_document(document)?;
    let signature = signature.ok_or_else(|| {
        "authenticated measured dataset requires a detached signature".to_string()
    })?;
    AuthenticatedMeasuredResponseDataset::from_parts(dataset, signature, policy)
}

/// Load a calibrated-data admission document from disk.
pub fn load_measured_response_dataset_json(
    path: impl AsRef<Path>,
) -> Result<MeasuredResponseDataset, String> {
    let path = path.as_ref();
    let document = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read measured-response dataset {:?}: {error}",
            path
        )
    })?;
    parse_measured_response_dataset_json(&document)
}

/// Load and authenticate a signed measured-response dataset from disk.
pub fn load_authenticated_measured_response_dataset_json(
    path: impl AsRef<Path>,
    policy: &MeasuredDatasetTrustPolicy,
) -> Result<AuthenticatedMeasuredResponseDataset, String> {
    let path = path.as_ref();
    let document = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read authenticated measured-response dataset {:?}: {error}",
            path
        )
    })?;
    parse_authenticated_measured_response_dataset_json(&document, policy)
}

/// Compare a solved output against an admitted measured dataset while
/// checking listener identity before relying on positional tensor matching.
pub fn compare_solved_output_to_measured_dataset(
    output: &SimulationOutput,
    dataset: &MeasuredResponseDataset,
    model_listener_names: &[String],
    frequency_tolerance_hz: f64,
) -> Result<MeasuredComparisonReport, String> {
    dataset.validate()?;
    if model_listener_names != dataset.metadata.listener_names {
        return Err("measured dataset listener names differ from model listeners".into());
    }
    compare_solved_output_to_measured(output, &dataset.response, frequency_tolerance_hz)
}

/// Compare a solved output against a dataset whose detached signature has
/// already been verified against the supplied trust policy.
pub fn compare_solved_output_to_authenticated_measured_dataset(
    output: &SimulationOutput,
    dataset: &AuthenticatedMeasuredResponseDataset,
    policy: &MeasuredDatasetTrustPolicy,
    model_listener_names: &[String],
    frequency_tolerance_hz: f64,
) -> Result<MeasuredComparisonReport, String> {
    dataset.validate(policy)?;
    compare_solved_output_to_measured_dataset(
        output,
        dataset.dataset(),
        model_listener_names,
        frequency_tolerance_hz,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture() -> &'static str {
        r#"
        {
          "schema_version": "roomsim.measured-response.v1",
          "frequencies_hz": [100.0, 200.0],
          "source_names": ["left"],
          "pressures": [
            [
              [
                {"real_pa": 1.25, "imag_pa": -0.5},
                {"real_pa": 0.25, "imag_pa": 2.0}
              ]
            ]
          ],
          "provenance_label": "bench-mic-2026-08-24"
        }
        "#
    }

    fn dataset_fixture() -> String {
        let fixture = r#"
        {
          "schema_version": "roomsim.measured-response-dataset.v2",
          "frequencies_hz": [100.0, 200.0],
          "source_names": ["left"],
          "pressures": [
            [
              [
                {"real_pa": 1.25, "imag_pa": -0.5},
                {"real_pa": 0.25, "imag_pa": 2.0}
              ]
            ]
          ],
          "provenance_label": "bench-mic-2026-08-24",
          "dataset_id": "room-a-take-01",
          "instrument_id": "interface-7-mic-2",
          "calibration_reference": "cal-2026-08-24-01",
          "pressure_unit": "Pa",
          "sample_convention": "peak_complex",
          "phase_reference": "source-trigger-channel",
          "listener_names": ["left-seat"],
          "content_sha256": ""
        }
        "#;
        let response = MeasuredResponseSet::new(
            vec![100.0, 200.0],
            vec!["left".into()],
            vec![vec![vec![
                Complex64::new(1.25, -0.5),
                Complex64::new(0.25, 2.0),
            ]]],
            "bench-mic-2026-08-24",
        )
        .unwrap();
        let metadata = MeasuredDatasetMetadata {
            dataset_id: "room-a-take-01".into(),
            instrument_id: "interface-7-mic-2".into(),
            calibration_reference: "cal-2026-08-24-01".into(),
            pressure_unit: "Pa".into(),
            sample_convention: "peak_complex".into(),
            phase_reference: "source-trigger-channel".into(),
            listener_names: vec!["left-seat".into()],
            content_sha256: String::new(),
        };
        let digest = canonical_content_sha256(&response, &metadata);
        fixture.replace(
            "\"content_sha256\": \"\"",
            &format!("\"content_sha256\": \"{digest}\""),
        )
    }

    #[test]
    fn parses_and_validates_calibrated_dataset_envelope() {
        let dataset = parse_measured_response_dataset_json(&dataset_fixture()).unwrap();

        assert_eq!(dataset.metadata.dataset_id, "room-a-take-01");
        assert_eq!(dataset.metadata.instrument_id, "interface-7-mic-2");
        assert_eq!(dataset.metadata.listener_names, vec!["left-seat"]);
        assert_eq!(
            dataset.response.pressures[0][0][1],
            Complex64::new(0.25, 2.0)
        );
        dataset.validate().unwrap();
    }

    #[test]
    fn compares_solved_output_only_after_listener_identity_check() {
        let dataset = parse_measured_response_dataset_json(&dataset_fixture()).unwrap();
        let output = SimulationOutput::solved(
            vec![100.0, 200.0],
            vec![vec![vec![
                Complex64::new(1.25, -0.5),
                Complex64::new(0.25, 2.0),
            ]]],
            vec!["left".into()],
        );

        let report = compare_solved_output_to_measured_dataset(
            &output,
            &dataset,
            &["left-seat".into()],
            0.0,
        )
        .unwrap();
        assert_eq!(report.sample_count, 2);
        assert_eq!(report.max_abs_magnitude_error_db, 0.0);

        let error = compare_solved_output_to_measured_dataset(
            &output,
            &dataset,
            &["right-seat".into()],
            0.0,
        )
        .unwrap_err();
        assert!(error.contains("listener names differ"));
    }

    #[test]
    fn rejects_dataset_schema_and_unknown_fields() {
        let old_schema = dataset_fixture().replace(
            "roomsim.measured-response-dataset.v2",
            "roomsim.measured-response-dataset.v1",
        );
        let error = parse_measured_response_dataset_json(&old_schema).unwrap_err();
        assert!(error.contains("expected \"roomsim.measured-response-dataset.v2\""));

        let wrong_schema = dataset_fixture().replace(
            "roomsim.measured-response-dataset.v2",
            "roomsim.measured-response-dataset.v0",
        );
        let error = parse_measured_response_dataset_json(&wrong_schema).unwrap_err();
        assert!(error.contains("unsupported measured-response dataset schema"));

        let unknown_field = dataset_fixture().replace(
            "\"listener_names\": [\"left-seat\"]",
            "\"listener_names\": [\"left-seat\"],\n          \"temperature_c\": 20.0",
        );
        let error = parse_measured_response_dataset_json(&unknown_field).unwrap_err();
        assert!(error.contains("invalid measured-response dataset JSON"));
    }

    #[test]
    fn rejects_unproven_calibration_metadata() {
        for (field, value, expected) in [
            (
                "\"calibration_reference\": \"cal-2026-08-24-01\"",
                "\"calibration_reference\": \" \"",
                "calibration reference",
            ),
            (
                "\"pressure_unit\": \"Pa\"",
                "\"pressure_unit\": \"dB\"",
                "pressure unit must be Pa",
            ),
            (
                "\"sample_convention\": \"peak_complex\"",
                "\"sample_convention\": \"rms_complex\"",
                "sample convention must be peak_complex",
            ),
            (
                "\"phase_reference\": \"source-trigger-channel\"",
                "\"phase_reference\": \" \"",
                "phase reference",
            ),
        ] {
            let malformed = dataset_fixture().replace(field, value);
            let error = parse_measured_response_dataset_json(&malformed).unwrap_err();
            assert!(error.contains(expected), "{field}: {error}");
        }
    }

    #[test]
    fn rejects_listener_name_duplicates_and_axis_mismatch() {
        let duplicate = dataset_fixture().replace(
            "\"listener_names\": [\"left-seat\"]",
            "\"listener_names\": [\"left-seat\", \"LEFT-SEAT\"]",
        );
        let error = parse_measured_response_dataset_json(&duplicate).unwrap_err();
        assert!(error.contains("listener names must be non-empty and unique"));

        let mismatch = dataset_fixture().replace(
            "\"listener_names\": [\"left-seat\"]",
            "\"listener_names\": [\"left-seat\", \"right-seat\"]",
        );
        let error = parse_measured_response_dataset_json(&mismatch).unwrap_err();
        assert!(error.contains("listener metadata does not match tensor"));
    }

    #[test]
    fn rejects_malformed_and_tampered_content_digests() {
        let malformed = dataset_fixture().replace(
            "\"content_sha256\": \"",
            "\"content_sha256\": \"not-a-digest",
        );
        let error = parse_measured_response_dataset_json(&malformed).unwrap_err();
        assert!(error.contains("content_sha256 must be 64 lowercase hexadecimal"));

        let valid_digest = canonical_content_sha256(
            &MeasuredResponseSet::new(
                vec![100.0, 200.0],
                vec!["left".into()],
                vec![vec![vec![
                    Complex64::new(1.25, -0.5),
                    Complex64::new(0.25, 2.0),
                ]]],
                "bench-mic-2026-08-24",
            )
            .unwrap(),
            &MeasuredDatasetMetadata {
                dataset_id: "room-a-take-01".into(),
                instrument_id: "interface-7-mic-2".into(),
                calibration_reference: "cal-2026-08-24-01".into(),
                pressure_unit: "Pa".into(),
                sample_convention: "peak_complex".into(),
                phase_reference: "source-trigger-channel".into(),
                listener_names: vec!["left-seat".into()],
                content_sha256: String::new(),
            },
        );
        let tampered_digest = format!(
            "{}{}",
            if valid_digest.starts_with('0') {
                '1'
            } else {
                '0'
            },
            &valid_digest[1..]
        );
        let tampered = dataset_fixture().replace(&valid_digest, &tampered_digest);
        let error = parse_measured_response_dataset_json(&tampered).unwrap_err();
        assert!(error.contains("content_sha256 mismatch"));
    }

    #[test]
    fn dataset_json_round_trips_canonical_content_digest() {
        let dataset = parse_measured_response_dataset_json(&dataset_fixture()).unwrap();
        let encoded = dataset.to_json().unwrap();
        let decoded = parse_measured_response_dataset_json(&encoded).unwrap();
        assert_eq!(decoded, dataset);
        assert_eq!(decoded.metadata.content_sha256.len(), 64);
    }

    #[test]
    fn loads_calibrated_dataset_from_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(dataset_fixture().as_bytes()).unwrap();

        let dataset = load_measured_response_dataset_json(file.path()).unwrap();
        assert_eq!(dataset.metadata.calibration_reference, "cal-2026-08-24-01");
    }

    #[test]
    fn parses_checked_complex_response_document() {
        let measured = parse_measured_response_json(fixture()).unwrap();

        assert_eq!(measured.frequencies, vec![100.0, 200.0]);
        assert_eq!(measured.source_names, vec!["left"]);
        assert_eq!(measured.pressures[0][0][0], Complex64::new(1.25, -0.5));
        assert_eq!(measured.pressures[0][0][1], Complex64::new(0.25, 2.0));
        assert_eq!(measured.provenance_label, "bench-mic-2026-08-24");
    }

    #[test]
    fn rejects_schema_drift_and_unknown_fields() {
        let wrong_schema = fixture().replace(
            "roomsim.measured-response.v1",
            "roomsim.measured-response.v0",
        );
        let error = parse_measured_response_json(&wrong_schema).unwrap_err();
        assert!(error.contains("unsupported measured-response schema"));

        let unknown_field = fixture().replace(
            "\"provenance_label\": \"bench-mic-2026-08-24\"",
            "\"provenance_label\": \"bench-mic-2026-08-24\",\n          \"units\": \"Pa\"",
        );
        let error = parse_measured_response_json(&unknown_field).unwrap_err();
        assert!(error.contains("invalid measured-response JSON"));
    }

    #[test]
    fn delegates_topology_and_provenance_validation() {
        let malformed = fixture()
            .replace("[100.0, 200.0]", "[200.0, 100.0]")
            .replace("bench-mic-2026-08-24", "  ");
        let error = parse_measured_response_json(&malformed).unwrap_err();
        assert!(
            error
                .contains("measured frequencies must be finite, positive, and strictly increasing")
                || error.contains("measured response requires a provenance label")
        );
    }

    #[test]
    fn loads_document_from_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(fixture().as_bytes()).unwrap();

        let measured = load_measured_response_json(file.path()).unwrap();
        assert_eq!(measured.pressures[0][0][1], Complex64::new(0.25, 2.0));
    }

    #[test]
    fn reports_file_read_errors() {
        let error = load_measured_response_json("/definitely/not/a/roomsim-measured-response.json")
            .unwrap_err();
        assert!(error.contains("failed to read measured-response"));
    }

    fn signing_key_and_policy() -> (ed25519_zebra::SigningKey, MeasuredDatasetTrustPolicy) {
        let signing_key = ed25519_zebra::SigningKey::from([7_u8; 32]);
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let policy = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
            key_id: "lab-key-v1".into(),
            public_key: hex(&public_key),
        }])
        .unwrap();
        (signing_key, policy)
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn signed_dataset_fixture() -> (String, MeasuredDatasetTrustPolicy) {
        let dataset = parse_measured_response_dataset_json(&dataset_fixture()).unwrap();
        let (signing_key, policy) = signing_key_and_policy();
        let payload = measured_dataset_signature_payload(&dataset, "lab-key-v1").unwrap();
        let signature = signing_key.sign(&payload);
        let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
        let detached = MeasuredDatasetSignature {
            algorithm: MEASURED_DATASET_SIGNATURE_ALGORITHM.into(),
            key_id: "lab-key-v1".into(),
            public_key: hex(&public_key),
            signature: hex(&signature.to_bytes()),
        };
        let authenticated =
            AuthenticatedMeasuredResponseDataset::from_parts(dataset, detached, &policy).unwrap();
        (authenticated.to_json().unwrap(), policy)
    }

    #[test]
    fn authenticates_dataset_with_trusted_ed25519_signature() {
        let (document, policy) = signed_dataset_fixture();
        let authenticated =
            parse_authenticated_measured_response_dataset_json(&document, &policy).unwrap();

        assert_eq!(authenticated.signature().key_id, "lab-key-v1");
        assert_eq!(
            authenticated.dataset().metadata.dataset_id,
            "room-a-take-01"
        );
        authenticated.validate(&policy).unwrap();
    }

    #[test]
    fn rejects_signature_tampering_and_unsigned_authentication() {
        let (document, policy) = signed_dataset_fixture();
        let tampered = document.replacen("\"real_pa\": 1.25", "\"real_pa\": 1.5", 1);
        let error =
            parse_authenticated_measured_response_dataset_json(&tampered, &policy).unwrap_err();
        assert!(
            error.contains("content_sha256 mismatch"),
            "tampering should fail before signature verification: {error}"
        );

        let error = parse_authenticated_measured_response_dataset_json(&dataset_fixture(), &policy)
            .unwrap_err();
        assert!(error.contains("requires a detached signature"));

        let unsigned = parse_measured_response_dataset_json(&dataset_fixture()).unwrap();
        let error = AuthenticatedMeasuredResponseDataset::from_parts(
            unsigned,
            MeasuredDatasetSignature {
                algorithm: MEASURED_DATASET_SIGNATURE_ALGORITHM.into(),
                key_id: "lab-key-v1".into(),
                public_key: "00".repeat(32),
                signature: "00".repeat(64),
            },
            &policy,
        )
        .unwrap_err();
        assert!(error.contains("public key does not match"));
    }

    #[test]
    fn rejects_wrong_key_and_untrusted_key_id() {
        let (document, policy) = signed_dataset_fixture();
        let value: serde_json::Value = serde_json::from_str(&document).unwrap();
        let signature: MeasuredDatasetSignature =
            serde_json::from_value(value["signature"].clone()).unwrap();
        let dataset = parse_measured_response_dataset_json(&document).unwrap();

        let wrong_key_policy = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
            key_id: "lab-key-v1".into(),
            public_key: "11".repeat(32),
        }])
        .unwrap();
        let error =
            verify_measured_dataset_signature(&dataset, &signature, &wrong_key_policy).unwrap_err();
        assert!(error.contains("public key does not match"));

        let untrusted_policy = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
            key_id: "other-key".into(),
            public_key: policy.trusted_keys[0].public_key.clone(),
        }])
        .unwrap();
        let error =
            verify_measured_dataset_signature(&dataset, &signature, &untrusted_policy).unwrap_err();
        assert!(error.contains("is not trusted"));
    }

    #[test]
    fn rejects_malformed_signature_hex_and_unsupported_algorithm() {
        let (document, policy) = signed_dataset_fixture();
        let dataset = parse_measured_response_dataset_json(&document).unwrap();
        let value: serde_json::Value = serde_json::from_str(&document).unwrap();
        let mut signature: MeasuredDatasetSignature =
            serde_json::from_value(value["signature"].clone()).unwrap();

        signature.signature = "zz".repeat(64);
        let error = verify_measured_dataset_signature(&dataset, &signature, &policy).unwrap_err();
        assert!(error.contains("signature must use lowercase hexadecimal"));

        signature.signature = value["signature"].as_object().unwrap()["signature"]
            .as_str()
            .unwrap()
            .into();
        signature.algorithm = "rsa-sha256".into();
        let error = verify_measured_dataset_signature(&dataset, &signature, &policy).unwrap_err();
        assert!(error.contains("algorithm must be ed25519"));
    }
}
