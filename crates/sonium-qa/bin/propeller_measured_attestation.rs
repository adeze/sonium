//! Compile-only QA wrapper for authenticated measured-dataset attestations.

use std::time::Instant;

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MeasuredCalibrationDataset, MeasuredCalibrationDatasetMetadata, MeasuredCalibrationSample,
    MeasuredDatasetAttestation, MeasuredDatasetTrustAnchor, MeasuredDatasetTrustPolicy,
    measured_dataset_attestation_payload,
};

const CASE_ID: &str = "propeller.measured-dataset-attestation.v1";

fn run() -> Result<(f64, f64), String> {
    let dataset = MeasuredCalibrationDataset::from_parts(
        vec![
            MeasuredCalibrationSample {
                operating_point: 1_000.0,
                predicted_value: 1.0,
                measured_value: 1.1,
            },
            MeasuredCalibrationSample {
                operating_point: 2_000.0,
                predicted_value: 2.0,
                measured_value: 2.2,
            },
            MeasuredCalibrationSample {
                operating_point: 3_000.0,
                predicted_value: 3.0,
                measured_value: 3.3,
            },
        ],
        MeasuredCalibrationDatasetMetadata {
            dataset_id: "qa-attestation-fixture-v1".into(),
            source_uri: "https://example.invalid/propeller/qa-attestation-fixture-v1".into(),
            license: "CC-BY-4.0".into(),
            instrument_id: "rig-fixture-01".into(),
            calibration_reference: "certificate-fixture-v1".into(),
            channel: "thrust".into(),
            unit: "N".into(),
            content_sha256: String::new(),
        },
    )
    .map_err(|error| error.to_string())?;
    let key_id = "qa-attestation-key-v1";
    let signing_key = SigningKey::from([9_u8; 32]);
    let payload = measured_dataset_attestation_payload(&dataset, key_id)
        .map_err(|error| error.to_string())?;
    let detached = signing_key.sign(&payload);
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let attestation = MeasuredDatasetAttestation::from_signed_parts(
        &dataset,
        key_id,
        public_key,
        detached.to_bytes(),
    )
    .map_err(|error| error.to_string())?;
    let policy = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
        key_id: key_id.into(),
        public_key: public_key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    }])
    .map_err(|error| error.to_string())?;
    attestation
        .verify_with_policy(&policy)
        .map_err(|error| error.to_string())?;
    let json = attestation.to_json().map_err(|error| error.to_string())?;
    let decoded =
        MeasuredDatasetAttestation::from_json(&json).map_err(|error| error.to_string())?;
    let verified = decoded.verify().map_err(|error| error.to_string())?;
    if verified.samples.len() != 3 || json.is_empty() {
        return Err("attestation round-trip did not retain dataset".into());
    }
    Ok((payload.len() as f64, json.len() as f64))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((payload_bytes, document_bytes)) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:propeller-measured-attestation-v1")
                .reference_checksum("ed25519/invariant:propeller-measured-attestation-v1")
                .dimension("three measured samples, Ed25519 detached signature")
                .formulation("canonical dataset JSON domain-separated Ed25519 verification")
                .tolerance("signature and canonical dataset round-trip verify exactly")
                .metric("payload_bytes", payload_bytes)
                .metric("document_bytes", document_bytes)
                .seed("propeller-measured-attestation-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serializes")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
