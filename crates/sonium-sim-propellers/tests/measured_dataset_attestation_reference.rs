use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use sonium_sim_propellers::{
    MeasuredCalibrationDataset, MeasuredCalibrationDatasetMetadata, MeasuredCalibrationSample,
    MeasuredDatasetAttestation, MeasuredDatasetAttestationError, MeasuredDatasetTrustAnchor,
    MeasuredDatasetTrustPolicy, measured_dataset_attestation_payload,
};

fn dataset() -> MeasuredCalibrationDataset {
    MeasuredCalibrationDataset::from_parts(
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
            dataset_id: "attestation-fixture-v1".into(),
            source_uri: "https://example.invalid/propeller/attestation-fixture-v1".into(),
            license: "CC-BY-4.0".into(),
            instrument_id: "rig-fixture-01".into(),
            calibration_reference: "certificate-fixture-v1".into(),
            channel: "thrust".into(),
            unit: "N".into(),
            content_sha256: String::new(),
        },
    )
    .expect("dataset fixture")
}

fn attestation() -> MeasuredDatasetAttestation {
    let signing_key = SigningKey::from([7_u8; 32]);
    let key_id = "fixture-key-v1";
    let payload = measured_dataset_attestation_payload(&dataset(), key_id).expect("payload");
    let detached = signing_key.sign(&payload);
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    MeasuredDatasetAttestation::from_signed_parts(
        &dataset(),
        key_id,
        public_key,
        detached.to_bytes(),
    )
    .expect("signed attestation")
}

#[test]
fn signed_dataset_verifies_and_round_trips() {
    let attestation = attestation();
    let json = attestation.to_json().expect("attestation JSON");
    let decoded = MeasuredDatasetAttestation::from_json(&json).expect("decoded attestation");

    assert_eq!(decoded, attestation);
    assert_eq!(decoded.verify().expect("verified dataset").samples.len(), 3);
}

#[test]
fn signed_dataset_payload_is_domain_and_key_bound() {
    let dataset = dataset();
    let first = measured_dataset_attestation_payload(&dataset, "key-a").expect("first payload");
    let second = measured_dataset_attestation_payload(&dataset, "key-b").expect("second payload");

    assert_ne!(first, second);
    assert!(first.starts_with(b"sonium-propeller-measured-dataset-attestation-v1\0"));
}

#[test]
fn tampered_signature_and_dataset_fail_closed() {
    let mut tampered = attestation();
    tampered.signature.replace_range(0..2, "00");
    assert!(matches!(
        tampered.verify(),
        Err(MeasuredDatasetAttestationError::SignatureMismatch)
    ));

    let mut changed = attestation();
    changed.dataset_json = changed.dataset_json.replace("1.1", "1.2");
    assert!(matches!(
        changed.verify(),
        Err(MeasuredDatasetAttestationError::Dataset(_))
            | Err(MeasuredDatasetAttestationError::NonCanonicalDataset)
            | Err(MeasuredDatasetAttestationError::SignatureMismatch)
    ));
}

#[test]
fn wrong_public_key_and_hex_case_are_rejected() {
    let mut wrong_key = attestation();
    wrong_key.public_key = "11".repeat(32);
    assert!(matches!(
        wrong_key.verify(),
        Err(MeasuredDatasetAttestationError::SignatureMismatch)
            | Err(MeasuredDatasetAttestationError::InvalidHex { .. })
    ));

    let mut uppercase = attestation();
    uppercase.public_key = uppercase.public_key.to_ascii_uppercase();
    assert!(matches!(
        uppercase.verify(),
        Err(MeasuredDatasetAttestationError::InvalidHex {
            field: "public_key"
        })
    ));
}

#[test]
fn unsupported_algorithm_schema_and_key_id_are_rejected() {
    let mut algorithm = attestation();
    algorithm.algorithm = "rsa".into();
    assert!(matches!(
        algorithm.verify(),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "algorithm",
            ..
        })
    ));

    let mut schema = attestation();
    schema.schema_version = "future".into();
    assert!(matches!(
        schema.verify(),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "schema_version",
            ..
        })
    ));

    let mut key_id = attestation();
    key_id.key_id = "bad\nkey".into();
    assert!(matches!(
        key_id.verify(),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "key_id",
            ..
        })
    ));
}

#[test]
fn signing_payload_rejects_noncanonical_dataset_json() {
    let mut attestation = attestation();
    attestation.dataset_json = format!(" {}", attestation.dataset_json);
    assert!(matches!(
        attestation.signing_payload(),
        Err(MeasuredDatasetAttestationError::NonCanonicalDataset)
    ));
}
#[test]
fn trust_policy_pins_key_id_and_public_key() {
    let signed = attestation();
    let policy = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
        key_id: signed.key_id.clone(),
        public_key: signed.public_key.clone(),
    }])
    .expect("valid trust policy");

    assert_eq!(signed.verify_with_policy(&policy).unwrap(), dataset());

    let untrusted = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
        key_id: "other-key".into(),
        public_key: signed.public_key.clone(),
    }])
    .expect("valid alternate trust policy");
    assert!(matches!(
        signed.verify_with_policy(&untrusted),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "key_id",
            ..
        })
    ));

    let mismatched = MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
        key_id: signed.key_id.clone(),
        public_key: "00".repeat(32),
    }])
    .expect("valid alternate public key");
    assert!(matches!(
        signed.verify_with_policy(&mismatched),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "public_key",
            ..
        })
    ));
}

#[test]
fn trust_policy_rejects_empty_duplicate_and_malformed_anchors() {
    assert!(matches!(
        MeasuredDatasetTrustPolicy::new(Vec::new()),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "trusted_keys",
            ..
        })
    ));

    let anchor = MeasuredDatasetTrustAnchor {
        key_id: "duplicate".into(),
        public_key: "00".repeat(32),
    };
    assert!(matches!(
        MeasuredDatasetTrustPolicy::new(vec![anchor.clone(), anchor]),
        Err(MeasuredDatasetAttestationError::InvalidField {
            field: "trusted_keys",
            ..
        })
    ));

    assert!(matches!(
        MeasuredDatasetTrustPolicy::new(vec![MeasuredDatasetTrustAnchor {
            key_id: "malformed".into(),
            public_key: "GG".repeat(32),
        }]),
        Err(MeasuredDatasetAttestationError::InvalidHex { .. })
    ));
}
