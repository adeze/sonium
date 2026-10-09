use sonium_sim_propellers::{
    MEASURED_CALIBRATION_DATASET_SCHEMA_V1, MeasuredCalibrationDataset,
    MeasuredCalibrationDatasetError, MeasuredCalibrationDatasetMetadata,
    MeasuredCalibrationOptions, MeasuredCalibrationSample,
};

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
        MeasuredCalibrationSample {
            operating_point: 0.6,
            predicted_value: 3.0,
            measured_value: 3.6,
        },
    ]
}

#[test]
fn public_dataset_binds_provenance_and_calibration_content() {
    let dataset =
        MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset envelope");
    dataset.validate().expect("dataset validation");
    assert_eq!(dataset.metadata.channel, "thrust");
    assert_eq!(dataset.metadata.unit, "N");
    assert_eq!(dataset.metadata.content_sha256.len(), 64);
    let result = dataset
        .calibrate(MeasuredCalibrationOptions::default())
        .expect("calibration");
    assert!(result.is_finite());
    assert!(result.passed);
}

#[test]
fn public_dataset_json_round_trip_preserves_digest_and_metadata() {
    let dataset =
        MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset envelope");
    let document = dataset.to_json().expect("dataset JSON");
    assert!(document.contains(MEASURED_CALIBRATION_DATASET_SCHEMA_V1));
    let restored = MeasuredCalibrationDataset::from_json(&document).expect("round trip");
    assert_eq!(restored, dataset);
}

#[test]
fn public_dataset_rejects_sample_metadata_and_digest_tampering() {
    let dataset =
        MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset envelope");
    let document = dataset.to_json().expect("dataset JSON");
    for tampered in [
        document.replace("\"measured_value\": 1.2", "\"measured_value\": 1.25"),
        document.replace("public-fixture-v1", "other-fixture-v1"),
        document.replace(&dataset.metadata.content_sha256, &"0".repeat(64)),
    ] {
        assert!(matches!(
            MeasuredCalibrationDataset::from_json(&tampered),
            Err(MeasuredCalibrationDatasetError::DigestMismatch { .. })
                | Err(MeasuredCalibrationDatasetError::InvalidDigest)
        ));
    }
}

#[test]
fn public_dataset_rejects_untrusted_metadata_and_duplicate_conditions() {
    let mut invalid_uri = metadata();
    invalid_uri.source_uri = "synthetic-fixture".into();
    assert!(matches!(
        MeasuredCalibrationDataset::from_parts(samples(), invalid_uri),
        Err(MeasuredCalibrationDatasetError::InvalidMetadata {
            field: "source_uri",
            ..
        })
    ));

    let mut duplicate = samples();
    duplicate[1].operating_point = duplicate[0].operating_point;
    assert!(matches!(
        MeasuredCalibrationDataset::from_parts(duplicate, metadata()),
        Err(MeasuredCalibrationDatasetError::InvalidMetadata {
            field: "operating_point",
            ..
        })
    ));
}

#[test]
fn public_dataset_rejects_unknown_schema_and_json_fields() {
    let dataset =
        MeasuredCalibrationDataset::from_parts(samples(), metadata()).expect("dataset envelope");
    let document = dataset.to_json().expect("dataset JSON");
    let wrong_schema = document.replace(
        MEASURED_CALIBRATION_DATASET_SCHEMA_V1,
        "propeller.measured-calibration-dataset.v0",
    );
    assert!(matches!(
        MeasuredCalibrationDataset::from_json(&wrong_schema),
        Err(MeasuredCalibrationDatasetError::InvalidDocument(_))
    ));
    let unknown_field = document.replace("\"samples\":", "\"unexpected\": 1, \"samples\":");
    assert!(matches!(
        MeasuredCalibrationDataset::from_json(&unknown_field),
        Err(MeasuredCalibrationDatasetError::InvalidDocument(_))
    ));
}
