use sonium_sim_structural_mechanics::{
    MEASURED_FIELD_VALIDATION_DIGEST_ALGORITHM, MEASURED_FIELD_VALIDATION_SCHEMA_VERSION,
    MeasuredFieldChannel, MeasuredFieldEvidenceSource, MeasuredFieldPoint, MeasuredFieldQuantity,
    MeasuredFieldValidationIntegrity, MeasuredFieldValidationStudy,
    compare_measured_field_validation, measured_field_input_sha256,
};

fn study() -> MeasuredFieldValidationStudy {
    MeasuredFieldValidationStudy {
        case_id: "fixture:multi-channel-structural-validation".to_string(),
        source_id: "fixture:caller-supplied".to_string(),
        input_sha256: None,
        channels: vec![
            MeasuredFieldChannel {
                channel_id: "displacement-x".to_string(),
                quantity: MeasuredFieldQuantity::DisplacementComponent { axis: 0 },
                points: vec![
                    MeasuredFieldPoint {
                        coordinate_m: [0.0, 0.0, 0.0],
                        predicted: 1.0e-3,
                        measured: 1.001e-3,
                    },
                    MeasuredFieldPoint {
                        coordinate_m: [0.1, 0.0, 0.0],
                        predicted: 2.0e-3,
                        measured: 2.001e-3,
                    },
                ],
                absolute_tolerance: 5.0e-6,
                relative_tolerance: 0.01,
            },
            MeasuredFieldChannel {
                channel_id: "von-mises".to_string(),
                quantity: MeasuredFieldQuantity::VonMisesStress,
                points: vec![
                    MeasuredFieldPoint {
                        coordinate_m: [0.0, 0.1, 0.0],
                        predicted: 100.0e6,
                        measured: 100.2e6,
                    },
                    MeasuredFieldPoint {
                        coordinate_m: [0.1, 0.1, 0.0],
                        predicted: 120.0e6,
                        measured: 120.1e6,
                    },
                ],
                absolute_tolerance: 1.0e6,
                relative_tolerance: 0.01,
            },
            MeasuredFieldChannel {
                channel_id: "mode-1-frequency".to_string(),
                quantity: MeasuredFieldQuantity::ModalFrequencyHz,
                points: vec![MeasuredFieldPoint {
                    coordinate_m: [0.0, 0.0, 0.0],
                    predicted: 42.0,
                    measured: 42.1,
                }],
                absolute_tolerance: 0.5,
                relative_tolerance: 0.01,
            },
            MeasuredFieldChannel {
                channel_id: "mode-1-shape-x".to_string(),
                quantity: MeasuredFieldQuantity::ModalShapeComponent { mode: 1, axis: 0 },
                points: vec![
                    MeasuredFieldPoint {
                        coordinate_m: [0.0, 0.0, 0.0],
                        predicted: 0.0,
                        measured: 0.0,
                    },
                    MeasuredFieldPoint {
                        coordinate_m: [0.1, 0.0, 0.0],
                        predicted: 0.5,
                        measured: 0.501,
                    },
                ],
                absolute_tolerance: 0.01,
                relative_tolerance: 0.02,
            },
        ],
    }
}

#[test]
fn multi_channel_validation_accepts_displacement_stress_and_modal_channels() {
    let mut input = study();
    input.input_sha256 = Some(measured_field_input_sha256(&input));
    let result = compare_measured_field_validation(&input).expect("fixture should validate");

    assert_eq!(
        result.schema_version,
        MEASURED_FIELD_VALIDATION_SCHEMA_VERSION
    );
    assert_eq!(result.channels.len(), 4);
    assert!(result.accepted);
    assert_eq!(
        result.evidence_source,
        MeasuredFieldEvidenceSource::CallerSupplied
    );
    assert_eq!(
        result.integrity,
        MeasuredFieldValidationIntegrity::Sha256Validated
    );
    assert!(
        result
            .execution
            .capability_id
            .contains("measured-field-validation")
    );
    assert!(result.channels.iter().all(|channel| channel.accepted));
}

#[test]
fn one_failed_channel_blocks_aggregate_acceptance_without_hiding_metrics() {
    let mut input = study();
    input.channels[1].points[0].measured += 5.0e6;
    let result = compare_measured_field_validation(&input).expect("finite data should compare");

    assert!(!result.accepted);
    assert!(result.channels[0].accepted);
    assert!(!result.channels[1].accepted);
    assert!(result.channels[1].max_abs_error > input.channels[1].absolute_tolerance);
}

#[test]
fn validation_rejects_duplicate_ids_invalid_modal_shape_and_duplicate_coordinates() {
    let mut duplicate_id = study();
    duplicate_id.channels[1].channel_id = duplicate_id.channels[0].channel_id.clone();
    assert!(compare_measured_field_validation(&duplicate_id).is_err());

    let mut invalid_modal_shape = study();
    invalid_modal_shape.channels[2]
        .points
        .push(MeasuredFieldPoint {
            coordinate_m: [0.2, 0.0, 0.0],
            predicted: 42.0,
            measured: 42.0,
        });
    assert!(compare_measured_field_validation(&invalid_modal_shape).is_err());

    let mut duplicate_coordinate = study();
    duplicate_coordinate.channels[0].points[1].coordinate_m =
        duplicate_coordinate.channels[0].points[0].coordinate_m;
    assert!(compare_measured_field_validation(&duplicate_coordinate).is_err());
}

#[test]
fn tampered_digest_is_rejected_and_bound_result_round_trips() {
    let mut input = study();
    input.input_sha256 = Some(measured_field_input_sha256(&input));
    input.channels[0].points[0].measured += 1.0e-4;
    assert!(
        compare_measured_field_validation(&input)
            .expect_err("tampering must fail closed")
            .to_string()
            .contains("input_sha256")
    );

    let mut valid = study();
    valid.input_sha256 = Some(measured_field_input_sha256(&valid));
    let result = compare_measured_field_validation(&valid).expect("bound fixture should compare");
    let encoded = serde_json::to_string(&result).expect("result should serialize");
    let decoded: sonium_sim_structural_mechanics::MeasuredFieldValidationResult =
        serde_json::from_str(&encoded).expect("result should deserialize");
    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study.case_id, result.study.case_id);
    assert_eq!(decoded.study.input_sha256, result.study.input_sha256);
    assert_eq!(decoded.channels.len(), result.channels.len());
    assert_eq!(decoded.accepted, result.accepted);
    for (decoded_channel, result_channel) in decoded.channels.iter().zip(&result.channels) {
        assert_eq!(decoded_channel.channel_id, result_channel.channel_id);
        assert_eq!(decoded_channel.quantity, result_channel.quantity);
        assert_eq!(decoded_channel.point_count, result_channel.point_count);
        assert_eq!(decoded_channel.accepted, result_channel.accepted);
        assert!(
            (decoded_channel.rmse - result_channel.rmse).abs()
                <= 1.0e-12 * (1.0 + result_channel.rmse.abs())
        );
        assert!(
            (decoded_channel.max_abs_error - result_channel.max_abs_error).abs()
                <= 1.0e-12 * (1.0 + result_channel.max_abs_error.abs())
        );
    }
    assert_eq!(MEASURED_FIELD_VALIDATION_DIGEST_ALGORITHM, "sha256");
}
