use sonium_sim_structural_mechanics::{
    MEASURED_VALIDATION_DIGEST_ALGORITHM, MEASURED_VALIDATION_SCHEMA_VERSION,
    MeasuredValidationEvidenceSource, MeasuredValidationIntegrity, MeasuredValidationStudy,
    compare_measured_displacements, measured_validation_input_sha256,
};

#[test]
fn measured_displacement_validation_accepts_bounded_curve() {
    let mut study = MeasuredValidationStudy::introductory();
    study.input_sha256 = Some(measured_validation_input_sha256(&study));
    let result = compare_measured_displacements(&study).expect("measured curve should validate");

    assert_eq!(result.schema_version, MEASURED_VALIDATION_SCHEMA_VERSION);
    assert_eq!(result.point_count, 5);
    assert_eq!(
        result.evidence_source,
        MeasuredValidationEvidenceSource::CallerSupplied
    );
    assert!(result.absolute_gate_passed);
    assert!(result.relative_gate_passed);
    assert!(result.accepted);
    assert!(result.rmse_m.is_finite());
}

#[test]
fn measured_displacement_validation_reports_gate_failure_without_claiming_acceptance() {
    let mut study = MeasuredValidationStudy::introductory();
    study.points[3].measured_m += 1.0e-3;
    let result = compare_measured_displacements(&study).expect("finite data should compare");

    assert!(!result.absolute_gate_passed);
    assert!(!result.relative_gate_passed);
    assert!(!result.accepted);
    assert!(result.max_abs_error_m > study.absolute_tolerance_m);
}

#[test]
fn measured_displacement_validation_result_round_trips() {
    let result = compare_measured_displacements(&MeasuredValidationStudy::introductory())
        .expect("measured curve should validate");
    let encoded = serde_json::to_string(&result).expect("serialize measured result");
    let decoded: sonium_sim_structural_mechanics::MeasuredValidationResult =
        serde_json::from_str(&encoded).expect("deserialize measured result");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study.case_id, result.study.case_id);
    assert_eq!(decoded.study.source_id, result.study.source_id);
    assert_eq!(decoded.point_count, result.point_count);
    assert_eq!(decoded.accepted, result.accepted);
    assert_eq!(decoded.execution, result.execution);
    assert_eq!(decoded.integrity, result.integrity);
}

#[test]
fn measured_displacement_validation_rejects_invalid_grid_and_tolerances() {
    let mut duplicate_coordinate = MeasuredValidationStudy::introductory();
    duplicate_coordinate.points[2].coordinate_m = duplicate_coordinate.points[1].coordinate_m;
    assert!(compare_measured_displacements(&duplicate_coordinate).is_err());

    let mut invalid_tolerance = MeasuredValidationStudy::introductory();
    invalid_tolerance.relative_tolerance = 0.0;
    assert!(compare_measured_displacements(&invalid_tolerance).is_err());
}

#[test]
fn measured_displacement_validation_rejects_tampered_digest_bound_inputs() {
    let mut study = MeasuredValidationStudy::introductory();
    study.input_sha256 = Some(measured_validation_input_sha256(&study));
    study.points[2].measured_m += 1.0e-6;

    let error = compare_measured_displacements(&study).expect_err("tampering must fail closed");
    assert!(error.to_string().contains("input_sha256"));
}

#[test]
fn measured_displacement_validation_reports_digest_integrity_state() {
    let mut study = MeasuredValidationStudy::introductory();
    let result = compare_measured_displacements(&study).expect("unbound fixture should compare");
    assert_eq!(result.integrity, MeasuredValidationIntegrity::Unbound);

    study.input_sha256 = Some(measured_validation_input_sha256(&study));
    let result = compare_measured_displacements(&study).expect("bound fixture should compare");
    assert_eq!(
        result.integrity,
        MeasuredValidationIntegrity::Sha256Validated
    );
    assert_eq!(MEASURED_VALIDATION_DIGEST_ALGORITHM, "sha256");
}
