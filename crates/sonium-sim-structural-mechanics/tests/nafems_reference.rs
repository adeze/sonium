use sonium_sim_structural_mechanics::{
    NafemsBenchmarkKind, NafemsBenchmarkSuite, PublishedReferenceAuthority, PublishedReferenceCase,
    PublishedReferenceManifest, ValidationReferenceKind,
    compare_nafems_suite_to_published_manifest, run_nafems_benchmark_suite,
};

fn surrogate_manifest(suite: &NafemsBenchmarkSuite) -> PublishedReferenceManifest {
    PublishedReferenceManifest {
        schema_version: 1,
        manifest_id: "public-analytical-surrogate-fixture".to_string(),
        title: "Public analytical surrogate fixture".to_string(),
        authority: PublishedReferenceAuthority::UserSupplied,
        citation: "Test fixture; not official NAFEMS data".to_string(),
        source_uri: Some("https://example.invalid/structural-reference.json".to_string()),
        source_revision: "fixture-1".to_string(),
        cases: suite
            .cases
            .iter()
            .map(|case| PublishedReferenceCase {
                id: case.id.clone(),
                kind: case.kind,
                quantity: "native validation quantity".to_string(),
                units: "SI".to_string(),
                reference_value: case.computed_value,
                tolerance: 1.0e-12,
                applicability: "matches the native analytical surrogate fixture".to_string(),
            })
            .collect(),
        applicability_limits: vec!["Synthetic test fixture only".to_string()],
        manifest_sha256: None,
    }
}

fn close_after_json_round_trip(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12 * (1.0 + left.abs().max(right.abs()))
}

#[test]
fn public_nafems_axis_catalogue_is_complete_and_explicitly_surrogate() {
    let suite = run_nafems_benchmark_suite().expect("analytical validation catalogue should run");

    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.cases.len(), 4);
    assert!(suite.all_passed);
    assert!(!suite.official_reference_data_loaded);
    assert!(!suite.official_parity_passed());
    assert!(
        suite
            .cases
            .iter()
            .all(|case| case.reference_kind == ValidationReferenceKind::AnalyticalSurrogate)
    );
    assert!(
        suite
            .applicability_limits
            .iter()
            .any(|limit| limit.contains("not official NAFEMS"))
    );

    let kinds = suite.cases.iter().map(|case| case.kind).collect::<Vec<_>>();
    assert_eq!(
        kinds,
        vec![
            NafemsBenchmarkKind::Static,
            NafemsBenchmarkKind::Modal,
            NafemsBenchmarkKind::ThermalStress,
            NafemsBenchmarkKind::Nonlinear,
        ]
    );

    let encoded = serde_json::to_string(&suite).expect("catalogue should serialize");
    let decoded: NafemsBenchmarkSuite =
        serde_json::from_str(&encoded).expect("catalogue should deserialize");
    assert_eq!(decoded.schema_version, suite.schema_version);
    assert_eq!(decoded.execution, suite.execution);
    assert_eq!(
        decoded.official_reference_data_loaded,
        suite.official_reference_data_loaded
    );
    assert_eq!(decoded.applicability_limits, suite.applicability_limits);
    assert_eq!(decoded.cases.len(), suite.cases.len());
    for (decoded, original) in decoded.cases.iter().zip(&suite.cases) {
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.kind, original.kind);
        assert_eq!(decoded.reference_kind, original.reference_kind);
        assert_eq!(decoded.reference_source, original.reference_source);
        assert!(close_after_json_round_trip(
            decoded.computed_value,
            original.computed_value
        ));
        assert!(close_after_json_round_trip(
            decoded.reference_value,
            original.reference_value
        ));
        assert!(close_after_json_round_trip(
            decoded.relative_error,
            original.relative_error
        ));
        assert_eq!(decoded.tolerance, original.tolerance);
        assert_eq!(
            decoded.solver_residual.is_some(),
            original.solver_residual.is_some()
        );
        assert_eq!(
            decoded.energy_error.is_some(),
            original.energy_error.is_some()
        );
        assert_eq!(decoded.passed, original.passed);
        assert_eq!(decoded.applicability, original.applicability);
    }
}

#[test]
fn public_nafems_reference_comparison_fails_closed_on_incomplete_coverage() {
    let suite = run_nafems_benchmark_suite().expect("analytical validation catalogue should run");
    let manifest = surrogate_manifest(&suite);
    let encoded = serde_json::to_value(&manifest).expect("manifest should serialize");
    let decoded: PublishedReferenceManifest =
        serde_json::from_value(encoded).expect("manifest should deserialize");
    let report = compare_nafems_suite_to_published_manifest(&suite, &decoded)
        .expect("synthetic manifest should compare");
    assert!(report.all_passed);
    assert!(report.reference_manifest_validated);
    assert!(!report.provenance_verified);
    assert!(!report.parity_passed());

    let mut incomplete = decoded;
    incomplete.cases.pop();
    let report = compare_nafems_suite_to_published_manifest(&suite, &incomplete)
        .expect("incomplete manifest should produce a failed report");
    assert!(!report.all_passed);
    assert_eq!(report.unmatched_suite_case_ids.len(), 1);
    assert!(!report.parity_passed());

    let mut spoofed = surrogate_manifest(&suite);
    spoofed.cases[0].id = "official-nafems-static".to_string();
    let report = compare_nafems_suite_to_published_manifest(&suite, &spoofed)
        .expect("mismatched manifest should produce a failed report");
    assert!(!report.all_passed);
    assert!(!report.cases[0].kind_matches);
    assert!(report.cases[0].computed_value.is_none());
    assert!(!report.parity_passed());
}
