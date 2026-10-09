use serde_json::Value;
use sonium_qa::{
    QA_RESULT_SCHEMA_ID, QA_RESULT_SCHEMA_VERSION, QaContextReuse, QaExecutionCounts,
    QaHardwareProvenance, QaMemoryEvidence, QaResult, QaRuntimeCounters, QaRuntimeEvidence,
    is_supported_schema_version,
};

const SCHEMA: &str = include_str!("../schema/qa-result-v1.schema.json");

#[test]
fn frozen_schema_artifact_matches_the_rust_contract() {
    let schema: Value = serde_json::from_str(SCHEMA).expect("QA schema must be valid JSON");
    assert_eq!(schema["$id"], QA_RESULT_SCHEMA_ID);
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        QA_RESULT_SCHEMA_VERSION
    );
    assert_eq!(
        schema["properties"]["status"]["enum"],
        serde_json::json!(["passed", "failed", "skipped"])
    );

    let required = schema["required"].as_array().expect("required array");
    for field in [
        "schema_version",
        "case_id",
        "status",
        "backend_outcomes",
        "metrics",
        "thresholds",
        "threshold_directions",
        "elapsed_ms",
        "git_sha",
        "environment",
        "artifacts",
        "evidence",
        "runtime",
    ] {
        assert!(
            required.iter().any(|value| value == field),
            "missing {field}"
        );
    }
    for field in ["topology", "peak_flops", "memory_bandwidth"] {
        assert!(
            schema["$defs"]["hardware"]["properties"]
                .get(field)
                .is_some(),
            "hardware schema omits {field}"
        );
    }
}

#[test]
fn canonical_result_serialization_contains_every_frozen_required_field() {
    let result = QaResult::passed("qa.schema.v1")
        .metric("relative_error", 1.0e-8)
        .maximum("relative_error", 1.0e-6)
        .backend_passed("cpu")
        .artifact_with_checksum("result.json", "sha256:test");
    let value: Value = serde_json::from_str(&result.to_json_line().unwrap()).unwrap();

    for field in [
        "schema_version",
        "case_id",
        "status",
        "backend_outcomes",
        "metrics",
        "thresholds",
        "threshold_directions",
        "elapsed_ms",
        "git_sha",
        "environment",
        "artifacts",
        "evidence",
        "runtime",
    ] {
        assert!(
            value.get(field).is_some(),
            "serialized result lacks {field}"
        );
    }
}

#[test]
fn strict_release_policy_requires_native_runtime_and_artifact_provenance() {
    let result = QaResult::passed("qa.cuda.v1")
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .minimum("error", 0.0)
        .metric("error", 0.0)
        .artifact_with_provenance(
            "report.json",
            "qa-test",
            "deterministic report",
            "v1",
            "sha256:test",
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Reused,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: 1,
                h2d_transfers: 1,
                d2h_transfers: 1,
                launches: 1,
                ..QaRuntimeCounters::default()
            },
            memory: QaMemoryEvidence {
                peak_bytes: 8,
                live_bytes: 4,
            },
            hardware: QaHardwareProvenance {
                device: Some("NVIDIA test device".into()),
                driver: Some("driver".into()),
                toolkit: Some("toolkit".into()),
                ..QaHardwareProvenance::default()
            },
            ..QaRuntimeEvidence::default()
        });
    assert!(
        result
            .release_eligible_for(Some("cuda"), true, true, true, true)
            .is_ok()
    );

    let mut stale = result.clone();
    stale
        .evidence
        .artifact_checksums
        .insert("report.json".into(), "sha256:stale".into());
    assert!(stale.validate().is_err());

    let mut missing_checksum = result.clone();
    missing_checksum.evidence.artifact_checksums.clear();
    assert!(missing_checksum.validate().is_err());

    let mut placeholder = result.clone();
    placeholder.runtime.hardware.device = Some("unknown".into());
    assert!(
        placeholder
            .release_eligible_for(Some("cuda"), true, true, true, true)
            .is_err()
    );

    let mut fallback = result;
    fallback.actual_backend = Some("cpu".into());
    assert!(
        fallback
            .release_eligible_for(Some("cuda"), true, true, true, true)
            .is_err()
    );
}

#[test]
fn unsupported_schema_versions_are_rejected_before_release_evaluation() {
    assert!(is_supported_schema_version(QA_RESULT_SCHEMA_VERSION));
    assert!(!is_supported_schema_version(QA_RESULT_SCHEMA_VERSION + 1));

    let mut result = QaResult::passed("qa.schema.v1");
    result.schema_version = QA_RESULT_SCHEMA_VERSION + 1;
    assert!(result.validate().is_err());
    assert!(result.release_eligible().is_err());
}
