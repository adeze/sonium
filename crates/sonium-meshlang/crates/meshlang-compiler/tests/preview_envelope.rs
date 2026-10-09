use meshlang_compiler::{
    compile_to_geo_full, preview_canonical_ir, CanonicalIrPreviewEnvelope,
    CANONICAL_IR_PREVIEW_SCHEMA_VERSION,
};
use serde_json::Value;

const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

#[test]
fn preview_json_round_trips_through_the_host_envelope() {
    let preview = preview_canonical_ir(SOURCE, "preview-envelope.mlg").unwrap();
    let json = preview.to_json().unwrap();
    let envelope = CanonicalIrPreviewEnvelope::from_json(&json).unwrap();

    assert_eq!(envelope.preview_schema, CANONICAL_IR_PREVIEW_SCHEMA_VERSION);
    assert_eq!(envelope.compiler, "meshlang-compiler");
    assert!(!envelope.compiler_version.is_empty());
    assert_eq!(envelope.diagnostics.len(), preview.diagnostics.len());
    assert!(envelope.sensitivity_manifest.is_some());
    assert!(envelope.provenance_manifest.is_some());
    assert!(envelope.sensitivity_manifest.as_ref().unwrap()["entries"].is_array());
    assert!(envelope.provenance_manifest.as_ref().unwrap()["source_map"].is_object());
    assert_eq!(
        envelope.canonical_ir_hash,
        envelope
            .canonical_ir
            .ir
            .structural_hash()
            .unwrap()
            .to_string()
    );
}

#[test]
fn preview_json_rejects_a_tampered_ir_hash() {
    let preview = preview_canonical_ir(SOURCE, "preview-envelope.mlg").unwrap();
    let mut value: Value = serde_json::from_str(&preview.to_json().unwrap()).unwrap();
    value["canonical_ir_hash"] = Value::String("0".repeat(64));

    let error = CanonicalIrPreviewEnvelope::from_json(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(error.contains("hash does not match embedded IR"), "{error}");
}

#[test]
fn preview_manifests_match_backend_sidecars() {
    let preview = preview_canonical_ir(SOURCE, "preview-envelope.mlg").unwrap();
    let envelope = CanonicalIrPreviewEnvelope::from_json(&preview.to_json().unwrap()).unwrap();
    let backend = compile_to_geo_full(SOURCE, "preview-envelope.mlg").unwrap();

    let preview_sensitivity = envelope.sensitivity_manifest.unwrap();
    let preview_provenance = envelope.provenance_manifest.unwrap();
    let backend_sensitivity: Value =
        serde_json::from_str(&backend.sensitivity_manifest_json).unwrap();
    let backend_provenance: Value =
        serde_json::from_str(&backend.provenance_manifest_json).unwrap();

    assert_eq!(preview_sensitivity, backend_sensitivity);
    assert_eq!(preview_provenance, backend_provenance);
}

#[test]
fn preview_json_rejects_unknown_schema_and_fields() {
    let preview = preview_canonical_ir(SOURCE, "preview-envelope.mlg").unwrap();
    let mut unknown_schema: Value = serde_json::from_str(&preview.to_json().unwrap()).unwrap();
    unknown_schema["preview_schema"] = Value::from(999);
    let error = CanonicalIrPreviewEnvelope::from_json(&unknown_schema.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("unsupported canonical IR preview schema"),
        "{error}"
    );

    let mut unknown_field: Value = serde_json::from_str(&preview.to_json().unwrap()).unwrap();
    unknown_field["future_field"] = Value::Bool(true);
    let error = CanonicalIrPreviewEnvelope::from_json(&unknown_field.to_string())
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown field"), "{error}");

    let mut unknown_document: Value = serde_json::from_str(&preview.to_json().unwrap()).unwrap();
    unknown_document["canonical_ir"]["document_version"] = Value::from(999);
    let error = CanonicalIrPreviewEnvelope::from_json(&unknown_document.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("unsupported canonical IR document version"),
        "{error}"
    );
}
