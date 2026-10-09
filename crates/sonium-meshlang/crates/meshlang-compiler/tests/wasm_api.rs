#![cfg(target_arch = "wasm32")]

use meshlang_compiler::{
    preview_canonical_ir_wasm, preview_canonical_ir_with_registry_wasm,
    preview_canonical_ir_with_signed_registry_wasm,
    preview_canonical_ir_with_trusted_registry_wasm,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node);

#[wasm_bindgen_test]
fn preview_wasm_returns_versioned_canonical_ir() {
    let json = preview_canonical_ir_wasm(
        "geometry model { let body = Box(1.0, 1.0, 1.0) }",
        "wasm-preview.mlg",
    )
    .expect("WASM compiler preview should accept valid source");
    let envelope: serde_json::Value = serde_json::from_str(&json).expect("valid IR JSON");
    assert_eq!(envelope["preview_schema"], 1);
    assert!(envelope["canonical_ir_hash"].is_string());
    assert_eq!(envelope["canonical_ir"]["document_version"], 1);
}

#[wasm_bindgen_test]
fn preview_wasm_rejects_filesystem_imports() {
    let result = preview_canonical_ir_wasm(
        "import \"stdlib/shapes.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "wasm-import.mlg",
    );
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn preview_wasm_accepts_versioned_source_registry() {
    let registry = r#"{"schema_version":1,"modules":[{"path":"pkg/part@1.mlg","source":"pub const PART = 1\n"}]}"#;
    let json = preview_canonical_ir_with_registry_wasm(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "wasm-registry.mlg",
        registry,
    )
    .expect("WASM registry preview should accept registered source");
    let envelope: serde_json::Value = serde_json::from_str(&json).expect("valid IR JSON");
    assert_eq!(envelope["preview_schema"], 1);
    assert!(envelope["canonical_ir_hash"].is_string());
}

#[wasm_bindgen_test]
fn signed_preview_wasm_rejects_unsigned_registry() {
    let registry = r#"{"schema_version":1,"modules":[]}"#;
    let result = preview_canonical_ir_with_signed_registry_wasm(
        "geometry model { Box(1.0, 1.0, 1.0) }",
        "wasm-signed-registry.mlg",
        registry,
    );
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn trusted_preview_wasm_rejects_unpinned_registry_key() {
    let registry = r#"{"schema_version":1,"modules":[],"signature":{"algorithm":"ed25519","key_id":"unknown","public_key":"0000000000000000000000000000000000000000000000000000000000000000","signature":"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}}"#;
    let policy = r#"{"trusted_keys":[]}"#;
    let result = preview_canonical_ir_with_trusted_registry_wasm(
        "geometry model { Box(1.0, 1.0, 1.0) }",
        "wasm-trusted-registry.mlg",
        registry,
        policy,
    );
    assert!(result.is_err());
}
