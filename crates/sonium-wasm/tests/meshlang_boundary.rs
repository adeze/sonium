#![cfg(feature = "meshlang")]

use serde_json::Value;
use sonium_roomsim::{
    MESHLANG_FRONTEND_API_SCHEMA_VERSION, mesh_meshlang_projection_json,
    meshlang_frontend_contract_json, parse_meshlang_cache_stats_json, parse_meshlang_cached_json,
    parse_meshlang_cached_subtree_json, parse_meshlang_json, parse_meshlang_subtree_json,
    preview_meshlang_json, preview_meshlang_with_registry_json,
    preview_meshlang_with_signed_registry_json, preview_meshlang_with_trusted_package_json,
    preview_meshlang_with_trusted_registry_json, validate_meshlang_grammar_json,
};

const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

#[test]
fn product_boundary_exposes_projection_mesh_artifacts() {
    let source = r#"
        geometry model {
            let solid = Sphere(r: 0.5)
            let surface = solid |> project_iso_surface()
        }
    "#;
    let request = r#"{
        "bounds": [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        "cells": [4, 4, 4],
        "tolerance": 0.01
    }"#;
    let artifact: Value = serde_json::from_str(
        &mesh_meshlang_projection_json(source, "projection.mlg", "model.surface", request)
            .expect("projection artifact"),
    )
    .expect("projection artifact JSON");
    assert_eq!(artifact["schema_version"], 1);
    assert_eq!(artifact["mesh"]["cells"], serde_json::json!([4, 4, 4]));
}

#[test]
fn product_contract_declares_stable_pure_frontend_operations() {
    let contract: Value =
        serde_json::from_str(&meshlang_frontend_contract_json().expect("contract JSON"))
            .expect("contract object");
    assert_eq!(contract["api_schema"], MESHLANG_FRONTEND_API_SCHEMA_VERSION);
    assert!(contract["filesystem_free"].as_bool().unwrap_or(false));
    assert!(!contract["production_meshing"].as_bool().unwrap_or(true));
    assert_eq!(contract["operations"].as_array().unwrap().len(), 10);
}

#[test]
fn product_parser_and_preview_preserve_upstream_schema_versions() {
    let parse: Value =
        serde_json::from_str(&parse_meshlang_json(SOURCE, "boundary.mlg").expect("parse JSON"))
            .expect("parse object");
    assert_eq!(parse["parse_schema"], 1);
    assert!(parse["report"]["source_file"].is_object());

    let preview: Value =
        serde_json::from_str(&preview_meshlang_json(SOURCE, "boundary.mlg").expect("IR JSON"))
            .expect("preview object");
    assert_eq!(preview["preview_schema"], 1);
    assert_eq!(preview["canonical_ir"]["document_version"], 1);
    assert!(preview["canonical_ir_hash"].is_string());
}

#[test]
fn product_cached_parser_preserves_envelope_and_reports_hit() {
    let mut cache = meshlang_parser::ParseCache::default();
    let first = parse_meshlang_cached_json(&mut cache, SOURCE, "cached.mlg").expect("first parse");
    let second =
        parse_meshlang_cached_json(&mut cache, SOURCE, "cached.mlg").expect("cache hit parse");
    assert_eq!(first, second);
    let stats: Value =
        serde_json::from_str(&parse_meshlang_cache_stats_json(&cache).expect("cache stats"))
            .expect("cache stats JSON");
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);
}

#[test]
fn product_boundary_subtree_parse_keeps_absolute_spans_and_cache_parity() {
    let source = "param before = 1\nparam selected = 2\n";
    let start = source.find("param selected").unwrap();
    let end = source.len();
    let direct: Value = serde_json::from_str(
        &parse_meshlang_subtree_json(source, "subtree.mlg", start, end).expect("subtree parse"),
    )
    .expect("subtree envelope");
    assert_eq!(direct["report"]["source_file"]["span"]["start"], start);

    let mut cache = meshlang_parser::ParseCache::default();
    let first = parse_meshlang_cached_subtree_json(&mut cache, source, "subtree.mlg", start, end)
        .expect("cached subtree parse");
    assert_eq!(
        first,
        parse_meshlang_cached_subtree_json(&mut cache, source, "subtree.mlg", start, end).unwrap()
    );
    let stats: Value =
        serde_json::from_str(&parse_meshlang_cache_stats_json(&cache).expect("cache stats"))
            .expect("cache stats JSON");
    assert_eq!(stats["subtree_entries"], 1);
    assert!(parse_meshlang_subtree_json(source, "subtree.mlg", 0, source.len() + 1).is_err());
}

#[test]
fn product_boundary_rejects_malformed_trusted_package_envelopes() {
    let policy = r#"{"trusted_keys":[]}"#;
    let result = preview_meshlang_with_trusted_package_json(
        SOURCE,
        "package.mlg",
        r#"{"index":{},"signature":{}}"#,
        policy,
    );
    assert!(result.is_err());
}

#[test]
fn product_boundary_rejects_filesystem_imports() {
    let result = preview_meshlang_json(
        "import \"stdlib/shapes.mlg\"\ngeometry Box(1.0, 1.0, 1.0)",
        "filesystem.mlg",
    );
    assert!(result.is_err());
}

#[test]
fn product_boundary_accepts_explicit_registry_transport() {
    let registry = r#"{"schema_version":1,"modules":[{"path":"pkg/part@1.mlg","source":"pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n"}]}"#;
    let preview = preview_meshlang_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "registry.mlg",
        registry,
    )
    .expect("registry preview");
    let value: Value = serde_json::from_str(&preview).expect("preview object");
    assert!(value["registry_manifest_hash"].is_string());
}

#[test]
fn product_boundary_keeps_authenticated_registry_variants_fail_closed() {
    let unsigned = r#"{"schema_version":1,"modules":[]}"#;
    assert!(
        preview_meshlang_with_signed_registry_json(
            "geometry Box(1.0, 1.0, 1.0)",
            "unsigned.mlg",
            unsigned,
        )
        .is_err()
    );

    let malformed_signature = r#"{"schema_version":1,"modules":[],"signature":{"algorithm":"ed25519","key_id":"unknown","public_key":"0000000000000000000000000000000000000000000000000000000000000000","signature":"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}}"#;
    let empty_policy = r#"{"trusted_keys":[]}"#;
    assert!(
        preview_meshlang_with_trusted_registry_json(
            "geometry Box(1.0, 1.0, 1.0)",
            "untrusted.mlg",
            malformed_signature,
            empty_policy,
        )
        .is_err()
    );
}

#[test]
fn product_boundary_exposes_grammar_contract() {
    let grammar: Value =
        serde_json::from_str(&validate_meshlang_grammar_json().expect("grammar JSON"))
            .expect("grammar object");
    assert_eq!(grammar["schema_version"], 1);
    assert!(grammar["operator_spellings"].as_array().unwrap().len() >= 7);
}
