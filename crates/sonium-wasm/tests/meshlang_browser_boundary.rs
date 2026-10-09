#![cfg(all(feature = "meshlang", target_arch = "wasm32"))]

use serde_json::Value;
use sonium_roomsim::{
    MESHLANG_FRONTEND_API_SCHEMA_VERSION, MeshLangEditor, mesh_meshlang_projection_wasm,
    meshlang_frontend_contract_wasm, parse_meshlang_wasm, preview_meshlang_wasm,
    validate_meshlang_grammar_wasm,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

#[wasm_bindgen_test]
fn browser_contract_and_frontend_operations_return_json() {
    let contract: Value =
        serde_json::from_str(&meshlang_frontend_contract_wasm().expect("contract"))
            .expect("contract JSON");
    assert_eq!(contract["api_schema"], MESHLANG_FRONTEND_API_SCHEMA_VERSION);
    assert!(contract["filesystem_free"].as_bool().unwrap_or(false));
    assert_eq!(contract["operations"].as_array().unwrap().len(), 10);

    let parse: Value =
        serde_json::from_str(&parse_meshlang_wasm(SOURCE, "browser.mlg").expect("parse envelope"))
            .expect("parse JSON");
    assert_eq!(parse["parse_schema"], 1);

    let preview: Value = serde_json::from_str(
        &preview_meshlang_wasm(SOURCE, "browser.mlg").expect("preview envelope"),
    )
    .expect("preview JSON");
    assert_eq!(preview["preview_schema"], 1);
    assert!(preview["canonical_ir_hash"].is_string());

    let grammar: Value =
        serde_json::from_str(&validate_meshlang_grammar_wasm().expect("grammar contract"))
            .expect("grammar JSON");
    assert_eq!(grammar["schema_version"], 1);
}

#[wasm_bindgen_test]
fn browser_editor_cache_reports_hits_and_invalidation() {
    let mut editor = MeshLangEditor::new();
    let first = editor
        .parse(SOURCE, "editor.mlg")
        .expect("first editor parse");
    let second = editor
        .parse(SOURCE, "editor.mlg")
        .expect("cache hit editor parse");
    assert_eq!(first, second);

    let stats: Value =
        serde_json::from_str(&editor.stats().expect("editor stats")).expect("editor stats JSON");
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);
    assert_eq!(stats["entries"], 1);
    assert_eq!(stats["subtree_entries"], 0);
    assert!(editor.invalidate("editor.mlg"));
    assert!(!editor.invalidate("editor.mlg"));
    editor.clear();
}

#[wasm_bindgen_test]
fn browser_editor_subtree_parse_keeps_absolute_span_and_cache_entry() {
    let source = "param before = 1\nparam selected = 2\n";
    let start = source.find("param selected").unwrap();
    let mut editor = MeshLangEditor::new();
    let first = editor
        .parse_subtree(source, "subtree.mlg", start, source.len())
        .expect("subtree parse");
    let second = editor
        .parse_subtree(source, "subtree.mlg", start, source.len())
        .expect("cached subtree parse");
    assert_eq!(first, second);
    let value: Value = serde_json::from_str(&first).expect("subtree JSON");
    assert_eq!(value["report"]["source_file"]["span"]["start"], start);
    let stats: Value =
        serde_json::from_str(&editor.stats().expect("editor stats")).expect("stats JSON");
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);
    assert_eq!(stats["subtree_entries"], 1);
}

#[wasm_bindgen_test]
fn browser_preview_fails_closed_for_filesystem_imports() {
    let result = preview_meshlang_wasm(
        "import \"stdlib/shapes.mlg\"\ngeometry Box(1.0, 1.0, 1.0)",
        "browser-import.mlg",
    );
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn browser_projection_mesh_returns_versioned_artifact() {
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
        &mesh_meshlang_projection_wasm(source, "browser-projection.mlg", "model.surface", request)
            .expect("projection artifact"),
    )
    .expect("projection JSON");
    assert_eq!(artifact["schema_version"], 1);
    assert_eq!(artifact["mesh"]["cells"], serde_json::json!([4, 4, 4]));
}
