#![cfg(target_arch = "wasm32")]

use meshlang_parser::{
    GRAMMAR_CONTRACT_SCHEMA_VERSION, PARSE_JSON_SCHEMA_VERSION, parse_wasm, validate_grammar_wasm,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node);

#[wasm_bindgen_test]
fn parse_wasm_returns_the_versioned_editor_envelope() {
    let json = parse_wasm(
        "geometry model { let body = Box(1.0, 1.0, 1.0) }",
        "wasm-contract.mlg",
    )
    .expect("WASM parser API should accept valid source");
    let envelope: serde_json::Value = serde_json::from_str(&json).expect("valid JSON envelope");
    assert_eq!(envelope["parse_schema"], PARSE_JSON_SCHEMA_VERSION);
    assert!(envelope["grammar_version"].is_string());
    assert!(envelope["diagnostic_schema"].is_number());
    assert_eq!(
        envelope["report"]["source_file"]["metadata"]["identity"]["display_name"],
        "wasm-contract.mlg"
    );
}

#[wasm_bindgen_test]
fn parse_wasm_turns_invalid_source_into_a_js_error() {
    assert!(parse_wasm("param x = 1e999", "wasm-invalid.mlg").is_err());
}

#[wasm_bindgen_test]
fn grammar_wasm_returns_validated_contract() {
    let json = validate_grammar_wasm().expect("WASM grammar contract should validate");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid grammar JSON");
    assert_eq!(value["schema_version"], GRAMMAR_CONTRACT_SCHEMA_VERSION);
    assert_eq!(value["precedence_rows"].as_array().unwrap().len(), 11);
    assert_eq!(value["operator_spellings"].as_array().unwrap().len(), 7);
}
