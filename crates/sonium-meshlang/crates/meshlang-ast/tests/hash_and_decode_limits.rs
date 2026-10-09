//! AST identity and decode-admission tests: structural-hash stability across
//! serialization, current-version round-trip identity, byte-limit admission,
//! and bounded decoding of deeply nested payloads.
//!
//! Serialized version migration and public AST construction are distinct
//! admission routes: byte limits on decoding do not automatically bound
//! recursively constructed trees, so both are exercised here.

use meshlang_ast::span::Span;
use meshlang_ast::{
    CURRENT_AST_SCHEMA_VERSION, SchemaError, SourceFile, StructuralHash, VersionedAst,
};

fn source_file() -> SourceFile {
    SourceFile::new(Vec::new(), Span::DUMMY)
}

#[test]
fn structural_hash_is_stable_across_json_round_trip() {
    let document = VersionedAst::current(source_file());
    let before = StructuralHash::of(&document).expect("hash before serialization");

    let bytes = document.to_json_vec().expect("serialize AST");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON value");
    // Re-hash through a fresh serialize/deserialize cycle, as a second
    // target or consumer would observe it.
    let round_tripped: VersionedAst = serde_json::from_value(value).expect("deserialize AST value");
    let after = StructuralHash::of(&round_tripped).expect("hash after round trip");

    assert_eq!(before, after);
    assert_eq!(before.to_hex(), after.to_hex());
}

#[test]
fn current_version_decode_is_identity_without_migration() {
    let document = VersionedAst::current(source_file());
    let bytes = document.to_json_vec().expect("serialize AST");

    let (decoded, report) = VersionedAst::from_json_slice(&bytes).expect("decode current AST");
    assert!(!report.was_migrated());
    assert_eq!(decoded.schema_version, CURRENT_AST_SCHEMA_VERSION);
    assert_eq!(
        serde_json::to_value(&decoded.source_file).expect("decoded source JSON"),
        serde_json::to_value(source_file()).expect("expected source JSON")
    );
}

#[test]
fn byte_limit_admits_exact_size_and_rejects_one_byte_more() {
    let document = VersionedAst::current(source_file());
    let bytes = document.to_json_vec().expect("serialize AST");

    let (decoded, _) = VersionedAst::from_json_slice_with_limit(&bytes, bytes.len())
        .expect("exact-size payload must decode");
    assert_eq!(
        serde_json::to_value(&decoded.source_file).expect("decoded source JSON"),
        serde_json::to_value(source_file()).expect("expected source JSON")
    );

    match VersionedAst::from_json_slice_with_limit(&bytes, bytes.len().saturating_sub(1)) {
        Err(SchemaError::ResourceLimit { actual, maximum }) => {
            assert_eq!(actual, bytes.len());
            assert_eq!(maximum, bytes.len().saturating_sub(1));
        }
        other => panic!("expected ResourceLimit, got: {other:?}"),
    }
}

#[test]
fn deeply_nested_payload_fails_bounded_without_panic() {
    // A recursively constructed tree far beyond any legitimate AST depth
    // must produce a bounded error, never a stack overflow or hang.
    const DEPTH: usize = 10_000;
    let mut nested = String::from("{\"schema_version\":1,\"source_file\":");
    nested.push_str(&"[".repeat(DEPTH));
    nested.push_str(&"]".repeat(DEPTH));
    nested.push('}');

    let result = VersionedAst::from_json_slice_with_limit(nested.as_bytes(), 8 * 1024 * 1024);
    assert!(
        result.is_err(),
        "10k-deep payload must not decode successfully"
    );
}
