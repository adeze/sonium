use meshlang_ast::span::Span;
use meshlang_ast::{
    CURRENT_AST_SCHEMA_VERSION, CURRENT_IR_DOCUMENT_VERSION, CURRENT_IR_SCHEMA_VERSION,
    CanonicalGeometryIr, IrDocumentVersion, IrPersistenceError, LEGACY_AST_SCHEMA_VERSION,
    SchemaError, SourceFile, SourceMetadata, VersionedAst, VersionedCanonicalIr,
};
use serde_json::json;

fn source_file() -> SourceFile {
    SourceFile::new(Vec::new(), Span::DUMMY)
}

#[test]
fn ast_version_matrix_accepts_current_and_legacy_rejects_future() {
    let source = source_file();
    let source_value = serde_json::to_value(&source).expect("source JSON");
    let cases = [
        (
            "current",
            serde_json::to_vec(&VersionedAst::current(source.clone())).expect("current JSON"),
            false,
        ),
        (
            "raw-legacy",
            serde_json::to_vec(&source_value).expect("raw legacy JSON"),
            true,
        ),
        (
            "labelled-legacy",
            serde_json::to_vec(&json!({
                "schema_version": LEGACY_AST_SCHEMA_VERSION.0,
                "source_file": source_value,
            }))
            .expect("labelled legacy JSON"),
            true,
        ),
    ];

    for (label, bytes, migrated) in cases {
        let (decoded, report) = VersionedAst::from_json_slice(&bytes)
            .unwrap_or_else(|error| panic!("{label} AST case failed: {error}"));
        assert_eq!(
            decoded.schema_version, CURRENT_AST_SCHEMA_VERSION,
            "{label}"
        );
        assert_eq!(report.was_migrated(), migrated, "{label}");
    }

    let future = json!({
        "schema_version": CURRENT_AST_SCHEMA_VERSION.0 + 1,
        "source_file": source,
    });
    assert!(matches!(
        VersionedAst::from_json_slice(&serde_json::to_vec(&future).expect("future AST JSON")),
        Err(SchemaError::UnsupportedVersion(version)) if version.0 == CURRENT_AST_SCHEMA_VERSION.0 + 1
    ));
}

#[test]
fn ir_document_matrix_accepts_current_and_raw_legacy_rejects_future() {
    let ir = CanonicalGeometryIr::new(SourceMetadata::for_source("matrix.mlg", ""));
    assert_eq!(ir.schema_version, CURRENT_IR_SCHEMA_VERSION);
    let current = VersionedCanonicalIr::current(ir.clone());
    let raw = serde_json::to_vec(&ir).expect("raw IR JSON");
    let current_bytes = current.to_json_vec().expect("current IR JSON");

    for (label, bytes, migrated) in [("current", current_bytes, false), ("raw-legacy", raw, true)] {
        let (decoded, report) = VersionedCanonicalIr::from_json_slice(&bytes)
            .unwrap_or_else(|error| panic!("{label} IR case failed: {error}"));
        assert_eq!(
            decoded.document_version, CURRENT_IR_DOCUMENT_VERSION,
            "{label}"
        );
        assert_eq!(report.was_migrated(), migrated, "{label}");
        assert_eq!(
            decoded.ir.structural_hash().expect("decoded IR hash"),
            ir.structural_hash().expect("source IR hash"),
            "{label}"
        );
    }

    let future = json!({
        "document_version": CURRENT_IR_DOCUMENT_VERSION.0 + 1,
        "ir": ir,
    });
    assert!(matches!(
        VersionedCanonicalIr::from_json_slice(&serde_json::to_vec(&future).expect("future IR JSON")),
        Err(IrPersistenceError::UnsupportedDocumentVersion(version))
            if version == IrDocumentVersion(CURRENT_IR_DOCUMENT_VERSION.0 + 1)
    ));
}
#[test]
fn legacy_ir_without_source_identities_recovers_root_identity() {
    let ir = CanonicalGeometryIr::new(SourceMetadata::for_source(
        "legacy-root.mlg",
        "geometry Root {}",
    ));
    let mut value =
        serde_json::to_value(VersionedCanonicalIr::current(ir.clone())).expect("current IR JSON");
    value["ir"]
        .as_object_mut()
        .expect("IR envelope")
        .remove("source_identities");

    let (decoded, report) = VersionedCanonicalIr::from_json_slice(
        &serde_json::to_vec(&value).expect("legacy IR bytes"),
    )
    .expect("legacy IR with omitted identity list");
    assert!(!report.was_migrated());
    assert_eq!(decoded.ir.source_identities, vec![ir.source.identity]);
}
