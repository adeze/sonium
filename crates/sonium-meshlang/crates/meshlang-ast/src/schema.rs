//! Explicit serialized-AST versioning and migrations.

use crate::SourceFile;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;

pub const LEGACY_AST_SCHEMA_VERSION: AstSchemaVersion = AstSchemaVersion(1);
pub const CURRENT_AST_SCHEMA_VERSION: AstSchemaVersion = AstSchemaVersion(2);
pub const DEFAULT_MAX_SERIALIZED_AST_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct AstSchemaVersion(pub u32);

/// The only supported persistence boundary for a MeshLang AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedAst {
    pub schema_version: AstSchemaVersion,
    pub source_file: SourceFile,
}

impl VersionedAst {
    pub fn current(source_file: SourceFile) -> Self {
        Self {
            schema_version: CURRENT_AST_SCHEMA_VERSION,
            source_file,
        }
    }

    pub fn to_json_vec(&self) -> Result<Vec<u8>, SchemaError> {
        if self.schema_version != CURRENT_AST_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion(self.schema_version));
        }
        serde_json::to_vec(self).map_err(SchemaError::Json)
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<(Self, MigrationReport), SchemaError> {
        Self::from_json_slice_with_limit(bytes, DEFAULT_MAX_SERIALIZED_AST_BYTES)
    }

    pub fn from_json_slice_with_limit(
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<(Self, MigrationReport), SchemaError> {
        if bytes.len() > max_bytes {
            return Err(SchemaError::ResourceLimit {
                actual: bytes.len(),
                maximum: max_bytes,
            });
        }
        let value: Value = serde_json::from_slice(bytes).map_err(SchemaError::Json)?;
        let (migrated, report) = migrate_value(value)?;
        let document = serde_json::from_value(migrated).map_err(SchemaError::Json)?;
        Ok((document, report))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MigrationReport {
    pub steps: Vec<MigrationStep>,
}

impl MigrationReport {
    pub fn was_migrated(&self) -> bool {
        !self.steps.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationStep {
    pub from: AstSchemaVersion,
    pub to: AstSchemaVersion,
    pub description: &'static str,
    /// Whether the step invents data the source document did not carry
    /// (v1→v2 stamps `Span::DUMMY` on migrated `Literal`/`Paren` payloads).
    pub was_lossy: bool,
}

/// Migrate a serialized AST value to the current explicit envelope.
///
/// Schema v1 was the historical raw `SourceFile` representation. A v1
/// envelope is also accepted to support clients that began labelling stored
/// documents before the migration API existed. No future version is guessed.
pub fn migrate_value(value: Value) -> Result<(Value, MigrationReport), SchemaError> {
    let (version, source_file) = match value {
        Value::Object(mut object) => match object.remove("schema_version") {
            Some(version) => {
                let version = serde_json::from_value::<AstSchemaVersion>(version)
                    .map_err(SchemaError::Json)?;
                let source_file = object
                    .remove("source_file")
                    .ok_or(SchemaError::MissingSourceFile)?;
                if !object.is_empty() {
                    return Err(SchemaError::InvalidEnvelope);
                }
                (version, source_file)
            }
            None => (LEGACY_AST_SCHEMA_VERSION, Value::Object(object)),
        },
        _ => return Err(SchemaError::InvalidEnvelope),
    };

    match version {
        CURRENT_AST_SCHEMA_VERSION => Ok((
            envelope(CURRENT_AST_SCHEMA_VERSION, source_file),
            MigrationReport::default(),
        )),
        LEGACY_AST_SCHEMA_VERSION => Ok((
            envelope(
                CURRENT_AST_SCHEMA_VERSION,
                migrate_v1_source_file(source_file),
            ),
            MigrationReport {
                steps: vec![MigrationStep {
                    from: LEGACY_AST_SCHEMA_VERSION,
                    to: CURRENT_AST_SCHEMA_VERSION,
                    description: "wrap the raw SourceFile in the explicit versioned AST envelope; \
                        migrated Literal/Paren payloads gain Span::DUMMY provenance (lossy)",
                    was_lossy: true,
                }],
            },
        )),
        other => Err(SchemaError::UnsupportedVersion(other)),
    }
}

fn migrate_v1_source_file(mut value: Value) -> Value {
    migrate_v1_expr_payloads(&mut value);
    value
}

fn migrate_v1_expr_payloads(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                migrate_v1_expr_payloads(value);
            }
        }
        Value::Object(object) => {
            for value in object.values_mut() {
                migrate_v1_expr_payloads(value);
            }

            if let Some(literal) = object.get_mut("Literal")
                && !literal.as_object().is_some_and(|payload| {
                    payload.contains_key("value") && payload.contains_key("span")
                })
            {
                let value = literal.take();
                *literal = serde_json::json!({
                    "value": value,
                    "span": crate::Span::DUMMY,
                });
            }

            if let Some(parenthesized) = object.get_mut("Paren")
                && !parenthesized.as_object().is_some_and(|payload| {
                    payload.contains_key("value") && payload.contains_key("span")
                })
            {
                let value = parenthesized.take();
                *parenthesized = serde_json::json!({
                    "value": value,
                    "span": crate::Span::DUMMY,
                });
            }
        }
        _ => {}
    }
}

fn envelope(version: AstSchemaVersion, source_file: Value) -> Value {
    let mut object = Map::new();
    object.insert(
        "schema_version".into(),
        Value::Number(serde_json::Number::from(version.0)),
    );
    object.insert("source_file".into(), source_file);
    Value::Object(object)
}

#[derive(Debug)]
pub enum SchemaError {
    Json(serde_json::Error),
    InvalidEnvelope,
    MissingSourceFile,
    UnsupportedVersion(AstSchemaVersion),
    ResourceLimit { actual: usize, maximum: usize },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid serialized AST: {error}"),
            Self::InvalidEnvelope => f.write_str("serialized AST must be a JSON object envelope"),
            Self::MissingSourceFile => f.write_str("serialized AST envelope has no `source_file`"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported AST schema version {}", version.0)
            }
            Self::ResourceLimit { actual, maximum } => write!(
                f,
                "serialized AST is {actual} bytes, exceeding the {maximum}-byte limit"
            ),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{
        AstNode, Comment, CommentKind, ProvenanceOperation, ProvenanceRecord, SourceMetadata,
        SourceReference,
    };
    use crate::span::Span;

    fn source_file() -> SourceFile {
        SourceFile::new(Vec::new(), Span::with_file(0, 12, 7))
    }

    #[test]
    fn current_schema_round_trips_without_migration() {
        let document = VersionedAst::current(source_file());
        let encoded = document.to_json_vec().unwrap();
        let (decoded, report) = VersionedAst::from_json_slice(&encoded).unwrap();
        assert_eq!(decoded.schema_version, CURRENT_AST_SCHEMA_VERSION);
        assert_eq!(decoded.source_file.span, Span::with_file(0, 12, 7));
        assert!(!report.was_migrated());
    }

    #[test]
    fn raw_v1_source_file_migrates_explicitly() {
        let legacy = serde_json::to_vec(&serde_json::json!({
            "declarations": [],
            "span": Span::with_file(0, 12, 7),
        }))
        .unwrap();
        let (decoded, report) = VersionedAst::from_json_slice(&legacy).unwrap();
        assert_eq!(decoded.schema_version, CURRENT_AST_SCHEMA_VERSION);
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.steps[0].from, LEGACY_AST_SCHEMA_VERSION);
        assert!(decoded.source_file.comments.is_empty());
        assert!(decoded.source_file.provenance.records.is_empty());
    }

    #[test]
    fn labelled_v1_envelope_migrates_without_guessing_future_versions() {
        let legacy = serde_json::json!({
            "schema_version": LEGACY_AST_SCHEMA_VERSION.0,
            "source_file": {
                "declarations": [],
                "span": Span::with_file(0, 4, 12)
            }
        });
        let (decoded, report) = VersionedAst::from_json_slice(
            &serde_json::to_vec(&legacy).expect("legacy envelope is serializable"),
        )
        .expect("labelled v1 envelope must migrate");
        assert_eq!(decoded.schema_version, CURRENT_AST_SCHEMA_VERSION);
        assert_eq!(decoded.source_file.span.file_id, 12);
        assert_eq!(
            report.steps.as_slice(),
            &[MigrationStep {
                from: LEGACY_AST_SCHEMA_VERSION,
                to: CURRENT_AST_SCHEMA_VERSION,
                description: "wrap the raw SourceFile in the explicit versioned AST envelope; \
                    migrated Literal/Paren payloads gain Span::DUMMY provenance (lossy)",
                was_lossy: true,
            }]
        );
    }

    #[test]
    fn v1_literal_and_parenthesis_payloads_gain_explicit_spans() {
        let span = Span::with_file(0, 9, 3);
        let legacy = serde_json::json!({
            "declarations": [{
                "Param": {
                    "name": { "name": "x", "span": span },
                    "type_ann": null,
                    "value": { "Paren": { "Literal": { "Int": 7 } } },
                    "unit": null,
                    "span": span
                }
            }],
            "span": span
        });
        let (decoded, report) =
            VersionedAst::from_json_slice(&serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert!(report.was_migrated());
        let crate::Declaration::Param(param) = &decoded.source_file.declarations[0] else {
            panic!("expected migrated parameter")
        };
        let crate::Expr::Paren(parenthesized) = &param.value else {
            panic!("expected migrated parenthesized expression")
        };
        assert_eq!(parenthesized.span, Span::DUMMY);
        let crate::Expr::Literal(crate::LiteralExpr { value, span }) = &*parenthesized.value else {
            panic!("expected migrated literal")
        };
        assert_eq!(*span, Span::DUMMY);
        assert!(matches!(value, crate::Literal::Int(7)));
    }

    #[test]
    fn future_schema_versions_are_rejected() {
        let value = serde_json::json!({
            "schema_version": CURRENT_AST_SCHEMA_VERSION.0 + 1,
            "source_file": source_file(),
        });
        let error =
            VersionedAst::from_json_slice(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(matches!(error, SchemaError::UnsupportedVersion(_)));
    }

    #[test]
    fn current_round_trip_preserves_comments_docstrings_ids_and_provenance() {
        let metadata = SourceMetadata::for_source("design.mlg", "/// domain\ngeometry domain {}\n");
        let mut file =
            SourceFile::new(Vec::new(), Span::with_file(0, 31, 9)).with_source_metadata(metadata);
        let node = file.node_id(&file.metadata.identity);
        file.comments.push(
            Comment::new(
                CommentKind::DocLine,
                "/// domain",
                Span::with_file(0, 10, 9),
            )
            .documenting(node),
        );
        file.comments.push(Comment::new(
            CommentKind::Line,
            "// trailing",
            Span::with_file(20, 31, 9),
        ));
        file.provenance.push(ProvenanceRecord {
            node,
            sources: vec![SourceReference {
                source: file.metadata.identity.id,
                node,
                span: file.span,
            }],
            operation: ProvenanceOperation::Parsed,
        });

        let before_hash = file.structural_hash().unwrap();
        let encoded = VersionedAst::current(file).to_json_vec().unwrap();
        let (decoded, report) = VersionedAst::from_json_slice(&encoded).unwrap();
        assert!(!report.was_migrated());
        assert_eq!(decoded.source_file.structural_hash().unwrap(), before_hash);
        assert_eq!(decoded.source_file.comments.len(), 2);
        assert!(decoded.source_file.comments[0].is_doc());
        assert_eq!(decoded.source_file.comments[0].target, Some(node));
        assert_eq!(decoded.source_file.provenance.records[0].node, node);
        assert_eq!(decoded.source_file.span.file_id, 9);
        assert!(decoded.source_file.metadata.identity.content_hash.is_some());
    }

    #[test]
    fn deserialization_is_bounded_and_total_for_arbitrary_bytes() {
        assert!(matches!(
            VersionedAst::from_json_slice_with_limit(&[b' '; 17], 16),
            Err(SchemaError::ResourceLimit { .. })
        ));

        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for length in 0..512 {
            let mut bytes = Vec::with_capacity(length);
            for _ in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bytes.push(state as u8);
            }
            let result = std::panic::catch_unwind(|| VersionedAst::from_json_slice(&bytes));
            assert!(
                result.is_ok(),
                "deserializer panicked for {length} arbitrary bytes"
            );
        }
    }
}
