//! Dependency-light schema and cross-language tooling artifacts.
//!
//! The generated bindings describe the stable persistence envelope and use
//! JSON-compatible payloads for individual variant bodies. Tooling clients can
//! therefore inspect, migrate, and route every AST variant without depending
//! on Rust layout details.

use crate::{CURRENT_AST_SCHEMA_VERSION, CURRENT_IR_SCHEMA_VERSION};
use serde_json::{Value, json};
use std::fmt::Write;

pub const DECLARATION_VARIANTS: &[&str] = crate::Declaration::VARIANT_NAMES;
pub const STATEMENT_VARIANTS: &[&str] = crate::Statement::VARIANT_NAMES;
pub const MESH_STATEMENT_VARIANTS: &[&str] = crate::MeshStatement::VARIANT_NAMES;
pub const EXPR_VARIANTS: &[&str] = crate::Expr::VARIANT_NAMES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolingArtifacts {
    pub json_schema: String,
    pub typescript: String,
    pub python: String,
}

/// Error returned when generated cross-language artifacts drift from the
/// versioned AST/IR contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolingValidationError(pub String);

impl std::fmt::Display for ToolingValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ToolingValidationError {}

/// Build every cross-language artifact. Infallible by construction:
/// `serde_json::Value` serialization cannot fail, so the `expect` below is
/// unreachable; the signature stays non-`Result` so codegen tooling keeps a
/// total function over the static schema.
pub fn generate_tooling_artifacts() -> ToolingArtifacts {
    ToolingArtifacts {
        json_schema: serde_json::to_string_pretty(&json_schema())
            .expect("MeshLang's static JSON schema is serializable"),
        typescript: typescript_bindings(),
        python: python_bindings(),
    }
}

/// Validate generated schema and language bindings against the current
/// versioned variant inventories.
pub fn validate_tooling_artifacts(
    artifacts: &ToolingArtifacts,
) -> Result<(), ToolingValidationError> {
    let schema: Value = serde_json::from_str(&artifacts.json_schema).map_err(|error| {
        ToolingValidationError(format!("generated JSON schema is invalid JSON: {error}"))
    })?;
    if schema["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        || schema["$id"] != "https://sonium.io/schema/meshlang/ast-v2.json"
    {
        return Err(ToolingValidationError(
            "generated JSON schema has an unexpected identity".into(),
        ));
    }
    if schema["properties"]["schema_version"]["const"] != CURRENT_AST_SCHEMA_VERSION.0 {
        return Err(ToolingValidationError(
            "generated AST schema version does not match the Rust contract".into(),
        ));
    }

    let variants = DECLARATION_VARIANTS
        .iter()
        .chain(STATEMENT_VARIANTS)
        .chain(MESH_STATEMENT_VARIANTS)
        .chain(EXPR_VARIANTS);
    for variant in variants {
        if !artifacts.json_schema.contains(variant)
            || !artifacts.typescript.contains(variant)
            || !artifacts.python.contains(variant)
        {
            return Err(ToolingValidationError(format!(
                "generated tooling artifacts omit AST variant '{variant}'"
            )));
        }
    }
    let ast_version = CURRENT_AST_SCHEMA_VERSION.0.to_string();
    let ir_version = CURRENT_IR_SCHEMA_VERSION.0.to_string();
    if !artifacts
        .typescript
        .contains(&format!("schema_version: {ast_version}"))
        || !artifacts
            .typescript
            .contains(&format!("schema_version: {ir_version}"))
        || !artifacts
            .python
            .contains(&format!("AST_SCHEMA_VERSION: Literal[{ast_version}]"))
        || !artifacts
            .python
            .contains(&format!("IR_SCHEMA_VERSION: Literal[{ir_version}]"))
    {
        return Err(ToolingValidationError(
            "generated language bindings omit current AST/IR versions".into(),
        ));
    }
    Ok(())
}

/// Draft 2020-12 JSON Schema for the versioned persistence boundary.
pub fn json_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://sonium.io/schema/meshlang/ast-v2.json",
        "title": "MeshLang Versioned AST",
        "type": "object",
        "additionalProperties": false,
        "required": ["schema_version", "source_file"],
        "properties": {
            "schema_version": { "const": CURRENT_AST_SCHEMA_VERSION.0 },
            "source_file": { "$ref": "#/$defs/SourceFile" }
        },
        "$defs": {
            "Span": {
                "type": "object",
                "additionalProperties": false,
                "required": ["start", "end", "file_id"],
                "properties": {
                    "start": { "type": "integer", "minimum": 0 },
                    "end": { "type": "integer", "minimum": 0 },
                    "file_id": { "type": "integer", "minimum": 0, "maximum": 4294967295_u64 }
                }
            },
            "SourceFile": {
                "type": "object",
                "additionalProperties": false,
                "required": ["declarations", "metadata", "comments", "provenance", "span"],
                "properties": {
                    "declarations": {
                        "type": "array",
                        "items": variant_schema(DECLARATION_VARIANTS)
                    },
                    "metadata": { "type": "object" },
                    "comments": { "type": "array", "items": { "$ref": "#/$defs/Comment" } },
                    "source_identities": { "type": "array", "items": { "$ref": "#/$defs/SourceIdentity" } },
                    "provenance": { "type": "object" },
                    "span": { "$ref": "#/$defs/Span" }
                }
            },
            "SourceIdentity": {
                "type": "object",
                "additionalProperties": false,
                "required": ["id", "display_name", "content_hash"],
                "properties": {
                    "id": { "type": "integer", "minimum": 0 },
                    "display_name": { "type": "string" },
                    "content_hash": { "type": ["array", "null"] }
                }
            },
            "Comment": {
                "type": "object",
                "required": ["kind", "text", "span", "target"],
                "properties": {
                    "kind": { "enum": ["Line", "Block", "DocLine", "DocBlock"] },
                    "text": { "type": "string" },
                    "span": { "$ref": "#/$defs/Span" },
                    "target": { "type": ["integer", "null"] }
                }
            },
            "Statement": variant_schema(STATEMENT_VARIANTS),
            "MeshStatement": variant_schema(MESH_STATEMENT_VARIANTS),
            "Expr": variant_schema(EXPR_VARIANTS),
            "CanonicalGeometryIr": {
                "type": "object",
                "required": ["schema_version", "source", "source_identities", "parameters", "entities", "physical_groups", "boundaries", "interfaces", "periodic_pairs", "coordinate_systems", "material_regions", "fields", "mesh", "optimization", "uncertainties", "sensitivity"],
                "properties": {
                    "schema_version": { "const": CURRENT_IR_SCHEMA_VERSION.0 },
                    "source": { "type": "object" },
                    "source_identities": { "type": "array", "items": { "$ref": "#/$defs/SourceIdentity" } },
                    "parameters": { "type": "array" },
                    "entities": { "type": "array" },
                    "controls": { "type": "array" },
                    "physical_groups": { "type": "array" },
                    "boundaries": { "type": "array" },
                    "interfaces": { "type": "array" },
                    "periodic_pairs": { "type": "array" },
                    "coordinate_systems": { "type": "array" },
                    "material_regions": { "type": "array" },
                    "fields": { "type": "array" },
                    "mesh": { "type": "object" },
                    "optimization": { "type": "object" },
                    "uncertainties": { "type": "array" },
                    "sensitivity": { "type": "object" }
                }
            }
        }
    })
}

fn variant_schema(variants: &[&str]) -> Value {
    Value::Object(serde_json::Map::from_iter([(
        "oneOf".into(),
        Value::Array(
            variants
                .iter()
                .map(|variant| {
                    let variant = *variant;
                    // Payloads are Rust-typed bodies carried as JSON objects;
                    // constraining them to `object` (rather than `{}`) means
                    // a payload-shape regression that emits a scalar fails
                    // structural validation instead of passing silently.
                    // Full per-field payload schemas await schemars-derived
                    // definitions (review mid-term item).
                    json!({
                        "type": "object",
                        "additionalProperties": false,
                        "required": [variant],
                        "properties": { (variant): { "type": "object" } }
                    })
                })
                .collect(),
        ),
    )]))
}

pub fn typescript_bindings() -> String {
    let mut output = String::from(
        "// Generated by meshlang_ast::tooling. Do not edit.\n\
         export type U64 = number;\n\
         export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };\n\
         export interface Span { start: number; end: number; file_id: number }\n\
         export type CommentKind = 'Line' | 'Block' | 'DocLine' | 'DocBlock';\n\
         export interface Comment { kind: CommentKind; text: string; span: Span; target: U64 | null }\n\
         export interface SourceMetadata { identity: unknown; language_version: { major: number; minor: number } }\n",
    );
    write_external_union(&mut output, "Declaration", DECLARATION_VARIANTS);
    write_external_union(&mut output, "Statement", STATEMENT_VARIANTS);
    write_external_union(&mut output, "MeshStatement", MESH_STATEMENT_VARIANTS);
    write_external_union(&mut output, "Expr", EXPR_VARIANTS);
    writeln!(
        &mut output,
        "export interface SourceIdentity {{ id: number; display_name: string; content_hash: number[] | null }}"
    )
    .unwrap();
    writeln!(
        &mut output,
        "export interface SourceFile {{ declarations: Declaration[]; metadata: SourceMetadata; comments: Comment[]; source_identities?: SourceIdentity[]; provenance: unknown; span: Span }}"
    )
    .unwrap();
    writeln!(
        &mut output,
        "export interface VersionedAst {{ schema_version: {}; source_file: SourceFile }}",
        CURRENT_AST_SCHEMA_VERSION.0
    )
    .unwrap();
    writeln!(
        &mut output,
        "export interface CanonicalGeometryIr {{ schema_version: {}; source: SourceMetadata; source_identities: SourceIdentity[]; parameters: JsonValue[]; entities: JsonValue[]; controls: JsonValue[]; loops: JsonValue[]; physical_groups: JsonValue[]; boundaries: JsonValue[]; interfaces: JsonValue[]; periodic_pairs: JsonValue[]; coordinate_systems: JsonValue[]; material_regions: JsonValue[]; fields: JsonValue[]; mesh: JsonValue; optimization: JsonValue; uncertainties: JsonValue[]; sensitivity: JsonValue }}",
        CURRENT_IR_SCHEMA_VERSION.0
    )
    .unwrap();
    output
}

fn write_external_union(output: &mut String, name: &str, variants: &[&str]) {
    write!(output, "export type {name} = ").unwrap();
    for (index, variant) in variants.iter().enumerate() {
        if index > 0 {
            output.push_str(" | ");
        }
        write!(output, "{{ {variant}: JsonValue }}").unwrap();
    }
    output.push_str(";\n");
}

pub fn python_bindings() -> String {
    let mut output = String::from(
        r#"# Generated by meshlang_ast.tooling. Do not edit.
from __future__ import annotations
from dataclasses import dataclass
from types import NoneType
from typing import Literal, TypeAlias

JsonValue: TypeAlias = NoneType | bool | int | float | str | list['JsonValue'] | dict[str, 'JsonValue']

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    file_id: int

CommentKind: TypeAlias = Literal['Line', 'Block', 'DocLine', 'DocBlock']

@dataclass
class Comment:
    kind: CommentKind
    text: str
    span: Span
    target: int | None

"#,
    );
    write_python_union(&mut output, "Declaration", DECLARATION_VARIANTS);
    write_python_union(&mut output, "Statement", STATEMENT_VARIANTS);
    write_python_union(&mut output, "MeshStatement", MESH_STATEMENT_VARIANTS);
    write_python_union(&mut output, "Expr", EXPR_VARIANTS);
    writeln!(
        &mut output,
        "\n@dataclass\nclass SourceMetadata:\n    identity: JsonValue\n    language_version: dict[str, int]\n\n\
         @dataclass\nclass SourceIdentity:\n    id: int\n    display_name: str\n    content_hash: list[int] | None\n\n\
         @dataclass\nclass SourceFile:\n    declarations: list[Declaration]\n    metadata: SourceMetadata\n    comments: list[Comment]\n    source_identities: list[SourceIdentity]\n    provenance: JsonValue\n    span: Span\n\n\
         @dataclass\nclass VersionedAst:\n    schema_version: Literal[{}]\n    source_file: SourceFile\n\n\
         @dataclass\nclass CanonicalGeometryIr:\n    schema_version: Literal[{}]\n    source: SourceMetadata\n    source_identities: list[SourceIdentity]\n    payload: dict[str, JsonValue]\n\n\
         AST_SCHEMA_VERSION: Literal[{}] = {}\nIR_SCHEMA_VERSION: Literal[{}] = {}",
        CURRENT_AST_SCHEMA_VERSION.0,
        CURRENT_IR_SCHEMA_VERSION.0,
        CURRENT_AST_SCHEMA_VERSION.0,
        CURRENT_AST_SCHEMA_VERSION.0,
        CURRENT_IR_SCHEMA_VERSION.0,
        CURRENT_IR_SCHEMA_VERSION.0
    )
    .unwrap();
    output
}

fn write_python_union(output: &mut String, name: &str, variants: &[&str]) {
    writeln!(
        output,
        "{name}: TypeAlias = dict[str, JsonValue]  # {}",
        variants.join(" | ")
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tooling_artifacts_are_deterministic_and_cover_all_public_variants() {
        // Deliberate mutation guard: enum declarations generate these lists,
        // while the fixed counts require every schema change to update this
        // compatibility test. The exhaustive visitor matches provide the
        // corresponding compile-time traversal gate.
        assert_eq!(crate::Declaration::VARIANT_NAMES.len(), 20);
        assert_eq!(crate::Statement::VARIANT_NAMES.len(), 6);
        assert_eq!(crate::MeshStatement::VARIANT_NAMES.len(), 8);
        assert_eq!(crate::Expr::VARIANT_NAMES.len(), 13);
        assert_eq!(
            crate::Declaration::VARIANT_NAMES.len(),
            crate::Declaration::VARIANT_PAYLOADS.len()
        );
        let first = generate_tooling_artifacts();
        let repeated = generate_tooling_artifacts();
        assert_eq!(first, repeated);
        validate_tooling_artifacts(&first).expect("tooling artifacts must match the AST contract");
        serde_json::from_str::<Value>(&first.json_schema).unwrap();
        for variant in DECLARATION_VARIANTS
            .iter()
            .chain(MESH_STATEMENT_VARIANTS)
            .chain(EXPR_VARIANTS)
        {
            assert!(first.json_schema.contains(variant));
            assert!(first.typescript.contains(variant));
        }
        assert!(first.python.contains("AST_SCHEMA_VERSION"));
        assert!(first.python.contains("IR_SCHEMA_VERSION"));
    }

    #[test]
    fn real_artifact_matches_schema_discriminators_structurally() {
        use crate::{
            BinaryExpr, BinaryOp, ConstDecl, Declaration, Expr, Ident, Literal, ParamDecl,
            SourceFile, Span,
        };

        // A real artifact with nested expressions: discriminators must be
        // single-key objects drawn from the variant inventories, never bare
        // strings, so this test exercises structure instead of substrings.
        let ident = |name: &str| Ident {
            name: name.into(),
            span: Span::DUMMY,
        };
        let file = SourceFile::new(
            vec![
                Declaration::Param(ParamDecl {
                    name: ident("pitch"),
                    type_ann: None,
                    value: Expr::Binary(BinaryExpr {
                        left: Box::new(Expr::literal(Literal::Int(18), Span::DUMMY)),
                        op: BinaryOp::Add,
                        right: Box::new(Expr::literal(Literal::Int(12), Span::DUMMY)),
                        span: Span::DUMMY,
                    }),
                    unit: None,
                    span: Span::DUMMY,
                }),
                Declaration::Const(ConstDecl {
                    name: ident("steps"),
                    type_ann: None,
                    value: Expr::literal(Literal::Int(4), Span::DUMMY),
                    unit: None,
                    span: Span::DUMMY,
                }),
            ],
            Span::DUMMY,
        );
        let artifact = serde_json::to_value(&file).expect("fixture SourceFile must serialize");
        let declarations = artifact["declarations"]
            .as_array()
            .expect("artifact declarations must be an array");
        assert_eq!(declarations.len(), 2);
        for declaration in declarations {
            let body = declaration
                .as_object()
                .expect("artifact declaration must be an object");
            assert_eq!(body.len(), 1, "declaration must carry one discriminator");
            let (discriminator, payload) = body.iter().next().expect("non-empty object");
            assert!(
                DECLARATION_VARIANTS.contains(&discriminator.as_str()),
                "unknown declaration discriminator: {discriminator}"
            );
            assert!(
                payload.is_object(),
                "declaration payload must be an object, not {payload}"
            );
        }

        // The generated schema must constrain the same discriminators
        // structurally: one branch per variant, each requiring its key with
        // an object payload. Parsed field-by-field, not substring-matched.
        let schema = json_schema();
        let declaration_branches =
            schema["$defs"]["SourceFile"]["properties"]["declarations"]["items"]["oneOf"]
                .as_array()
                .expect("declaration items must be a oneOf schema");
        assert_eq!(declaration_branches.len(), DECLARATION_VARIANTS.len());
        for (branch, variant) in declaration_branches.iter().zip(DECLARATION_VARIANTS) {
            assert_eq!(branch["type"], "object");
            assert_eq!(branch["additionalProperties"], false);
            assert_eq!(branch["required"], serde_json::json!([variant]));
            assert_eq!(branch["properties"][variant]["type"], "object");
        }
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            CURRENT_AST_SCHEMA_VERSION.0
        );
    }
}
