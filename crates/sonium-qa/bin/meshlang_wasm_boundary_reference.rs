//! Bounded CPU contract for the product-facing MeshLang WASM frontend API.
//!
//! This exercises the same pure functions used by wasm-bindgen. It does not
//! build a WASM module, start a browser, read a path, invoke Gmsh, or run a
//! room simulation.

use ed25519_zebra::{SigningKey, VerificationKeyBytes};
use meshlang_compiler::{
    REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION, RegistryPackageIndex, RegistryPackageIndexEnvelope,
    RegistryPackageIndexSignature, RegistryPackageManifest, RegistryPackageModule,
    RegistrySignaturePolicy, RegistryTrustAnchor,
};
use meshlang_parser::ParseCache;
use serde_json::Value;
use sonium_qa::QaResult;
use sonium_roomsim::{
    mesh_meshlang_projection_json, meshlang_frontend_contract_json,
    parse_meshlang_cache_stats_json, parse_meshlang_cached_json, parse_meshlang_json,
    preview_meshlang_json, preview_meshlang_with_registry_json,
    preview_meshlang_with_trusted_package_json, validate_meshlang_grammar_json,
};

const CASE_ID: &str = "meshlang.wasm-boundary-reference.v1";
const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn run() -> Result<QaResult, String> {
    let contract: Value = serde_json::from_str(
        &meshlang_frontend_contract_json().map_err(|error| format!("contract: {error}"))?,
    )
    .map_err(|error| format!("contract JSON: {error}"))?;
    if contract["api_schema"] != 2
        || !contract["filesystem_free"].as_bool().unwrap_or(false)
        || contract["production_meshing"].as_bool().unwrap_or(true)
        || contract["operations"].as_array().map_or(0, Vec::len) != 10
    {
        return Err("product frontend capability contract drifted".into());
    }

    let parse: Value = serde_json::from_str(
        &parse_meshlang_json(SOURCE, "qa-boundary.mlg")
            .map_err(|error| format!("parse envelope: {error}"))?,
    )
    .map_err(|error| format!("parse JSON: {error}"))?;
    if parse["parse_schema"] != 1 || !parse["report"]["source_file"].is_object() {
        return Err("parser envelope contract drifted".into());
    }

    let preview: Value = serde_json::from_str(
        &preview_meshlang_json(SOURCE, "qa-boundary.mlg")
            .map_err(|error| format!("preview envelope: {error}"))?,
    )
    .map_err(|error| format!("preview JSON: {error}"))?;
    if preview["preview_schema"] != 1
        || preview["canonical_ir"]["document_version"] != 1
        || !preview["canonical_ir_hash"].is_string()
        || !preview["sensitivity_manifest"]["entries"].is_array()
        || !preview["provenance_manifest"]["source_map"].is_object()
    {
        return Err("canonical IR preview manifest contract drifted".into());
    }

    let projection_source = r#"geometry model {
        let solid = Sphere(r: 0.5)
        let surface = solid |> project_iso_surface()
    }"#;
    let projection_request = r#"{
        "bounds": [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        "cells": [4, 4, 4],
        "tolerance": 0.01
    }"#;
    let projection: Value = serde_json::from_str(
        &mesh_meshlang_projection_json(
            projection_source,
            "qa-projection.mlg",
            "model.surface",
            projection_request,
        )
        .map_err(|error| format!("projection artifact: {error}"))?,
    )
    .map_err(|error| format!("projection artifact JSON: {error}"))?;
    if projection["schema_version"] != 1
        || projection["mesh"]["cells"] != serde_json::json!([4, 4, 4])
        || !projection["mesh"]["triangles"].is_array()
    {
        return Err("projection artifact contract drifted".into());
    }
    let import_result = preview_meshlang_json(
        "import \"stdlib/shapes.mlg\"\ngeometry Box(1.0, 1.0, 1.0)",
        "qa-import.mlg",
    );
    if import_result.is_ok() {
        return Err("filesystem import was accepted by single-source preview".into());
    }

    let registry = r#"{"schema_version":1,"modules":[{"path":"pkg/part@1.mlg","source":"pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n"}]}"#;
    let registry_preview = preview_meshlang_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "qa-registry.mlg",
        registry,
    )
    .map_err(|error| format!("registry preview: {error}"))?;
    let registry_preview: Value = serde_json::from_str(&registry_preview)
        .map_err(|error| format!("registry preview JSON: {error}"))?;
    if !registry_preview["registry_manifest_hash"].is_string() {
        return Err("registry preview did not expose a manifest hash".into());
    }

    let mut package = RegistryPackageManifest {
        name: "pkg".into(),
        version: "1".into(),
        modules: vec![RegistryPackageModule {
            path: "pkg/part@1.mlg".into(),
            source: "pub geometry Part { let solid = Box(1.0, 1.0, 1.0) }\n".into(),
        }],
        dependencies: Vec::new(),
        manifest_hash: String::new(),
    };
    package.manifest_hash = package
        .canonical_hash()
        .map_err(|error| format!("package manifest hash: {error}"))?
        .to_string();
    let index = RegistryPackageIndex {
        schema_version: REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION,
        packages: vec![package],
    }
    .validate_and_canonicalize()
    .map_err(|error| format!("package index: {error}"))?;
    let signing_key = SigningKey::from([9_u8; 32]);
    let key_id = "qa-wasm-package-key-v1";
    let detached = signing_key.sign(
        &index
            .signing_payload(key_id)
            .map_err(|error| error.to_string())?,
    );
    let public_key: [u8; 32] = VerificationKeyBytes::from(&signing_key).into();
    let envelope = RegistryPackageIndexEnvelope {
        index,
        signature: RegistryPackageIndexSignature {
            algorithm: "ed25519".into(),
            key_id: key_id.into(),
            public_key: hex(&public_key),
            signature: hex(&detached.to_bytes()),
        },
    };
    let policy = RegistrySignaturePolicy {
        trusted_keys: vec![RegistryTrustAnchor {
            key_id: key_id.into(),
            public_key: hex(&public_key),
        }],
    };
    let trusted_package_preview = preview_meshlang_with_trusted_package_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "qa-package.mlg",
        &envelope
            .to_json()
            .map_err(|error| format!("package envelope JSON: {error}"))?,
        &serde_json::to_string(&policy).map_err(|error| format!("package policy JSON: {error}"))?,
    )
    .map_err(|error| format!("trusted package preview: {error}"))?;
    let trusted_package_preview: Value = serde_json::from_str(&trusted_package_preview)
        .map_err(|error| format!("trusted package preview JSON: {error}"))?;
    if trusted_package_preview["registry_signature_key_id"] != key_id
        || !trusted_package_preview["registry_manifest_hash"].is_string()
    {
        return Err("trusted package preview authentication metadata drifted".into());
    }

    let mut cache = ParseCache::default();
    let first = parse_meshlang_cached_json(&mut cache, SOURCE, "qa-cache.mlg")
        .map_err(|error| format!("cached parse: {error}"))?;
    let second = parse_meshlang_cached_json(&mut cache, SOURCE, "qa-cache.mlg")
        .map_err(|error| format!("cached hit parse: {error}"))?;
    if first != second {
        return Err("cached parser changed the JSON envelope".into());
    }
    let cache_stats: Value = serde_json::from_str(
        &parse_meshlang_cache_stats_json(&cache)
            .map_err(|error| format!("cache stats: {error}"))?,
    )
    .map_err(|error| format!("cache stats JSON: {error}"))?;
    if cache_stats["hits"] != 1 || cache_stats["misses"] != 1 || cache_stats["entries"] != 1 {
        return Err(format!("unexpected cache stats: {cache_stats}"));
    }

    let grammar: Value = serde_json::from_str(
        &validate_meshlang_grammar_json().map_err(|error| format!("grammar: {error}"))?,
    )
    .map_err(|error| format!("grammar JSON: {error}"))?;
    if grammar["schema_version"] != 1
        || grammar["operator_spellings"].as_array().map_or(0, Vec::len) < 7
    {
        return Err("grammar contract drifted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("contract_checks", 4.0)
        .metric("parse_preview_checks", 5.0)
        .metric("registry_checks", 2.0)
        .metric("package_checks", 3.0)
        .metric("grammar_checks", 2.0)
        .metric("cache_checks", 3.0)
        .metric("checks_passed", 20.0)
        .input_checksum("source:meshlang-wasm-boundary:v1")
        .reference("sonium-roomsim pure MeshLang frontend product boundary")
        .reference_checksum("analytic/deterministic-fixture:meshlang-wasm-boundary-v1")
        .dimension("host forwarding functions exercised with filesystem/import rejection")
        .formulation("wasm-bindgen JSON envelope compatibility")
        .tolerance(
            "nineteen version, capability, manifest, rejection, schema, package, and cache checks",
        )
        .seed("meshlang-wasm-boundary-reference:cpu:v1"))
}

fn main() {
    let started = std::time::Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
