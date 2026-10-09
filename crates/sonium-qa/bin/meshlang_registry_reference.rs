//! Release-facing MeshLang source-registry resolution contract.

use std::path::{Path, PathBuf};

use meshlang_compiler::imports::ImportResolver;
use meshlang_compiler::{
    CANONICAL_IR_PREVIEW_SCHEMA_VERSION, CanonicalIrPreviewEnvelope,
    REGISTRY_PREVIEW_SCHEMA_VERSION, RegistryPreviewModule,
    preview_canonical_ir_with_registry_json, registry_preview_manifest_hash,
};
use serde_json::json;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.registry-reference.v1";

fn registry_path(path: &str) -> PathBuf {
    PathBuf::from("<meshlang-registry>").join(path)
}

fn run() -> Result<QaResult, String> {
    let mut resolver = ImportResolver::new();
    resolver
        .register_module("pkg/root@1.mlg", "import \"pkg/part@1.mlg\"\n")
        .map_err(|error| format!("register root: {error}"))?;
    resolver
        .register_module("pkg/part@1.mlg", "/// registry part\npub const PART = 1\n")
        .map_err(|error| format!("register part: {error}"))?;
    resolver
        .register_module("pkg/part@2.mlg", "pub const PART = 2\n")
        .map_err(|error| format!("register part v2: {error}"))?;
    resolver
        .register_module("std/acoustics@2.mlg", "pub const C_AIR = 343.0\n")
        .map_err(|error| format!("register stdlib: {error}"))?;

    let from = Path::new("/virtual/main.mlg");
    for requested in ["pkg/part@1.mlg", "pkg/part@2.mlg", "std.acoustics@2.mlg"] {
        resolver
            .resolve_path(requested, from)
            .map_err(|error| format!("exact registry resolution for {requested}: {error}"))?;
    }
    if resolver.resolve_path("pkg/part@3.mlg", from).is_ok()
        || resolver.resolve_path("pkg/part.mlg", from).is_ok()
    {
        return Err("registry resolution fell back across an exact version boundary".into());
    }

    let invalid = [
        "../escape.mlg",
        "/absolute.mlg",
        "pkg\\windows.mlg",
        "pkg/no-extension",
    ];
    for path in invalid {
        if resolver.register_module(path, "").is_ok() {
            return Err(format!("host-dependent registry path accepted: {path}"));
        }
    }

    let keys = resolver.registered_modules();
    if keys
        != vec![
            "pkg/part@1.mlg".to_owned(),
            "pkg/part@2.mlg".to_owned(),
            "pkg/root@1.mlg".to_owned(),
            "std/acoustics@2.mlg".to_owned(),
        ]
    {
        return Err(format!("registry ordering is not deterministic: {keys:?}"));
    }

    let root = resolver
        .load_module("pkg/root@1.mlg", from)
        .map_err(|error| format!("load nested registry root: {error}"))?
        .clone();
    if resolver.cache_stats().misses != 2 {
        return Err(format!(
            "nested registry load expected two misses, got {:?}",
            resolver.cache_stats()
        ));
    }
    let flattened = resolver
        .flatten(&root, &registry_path("pkg/root@1.mlg"))
        .map_err(|error| format!("flatten nested registry modules: {error}"))?;
    if flattened.comments.len() != 1 || flattened.comments[0].text != "/// registry part" {
        return Err("registry comment provenance was not retained".into());
    }
    if flattened.source_identities.len() < 2 || resolver.cache_stats().hits == 0 {
        return Err("registry source identities/cache evidence incomplete".into());
    }

    let preview_modules = vec![RegistryPreviewModule {
        path: "pkg/part@1.mlg".into(),
        source: "pub const PART = 1\n".into(),
    }];
    let preview_manifest = registry_preview_manifest_hash(&preview_modules)
        .map_err(|error| format!("registry manifest hash: {error}"))?;
    let preview_manifest_hex = preview_manifest.to_string();
    let preview_registry = json!({
        "schema_version": REGISTRY_PREVIEW_SCHEMA_VERSION,
        "manifest_hash": preview_manifest_hex,
        "modules": preview_modules,
    })
    .to_string();
    let preview = preview_canonical_ir_with_registry_json(
        "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
        "<preview>/root.mlg",
        &preview_registry,
    )
    .map_err(|error| format!("registry preview API: {error}"))?;
    let preview_json: serde_json::Value =
        serde_json::from_str(&preview).map_err(|error| format!("preview JSON: {error}"))?;
    if preview_json["preview_schema"] != 1 || preview_json["canonical_ir_hash"].as_str().is_none() {
        return Err("registry preview envelope is not versioned canonical IR".into());
    }
    let envelope = CanonicalIrPreviewEnvelope::from_json(&preview)
        .map_err(|error| format!("preview envelope validation: {error}"))?;
    if envelope.preview_schema != CANONICAL_IR_PREVIEW_SCHEMA_VERSION
        || envelope.canonical_ir_hash
            != envelope
                .canonical_ir
                .ir
                .structural_hash()
                .map_err(|error| format!("preview envelope hash: {error}"))?
                .to_string()
    {
        return Err("preview envelope hash/schema contract failed".into());
    }
    if envelope.registry_manifest_hash.as_deref() != Some(preview_manifest_hex.as_str()) {
        return Err("preview envelope registry manifest hash missing or incorrect".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("exact_version_checks", 3.0)
        .minimum("exact_version_checks", 3.0)
        .metric("no_fallback_checks", 2.0)
        .minimum("no_fallback_checks", 2.0)
        .metric("registry_validation_checks", 4.0)
        .minimum("registry_validation_checks", 4.0)
        .metric("nested_provenance_checks", 4.0)
        .minimum("nested_provenance_checks", 4.0)
        .metric("registry_preview_checks", 7.0)
        .minimum("registry_preview_checks", 7.0)
        .input_checksum("source:meshlang-registry-fixtures:v1")
        .reference("cross-platform source registry exact-version resolution and nested provenance")
        .reference_checksum("analytic/deterministic-fixture:meshlang-registry-v1")
        .dimension("registry keys, versions, synthetic paths, cache, source identities, comments")
        .formulation("MeshLang package/source registry import boundary")
        .tolerance("all registry ordering, no-fallback, and provenance checks pass")
        .seed("meshlang-registry-reference:cpu:v1"))
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
