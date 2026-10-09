//! Release-facing deterministic MeshLang import path/version policy reference.

use meshlang_compiler::imports::ImportResolver;
use sonium_qa::QaResult;
use std::fs;
use std::time::Instant;
use tempfile::tempdir;

const CASE_ID: &str = "meshlang.import-policy-reference.v1";

fn run() -> Result<QaResult, String> {
    let directory = tempdir().map_err(|error| format!("temporary directory: {error}"))?;
    let main = directory.path().join("main.mlg");
    let exact_v2 = directory.path().join("module@2.mlg");
    let exact_v12 = directory.path().join("module@1.2.mlg");
    fs::write(&exact_v2, "const VALUE = 2\n").map_err(|error| format!("write v2: {error}"))?;
    fs::write(&exact_v12, "const VALUE = 1.2\n").map_err(|error| format!("write v1.2: {error}"))?;
    fs::write(directory.path().join("module.mlg"), "const VALUE = 0\n")
        .map_err(|error| format!("write fallback: {error}"))?;

    let resolver = ImportResolver::new();
    let v2 = resolver
        .resolve_path("module@2.mlg", &main)
        .map_err(|error| format!("resolve @2: {error}"))?;
    let v12 = resolver
        .resolve_path("module@1.2.mlg", &main)
        .map_err(|error| format!("resolve @1.2: {error}"))?;
    if v2
        != exact_v2
            .canonicalize()
            .map_err(|error| format!("canonical v2: {error}"))?
        || v12
            != exact_v12
                .canonicalize()
                .map_err(|error| format!("canonical v1.2: {error}"))?
    {
        return Err("versioned imports resolved to the wrong exact module".into());
    }

    let missing = resolver
        .resolve_path("module@3.mlg", &main)
        .expect_err("missing exact version must not fall back to module.mlg")
        .to_string();
    if !missing.contains("module@3.mlg") {
        return Err(format!(
            "missing-version diagnostic lost requested path: {missing}"
        ));
    }

    let mut rejected = 0usize;
    for path in [
        "module@beta.mlg",
        "module@1@2.mlg",
        "vendor@2/module.mlg",
        "@2.mlg",
    ] {
        let error = resolver
            .resolve_path(path, &main)
            .expect_err("invalid version selector must be rejected")
            .to_string();
        if !error.contains("version") && !error.contains("selector") {
            return Err(format!(
                "invalid selector {path:?} had weak diagnostic: {error}"
            ));
        }
        rejected += 1;
    }
    let separator_error = resolver
        .resolve_path("pkg\\part.mlg", &main)
        .expect_err("host-specific separator must be rejected")
        .to_string();
    if !separator_error.contains("'/' separators") {
        return Err(format!(
            "backslash policy diagnostic was weak: {separator_error}"
        ));
    }

    let stdlib = directory.path().join("stdlib");
    fs::create_dir_all(&stdlib).map_err(|error| format!("create stdlib: {error}"))?;
    let stdlib_v2 = stdlib.join("acoustics@2.mlg");
    fs::write(&stdlib_v2, "const C_AIR = 343.0\n")
        .map_err(|error| format!("write stdlib: {error}"))?;
    let mut stdlib_resolver = ImportResolver::new();
    stdlib_resolver.set_stdlib_path(&stdlib);
    let resolved_stdlib = stdlib_resolver
        .resolve_path("std/acoustics@2.mlg", &main)
        .map_err(|error| format!("resolve stdlib version: {error}"))?;
    if resolved_stdlib
        != stdlib_v2
            .canonicalize()
            .map_err(|error| format!("canonical stdlib: {error}"))?
    {
        return Err("versioned stdlib import resolved to the wrong module".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("exact_version_checks", 2.0)
        .minimum("exact_version_checks", 2.0)
        .metric("no_fallback_checks", 1.0)
        .minimum("no_fallback_checks", 1.0)
        .metric("selector_rejection_checks", rejected as f64)
        .minimum("selector_rejection_checks", 4.0)
        .metric("portable_separator_checks", 1.0)
        .minimum("portable_separator_checks", 1.0)
        .metric("stdlib_version_checks", 1.0)
        .minimum("stdlib_version_checks", 1.0)
        .input_checksum("source:meshlang-import-policy-fixtures:v1")
        .reference("exact numeric module-version selectors, strict rejection, and stdlib parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-import-policy-v1")
        .dimension("relative, search-path, stdlib, invalid-selector, and no-fallback resolution")
        .formulation("MeshLang deterministic import path/version policy")
        .tolerance("all exact and rejection checks pass without fallback")
        .seed("meshlang-import-policy-reference:cpu:v1"))
}

fn main() {
    let started = Instant::now();
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
