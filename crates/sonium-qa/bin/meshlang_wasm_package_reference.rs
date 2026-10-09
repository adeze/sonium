//! Static release contract for the publishable MeshLang WASM frontend package.
//!
//! The package itself is exercised by the Node contract tests. This bounded
//! CPU QA case verifies the checked-in package manifest, runtime adapter, and
//! TypeScript declarations stay aligned with the Rust capability contract. It
//! intentionally does not build WebAssembly, launch a browser, invoke Gmsh,
//! or start a simulation.

use serde_json::Value;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.wasm-package-reference.v1";
const PACKAGE_JSON: &str = include_str!("../../sonium-wasm/frontend/package.json");
const RUNTIME: &str = include_str!("../../sonium-wasm/frontend/index.mjs");
const DECLARATIONS: &str = include_str!("../../sonium-wasm/frontend/index.d.ts");
const NODE_TESTS: &str = include_str!("../../sonium-wasm/frontend/test/meshlang_frontend.test.mjs");

fn run() -> Result<QaResult, String> {
    let package: Value = serde_json::from_str(PACKAGE_JSON)
        .map_err(|error| format!("package.json is invalid: {error}"))?;
    let package_name = package["name"]
        .as_str()
        .ok_or_else(|| "package name is missing".to_owned())?;
    if package_name != "@sonium/meshlang-wasm-frontend" {
        return Err(format!("unexpected package name: {package_name}"));
    }
    if package["type"] != "module" {
        return Err("package must publish native ESM".into());
    }
    if package["exports"]["."]["import"] != "./index.mjs"
        || package["exports"]["."]["types"] != "./index.d.ts"
    {
        return Err("package exports do not point at the stable adapter".into());
    }
    let files = package["files"]
        .as_array()
        .ok_or_else(|| "package files allow-list is missing".to_owned())?;
    if !files.iter().any(|file| file == "pkg") {
        return Err("generated pkg/ output is not publishable".into());
    }

    let required_runtime_tokens = [
        "createMeshLangFrontend",
        "loadGeneratedMeshLangFrontend",
        "meshlangFrontendContract",
        "parseMeshLang",
        "previewCanonicalIrWithTrustedRegistry",
        "fetchTrustedPackageJson",
        "previewWithTrustedPackageFromFetch",
        "MAX_TRUSTED_PACKAGE_BYTES",
        "MeshLangEditor",
        "loadModule",
        "failed to load the MeshLang WASM module",
        "filesystem-free",
    ];
    let runtime_checks = required_runtime_tokens
        .iter()
        .filter(|token| RUNTIME.contains(**token))
        .count();
    if runtime_checks != required_runtime_tokens.len() {
        return Err("runtime adapter is missing a required capability token".into());
    }

    let required_type_tokens = [
        "MeshLangFrontendContract",
        "MeshLangWasmEditor",
        "createMeshLangFrontend",
        "loadGeneratedMeshLangFrontend",
        "MeshLangWasmModuleLoader",
        "MeshLangPackageFetcher",
    ];
    let type_checks = required_type_tokens
        .iter()
        .filter(|token| DECLARATIONS.contains(**token))
        .count();
    if type_checks != required_type_tokens.len() {
        return Err("TypeScript declarations are incomplete".into());
    }

    let node_test_checks = [
        "validates the Rust contract",
        "editor adapter exposes cache invalidation",
        "rejects a downgraded",
        "rejects operation-order drift",
        "initializes an injected generated module",
        "reports host module-loader failures",
        "fetches trusted package JSON",
    ]
    .iter()
    .filter(|token| NODE_TESTS.contains(**token))
    .count();
    if node_test_checks != 7 {
        return Err("Node contract tests do not cover the package gates".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("manifest_checks", 4.0)
        .metric("runtime_adapter_checks", runtime_checks as f64)
        .metric("typescript_declaration_checks", type_checks as f64)
        .metric("node_contract_test_checks", node_test_checks as f64)
        .metric(
            "checks_passed",
            (4 + runtime_checks + type_checks + node_test_checks) as f64,
        )
        .minimum("manifest_checks", 4.0)
        .minimum(
            "runtime_adapter_checks",
            required_runtime_tokens.len() as f64,
        )
        .minimum(
            "typescript_declaration_checks",
            required_type_tokens.len() as f64,
        )
        .minimum("node_contract_test_checks", 7.0)
        .input_checksum("source:meshlang-wasm-package:v2")
        .reference("publishable ESM adapter, TypeScript declarations, and Node contract tests")
        .reference_checksum("analytic/deterministic-fixture:meshlang-wasm-package-v2")
        .dimension(
            "package exports, runtime capability validation, editor cache, transport, declarations",
        )
        .formulation("MeshLang browser package contract")
        .tolerance("all static package/runtime/type/test contract checks pass")
        .seed("meshlang-wasm-package-reference:cpu:v1"))
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
