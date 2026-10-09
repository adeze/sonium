//! Release-backed MeshLang source-map and provenance sidecar round trip.

use meshlang_ast::{CommentKind, PROVENANCE_CONTRACT_VERSION, ProvenanceOperation, SourceMap};
use meshlang_compiler::{CompileResult, compile_to_geo_full, compile_to_python_full};
use serde_json::Value;
use sonium_qa::QaResult;
use std::fs;
use std::path::Path;

const CASE_ID: &str = "meshlang.source-provenance-roundtrip.v1";

fn backend_source_map(result: &CompileResult) -> Result<SourceMap, String> {
    let metadata: Value = serde_json::from_str(&result.backend_metadata_json)
        .map_err(|error| format!("backend metadata JSON: {error}"))?;
    serde_json::from_value(metadata["source_map"]["source_file"].clone())
        .map_err(|error| format!("backend source map: {error}"))
}

fn provenance_source_map(result: &CompileResult) -> Result<SourceMap, String> {
    let manifest: Value = serde_json::from_str(&result.provenance_manifest_json)
        .map_err(|error| format!("provenance manifest JSON: {error}"))?;
    serde_json::from_value(manifest["source_map"].clone())
        .map_err(|error| format!("provenance source map: {error}"))
}

fn run() -> Result<QaResult, String> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let root_path = directory.path().join("root.mlg");
    let imported_path = directory.path().join("part.mlg");
    let root_source = r#"
        // root import
        import "part.mlg"

        /// root geometry
        geometry model {
            // root body
            let body = Box(2.0, 2.0, 2.0)
        }
    "#;
    let imported_source = r#"
        /// imported geometry
        pub geometry Part {
            // imported body
            let body = Box(1.0, 1.0, 1.0)
        }
    "#;
    fs::write(&root_path, root_source).map_err(|error| error.to_string())?;
    fs::write(&imported_path, imported_source).map_err(|error| error.to_string())?;

    let filename = root_path
        .to_str()
        .ok_or_else(|| "temporary MeshLang path is not UTF-8".to_string())?;
    let geo = compile_to_geo_full(root_source, filename).map_err(|error| error.to_string())?;
    let python =
        compile_to_python_full(root_source, filename).map_err(|error| error.to_string())?;
    let geo_map = backend_source_map(&geo)?;
    let python_map = backend_source_map(&python)?;
    let geo_provenance = provenance_source_map(&geo)?;
    let python_provenance = provenance_source_map(&python)?;

    let round_trip = serde_json::from_slice::<SourceMap>(
        &serde_json::to_vec(&geo_map).map_err(|error| format!("serialize source map: {error}"))?,
    )
    .map_err(|error| format!("deserialize source map: {error}"))?;
    let geo_canonical = geo_map
        .canonicalized()
        .map_err(|error| format!("canonicalize GEO source map: {error:?}"))?;
    let python_canonical = python_map
        .canonicalized()
        .map_err(|error| format!("canonicalize Python source map: {error:?}"))?;
    let identities = geo_map
        .source_identities
        .iter()
        .map(|identity| identity.display_name.as_str())
        .collect::<Vec<_>>();
    let imported_id = geo_map
        .source_identities
        .iter()
        .find(|identity| Path::new(&identity.display_name).ends_with("part.mlg"))
        .map(|identity| identity.id)
        .ok_or_else(|| "imported source identity missing".to_string())?;

    let checks = [
        geo_map == python_map,
        geo_map == geo_provenance,
        python_map == python_provenance,
        round_trip == geo_map,
        geo_map.validate().is_ok(),
        python_map.validate().is_ok(),
        geo_canonical == geo_map,
        python_canonical == python_map,
        identities.len() == 2,
        identities
            .iter()
            .any(|name| Path::new(name).ends_with("root.mlg")),
        identities
            .iter()
            .any(|name| Path::new(name).ends_with("part.mlg")),
        geo_map.comments.iter().any(|comment| {
            comment.kind == CommentKind::DocLine
                && comment.text == "/// imported geometry"
                && comment.span.file_id != 0
                && comment.target.is_some()
        }),
        geo_map.comments.iter().any(|comment| {
            comment.kind == CommentKind::DocLine
                && comment.text == "/// root geometry"
                && comment.span.file_id == 0
                && comment.target.is_some()
        }),
        geo_map
            .comments
            .iter()
            .any(|comment| comment.text == "// imported body" && comment.span.file_id != 0),
        geo_map
            .comments
            .iter()
            .any(|comment| comment.text == "// root body" && comment.span.file_id == 0),
        geo_map.provenance.records.iter().any(|record| {
            record.operation == ProvenanceOperation::Parsed
                && record
                    .sources
                    .iter()
                    .any(|source| source.source == imported_id)
        }),
        geo_map.provenance.records.iter().any(|record| {
            record.operation == ProvenanceOperation::Expanded
                && record
                    .sources
                    .iter()
                    .any(|source| source.source == imported_id)
        }),
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "source/provenance round-trip checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("source_identity_count", identities.len() as f64)
        .metric(
            "provenance_contract_version",
            PROVENANCE_CONTRACT_VERSION as f64,
        )
        .minimum("checks_passed", checks.len() as f64)
        .minimum("source_identity_count", 2.0)
        .minimum(
            "provenance_contract_version",
            PROVENANCE_CONTRACT_VERSION as f64,
        )
        .input_checksum("source:root-and-imported-comments:meshlang-source-provenance-v1")
        .reference("independent source-map JSON round trip and GEO/Python provenance parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-source-provenance-v2")
        .dimension("one root and one imported MeshLang source with four comments")
        .formulation(
            "MeshLang compiler source-map JSON round trip plus GEO/Python backend and parsed/expanded provenance parity",
        )
        .tolerance("seventeen canonical-contract, identity, backend, comment, and provenance checks")
        .seed("meshlang-source-provenance-roundtrip:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("MeshLang failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
