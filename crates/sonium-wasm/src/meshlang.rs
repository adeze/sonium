//! Filesystem-free MeshLang editor/frontend bindings for the product WASM crate.
//!
//! The parser and compiler own the individual wire formats.  This module keeps
//! the product boundary deliberately thin: it forwards those versioned JSON
//! envelopes, publishes one capability contract, and never touches a file,
//! Gmsh, a mesh worker, or a simulation solver.

use meshlang_ast::{GeometryOperation, ProjectionMethod};
use meshlang_compiler::{
    CANONICAL_IR_PREVIEW_SCHEMA_VERSION, ProjectionMeshArtifact, REGISTRY_PREVIEW_SCHEMA_VERSION,
    compile_projection_mesh_artifact, mesh_projection_iso_surface, preview_canonical_ir,
    preview_canonical_ir_json, preview_canonical_ir_with_registry_json,
    preview_canonical_ir_with_signed_registry_json, preview_canonical_ir_with_trusted_package_json,
    preview_canonical_ir_with_trusted_registry_json,
};
use meshlang_parser::{
    GRAMMAR_CONTRACT_SCHEMA_VERSION, PARSE_JSON_SCHEMA_VERSION, ParseCache, ParseOptions,
    SourceRange, grammar_contract_json, parse_json, parse_subtree_json,
};
use serde::{Deserialize, Serialize};

/// Version of the product-facing capability/operation contract.
pub const MESHLANG_FRONTEND_API_SCHEMA_VERSION: u32 = 2;

/// Stable identifier for the pure frontend surface.
pub const MESHLANG_FRONTEND_API_NAME: &str = "sonium-meshlang-frontend-v2";

#[derive(Debug, Serialize)]
struct FrontendContract {
    api_schema: u32,
    api_name: &'static str,
    filesystem_free: bool,
    production_meshing: bool,
    operations: [&'static str; 10],
    parse_schema: u32,
    grammar_contract_schema: u32,
    canonical_ir_preview_schema: u32,
    registry_preview_schema: u32,
}

/// Return the stable product/frontend capability contract as JSON.
pub fn meshlang_frontend_contract_json() -> Result<String, String> {
    serde_json::to_string(&FrontendContract {
        api_schema: MESHLANG_FRONTEND_API_SCHEMA_VERSION,
        api_name: MESHLANG_FRONTEND_API_NAME,
        filesystem_free: true,
        production_meshing: false,
        operations: [
            "meshlangFrontendContract",
            "parseMeshLang",
            "validateMeshLangGrammar",
            "previewCanonicalIr",
            "meshProjection",
            "previewCanonicalIrWithRegistry",
            "previewCanonicalIrWithSignedRegistry",
            "previewCanonicalIrWithTrustedRegistry",
            "previewCanonicalIrWithTrustedPackage",
            "MeshLangEditor",
        ],
        parse_schema: PARSE_JSON_SCHEMA_VERSION,
        grammar_contract_schema: GRAMMAR_CONTRACT_SCHEMA_VERSION,
        canonical_ir_preview_schema: CANONICAL_IR_PREVIEW_SCHEMA_VERSION,
        registry_preview_schema: REGISTRY_PREVIEW_SCHEMA_VERSION,
    })
    .map_err(|error| format!("failed to serialize MeshLang frontend contract: {error}"))
}

/// Build a bounded, versioned projection mesh artifact from canonical IR.
///
/// This is an analytic consumer rather than a native GEO or Python emitter:
/// it is filesystem-free and uses the compiler's bounded
/// marching-tetrahedra implementation. The projection is selected by its
/// resolved canonical name (for example, `model.surface`).
pub fn mesh_meshlang_projection_json(
    source: &str,
    filename: &str,
    projection_name: &str,
    request_json: &str,
) -> Result<String, String> {
    let request: ProjectionMeshRequest = serde_json::from_str(request_json)
        .map_err(|error| format!("invalid projection mesh request: {error}"))?;
    compile_projection_mesh_artifact(
        source,
        filename,
        projection_name,
        request.bounds,
        request.cells,
        request.tolerance,
    )
    .map_err(|error| error.to_string())?
    .to_json()
    .map_err(|error| error.to_string())
}

pub fn mesh_meshlang_projection_json_legacy(
    source: &str,
    filename: &str,
    projection_name: &str,
    request_json: &str,
) -> Result<String, String> {
    let request: ProjectionMeshRequest = serde_json::from_str(request_json)
        .map_err(|error| format!("invalid projection mesh request: {error}"))?;
    let preview = preview_canonical_ir(source, filename).map_err(|error| error.to_string())?;
    let node = preview
        .canonical_ir
        .entities
        .iter()
        .find(|node| node.name == projection_name)
        .ok_or_else(|| format!("unknown projection entity '{projection_name}'"))?;
    match node.operation {
        GeometryOperation::Project {
            method: ProjectionMethod::IsoSurface,
        } => {}
        GeometryOperation::Project { method } => {
            return Err(format!(
                "projection '{projection_name}' uses {method:?}; mesh export requires iso-surface"
            ));
        }
        _ => {
            return Err(format!(
                "entity '{projection_name}' is not a projection operation"
            ));
        }
    }
    let mesh = mesh_projection_iso_surface(
        &preview.canonical_ir,
        node.id,
        request.bounds,
        request.cells,
        request.tolerance,
    )
    .map_err(|error| error.to_string())?;
    ProjectionMeshArtifact::from_mesh(mesh)
        .and_then(|artifact| artifact.to_json())
        .map_err(|error| error.to_string())
}

/// Parse source into the parser's recovering, versioned JSON envelope.
pub fn parse_meshlang_json(source: &str, filename: &str) -> Result<String, String> {
    parse_json(source, filename, ParseOptions::default())
        .map_err(|error| format!("failed to serialize MeshLang parse envelope: {error}"))
}

/// Parse through a caller-owned deterministic file cache. The returned
/// envelope is byte-compatible with [`parse_meshlang_json`].
pub fn parse_meshlang_cached_json(
    cache: &mut ParseCache,
    source: &str,
    filename: &str,
) -> Result<String, String> {
    cache
        .parse_json(source, filename, ParseOptions::default())
        .map_err(|error| format!("failed to serialize cached MeshLang parse envelope: {error}"))
}

/// Parse one checked byte range while retaining original-file spans and
/// source identity. This is a bounded editor primitive; it does not perform
/// dependency-aware incremental compilation.
pub fn parse_meshlang_subtree_json(
    source: &str,
    filename: &str,
    start: usize,
    end: usize,
) -> Result<String, String> {
    parse_subtree_json(
        source,
        filename,
        SourceRange::new(start, end),
        ParseOptions::default(),
    )
    .map_err(|error| format!("failed to serialize MeshLang subtree parse envelope: {error}"))
}

/// Parse one checked byte range through a caller-owned deterministic cache.
pub fn parse_meshlang_cached_subtree_json(
    cache: &mut ParseCache,
    source: &str,
    filename: &str,
    start: usize,
    end: usize,
) -> Result<String, String> {
    cache
        .parse_subtree_json(
            source,
            filename,
            SourceRange::new(start, end),
            ParseOptions::default(),
        )
        .map_err(|error| format!("failed to serialize cached MeshLang subtree envelope: {error}"))
}

/// Serialize cache hit/miss counters for editor telemetry.
pub fn parse_meshlang_cache_stats_json(cache: &ParseCache) -> Result<String, String> {
    let stats = cache.stats();
    serde_json::to_string(&serde_json::json!({
        "hits": stats.hits,
        "misses": stats.misses,
        "entries": cache.len(),
        "subtree_entries": cache.subtree_len(),
    }))
    .map_err(|error| format!("failed to serialize MeshLang parse-cache stats: {error}"))
}

/// Return the parser/tree-sitter grammar contract used by editor tooling.
pub fn validate_meshlang_grammar_json() -> Result<String, String> {
    grammar_contract_json()
}

/// Preview canonical IR without resolving filesystem imports.
pub fn preview_meshlang_json(source: &str, filename: &str) -> Result<String, String> {
    preview_canonical_ir_json(source, filename).map_err(|error| error.to_string())
}

/// Preview canonical IR against an explicit, versioned in-memory registry.
pub fn preview_meshlang_with_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, String> {
    preview_canonical_ir_with_registry_json(source, filename, registry_json)
        .map_err(|error| error.to_string())
}

/// Preview canonical IR against a signed in-memory registry.
pub fn preview_meshlang_with_signed_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, String> {
    preview_canonical_ir_with_signed_registry_json(source, filename, registry_json)
        .map_err(|error| error.to_string())
}

/// Preview canonical IR against a signed registry and pinned trust policy.
pub fn preview_meshlang_with_trusted_registry_json(
    source: &str,
    filename: &str,
    registry_json: &str,
    trust_policy_json: &str,
) -> Result<String, String> {
    preview_canonical_ir_with_trusted_registry_json(
        source,
        filename,
        registry_json,
        trust_policy_json,
    )
    .map_err(|error| error.to_string())
}

/// Preview canonical IR against a host-fetched signed package index.
pub fn preview_meshlang_with_trusted_package_json(
    source: &str,
    filename: &str,
    package_json: &str,
    trust_policy_json: &str,
) -> Result<String, String> {
    preview_canonical_ir_with_trusted_package_json(
        source,
        filename,
        package_json,
        trust_policy_json,
    )
    .map_err(|error| error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn js_error(error: String) -> wasm_bindgen::JsValue {
    wasm_bindgen::JsValue::from_str(&error)
}

/// Browser/editor capability and schema contract.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = meshlangFrontendContract)]
pub fn meshlang_frontend_contract_wasm() -> Result<String, wasm_bindgen::JsValue> {
    meshlang_frontend_contract_json().map_err(js_error)
}

/// Browser/editor recovering parser API.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = parseMeshLang)]
pub fn parse_meshlang_wasm(source: &str, filename: &str) -> Result<String, wasm_bindgen::JsValue> {
    parse_meshlang_json(source, filename).map_err(js_error)
}

/// Stateful editor parser backed by deterministic file and bounded-subtree
/// caches.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct MeshLangEditor {
    cache: ParseCache,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionMeshRequest {
    bounds: [[f64; 3]; 2],
    cells: [usize; 3],
    tolerance: f64,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl MeshLangEditor {
    #[wasm_bindgen::prelude::wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cache: ParseCache::default(),
        }
    }

    /// Parse a file and return the versioned recovering JSON envelope.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = parse)]
    pub fn parse(&mut self, source: &str, filename: &str) -> Result<String, wasm_bindgen::JsValue> {
        parse_meshlang_cached_json(&mut self.cache, source, filename).map_err(js_error)
    }

    /// Parse one source range and return the versioned recovering JSON
    /// envelope with absolute original-file spans.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = parseSubtree)]
    pub fn parse_subtree(
        &mut self,
        source: &str,
        filename: &str,
        start: usize,
        end: usize,
    ) -> Result<String, wasm_bindgen::JsValue> {
        parse_meshlang_cached_subtree_json(&mut self.cache, source, filename, start, end)
            .map_err(js_error)
    }

    /// Remove one cached file; returns whether an entry was removed.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn invalidate(&mut self, filename: &str) -> bool {
        self.cache.invalidate(filename)
    }

    /// Remove every cached report while retaining cumulative counters.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Return deterministic hit/miss counters and entry count.
    #[wasm_bindgen::prelude::wasm_bindgen]
    pub fn stats(&self) -> Result<String, wasm_bindgen::JsValue> {
        parse_meshlang_cache_stats_json(&self.cache).map_err(js_error)
    }
}

/// Browser/editor grammar parity API.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = validateMeshLangGrammar)]
pub fn validate_meshlang_grammar_wasm() -> Result<String, wasm_bindgen::JsValue> {
    validate_meshlang_grammar_json().map_err(js_error)
}

/// Browser/editor canonical IR preview API.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIr)]
pub fn preview_meshlang_wasm(
    source: &str,
    filename: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_meshlang_json(source, filename).map_err(js_error)
}

/// Browser/editor bounded projection mesh artifact API.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = meshProjection)]
pub fn mesh_meshlang_projection_wasm(
    source: &str,
    filename: &str,
    projection_name: &str,
    request_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    mesh_meshlang_projection_json(source, filename, projection_name, request_json).map_err(js_error)
}

/// Browser/editor canonical IR preview with an explicit in-memory registry.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithRegistry)]
pub fn preview_meshlang_with_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_meshlang_with_registry_json(source, filename, registry_json).map_err(js_error)
}

/// Browser/editor canonical IR preview with signature verification.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithSignedRegistry)]
pub fn preview_meshlang_with_signed_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_meshlang_with_signed_registry_json(source, filename, registry_json).map_err(js_error)
}

/// Browser/editor canonical IR preview with pinned trust anchors.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithTrustedRegistry)]
pub fn preview_meshlang_with_trusted_registry_wasm(
    source: &str,
    filename: &str,
    registry_json: &str,
    trust_policy_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_meshlang_with_trusted_registry_json(source, filename, registry_json, trust_policy_json)
        .map_err(js_error)
}

/// Browser/editor canonical IR preview with a host-fetched signed package index.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = previewCanonicalIrWithTrustedPackage)]
pub fn preview_meshlang_with_trusted_package_wasm(
    source: &str,
    filename: &str,
    package_json: &str,
    trust_policy_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    preview_meshlang_with_trusted_package_json(source, filename, package_json, trust_policy_json)
        .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";
    const PROJECTION_SOURCE: &str = r#"
        geometry model {
            let solid = Sphere(r: 0.5)
            let surface = solid |> project_iso_surface()
        }
    "#;

    #[test]
    fn contract_is_versioned_and_filesystem_free() {
        let contract: Value = serde_json::from_str(&meshlang_frontend_contract_json().unwrap())
            .expect("frontend contract JSON");
        assert_eq!(contract["api_schema"], MESHLANG_FRONTEND_API_SCHEMA_VERSION);
        assert_eq!(contract["api_name"], MESHLANG_FRONTEND_API_NAME);
        assert!(contract["filesystem_free"].as_bool().unwrap_or(false));
        assert!(!contract["production_meshing"].as_bool().unwrap_or(true));
        assert_eq!(contract["operations"].as_array().unwrap().len(), 10);
    }

    #[test]
    fn parse_and_preview_forward_versioned_envelopes() {
        let parse: Value = serde_json::from_str(&parse_meshlang_json(SOURCE, "test.mlg").unwrap())
            .expect("parse envelope JSON");
        assert_eq!(parse["parse_schema"], PARSE_JSON_SCHEMA_VERSION);
        assert_eq!(parse["grammar_version"], "1.0");

        let preview: Value =
            serde_json::from_str(&preview_meshlang_json(SOURCE, "test.mlg").unwrap())
                .expect("preview envelope JSON");
        assert_eq!(
            preview["preview_schema"],
            CANONICAL_IR_PREVIEW_SCHEMA_VERSION
        );
        assert_eq!(preview["canonical_ir"]["document_version"], 1);
    }

    #[test]
    fn projection_mesh_boundary_returns_validated_artifact() {
        let request = r#"{
            "bounds": [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
            "cells": [4, 4, 4],
            "tolerance": 0.01
        }"#;
        let artifact: Value = serde_json::from_str(
            &mesh_meshlang_projection_json(
                PROJECTION_SOURCE,
                "projection.mlg",
                "model.surface",
                request,
            )
            .expect("projection artifact"),
        )
        .expect("projection artifact JSON");
        assert_eq!(artifact["schema_version"], 1);
        assert_eq!(artifact["mesh"]["cells"], serde_json::json!([4, 4, 4]));
        assert!(!artifact["mesh"]["triangles"].as_array().unwrap().is_empty());
    }

    #[test]
    fn projection_mesh_boundary_rejects_non_iso_and_unknown_entities() {
        let request = r#"{
            "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "cells": [2, 2, 2],
            "tolerance": 0.01
        }"#;
        let source =
            PROJECTION_SOURCE.replace("project_iso_surface()", "project_signed_distance()");
        let error =
            mesh_meshlang_projection_json(&source, "projection.mlg", "model.surface", request)
                .expect_err("signed-distance projection must not mesh");
        assert!(error.contains("requires iso-surface"));
        let error = mesh_meshlang_projection_json(
            PROJECTION_SOURCE,
            "projection.mlg",
            "model.missing",
            request,
        )
        .expect_err("unknown projection must fail");
        assert!(error.contains("unknown projection entity"));
    }

    #[test]
    fn single_source_preview_rejects_filesystem_imports() {
        let result = preview_meshlang_json(
            "import \"stdlib/shapes.mlg\"\ngeometry Box(1.0, 1.0, 1.0)",
            "import.mlg",
        );
        assert!(result.is_err());
    }

    #[test]
    fn registry_preview_accepts_only_explicit_registry_payloads() {
        let registry = r#"{"schema_version":1,"modules":[{"path":"pkg/part@1.mlg","source":"pub const PART = 1\n"}]}"#;
        let result = preview_meshlang_with_registry_json(
            "import \"pkg/part@1.mlg\"\ngeometry model { let body = Box(1.0, 1.0, 1.0) }",
            "registry.mlg",
            registry,
        )
        .expect("explicit registry preview");
        let preview: Value = serde_json::from_str(&result).expect("preview envelope JSON");
        assert_eq!(
            preview["preview_schema"],
            CANONICAL_IR_PREVIEW_SCHEMA_VERSION
        );
        assert!(preview["registry_manifest_hash"].is_string());
    }

    #[test]
    fn cached_product_parse_reuses_report_and_exposes_stats() {
        let mut cache = ParseCache::default();
        let first = parse_meshlang_cached_json(&mut cache, SOURCE, "cached.mlg").unwrap();
        let second = parse_meshlang_cached_json(&mut cache, SOURCE, "cached.mlg").unwrap();
        assert_eq!(first, second);
        let stats: serde_json::Value =
            serde_json::from_str(&parse_meshlang_cache_stats_json(&cache).unwrap()).unwrap();
        assert_eq!(stats["hits"], 1);
        assert_eq!(stats["misses"], 1);
        assert_eq!(stats["subtree_entries"], 0);
    }

    #[test]
    fn cached_product_subtree_parse_preserves_absolute_range_and_telemetry() {
        let source = "param before = 1\nparam selected = 2\n";
        let start = source.find("param selected").unwrap();
        let end = source.len();
        let mut cache = ParseCache::default();
        let first =
            parse_meshlang_cached_subtree_json(&mut cache, source, "subtree.mlg", start, end)
                .unwrap();
        let second =
            parse_meshlang_cached_subtree_json(&mut cache, source, "subtree.mlg", start, end)
                .unwrap();
        assert_eq!(first, second);
        let value: Value = serde_json::from_str(&first).unwrap();
        assert_eq!(value["report"]["source_file"]["span"]["start"], start);
        let stats: Value =
            serde_json::from_str(&parse_meshlang_cache_stats_json(&cache).unwrap()).unwrap();
        assert_eq!(stats["hits"], 1);
        assert_eq!(stats["misses"], 1);
        assert_eq!(stats["subtree_entries"], 1);
    }
}
