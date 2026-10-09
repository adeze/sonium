//! MeshLang Compiler
//!
//! Compiles MeshLang AST to Gmsh `.geo` files or Python API scripts.

pub mod batch;
mod canonical_ast;
pub mod codegen_geo;
pub mod codegen_python;
pub mod diagnostics;
pub mod error;
pub mod imports;
pub mod incremental;
pub mod lower_ir;
pub mod name_resolution;
pub mod projection;
pub mod projection_native;
pub mod projection_volume;
pub mod registry;
pub mod tag_allocator;
pub mod topology;
pub mod type_check;

pub use batch::{
    parse_source_files, parse_source_files_cached, BatchParseInput, BatchParseOutput, BatchParser,
};
pub use codegen_geo::GeoCodegen;
pub use codegen_python::PythonCodegen;
pub use error::CompileError;
pub use incremental::{
    preview_canonical_ir, preview_canonical_ir_json, preview_canonical_ir_with_options,
    preview_canonical_ir_with_registry_json, preview_canonical_ir_with_signed_registry_json,
    preview_canonical_ir_with_trusted_package_json,
    preview_canonical_ir_with_trusted_registry_json, registry_manifest_signing_payload,
    registry_preview_manifest_hash, verify_registry_manifest_signature, CanonicalIrPreview,
    CanonicalIrPreviewDiagnostic, CanonicalIrPreviewDiagnosticLevel, CanonicalIrPreviewEnvelope,
    IncrementalCompiler, IncrementalCompilerStats, RegistryKeyRotation,
    RegistryKeyRotationEnvelope, RegistryManifestSignature, RegistryPreviewModule,
    RegistrySignatureError, RegistrySignaturePolicy, RegistryTrustAnchor,
    CANONICAL_IR_PREVIEW_SCHEMA_VERSION, MAX_REGISTRY_KEY_ROTATION_BYTES,
    REGISTRY_KEY_ROTATION_SCHEMA_VERSION, REGISTRY_PREVIEW_SCHEMA_VERSION,
};
pub use lower_ir::{lower_to_canonical_ir, CanonicalIrCache};
pub use projection::{
    entity_node_index, evaluate_projection, evaluate_projection_with_index,
    mesh_projection_iso_surface, mesh_projection_iso_surface_with_index, EntityNodeIndex,
    ProjectionArtifactError, ProjectionError, ProjectionEvaluation, ProjectionMesh,
    ProjectionMeshArtifact, MAX_PROJECTION_MESH_ARTIFACT_BYTES,
    PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
};
pub use projection_native::{
    compile_projection_native_artifacts, projection_mesh_to_native, NativeProjectionArtifacts,
    MAX_NATIVE_PROJECTION_SOURCE_BYTES,
};
pub use projection_volume::{
    compile_projection_native_volume_artifacts, projection_mesh_to_native_volume,
    NativeProjectionVolumeArtifacts,
};
#[cfg(not(target_arch = "wasm32"))]
pub use registry::HttpPackageTransport;
#[cfg(not(target_arch = "wasm32"))]
pub use registry::HttpRegistryKeyTransport;
pub use registry::{
    FilesystemPackageRegistry, FilesystemRegistryKeyTransport, RegistryKeyTransport,
    RegistryKeyTransportError, RegistryPackageDependency, RegistryPackageDiscovery,
    RegistryPackageDiscoveryError, RegistryPackageIndex, RegistryPackageIndexEnvelope,
    RegistryPackageIndexSignature, RegistryPackageManifest, RegistryPackageMigrationModule,
    RegistryPackageMigrationPlan, RegistryPackageMigrationPolicy, RegistryPackageModule,
    RegistryPackageModulesWithAliases, RegistryPackageResolution, RegistryPackageTransport,
    RegistryPackageTransportError, RegistryPackageVersionPolicy,
    REGISTRY_PACKAGE_INDEX_SCHEMA_VERSION, REGISTRY_PACKAGE_MIGRATION_SCHEMA_VERSION,
};
pub use topology::TopologyTracker;
pub use type_check::{DiagLevel, Diagnostic, TypeChecker};

use meshlang_ast::{
    CanonicalGeometryIr, Expr, GeometryOperation, Ident, ProjectionMethod, SourceFile, SourceMap,
    StructuralHash, VersionedCanonicalIr, CURRENT_IR_DOCUMENT_VERSION,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Backend-neutral request passed to a Hessian mesh-field provider.
///
/// Providers return the six symmetric tensor components as MeshLang scalar
/// expressions (`xx`, `yy`, `zz`, `xy`, `xz`, `yz`). The same resolved
/// expressions are then emitted by both native backends.
#[derive(Debug, Clone, Copy)]
pub struct HessianMetricRequest<'a> {
    pub target: &'a Expr,
    pub field: &'a Expr,
    pub minimum_size: &'a Expr,
    pub maximum_size: &'a Expr,
    pub coordinate_system: Option<&'a Ident>,
}

/// Deterministic provider for Hessian-derived anisotropic mesh metrics.
///
/// Implementations may bridge an in-process field evaluator or an external
/// field artifact, but must return deterministic scalar expressions for an
/// identical request. Returning expressions (rather than backend text) keeps
/// GEO and Python-Gmsh semantics on one shared plan.
pub trait HessianMetricProvider: std::fmt::Debug + Send + Sync {
    fn resolve(&self, request: HessianMetricRequest<'_>) -> Result<[Expr; 6], String>;
}

/// Filesystem and validation policy for one compilation.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub import_search_paths: Vec<PathBuf>,
    pub import_sandbox_root: Option<PathBuf>,
    pub allow_absolute_imports: bool,
    pub stdlib_path: Option<PathBuf>,
    pub validate_canonical_ir: bool,
    pub hessian_metric_provider: Option<Arc<dyn HessianMetricProvider>>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            import_search_paths: Vec::new(),
            import_sandbox_root: None,
            allow_absolute_imports: false,
            stdlib_path: None,
            validate_canonical_ir: true,
            hessian_metric_provider: None,
        }
    }
}

/// Compilation result including output code and diagnostics.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub code: String,
    pub diagnostics: Vec<Diagnostic>,
    pub tag_map_json: Option<String>,
    pub canonical_ir_json: String,
    pub canonical_ir_hash: StructuralHash,
    pub sensitivity_manifest_json: String,
    pub provenance_manifest_json: String,
    pub backend_metadata_json: String,
}

/// Backend-neutral serialized sections that can be reused independently of
/// backend code, tag allocation, provenance, and backend metadata.
///
/// The cache key is the canonical IR structural hash.  Source comments and
/// source-map records are intentionally excluded: those remain rebuilt for
/// every artifact so comment/provenance edits cannot reuse stale mappings.
#[derive(Debug, Clone)]
pub(crate) struct CachedArtifactSections {
    pub(crate) canonical_ir_json: String,
    pub(crate) canonical_ir_hash: StructuralHash,
    pub(crate) sensitivity_manifest_json: String,
}

/// Compile a MeshLang source file to Gmsh .geo format.
///
/// Side-effect free: warnings are returned in [`CompileResult::diagnostics`]
/// via [`compile_to_geo_full`], never printed. The CLI binary performs its
/// own stderr reporting so library and WASM callers stay quiet.
pub fn compile_to_geo(source: &str, filename: &str) -> Result<String, CompileError> {
    Ok(compile_to_geo_full(source, filename)?.code)
}

/// Compile with full diagnostics and tag map.
pub fn compile_to_geo_full(source: &str, filename: &str) -> Result<CompileResult, CompileError> {
    compile_to_geo_full_with_options(source, filename, &CompileOptions::default())
}

/// Materialize a source file through the canonical geometry IR without
/// emitting a backend artifact. This tooling/QA boundary keeps generated
/// expressions traceable to the canonical entity's source span.
pub fn materialize_canonical_geometry(
    source: &str,
    filename: &str,
) -> Result<SourceFile, CompileError> {
    let options = CompileOptions::default();
    let frontend = prepare_frontend(source, filename, &options)?;
    canonical_ast::validate_backend_capabilities_with_hessian_provider(
        &frontend.ir,
        options.hessian_metric_provider.is_some(),
    )?;
    canonical_ast::materialize_geometry(&frontend.ast, &frontend.ir)
}

/// Build a bounded, backend-neutral projection mesh directly from MeshLang
/// source through the canonical IR.
///
/// This is the single source-to-artifact path used by filesystem-free
/// frontends. Native GEO/Python code generation intentionally remains a
/// metadata-only capability for projection operations; this helper does not
/// imply Gmsh-native meshing.
pub fn compile_projection_mesh_artifact(
    source: &str,
    filename: &str,
    projection_name: &str,
    bounds: [[f64; 3]; 2],
    cells: [usize; 3],
    tolerance: f64,
) -> Result<ProjectionMeshArtifact, CompileError> {
    let options = CompileOptions::default();
    let frontend = prepare_frontend(source, filename, &options)?;
    let projection = frontend
        .ir
        .entities
        .iter()
        .find(|node| node.name == projection_name)
        .ok_or_else(|| {
            CompileError::Unsupported(format!("unknown projection entity '{projection_name}'"))
        })?;
    match projection.operation {
        GeometryOperation::Project {
            method: ProjectionMethod::IsoSurface,
        } => {}
        GeometryOperation::Project { method } => {
            return Err(CompileError::Unsupported(format!(
                "projection '{projection_name}' uses {method:?}; mesh export requires iso-surface"
            )));
        }
        _ => {
            return Err(CompileError::Unsupported(format!(
                "entity '{projection_name}' not projection operation"
            )));
        }
    }
    let mesh = mesh_projection_iso_surface(&frontend.ir, projection.id, bounds, cells, tolerance)
        .map_err(|error| {
        CompileError::Unsupported(format!(
            "projection '{projection_name}' mesh evaluation failed: {error}"
        ))
    })?;
    ProjectionMeshArtifact::from_mesh(mesh).map_err(|error| {
        CompileError::Unsupported(format!(
            "projection '{projection_name}' artifact validation failed: {error}"
        ))
    })
}

/// Compile GEO output with explicit import and IR-validation policy.
pub fn compile_to_geo_full_with_options(
    source: &str,
    filename: &str,
    options: &CompileOptions,
) -> Result<CompileResult, CompileError> {
    let frontend = prepare_frontend(source, filename, options)?;

    // Codegen consumes the canonical dimension-qualified ID plan.
    let mut codegen = GeoCodegen::new();
    codegen.apply_canonical_plan(&frontend.ir)?;
    canonical_ast::validate_backend_capabilities_with_hessian_provider(
        &frontend.ir,
        options.hessian_metric_provider.is_some(),
    )?;
    let mut backend_ast = canonical_ast::materialize_geometry(&frontend.ast, &frontend.ir)?;
    canonical_ast::materialize_hessian_metrics(
        &mut backend_ast,
        options.hessian_metric_provider.as_deref(),
    )?;
    codegen.emit_source_file(&backend_ast)?;
    codegen.emit_canonical_controls(&frontend.ir)?;
    codegen.validate_canonical_output(&frontend.ir)?;

    codegen
        .tag_allocator()
        .validate_reverse_mappings()
        .map_err(CompileError::Internal)?;
    let tag_map = codegen
        .tag_allocator()
        .export_map_with_provenance(&frontend.ir);
    let tag_map_json = Some(serde_json::to_string_pretty(&tag_map).map_err(|error| {
        CompileError::Internal(format!("failed to serialize tag map: {error}"))
    })?);
    build_result(
        codegen.output(),
        frontend.diagnostics,
        tag_map_json,
        &frontend.ast,
        &frontend.ir,
        "geo",
    )
}

/// Compile a MeshLang source file to Gmsh Python API script.
pub fn compile_to_python(source: &str, filename: &str) -> Result<String, CompileError> {
    let result = compile_to_python_full(source, filename)?;
    for d in &result.diagnostics {
        if d.level == DiagLevel::Warning {
            eprintln!("{}", d);
        }
    }
    Ok(result.code)
}

/// Compile Python API output with the same artifact metadata as GEO output.
pub fn compile_to_python_full(source: &str, filename: &str) -> Result<CompileResult, CompileError> {
    compile_to_python_full_with_options(source, filename, &CompileOptions::default())
}

/// Compile Python API output with explicit import and IR-validation policy.
pub fn compile_to_python_full_with_options(
    source: &str,
    filename: &str,
    options: &CompileOptions,
) -> Result<CompileResult, CompileError> {
    let frontend = prepare_frontend(source, filename, options)?;

    let mut codegen = PythonCodegen::new();
    codegen.apply_canonical_plan(&frontend.ir)?;
    canonical_ast::validate_backend_capabilities_with_hessian_provider(
        &frontend.ir,
        options.hessian_metric_provider.is_some(),
    )?;
    let mut backend_ast = canonical_ast::materialize_geometry(&frontend.ast, &frontend.ir)?;
    canonical_ast::materialize_hessian_metrics(
        &mut backend_ast,
        options.hessian_metric_provider.as_deref(),
    )?;
    codegen.emit_source_file(&backend_ast)?;
    codegen.emit_canonical_controls(&frontend.ir)?;
    codegen.validate_canonical_output(&frontend.ir)?;
    build_result(
        codegen.output(),
        frontend.diagnostics,
        None,
        &frontend.ast,
        &frontend.ir,
        "python",
    )
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn fail_on_type_errors(diagnostics: &[Diagnostic]) -> Result<(), CompileError> {
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.level == DiagLevel::Error)
        .collect();
    if errors.is_empty() {
        return Ok(());
    }

    Err(CompileError::TypeError {
        message: errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
            .join("; "),
    })
}

pub(crate) struct FrontendProduct {
    pub(crate) ast: meshlang_ast::SourceFile,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) ir: Arc<CanonicalGeometryIr>,
}

fn prepare_frontend(
    source: &str,
    filename: &str,
    options: &CompileOptions,
) -> Result<FrontendProduct, CompileError> {
    let ast =
        meshlang_parser::parse(source, filename).map_err(|e| CompileError::Parse(e.message))?;
    let ast = flatten_imports_with_options(ast, filename, options)?;
    let mut ir_cache = CanonicalIrCache::default();
    analyze_flattened_ast(ast, options, &mut ir_cache)
}

pub(crate) fn analyze_flattened_ast(
    ast: meshlang_ast::SourceFile,
    options: &CompileOptions,
    ir_cache: &mut CanonicalIrCache,
) -> Result<FrontendProduct, CompileError> {
    // Resolve duplicate/module symbols before lowering metadata expressions so
    // source-level diagnostics are not masked by downstream IR failures.
    let _scope = name_resolution::resolve(&ast)?;
    let mut checker = TypeChecker::new();
    let diagnostics = checker.check(&ast);
    fail_on_type_errors(&diagnostics)?;
    let ir = ir_cache.lower(&ast)?;
    ir.validate_provenance().map_err(|errors| {
        CompileError::Internal(
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;
    if options.validate_canonical_ir {
        ir.validate().map_err(|errors| {
            CompileError::Internal(
                errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
    }
    Ok(FrontendProduct {
        ast,
        diagnostics,
        ir,
    })
}

fn entity_provenance_entries(
    ir: &CanonicalGeometryIr,
) -> Result<Vec<serde_json::Value>, CompileError> {
    let nodes = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<BTreeMap<_, _>>();

    fn collect_parent_chain(
        entity: meshlang_ast::EntityId,
        nodes: &BTreeMap<meshlang_ast::EntityId, &meshlang_ast::GeometryNode>,
        visiting: &mut BTreeSet<meshlang_ast::EntityId>,
        emitted: &mut BTreeSet<meshlang_ast::EntityId>,
        chain: &mut Vec<meshlang_ast::EntityId>,
    ) -> Result<(), CompileError> {
        if !visiting.insert(entity) {
            return Err(CompileError::Internal(format!(
                "canonical entity parent cycle at {entity:?}"
            )));
        }
        let node = nodes.get(&entity).ok_or_else(|| {
            CompileError::Internal(format!("canonical entity parent {entity:?} is missing"))
        })?;
        for parent in &node.inputs {
            if emitted.insert(*parent) {
                chain.push(*parent);
            }
            collect_parent_chain(*parent, nodes, visiting, emitted, chain)?;
        }
        visiting.remove(&entity);
        Ok(())
    }

    ir.entities
        .iter()
        .map(|node| {
            let mut chain = Vec::new();
            collect_parent_chain(
                node.id,
                &nodes,
                &mut BTreeSet::new(),
                &mut BTreeSet::new(),
                &mut chain,
            )?;
            Ok(json!({
                "id": node.id,
                "name": node.name,
                "operation": node.operation,
                "parents": node.inputs,
                "parent_chain": chain,
                "provenance": node.provenance,
            }))
        })
        .collect()
}

/// Return the explicit backend-neutral projection execution descriptors.
///
/// Projection nodes describe a field/mesh consumer, not a native Gmsh
/// entity. Keeping this contract in the backend sidecar lets preview,
/// optimization, and WASM clients consume the same canonical request while
/// native geometry code remains fail-closed for accidental CSG use.
/// Build deterministic backend-neutral projection descriptors for a
/// canonical IR. Native GEO/Python consumers use this sidecar contract to
/// hand projection fields to preview/optimization/WASM clients without
/// treating them as native Gmsh geometry.
pub fn projection_backend_descriptors(ir: &CanonicalGeometryIr) -> Vec<serde_json::Value> {
    ir.entities
        .iter()
        .filter_map(|node| {
            let GeometryOperation::Project { method } = &node.operation else {
                return None;
            };
            let method = match method {
                ProjectionMethod::IsoSurface => "iso_surface",
                ProjectionMethod::SignedDistance => "signed_distance",
                ProjectionMethod::ClosestPoint => "closest_point",
            };
            Some(json!({
                "id": node.id,
                "name": node.name,
                "method": method,
                "inputs": node.inputs,
                "arguments": node.arguments,
                "provenance": node.provenance,
                "native_geometry": false,
                "consumer": "ProjectionMeshArtifact",
                "consumer_schema_version": PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
                "consumer_max_bytes": MAX_PROJECTION_MESH_ARTIFACT_BYTES,
            }))
        })
        .collect()
}

pub(crate) fn build_artifact_sections(
    ir: &CanonicalGeometryIr,
) -> Result<CachedArtifactSections, CompileError> {
    let canonical_ir_json = String::from_utf8(
        VersionedCanonicalIr::current(ir.clone())
            .to_json_vec()
            .map_err(|error| {
                CompileError::Internal(format!("failed to serialize canonical IR: {error}"))
            })?,
    )
    .map_err(|error| CompileError::Internal(format!("canonical IR JSON was not UTF-8: {error}")))?;
    let canonical_ir_hash = ir
        .structural_hash()
        .map_err(|error| CompileError::Internal(format!("failed to hash canonical IR: {error}")))?;
    let sensitivity_manifest_json =
        serde_json::to_string_pretty(&ir.sensitivity).map_err(|error| {
            CompileError::Internal(format!("failed to serialize sensitivity manifest: {error}"))
        })?;

    Ok(CachedArtifactSections {
        canonical_ir_json,
        canonical_ir_hash,
        sensitivity_manifest_json,
    })
}

pub(crate) fn build_result(
    code: String,
    diagnostics: Vec<Diagnostic>,
    tag_map_json: Option<String>,
    source_file: &SourceFile,
    ir: &CanonicalGeometryIr,
    backend: &str,
) -> Result<CompileResult, CompileError> {
    let sections = build_artifact_sections(ir)?;
    build_result_with_cached_sections(
        code,
        diagnostics,
        tag_map_json,
        source_file,
        ir,
        backend,
        sections,
    )
}

pub(crate) fn build_result_with_cached_sections(
    code: String,
    diagnostics: Vec<Diagnostic>,
    tag_map_json: Option<String>,
    source_file: &SourceFile,
    ir: &CanonicalGeometryIr,
    backend: &str,
    sections: CachedArtifactSections,
) -> Result<CompileResult, CompileError> {
    let actual_hash = ir
        .structural_hash()
        .map_err(|error| CompileError::Internal(format!("failed to hash canonical IR: {error}")))?;
    if actual_hash != sections.canonical_ir_hash {
        return Err(CompileError::Internal(
            "cached artifact sections do not match canonical IR".into(),
        ));
    }
    let CachedArtifactSections {
        canonical_ir_json: cached_canonical_ir_json,
        canonical_ir_hash: cached_canonical_ir_hash,
        sensitivity_manifest_json: cached_sensitivity_manifest_json,
    } = sections;

    let canonical_ir_json = cached_canonical_ir_json;
    let canonical_ir_hash = cached_canonical_ir_hash;
    let sensitivity_manifest_json = cached_sensitivity_manifest_json;

    // Sidecars are a persistence/API boundary, so do not serialize parser or
    // import provenance until the stable contract has been canonicalized and
    // validated. This keeps GEO and Python metadata byte-deterministic and
    // prevents malformed ancestry from escaping through a successful compile.
    let source_map = SourceMap::from_source_file(source_file)
        .canonicalized()
        .map_err(|error| {
            CompileError::Internal(format!("invalid source/provenance map: {error:?}"))
        })?;
    let provenance_manifest = json!({
        "source": &ir.source,
        "source_map": &source_map,
        "entities": entity_provenance_entries(ir)?,
        "physical_groups": ir.physical_groups.iter().map(|group| json!({
            "id": group.id,
            "name": group.name,
            "material": group.material,
            "provenance": group.provenance,
        })).collect::<Vec<_>>(),
        "material_regions": ir.material_regions.iter().map(|region| json!({
            "name": region.name,
            "material": region.material,
            "entities": region.entities,
            "properties": region.properties,
            "provenance": region.provenance,
        })).collect::<Vec<_>>(),
    });
    let provenance_manifest_json =
        serde_json::to_string_pretty(&provenance_manifest).map_err(|error| {
            CompileError::Internal(format!("failed to serialize provenance manifest: {error}"))
        })?;
    let backend_metadata_json = serde_json::to_string_pretty(&json!({
        "artifact_schema": 1,
        "backend": backend,
        "compiler": "meshlang-compiler",
        "compiler_version": env!("CARGO_PKG_VERSION"),
        "canonical_ir_schema": ir.schema_version,
        "canonical_ir_document_version": CURRENT_IR_DOCUMENT_VERSION,
        "canonical_ir_hash": canonical_ir_hash,
        // Keep a deterministic digest of the emitted product alongside the
        // IR identity.  Consumers can use this to detect accidental changes
        // in backend output while retaining the source/IR provenance below.
        "reproducibility": {
            "artifact_hash": StructuralHash::from_bytes(code.as_bytes()).to_string(),
            "hash_kind": "meshlang-structural-fnv1a-256",
        },
        "toolchain": {
            // Executed-artifact CI supplies these values through the
            // environment; ordinary compilation remains Gmsh-independent.
            "gmsh_version": std::env::var("MESHLANG_GMSH_VERSION").ok(),
            "python_gmsh_version": std::env::var("MESHLANG_PYTHON_GMSH_VERSION").ok(),
        },
        "source_map": {
            "source": &ir.source,
            "source_file": &source_map,
            "entities": entity_provenance_entries(ir)?,
            "physical_groups": ir.physical_groups.iter().map(|group| json!({
                "id": group.id,
                "name": group.name,
                "provenance": group.provenance,
            })).collect::<Vec<_>>(),
        },
        "projections": projection_backend_descriptors(ir),
        "gmsh_compatibility": ">=4.8",
        // Keep solver-facing metadata in the backend sidecar as well as in
        // the canonical IR.  Gmsh physical groups have no material-property
        // channel of their own.
        "physical_groups": ir.physical_groups,
        "material_regions": ir.material_regions,
        "mesh_controls": ir.mesh,
    }))
    .map_err(|error| {
        CompileError::Internal(format!("failed to serialize backend metadata: {error}"))
    })?;

    Ok(CompileResult {
        code,
        diagnostics,
        tag_map_json,
        canonical_ir_json,
        canonical_ir_hash,
        sensitivity_manifest_json,
        provenance_manifest_json,
        backend_metadata_json,
    })
}

/// Build the provenance manifest used by the filesystem-free preview
/// envelope. Backend sidecars use the same canonical fields in `build_result`.
pub(crate) fn preview_provenance_manifest_value(
    source_file: &SourceFile,
    ir: &CanonicalGeometryIr,
) -> Result<serde_json::Value, CompileError> {
    let source_map = SourceMap::from_source_file(source_file)
        .canonicalized()
        .map_err(|error| {
            CompileError::Internal(format!("invalid source/provenance map: {error:?}"))
        })?;
    Ok(json!({
        "source": &ir.source,
        "source_map": &source_map,
        "entities": entity_provenance_entries(ir)?,
        "physical_groups": ir.physical_groups.iter().map(|group| json!({
            "id": group.id,
            "name": group.name,
            "provenance": group.provenance,
        })).collect::<Vec<_>>(),
        "material_regions": ir.material_regions.iter().map(|region| json!({
            "name": region.name,
            "material": region.material,
            "entities": region.entities,
            "properties": region.properties,
            "provenance": region.provenance,
        })).collect::<Vec<_>>(),
    }))
}

/// Run the import-flattening pass on a freshly parsed AST.
///
/// The base directory for resolving relative import paths is derived from
/// `filename`. When `filename` has no directory component (e.g. `"test.mlg"`
/// in unit tests), imports are resolved relative to the working directory.
fn flatten_imports_with_options(
    ast: meshlang_ast::SourceFile,
    filename: &str,
    options: &CompileOptions,
) -> Result<meshlang_ast::SourceFile, CompileError> {
    // Quick check: skip the filesystem work entirely when there are no imports.
    let has_imports = ast
        .declarations
        .iter()
        .any(|d| matches!(d, meshlang_ast::Declaration::Import(_)));
    if !has_imports {
        return Ok(ast);
    }

    let base_path = Path::new(filename);
    let base_dir = base_path.parent().unwrap_or(Path::new("."));
    let mut resolver = imports::ImportResolver::with_defaults(base_dir);
    for path in &options.import_search_paths {
        resolver.add_search_path(path);
    }
    if let Some(path) = &options.stdlib_path {
        resolver.set_stdlib_path(path);
    }
    if let Some(root) = &options.import_sandbox_root {
        resolver.set_sandbox_root(root)?;
    }
    resolver.set_allow_absolute_paths(options.allow_absolute_imports);
    flatten_imports_with_resolver(ast, filename, &mut resolver)
}

pub(crate) fn flatten_imports_with_resolver(
    ast: meshlang_ast::SourceFile,
    filename: &str,
    resolver: &mut imports::ImportResolver,
) -> Result<meshlang_ast::SourceFile, CompileError> {
    let has_imports = ast
        .declarations
        .iter()
        .any(|d| matches!(d, meshlang_ast::Declaration::Import(_)));
    if !has_imports {
        return Ok(ast);
    }
    resolver.flatten(&ast, Path::new(filename))
}

#[cfg(test)]
mod projection_sidecar_tests {
    use super::{
        lower_to_canonical_ir, projection_backend_descriptors, MAX_PROJECTION_MESH_ARTIFACT_BYTES,
        PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
    };

    #[test]
    fn projection_backend_descriptors_are_stable_and_backend_neutral() {
        let source = r#"
            geometry model {
                let solid = Box(1.0, 1.0, 1.0)
                let iso = solid |> project_iso_surface(tolerance: 0.01)
                let signed = solid |> project_signed_distance(tolerance: 0.02)
                let closest = solid |> project_closest_point(tolerance: 0.03)
            }
        "#;
        let ast = meshlang_parser::parse(source, "projection-sidecar.mlg")
            .expect("projection fixture should parse");
        let ir = lower_to_canonical_ir(&ast).expect("projection fixture should lower");
        let first = projection_backend_descriptors(&ir);
        let second = projection_backend_descriptors(&ir);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert_eq!(
            first
                .iter()
                .map(|descriptor| descriptor["method"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["iso_surface", "signed_distance", "closest_point"]
        );
        assert!(first.iter().all(|descriptor| {
            descriptor["native_geometry"] == false
                && descriptor["consumer"] == "ProjectionMeshArtifact"
                && descriptor["consumer_schema_version"] == PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION
                && descriptor["consumer_max_bytes"] == MAX_PROJECTION_MESH_ARTIFACT_BYTES
                && descriptor["inputs"].is_array()
                && descriptor["arguments"].is_array()
                && descriptor["provenance"].is_array()
        }));
    }
}
