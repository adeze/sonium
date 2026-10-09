# meshlang-compiler

MeshLang compiler: parses `.mlg` source, lowers a backend-neutral canonical
geometry IR, and emits Gmsh `.geo` files or Python API scripts.

## Installation

```toml
[dependencies]
meshlang-compiler = { workspace = true }
```

Provides the `meshlang` binary:

```bash
cargo run -p meshlang-compiler --bin meshlang -- input.mlg --backend geo --out output.geo
```

## Modules

- `imports` — module resolution and `use` statement handling

### Import path/version policy

Filesystem and standard-library imports resolve in deterministic order: the
importing file, configured search paths, then the configured stdlib root. An
explicit numeric selector on the final path component (for example
`module@2.mlg`, `module@1.2.mlg`, or `std/acoustics@2.mlg`) selects that exact
file. Selectors with multiple `@` characters, non-numeric versions, or a
directory-level selector are rejected before filesystem access. A missing
requested version never falls back to an unversioned module. Browser/WASM
preview APIs remain intentionally import-free.

Package and editor clients can instead call `ImportResolver::register_module` to
install a filesystem-independent source registry. Registry keys always use `/`
separators and `.mlg` names, resolve through deterministic synthetic paths, and
share the normal cache, dependency, source-identity, and comment-provenance
pipeline. Registry version selectors are exact and duplicate or host-dependent
keys fail closed. Trusted hosts may extend this boundary with
`register_trusted_package_from_discovery`: a discovery callback returns only
candidate versions, the caller explicitly chooses `Exact` or `HighestAtMost`,
and the selected exact version still crosses signed transport before any
    registry mutation. The validated package index also exposes a deterministic
    dependency-first transitive closure for hosts that need to register packages
    in dependency order. Network clients, key distribution, and migration policy
remain host-owned.
- The single-source preview remains import-free; registry clients can use
  `preview_canonical_ir_with_registry_json` (exported to WASM as
  `previewCanonicalIrWithRegistry`) with a bounded, versioned module request.
- `name_resolution` — symbol table, scope tracking
- `type_check` — static type checking over the AST
- `lower_ir` — deterministic dimension-qualified canonical geometry IR
- `incremental` — long-lived module/IR caches and preview APIs
- `topology` — geometry topology bookkeeping (entity dimensions, parent/child)
- `tag_allocator` — Gmsh entity tag allocation, dimension-aware
- `codegen_geo` — Gmsh `.geo` emission (Point, Curve, Surface, Volume, CSG, Compound, …)
- `codegen_python` — Gmsh Python API emission
- `diagnostics` — span-anchored compiler diagnostics
- `error` — `CompileError`

## Pipeline

```
source (.mlg)
   ├─ meshlang-parser ──▶ AST
   ├─ imports ──▶ resolved module graph
   ├─ name_resolution ──▶ symbol table
   ├─ type_check ──▶ typed AST
   ├─ lower_ir ──▶ canonical geometry IR + manifests
   ├─ topology + tag_allocator consume the canonical entity plan
   └─ codegen_geo | codegen_python ──▶ output
```

## Library APIs

One-shot compilation returns backend code plus the shared canonical IR,
sensitivity/provenance manifests, deterministic backend metadata, and the GEO
tag map:

```rust
let result = meshlang_compiler::compile_to_geo_full(source, "model.mlg")?;
println!("{}", result.canonical_ir_hash);
```

Editors and optimization loops should retain an `IncrementalCompiler`. Imported
modules are revalidated by source-content hash and flattened ASTs reuse IR by
structural hash:

```rust
use meshlang_compiler::IncrementalCompiler;

let mut compiler = IncrementalCompiler::new("project");
let preview = compiler.preview_canonical_ir(source, "project/model.mlg")?;
let geo = compiler.compile_to_geo(source, "project/model.mlg")?;
println!("{:?}", compiler.stats());
```

`preview_canonical_ir_json` is an import-free, ownership-neutral frontend API.
On `wasm32` it is exported as `previewCanonicalIr`. Browser callers must provide
one self-contained source file; native module graphs use `IncrementalCompiler`.

## Inline-function control flow

Inline functions may end in a value-producing compile-time-resolvable
`if`/`else` or compile-time `for`. Their local bindings and branch/loop values are substituted before
canonical lowering; GEO and Python never independently interpret those
function-control-flow constructs. A final compile-time loop folds entity
results with canonical CSG union semantics.

## Parameterized geometry ports

Templates can expose multiple typed entity ports with `-> (...)`. Their body is
lowered once into the canonical instance namespace, so both backends consume
the same ports without source-level template calls:

```meshlang
geometry Frame(w: Float) -> (body: Solid, inlet: Solid) {
    let body = Box(w, 1.0, 1.0)
    let inlet = Cylinder(r: 0.1, h: 0.5)
}
geometry model { let frame = Frame(2.0) }
```

Ports are read as `frame.body` and `frame.inlet`; the first declared port is
also the compatibility primary result of `Frame(...)`. Ports must be direct
bindings with concrete geometry types and no default values.

## Testing

## Bounded projection consumer

`evaluate_projection` provides a backend-neutral, in-process consumer for canonical `GeometryOperation::Project` nodes. It supports signed-distance, iso-surface admission, closest-point queries, Point/Sphere/Box/Cylinder SDFs, Boolean and Array composition, nested projection, Translate, uniform Scale, and resolved scalar/vector/parameter arguments. Evaluation is bounded by geometry-depth and closest-point iteration caps and rejects cycles, dynamic values, non-finite inputs/results, invalid tolerances, and unsupported operations explicitly. Native GEO/Python/Gmsh projected-operation emission and full remeshing remain separate backend/product work.

```bash
cargo test -p meshlang-compiler --lib
cargo test -p meshlang-compiler --test integration
cargo test -p meshlang-compiler --test incremental
cargo check -p meshlang-compiler --all-targets
cargo clippy -p meshlang-compiler --all-targets -- -D warnings
cargo bench -p meshlang-compiler --bench compiler_bench
```

The offline fuzz harness is under `fuzz/`; see its README. Performance scenarios
and cache interpretation are documented in `PERFORMANCE.md`.

`mesh_projection_iso_surface` adds a deterministic, resource-bounded marching-tetrahedra consumer for scalar iso-surfaces; it validates finite bounds, grid limits, duplicate-free indexed triangles, and oriented output.

`ProjectionMeshArtifact` adds a versioned JSON hand-off for that bounded mesh. It validates the schema, finite bounds/grid, finite vertices, triangle indices, and a 64 MiB serialized-size limit before serialization or consumption; this remains a backend-neutral preview/optimization artifact, not native Gmsh projection emission.

Projection descriptors in backend metadata are explicit and deterministic: they
record the canonical entity, method, inputs, arguments, provenance, the
`ProjectionMeshArtifact` consumer, its schema version, and its byte budget
while marking `native_geometry: false`. Consumers can reject an artifact
contract they do not understand before attempting to decode it.
Native backends therefore cannot silently treat a projection field as a Gmsh
entity; CSG use remains a typed capability error.

## Incremental artifact sections

`IncrementalCompiler` also shares serialized canonical-IR and sensitivity
sections between GEO and Python artifacts keyed by canonical IR hash. Backend
code, tag maps, source/comment provenance, and backend metadata remain rebuilt
per artifact; `IncrementalCompilerStats` exposes section hit/miss counters.

## Current limitations

- The browser preview intentionally does not resolve filesystem imports; the
  registry preview accepts only explicitly registered source modules.
- Sensitivity manifests track geometry input/argument dataflow and mesh/material
  metadata references; topology-dependent variables retain explicit remesh and
  state-transfer responsibilities.
- Hessian mesh metrics accept a deterministic in-process or external adapter
  through `CompileOptions::hessian_metric_provider`. It resolves the symmetric
  tensor once for shared GEO/Python emission; missing and failed providers are
  typed errors. Affine transforms are emitted by
  both native backends, and cyclic-symmetry regions emit rotated native sector
  copies in GEO/Python; executed Gmsh parity remains release-QA work.
- Material properties are preserved in canonical IR/manifests and the backend
  metadata sidecar; Gmsh itself has no native physical-group material channel.
- Generated GEO/Python parity has configured execution coverage; release QA
  still needs a versioned Gmsh execution matrix and downstream MSH4 consumer
  smoke tests.

## Perturbation and topology hand-off

Optimization or remeshing consumers can compare two decoded
`CanonicalGeometryIr` values with `first.topology_diff(&candidate)`. The
result ignores numeric argument changes, preserves stable entity IDs as a
continuity signal, and reports added/removed/modified entity nodes plus
changed dynamic-branch membership. `topology_changed()` is the explicit
admission signal for a topology-aware remesh, while `markers_changed()`
identifies physical-group continuity changes without falsely classifying them
as geometry topology changes. Sensitivity entries continue to carry the
required remesh and state-transfer responsibility for each design variable.
`compile_projection_mesh_artifact` centralizes the source-to-canonical-IR lookup and
artifact validation for filesystem-free callers. It is deterministic for identical
source, named projection, bounds, cells, and tolerance, and reports unknown/non-iso
projections before sampling. This remains distinct from native GEO/Python/Gmsh
projection execution.
