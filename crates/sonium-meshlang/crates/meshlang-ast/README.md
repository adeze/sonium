# meshlang-ast

Abstract Syntax Tree types for MeshLang — a small geometry/meshing DSL that compiles to Gmsh `.geo` and a Python API.

## Installation

```toml
[dependencies]
meshlang-ast = { workspace = true }
```

## Modules

- `types` — AST node definitions (statements, expressions, geometry, CSG, kernels)
- `span` — source-location tracking for diagnostics
- `units` — physical-unit parsing (`mm`, `m`, `deg`, `rad`, …)
- `visitor` — visitor trait for AST walks
- `provenance` — source identities, node IDs, comments, metadata
- `metadata` — design variables, distributions, objectives, regions, boundaries, fields
- `schema` — versioned AST envelope plus v1 migration
- `hash` — structural (non-cryptographic) cache identity for AST values
- `ir` — canonical backend-neutral geometry IR with validation and diff
- `state_transfer` — state-transfer plan application and IDW weights
- `tooling` — JSON Schema plus TypeScript/Python artifact generation

## Notes

This crate has no behavior — it is the shared data model used by `meshlang-parser` and `meshlang-compiler`. All AST nodes derive `serde::Serialize` for snapshot tooling.

`GeometryBlock::ports` is the defaulted, serialized contract for named results
of a parameterized geometry template. It preserves v2-envelope compatibility:
older serialized ASTs deserialize with an empty port list.

## Testing

```bash
cargo test -p meshlang-ast --lib
cargo check -p meshlang-ast && cargo clippy -p meshlang-ast
```
