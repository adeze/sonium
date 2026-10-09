# meshlang-compiler (lib: `meshlang_compiler`, version: 0.1.7)

MeshLang compiler. Frontend wraps `meshlang-parser`; backend emits Gmsh `.geo` or a Python API description.

## Layout

```
src/
  lib.rs              Public entry points (compile_to_geo, compile_to_python)
  main.rs             `meshlang` binary
  imports.rs          Module resolution / `use` handling
  incremental.rs      Long-lived compiler facade + JSON/WASM IR preview
  lower_ir.rs         Backend-neutral canonical geometry IR lowering/cache
  name_resolution.rs  Symbol table, scope tracking
  type_check.rs       Static type checking
  topology.rs         Entity dimension bookkeeping (parent/child)
  tag_allocator.rs    Gmsh entity tag allocation per dimension
  codegen_geo.rs      Gmsh .geo emission
  codegen_python.rs   Gmsh Python API emission
  diagnostics.rs      Span-anchored compiler diagnostics
  error.rs            CompileError
```

## Compiler Pipeline

```
.mlg source
  → meshlang-parser → AST
  → imports → resolved module graph
  → name_resolution → symbol table
  → type_check → typed AST
  → lower_ir → canonical geometry IR + manifests
  → topology + tag_allocator consume canonical IDs/dimensions
  → codegen_geo | codegen_python
```

## Binary

```
cargo run -p meshlang-compiler --bin meshlang -- input.mlg --backend geo --out out.geo
```

## Testing

```bash
cargo test -p meshlang-compiler --lib
cargo test -p meshlang-compiler --test integration
cargo test -p meshlang-compiler --test incremental
cargo check -p meshlang-compiler --all-targets
cargo clippy -p meshlang-compiler --all-targets -- -D warnings
cargo bench -p meshlang-compiler --bench compiler_bench
```

`tests/integration.rs` runs a wide set of `examples/*.mlg` programs through the full compiler.

`tests/integration.rs` covers shipping examples and backend semantics;
`tests/incremental.rs` covers cache reuse/invalidation and preview contracts.
`benches/compiler_bench.rs` covers large fragment arrays, nested CSG, import
graphs, and candidate recompilation. The bounded fuzz target and seed corpus are
under `fuzz/`.

## Current limitations

- Browser/WASM preview is single-source and import-free.
- Sensitivity dependencies use geometry input/argument dataflow and include
  mesh/material metadata references.
- Hessian metrics require `CompileOptions::hessian_metric_provider`; the
  provider resolves one backend-neutral six-component tensor plan shared by
  GEO and Python, while a missing or failed provider remains a typed error.
  Affine transforms are emitted by both native backends; cyclic
  symmetry regions emit deterministic rotated sector copies in both native
  backends, with executed Gmsh parity still owned by release QA.
- Real Gmsh execution parity and downstream MSH4 consumer smoke tests remain
  release-QA responsibilities.

## Notes

- Tag allocation is dimension-keyed; do not collapse the per-dimension counters.
- Public entity IDs start at 1 within each dimension; internal lowering IDs
  start at `1 << 32`.
- Keep GEO and Python codegen on the canonical ID/dimension plan.
- Keep inline-function local substitution and value-producing compile-time
  `if`/`for` expansion in canonical IR; backends must not become the authority for
  multi-statement function semantics.
- Imported-module cache keys use source hashes; IR cache keys use flattened AST
  structural hashes. Do not replace either with timestamps.
- All AST visitors should preserve span info — diagnostics depend on it.
- The `meshlang` feature in `sonium-mesher` shells out to this compiler at build/test time.
