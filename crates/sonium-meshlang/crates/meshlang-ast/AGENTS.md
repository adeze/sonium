# meshlang-ast (lib: `meshlang_ast`, version: 0.2.0)

Abstract Syntax Tree types for MeshLang. Pure data model — no parsing, no codegen.

## Layout

```
src/
  types.rs       AST node definitions (Stmt, Expr, GeoStmt, Csg, Kernel, …)
  span.rs        Source position tracking (start, end byte offsets)
  units.rs       Unit parsing (mm, cm, m, deg, rad, …)
  visitor.rs     Visitor trait
  provenance.rs  Source identity, comments, stable node IDs, provenance
  metadata.rs    Design, physics, manufacturing, and mesh metadata
  schema.rs      Versioned AST envelopes and migrations
  hash.rs        Deterministic structural hashes
  ir.rs          Canonical backend-neutral geometry IR
  tooling.rs     JSON Schema and TypeScript/Python artifact generation
  lib.rs         Re-exports
```

## Dependencies

- `serde` / `serde_json` — deterministic versioned persistence, migrations, structural hashes,
  snapshots, and tooling artifacts.

## Notes

- Owned only by `meshlang-parser` (constructs AST) and `meshlang-compiler` (consumes AST).
- Public sum types are declared through `ast_enum!`, which generates the tooling variant
  inventory. Exhaustive visitor matches and fixed compatibility counts must be updated for every
  new variant.
- `Unit::parse` is fallible and must never be converted into silent token dropping by callers.
- Persist ASTs through `VersionedAst`; do not serialize a raw `SourceFile` for storage or IPC.
- Keep the canonical IR free of AST expressions and backend-specific escape strings.

## Testing

```bash
cargo test -p meshlang-ast --lib
cargo check -p meshlang-ast && cargo clippy -p meshlang-ast
cargo run -p meshlang-ast --example generate_tooling -- json-schema
```
