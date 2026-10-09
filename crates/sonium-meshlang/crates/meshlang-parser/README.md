# meshlang-parser

Recursive-descent parser for MeshLang.

Source `.mlg` text → `meshlang_ast::SourceFile` (with span-annotated nodes and structured diagnostics).

## Installation

```toml
[dependencies]
meshlang-parser = { workspace = true }
```

## Modules

- `grammar::validate_tree_sitter_grammar` checks the checked-in editor grammar against the Rust operator and precedence contract without requiring Node.js.

- `lexer` — token stream, handles strings, numbers with units, identifiers, operators (including pipe-arrow `|>`)
- `parser` — recursive-descent parser producing `meshlang_ast::SourceFile`
- `error` — `ParseError` with span-anchored diagnostics
- `GRAMMAR.md` — normative language grammar and precedence for `version 1`
- `fuzz/` — coverage-guided target and permanent minimized regression corpus
- `benches/parser_bench.rs` — generated large-program throughput benchmark

## Usage

```rust
use meshlang_parser::parse;

let program = parse(source_text)?;
```

For editor and batch use, `parse_recovering` returns a partial `SourceFile`
plus bounded typed diagnostics. `ParseOptions` also enforces a source-byte
limit before lexing, preventing oversized inputs from allocating unbounded
token buffers. The diagnostic wire schema is version 2 and includes the
source-size failure category. `ParseCache` provides deterministic file-level
caching, and the `wasm32` build exports both `parseMeshLang` and
`validateMeshLangGrammar`; the latter returns a versioned, validated
Rust/tree-sitter contract for editor clients.
The diagnostic wire contract is identified by `error::DIAGNOSTIC_SCHEMA_VERSION`,
and the grammar contract by `GRAMMAR_CONTRACT_SCHEMA_VERSION`.

## Testing

```bash
cargo test -p meshlang-parser --lib
cargo test -p meshlang-parser --test roadmap
cargo check -p meshlang-parser && cargo clippy -p meshlang-parser
```
