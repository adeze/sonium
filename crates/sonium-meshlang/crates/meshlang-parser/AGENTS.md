# meshlang-parser (lib: `meshlang_parser`, version: 0.1.4)

Recursive-descent parser for MeshLang. Produces `meshlang_ast::SourceFile` from source text.

## Layout

```
src/
  lexer.rs    Token stream (strings, numbers+units, identifiers, |>)
  parser.rs   Recursive-descent parser
  error.rs    ParseError + span-anchored diagnostics
  lib.rs      Public entry point
```

## Public API

- `parse(source, filename) -> Result<SourceFile, ParseError>`
- `parse_recovering(source, filename, options) -> ParseReport`
- `ParseCache` for stable content-hash file caching

## Token Notes

- `|>` (pipe-arrow) is lexed only as `PipeArrow`; `|` is always CSG union.
- Modelling declaration keywords are contextual and remain usable as local identifiers.
- Lexer token spellings are session-interned and all error-token spans end on UTF-8 boundaries.
- Grammar 1.0 bounds nesting and recovering diagnostics; see `GRAMMAR.md`.
- Strings must be terminated; unterminated literals now produce a parse error rather than silently truncating.
- Composite expression spans cover complete operands, not just the operator or postfix token.
- Empty argument lists (`f(,)` or trailing commas) produce a clean error rather than crashing.

## Testing

```bash
cargo test -p meshlang-parser --lib
cargo check -p meshlang-parser && cargo clippy -p meshlang-parser
```
