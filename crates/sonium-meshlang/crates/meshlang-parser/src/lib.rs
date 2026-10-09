//! MeshLang Parser
//!
//! Provides lexing and parsing of `.mlg` source files into the MeshLang AST.
//!
//! # Usage
//! ```ignore
//! use meshlang_parser::parse;
//!
//! let source = r#"
//!     param freq = 1000.0 Hz
//!     geometry box {
//!         let outer = Box(0.3 m, 0.4 m, 0.25 m)
//!     }
//! "#;
//!
//! let ast = parse(source, "example.mlg").unwrap();
//! ```

pub mod error;
/// Normative precedence and token contract used by parser/tooling tests.
pub mod grammar;
pub mod lexer;
pub mod parser;

use std::collections::HashMap;

use error::ParseError;
use lexer::Lexer;
use meshlang_ast::{
    AstNode, Comment, ProvenanceOperation, ProvenanceRecord, SourceFile, SourceMetadata,
    SourceReference, StructuralHash, span::Span,
};
use parser::Parser;
pub use parser::{DEFAULT_MAX_NESTING, GRAMMAR_VERSION, LANGUAGE_VERSION, ParseReport};
use serde::Serialize;
use thiserror::Error;

pub const DEFAULT_MAX_DIAGNOSTICS: usize = 32;
/// Default source-size guard protecting lexer/parser allocations.
pub const DEFAULT_MAX_SOURCE_BYTES: usize = 8 * 1024 * 1024;
/// Version of the ownership-neutral JSON envelope used by editor/WASM tools.
pub const PARSE_JSON_SCHEMA_VERSION: u32 = 1;
/// Versioned JSON contract for editor grammar parity metadata.
pub const GRAMMAR_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Resource and identity limits for a single parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ParseOptions {
    pub file_id: u32,
    pub max_diagnostics: usize,
    pub max_nesting: usize,
    /// Maximum source size accepted by the lexer/parser in bytes.
    pub max_source_bytes: usize,
}

/// A byte range into an original MeshLang source file.
///
/// Ranges are deliberately byte-based because AST and diagnostic spans use
/// byte offsets. [`parse_subtree`] validates both bounds and UTF-8 character
/// boundaries before borrowing the selected slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SourceRange {
    pub start: usize,
    pub end: usize,
}

impl SourceRange {
    pub const fn new(start: usize, end: usize) -> Self {
        // `const` constructors cannot panic or validate; inverted ranges
        // report `len() == 0` and are rejected later by `validate`.
        debug_assert!(start <= end);
        Self { start, end }
    }

    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    fn validate(self, source: &str) -> Result<(), SubtreeRangeError> {
        if self.start > self.end || self.end > source.len() {
            return Err(SubtreeRangeError::OutOfBounds {
                start: self.start,
                end: self.end,
                source_len: source.len(),
            });
        }
        if !source.is_char_boundary(self.start) {
            return Err(SubtreeRangeError::NotUtf8Boundary { offset: self.start });
        }
        if !source.is_char_boundary(self.end) {
            return Err(SubtreeRangeError::NotUtf8Boundary { offset: self.end });
        }
        Ok(())
    }

    fn span(self, file_id: u32) -> Span {
        Span::with_file(self.start, self.end, file_id)
    }
}

/// Validation failures returned before subtree lexing starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, Serialize)]
pub enum SubtreeRangeError {
    #[error("subtree range {start}..{end} is outside the {source_len}-byte source")]
    OutOfBounds {
        start: usize,
        end: usize,
        source_len: usize,
    },
    #[error("subtree range boundary {offset} is not a UTF-8 character boundary")]
    NotUtf8Boundary { offset: usize },
}

/// Errors from subtree JSON serialization, including range validation.
#[derive(Debug, Error)]
pub enum SubtreeJsonError {
    #[error(transparent)]
    Range(#[from] SubtreeRangeError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            file_id: 0,
            max_diagnostics: DEFAULT_MAX_DIAGNOSTICS,
            max_nesting: DEFAULT_MAX_NESTING,
            max_source_bytes: DEFAULT_MAX_SOURCE_BYTES,
        }
    }
}

/// Parse a MeshLang source string into an AST.
pub fn parse(source: &str, filename: &str) -> Result<SourceFile, ParseError> {
    parse_with_options(source, filename, ParseOptions::default())
}

/// Strict parsing with explicit file identity and resource limits.
pub fn parse_with_options(
    source: &str,
    filename: &str,
    options: ParseOptions,
) -> Result<SourceFile, ParseError> {
    if source.len() > options.max_source_bytes {
        return Err(source_too_large_error(source, filename, options));
    }
    let mut lexer = Lexer::new(source, options.file_id);
    let tokens = lexer.tokenize();
    let comments = lexer.take_comments();
    let mut parser = Parser::with_max_nesting(tokens, options.file_id, options.max_nesting);
    parser.set_source_name(filename);
    parser
        .parse_source_file()
        .map(|source_file| finalize_source_file(source_file, source, filename, comments))
}

/// Recovering parse for editors, language servers, and batch diagnostics.
pub fn parse_recovering(source: &str, filename: &str, options: ParseOptions) -> ParseReport {
    if source.len() > options.max_source_bytes {
        return ParseReport {
            source_file: limited_source_file(source, filename, options.file_id),
            diagnostics: vec![source_too_large_error(source, filename, options)],
        };
    }
    let mut lexer = Lexer::new(source, options.file_id);
    let tokens = lexer.tokenize();
    let comments = lexer.take_comments();
    let mut parser = Parser::with_max_nesting(tokens, options.file_id, options.max_nesting);
    parser.set_source_name(filename);
    let mut report = parser.parse_source_file_recovering(options.max_diagnostics);
    report.source_file = finalize_source_file(report.source_file, source, filename, comments);
    report
}

/// Parse only a checked byte range of an original source file.
///
/// This is a bounded editor primitive, not dependency-aware incremental
/// compilation: imports, symbols, and declarations outside `range` are not
/// available to the parser. The borrowed range is lexed directly, while all
/// token, comment, diagnostic, AST, and provenance spans retain absolute
/// offsets into `source` and the source identity hashes the complete file.
///
/// Cost note: the oversize branch checks `range.len()`, not `source.len()` —
/// a small range of a huge file still parses while holding the full source
/// string and hashing it for identity, i.e. subtree parsing costs O(file)
/// memory, not O(range).
pub fn parse_subtree(
    source: &str,
    filename: &str,
    range: SourceRange,
    options: ParseOptions,
) -> Result<ParseReport, SubtreeRangeError> {
    range.validate(source)?;

    let source_file = if range.len() > options.max_source_bytes {
        let mut source_file = SourceFile::new(Vec::new(), range.span(options.file_id))
            .with_source_metadata(SourceMetadata::for_source(filename, source));
        let diagnostic = source_too_large_error_for_range(filename, options, range);
        source_file = finalize_source_file(source_file, source, filename, Vec::new());
        return Ok(ParseReport {
            source_file,
            diagnostics: vec![diagnostic],
        });
    } else {
        let selected = &source[range.start..range.end];
        let mut lexer = Lexer::with_offset(selected, options.file_id, range.start);
        let tokens = lexer.tokenize();
        let comments = lexer.take_comments();
        let mut parser = Parser::with_max_nesting(tokens, options.file_id, options.max_nesting);
        parser.set_source_name(filename);
        let mut report = parser.parse_source_file_recovering(options.max_diagnostics);
        // Empty and comment-only ranges still identify the exact editor span.
        report.source_file.span = range.span(options.file_id);
        report.source_file = finalize_source_file(report.source_file, source, filename, comments);
        report
    };

    Ok(source_file)
}

/// Serialize a bounded subtree parse through the same versioned envelope as a
/// complete-file parse.
pub fn parse_subtree_json(
    source: &str,
    filename: &str,
    range: SourceRange,
    options: ParseOptions,
) -> Result<String, SubtreeJsonError> {
    Ok(parse_report_json(parse_subtree(
        source, filename, range, options,
    )?)?)
}

#[derive(Debug, Serialize)]
struct ParseJsonEnvelope {
    parse_schema: u32,
    grammar_version: &'static str,
    diagnostic_schema: u32,
    report: ParseReport,
}

/// Serialize a recovering parse into a stable, versioned JSON envelope for
/// editors, browser previews, and non-Rust tooling. The envelope keeps the
/// AST and typed diagnostics together while allowing either payload to evolve
/// independently in future schema versions.
pub fn parse_json(
    source: &str,
    filename: &str,
    options: ParseOptions,
) -> Result<String, serde_json::Error> {
    parse_report_json(parse_recovering(source, filename, options))
}

/// Serialize an already computed recovering report into the same stable,
/// versioned envelope used by [`parse_json`]. This is the cache/editor bridge:
/// callers can parse once, retain the typed report, and serialize it without
/// repeating lexing or recovery.
pub fn parse_report_json(report: ParseReport) -> Result<String, serde_json::Error> {
    serde_json::to_string(&ParseJsonEnvelope {
        parse_schema: PARSE_JSON_SCHEMA_VERSION,
        grammar_version: GRAMMAR_VERSION,
        diagnostic_schema: error::DIAGNOSTIC_SCHEMA_VERSION,
        report,
    })
}

/// Serialize the checked-in parser/editor grammar parity contract.
///
/// The tree-sitter source is validated before it is exposed, so editor
/// clients receive an explicit failure rather than stale precedence metadata.
pub fn grammar_contract_json() -> Result<String, String> {
    let tree_sitter_source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tree-sitter-meshlang/grammar.js"
    ));
    grammar::validate_tree_sitter_grammar(tree_sitter_source).map_err(|error| error.to_string())?;
    serde_json::to_string(&serde_json::json!({
        "schema_version": GRAMMAR_CONTRACT_SCHEMA_VERSION,
        "grammar_version": GRAMMAR_VERSION,
        "operator_spellings": grammar::OPERATOR_SPELLINGS,
        "precedence_rows": grammar::PRECEDENCE_ROWS,
    }))
    .map_err(|error| error.to_string())
}

fn finalize_source_file(
    source_file: SourceFile,
    source: &str,
    filename: &str,
    comments: Vec<Comment>,
) -> SourceFile {
    let metadata = SourceMetadata::for_source(filename, source);
    let identity = metadata.identity.clone();
    let mut source_file = source_file.with_source_metadata(metadata);
    let source_id = source_file.node_id(&identity);
    source_file.provenance.push(ProvenanceRecord {
        node: source_id,
        sources: vec![SourceReference {
            source: identity.id,
            node: source_id,
            span: source_file.span,
        }],
        operation: ProvenanceOperation::Parsed,
    });
    for declaration in &source_file.declarations {
        let node = declaration.node_id(&identity);
        source_file.provenance.push(ProvenanceRecord {
            node,
            sources: vec![SourceReference {
                source: identity.id,
                node,
                span: declaration.span(),
            }],
            operation: ProvenanceOperation::Parsed,
        });
    }
    // Doc-comment attachment in O((D+C) log) instead of O(D·C): editor
    // files reparse on every keystroke, so the per-comment declaration
    // scans used to dominate recovery cost. Declaration spans are sorted
    // once; nesting is resolved per declaration over comment-span order
    // and successors by binary search. Tie order keeps declaration order,
    // exactly matching the old `min_by_key` scan.
    let decl_spans: Vec<(u32, usize, usize)> = source_file
        .declarations
        .iter()
        .map(|declaration| {
            let span = declaration.span();
            (span.file_id, span.start, span.end)
        })
        .collect();
    let mut succ_order: Vec<usize> = (0..decl_spans.len()).collect();
    succ_order.sort_by_key(|&index| (decl_spans[index].0, decl_spans[index].1, index));
    let mut comment_order: Vec<usize> = (0..comments.len()).collect();
    comment_order.sort_by_key(|&index| {
        (
            comments[index].span.file_id,
            comments[index].span.start,
            index,
        )
    });
    let mut nested = vec![false; comments.len()];
    for &(file_id, start, end) in &decl_spans {
        let lower = comment_order.partition_point(|&index| {
            let span = &comments[index].span;
            (span.file_id, span.start) < (file_id, start)
        });
        let upper = comment_order.partition_point(|&index| {
            let span = &comments[index].span;
            (span.file_id, span.start) <= (file_id, end)
        });
        for &index in &comment_order[lower..upper] {
            if comments[index].span.end <= end {
                nested[index] = true;
            }
        }
    }
    let successor = |comment: &Comment| -> Option<usize> {
        let position = succ_order.partition_point(|&index| {
            (decl_spans[index].0, decl_spans[index].1) < (comment.span.file_id, comment.span.end)
        });
        succ_order
            .get(position)
            .copied()
            .filter(|&index| decl_spans[index].0 == comment.span.file_id)
    };
    let comments: Vec<_> = comments
        .into_iter()
        .enumerate()
        .map(|(index, comment)| {
            if !comment.is_doc() || nested[index] {
                return comment;
            }
            successor(&comment).map_or(comment.clone(), |declaration| {
                comment.documenting(source_file.declarations[declaration].node_id(&identity))
            })
        })
        .collect();
    source_file.extend_comments(comments);
    source_file
}

fn source_too_large_error(source: &str, filename: &str, options: ParseOptions) -> ParseError {
    ParseError::new(
        error::DiagnosticCode::SourceTooLarge,
        format!(
            "source is {} bytes; maximum is {} bytes",
            source.len(),
            options.max_source_bytes
        ),
        Span::with_file(0, source.len(), options.file_id),
    )
    .with_source_name(filename)
}

fn source_too_large_error_for_range(
    filename: &str,
    options: ParseOptions,
    range: SourceRange,
) -> ParseError {
    ParseError::new(
        error::DiagnosticCode::SourceTooLarge,
        format!(
            "subtree is {} bytes; maximum is {} bytes",
            range.len(),
            options.max_source_bytes
        ),
        range.span(options.file_id),
    )
    .with_source_name(filename)
}

fn limited_source_file(source: &str, filename: &str, file_id: u32) -> SourceFile {
    SourceFile::new(Vec::new(), Span::with_file(0, source.len(), file_id))
        .with_source_metadata(SourceMetadata::for_source(filename, source))
}

/// Lex a source string into tokens (useful for testing / tooling).
pub fn lex(source: &str) -> Vec<lexer::Token> {
    lex_with_file_id(source, 0)
}

pub fn lex_with_file_id(source: &str, file_id: u32) -> Vec<lexer::Token> {
    let mut lexer = Lexer::new(source, file_id);
    lexer.tokenize()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Bulk evictions performed when a capacity cap was reached.
    pub evictions: u64,
}

/// Maximum cached files. Past this bound the file map is bulk-evicted so an
/// editor opening many files cannot grow the cache without limit.
pub const MAX_CACHED_FILES: usize = 64;
/// Maximum cached subtree ranges. Past this bound the subtree map is
/// bulk-evicted so range scrubbing cannot leak one cloned report per range.
pub const MAX_CACHED_SUBTREES: usize = 256;

#[derive(Debug, Clone)]
struct CacheEntry {
    content_hash: StructuralHash,
    options: ParseOptions,
    report: ParseReport,
}

/// Deterministic parse cache for complete files and bounded editor subtrees.
///
/// Subtree entries are keyed by filename, original-file content hash, source
/// range, and parse options. They intentionally do not attempt dependency or
/// symbol invalidation; callers needing that behavior must use the compiler's
/// incremental layer.
#[derive(Debug, Default)]
pub struct ParseCache {
    files: HashMap<String, CacheEntry>,
    subtrees: HashMap<(String, SourceRange), CacheEntry>,
    stats: CacheStats,
}

impl ParseCache {
    pub fn parse(&mut self, source: &str, filename: &str, options: ParseOptions) -> ParseReport {
        let content_hash = StructuralHash::from_bytes(source.as_bytes());
        if let Some(entry) = self.files.get(filename)
            && entry.content_hash == content_hash
            && entry.options == options
        {
            self.stats.hits += 1;
            return entry.report.clone();
        }

        self.stats.misses += 1;
        let report = parse_recovering(source, filename, options);
        if self.files.len() >= MAX_CACHED_FILES && !self.files.contains_key(filename) {
            self.files.clear();
            self.stats.evictions += 1;
        }
        self.files.insert(
            filename.to_string(),
            CacheEntry {
                content_hash,
                options,
                report: report.clone(),
            },
        );
        report
    }

    /// Parse through the deterministic file cache and return the stable JSON
    /// envelope used by editor/WASM clients.
    pub fn parse_json(
        &mut self,
        source: &str,
        filename: &str,
        options: ParseOptions,
    ) -> Result<String, serde_json::Error> {
        parse_report_json(self.parse(source, filename, options))
    }

    /// Parse and cache one checked source range while retaining original-file
    /// spans and source identity.
    pub fn parse_subtree(
        &mut self,
        source: &str,
        filename: &str,
        range: SourceRange,
        options: ParseOptions,
    ) -> Result<ParseReport, SubtreeRangeError> {
        range.validate(source)?;
        let content_hash = StructuralHash::from_bytes(source.as_bytes());
        let key = (filename.to_string(), range);
        if let Some(entry) = self.subtrees.get(&key)
            && entry.content_hash == content_hash
            && entry.options == options
        {
            self.stats.hits += 1;
            return Ok(entry.report.clone());
        }

        self.stats.misses += 1;
        let report = parse_subtree(source, filename, range, options)?;
        if self.subtrees.len() >= MAX_CACHED_SUBTREES && !self.subtrees.contains_key(&key) {
            self.subtrees.clear();
            self.stats.evictions += 1;
        }
        self.subtrees.insert(
            key,
            CacheEntry {
                content_hash,
                options,
                report: report.clone(),
            },
        );
        Ok(report)
    }

    /// Serialize a cached bounded subtree using the stable parse envelope.
    pub fn parse_subtree_json(
        &mut self,
        source: &str,
        filename: &str,
        range: SourceRange,
        options: ParseOptions,
    ) -> Result<String, SubtreeJsonError> {
        Ok(parse_report_json(
            self.parse_subtree(source, filename, range, options)?,
        )?)
    }

    pub fn invalidate(&mut self, filename: &str) -> bool {
        let removed_file = self.files.remove(filename).is_some();
        let before = self.subtrees.len();
        self.subtrees.retain(|(name, _), _| name != filename);
        removed_file || self.subtrees.len() != before
    }

    pub fn clear(&mut self) {
        self.files.clear();
        self.subtrees.clear();
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Number of cached bounded source ranges.
    pub fn subtree_len(&self) -> usize {
        self.subtrees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.subtrees.is_empty()
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }
}

/// Browser/editor API. Returning JSON avoids JS object ownership ambiguities
/// while preserving the same versioned AST and typed diagnostics as native.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = parseMeshLang)]
pub fn parse_wasm(source: &str, filename: &str) -> Result<String, wasm_bindgen::JsValue> {
    parse_json(source, filename, ParseOptions::default())
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Browser/editor API for the validated Rust/tree-sitter grammar contract.
#[cfg(all(target_arch = "wasm32", feature = "wasm-exports"))]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = validateMeshLangGrammar)]
pub fn validate_grammar_wasm() -> Result<String, wasm_bindgen::JsValue> {
    grammar_contract_json().map_err(|error| wasm_bindgen::JsValue::from_str(&error))
}
