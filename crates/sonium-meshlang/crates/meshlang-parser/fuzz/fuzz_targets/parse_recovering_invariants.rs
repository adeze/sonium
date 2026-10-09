#![no_main]

use libfuzzer_sys::fuzz_target;
use meshlang_parser::{parse_recovering, parse_report_json, ParseOptions};

fuzz_target!(|bytes: &[u8]| {
    // Keep the fuzz target aligned with the parser's bounded-release contract;
    // larger inputs are covered by the explicit source-size rejection path.
    if bytes.len() > 4096 {
        return;
    }
    let source = String::from_utf8_lossy(bytes);
    let options = ParseOptions {
        file_id: 211,
        max_diagnostics: 24,
        max_nesting: 48,
        max_source_bytes: 4096,
    };
    let first = parse_recovering(&source, "fuzz-invariants.mlg", options);
    assert!(first.diagnostics.len() <= options.max_diagnostics);
    assert!(first.source_file.structural_hash().is_ok());
    assert!(first
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.span.start <= diagnostic.span.end));
    assert!(first
        .source_file
        .comments
        .iter()
        .all(|comment| comment.span.start <= comment.span.end));
    let first_json = parse_report_json(first).expect("serialize first fuzz report");
    let second = parse_recovering(&source, "fuzz-invariants.mlg", options);
    assert_eq!(
        first_json,
        parse_report_json(second).expect("serialize second fuzz report")
    );
});
