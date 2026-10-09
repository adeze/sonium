//! Adversarial parser inputs: nesting depth, oversize sources, pathological
//! comments and Unicode, and the strict-vs-recovered contract.
//!
//! The parser is deliberately sequential per file; these tests bound the
//! worst case (bounded diagnostics, no hangs, no panics) rather than
//! measuring throughput on flat generated corpora.

use meshlang_parser::error::DiagnosticCode;
use meshlang_parser::{ParseOptions, parse_recovering, parse_with_options};

fn options() -> ParseOptions {
    ParseOptions {
        file_id: 0,
        max_diagnostics: 32,
        max_nesting: 16,
        max_source_bytes: 1024 * 1024,
    }
}

/// `depth` nested `if` blocks inside a geometry model.
fn nested_ifs(depth: usize) -> String {
    let mut source = String::from("geometry model { ");
    for _ in 0..depth {
        source.push_str("if enabled { ");
    }
    source.push_str("let body = Box(1.0, 1.0, 1.0) ");
    for _ in 0..depth {
        source.push_str("} ");
    }
    source.push_str("}\n");
    source
}

#[test]
fn deep_nesting_reports_nesting_limit_and_stays_bounded() {
    let source = nested_ifs(256);
    let report = parse_recovering(&source, "deep.mlg", options());
    // Must terminate with a bounded diagnostic set, not a stack overflow or
    // runaway recovery.
    assert!(!report.is_success());
    assert!(report.diagnostics.len() <= options().max_diagnostics);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::NestingLimit),
        "expected a NestingLimit diagnostic, got: {:?}",
        report
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code)
            .collect::<Vec<_>>()
    );
}

#[test]
fn nesting_at_default_limits_still_terminates() {
    let defaults = ParseOptions {
        max_diagnostics: 16,
        ..ParseOptions::default()
    };
    let source = nested_ifs(10_000);
    let report = parse_recovering(&source, "very_deep.mlg", defaults);
    assert!(!report.is_success());
    assert!(report.diagnostics.len() <= defaults.max_diagnostics);
}

#[test]
fn oversize_source_is_rejected_without_lexing() {
    let mut tight = options();
    tight.max_source_bytes = 16;
    let source = "geometry model { let body = Box(1.0, 1.0, 1.0) }\n";
    assert!(source.len() > tight.max_source_bytes);

    let error = parse_with_options(source, "big.mlg", tight)
        .expect_err("oversize source must fail strict parsing");
    assert_eq!(error.code, DiagnosticCode::SourceTooLarge);

    let report = parse_recovering(source, "big.mlg", tight);
    assert!(!report.is_success());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::SourceTooLarge)
    );
}

#[test]
fn pathological_comments_and_unicode_terminate_bounded() {
    let giant_line_comment = format!("//{}\n", "x".repeat(200_000));
    let mut source = String::from("geometry model { let body = Box(1.0, 1.0, 1.0) }\n");
    source.push_str(&giant_line_comment);
    // Unterminated block comment plus combining characters.
    source.push_str("/* unterminated ");
    source.push_str(&"\u{301}".repeat(10_000));

    let report = parse_recovering(&source, "pathological.mlg", options());
    assert!(report.diagnostics.len() <= options().max_diagnostics);
    for diagnostic in &report.diagnostics {
        assert!(diagnostic.span.start <= diagnostic.span.end);
        assert!(diagnostic.span.end <= source.len());
        assert!(source.is_char_boundary(diagnostic.span.start));
        assert!(source.is_char_boundary(diagnostic.span.end));
    }
}

#[test]
fn strict_parse_and_recovered_preview_disagree_on_malformed_input() {
    // Consumers must keep this distinction: strict parsing fails while
    // recovery returns a partial preview with diagnostics. A recovered
    // report must never be mistaken for a successful strict parse.
    let source = "geometry model { let body = Box(1.0, } garbage (((";
    let strict = parse_with_options(source, "broken.mlg", options());
    assert!(strict.is_err(), "malformed input must fail strict parsing");

    let report = parse_recovering(source, "broken.mlg", options());
    assert!(!report.is_success());
    assert!(!report.diagnostics.is_empty());
    assert!(report.diagnostics.len() <= options().max_diagnostics);
}
