use meshlang_ast::{AstNode, span::Span};
use meshlang_parser::{
    ParseCache, ParseOptions, SourceRange, SubtreeRangeError, parse, parse_subtree,
    parse_subtree_json,
};
use serde_json::Value;

const SOURCE: &str = "param before = 1\n/// geometry documentation\ngeometry model { let body = Box(1.0, 1.0, 1.0) }\nparam after = 2\n";

fn geometry_range() -> SourceRange {
    let start = SOURCE.find("/// geometry").expect("documentation start");
    let end = SOURCE.find("\nparam after").expect("following declaration");
    SourceRange::new(start, end)
}

#[test]
fn subtree_preserves_absolute_spans_identity_comments_and_hash() {
    let range = geometry_range();
    let options = ParseOptions {
        file_id: 19,
        ..ParseOptions::default()
    };
    let first = parse_subtree(SOURCE, "subtree.mlg", range, options).expect("subtree parse");
    let repeated = parse_subtree(SOURCE, "subtree.mlg", range, options).expect("repeat parse");
    let whole = parse(SOURCE, "subtree.mlg").expect("whole parse");

    assert_eq!(
        first.source_file.span,
        Span::with_file(range.start, range.end, 19)
    );
    assert_eq!(first.source_file.metadata.identity, whole.metadata.identity);
    assert_eq!(first.source_file.declarations.len(), 1);
    assert!(first.diagnostics.is_empty());
    assert_eq!(first.source_file.comments.len(), 1);
    assert!(first.source_file.comments[0].target.is_some());
    assert_eq!(
        first.source_file.structural_hash().expect("first hash"),
        repeated.source_file.structural_hash().expect("repeat hash")
    );

    let declaration_span = first.source_file.declarations[0].span();
    assert!(declaration_span.start >= range.start);
    assert!(declaration_span.end <= range.end);
    assert_eq!(declaration_span.file_id, options.file_id);
}

#[test]
fn subtree_does_not_parse_outside_selected_range_and_reports_absolute_errors() {
    let start = SOURCE.find("param after").expect("after declaration");
    let end = SOURCE.len();
    let report = parse_subtree(
        SOURCE,
        "subtree.mlg",
        SourceRange::new(start, end),
        ParseOptions {
            file_id: 7,
            ..ParseOptions::default()
        },
    )
    .expect("tail subtree");
    assert_eq!(report.source_file.declarations.len(), 1);
    assert!(report.diagnostics.is_empty());

    let bad_source = "param before = 1\nparam broken = @\nparam after = 2";
    let bad_start = bad_source.find("param broken").unwrap();
    let bad_end = bad_source.find("\nparam after").unwrap();
    let bad = parse_subtree(
        bad_source,
        "bad.mlg",
        SourceRange::new(bad_start, bad_end),
        ParseOptions::default(),
    )
    .expect("recovering bad subtree");
    assert!(!bad.diagnostics.is_empty());
    assert!(
        bad.diagnostics
            .iter()
            .all(|diagnostic| diagnostic.span.start >= bad_start)
    );
}

#[test]
fn subtree_range_validation_rejects_bounds_and_split_utf8() {
    let source = "param räume = 1";
    assert!(matches!(
        parse_subtree(
            source,
            "invalid.mlg",
            SourceRange::new(0, source.len() + 1),
            ParseOptions::default(),
        ),
        Err(SubtreeRangeError::OutOfBounds { .. })
    ));
    assert!(matches!(
        parse_subtree(
            source,
            "invalid.mlg",
            SourceRange::new(8, source.len()),
            ParseOptions::default(),
        ),
        Err(SubtreeRangeError::NotUtf8Boundary { offset: 8 })
    ));
}

#[test]
fn subtree_json_and_cache_are_deterministic_and_option_sensitive() {
    let range = geometry_range();
    let options = ParseOptions::default();
    let direct = parse_subtree_json(SOURCE, "cache.mlg", range, options).expect("direct JSON");
    let mut cache = ParseCache::default();
    let first = cache
        .parse_subtree_json(SOURCE, "cache.mlg", range, options)
        .expect("first cached JSON");
    let second = cache
        .parse_subtree_json(SOURCE, "cache.mlg", range, options)
        .expect("cached JSON");
    assert_eq!(direct, first);
    assert_eq!(first, second);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.subtree_len(), 1);

    let mut limited = options;
    limited.max_diagnostics = 1;
    cache
        .parse_subtree(SOURCE, "cache.mlg", range, limited)
        .unwrap();
    assert_eq!(cache.stats().misses, 2);

    let envelope: Value = serde_json::from_str(&first).expect("subtree envelope");
    assert_eq!(envelope["parse_schema"], 1);
    assert_eq!(
        envelope["report"]["source_file"]["span"]["start"],
        range.start
    );
    assert!(cache.invalidate("cache.mlg"));
    assert!(cache.is_empty());
    assert_eq!(cache.subtree_len(), 0);
}
