use std::panic::{AssertUnwindSafe, catch_unwind};

use meshlang_parser::{ParseOptions, parse_recovering, parse_report_json};

const MAX_DIAGNOSTICS: usize = 24;
const MAX_NESTING: usize = 48;
const MAX_SOURCE_BYTES: usize = 4096;

fn assert_span(span: meshlang_ast::span::Span, source_len: usize, file_id: u32) {
    assert_eq!(span.file_id, file_id);
    assert!(span.start <= span.end);
    assert!(span.end <= source_len);
}

fn generated_bytes(seed: u64, length: usize) -> Vec<u8> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            (state as u8).wrapping_add(0x20)
        })
        .collect()
}

fn assert_recovery_invariants(source: &str, filename: &str, options: ParseOptions) {
    let first = catch_unwind(AssertUnwindSafe(|| {
        parse_recovering(source, filename, options)
    }))
    .expect("recovering parser must not panic");
    assert!(first.diagnostics.len() <= options.max_diagnostics);
    assert_span(first.source_file.span, source.len(), options.file_id);
    assert!(first.source_file.structural_hash().is_ok());
    for diagnostic in &first.diagnostics {
        assert_span(diagnostic.span, source.len(), options.file_id);
    }
    for comment in &first.source_file.comments {
        assert_span(comment.span, source.len(), options.file_id);
    }

    let second = catch_unwind(AssertUnwindSafe(|| {
        parse_recovering(source, filename, options)
    }))
    .expect("repeated recovering parser must not panic");
    assert_eq!(
        parse_report_json(first).expect("first report JSON"),
        parse_report_json(second).expect("second report JSON"),
        "recovery output must be deterministic"
    );
}

/// Deterministic bounded fuzz-smoke corpus for parser totality and recovery.
/// The libFuzzer target remains available for sustained campaigns; this corpus
/// is intentionally finite so release QA can exercise the same invariants on
/// every CPU run without a time-based fuzzing budget.
pub fn deterministic_recovery_fuzz_corpus() {
    let options = ParseOptions {
        file_id: 173,
        max_diagnostics: MAX_DIAGNOSTICS,
        max_nesting: MAX_NESTING,
        max_source_bytes: MAX_SOURCE_BYTES,
    };
    let seeds = [
        "geometry model { let body = Box(1.0, 1.0, 1.0) }",
        "param width = 1.0\ngeometry model { let body = Box(width, 1.0, 1.0) }",
        "geometry model { let body = Box(1.0, 1.0, 1.0) } // trailing",
        "geometry model { let body = \"unterminated }",
    ];
    for (index, seed) in seeds.iter().enumerate() {
        assert_recovery_invariants(seed, &format!("seed-{index}.mlg"), options);
    }
    for index in 0..96_u64 {
        let bytes = generated_bytes(0x9e37_79b9_7f4a_7c15 ^ index, (index as usize * 37) % 512);
        let arbitrary = String::from_utf8_lossy(&bytes);
        let source =
            format!("geometry g{index} {{ let x = Box({index}.0, 1.0, 1.0) {arbitrary} }}");
        assert_recovery_invariants(&source, &format!("generated-{index}.mlg"), options);
    }
}

#[test]
fn deterministic_recovery_fuzz_corpus_is_bounded_and_stable() {
    deterministic_recovery_fuzz_corpus();
}
