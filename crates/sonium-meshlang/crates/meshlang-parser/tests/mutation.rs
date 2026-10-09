use std::panic::{AssertUnwindSafe, catch_unwind};

use meshlang_ast::AstNode;
use meshlang_ast::span::Span;
use meshlang_parser::{ParseOptions, parse_recovering};

const SEEDS: &[&str] = &[
    "geometry model { let body = Box(1.0, 1.0, 1.0) }",
    "param width = 1.0\ngeometry model { let body = Box(width, 1.0, 1.0) }",
    "geometry frame(width: Float = 1.0) -> (body: Solid, inlet: Surface) { let body = Box(width, 1.0, 1.0) let inlet = Rectangle(1.0, 1.0) }",
    "geometry model { if enabled { let body = Box(1.0, 1.0, 1.0) } else { let body = Sphere(r: 0.5) } }",
    "import \"stdlib/shapes.mlg\"\ngeometry model { let body = RoundedBox(1.0, 1.0, 1.0, radius: 0.1) }",
    "geometry model { let shell = Box(1.0, 1.0, 1.0) |> fillet(radius: 0.05) }\nmesh controls { size = 0.1 }",
    "geometry frame(width: Float = 1.0) -> (body: Solid, port: Surface) { let body = Box(width, 1.0, 1.0) let port = body.faces.top }",
    "param label = \"µ\u{301}\"\ngeometry model { let body = Sphere(r: 0.5) } // comment",
];

const MUTATIONS_PER_SEED: usize = 32;

fn mutate(seed: &str, mutation: usize) -> String {
    let mut chars = seed.chars().collect::<Vec<_>>();
    let position = mutation % (chars.len() + 1);
    match mutation % 4 {
        0 if !chars.is_empty() => {
            let index = position % chars.len();
            chars[index] = ['?', '|', '}', '(', ':'][mutation % 5];
        }
        1 => chars.insert(position, [';', ',', ')', '='][mutation % 4]),
        2 if !chars.is_empty() => {
            chars.remove(position % chars.len());
        }
        _ => chars.insert(position, '\u{0301}'),
    }
    chars.into_iter().collect()
}

fn assert_span(span: Span, source: &str, file_id: u32) {
    assert_eq!(span.file_id, file_id);
    assert!(span.start <= span.end);
    assert!(span.end <= source.len());
    assert!(source.is_char_boundary(span.start));
    assert!(source.is_char_boundary(span.end));
}

fn assert_bounded_report(
    source: &str,
    report: &meshlang_parser::ParseReport,
    options: ParseOptions,
) {
    assert!(report.diagnostics.len() <= options.max_diagnostics);
    assert_span(report.source_file.span, source, options.file_id);
    for diagnostic in &report.diagnostics {
        assert_span(diagnostic.span, source, options.file_id);
    }
    for declaration in &report.source_file.declarations {
        assert_span(declaration.span(), source, options.file_id);
    }
    for comment in &report.source_file.comments {
        assert_span(comment.span, source, options.file_id);
    }
    assert!(report.source_file.structural_hash().is_ok());
}

pub fn deterministic_mutation_corpus_is_bounded_and_stable() {
    let options = ParseOptions {
        file_id: 91,
        max_diagnostics: 12,
        max_nesting: 32,
        max_source_bytes: 4096,
    };
    for (seed_index, seed) in SEEDS.iter().enumerate() {
        for mutation in 0..MUTATIONS_PER_SEED {
            let source = mutate(seed, mutation + seed_index * 17);
            let first = catch_unwind(AssertUnwindSafe(|| {
                parse_recovering(&source, "mutation.mlg", options)
            }))
            .expect("deterministic mutation must not panic");
            assert_bounded_report(&source, &first, options);
            let second = parse_recovering(&source, "mutation.mlg", options);
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap(),
                "mutation result must be deterministic for seed {seed_index}, mutation {mutation}"
            );
        }
    }
}

#[test]
fn deterministic_mutation_corpus_is_bounded_and_stable_test() {
    deterministic_mutation_corpus_is_bounded_and_stable();
}
