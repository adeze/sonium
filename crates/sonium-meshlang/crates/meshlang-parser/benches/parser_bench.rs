use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use meshlang_parser::{
    DEFAULT_MAX_NESTING, ParseCache, ParseOptions, lex, parse, parse_recovering, parse_with_options,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

fn generated_program(declarations: usize) -> String {
    let mut source = String::with_capacity(declarations * 48);
    source.push_str("version 1\n");
    for index in 0..declarations {
        source.push_str(&format!("param value_{index} = {index}.0 m\n"));
    }
    source
}

fn benchmark_large_programs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("generated_large_program");
    for declarations in [1_000, 10_000] {
        let source = generated_program(declarations);
        group.bench_with_input(
            BenchmarkId::new("lex", declarations),
            &source,
            |bencher, source| bencher.iter(|| lex(source)),
        );
        group.bench_with_input(
            BenchmarkId::new("parse", declarations),
            &source,
            |bencher, source| bencher.iter(|| parse(source, "generated.mlg")),
        );
    }
    group.finish();

    let graph: Vec<_> = (0..100)
        .map(|file| {
            let mut source = format!("version 1\nimport \"file_{}.mlg\"\n", (file + 1) % 100);
            for declaration in 0..100 {
                source.push_str(&format!(
                    "const value_{file}_{declaration} = {declaration}\n"
                ));
            }
            (format!("file_{file}.mlg"), source)
        })
        .collect();
    let mut import_group = criterion.benchmark_group("generated_import_graph");
    import_group.bench_function("strict_parse_100_files", |bencher| {
        bencher.iter(|| {
            for (filename, source) in &graph {
                let _ = parse(source, filename);
            }
        });
    });
    let mut cache = ParseCache::default();
    for (filename, source) in &graph {
        cache.parse(source, filename, ParseOptions::default());
    }
    import_group.bench_function("warm_file_cache_100_files", |bencher| {
        bencher.iter(|| {
            for (filename, source) in &graph {
                let _ = cache.parse(source, filename, ParseOptions::default());
            }
        });
    });
    import_group.finish();
}

/// Counting allocator for the allocation survey: allocations and payload
/// bytes attributable to in-process parsing, excluding Cargo/rustc itself
/// (the audit's "excluding Cargo" clause). Sampled outside criterion iters;
/// deallocations are not subtracted, so bytes are cumulative allocated.
struct CountingAlloc;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn reset_alloc_counters() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn survey_allocations(label: &str, decls: usize, source: &str) {
    reset_alloc_counters();
    let parsed = parse(source, "alloc-survey.mlg").expect("survey source must parse");
    drop(parsed);
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    eprintln!(
        "PARSEALLOCS {label} decls={decls} allocs={allocs} bytes={bytes} bytes_per_decl={}",
        bytes / decls.max(1) as u64
    );
}

/// `depth` nested `if` blocks inside a geometry model (editor/AST stress shape).
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

/// Simulate an editor session: 32 keystrokes, each appending one declaration
/// (`append`) or rewriting one value (`modify`). One bench iter replays the
/// whole session; throughput is per keystroke.
fn editor_session(base_decls: usize, keystrokes: usize, modify: bool) -> Vec<String> {
    let mut source = generated_program(base_decls);
    let mut session = Vec::with_capacity(keystrokes);
    let mut previous = String::from("param value_0 = 0.0 m\n");
    for key in 0..keystrokes {
        if modify {
            let current = format!("param value_0 = {key}.0 m\n");
            source = source.replace(&previous, &current);
            previous = current;
        } else {
            source.push_str(&format!("param edit_{key} = {key}.0 m\n"));
        }
        session.push(source.clone());
    }
    session
}

fn benchmark_editor_trace(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("parser_editor_trace");
    for (kind, modify) in [("append", false), ("modify", true)] {
        let session = editor_session(200, 32, modify);
        group.throughput(criterion::Throughput::Elements(32));
        group.bench_function(format!("reparse_per_keystroke_{kind}"), |bencher| {
            bencher.iter(|| {
                for source in &session {
                    let _ = parse(source, "edited.mlg");
                }
            });
        });
        // Replay once up front so the stats line describes a real session
        // (mutated sources miss the content-keyed cache and reparse fully).
        let mut warm_cache = ParseCache::default();
        for source in &session {
            warm_cache.invalidate("edited.mlg");
            let _ = warm_cache.parse(source, "edited.mlg", ParseOptions::default());
        }
        let stats = warm_cache.stats();
        eprintln!(
            "EDITTRACE {kind} cache_hits={} cache_misses={}",
            stats.hits, stats.misses
        );
        let mut cache = ParseCache::default();
        group.bench_function(format!("file_cache_per_keystroke_{kind}"), |bencher| {
            bencher.iter(|| {
                for source in &session {
                    cache.invalidate("edited.mlg");
                    let _ = cache.parse(source, "edited.mlg", ParseOptions::default());
                }
            });
        });
    }
    group.finish();
}

fn benchmark_nesting_depth_sweep(criterion: &mut Criterion) {
    // Depths past DEFAULT_MAX_NESTING take the typed depth-error path, not
    // the success path: the sweep documents the cliff honestly instead of
    // mixing paths silently.
    let mut group = criterion.benchmark_group("parser_nesting_depth");
    for depth in [16usize, 64, 128, 256, 1024] {
        let source = nested_ifs(depth);
        let probe = parse_with_options(&source, "deep.mlg", ParseOptions::default());
        eprintln!(
            "NESTDEPTH depth={depth} limit={DEFAULT_MAX_NESTING} parses_ok={} bytes={}",
            probe.is_ok(),
            source.len()
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &source,
            |bencher, source| {
                bencher.iter(|| {
                    let _ = parse_with_options(source, "deep.mlg", ParseOptions::default());
                });
            },
        );
    }
    group.finish();
}

fn benchmark_unicode_sweep(criterion: &mut Criterion) {
    // Combining-character runs in comments (proven-legal shape) plus a CJK
    // comment payload, at three scales; the recovery entry means a lexer
    // surprise can never panic the bench.
    let mut group = criterion.benchmark_group("parser_unicode");
    for combining in [1_000usize, 10_000, 100_000] {
        let mut source = String::from("geometry model { let body = Box(1.0, 1.0, 1.0) }\n");
        source.push_str("// observable \u{4e2d}\u{6587} payload\n");
        source.push_str(&format!("//{}\n", "\u{301}".repeat(combining)));
        let diagnostics = parse_recovering(&source, "unicode.mlg", ParseOptions::default())
            .diagnostics
            .len();
        eprintln!(
            "UNICODE combining={combining} bytes={} diagnostics={diagnostics}",
            source.len()
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(combining),
            &source,
            |bencher, source| {
                bencher.iter(|| {
                    let _ = parse_recovering(source, "unicode.mlg", ParseOptions::default());
                });
            },
        );
    }
    group.finish();
}

fn benchmark_recovery_sweep(criterion: &mut Criterion) {
    // Malformed shapes at two scales: truncated nesting, garbage parens,
    // unterminated string/comment. Recovery must terminate bounded with a
    // capped diagnostic set; the sweep times that contract.
    type ShapeBuilder = Box<dyn Fn(usize) -> String>;
    let shapes: [(&str, ShapeBuilder); 3] = [
        ("truncated_nesting", Box::new(|scale| nested_ifs(scale * 4))),
        (
            "garbage_parens",
            Box::new(|scale| {
                format!(
                    "geometry model {{ let body = Box(1.0, }} garbage ((( {}\n",
                    "( ".repeat(scale)
                )
            }),
        ),
        (
            "unterminated_string_comment",
            Box::new(|scale| {
                format!(
                    "geometry model {{ let name = \"open string {}\n/* unterminated {}\n",
                    "x".repeat(scale),
                    "y".repeat(scale)
                )
            }),
        ),
    ];
    let mut group = criterion.benchmark_group("parser_recovery");
    for (name, build) in shapes {
        for scale in [1usize, 100] {
            let source = build(scale);
            let report = parse_recovering(&source, "broken.mlg", ParseOptions::default());
            eprintln!(
                "RECOVERY shape={name} scale={scale} bytes={} diagnostics={} success={}",
                source.len(),
                report.diagnostics.len(),
                report.is_success()
            );
            group.bench_with_input(BenchmarkId::new(name, scale), &source, |bencher, source| {
                bencher.iter(|| {
                    let _ = parse_recovering(source, "broken.mlg", ParseOptions::default());
                });
            });
        }
    }
    group.finish();
}

fn benchmark_allocation_survey(criterion: &mut Criterion) {
    // Allocation survey runs outside iters (measured, not timed): the group
    // still runs a timing bench so the series stays comparable.
    let mut group = criterion.benchmark_group("parser_allocations");
    for decls in [1_000usize, 10_000] {
        let source = generated_program(decls);
        survey_allocations("strict_parse", decls, &source);
        group.bench_with_input(
            BenchmarkId::from_parameter(decls),
            &source,
            |bencher, source| {
                bencher.iter(|| {
                    let _ = parse(source, "generated.mlg");
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_large_programs,
    benchmark_editor_trace,
    benchmark_nesting_depth_sweep,
    benchmark_unicode_sweep,
    benchmark_recovery_sweep,
    benchmark_allocation_survey
);
criterion_main!(benches);
