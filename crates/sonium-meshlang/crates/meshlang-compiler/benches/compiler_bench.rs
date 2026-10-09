use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use meshlang_compiler::{preview_canonical_ir, CompileOptions, IncrementalCompiler};
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use tempfile::TempDir;

fn large_fragment_array(entities: usize) -> String {
    let mut source = String::with_capacity(entities * 96);
    source.push_str("geometry Main {\n  let base = Box(10.0, 1.0, 1.0)\n");
    for index in 0..entities {
        writeln!(
            source,
            "  let inclusion_{index} = Sphere(r: 0.01) |> translate({}, 0.5, 0.5)",
            index as f64 * 0.02
        )
        .unwrap();
    }
    source.push_str("  let result = base |> fragment([");
    for index in 0..entities {
        if index != 0 {
            source.push_str(", ");
        }
        write!(source, "inclusion_{index}").unwrap();
    }
    source.push_str("])\n}\n");
    source
}

fn nested_csg(operands: usize) -> String {
    let mut source = String::with_capacity(operands * 72);
    source.push_str("geometry Main {\n");
    for index in 0..operands {
        writeln!(
            source,
            "  let solid_{index} = Box(1.0, 1.0, 1.0) |> translate({}, 0.0, 0.0)",
            index as f64 * 0.01
        )
        .unwrap();
    }
    source.push_str("  let result = solid_0");
    for index in 1..operands {
        write!(source, " - solid_{index}").unwrap();
    }
    source.push_str("\n}\n");
    source
}

fn import_chain(modules: usize) -> (TempDir, String, String) {
    let directory = tempfile::tempdir().unwrap();
    for index in 0..modules {
        let mut source = String::new();
        if index + 1 < modules {
            writeln!(source, "import \"module_{}.mlg\"", index + 1).unwrap();
        }
        writeln!(
            source,
            "pub geometry Module{index} {{ let solid = Box(1.0, 1.0, 1.0) }}"
        )
        .unwrap();
        fs::write(directory.path().join(format!("module_{index}.mlg")), source).unwrap();
    }
    let filename = directory.path().join("root.mlg");
    (
        directory,
        "import \"module_0.mlg\"\n".to_owned(),
        filename.to_string_lossy().into_owned(),
    )
}

fn benchmark_frontend_shapes(criterion: &mut Criterion) {
    let mut arrays = criterion.benchmark_group("compiler_large_fragment_array");
    for entities in [64, 256] {
        let source = large_fragment_array(entities);
        arrays.bench_with_input(
            BenchmarkId::from_parameter(entities),
            &source,
            |bencher, source| {
                bencher
                    .iter(|| black_box(preview_canonical_ir(source, "large-array.mlg").unwrap()));
            },
        );
    }
    arrays.finish();

    let mut csg = criterion.benchmark_group("compiler_nested_csg");
    for operands in [32, 128] {
        let source = nested_csg(operands);
        csg.bench_with_input(
            BenchmarkId::from_parameter(operands),
            &source,
            |bencher, source| {
                bencher.iter(|| black_box(preview_canonical_ir(source, "nested-csg.mlg").unwrap()));
            },
        );
    }
    csg.finish();
}

fn benchmark_import_graph(criterion: &mut Criterion) {
    let (directory, source, filename) = import_chain(64);
    let options = CompileOptions {
        import_sandbox_root: Some(directory.path().to_path_buf()),
        ..CompileOptions::default()
    };
    let mut group = criterion.benchmark_group("compiler_import_chain_64");
    group.bench_function("cold", |bencher| {
        bencher.iter_batched(
            || IncrementalCompiler::with_options(directory.path(), options.clone()).unwrap(),
            |mut compiler| black_box(compiler.preview_canonical_ir(&source, &filename).unwrap()),
            BatchSize::SmallInput,
        );
    });

    let mut compiler =
        IncrementalCompiler::with_options(directory.path(), options.clone()).unwrap();
    compiler.preview_canonical_ir(&source, &filename).unwrap();
    group.bench_function("warm_module_and_ir_caches", |bencher| {
        bencher.iter(|| black_box(compiler.preview_canonical_ir(&source, &filename).unwrap()));
    });
    group.finish();
}

fn benchmark_candidate_recompilation(criterion: &mut Criterion) {
    let (directory, _, filename) = import_chain(1);
    let options = CompileOptions {
        import_sandbox_root: Some(directory.path().to_path_buf()),
        ..CompileOptions::default()
    };
    let candidates = (0..64)
        .map(|index| {
            format!(
                "import \"module_0.mlg\"\nparam size = {}\ngeometry Candidate {{ let solid = Box(size, size, size) }}\n",
                1.0 + index as f64 * 0.001
            )
        })
        .collect::<Vec<_>>();
    let mut compiler = IncrementalCompiler::with_options(directory.path(), options).unwrap();
    for source in &candidates {
        compiler.preview_canonical_ir(source, &filename).unwrap();
    }
    let mut index = 0usize;
    criterion.bench_function(
        "compiler_candidate_recompile_64_variant_working_set",
        |bencher| {
            bencher.iter(|| {
                let source = &candidates[index];
                index = (index + 1) % candidates.len();
                black_box(compiler.preview_canonical_ir(source, &filename).unwrap())
            });
        },
    );
}

/// Real consumer traces (not synthetic): a 22 KB PC-fan shape, a two-way
/// loudspeaker example, and the stdlib shapes module. Paths are relative to
/// this bench file (five levels up to the workspace root).
const REAL_TRACES: [(&str, &str, &str); 3] = [
    (
        "pcfan",
        "../../../../../crates/sonium-sim-pcfan/shapes/pcfan.mlg",
        "pcfan.mlg",
    ),
    (
        "two-way-speaker",
        "../../../../../crates/sonium-sim-speaker/examples/two_way_speaker.mlg",
        "two_way_speaker.mlg",
    ),
    (
        "stdlib-shapes",
        "../../../../../crates/sonium-meshlang/stdlib/shapes.mlg",
        "shapes.mlg",
    ),
];

fn benchmark_real_trace_cold_warm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("compiler_real_trace");
    for (name, path, filename) in REAL_TRACES {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let trace_path = manifest.join("benches").join(path);
        let base_dir = trace_path.parent().expect("trace must have a parent dir");
        let source = fs::read_to_string(&trace_path).expect("real trace must be readable");
        let options = CompileOptions {
            import_sandbox_root: Some(base_dir.to_path_buf()),
            ..CompileOptions::default()
        };
        group.bench_function(format!("{name}_cold"), |bencher| {
            bencher.iter_batched(
                || IncrementalCompiler::with_options(base_dir, options.clone()).unwrap(),
                |mut compiler| black_box(compiler.preview_canonical_ir(&source, filename).unwrap()),
                BatchSize::SmallInput,
            );
        });

        let mut compiler = IncrementalCompiler::with_options(base_dir, options.clone()).unwrap();
        compiler.preview_canonical_ir(&source, filename).unwrap();
        // Second pass engages the lowering cache; sample after it so the log
        // proves the warm path (ir_hits >= 1). Note the warm bench still
        // re-parses, re-flattens, re-resolves and re-typechecks by design —
        // only lowering is cached — so cold-vs-warm isolates lowering cost.
        compiler.preview_canonical_ir(&source, filename).unwrap();
        let stats = compiler.stats();
        eprintln!(
            "REALTRACE {name} warm modules={} ir_entries={} ir_hits={}",
            stats.parsed_module_entries, stats.canonical_ir_entries, stats.canonical_ir_hits
        );
        group.bench_function(format!("{name}_warm"), |bencher| {
            bencher.iter(|| black_box(compiler.preview_canonical_ir(&source, filename).unwrap()));
        });
    }
    group.finish();
}

/// Peak RSS (VmHWM) of this process in KiB. Linux-only; other platforms
/// report 0 and the growth curve degrades to the timing series.
#[cfg(target_os = "linux")]
fn peak_rss_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kib() -> u64 {
    0
}

fn benchmark_memory_growth(criterion: &mut Criterion) {
    // Cumulative working sets: each point builds a fresh compiler, compiles
    // K distinct variants, and samples peak-RSS delta. VmHWM is monotonic,
    // so the end sample is the curve value; the K=0 sample is the harness
    // baseline (criterion RSS is large — deltas are what matter).
    let baseline = peak_rss_kib();
    let mut group = criterion.benchmark_group("compiler_memory_growth");
    for variants in [16usize, 32, 64, 128] {
        let directory = tempfile::tempdir().unwrap();
        let options = CompileOptions {
            import_sandbox_root: Some(directory.path().to_path_buf()),
            ..CompileOptions::default()
        };
        let sources = (0..variants)
            .map(|index| {
                format!(
                    "param size = {}\ngeometry Candidate {{ let solid = Box(size, size, size) }}\n",
                    1.0 + index as f64 * 0.001
                )
            })
            .collect::<Vec<_>>();
        let mut compiler = IncrementalCompiler::with_options(directory.path(), options).unwrap();
        for source in &sources {
            compiler
                .preview_canonical_ir(source, "candidate.mlg")
                .unwrap();
        }
        let delta = peak_rss_kib().saturating_sub(baseline);
        eprintln!("MEMGROWTH variants={variants} vmhwm_delta_kib={delta}");
        group.bench_function(BenchmarkId::from_parameter(variants), |bencher| {
            bencher.iter_batched(
                || {
                    IncrementalCompiler::with_options(
                        directory.path(),
                        CompileOptions {
                            import_sandbox_root: Some(directory.path().to_path_buf()),
                            ..CompileOptions::default()
                        },
                    )
                    .unwrap()
                },
                |mut compiler| {
                    for source in &sources {
                        black_box(
                            compiler
                                .preview_canonical_ir(source, "candidate.mlg")
                                .unwrap(),
                        );
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_frontend_shapes,
    benchmark_import_graph,
    benchmark_candidate_recompilation,
    benchmark_real_trace_cold_warm,
    benchmark_memory_growth
);
criterion_main!(benches);
