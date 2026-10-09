//! CPU-only bounded MeshLang subtree parser/cache reference.
//!
//! This checks the editor-facing range contract without invoking Gmsh,
//! meshing, a solver, or any simulation workload.

use meshlang_parser::{ParseCache, ParseOptions, SourceRange, parse_subtree, parse_subtree_json};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "meshlang.parser-subtree-reference.v1";
const SOURCE: &str = "param before = 1\n/// selected\nparam selected = 2\nparam after = 3\n";

fn run() -> Result<(), String> {
    let start = SOURCE.find("/// selected").ok_or("selected range start")?;
    let end = SOURCE.find("\nparam after").ok_or("selected range end")?;
    let range = SourceRange::new(start, end);
    let options = ParseOptions {
        file_id: 41,
        ..ParseOptions::default()
    };

    let first = parse_subtree(SOURCE, "qa-subtree.mlg", range, options)
        .map_err(|error| error.to_string())?;
    let repeated = parse_subtree(SOURCE, "qa-subtree.mlg", range, options)
        .map_err(|error| error.to_string())?;
    if first.source_file.span.start != start
        || first.source_file.span.end != end
        || first.source_file.declarations.len() != 1
        || !first.diagnostics.is_empty()
        || first
            .source_file
            .structural_hash()
            .map_err(|error| error.to_string())?
            != repeated
                .source_file
                .structural_hash()
                .map_err(|error| error.to_string())?
    {
        return Err("subtree span, declaration, diagnostics, or hash contract failed".into());
    }

    let direct_json = parse_subtree_json(SOURCE, "qa-subtree.mlg", range, options)
        .map_err(|error| error.to_string())?;
    let mut cache = ParseCache::default();
    let cached_json = cache
        .parse_subtree_json(SOURCE, "qa-subtree.mlg", range, options)
        .map_err(|error| error.to_string())?;
    let cached_again = cache
        .parse_subtree_json(SOURCE, "qa-subtree.mlg", range, options)
        .map_err(|error| error.to_string())?;
    if direct_json != cached_json
        || cached_json != cached_again
        || cache.stats().hits != 1
        || cache.stats().misses != 1
        || cache.subtree_len() != 1
    {
        return Err("subtree cache envelope or telemetry contract failed".into());
    }

    let unicode = "param räume = 1";
    if parse_subtree(
        unicode,
        "utf8.mlg",
        SourceRange::new(8, unicode.len()),
        ParseOptions::default(),
    )
    .is_ok()
    {
        return Err("split UTF-8 range was accepted".into());
    }
    Ok(())
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(Ok(())) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("absolute_span_checks", 4.0)
            .metric("cache_determinism_checks", 5.0)
            .metric("range_validation_checks", 2.0)
            .minimum("absolute_span_checks", 4.0)
            .minimum("cache_determinism_checks", 5.0)
            .minimum("range_validation_checks", 2.0)
            .input_checksum("source:meshlang-parser-subtree-reference:v1")
            .reference("bounded source-range parsing with absolute spans, full-file identity, deterministic cache, and UTF-8 validation")
            .reference_checksum("analytic/deterministic-fixture:meshlang-parser-subtree-v1")
            .dimension("editor subtree parsing and cache contract")
            .formulation("MeshLang bounded subtree parser")
            .tolerance("zero diagnostics for selected declaration; repeated JSON and structural hashes byte-identical")
            .seed("meshlang-parser-subtree-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(CASE_ID, format!("subtree reference panicked: {payload:?}"))
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(elapsed_ms),
    };
    println!("{}", result.to_json_line().expect("QA_RESULT JSON"));
    if !matches!(result.status, QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
