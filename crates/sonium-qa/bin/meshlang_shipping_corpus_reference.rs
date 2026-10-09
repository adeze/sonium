//! Release-facing compiler parity reference for the checked-in MeshLang corpus.
//!
//! This CPU-only gate exercises every product/stdlib source in the compiler's
//! shipping manifest through both canonical GEO and Python lowering. It does
//! not launch Gmsh, import Python, or start a simulator; those execution lanes
//! remain separate environment-dependent gates.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use serde_json::Value;
use sonium_qa::QaResult;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CASE_ID: &str = "meshlang.shipping-corpus-reference.v1";

const CORPUS: &[&str] = &[
    "../sonium-sim-pcfan/shapes/pcfan.mlg",
    "../sonium-sim-speaker/examples/two_way_speaker.mlg",
    "examples/gmsh_helmholtz_resonator.mlg",
    "examples/gmsh_t16_csg.mlg",
    "examples/gmsh_t19_loft_pipe.mlg",
    "examples/gmsh_t5_cheese.mlg",
    "examples/ported_acoustic_frame.mlg",
    "examples/room_acoustics.mlg",
    "examples/speaker_cabinet.mlg",
    "examples/sphere_scattering.mlg",
    "stdlib/acoustics.mlg",
    "stdlib/shapes.mlg",
];

fn meshlang_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../sonium-meshlang")
}

fn entity_count(value: &Value) -> usize {
    if let Some(entities) = value.get("entities").and_then(Value::as_array) {
        return entities.len();
    }
    value.get("ir").map_or(0, entity_count)
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let root = meshlang_root();
    let mut parity_files = 0usize;
    let mut deterministic_files = 0usize;
    let mut entity_bearing_files = 0usize;
    let mut max_ir_bytes = 0usize;

    for relative in CORPUS {
        let path = root.join(relative);
        let source = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {relative}: {error}"))?;
        let geo = compile_to_geo_full(&source, relative)
            .map_err(|error| format!("GEO compilation failed for {relative}: {error}"))?;
        let python = compile_to_python_full(&source, relative)
            .map_err(|error| format!("Python compilation failed for {relative}: {error}"))?;

        if geo.code.is_empty() || python.code.is_empty() {
            return Err(format!("empty backend output for {relative}"));
        }
        if geo.canonical_ir_hash != python.canonical_ir_hash
            || geo.canonical_ir_json != python.canonical_ir_json
        {
            return Err(format!("canonical IR parity drift for {relative}"));
        }
        let canonical: Value = serde_json::from_str(&geo.canonical_ir_json)
            .map_err(|error| format!("invalid canonical IR for {relative}: {error}"))?;
        let entities = entity_count(&canonical);
        if entities > 0 {
            entity_bearing_files += 1;
        }
        max_ir_bytes = max_ir_bytes.max(geo.canonical_ir_json.len());
        parity_files += 1;

        let repeat = compile_to_geo_full(&source, relative)
            .map_err(|error| format!("repeat GEO compilation failed for {relative}: {error}"))?;
        if repeat.code != geo.code || repeat.canonical_ir_hash != geo.canonical_ir_hash {
            return Err(format!("non-deterministic GEO output for {relative}"));
        }
        deterministic_files += 1;
    }

    if parity_files != CORPUS.len() || deterministic_files != CORPUS.len() {
        return Err("shipping corpus did not cover every manifest entry".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("corpus_files", CORPUS.len() as f64)
        .metric("backend_parity_files", parity_files as f64)
        .metric("deterministic_repeat_files", deterministic_files as f64)
        .metric("entity_bearing_files", entity_bearing_files as f64)
        .metric("max_canonical_ir_bytes", max_ir_bytes as f64)
        .minimum("corpus_files", CORPUS.len() as f64)
        .minimum("backend_parity_files", CORPUS.len() as f64)
        .minimum("deterministic_repeat_files", CORPUS.len() as f64)
        .reference("checked-in MeshLang product and stdlib corpus canonical GEO/Python parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-shipping-corpus-v1")
        .input_checksum("source:meshlang-shipping-corpus-manifest:v1")
        .dimension("12 product/stdlib MeshLang sources")
        .formulation("backend-neutral canonical IR parity and deterministic repeat compilation")
        .tolerance(
            "every manifest source compiles through both backends with exact canonical IR parity",
        )
        .seed("meshlang-shipping-corpus-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let started = Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (result, true),
        Err(error) => (QaResult::failed(CASE_ID, error), false),
    };
    println!(
        "{}",
        result
            .elapsed_ms(started.elapsed().as_millis() as u64)
            .to_json_line()
            .expect("serialize QA result JSON")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
