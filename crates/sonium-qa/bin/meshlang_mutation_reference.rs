//! Release-facing deterministic MeshLang parser mutation/recovery reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-parser/tests/mutation.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.parser-mutation-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome =
        std::panic::catch_unwind(reference::deterministic_mutation_corpus_is_bounded_and_stable);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let (result, passed) = match outcome {
        Ok(()) => (
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("mutation_seeds", 8.0)
                .minimum("mutation_seeds", 8.0)
                .metric("mutations_per_seed", 32.0)
                .minimum("mutations_per_seed", 32.0)
                .metric("bounded_recovery_checks", 256.0)
                .minimum("bounded_recovery_checks", 256.0)
                .input_checksum("source:meshlang-parser-mutation-seeds:v2")
                .reference("deterministic single-character mutation corpus with bounded recovery and stable JSON")
                .reference_checksum("analytic/deterministic-fixture:meshlang-parser-mutation-v2")
                .dimension("UTF-8 spans, diagnostics, AST recovery, structural hash")
                .formulation("MeshLang parser totality and deterministic mutation recovery")
                .tolerance("256 mutations return without panic, stay bounded, and serialize deterministically")
                .seed("meshlang-parser-mutation-reference:cpu:v1")
                .elapsed_ms(elapsed_ms),
            true,
        ),
        Err(payload) => (
            QaResult::failed(CASE_ID, format!("mutation reference panic: {payload:?}")).elapsed_ms(elapsed_ms),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
