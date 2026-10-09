//! Release-facing bounded MeshLang parser recovery fuzz-smoke reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-parser/tests/recovery_fuzz.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "meshlang.parser-recovery-fuzz-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(
        reference::deterministic_recovery_fuzz_corpus,
    ));
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("fixed_seed_cases", 4.0)
            .metric("generated_cases", 96.0)
            .metric("repeated_determinism_checks", 100.0)
            .minimum("fixed_seed_cases", 4.0)
            .minimum("generated_cases", 96.0)
            .minimum("repeated_determinism_checks", 100.0)
            .input_checksum("source:meshlang-parser-recovery-fuzz-smoke:v1")
            .reference("bounded UTF-8-lossy parser recovery corpus with span, panic, and deterministic JSON invariants")
            .reference_checksum("analytic/deterministic-fixture:meshlang-parser-recovery-fuzz-v1")
            .dimension("recovering parser totality, resource bounds, span validity, deterministic reports")
            .formulation("MeshLang parser parse_recovering deterministic fuzz-smoke")
            .tolerance("zero panics, all spans in source, diagnostics capped, repeated reports byte-identical")
            .seed("meshlang-parser-recovery-fuzz-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("parser recovery fuzz-smoke panicked: {payload:?}"),
        )
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
