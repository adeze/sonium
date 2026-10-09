//! Release-style MeshLang imported nested-provenance reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/imported_provenance_packet.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "meshlang.imported-provenance-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::imported_nested_transforms_preserve_comments_spans_and_parent_chain();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("backend_parity", 1.0)
            .metric("imported_comments", 4.0)
            .metric("nested_parent_chain_depth", 2.0)
            .metric("imported_source_identity", 1.0)
            .minimum("backend_parity", 1.0)
            .minimum("imported_comments", 4.0)
            .minimum("nested_parent_chain_depth", 2.0)
            .minimum("imported_source_identity", 1.0)
            .input_checksum("source:imported-nested-provenance-packet.mlg:v1")
            .reference(
                "filesystem import flattening preserves nested transformed-parent chains, comments, spans, and GEO/Python parity",
            )
            .reference_checksum("analytic/deterministic-fixture:meshlang-imported-provenance-v1")
            .dimension("imported source identity, comment targets, nested parent-chain metadata")
            .formulation("MeshLang ImportResolver flattening and canonical backend sidecars")
            .tolerance("exact GEO/Python canonical, provenance, sensitivity, and comment parity")
            .seed("meshlang-imported-provenance-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("imported provenance regression panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA_RESULT serialization")
    );
    let passed = matches!(result.status, QaStatus::Passed);
    println!("{}", if passed { "SUCCESS" } else { "FAILED" });
    if !passed {
        std::process::exit(1);
    }
}
