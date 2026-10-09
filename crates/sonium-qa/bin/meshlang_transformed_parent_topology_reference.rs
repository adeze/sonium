//! Release-style MeshLang transformed-parent topology/provenance reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/transformed_parent_topology_packet.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.transformed-parent-topology-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::transformed_parent_topology_has_backend_and_provenance_parity();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("backend_parity", 1.0)
            .metric("transformed_parent_chain", 1.0)
            .metric("derived_topology_groups", 2.0)
            .metric("multi_binding_provenance", 1.0)
            .minimum("backend_parity", 1.0)
            .minimum("transformed_parent_chain", 1.0)
            .minimum("derived_topology_groups", 2.0)
            .minimum("multi_binding_provenance", 1.0)
            .input_checksum("source:transformed-parent-topology-packet.mlg:v1")
            .reference(
                "two-step transformed-parent topology selectors with GEO/Python canonical, comment, and provenance parity",
            )
            .reference_checksum("analytic/deterministic-fixture:meshlang-transformed-parent-topology-v1")
            .dimension("transformed parent chain, derived topology groups, multi-binding provenance")
            .formulation("MeshLang canonical IR with GEO/Python topology-query lowering")
            .tolerance("exact canonical/backend sidecar parity and two derived topology groups")
            .seed("meshlang-transformed-parent-topology-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("transformed-parent topology reference panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA result must serialize")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
