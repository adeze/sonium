//! Release-backed nested MeshLang control-flow and canonical-IR reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/nested_control_flow_packet.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.nested-control-flow-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::nested_compile_time_control_flow_has_composite_bindings_and_backend_parity();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("backend_parity", 1.0)
            .minimum("backend_parity", 1.0)
            .metric("composite_bindings", 4.0)
            .minimum("composite_bindings", 4.0)
            .metric("provenance_complete", 1.0)
            .minimum("provenance_complete", 1.0)
            .input_checksum("source:nested-control-flow-packet.mlg:v1")
            .reference("canonical IR, GEO/Python lowering, provenance, sensitivity, and composite-binding parity")
            .reference_checksum("analytic/deterministic-fixture:meshlang-nested-control-flow-v1")
            .dimension("two nested compile-time loops with if/else Box/Cylinder bindings")
            .formulation("MeshLang parser/compiler canonical IR with GEO and Python lowering")
            .tolerance("four composite bindings; backend, provenance, and sensitivity parity")
            .seed("meshlang-nested-control-flow-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("nested control-flow reference panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
