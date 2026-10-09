//! Release-backed MeshLang static-control-flow/dynamic-binding reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/dynamic_binding_packet.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.dynamic-binding-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::static_control_flow_with_dynamic_multi_binding_is_canonical_and_deterministic();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("canonical_backend_checks", 8.0)
            .minimum("canonical_backend_checks", 8.0)
            .metric("backend_box_emissions", 2.0)
            .minimum("backend_box_emissions", 2.0)
            .metric("non_leaked_control_syntax_checks", 1.0)
            .minimum("non_leaked_control_syntax_checks", 1.0)
            .reference(
                "deterministic canonical IR and GEO/Python backend parity for static control flow with dynamic multi-binding",
            )
            .reference_checksum("analytic/deterministic-fixture:meshlang-dynamic-binding-v1")
            .dimension("two Box bindings, BooleanUnion, translate pipeline, one static if block")
            .formulation("MeshLang parser/compiler canonical IR with GEO and Python lowering")
            .tolerance("eight parity checks, two backend Box emissions, and non-leaked control syntax")
            .seed("meshlang-dynamic-binding-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("MeshLang static dynamic-binding reference panicked: {payload:?}"),
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
