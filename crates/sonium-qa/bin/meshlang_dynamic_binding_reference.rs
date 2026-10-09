//! Release-backed MeshLang dynamic branch/multi-binding canonical reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/dynamic_binding_packet.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.dynamic-multi-binding-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::static_control_flow_with_dynamic_multi_binding_is_canonical_and_deterministic();
        reference::mixed_control_flow_loop_bindings_match_backends();
        reference::dynamic_else_if_multi_bindings_have_one_canonical_branch_plan();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dynamic_binding_checks", 3.0)
            .minimum("dynamic_binding_checks", 3.0)
            .metric("geo_python_canonical_parity", 1.0)
            .minimum("geo_python_canonical_parity", 1.0)
            .metric("provenance_sensitivity_preservation", 1.0)
            .minimum("provenance_sensitivity_preservation", 1.0)
            .input_checksum("source:dynamic-binding-packet.mlg:v1")
            .reference("canonical IR branch topology, GEO/Python parity, deterministic multi-binding names, provenance, and sensitivity dependencies")
            .reference_checksum("analytic/deterministic-fixture:meshlang-dynamic-multi-binding-v1")
            .dimension("dynamic else-if/else branch plans with loop-produced multi-bindings")
            .formulation("MeshLang canonical IR dynamic control lowering with backend-neutral provenance")
            .tolerance("three dynamic binding checks plus canonical parity and dependency preservation")
            .seed("meshlang-dynamic-multi-binding-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("dynamic MeshLang binding reference panicked: {payload:?}"),
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
