//! Release-facing MeshLang parser/materializer/backend capability contract.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/capability_contract.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.capability-contract-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::native_backends_preserve_projected_geometry_as_metadata_only();
        reference::typed_ports_parse_and_legacy_ast_roundtrip();
        reference::canonical_control_parent_validation();
        reference::canonical_control_branch_validation();
        reference::canonical_mixed_control_parent_links();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let (result, passed) = match outcome {
        Ok(()) => (
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("capability_rejection_checks", 2.0)
                .minimum("capability_rejection_checks", 2.0)
                .metric("typed_port_schema_checks", 1.0)
                .minimum("typed_port_schema_checks", 1.0)
                .metric("control_parent_validation_checks", 1.0)
                .minimum("control_parent_validation_checks", 1.0)
        .metric("control_branch_validation_checks", 1.0)
        .minimum("control_branch_validation_checks", 1.0)
        .metric("mixed_control_parent_checks", 4.0)
        .minimum("mixed_control_parent_checks", 4.0)
                .input_checksum("source:meshlang-capability-contract:v1")
                .reference(
                    "native Project rejection, typed-port persistence, and canonical mixed-control topology",
                )
                .reference_checksum("analytic/deterministic-fixture:meshlang-capability-v1")
                .dimension("GEO/Python capability admission and versioned AST port fields")
                .formulation("MeshLang canonical backend gate, parser, and AST persistence")
                .tolerance(
                    "two backend rejection checks, one typed-port schema check, one branch-order check, four mixed-parent checks",
                )
                .seed("meshlang-capability-contract-reference:cpu:v1")
                .elapsed_ms(elapsed_ms),
            true,
        ),
        Err(payload) => (
            QaResult::failed(CASE_ID, format!("reference panic: {payload:?}"))
                .elapsed_ms(elapsed_ms),
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
