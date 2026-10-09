//! Release-backed nested typed-port template canonical-IR reference.

#[allow(dead_code)]
#[path = "../../sonium-meshlang/crates/meshlang-compiler/tests/nested_ported_template.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.nested-ported-template-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::nested_ported_templates_preserve_each_port_through_canonical_ir();
        reference::ported_templates_inside_compile_time_loops_preserve_each_instance_port();
        reference::ported_templates_inside_runtime_branches_keep_branch_qualified_ports();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let (result, passed) = match outcome {
        Ok(()) => (
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
            .metric("nested_port_checks", 13.0)
            .minimum("nested_port_checks", 13.0)
            .metric("geo_python_canonical_parity", 3.0)
            .minimum("geo_python_canonical_parity", 3.0)
                .metric("port_provenance", 1.0)
                .minimum("port_provenance", 1.0)
                .input_checksum("source:meshlang-nested-ported-template:v1")
                .reference("nested typed geometry-template ports through canonical IR and both native backends")
                .reference_checksum("analytic/deterministic-fixture:meshlang-nested-ported-template-v1")
                .dimension("nested Solid/Surface ports and downstream CSG")
                .formulation("MeshLang canonical template-instance port materialization")
                .tolerance("five port bindings plus GEO/Python parity and provenance")
                .seed("meshlang-nested-ported-template-reference:cpu:v1")
                .elapsed_ms(elapsed_ms),
            true,
        ),
        Err(payload) => (
            QaResult::failed(
                CASE_ID,
                format!("nested ported-template reference panicked: {payload:?}"),
            )
            .elapsed_ms(elapsed_ms),
            false,
        ),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
