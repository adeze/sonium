//! Release-backed RT1 H(div)-scalar mixed-conservation reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_rt1_mixed_conservation.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.hdiv-rt1-mixed-conservation.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::rt1_mixed_conservation_reference);
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rt1_mixed_domain_checks", 4.0)
            .minimum("rt1_mixed_domain_checks", 4.0)
            .metric("rt1_constant_field_conservation", 1.0)
            .minimum("rt1_constant_field_conservation", 1.0)
            .input_checksum("mesh:unit-square-and-unit-cube:rt1-mixed-conservation-v1")
            .reference(
                "independent quadrature of RT1 Piola divergence against constant P1/Q1 field",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-mixed-conservation-v1")
            .dimension("2-D triangles/quadrilaterals and 3-D tetrahedra/hexahedra")
            .formulation("checked mixed H(div)-scalar divergence block")
            .seed("fem-hdiv-rt1-mixed-conservation:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RT1 mixed-conservation reference panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("fem-hdiv-rt1-mixed-conservation:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
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
