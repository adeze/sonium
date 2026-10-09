//! Release-ready wrapper for the bounded tensor-product H(curl) packet.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hcurl_tensor_curved_p2_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-tensor-curved-p2-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::tensor_product_hcurl_curved_p2_reference);
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_curved_tensor_hcurl_checks", 2.0)
            .minimum("independent_curved_tensor_hcurl_checks", 2.0)
            .metric("curved_q2_quadrilateral_reference", 1.0)
            .minimum("curved_q2_quadrilateral_reference", 1.0)
            .metric("curved_q2_hexahedron_reference", 1.0)
            .minimum("curved_q2_hexahedron_reference", 1.0)
            .input_checksum("mesh:curved-q2-quad-hex:hcurl-reference-v1")
            .reference("independent Q2 isoparametric map and tensor-product Nedelec reference")
            .reference_checksum("analytic/deterministic-fixture:fem-hcurl-tensor-curved-p2-v1")
            .dimension("curved 2-D Q2 quadrilateral and 3-D Q2 hexahedron")
            .formulation("lowest-order tensor-product H(curl) with covariant Piola value/curl map")
            .tolerance(
                "all coordinate, value, and curl checks pass at the focused numerical thresholds",
            )
            .seed("fem-hcurl-tensor-curved-p2-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("tensor-product curved-Q2 H(curl) reference panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("fem-hcurl-tensor-curved-p2-reference:cpu:v1")
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
