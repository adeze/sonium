//! Release-backed curved-P2 H(curl) geometry reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hcurl_curved_p2_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-curved-p2-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::curved_p2_hcurl_uses_physical_map_and_jacobian();
        reference::curved_p2_hcurl_tetrahedral_map_for_both_orders();
    });
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_curved_hcurl_checks", 4.0)
            .minimum("independent_curved_hcurl_checks", 4.0)
            .metric("p2_geometry_reference", 1.0)
            .minimum("p2_geometry_reference", 1.0)
            .metric("covariant_piola_jacobian_reference", 1.0)
            .minimum("covariant_piola_jacobian_reference", 1.0)
            .input_checksum("mesh:curved-p2-unit-square-triangle:hcurl-reference-v1")
            .reference(
                "independent P2 isoparametric physical-coordinate and Jacobian reconstruction",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-hcurl-curved-p2-v1")
            .dimension("curved 2-D/3-D P2 geometry with order-1/order-2 H(curl) reference bases")
            .formulation("covariant Piola mapping on explicit P2 geometry")
            .seed("fem-hcurl-curved-p2-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved P2 H(curl) reference panicked: {payload:?}"),
        )
        .seed("fem-hcurl-curved-p2-reference:cpu:v1")
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
