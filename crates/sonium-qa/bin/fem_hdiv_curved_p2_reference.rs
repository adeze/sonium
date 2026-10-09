//! Release-backed curved-P2 RT1 H(div) reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_curved_p2_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-curved-p2-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::curved_rt1_hdiv_uses_p2_geometry_and_admits_hex();
        reference::curved_rt1_hdiv_piola_divergence_matches_physical_finite_difference();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_curved_hdiv_checks", 2.0)
            .minimum("independent_curved_hdiv_checks", 2.0)
            .metric("p2_geometry_reference", 1.0)
            .minimum("p2_geometry_reference", 1.0)
            .metric("physical_divergence_piola_reference", 1.0)
            .minimum("physical_divergence_piola_reference", 1.0)
            .input_checksum("mesh:curved-p2-unit-cube-hexahedron:rt1-reference-v1")
            .reference(
                "independent P2 isoparametric physical-coordinate reconstruction and physical-divergence finite difference",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-curved-p2-v1")
            .dimension("curved 3-D P2 geometry with affine RT1 H(div) reference basis")
            .formulation("contravariant Piola mapping on explicit P2/Q2 geometry")
            .seed("fem-hdiv-curved-p2-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved RT1 H(div) reference panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
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
