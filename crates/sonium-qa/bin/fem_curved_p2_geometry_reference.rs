//! Release-facing curved P2 isoparametric H1 geometry reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/curved_geometry_contract.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.curved-p2-h1-geometry.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::w1_05_h1_accepts_curved_p2_geometry_across_supported_cells();
        reference::w1_05_curved_p2_hessians_match_reference_coordinate_finite_difference();
    });
    let passed = outcome.is_ok();
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("element_families_checked", 4.0)
            .minimum("element_families_checked", 4.0)
            .metric("curved_p2_geometry_interpolation", 1.0)
            .minimum("curved_p2_geometry_interpolation", 1.0)
            .metric("curved_p2_hessian_reference", 1.0)
            .minimum("curved_p2_hessian_reference", 1.0)
            .input_checksum("mesh:promote-to-p2:triangle-quad-tetrahedron-hexahedron")
            .reference("independent P2 nodal interpolation and reference-coordinate gradient finite difference")
            .reference_checksum("analytic/deterministic-fixture:fem-curved-p2-h1-geometry-v1")
            .formulation("production H1 P1/P2 fields on explicit isoparametric P2 geometry")
            .dimension("2-D triangle/quadrilateral and 3-D tetrahedron/hexahedron")
            .tolerance("positive Jacobian and finite-difference Hessian error")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved P2 H1 geometry reference panicked: {payload:?}"),
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
