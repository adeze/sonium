//! Release-backed affine-quadrilateral RT1 H(div) reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_rt1_quadrilateral_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.hdiv-rt1-quadrilateral-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::rt1_quadrilateral_reference_basis_has_identity_moments();
        reference::rt1_quadrilateral_reference_divergence_matches_finite_difference();
        reference::rt1_quadrilateral_space_preserves_edge_and_cell_topology();
        reference::rt1_quadrilateral_shared_edge_traces_survive_local_reversal();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_rt1_quadrilateral_checks", 4.0)
            .minimum("independent_rt1_quadrilateral_checks", 4.0)
            .metric("edge_and_cell_moment_unisolvence", 1.0)
            .minimum("edge_and_cell_moment_unisolvence", 1.0)
            .metric("shared_edge_orientation", 1.0)
            .minimum("shared_edge_orientation", 1.0)
            .input_checksum("mesh:unit-square-quadrilateral:rt1-reference-v1")
            .reference("independent edge/cell moment quadrature, finite-difference divergence, topology, and shared-edge orientation")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-quadrilateral-v1")
            .dimension("2-D affine quadrilaterals; Q2,1 x Q1,2 RT1 H(div) with twelve local DOFs")
 .formulation("Piola-mapped quadrilateral RT1 basis with oriented edge moments, cell moments, divergence, assembly, and orientation checks")
            .tolerance("four independent RT1 quadrilateral reference checks pass")
            .seed("fem-hdiv-rt1-quadrilateral-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RT1 quadrilateral reference panicked: {payload:?}"),
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
