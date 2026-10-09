//! Release-backed affine-triangle RT1 H(div) reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_rt1_triangle_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.hdiv-rt1-triangle-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::rt1_reference_basis_has_identity_edge_and_cell_moments();
        reference::rt1_reference_divergence_matches_independent_finite_difference();
        reference::rt1_triangle_space_has_stable_edge_and_cell_topology();
        reference::rt1_shared_edge_traces_are_invariant_under_local_reversal();
        reference::rt1_admission_accepts_implemented_cell_families();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_rt1_checks", 5.0)
            .minimum("independent_rt1_checks", 5.0)
            .metric("edge_and_cell_moment_unisolvence", 1.0)
            .minimum("edge_and_cell_moment_unisolvence", 1.0)
            .metric("shared_edge_orientation", 1.0)
            .minimum("shared_edge_orientation", 1.0)
            .metric("unsupported_element_admission", 1.0)
            .minimum("unsupported_element_admission", 1.0)
            .input_checksum("mesh:unit-square-triangles:rt1-reference-v1")
            .reference("independent edge/cell moment quadrature, finite-difference divergence, topology, orientation, and admission checks")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-triangle-v1")
            .dimension("affine 2-D triangles; RT1 H(div) with eight local DOFs")
            .formulation("Piola-mapped RT1 triangle basis with oriented two-moment edges and two cell moments")
            .tolerance("five independent RT1 reference checks pass")
            .seed("fem-hdiv-rt1-triangle-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RT1 triangle reference panicked: {payload:?}"),
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
