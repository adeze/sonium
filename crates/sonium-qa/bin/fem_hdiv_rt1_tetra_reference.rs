//! Release-backed affine-tetrahedron RT1 H(div) reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_rt1_tetra_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-tetra-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::rt1_tetra_reference_basis_has_identity_face_and_cell_moments();
        reference::rt1_tetra_reference_divergence_matches_finite_difference();
        reference::rt1_tetra_space_has_face_cell_numbering_and_piola_tabulation();
    });
    let passed = outcome.is_ok();
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_rt1_checks", 3.0)
            .minimum("independent_rt1_checks", 3.0)
            .metric("face_and_cell_moment_unisolvence", 1.0)
            .minimum("face_and_cell_moment_unisolvence", 1.0)
            .metric("finite_difference_divergence", 1.0)
            .minimum("finite_difference_divergence", 1.0)
            .metric("topology_and_piola_contract", 1.0)
            .minimum("topology_and_piola_contract", 1.0)
            .input_checksum("mesh:unit-cube-tetrahedra:rt1-reference-v1")
            .reference(
                "independent tetrahedral face/cell quadrature, finite-difference divergence, topology and Piola checks",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt1-tetra-v1")
            .dimension("affine 3-D RT1 H(div) DOFs")
            .formulation("RT1 face moments plus Cartesian cell moments with contravariant Piola mapping")
            .seed("fem-hdiv-rt1-tetra-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RT1 tetrahedron reference panic: {payload:?}"),
        ),
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
