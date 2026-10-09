//! Release-facing three-dimensional P2/Q2 manufactured Poisson convergence reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/3d_p2_poisson_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.p2-poisson-convergence-3d.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::p2_poisson_manufactured_solution_converges_in_3d_tetrahedra_and_hexahedra();
    });
    let passed = outcome.is_ok();
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("families_checked", 2.0)
            .minimum("families_checked", 2.0)
            .metric("three_dimensional_manufactured_reference", 1.0)
            .minimum("three_dimensional_manufactured_reference", 1.0)
            .input_checksum("mesh:unit-cube-tetrahedra-and-hexahedra:1-4")
            .reference("independent quartic manufactured Poisson solution and refinement rates")
            .reference_checksum("analytic/deterministic-fixture:fem-p2-poisson-convergence-3d-v1")
            .formulation("production 3-D P2 tetrahedron and Q2 hexahedron scalar FEM path")
            .dimension("unit cube tetrahedral and hexahedral refinements 1, 2, 4")
            .tolerance("finite residuals and expected L2/H1 refinement rates")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("3-D P2/Q2 Poisson convergence reference panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
