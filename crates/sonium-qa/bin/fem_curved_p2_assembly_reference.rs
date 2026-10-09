//! Release-facing curved-P2 assembled stiffness reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/curved_p2_assembly.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.curved-p2-assembly-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::curved_p2_stiffness_preserves_symmetry_conservation_and_energy();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("curved_p2_stiffness_reference", 1.0)
            .minimum("curved_p2_stiffness_reference", 1.0)
            .metric("symmetry_conservation_energy_contract", 1.0)
            .minimum("symmetry_conservation_energy_contract", 1.0)
            .input_checksum("mesh:unit-square-triangles:promote-to-p2:curved-midpoint")
            .reference("independent curved-P2 stiffness symmetry conservation energy checks")
            .reference_checksum("analytic/deterministic-fixture:fem-curved-p2-assembly-v1")
            .formulation("production H1 P2 Laplacian assembly on explicit isoparametric geometry")
            .dimension("2-D curved P2 triangular mesh")
            .tolerance("symmetry <= 2e-12; constant-field row sums <= 2e-11; positive energy")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved-P2 assembly reference panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };
    println!("QA_RESULT: {}", result.to_json_line().unwrap());
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
