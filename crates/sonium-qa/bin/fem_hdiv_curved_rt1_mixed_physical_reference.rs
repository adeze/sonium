//! Release-backed 3-D curved-Q2 RT1 mixed physical-coordinate reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/hdiv_curved_rt1_mixed_physical_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.hdiv-curved-rt1-mixed-physical-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_curved_rt1_mixed_physical_reference);
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric(
                "curved_rt1_mixed_block_entries",
                metrics.checked_entries as f64,
            )
            .minimum("curved_rt1_mixed_block_entries", 288.0)
            .metric("max_block_error", metrics.max_block_error)
            .maximum("max_block_error", 2.0e-10)
            .metric(
                "max_constant_conservation_error",
                metrics.max_constant_conservation_error,
            )
            .maximum("max_constant_conservation_error", 2.0e-10)
            .metric("minimum_curved_jacobian_det", metrics.min_jacobian_det)
            .minimum("minimum_curved_jacobian_det", 0.05)
            .input_checksum("mesh:curved-q2-unit-cube-hexahedron:rt1-mixed-v1")
            .reference("independent Q2 physical map and RT1/P1 mixed divergence quadrature")
            .reference_checksum("analytic/deterministic-fixture:fem-hdiv-curved-rt1-mixed-v1")
            .dimension("3-D curved Q2 hexahedron")
            .formulation("RT1 H(div)-P1 checked mixed divergence block with Piola map")
            .seed("fem-hdiv-curved-rt1-mixed-physical-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved RT1 mixed physical reference panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("fem-hdiv-curved-rt1-mixed-physical-reference:cpu:v1")
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
