//! Release QA wrapper for explicit P3 H1 isoparametric geometry.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/h1_p3_curved_geometry_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.h1-p3-curved-geometry-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::explicit_p3_geometry_reference);
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("explicit_p3_triangle_checks", 10.0)
            .minimum("explicit_p3_triangle_checks", 10.0)
            .metric("explicit_p3_hexahedron_checks", 64.0)
            .minimum("explicit_p3_hexahedron_checks", 64.0)
            .metric("assembly_symmetry_checks", 200.0)
            .minimum("assembly_symmetry_checks", 200.0)
            .input_checksum("mesh:curved-p3-triangle-and-hexahedron-v1")
            .reference("independent explicit cubic nodal geometry, partition, and assembled symmetry checks")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-p3-curved-geometry-v1")
            .dimension("2-D curved P3 triangle and 3-D curved Q3 hexahedron")
            .formulation("H1 Lagrange P3 isoparametric geometry with production stiffness and mass assembly")
            .seed("fem-h1-p3-curved-geometry-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(
            CASE_ID,
            format!("explicit P3 H1 geometry reference panicked: {error:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("fem-h1-p3-curved-geometry-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
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
