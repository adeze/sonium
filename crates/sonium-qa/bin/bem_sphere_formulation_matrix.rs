//! Release-facing rigid-sphere backend/formulation equivalence reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/formulation_equivalence.rs"]
mod formulation_equivalence;
#[allow(dead_code)]
#[path = "../../sonium-bem/tests/sphere_reciprocity_passivity.rs"]
mod sphere_reciprocity_passivity;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.sphere-backend-formulation-matrix.v1";
const FORMULATION_COUNT: f64 = 8.0;

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        sphere_reciprocity_passivity::bem_sphere_backend_formulation_matrix_v1();
        formulation_equivalence::run_regular_and_burton_miller_reference();
    });

    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent rigid-sphere Mie scattering series")
            .reference_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
            .input_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
            .dimension("3D rigid sphere; ka=0.5; eight backend/formulation rows")
            .formulation("four CPU assembly backends × regular/Burton-Miller exterior formulation")
            .tolerance("all eight rows satisfy the production test gates")
            .metric("backend_formulations_checked", FORMULATION_COUNT)
            .minimum("backend_formulations_checked", FORMULATION_COUNT)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("backend/formulation matrix reference panicked: {payload:?}"),
        )
        .elapsed_ms(started.elapsed().as_millis() as u64),
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
