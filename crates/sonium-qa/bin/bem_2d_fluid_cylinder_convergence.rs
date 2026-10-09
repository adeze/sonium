//! Release-backed 2-D multi-region fluid-cylinder refinement reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/cpu_2d_fluid_cylinder_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.2d-fluid-cylinder-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::multi_region_constant_panel_refinement_converges_in_field_and_power();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("field_refinement_convergence", 1.0)
            .minimum("field_refinement_convergence", 1.0)
            .metric("power_reference_contract", 1.0)
            .minimum("power_reference_contract", 1.0)
            .metric("multi_region_residual_contract", 1.0)
            .minimum("multi_region_residual_contract", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-convergence-v1")
            .reference("independent cylindrical Fourier transmission field and scattered-power-per-length reference")
            .reference_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-convergence-v1")
            .dimension("2-D penetrable fluid cylinder; 64/128/256 constant-panel Müller refinement")
            .formulation("multi-region constant-panel transmission with explicit pressure, power, convention, and provenance contracts")
            .tolerance("field and power errors refine monotonically; fine field errors <= 0.12; residual <= 1e-3")
            .seed("bem-2d-fluid-cylinder-convergence:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("2-D fluid-cylinder convergence reference panicked: {payload:?}"),
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
