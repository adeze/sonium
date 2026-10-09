//! Release-backed discontinuous-linear and quadratic BEM panel reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/higher_order_formulation_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-formulation-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::linear_panels_converge_to_mie_and_preserve_formulation_and_axisymmetry();
        reference::quadratic_panels_run_against_independent_mie_reference();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_mie_reference", 1.0)
            .minimum("independent_mie_reference", 1.0)
        .metric("linear_panel_convergence", 1.0)
        .minimum("linear_panel_convergence", 1.0)
        .metric("quadratic_panel_mie_reference", 1.0)
        .minimum("quadratic_panel_mie_reference", 1.0)
            .metric("regular_burton_miller_equivalence", 1.0)
            .minimum("regular_burton_miller_equivalence", 1.0)
            .metric("axisymmetry_invariant", 1.0)
            .minimum("axisymmetry_invariant", 1.0)
        .input_checksum("bem:rigid-sphere:ka-0.35:linear-and-quadratic-panel-reference")
            .reference("independent spherical Mie fields, checked field evaluation, linear/quadratic panel paths, formulation equivalence, and axial invariance")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-formulation-v1")
            .dimension("3-D rigid sphere; discontinuous-linear and quadratic Tri3 TBEM panels")
            .formulation("production CPU TBEM linear/quadratic panel bases with regular and Burton-Miller rigid formulations")
            .tolerance("linear fine Mie/formulation thresholds plus bounded quadratic Mie reference")
            .seed("bem-higher-order-formulation-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("higher-order/formulation reference panicked: {payload:?}"),
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
