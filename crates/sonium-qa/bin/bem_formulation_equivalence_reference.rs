//! Release-facing regular/Burton-Miller formulation equivalence reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/formulation_equivalence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.formulation-equivalence-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::regular_and_burton_miller_match_independent_mie_reference();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("regular_burton_miller_equivalence", 1.0)
            .minimum("regular_burton_miller_equivalence", 1.0)
            .metric("independent_mie_reference", 1.0)
            .minimum("independent_mie_reference", 1.0)
            .input_checksum("bem:rigid-sphere:ka-0.5:regular-burton-miller")
            .reference("independent rigid-sphere Mie field and regular/Burton-Miller equivalence")
            .reference_checksum("analytic/deterministic-fixture:bem-formulation-equivalence-v1")
            .formulation("production constant-panel TBEM regular and Burton-Miller formulations")
            .dimension("3-D rigid sphere; ka=0.5; four exterior evaluation angles")
            .tolerance("Mie relative field error <= 1.5e-1; formulation relative error <= 8e-2")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("formulation-equivalence reference panicked: {payload:?}"),
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
