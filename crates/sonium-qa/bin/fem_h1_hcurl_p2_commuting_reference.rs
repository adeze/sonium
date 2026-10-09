include!("../../sonium-fem/tests/h1_hcurl_p2_commuting_reference.rs");

// Release-backed affine P2 H¹ to order-2 H(curl) commuting reference.

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "fem.h1-hcurl-p2-commuting-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(
        run_p2_gradient_commutes_against_independent_reference_on_oriented_patch,
    ));
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checks_passed", 1.0)
            .minimum("checks_passed", 1.0)
            .input_checksum("analytic/deterministic-fixture:fem-h1-hcurl-p2-commuting-v1")
            .reference("independent analytical P2 gradient edge moments and production order-2 H(curl) oriented patch")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-hcurl-p2-commuting-v1")
            .dimension("2D affine P2 triangle; H¹ quadratic scalar and order-2 Nédélec H(curl) spaces")
            .formulation("independent P2 H¹ gradient reconstruction against production oriented order-2 H(curl) tabulation")
            .tolerance("one independent P2 commuting-reference check passes")
            .seed("fem-h1-hcurl-p2-commuting-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "P2 H¹/H(curl) commuting reference panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("P2 commuting QA result must serialize")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
