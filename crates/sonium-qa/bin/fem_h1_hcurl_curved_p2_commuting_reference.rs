//! Release-backed curved-P2 H1 -> H(curl) commuting reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/h1_hcurl_curved_p2_commuting_reference.rs"]
mod reference;

use std::time::Instant;

use sonium_qa::QaResult;

const CASE_ID: &str = "fem.h1-hcurl-curved-p2-commuting-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_curved_p2_h1_hcurl_commuting_reference);
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("curved_commuting_samples", 8.0)
            .minimum("curved_commuting_samples", 8.0)
            .metric("curl_free_gradient_reference", 1.0)
            .minimum("curl_free_gradient_reference", 1.0)
            .input_checksum("mesh:curved-p2-unit-square-triangle:h1-hcurl-commuting-v1")
            .reference("independent reference-triangle scalar gradient and P2 map Jacobian")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-hcurl-curved-p2-v1")
            .dimension("curved 2-D P2 triangles; order-2 H1-to-H(curl); 2 elements x 4 samples")
            .formulation("reference quadratic gradient under production covariant Piola tabulation")
            .tolerance("field and curl residuals <= 3e-11")
            .seed("fem-h1-hcurl-curved-p2-commuting-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved-P2 H1-to-H(curl) reference panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .seed("fem-h1-hcurl-curved-p2-commuting-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("serialize QA result must succeed")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
