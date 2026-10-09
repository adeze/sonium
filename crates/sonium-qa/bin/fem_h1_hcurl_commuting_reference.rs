include!("../../sonium-fem/tests/h1_hcurl_commuting_reference.rs");

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "fem.h1-hcurl-commuting-reference.v1";
const CHECK_COUNT: f64 = 3.0;

fn main() {
    let started = Instant::now();
    let checks = [
        catch_unwind(AssertUnwindSafe(
            p1_gradient_commutes_with_affine_nedelec_interpolation,
        )),
        catch_unwind(AssertUnwindSafe(
            affine_nedelec_commuting_reference_is_invariant_under_edge_reversal,
        )),
        catch_unwind(AssertUnwindSafe(
            production_hcurl_p1_gradient_commutes_across_oriented_patch,
        )),
    ];
    let passed = checks.iter().filter(|result| result.is_ok()).count() as f64;

    if passed == CHECK_COUNT {
        let result = QaResult::passed(CASE_ID)
            .reference("independent analytical edge line integrals of the affine scalar gradient")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-hcurl-commuting-v1")
            .dimension("2D affine P1 triangle; H¹ scalar and first-kind Nédélec H(curl) spaces")
            .formulation("analytical affine P1 gradient edge moments, edge reversal, and production oriented patch")
            .tolerance("checks_passed == 3")
            .metric("checks_passed", passed)
            .maximum("checks_passed", CHECK_COUNT)
            .metric("checks_total", CHECK_COUNT)
            .elapsed_ms(started.elapsed().as_millis() as u64);
        println!("QA_RESULT: {}", result.to_json_line().unwrap());
        println!("SUCCESS");
    } else {
        let result = QaResult::failed(
            CASE_ID,
            format!("{passed} of {CHECK_COUNT} independent commuting checks passed"),
        )
        .reference("independent analytical edge line integrals of the affine scalar gradient")
        .reference_checksum("analytic/deterministic-fixture:fem-h1-hcurl-commuting-v1")
        .dimension("2D affine P1 triangle; H¹ scalar and first-kind Nédélec H(curl) spaces")
        .metric("checks_passed", passed)
        .metric("checks_total", CHECK_COUNT)
        .elapsed_ms(started.elapsed().as_millis() as u64);
        println!("QA_RESULT: {}", result.to_json_line().unwrap());
        eprintln!("H1/H(CURL) COMMUTING REFERENCE FAILED");
        println!("FAILED");
        std::process::exit(1);
    }
}
