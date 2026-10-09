//! Release-backed directional FEM PML coefficient contract.

#[path = "../../sonium-fem/tests/pml_directional_contract.rs"]
mod reference;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "fem.pml-directional-contract.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(reference::run_directional_pml_contract));
    let passed = outcome.is_ok();
    let result = if passed {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checks_passed", 28.0)
            .minimum("checks_passed", 28.0)
            .input_checksum("analytic/deterministic-fixture:fem-pml-directional-v1")
            .reference(
                "independent complex coordinate-stretch coefficient identities on all 2-D sides and XY corner",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-pml-directional-v1")
            .dimension("2-D positive/negative X/Y sides and XY polynomial PML corner")
            .formulation(
                "production PmlRegion stretching, mass, and directional stiffness coefficients on all 2-D sides",
            )
            .tolerance("28 finite coefficient/direction checks pass to 1e-12")
            .seed("fem-pml-directional-contract:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(CASE_ID, "directional PML contract panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("PML QA result serializes")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
