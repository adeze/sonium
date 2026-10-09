//! Release-backed CPU 2-D penetrable-cylinder BEM reference.

#[path = "../../sonium-bem/tests/cpu_2d_fluid_cylinder.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.2d-fluid-cylinder-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        packet::run_production_2d_transmission_matches_cylindrical_reference();
        packet::run_production_2d_transmission_matches_cylindrical_far_field();
        packet::run_production_2d_transmission_rejects_invalid_field_queries();
    }));
    let passed = outcome.is_ok();
    let result = if passed {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checks_passed", 6.0)
            .minimum("checks_passed", 6.0)
            .input_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-v1")
            .reference(
                "independent cylindrical Fourier pressure, far-field amplitude, and power reference",
            )
            .reference_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-v1")
            .dimension("2-D radius 0.4 m; exterior k=2/rho=1.21; interior k=3/rho=1.5; 128 panels")
            .formulation("production dense constant-panel CPU Müller transmission BEM")
            .tolerance(
                "exterior/interior fields, interface traces, far-field amplitude, power, and invalid-query admission pass documented bounds",
            )
            .seed("bem-2d-fluid-cylinder-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(CASE_ID, "2-D penetrable-cylinder reference packet panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA result JSON")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
