//! Release-backed CPU 2-D constant-panel rigid-cylinder BEM reference.

#[path = "../../sonium-bem/tests/cpu_2d_rigid_cylinder.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.2d-rigid-cylinder-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        packet::run_production_2d_line_bem_matches_cylinder_field_and_power();
        packet::run_production_2d_line_bem_matches_cylinder_far_field();
        packet::run_production_2d_rigid_rejects_invalid_field_queries();
        packet::run_production_2d_power_observes_xem_peak_rms_convention();
    }));
    let passed = outcome.is_ok();
    let result = if passed {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checks_passed", 4.0)
            .minimum("checks_passed", 4.0)
            .input_checksum("analytic/deterministic-fixture:bem-2d-rigid-cylinder-v1")
            .reference(
                "independent 2-D rigid-cylinder series field/far-field amplitude and RMS/peak per-length power",
            )
            .reference_checksum("analytic/deterministic-fixture:bem-2d-rigid-cylinder-v1")
            .dimension("2-D radius 0.4 m constant-panel cylinder; k=2; 96/128 panels")
            .formulation(
                "production constant-panel CPU 2-D rigid-cylinder BEM with typed XEM medium",
            )
            .tolerance(
                "four field/far-field/power and RMS/peak plus invalid-query admission checks pass",
            )
            .seed("bem-2d-rigid-cylinder-reference:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(CASE_ID, "2-D rigid-cylinder reference assertion failed")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64)
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("2-D BEM result must serialize")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
