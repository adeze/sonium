//! Release-facing P1 Helmholtz pollution/convergence reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/helmholtz_plane_wave_pollution.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.helmholtz-plane-wave-pollution.v1";
const REFINEMENT_LEVELS: f64 = 3.0;

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::p1_helmholtz_oblique_plane_wave_converges_against_exact_reference();
        reference::p2_helmholtz_oblique_plane_wave_converges_against_exact_reference();
    });
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent exact oblique complex plane wave")
            .reference_checksum("analytic/deterministic-fixture:fem-helmholtz-plane-wave-v1")
            .dimension("2D unit square; k=8; three refinement levels")
            .formulation(
                "P1/P2 Helmholtz production path and exact oblique plane-wave field evaluation",
            )
            .tolerance("focused test convergence and finite residual/phase gates pass")
            .metric("refinement_levels", REFINEMENT_LEVELS)
            .minimum("refinement_levels", REFINEMENT_LEVELS)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("P1 Helmholtz plane-wave reference panicked: {payload:?}"),
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
