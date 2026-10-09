//! Release-backed RoomSim fast-hybrid/FEM spatial-pattern correlation.

#[allow(dead_code)]
#[path = "../../sonium-sim-room/tests/cross_backend_correlation.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "roomsim.cross-backend-spatial-correlation.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_fast_hybrid_and_fem_correlation);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(correlation) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("normalized_listener_pattern_correlation", correlation)
            .minimum("normalized_listener_pattern_correlation", 0.75)
            .metric("backend_outputs_finite", 1.0)
            .minimum("backend_outputs_finite", 1.0)
            .input_checksum("analytic/deterministic-fixture:roomsim-cross-backend-correlation-v1")
            .reference("normalized listener-pattern Pearson correlation between fast-hybrid and FEM outputs")
            .reference_checksum("analytic/deterministic-fixture:roomsim-cross-backend-correlation-v1")
            .dimension("1.6 x 1.5 x 1.4 m rectangular room; two listeners; 200/400/800 Hz")
            .formulation("production CPU fast-hybrid and FEM RoomSim backends with shared source, boundary, and listener fixture")
            .tolerance("normalized listener-pattern correlation >= 0.75; finite output and spatial metrics")
            .seed("roomsim-cross-backend-correlation:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RoomSim cross-backend correlation panicked: {payload:?}"),
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
