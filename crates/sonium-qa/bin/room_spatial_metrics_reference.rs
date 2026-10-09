//! Release-backed RoomSim spatial RMS/SPL metric reference.

#[allow(dead_code)]
#[path = "../../sonium-sim-room/tests/room_metrics_reference.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "roomsim.spatial-metrics-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::known_listener_levels_reproduce_spatial_rms_and_spread_independently();
        reference::metric_reproduction_is_invariant_to_listener_order_and_phase();
        reference::fast_hybrid_output_reproduces_spatial_rms_independently();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("independent_metric_checks", 3.0)
            .minimum("independent_metric_checks", 3.0)
            .metric("listener_order_phase_invariance", 1.0)
            .minimum("listener_order_phase_invariance", 1.0)
            .metric("fast_hybrid_spatial_rms", 1.0)
            .minimum("fast_hybrid_spatial_rms", 1.0)
            .input_checksum("analytic/deterministic-fixture:roomsim-spatial-metrics-v1")
            .reference("independent RMS reduction, peak/RMS conversion, listener-order/phase invariance, and production fast-hybrid output")
            .reference_checksum("analytic/deterministic-fixture:roomsim-spatial-metrics-v1")
            .dimension("two-listener analytic identities plus three-listener fast-hybrid spatial output")
            .formulation("backend-neutral RoomSim spatial RMS/SPL and listener-spread metrics")
            .tolerance("three independent metric checks finite and equal to analytic reductions")
            .seed("roomsim-spatial-metrics-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("RoomSim spatial-metrics reference panicked: {payload:?}"),
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
