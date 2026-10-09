//! Release-facing P3 tetrahedral trace/orientation reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/p3_tetra_trace_orientation.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.p3-tetra-trace-orientation.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| reference::p3_tetra_trace_orientation_reference());
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent cubic polynomial and shared-face continuity reference")
            .reference_checksum("analytic/deterministic-fixture:fem-p3-tetra-trace-orientation-v1")
            .dimension(
                "two positively oriented 3D tetrahedra with reversed local edge/face ordering",
            )
            .formulation(
                "production P3 H1 orientation-aware global interpolation and face trace evaluation",
            )
            .tolerance("two reference checks pass below the focused trace threshold")
            .metric("reference_checks", 2.0)
            .minimum("reference_checks", 2.0)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("P3 tetrahedral trace/orientation reference panicked: {payload:?}"),
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
