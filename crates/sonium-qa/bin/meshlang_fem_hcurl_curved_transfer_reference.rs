//! Release-style MeshLang-to-FEM curved H(curl) transfer reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/meshlang_hcurl_curved_transfer.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-curved-transfer-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::meshlang_hcurl_bridge_transfers_curved_target_edges();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("curved_edge_moment_checks", 5.0)
            .metric("bridge_reference_parity", 1.0)
            .minimum("curved_edge_moment_checks", 5.0)
            .minimum("bridge_reference_parity", 1.0)
            .input_checksum("mesh:curved-p2-target:meshlang-hcurl-v1")
            .reference("curved target edge moments agree with the checked H(curl) kernel through the MeshLang plan bridge")
            .reference_checksum("analytic/deterministic-fixture:meshlang-fem-hcurl-curved-transfer-v1")
            .dimension("2-D affine P1 source to curved P2 target, order-1 H(curl) edge state")
            .formulation("IrStateTransferPlan order-1 H(curl) curved FEM consumer")
            .tolerance("all transferred edge moments agree within 2e-9")
            .seed("meshlang-fem-hcurl-curved-transfer-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved H(curl) transfer regression panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(elapsed_ms),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA_RESULT serialization")
    );
    let passed = matches!(result.status, QaStatus::Passed);
    println!("{}", if passed { "SUCCESS" } else { "FAILED" });
    if !passed {
        std::process::exit(1);
    }
}
