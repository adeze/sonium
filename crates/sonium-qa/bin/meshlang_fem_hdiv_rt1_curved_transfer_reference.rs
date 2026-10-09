//! Release-style MeshLang-to-FEM curved RT1 H(div) transfer reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/meshlang_hdiv_rt1_curved_transfer.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hdiv-rt1-curved-transfer-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::meshlang_hdiv_rt1_bridge_curved_reference);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok((max_error, transfer_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("affine_to_curved_checks", (transfer_checks / 2) as f64)
            .metric("curved_to_affine_checks", (transfer_checks / 2) as f64)
            .metric("max_bridge_reference_error", max_error)
            .minimum("affine_to_curved_checks", 1.0)
            .minimum("curved_to_affine_checks", 1.0)
            .maximum("max_bridge_reference_error", 2.0e-10)
            .input_checksum("mesh:affine-p1-curved-p2:meshlang-hdiv-rt1-v1")
            .reference(
                "curved RT1 edge/cell moments agree with the checked kernel in both directions",
            )
            .reference_checksum("analytic/deterministic-fixture:meshlang-fem-hdiv-rt1-curved-v1")
            .dimension("2-D affine P1 and curved P2 triangular RT1 H(div) state")
            .formulation("IrStateTransferPlan RT1 H(div) curved FEM consumer")
            .tolerance("all MeshLang bridge values agree with the checked RT1 kernel within 2e-10")
            .seed("meshlang-fem-hdiv-rt1-curved-transfer-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved RT1 reference panicked: {}", panic_message(payload)),
        )
        .elapsed_ms(elapsed_ms),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA_RESULT serialization")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}
