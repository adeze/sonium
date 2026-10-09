//! Release-style MeshLang-to-FEM curved H1 transfer reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/meshlang_h1_curved_transfer.rs"]
mod reference;

use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-curved-transfer-reference.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let p2 = reference::meshlang_h1_bridge_transfers_curved_p2_layouts();
        let p3 = reference::meshlang_h1_bridge_transfers_curved_p3_layouts();
        let mixed = reference::meshlang_h1_bridge_transfers_mixed_curved_layouts();
        (p2, p3, mixed)
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok((
            (p2_forward, p2_reverse),
            (p3_forward, p3_reverse),
            (mixed_forward, mixed_reverse),
        )) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("curved_p2_forward_checks", 6.0)
            .metric("curved_p2_reverse_checks", 6.0)
            .metric("curved_p3_forward_checks", 10.0)
            .metric("curved_p3_reverse_checks", 10.0)
            .metric("mixed_p1_to_curved_p2_checks", 6.0)
            .metric("mixed_curved_p2_to_p1_checks", 3.0)
            .metric("max_curved_p2_error", p2_forward.max(p2_reverse))
            .metric("max_curved_p3_error", p3_forward.max(p3_reverse))
            .metric("max_mixed_curved_error", mixed_forward.max(mixed_reverse))
            .minimum("curved_p2_forward_checks", 6.0)
            .minimum("curved_p2_reverse_checks", 6.0)
            .minimum("curved_p3_forward_checks", 10.0)
            .minimum("curved_p3_reverse_checks", 10.0)
            .minimum("mixed_p1_to_curved_p2_checks", 6.0)
            .minimum("mixed_curved_p2_to_p1_checks", 3.0)
            .maximum("max_curved_p2_error", 5.0e-4)
            .maximum("max_curved_p3_error", 5.0e-4)
            .maximum("max_mixed_curved_error", 2.0e-9)
            .input_checksum("curved-p2-p3-triangle-layouts:meshlang-h1-v1")
            .reference_checksum("analytic-quadratic-cubic-fields:meshlang-h1-curved-v1")
            .dimension("2-D curved P2/P3 H1 MeshLang transfer")
            .formulation("IrStateTransferPlan curved/isoparametric H1 FEM consumer")
            .tolerance("affine-to-curved exact; curved-to-affine bounded interpolation error")
            .seed("meshlang-fem-h1-curved-transfer-reference:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("curved H1 transfer regression panicked: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(elapsed_ms),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA_RESULT JSON serialization")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
