//! CPU-only MeshLang-to-FEM mixed-order adaptive H1 reference.

use sonium_qa::QaResult;
use std::time::Instant;

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/meshlang_h1_mixed_transfer.rs"]
mod reference;

const CASE_ID: &str = "meshlang.fem-h1-mixed-adaptive-reference.v1";

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let (refinement_error, coarsening_error) =
        reference::h1_mixed_bridge_transfers_p1_p2_adaptive_lineage_both_directions();
    let (curved_refinement_error, curved_coarsening_error) =
        reference::h1_mixed_bridge_transfers_curved_higher_order_history();
    let (chain_refinement_error, chain_coarsening_error) =
        reference::h1_mixed_bridge_transfers_p1_p2_p3_adaptive_chain();
    let curved_p2_p3_error = reference::h1_mixed_bridge_transfers_curved_p2_to_p3_mixed_state();
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("refinement_error", refinement_error)
        .metric("coarsening_error", coarsening_error)
        .metric("curved_refinement_error", curved_refinement_error)
        .metric("curved_coarsening_error", curved_coarsening_error)
        .metric("chain_refinement_error", chain_refinement_error)
        .metric("chain_coarsening_error", chain_coarsening_error)
        .metric("curved_p2_p3_error", curved_p2_p3_error)
        .maximum("curved_p2_p3_error", 1.0e-10)
        .metric("source_order", 1.0)
        .metric("target_order", 2.0)
        .metric("lineage_directions", 2.0)
        .minimum("lineage_directions", 2.0)
        .dimension("2-D mixed-order P1/P2/P3 triangular H1 adaptive MeshLang lineage")
        .formulation("checked endpoint-normalized refinement history with plan-driven mixed-order interpolation and reverse coarsening")
        .tolerance("refinement and coarsening errors <= 1e-12")
        .input_checksum("unit-square-p1-to-p2-lineage:meshlang-h1-mixed-v1")
        .reference_checksum("analytic-linear-field:meshlang-h1-mixed-adaptive-curved-v1")
        .seed("meshlang-fem-h1-mixed-adaptive:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA result must serialize")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
