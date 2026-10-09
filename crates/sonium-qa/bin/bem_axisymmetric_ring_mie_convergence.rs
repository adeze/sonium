//! Release-style wrapper for reduced axisymmetric ring-source convergence.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/axisymmetric_ring_mie_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.axisymmetric-ring-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_axisymmetric_ring_mie_convergence);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let success = outcome.is_ok();
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_l2_error", metrics.coarse_l2_error)
            .metric("fine_l2_error", metrics.fine_l2_error)
            .maximum("fine_l2_error", 1.0e-7)
            .metric("coarse_pointwise_error", metrics.coarse_pointwise_error)
            .metric("fine_pointwise_error", metrics.fine_pointwise_error)
            .maximum("fine_pointwise_error", 2.0e-7)
            .metric("coarse_far_field_error", metrics.coarse_far_field_error)
            .metric("fine_far_field_error", metrics.fine_far_field_error)
            .maximum("fine_far_field_error", 2.0e-7)
            .metric(
                "l2_refinement_ratio",
                metrics.fine_l2_error / metrics.coarse_l2_error,
            )
            .maximum("l2_refinement_ratio", 0.01)
            .metric(
                "pointwise_refinement_ratio",
                metrics.fine_pointwise_error / metrics.coarse_pointwise_error,
            )
            .maximum("pointwise_refinement_ratio", 0.01)
            .metric(
                "far_field_refinement_ratio",
                metrics.fine_far_field_error / metrics.coarse_far_field_error,
            )
            .maximum("far_field_refinement_ratio", 0.01)
            .metric("coarse_residual", metrics.coarse_residual)
            .maximum("coarse_residual", 1.0e-10)
            .metric("fine_residual", metrics.fine_residual)
            .maximum("fine_residual", 1.0e-10)
            .metric("coarse_dofs", metrics.coarse_dofs as f64)
            .metric("fine_dofs", metrics.fine_dofs as f64)
            .minimum("fine_dofs", (metrics.coarse_dofs + 1) as f64)
            .input_checksum("bem:rigid-sphere:ka-0.35:axisymmetric-rings:12-and-20")
            .reference_checksum("analytic/deterministic-fixture:bem-axisymmetric-ring-mie-v1")
            .dimension("reduced axisymmetric CPU ring-source BEM; rigid sphere")
            .seed("bem-axisymmetric-ring-mie:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "axisymmetric ring-source Mie gate panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference_checksum("analytic/deterministic-fixture:bem-axisymmetric-ring-mie-v1")
            .dimension("reduced axisymmetric CPU ring-source BEM; rigid sphere")
            .seed("bem-axisymmetric-ring-mie:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if success {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
