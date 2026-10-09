//! Release-style wrapper for penetrable-sphere multi-region BEM validation.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/penetrable_sphere_mie_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.penetrable-sphere-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_penetrable_sphere_mie_convergence);
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_l2_error", metrics.coarse_l2_error)
            .metric("coarse_pointwise_error", metrics.coarse_pointwise_error)
            .metric("fine_l2_error", metrics.fine_l2_error)
            .maximum("fine_l2_error", 0.005)
            .metric("fine_pointwise_error", metrics.fine_pointwise_error)
            .maximum("fine_pointwise_error", 0.01)
            .metric("fine_solve_residual", metrics.fine_residual)
            .maximum("fine_solve_residual", 1.0e-9)
            .metric(
                "mie_trace_refinement_ratio",
                metrics.fine_mie_trace_residual / metrics.coarse_mie_trace_residual,
            )
            .maximum("mie_trace_refinement_ratio", 0.50)
            .metric("fine_mie_trace_residual", metrics.fine_mie_trace_residual)
            .maximum("fine_mie_trace_residual", 0.03)
            .metric("axisymmetry_error", metrics.axisymmetry_error)
            .maximum("axisymmetry_error", 0.01)
            .metric("active_power_balance", metrics.active_power_balance)
            .maximum("active_power_balance", 0.02)
            .metric("coarse_panels", metrics.coarse_panels as f64)
            .metric("fine_panels", metrics.fine_panels as f64)
            .minimum("fine_panels", (metrics.coarse_panels + 1) as f64)
            .input_checksum("bem:penetrable-fluid-sphere:ka-1:rho-1.21-2.4:c-343-240:icosphere-1-2")
            .reference("independent two-medium spherical-Mie pressure/velocity traces and exterior fields")
            .reference_checksum("analytic/deterministic-fixture:bem-penetrable-sphere-mie-v1")
            .dimension("3-D penetrable lossless fluid sphere at exterior ka=1; 80/320 constant panels")
            .formulation("dense constant-panel Müller pressure/normal-velocity solve; exact-trace operator residual; exterior field and power gates")
            .seed("bem-penetrable-sphere-mie:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "penetrable-sphere Mie convergence packet panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent two-medium spherical-Mie pressure/velocity traces and exterior fields")
            .reference_checksum("analytic/deterministic-fixture:bem-penetrable-sphere-mie-v1")
            .dimension("3-D penetrable lossless fluid sphere at exterior ka=1; 80/320 constant panels")
            .formulation("dense constant-panel Müller pressure/normal-velocity solve; exact-trace operator residual; exterior field and power gates")
            .seed("bem-penetrable-sphere-mie:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if outcome.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
