//! Release-style wrapper for affine-Tri3 Galerkin Müller physical convergence.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/galerkin_muller_mie_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-muller-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_galerkin_muller_mie_convergence);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_l2_error", metrics.coarse_l2_error)
            .metric("coarse_pointwise_error", metrics.coarse_pointwise_error)
            .metric("fine_l2_error", metrics.fine_l2_error)
            .maximum("fine_l2_error", 0.8)
            .metric("fine_pointwise_error", metrics.fine_pointwise_error)
            .maximum("fine_pointwise_error", 1.0)
            .metric("fine_solve_residual", metrics.fine_residual)
            .maximum("fine_solve_residual", 1.0e-8)
            .metric("coarse_solve_residual", metrics.coarse_residual)
            .maximum("coarse_solve_residual", 1.0e-8)
            .metric(
                "field_refinement_ratio",
                metrics.fine_l2_error / metrics.coarse_l2_error.max(f64::MIN_POSITIVE),
            )
            .maximum("field_refinement_ratio", 1.05)
            .metric(
                "pointwise_refinement_ratio",
                metrics.fine_pointwise_error
                    / metrics.coarse_pointwise_error.max(f64::MIN_POSITIVE),
            )
            .maximum("pointwise_refinement_ratio", 1.10)
            .metric("maximum_fine_field_magnitude", metrics.maximum_fine_field_magnitude)
            .minimum("maximum_fine_field_magnitude", f64::MIN_POSITIVE)
            .metric("coarse_panels", metrics.coarse_panels as f64)
            .metric("fine_panels", metrics.fine_panels as f64)
            .minimum("fine_panels", (metrics.coarse_panels + 1) as f64)
            .metric("coarse_dofs", metrics.coarse_dofs as f64)
            .metric("fine_dofs", metrics.fine_dofs as f64)
            .minimum("fine_dofs", (metrics.coarse_dofs + 1) as f64)
            .input_checksum("bem:galerkin-muller-fluid-sphere:ka-0.7:icosphere-0-1")
            .reference("independent two-medium spherical-Mie exterior total pressure")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-muller-mie-convergence-v1")
            .dimension("3-D conforming affine-Tri3 Galerkin Müller; 20/80 panels")
            .formulation("dense shared-DOF Galerkin V/K/K-prime/W assembly, projected incident RHS, checked solve and exterior field")
            .seed("bem-galerkin-muller-mie-convergence:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "Galerkin Müller Mie convergence packet panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent two-medium spherical-Mie exterior total pressure")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-muller-mie-convergence-v1")
            .dimension("3-D conforming affine-Tri3 Galerkin Müller; 20/80 panels")
            .formulation("dense shared-DOF Galerkin V/K/K-prime/W assembly, projected incident RHS, checked solve and exterior field")
            .seed("bem-galerkin-muller-mie-convergence:cpu:v1")
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
