//! Release-style wrapper for the continuous-Q2 physical convergence gate.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/continuous_quadratic_mie_convergence.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.continuous-quadratic-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(reference::run_continuous_quadratic_mie_convergence);
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_l2_error", metrics.coarse_l2_error)
            .metric("coarse_pointwise_error", metrics.coarse_pointwise_error)
            .metric("fine_l2_error", metrics.fine_l2_error)
            .maximum("fine_l2_error", 0.01)
            .metric("fine_pointwise_error", metrics.fine_pointwise_error)
            .maximum("fine_pointwise_error", 0.02)
            .metric(
                "l2_refinement_ratio",
                metrics.fine_l2_error / metrics.coarse_l2_error,
            )
            .maximum("l2_refinement_ratio", 0.85)
            .metric(
                "pointwise_refinement_ratio",
                metrics.fine_pointwise_error / metrics.coarse_pointwise_error,
            )
            .maximum("pointwise_refinement_ratio", 0.85)
            .metric("axisymmetry_error", metrics.axisymmetry_error)
            .maximum("axisymmetry_error", 0.01)
            .metric("coarse_dofs", metrics.coarse_dofs as f64)
            .metric("fine_dofs", metrics.fine_dofs as f64)
            .minimum("fine_dofs", (metrics.coarse_dofs + 1) as f64)
            .input_checksum("bem:rigid-sphere:ka-0.35:continuous-q2:18-and-32-fibonacci-points")
            .reference("independent spherical-Mie exterior fields; two production continuous-Q2 TBEM meshes; rotational-invariance check")
            .reference_checksum("analytic/deterministic-fixture:bem-continuous-q2-mie-v1")
            .dimension("3-D rigid sphere; globally continuous quadratic Tri3 trace; coarse/fine CPU TBEM")
            .formulation("direct dense Burton-Miller solve; shared vertex/edge Q2 DOFs; checked exterior field")
            .seed("bem-continuous-q2-mie:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "continuous-Q2 Mie convergence packet panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent spherical-Mie exterior fields; two production continuous-Q2 TBEM meshes; rotational-invariance check")
            .reference_checksum("analytic/deterministic-fixture:bem-continuous-q2-mie-v1")
            .dimension("3-D rigid sphere; globally continuous quadratic Tri3 trace; coarse/fine CPU TBEM")
            .formulation("direct dense Burton-Miller solve; shared vertex/edge Q2 DOFs; checked exterior field")
            .seed("bem-continuous-q2-mie:cpu:v1")
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
