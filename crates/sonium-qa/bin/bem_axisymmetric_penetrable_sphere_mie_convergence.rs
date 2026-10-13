//! Release-backed reduced axisymmetric penetrable-sphere Mie convergence.

// Rust guideline compliant 2026-02-21

#[path = "../../sonium-bem/tests/axisymmetric_penetrable_sphere_mie_convergence.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.axisymmetric-penetrable-sphere-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        packet::run_axisymmetric_penetrable_sphere_mie_convergence()
    }));
    let mut result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("coarse_exterior_error", metrics.coarse_exterior_error)
            .metric("fine_exterior_error", metrics.fine_exterior_error)
            .metric("exterior_refinement_ratio",metrics.fine_exterior_error/metrics.coarse_exterior_error).maximum("exterior_refinement_ratio",0.1)
            .metric("interior_refinement_ratio",metrics.fine_interior_error/metrics.coarse_interior_error).maximum("interior_refinement_ratio",0.1)
            .metric("reference_truncation_change",metrics.reference_change).maximum("reference_truncation_change",1e-10)
            .metric("numerical_solve_executed",2.0).minimum("numerical_solve_executed",2.0)
            .metric("coarse_interior_error", metrics.coarse_interior_error)
        .metric("fine_interior_error", metrics.fine_interior_error)
        .metric("coarse_far_field_error", metrics.coarse_far_field_error)
        .metric("fine_far_field_error", metrics.fine_far_field_error)
        .metric(
            "far_field_refinement_ratio",
            metrics.fine_far_field_error / metrics.coarse_far_field_error,
        )
        .metric("fine_residual", metrics.fine_residual)
        .metric("forward_solution_error", metrics.forward_solution_error)
        .metric("pressure_continuity_error", metrics.pressure_continuity_error)
        .metric("velocity_continuity_error", metrics.velocity_continuity_error)
        .metric("adjoint_field_norm", metrics.adjoint_field_norm)
        .metric(
            "adjoint_invalid_point_rejections",
            metrics.adjoint_invalid_point_rejections as f64,
        )
        .metric("adjoint_residual", metrics.adjoint_residual)
        .metric("adjoint_batch_residual", metrics.adjoint_batch_residual)
        .metric("inner_product_error", metrics.inner_product_error)
        .metric("compressed_action_error", metrics.compressed_action_error)
        .metric(
            "compressed_block_count",
            metrics.compressed_block_count as f64,
        )
            .minimum("coarse_exterior_error", 0.0)
            .maximum("fine_exterior_error", 1e-4)
            .minimum("coarse_interior_error", 0.0)
        .maximum("fine_interior_error", 1e-4)
        .minimum("coarse_far_field_error", 0.0)
        .maximum("fine_far_field_error", 1e-5)
        .maximum("far_field_refinement_ratio", 0.01)
        .maximum("fine_residual", 1.0e-10)
        .maximum("forward_solution_error", 1.0e-10)
        .maximum("pressure_continuity_error", 1.0e-10)
        .maximum("velocity_continuity_error", 1.0e-10)
        .minimum("adjoint_field_norm", 0.0)
        .minimum("adjoint_invalid_point_rejections", 1.0)
            .maximum("adjoint_residual", 1.0e-10)
        .maximum("adjoint_batch_residual", 1.0e-10)
        .maximum("inner_product_error", 2.0e-12)
        .maximum("compressed_action_error", 2.0e-2)
        .minimum("compressed_block_count", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-axisymmetric-penetrable-sphere-v1")
            .reference("independent FluidSphereTransmission3d Mie pressure reference")
            .reference_checksum("analytic/deterministic-fixture:bem-axisymmetric-penetrable-sphere-v1")
            .dimension("axisymmetric sphere radius 0.2 m; 8/16 meridional points; 64 azimuthal samples")
            .formulation("production reduced ring-source CPU two-medium transmission BEM")
            .tolerance("near exterior/interior errors <= 1e-4; far field <= 1e-5; near/interior refinement ratios <=0.1; Mie truncation change <=1e-10; far-field refinement ratio <= 0.01; forward/adjoint residuals <= 1e-10; inner-product error <= 2e-12")
            .seed("bem-axisymmetric-penetrable-sphere-mie:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "axisymmetric penetrable-sphere Mie convergence panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    if result.status == sonium_qa::QaStatus::Passed
        && let Err(error) = result.release_eligible()
    {
        result = QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu");
    }
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA JSON serialization")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
