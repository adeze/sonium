//! Release-backed 2-D transmission matrix and Hermitian-adjoint contract.

#[path = "../../sonium-bem/tests/cpu_2d_transmission_adjoint.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.2d-transmission-adjoint.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        packet::run_two_d_transmission_adjoint_contract()
    }));
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("matrix_dimension", metrics.matrix_dimension as f64)
            .metric("forward_residual", metrics.forward_residual)
            .metric("forward_solution_error", metrics.forward_solution_error)
            .metric("adjoint_field_norm", metrics.adjoint_field_norm)
            .metric(
                "adjoint_invalid_point_rejections",
                metrics.adjoint_invalid_point_rejections as f64,
            )
            .metric("adjoint_residual", metrics.adjoint_residual)
            .metric("adjoint_batch_residual", metrics.adjoint_batch_residual)
            .metric("inner_product_error", metrics.inner_product_error)
            .minimum("matrix_dimension", 16.0)
            .maximum("forward_residual", 1.0e-10)
            .maximum("forward_solution_error", 1.0e-10)
            .minimum("adjoint_field_norm", 0.0)
            .minimum("adjoint_invalid_point_rejections", 1.0)
            .maximum("adjoint_residual", 1.0e-10)
            .maximum("adjoint_batch_residual", 1.0e-10)
            .maximum("inner_product_error", 2.0e-12)
            .input_checksum("analytic/deterministic-fixture:bem-2d-transmission-adjoint-v1")
            .reference("independent assembled transmission matrix and Hermitian-adjoint identity")
            .reference_checksum("analytic/deterministic-fixture:bem-2d-transmission-adjoint-v1")
            .dimension("2-D circular fluid interface; 32 constant panels; 64x64 operator")
            .formulation("production dense constant-panel CPU Müller transmission BEM")
            .tolerance("forward and adjoint residuals <= 1e-10; inner-product error <= 2e-12")
            .seed("bem-2d-transmission-adjoint:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "2-D transmission adjoint contract panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
