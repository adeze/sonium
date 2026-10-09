//! Release-backed compressed BEM Hermitian-adjoint field parity contract.

#[path = "../../sonium-bem/tests/compressed_adjoint_field_parity.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.adjoint-compressed-backends.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(
        packet::run_compressed_adjoint_field_parity,
    ));
    let result = match outcome {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("backend_count", metrics.backend_count as f64)
    .metric("matrix_dimension", metrics.matrix_dimension as f64)
    .metric("maximum_residual", metrics.maximum_residual)
    .metric(
        "maximum_exact_reference_residual",
        metrics.maximum_exact_reference_residual,
    )
    .metric("maximum_field_error", metrics.maximum_field_error)
            .metric(
                "invalid_point_rejections",
                metrics.invalid_point_rejections as f64,
            )
            .minimum("backend_count", 4.0)
            .minimum("matrix_dimension", 1.0)
    .maximum("maximum_residual", 1.0e-6)
    .maximum("maximum_exact_reference_residual", 1.0e-4)
    .maximum("maximum_field_error", 1.0e-6)
            .minimum("invalid_point_rejections", 8.0)
            .input_checksum("analytic/deterministic-fixture:bem-adjoint-compressed-backends-v1")
            .reference(
                "independent dense adjoint field, compressed and exact residual, checked invalid-point rejection",
            )
            .reference_checksum("analytic/deterministic-fixture:bem-adjoint-compressed-backends-v1")
            .dimension("3-D 48-DOF RHS; two-frequency H-matrix, SLFMM, MLFMM, directional-MLFMM adjoint field parity")
            .formulation("production high-level BemSolver compressed Hermitian-adjoint dispatch")
            .tolerance(
                "compressed residual <= 1e-6; exact-reference residual <= 1e-4; field error <= 1e-6; all backend/frequency points reject invalid input",
            )
    .seed("bem-adjoint-compressed-backends:cpu:v3")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(CASE_ID, "panic during compressed adjoint field parity validation")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
