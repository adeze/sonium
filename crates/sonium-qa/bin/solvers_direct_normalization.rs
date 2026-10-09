//! Release-backed direct solver compatibility/result normalization reference.

#[allow(dead_code)]
#[path = "../../sonium-solvers/crates/sonium-solvers-direct/tests/direct_normalization.rs"]
mod reference;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "solvers.direct-compatibility-normalization.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        reference::dense_facade_normalizes_dimensions_and_non_finite_inputs();
        reference::dense_backward_error_rejects_non_finite_metadata();
        reference::sparse_facade_normalizes_checked_solve_and_preserves_destination();
        reference::sparse_direct_facade_many_rhs_preserves_checked_contract();
        reference::dense_factorization_many_rhs_preserves_checked_contract();
        reference::sparse_multifrontal_adapter_contains_legacy_panicking_solve_path();
        reference::sparse_compatibility_uses_checked_direct_contract_and_matches_reference();
        reference::cholesky_rejects_multi_rhs_shape_overflow_without_panicking();
        reference::sparse_refinement_never_publishes_non_finite_residual_metadata();
        reference::direct_errors_expose_canonical_status_and_accuracy();
        reference::direct_composite_errors_preserve_nested_breakdown_and_aliases();
        reference::lu_multi_rhs_into_rejects_size_overflow();
        reference::dense_single_rhs_preserves_destination_after_lu_failure();
        reference::dense_single_rhs_preserves_destination_after_cholesky_failure();
        reference::dense_single_rhs_preserves_destination_after_qr_failure();
        reference::dense_many_rhs_preserves_destination_after_later_lu_failure();
        reference::dense_many_rhs_preserves_destination_after_later_cholesky_failure();
        reference::dense_many_rhs_preserves_destination_after_later_qr_failure();
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("direct_normalization_checks", 18.0)
            .minimum("direct_normalization_checks", 18.0)
            .metric("canonical_status_and_accuracy", 1.0)
            .minimum("canonical_status_and_accuracy", 1.0)
            .metric("transactional_error_contract", 1.0)
            .minimum("transactional_error_contract", 1.0)
            .input_checksum("analytic/deterministic-fixture:solvers-direct-normalization-v1")
            .reference("known-solution CSR residuals, dimension/input rejection, status normalization, and transactional destination preservation")
            .reference_checksum("analytic/deterministic-fixture:solvers-direct-normalization-v1")
            .dimension("dense, sparse, multifrontal, Cholesky, refinement, and multi-RHS direct solver adapters")
            .formulation("checked DirectFactorization/SparseFactorization compatibility facade with canonical SolverFailure")
            .tolerance("eighteen direct normalization and failure-boundary checks pass")
            .seed("solvers-direct-compatibility-normalization:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("direct compatibility normalization panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
