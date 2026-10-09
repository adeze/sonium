use std::sync::atomic::{AtomicUsize, Ordering};

use sonium_solvers_core::traits::{IdentityPreconditioner, LinearOperator, SolverError};
use sonium_solvers_iterative::{DeflationSubspace, GmresConfig, try_gmres_deflated_preconditioned};

struct FinalResidualFailure {
    calls: AtomicUsize,
}

impl LinearOperator<f64> for FinalResidualFailure {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        output[0] = if call >= 3 { f64::NAN } else { 1.0 };
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

#[test]
fn deflated_gmres_attributes_final_residual_callback_failure_to_operator() {
    let operator = FinalResidualFailure {
        calls: AtomicUsize::new(0),
    };
    let deflation = DeflationSubspace::try_new(Vec::new(), &operator)
        .expect("an empty deflation subspace is valid");

    let error = try_gmres_deflated_preconditioned(
        &operator,
        &IdentityPreconditioner,
        &deflation,
        &[1.0],
        None,
        &config(),
    )
    .expect_err("final residual callback failure must be typed");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "deflated GMRES operator",
            ..
        }
    ));
    assert!(error.to_string().contains("generated NaN"));
    assert_eq!(operator.calls.load(Ordering::SeqCst), 3);
}
