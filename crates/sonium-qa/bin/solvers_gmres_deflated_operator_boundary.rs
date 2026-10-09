//! Independent CPU evidence for the checked deflated-GMRES final residual boundary.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sonium_qa::QaResult;
use sonium_solvers::traits::{IdentityPreconditioner, LinearOperator};
use sonium_solvers::{
    DeflationSubspace, GmresConfig, SolverError, try_gmres_deflated_preconditioned,
};

const CASE_ID: &str = "solvers.gmres-deflated-operator-boundary.v1";

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

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator = FinalResidualFailure {
        calls: AtomicUsize::new(0),
    };
    let deflation = DeflationSubspace::try_new(Vec::new(), &operator)
        .map_err(|error| format!("empty deflation construction failed: {error:?}"))?;
    let config = GmresConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };

    let error = try_gmres_deflated_preconditioned(
        &operator,
        &IdentityPreconditioner,
        &deflation,
        &[1.0],
        None,
        &config,
    )
    .expect_err("final operator callback failure must be typed");
    if !matches!(
        error,
        SolverError::OperationFailed {
            operation: "deflated GMRES operator",
            ..
        }
    ) {
        return Err(format!("unexpected typed failure: {error:?}"));
    }
    let calls = operator.calls.load(Ordering::SeqCst);
    if calls != 3 {
        return Err(format!(
            "expected exactly three operator calls, got {calls}"
        ));
    }

    Ok(vec![
        ("operator_failure_typed", 1.0),
        ("final_residual_operator_calls", calls as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent 1x1 final-residual callback failure fixture")
                .input_checksum(
                    "analytic/deterministic-fixture:gmres-deflated-operator-boundary-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-deflated-operator-boundary-v1",
                )
                .dimension("1x1 identity-shaped operator; empty deflation subspace; restart=1")
                .tolerance(
                    "final operator callback failure is typed and attributed to the operator",
                )
                .seed("solvers-gmres-deflated-operator-boundary:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name == "operator_failure_typed" {
                    result.minimum(name, 1.0).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent 1x1 final-residual callback failure fixture")
                .input_checksum(
                    "analytic/deterministic-fixture:gmres-deflated-operator-boundary-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-deflated-operator-boundary-v1",
                )
                .tolerance(
                    "final operator callback failure is typed and attributed to the operator",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
