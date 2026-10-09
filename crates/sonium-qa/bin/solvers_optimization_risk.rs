//! Independent CPU QA for chance/CVaR and inequality aggregation primitives.

use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, RiskAggregationError, RiskObjectiveEvaluation, SolverErrorKind, SolverFailure,
    SolverResultContract, SolverStatus, chance_constraint_satisfied, empirical_cvar,
    empirical_cvar_with_gradients, ks_aggregate, ks_aggregate_with_gradients,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-risk-aggregation.v1";
const TOLERANCE: f64 = 1.0e-12;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let losses = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let cvar =
        empirical_cvar(&losses, 0.6).map_err(|error| format!("CVaR evaluation failed: {error}"))?;
    let gradients = vec![
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![4.0, 1.0],
        vec![5.0, 2.0],
    ];
    let (cvar_with_gradient, cvar_gradient) =
        empirical_cvar_with_gradients(&losses, &gradients, 0.6)
            .map_err(|error| format!("CVaR gradient evaluation failed: {error}"))?;
    let ks = ks_aggregate(&[1000.0_f64, 999.0, 997.0], 10.0_f64)
        .map_err(|error| format!("KS evaluation failed: {error}"))?;
    let (ks_with_gradient, ks_gradient) = ks_aggregate_with_gradients(
        &[1000.0_f64, 999.0, 997.0],
        &[vec![1.0_f64, 0.0], vec![0.0, 1.0], vec![2.0, 3.0]],
        10.0_f64,
    )
    .map_err(|error| format!("KS gradient evaluation failed: {error}"))?;
    let chance_pass = chance_constraint_satisfied(&losses, 3.0, 0.4)
        .map_err(|error| format!("chance evaluation failed: {error}"))?;
    let chance_fail = chance_constraint_satisfied(&losses, 3.0, 0.39)
        .map_err(|error| format!("chance evaluation failed: {error}"))?;
    let invalid_rejection = matches!(
        empirical_cvar(&losses, 0.0),
        Err(RiskAggregationError::InvalidConfidence)
    ) && matches!(
        ks_aggregate(&[1.0_f64], 0.0_f64),
        Err(RiskAggregationError::InvalidSharpness)
    );

    let cvar_error = (cvar.value - 4.5).abs();
    let threshold_error = (cvar.threshold - 3.0).abs();
    let cvar_gradient_error = (cvar_gradient[0] - 4.5)
        .abs()
        .max((cvar_gradient[1] - 1.5).abs());
    let cvar_parity_error = (cvar_with_gradient.value - cvar.value).abs();
    let ks_finite_error = if ks.value.is_finite() { 0.0 } else { 1.0 };
    let ks_gradient_error = (ks_gradient[0]
        - (ks_with_gradient.weights[0] + 2.0 * ks_with_gradient.weights[2]))
        .abs()
        .max(
            (ks_gradient[1] - (ks_with_gradient.weights[1] + 3.0 * ks_with_gradient.weights[2]))
                .abs(),
        );
    let weight_sum_error = (cvar.weights.iter().sum::<f64>() - 1.0)
        .abs()
        .max((ks.weights.iter().sum::<f64>() - 1.0).abs());
    let chance_error = if chance_pass && !chance_fail {
        0.0
    } else {
        1.0
    };
    let invalid_error = if invalid_rejection { 0.0 } else { 1.0 };
    let shared_failure_contract_error = [
        (
            RiskAggregationError::DimensionMismatch {
                expected: 2,
                got: 1,
            },
            SolverErrorKind::DimensionMismatch,
        ),
        (
            RiskAggregationError::DesignDimensionMismatch {
                expected: 3,
                got: 2,
            },
            SolverErrorKind::DimensionMismatch,
        ),
        (
            RiskAggregationError::EmptyInput,
            SolverErrorKind::OperationFailed,
        ),
        (
            RiskAggregationError::InvalidConfidence,
            SolverErrorKind::OperationFailed,
        ),
        (
            RiskAggregationError::InvalidSharpness,
            SolverErrorKind::OperationFailed,
        ),
        (
            RiskAggregationError::InvalidProbability,
            SolverErrorKind::OperationFailed,
        ),
        (
            RiskAggregationError::NonFiniteInput,
            SolverErrorKind::OperationFailed,
        ),
        (
            RiskAggregationError::CallbackPanicked,
            SolverErrorKind::OperationFailed,
        ),
    ]
    .into_iter()
    .filter(|(error, kind)| {
        error.error_kind() == *kind
            && error.status() == SolverStatus::NumericalError
            && error.accuracy_status() == sonium_solvers::AccuracyStatus::NotChecked
    })
    .count() as f64;
    let cvar_contract_error = if cvar.validate_contract().is_ok() {
        0.0
    } else {
        1.0
    };
    let cvar_status_error = if cvar.solver_status() == SolverStatus::Converged
        && cvar.accuracy_status() == AccuracyStatus::Attained
    {
        0.0
    } else {
        1.0
    };
    let ks_contract_error = if ks.validate_contract().is_ok() {
        0.0
    } else {
        1.0
    };
    let evaluation = RiskObjectiveEvaluation {
        value: cvar.value,
        gradient: cvar_gradient.clone(),
        weights: cvar.weights.clone(),
    };
    let evaluation_contract_error = if evaluation.validate_contract().is_ok() {
        0.0
    } else {
        1.0
    };
    let mut malformed = cvar.clone();
    malformed.weights[0] = f64::NAN;
    let malformed_contract_rejection = if malformed.validate_contract().is_err() {
        0.0
    } else {
        1.0
    };
    let metrics = vec![
        ("cvar_error", cvar_error),
        ("cvar_threshold_error", threshold_error),
        ("cvar_gradient_error", cvar_gradient_error),
        ("cvar_gradient_parity_error", cvar_parity_error),
        ("ks_finite_error", ks_finite_error),
        ("ks_gradient_error", ks_gradient_error),
        ("normalized_weight_error", weight_sum_error),
        ("chance_constraint_error", chance_error),
        ("invalid_input_rejection_error", invalid_error),
        (
            "shared_failure_contract_error",
            (8.0 - shared_failure_contract_error).abs(),
        ),
        ("cvar_result_contract_error", cvar_contract_error),
        ("cvar_result_status_error", cvar_status_error),
        ("ks_result_contract_error", ks_contract_error),
        ("risk_evaluation_contract_error", evaluation_contract_error),
        (
            "malformed_result_contract_error",
            malformed_contract_rejection,
        ),
    ];
    if metrics.iter().any(|(_, value)| *value > TOLERANCE) {
        return Err(format!("risk aggregation mismatch: {metrics:?}"));
    }
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent empirical CVaR/KS/chance oracle")
                .input_checksum("analytic/deterministic-fixture:optimization-risk-v1")
                .reference_checksum("analytic/deterministic-fixture:optimization-risk-v1")
                .formulation("fractional-tail CVaR, stable KS maximum, and empirical chance gate")
                .dimension("five scalar losses; two-component scenario gradients")
                .tolerance(
                    "all risk value, gradient, weight, chance, and rejection errors <= 1e-12",
                );
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            result = result
                .seed("solvers-optimization-risk:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA JSON must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA JSON must serialize")
            );
            eprintln!("FAILED");
            std::process::exit(1);
        }
    }
}
