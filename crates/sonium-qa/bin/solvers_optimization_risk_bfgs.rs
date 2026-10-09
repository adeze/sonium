//! Independent CPU QA for the checked risk-objective/BFGS integration seam.

use sonium_qa::QaResult;
use sonium_solvers::{
    BfgsConfig, OptimError, RiskAggregation, RiskAggregationError, RiskObjective,
    ScenarioEvaluation, bfgs_risk,
};
use std::cell::Cell;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-risk-bfgs.v1";
const TOLERANCE: f64 = 1.0e-8;

fn scenarios(x: &[f64]) -> Result<ScenarioEvaluation<f64>, RiskAggregationError> {
    let value = x
        .first()
        .copied()
        .ok_or(RiskAggregationError::DimensionMismatch {
            expected: 1,
            got: 0,
        })?;
    let delta = value - 1.0;
    let loss = delta * delta;
    Ok(ScenarioEvaluation::new(
        vec![loss, loss + 0.5, loss + 1.0],
        vec![vec![2.0 * delta]; 3],
    ))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let calls = Cell::new(0_usize);
    let objective = RiskObjective::new(
        |x: &[f64]| {
            calls.set(calls.get() + 1);
            scenarios(x)
        },
        RiskAggregation::Cvar { confidence: 0.5 },
    );
    let initial = objective
        .evaluate(&[3.0])
        .map_err(|error| format!("risk preflight failed: {error}"))?;
    let initial_value = initial.value;
    let initial_gradient = initial.gradient[0];
    let _ = objective
        .value(&[3.0])
        .map_err(|error| format!("cached risk value failed: {error}"))?;
    let _ = objective
        .gradient(&[3.0])
        .map_err(|error| format!("cached risk gradient failed: {error}"))?;
    let cache_error = if calls.get() == 1 { 0.0 } else { 1.0 };

    let result = bfgs_risk(
        scenarios,
        &[3.0],
        &BfgsConfig::default(),
        RiskAggregation::Ks { sharpness: 8.0 },
    )
    .map_err(|error| format!("risk BFGS failed: {error}"))?;
    let solution_error = (result.x[0] - 1.0).abs();
    let objective_error =
        (result.fun - (1.0 + (1.0 + (-4.0_f64).exp() + (-8.0_f64).exp()).ln() / 8.0)).abs();
    let gradient_error = result.grad_norm.abs();
    let convergence_error = if result.converged { 0.0 } else { 1.0 };
    let finite_error = if result.fun.is_finite() && result.x[0].is_finite() {
        0.0
    } else {
        1.0
    };

    let malformed = bfgs_risk(
        |_| Ok(ScenarioEvaluation::new(vec![1.0], vec![vec![1.0, 0.0]])),
        &[0.0_f64],
        &BfgsConfig::default(),
        RiskAggregation::Cvar { confidence: 0.5 },
    );
    let malformed_error = if matches!(
        malformed,
        Err(OptimError::DimensionMismatch {
            expected: 1,
            got: 2
        })
    ) {
        0.0
    } else {
        1.0
    };

    let metrics = vec![
        ("initial_value_error", (initial_value - (29.0 / 6.0)).abs()),
        ("initial_gradient_error", (initial_gradient - 4.0).abs()),
        ("cached_callback_error", cache_error),
        ("solution_error", solution_error),
        ("objective_error", objective_error),
        ("gradient_error", gradient_error),
        ("convergence_error", convergence_error),
        ("finite_result_error", finite_error),
        ("malformed_rejection_error", malformed_error),
    ];
    if metrics
        .iter()
        .any(|(_, value)| !value.is_finite() || *value > TOLERANCE)
    {
        return Err(format!("risk BFGS integration mismatch: {metrics:?}"));
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
                .reference("independent analytic scenario quadratic with CVaR/KS gradients")
                .input_checksum("analytic/deterministic-fixture:optimization-risk-bfgs-v1")
                .reference_checksum("analytic/deterministic-fixture:optimization-risk-bfgs-v1")
                .formulation("cached checked risk objective adapter and BFGS integration")
                .dimension("one design variable; three scenario losses")
                .tolerance(
                    "all objective, gradient, convergence, cache, and rejection errors <= 1e-8",
                );
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            result = result
                .seed("solvers-optimization-risk-bfgs:cpu:v1")
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
