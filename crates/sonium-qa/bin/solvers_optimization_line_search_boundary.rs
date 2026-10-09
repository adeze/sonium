//! Checked admission boundary for the public optimization line searches.

use sonium_qa::QaResult;
use sonium_solvers::SolverStatus;
use sonium_solvers::optimization::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-line-search-boundary.v1";
const CHECKS: f64 = 9.0;

fn run() -> Result<(), String> {
    let objective = |x: &[f64]| 0.5 * x.iter().map(|value| value * value).sum::<f64>();
    let gradient = |x: &[f64]| x.to_vec();
    let x = [1.0, 1.0];
    let direction = [-1.0, -1.0];
    let gradient_at_x = [1.0, 1.0];

    let mut backtracking_config = BacktrackingConfig::default();
    backtracking_config.c1 = f64::NAN;
    let result = backtracking(
        &objective,
        &x,
        &direction,
        1.0,
        &gradient_at_x,
        &backtracking_config,
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("backtracking accepted a non-finite c1 configuration".into());
    }

    let result = backtracking(
        &objective,
        &x,
        &direction,
        f64::NAN,
        &gradient_at_x,
        &BacktrackingConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("backtracking accepted a non-finite objective value".into());
    }

    let strong_wolfe_config = StrongWolfeConfig {
        alpha_init: f64::NAN,
        ..StrongWolfeConfig::default()
    };
    let result = strong_wolfe(
        &objective,
        &gradient,
        &x,
        &direction,
        1.0,
        &gradient_at_x,
        &strong_wolfe_config,
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("strong-Wolfe accepted a non-finite initial step".into());
    }

    let result = strong_wolfe(
        &objective,
        &gradient,
        &[f64::NAN, 1.0],
        &direction,
        1.0,
        &gradient_at_x,
        &StrongWolfeConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("strong-Wolfe accepted a non-finite initial point".into());
    }

    let hager_zhang_config = HagerZhangConfig {
        sigma: 0.05,
        ..HagerZhangConfig::default()
    };
    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &x,
        &direction,
        1.0,
        &gradient_at_x,
        &hager_zhang_config,
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("Hager-Zhang accepted an invalid curvature configuration".into());
    }

    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &x,
        &direction,
        1.0,
        &[f64::NAN, 1.0],
        &HagerZhangConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("Hager-Zhang accepted a non-finite initial gradient".into());
    }

    let result = backtracking(
        &objective,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &BacktrackingConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError || result.fun_evals != 0 {
        return Err("backtracking accepted mismatched vector dimensions".into());
    }

    let result = strong_wolfe(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &StrongWolfeConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError
        || result.fun_evals != 0
        || result.grad_evals != 0
    {
        return Err("strong-Wolfe accepted mismatched vector dimensions".into());
    }

    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &HagerZhangConfig::default(),
    );
    if result.status() != SolverStatus::NumericalError
        || result.fun_evals != 0
        || result.grad_evals != 0
    {
        return Err("Hager-Zhang accepted mismatched vector dimensions".into());
    }

    Ok(())
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok(())) => QaResult::passed(CASE_ID)
            .metric("checked_boundary_cases", CHECKS)
            .metric("backtracking_cases", 3.0)
            .metric("strong_wolfe_cases", 3.0)
            .metric("hager_zhang_cases", 3.0)
            .input_checksum(
                "analytic/deterministic-fixture:solvers-optimization-line-search-boundary-v1",
            )
            .reference("canonical SolverStatus::NumericalError for invalid line-search admission")
            .reference_checksum("analytic")
            .dimension("f64 two-variable quadratic; three public line-search families")
            .formulation(
                "finite initial-state and configuration rejection before callback evaluation",
            )
            .tolerance("9/9 invalid-input cases report NumericalError with zero evaluations")
            .seed("solvers-optimization-line-search-boundary:cpu:v1")
            .timing_ms("line_search_boundary", started.elapsed().as_millis() as u64),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "optimization line-search boundary panicked".to_owned()
            };
            QaResult::failed(CASE_ID, message)
        }
    }
    .requested_backend("cpu")
    .actual_backend("cpu")
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("line-search boundary result must serialize")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
