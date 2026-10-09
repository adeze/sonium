//! Independent CPU differential reference for smooth unconstrained optimization.
//!
//! The objective is a translated diagonal quadratic with a known minimizer.
//! BFGS and L-BFGS are checked against that minimizer and an independently
//! recomputed objective value.

use sonium_qa::QaResult;
use sonium_solvers::{BfgsConfig, LbfgsConfig, LineSearchMethod, bfgs, lbfgs};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-differential-reference.v1";
const TARGET: [f64; 3] = [0.25, -1.5, 2.75];
const WEIGHTS: [f64; 3] = [1.0, 3.0, 7.0];
const START: [f64; 3] = [3.0, -4.0, 0.5];

fn objective(x: &[f64]) -> f64 {
    x.iter()
        .zip(TARGET)
        .zip(WEIGHTS)
        .map(|((&value, target), weight)| 0.5 * weight * (value - target).powi(2))
        .sum()
}

fn gradient(x: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(TARGET)
        .zip(WEIGHTS)
        .map(|((&value, target), weight)| weight * (value - target))
        .collect()
}

fn max_coordinate_error(x: &[f64]) -> f64 {
    x.iter()
        .zip(TARGET)
        .map(|(&value, target)| (value - target).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let x0 = START.to_vec();
    let bfgs_config = BfgsConfig {
        max_iterations: 200,
        grad_tolerance: 1.0e-10,
        fun_tolerance: 1.0e-12,
        step_tolerance: 1.0e-12,
        line_search: LineSearchMethod::StrongWolfe,
    };
    let lbfgs_config = LbfgsConfig {
        max_iterations: 200,
        grad_tolerance: 1.0e-10,
        fun_tolerance: 1.0e-12,
        step_tolerance: 1.0e-12,
        memory: 8,
        line_search: LineSearchMethod::StrongWolfe,
    };

    let bfgs_result = bfgs(&objective, &gradient, &x0, &bfgs_config)
        .map_err(|error| format!("BFGS failed on independent quadratic: {error:?}"))?;
    let lbfgs_result = lbfgs(&objective, &gradient, &x0, &lbfgs_config)
        .map_err(|error| format!("L-BFGS failed on independent quadratic: {error:?}"))?;
    if !bfgs_result.converged || !lbfgs_result.converged {
        return Err(format!(
            "optimizer did not converge: BFGS={}, L-BFGS={}",
            bfgs_result.converged, lbfgs_result.converged
        ));
    }

    let bfgs_coordinate_error = max_coordinate_error(&bfgs_result.x);
    let lbfgs_coordinate_error = max_coordinate_error(&lbfgs_result.x);
    let bfgs_objective_error = objective(&bfgs_result.x);
    let lbfgs_objective_error = objective(&lbfgs_result.x);
    if bfgs_coordinate_error > 1.0e-7
        || lbfgs_coordinate_error > 1.0e-7
        || bfgs_objective_error > 1.0e-12
        || lbfgs_objective_error > 1.0e-12
    {
        return Err(format!(
            "independent quadratic reference failed: BFGS coordinate/objective = {bfgs_coordinate_error:.3e}/{bfgs_objective_error:.3e}, L-BFGS = {lbfgs_coordinate_error:.3e}/{lbfgs_objective_error:.3e}"
        ));
    }
    Ok((
        bfgs_coordinate_error,
        lbfgs_coordinate_error,
        bfgs_objective_error,
        lbfgs_objective_error,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok((
            bfgs_coordinate_error,
            lbfgs_coordinate_error,
            bfgs_objective_error,
            lbfgs_objective_error,
        ))) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("bfgs_coordinate_error", bfgs_coordinate_error)
                .metric("lbfgs_coordinate_error", lbfgs_coordinate_error)
                .metric("bfgs_independent_objective", bfgs_objective_error)
                .metric("lbfgs_independent_objective", lbfgs_objective_error)
                .maximum("bfgs_coordinate_error", 1.0e-7)
                .maximum("lbfgs_coordinate_error", 1.0e-7)
                .maximum("bfgs_independent_objective", 1.0e-12)
                .maximum("lbfgs_independent_objective", 1.0e-12)
                .input_checksum("analytic/deterministic-fixture:translated-diagonal-quadratic-v1")
                .reference("independent translated diagonal quadratic minimizer and objective recomputation")
                .reference_checksum("analytic/deterministic-fixture:translated-diagonal-quadratic-v1")
                .dimension("three-variable positive diagonal quadratic; BFGS and L-BFGS from one non-optimal start")
                .formulation("public BFGS/L-BFGS optimization paths against an independent analytic minimizer")
                .tolerance("both optimizers converge; coordinate error <= 1e-7; independently recomputed objective <= 1e-12")
                .seed("solvers-optimization-differential-reference:cpu:v1")
                .timing_ms("optimization_differential_reference", elapsed_ms)
                .elapsed_ms(elapsed_ms)
        }
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("optimization differential reference panicked: {payload:?}"),
        ),
    };
    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
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
