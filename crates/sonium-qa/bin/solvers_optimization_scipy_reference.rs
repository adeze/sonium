//! Optional live SciPy differential reference for unconstrained optimization.
//!
//! The fixture is intentionally the same translated diagonal quadratic used
//! by the analytic optimization packet. SciPy is an independent execution
//! reference here; absence of the requested Python/SciPy runtime is an
//! explicit skip unless `SONIUM_REQUIRE_SCIPY=1` is set.

use serde_json::Value;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{BfgsConfig, LbfgsConfig, LineSearchMethod, bfgs, lbfgs};
use std::process::Command;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-scipy-reference.v1";
const TARGET: [f64; 3] = [0.25, -1.5, 2.75];
const WEIGHTS: [f64; 3] = [1.0, 3.0, 7.0];
const START: [f64; 3] = [3.0, -4.0, 0.5];

#[derive(Debug)]
enum RunError {
    Unavailable(String),
    Failed(String),
}

fn objective(x: &[f64]) -> f64 {
    x.iter()
        .zip(TARGET)
        .zip(WEIGHTS)
        .map(|((&value, target), weight)| 0.5 * weight * (value - target).powi(2))
        .sum()
}

fn max_coordinate_error(x: &[f64]) -> f64 {
    x.iter()
        .zip(TARGET)
        .map(|(&value, target)| (value - target).abs())
        .fold(0.0, f64::max)
}

fn scipy_python() -> String {
    std::env::var("SONIUM_SCIPY_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_scipy() -> Result<(Vec<Vec<f64>>, Vec<f64>), RunError> {
    let script = r#"
import json
import numpy as np
from scipy.optimize import minimize

target = np.array([0.25, -1.5, 2.75], dtype=float)
weights = np.array([1.0, 3.0, 7.0], dtype=float)
start = np.array([3.0, -4.0, 0.5], dtype=float)

def fun(x):
    d = x - target
    return float(0.5 * np.dot(weights * d, d))

def jac(x):
    return weights * (x - target)

results = []
for method in ("BFGS", "L-BFGS-B"):
    result = minimize(fun, start, jac=jac, method=method,
                      options={"maxiter": 200, "gtol": 1e-10})
    results.append({"method": method, "success": bool(result.success),
                    "x": [float(v) for v in result.x],
                    "fun": float(result.fun), "nit": int(result.nit)})
print(json.dumps(results))
"#;
    let output = Command::new(scipy_python())
        .args(["-c", script])
        .output()
        .map_err(|error| RunError::Unavailable(format!("execute SciPy Python: {error}")))?;
    if !output.status.success() {
        return Err(RunError::Unavailable(format!(
            "SciPy process failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let line = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| RunError::Failed("SciPy produced no JSON result".into()))?;
    let value: Value = serde_json::from_str(line)
        .map_err(|error| RunError::Failed(format!("parse SciPy JSON: {error}")))?;
    let results = value
        .as_array()
        .ok_or_else(|| RunError::Failed("SciPy result is not an array".into()))?;
    if results.len() != 2 {
        return Err(RunError::Failed(format!(
            "SciPy returned {} methods, expected 2",
            results.len()
        )));
    }
    let mut points = Vec::with_capacity(2);
    let mut objectives = Vec::with_capacity(2);
    for result in results {
        if !result
            .get("success")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(RunError::Failed(format!(
                "SciPy did not converge: {result}"
            )));
        }
        let point = result
            .get("x")
            .and_then(Value::as_array)
            .ok_or_else(|| RunError::Failed("SciPy result has no x vector".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| RunError::Failed("SciPy x contains non-finite value".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if point.len() != TARGET.len() || point.iter().any(|value| !value.is_finite()) {
            return Err(RunError::Failed(
                "SciPy x has invalid dimension/value".into(),
            ));
        }
        let objective = result
            .get("fun")
            .and_then(Value::as_f64)
            .ok_or_else(|| RunError::Failed("SciPy result has no objective".into()))?;
        points.push(point);
        objectives.push(objective);
    }
    Ok((points, objectives))
}

fn run_sonium() -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let gradient = |x: &[f64]| {
        x.iter()
            .zip(TARGET)
            .zip(WEIGHTS)
            .map(|((&value, target), weight)| weight * (value - target))
            .collect::<Vec<_>>()
    };
    let bfgs = bfgs(
        &objective,
        &gradient,
        &START,
        &BfgsConfig {
            max_iterations: 200,
            grad_tolerance: 1.0e-10,
            fun_tolerance: 1.0e-12,
            step_tolerance: 1.0e-12,
            line_search: LineSearchMethod::StrongWolfe,
        },
    )
    .map_err(|error| format!("Sonium BFGS failed: {error:?}"))?;
    let lbfgs = lbfgs(
        &objective,
        &gradient,
        &START,
        &LbfgsConfig {
            max_iterations: 200,
            grad_tolerance: 1.0e-10,
            fun_tolerance: 1.0e-12,
            step_tolerance: 1.0e-12,
            memory: 8,
            line_search: LineSearchMethod::StrongWolfe,
        },
    )
    .map_err(|error| format!("Sonium L-BFGS failed: {error:?}"))?;
    if !bfgs.converged || !lbfgs.converged {
        return Err(format!(
            "Sonium did not converge: BFGS={}, L-BFGS={}",
            bfgs.converged, lbfgs.converged
        ));
    }
    Ok((
        vec![bfgs.x.clone(), lbfgs.x.clone()],
        vec![objective(&bfgs.x), objective(&lbfgs.x)],
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run_sonium() {
        Err(error) => QaResult::failed(CASE_ID, error),
        Ok((sonium_points, sonium_objectives)) => match run_scipy() {
            Err(RunError::Unavailable(reason)) => {
                if std::env::var("SONIUM_REQUIRE_SCIPY").as_deref() == Ok("1") {
                    QaResult::failed(CASE_ID, reason)
                } else {
                    QaResult::skipped(CASE_ID, reason)
                }
            }
            Err(RunError::Failed(reason)) => QaResult::failed(CASE_ID, reason),
            Ok((scipy_points, scipy_objectives)) => {
            let mut max_son_error: f64 = 0.0;
            let mut max_cross_error: f64 = 0.0;
            let mut max_objective_gap: f64 = 0.0;
            for index in 0..2 {
                max_son_error = max_son_error.max(max_coordinate_error(&sonium_points[index]));
                max_cross_error = max_cross_error.max(
                    sonium_points[index]
                        .iter()
                        .zip(&scipy_points[index])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max),
                );
                max_objective_gap =
                    max_objective_gap.max((sonium_objectives[index] - scipy_objectives[index]).abs());
            }
            QaResult::passed(CASE_ID)
                .requested_backend("cpu+scipy")
                .actual_backend("cpu+scipy")
                .reference("live SciPy 1.x BFGS and L-BFGS-B on the same translated diagonal quadratic")
                .reference_checksum("analytic/deterministic-fixture:optimization-scipy-v1")
                .input_checksum("analytic/deterministic-fixture:optimization-scipy-v1")
                .dimension("three-variable translated diagonal quadratic; BFGS and L-BFGS")
                .formulation("Sonium Strong-Wolfe BFGS/L-BFGS versus SciPy BFGS/L-BFGS-B")
                .tolerance("both external methods converge; Sonium coordinate error <= 1e-7; cross-method coordinate gap <= 1e-6")
                .metric("max_sonium_coordinate_error", max_son_error)
                .maximum("max_sonium_coordinate_error", 1.0e-7)
                .metric(
                    "max_scipy_coordinate_error",
                    scipy_points
                        .iter()
                        .map(|x| max_coordinate_error(x))
                        .fold(0.0, f64::max),
                )
                .maximum("max_scipy_coordinate_error", 1.0e-7)
                .metric("max_cross_coordinate_error", max_cross_error)
                .maximum("max_cross_coordinate_error", 1.0e-6)
                .metric("max_objective_gap", max_objective_gap)
                .maximum("max_objective_gap", 1.0e-10)
            }
        },
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    match result.status {
        QaStatus::Passed => println!("SUCCESS"),
        QaStatus::Skipped => println!("SKIPPED"),
        QaStatus::Failed => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
