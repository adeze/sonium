//! Independent BOBYQA bounded/unbounded and callback-contract validation.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{BobyqaConfig, bobyqa};
use sonium_solvers::{Bound, OptimError, OptimStatus};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-bobyqa.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:solvers-optimization-bobyqa-v1";

fn coordinate_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();

    let interior = |x: &[f64]| 2.0 * (x[0] - 0.75).powi(2) + 5.0 * (x[1] + 1.25).powi(2);
    let interior_config = BobyqaConfig {
        bounds: vec![Bound::both(-1.0, 1.0), Bound::both(-2.0, 2.0)],
        ..BobyqaConfig::default()
    };
    let interior_result = bobyqa(&interior, &[0.0, 0.0], &interior_config)
        .map_err(|error| format!("bounded interior BOBYQA failed: {error}"))?;
    if !interior_result.converged || interior_result.status != OptimStatus::Converged {
        return Err(format!(
            "bounded interior BOBYQA did not converge: {:?}",
            interior_result.status
        ));
    }
    let interior_error = coordinate_error(&interior_result.x, &[0.75, -1.25]);
    if interior_error > 2.0e-4 || interior_result.fun > 1.0e-7 {
        return Err(format!(
            "bounded interior solution mismatch: x={:?}, fun={}",
            interior_result.x, interior_result.fun
        ));
    }

    let boundary = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2);
    let boundary_result = bobyqa(
        &boundary,
        &[0.0, 0.0],
        &BobyqaConfig {
            bounds: vec![Bound::both(-1.0, 1.0), Bound::both(-2.0, 2.0)],
            ..BobyqaConfig::default()
        },
    )
    .map_err(|error| format!("active-bound BOBYQA failed: {error}"))?;
    let boundary_error = coordinate_error(&boundary_result.x, &[1.0, -2.0]);
    if !boundary_result.converged
        || boundary_result.status != OptimStatus::Converged
        || boundary_error > 2.0e-4
        || (boundary_result.fun - 2.0).abs() > 1.0e-7
    {
        return Err(format!(
            "active-bound solution mismatch: x={:?}, fun={}, status={:?}",
            boundary_result.x, boundary_result.fun, boundary_result.status
        ));
    }

    let unbounded = |x: &[f64]| (x[0] - 1.25).powi(2) + 3.0 * (x[1] + 0.5).powi(2);
    let unbounded_result = bobyqa(&unbounded, &[0.0, 0.0], &BobyqaConfig::default())
        .map_err(|error| format!("unbounded BOBYQA failed: {error}"))?;
    let unbounded_error = coordinate_error(&unbounded_result.x, &[1.25, -0.5]);
    if !unbounded_result.converged
        || unbounded_result.status != OptimStatus::Converged
        || unbounded_error > 2.0e-4
        || unbounded_result.fun > 1.0e-7
    {
        return Err(format!(
            "unbounded solution mismatch: x={:?}, fun={}, status={:?}",
            unbounded_result.x, unbounded_result.fun, unbounded_result.status
        ));
    }

    let repeat = bobyqa(&interior, &[0.0, 0.0], &interior_config)
        .map_err(|error| format!("repeat bounded BOBYQA failed: {error}"))?;
    let deterministic_eval_gap = interior_result.fun_evals.abs_diff(repeat.fun_evals) as f64;
    let deterministic_objective_gap = (interior_result.fun - repeat.fun).abs();
    if deterministic_eval_gap != 0.0 || deterministic_objective_gap != 0.0 {
        return Err(format!(
            "BOBYQA is not deterministic: eval gap={deterministic_eval_gap}, fun gap={deterministic_objective_gap}"
        ));
    }

    let generated_nonfinite = |x: &[f64]| {
        if x.len() == 1 && x[0] == 0.0 {
            0.0
        } else {
            f64::NAN
        }
    };
    let generated_error = bobyqa(&generated_nonfinite, &[0.0], &BobyqaConfig::default())
        .expect_err("generated non-finite objective must fail closed");
    if !matches!(generated_error, OptimError::NumericalError { .. }) {
        return Err(format!(
            "generated non-finite objective returned wrong error: {generated_error:?}"
        ));
    }

    let result = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("fixture_count", 4.0)
        .minimum("fixture_count", 4.0)
        .metric("converged_fixtures", 3.0)
        .minimum("converged_fixtures", 3.0)
        .metric("converged_status_contract", 3.0)
        .minimum("converged_status_contract", 3.0)
        .metric("interior_coordinate_error", interior_error)
        .maximum("interior_coordinate_error", 2.0e-4)
        .metric("boundary_coordinate_error", boundary_error)
        .maximum("boundary_coordinate_error", 2.0e-4)
        .metric("unbounded_coordinate_error", unbounded_error)
        .maximum("unbounded_coordinate_error", 2.0e-4)
        .metric("deterministic_eval_gap", deterministic_eval_gap)
        .maximum("deterministic_eval_gap", 0.0)
        .metric("deterministic_objective_gap", deterministic_objective_gap)
        .maximum("deterministic_objective_gap", 0.0)
        .metric("generated_nonfinite_rejections", 1.0)
        .minimum("generated_nonfinite_rejections", 1.0)
        .input_checksum(CHECKSUM)
        .reference("analytic bounded and unbounded quadratic minimizers")
        .reference_checksum(CHECKSUM)
        .dimension("f64; two-variable bound-constrained and unbounded BOBYQA")
        .formulation("BOBYQA objective accuracy, active bounds, deterministic evaluation accounting, and generated callback failure contract")
        .tolerance("coordinate error <= 2e-4; objective <= 1e-7; generated non-finite callback rejected")
        .seed("solvers-optimization-bobyqa:deterministic-fixture:v1")
        .timing_ms("elapsed", started.elapsed().as_millis() as u64);
    Ok(result)
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error).timing_ms("elapsed", 0);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
