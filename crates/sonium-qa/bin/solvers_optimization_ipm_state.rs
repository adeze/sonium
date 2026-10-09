//! Independent CPU evidence for interior-point state diagnostics.

use sonium_qa::QaResult;
use sonium_solvers::{Bound, Constraint, ConstraintKind, IpmConfig, OptimStatus, ipm};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::SolverResultContract;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-ipm-state.v1";

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let objective = |x: &[f64]| x[0] + x[1];
    let gradient = |_x: &[f64]| vec![1.0, 1.0];
    let hessian = |_: &[f64], multipliers: &[f64]| {
        DenseMatrix::from_diag(&[2.0 * multipliers[0], 2.0 * multipliers[0]])
    };
    let constraints = vec![Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![1.0 - x[0] * x[0] - x[1] * x[1]]),
        jac: Some(Box::new(|x: &[f64]| {
            DenseMatrix::from_row_major_vec(1, 2, vec![-2.0 * x[0], -2.0 * x[1]])
        })),
    }];
    let result = ipm(
        &objective,
        &gradient,
        Some(&hessian),
        &[1.0, 0.0],
        &constraints,
        &IpmConfig::default(),
    )
    .map_err(|error| format!("circle IPM failed: {error}"))?;
    if !result.converged {
        return Err(format!("circle IPM did not converge: {result:?}"));
    }
    result
        .validate_contract()
        .map_err(|error| format!("circle IPM result contract failed: {error}"))?;
    let report = result
        .kkt
        .as_ref()
        .ok_or_else(|| "IPM result has no KKT report".to_string())?;
    let state = report
        .ipm
        .as_ref()
        .ok_or_else(|| "IPM result has no state diagnostics".to_string())?;
    if state.iterations != result.iterations
        || state.active_inequalities != state.active_inequality_rows.len()
        || state.active_bounds != state.active_bound_rows.len()
        || state.active_inequality_rows != [0]
        || !state.barrier_parameter.is_finite()
        || !state.dual_residual.is_finite()
        || !state.primal_residual.is_finite()
        || !state.complementarity.is_finite()
        || !report.norm.is_finite()
    {
        return Err(format!(
            "invalid circle IPM diagnostics: {state:?}, {report:?}"
        ));
    }
    let circle_native_multiplier_error =
        (report.multipliers.constraints[0] - std::f64::consts::FRAC_1_SQRT_2).abs();
    if circle_native_multiplier_error > 1e-6 {
        return Err(format!(
            "circle native multiplier mismatch: got {}, error {}",
            report.multipliers.constraints[0], circle_native_multiplier_error
        ));
    }

    let bound_objective = |x: &[f64]| (x[0] - 2.0).powi(2);
    let bound_gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let bound_result = ipm(
        &bound_objective,
        &bound_gradient,
        None::<&fn(&[f64], &[f64]) -> DenseMatrix<f64>>,
        &[0.0],
        &[],
        &IpmConfig {
            bounds: vec![Bound::upper(1.0)],
            ..IpmConfig::default()
        },
    )
    .map_err(|error| format!("bound IPM failed: {error}"))?;
    if !bound_result.converged {
        return Err(format!("bound IPM did not converge: {bound_result:?}"));
    }
    bound_result
        .validate_contract()
        .map_err(|error| format!("bound IPM result contract failed: {error}"))?;
    let bound_report = bound_result
        .kkt
        .as_ref()
        .ok_or_else(|| "bound IPM result has no KKT report".to_string())?;
    let bound_state = bound_report
        .ipm
        .as_ref()
        .ok_or_else(|| "bound IPM result has no state diagnostics".to_string())?;
    if bound_state.active_bounds != bound_state.active_bound_rows.len()
        || bound_state.active_bound_rows != [0]
    {
        return Err(format!(
            "bound active-set diagnostics failed: {bound_state:?}"
        ));
    }
    let bound_native_multiplier_error = (bound_report.multipliers.upper_bounds[0] - 2.0).abs();
    if bound_native_multiplier_error > 1e-6 {
        return Err(format!(
            "bound native multiplier mismatch: got {}, error {}",
            bound_report.multipliers.upper_bounds[0], bound_native_multiplier_error
        ));
    }

    let infeasible_constraints = vec![
        Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_: &[f64]| {
                DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        },
        Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 2.0]),
            jac: Some(Box::new(|_: &[f64]| {
                DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        },
    ];
    let infeasible_hessian = |_: &[f64], _: &[f64]| DenseMatrix::from_diag(&[2.0, 2.0]);
    let infeasible = ipm(
        &|x: &[f64]| x[0].powi(2) + x[1].powi(2),
        &|x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]],
        Some(&infeasible_hessian),
        &[0.5, 0.5],
        &infeasible_constraints,
        &IpmConfig::default(),
    )
    .map_err(|error| format!("infeasible IPM unexpectedly returned an error: {error}"))?;
    if infeasible.converged
        || infeasible.status != OptimStatus::RestorationFailed
        || infeasible.x.iter().any(|value| !value.is_finite())
        || !infeasible.fun.is_finite()
    {
        return Err(format!(
            "infeasible IPM was not a finite typed restoration failure: {:?}",
            infeasible.status
        ));
    }
    let infeasible_state = infeasible
        .kkt
        .as_ref()
        .and_then(|report| report.ipm.as_ref())
        .ok_or_else(|| "infeasible IPM result has no diagnostics".to_string())?;
    if infeasible_state.restoration_attempts == 0
        || infeasible_state.restoration_successes != 0
        || !infeasible_state.primal_residual.is_finite()
    {
        return Err(format!(
            "infeasible IPM restoration diagnostics are invalid: {:?}",
            infeasible_state
        ));
    }

    Ok(vec![
        ("ipm_cases", 3.0),
        ("circle_iterations", state.iterations as f64),
        (
            "circle_native_multiplier_error",
            circle_native_multiplier_error,
        ),
        (
            "circle_active_inequalities",
            state.active_inequalities as f64,
        ),
        ("bound_active_bounds", bound_state.active_bounds as f64),
        (
            "bound_native_multiplier_error",
            bound_native_multiplier_error,
        ),
        (
            "max_restoration_failures",
            state
                .restoration_failures
                .max(bound_state.restoration_failures) as f64,
        ),
        (
            "max_line_search_backtracks",
            state
                .line_search_backtracks
                .max(bound_state.line_search_backtracks) as f64,
        ),
        (
            "max_kkt_norm",
            report.norm.max(bound_result.kkt.as_ref().unwrap().norm),
        ),
        ("infeasible_restoration_failure", 1.0),
        (
            "infeasible_restoration_attempts",
            infeasible_state.restoration_attempts as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            result = result
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent IPM state/KKT diagnostics")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-optimization-ipm-state-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let mut result = QaResult::failed(CASE_ID, error);
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
