//! Independent CPU evidence for native-multiplier KKT coverage.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    AdvancedSqpConfig, Bound, Constraint, ConstraintKind, LbfgsbConfig, OptimResult, SlsqpConfig,
    SqpConfig, lbfgsb, slsqp, sqp, sqp_advanced,
};
use sonium_solvers_core::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-kkt-matrix.v1";

#[derive(Clone, Copy)]
enum ConstraintCase {
    Equality,
    ActiveInequality,
    InactiveInequality,
}

#[derive(Clone, Copy, Debug)]
enum MultidimensionalConstraintCase {
    Equality,
    ActiveInequality,
    InactiveInequality,
}

fn constraint(case: ConstraintCase) -> Constraint<f64> {
    let (kind, slope, offset) = match case {
        ConstraintCase::Equality => (ConstraintKind::Equality, 1.0, -1.0),
        ConstraintCase::ActiveInequality | ConstraintCase::InactiveInequality => {
            (ConstraintKind::Inequality, -1.0, 1.0)
        }
    };
    Constraint {
        kind,
        dim: 1,
        fun: Box::new(move |x: &[f64]| vec![slope * x[0] + offset]),
        jac: Some(Box::new(move |_| {
            DenseMatrix::from_row_major_vec(1, 1, vec![slope])
        })),
    }
}

fn multidimensional_constraint(case: MultidimensionalConstraintCase) -> Constraint<f64> {
    let (kind, coefficients, offset) = match case {
        MultidimensionalConstraintCase::Equality => (ConstraintKind::Equality, [1.0, 2.0], -1.0),
        MultidimensionalConstraintCase::ActiveInequality => {
            (ConstraintKind::Inequality, [1.0, 0.0], -3.0)
        }
        MultidimensionalConstraintCase::InactiveInequality => {
            (ConstraintKind::Inequality, [1.0, 0.0], 1.0)
        }
    };
    Constraint {
        kind,
        dim: 1,
        fun: Box::new(move |x: &[f64]| {
            vec![coefficients[0] * x[0] + coefficients[1] * x[1] + offset]
        }),
        jac: Some(Box::new(move |_| {
            DenseMatrix::from_row_major_vec(1, 2, coefficients.to_vec())
        })),
    }
}

fn multidimensional_objective(x: &[f64]) -> f64 {
    (x[0] - 2.0).powi(2) + 4.0 * (x[1] + 1.0).powi(2)
}

fn multidimensional_gradient(x: &[f64]) -> Vec<f64> {
    vec![2.0 * (x[0] - 2.0), 8.0 * (x[1] + 1.0)]
}

fn objective(target: f64, x: &[f64]) -> f64 {
    (x[0] - target).powi(2)
}

fn gradient(target: f64, x: &[f64]) -> Vec<f64> {
    vec![2.0 * (x[0] - target)]
}

fn check_kkt(result: &OptimResult<f64>, case: ConstraintCase, target: f64) -> Result<f64, String> {
    if !result.converged {
        return Err(format!("optimizer did not converge: {result:?}"));
    }
    let report = result
        .kkt
        .as_ref()
        .ok_or_else(|| "successful result has no KKT report".to_string())?;
    let x = result.x[0];
    let gradient = 2.0 * (x - target);
    let (kind, value, jacobian) = match case {
        ConstraintCase::Equality => (ConstraintKind::Equality, x - 1.0, 1.0),
        ConstraintCase::ActiveInequality | ConstraintCase::InactiveInequality => {
            (ConstraintKind::Inequality, 1.0 - x, -1.0)
        }
    };
    let multiplier = report.multipliers.constraints[0];
    let primal = match kind {
        ConstraintKind::Equality => value.abs(),
        ConstraintKind::Inequality => (-value).max(0.0),
    };
    let dual = match kind {
        ConstraintKind::Equality => 0.0,
        ConstraintKind::Inequality => (-multiplier).max(0.0),
    };
    let stationarity = (gradient - multiplier * jacobian).abs();
    let complementarity = (multiplier * value).abs();
    let independent = primal.max(dual).max(stationarity).max(complementarity);
    if independent > 2.0e-5 || !report.norm.is_finite() || report.norm > 2.0e-5 {
        return Err(format!(
            "KKT mismatch independent={independent:e} report={:e}",
            report.norm
        ));
    }
    match case {
        ConstraintCase::Equality if multiplier.abs() <= 1.0e-3 => {
            return Err(format!(
                "equality multiplier unexpectedly inactive: {multiplier:e}"
            ));
        }
        ConstraintCase::ActiveInequality if multiplier <= 1.0e-3 => {
            return Err(format!(
                "active inequality multiplier too small: {multiplier:e}"
            ));
        }
        ConstraintCase::InactiveInequality if multiplier.abs() > 2.0e-5 => {
            return Err(format!(
                "inactive inequality multiplier nonzero: {multiplier:e}"
            ));
        }
        _ => {}
    }
    Ok(independent.max(report.norm))
}

fn check_multidimensional_kkt(
    result: &OptimResult<f64>,
    case: MultidimensionalConstraintCase,
) -> Result<f64, String> {
    if !result.converged {
        return Err(format!(
            "multidimensional optimizer did not converge: {result:?}"
        ));
    }
    let report = result
        .kkt
        .as_ref()
        .ok_or_else(|| "multidimensional result has no KKT report".to_string())?;
    let x = &result.x;
    let gradient = multidimensional_gradient(x);
    let (value, jacobian) = match case {
        MultidimensionalConstraintCase::Equality => (x[0] + 2.0 * x[1] - 1.0, [1.0, 2.0]),
        MultidimensionalConstraintCase::ActiveInequality => (x[0] - 3.0, [1.0, 0.0]),
        MultidimensionalConstraintCase::InactiveInequality => (x[0] + 1.0, [1.0, 0.0]),
    };
    let multiplier = report.multipliers.constraints[0];
    let primal = match case {
        MultidimensionalConstraintCase::Equality => value.abs(),
        MultidimensionalConstraintCase::ActiveInequality
        | MultidimensionalConstraintCase::InactiveInequality => (-value).max(0.0),
    };
    let dual = match case {
        MultidimensionalConstraintCase::Equality => 0.0,
        MultidimensionalConstraintCase::ActiveInequality
        | MultidimensionalConstraintCase::InactiveInequality => (-multiplier).max(0.0),
    };
    let stationarity = gradient
        .iter()
        .zip(jacobian)
        .map(|(&g, j)| (g - multiplier * j).abs())
        .fold(0.0, f64::max);
    let complementarity = (multiplier * value).abs();
    let independent = primal.max(dual).max(stationarity).max(complementarity);
    if !independent.is_finite()
        || independent > 2.0e-5
        || !report.norm.is_finite()
        || report.norm > 2.0e-5
    {
        return Err(format!(
            "multidimensional KKT failed for {case:?}: independent={independent:e}, report={report:?}"
        ));
    }
    match case {
        MultidimensionalConstraintCase::Equality if multiplier.abs() <= 1.0e-3 => {
            return Err(format!("equality multiplier too small: {multiplier:e}"));
        }
        MultidimensionalConstraintCase::ActiveInequality if multiplier <= 1.0e-3 => {
            return Err(format!("active multiplier too small: {multiplier:e}"));
        }
        MultidimensionalConstraintCase::InactiveInequality if multiplier.abs() > 2.0e-5 => {
            return Err(format!("inactive multiplier nonzero: {multiplier:e}"));
        }
        _ => {}
    }
    Ok(independent.max(report.norm))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let cases = [
        (ConstraintCase::Equality, 2.0),
        (ConstraintCase::ActiveInequality, 2.0),
        (ConstraintCase::InactiveInequality, 0.25),
    ];
    let mut max_kkt = 0.0_f64;
    let mut runs = 0.0;
    let mut multidimensional_runs = 0.0;
    for (case, target) in cases {
        let objective_fn = |x: &[f64]| objective(target, x);
        let gradient_fn = |x: &[f64]| gradient(target, x);

        let mut sqp_config = SqpConfig::default();
        sqp_config.max_iterations = 100;
        sqp_config.grad_tolerance = 1.0e-9;
        sqp_config.constraint_tolerance = 1.0e-9;
        let sqp_result = sqp(
            &objective_fn,
            &gradient_fn,
            &[0.0],
            &[constraint(case)],
            &sqp_config,
        )
        .map_err(|error| format!("SQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_kkt(&sqp_result, case, target)?);
        runs += 1.0;

        let hessian_fn = |_: &[f64], _: &[f64]| DenseMatrix::from_row_major_vec(1, 1, vec![2.0]);
        let mut advanced_config = AdvancedSqpConfig::default();
        advanced_config.max_iterations = 100;
        advanced_config.grad_tolerance = 1.0e-9;
        advanced_config.constraint_tolerance = 1.0e-9;
        let advanced_result = sqp_advanced(
            &objective_fn,
            &gradient_fn,
            &hessian_fn,
            &[0.0],
            &[constraint(case)],
            &advanced_config,
        )
        .map_err(|error| format!("advanced SQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_kkt(&advanced_result, case, target)?);
        runs += 1.0;

        let mut slsqp_config = SlsqpConfig::default();
        slsqp_config.max_iterations = 100;
        slsqp_config.grad_tolerance = 1.0e-9;
        slsqp_config.constraint_tolerance = 1.0e-9;
        let slsqp_result = slsqp(
            &objective_fn,
            &gradient_fn,
            &[0.0],
            &[constraint(case)],
            &slsqp_config,
        )
        .map_err(|error| format!("SLSQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_kkt(&slsqp_result, case, target)?);
        runs += 1.0;
    }

    for case in [
        MultidimensionalConstraintCase::Equality,
        MultidimensionalConstraintCase::ActiveInequality,
        MultidimensionalConstraintCase::InactiveInequality,
    ] {
        let sqp_config = SqpConfig {
            max_iterations: 100,
            grad_tolerance: 1.0e-9,
            constraint_tolerance: 1.0e-9,
            ..SqpConfig::default()
        };
        let sqp_result = sqp(
            &multidimensional_objective,
            &multidimensional_gradient,
            &[0.0, 0.0],
            &[multidimensional_constraint(case)],
            &sqp_config,
        )
        .map_err(|error| format!("multidimensional SQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_multidimensional_kkt(&sqp_result, case)?);
        multidimensional_runs += 1.0;

        let hessian_fn =
            |_: &[f64], _: &[f64]| DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 0.0, 0.0, 8.0]);
        let advanced_config = AdvancedSqpConfig {
            max_iterations: 100,
            grad_tolerance: 1.0e-9,
            constraint_tolerance: 1.0e-9,
            ..AdvancedSqpConfig::default()
        };
        let advanced_result = sqp_advanced(
            &multidimensional_objective,
            &multidimensional_gradient,
            &hessian_fn,
            &[0.0, 0.0],
            &[multidimensional_constraint(case)],
            &advanced_config,
        )
        .map_err(|error| format!("multidimensional advanced SQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_multidimensional_kkt(&advanced_result, case)?);
        multidimensional_runs += 1.0;

        let slsqp_config = SlsqpConfig {
            max_iterations: 100,
            grad_tolerance: 1.0e-9,
            constraint_tolerance: 1.0e-9,
            ..SlsqpConfig::default()
        };
        let slsqp_result = slsqp(
            &multidimensional_objective,
            &multidimensional_gradient,
            &[0.0, 0.0],
            &[multidimensional_constraint(case)],
            &slsqp_config,
        )
        .map_err(|error| format!("multidimensional SLSQP failed: {error}"))?;
        max_kkt = max_kkt.max(check_multidimensional_kkt(&slsqp_result, case)?);
        multidimensional_runs += 1.0;
    }

    let multidimensional_bound_config = LbfgsbConfig {
        bounds: vec![Bound::upper(1.0), Bound::unbounded()],
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        ..LbfgsbConfig::default()
    };
    let multidimensional_bound_result = lbfgsb(
        &multidimensional_objective,
        &multidimensional_gradient,
        &[0.0, 0.0],
        &multidimensional_bound_config,
    )
    .map_err(|error| format!("multidimensional L-BFGS-B failed: {error}"))?;
    if !multidimensional_bound_result.converged {
        return Err(format!(
            "multidimensional L-BFGS-B did not converge: {multidimensional_bound_result:?}"
        ));
    }
    let multidimensional_bound_report = multidimensional_bound_result
        .kkt
        .as_ref()
        .ok_or_else(|| "multidimensional bound result has no KKT report".to_string())?;
    if multidimensional_bound_report
        .diagnostics
        .active_upper_bounds
        != [0]
        || multidimensional_bound_report.multipliers.upper_bounds[0] <= 1.0e-3
        || !multidimensional_bound_report.norm.is_finite()
        || multidimensional_bound_report.norm > 2.0e-5
    {
        return Err(format!(
            "multidimensional bound KKT failed: {multidimensional_bound_report:?}"
        ));
    }
    max_kkt = max_kkt.max(multidimensional_bound_report.norm);
    multidimensional_runs += 1.0;

    let objective_fn = |x: &[f64]| objective(2.0, x);
    let gradient_fn = |x: &[f64]| gradient(2.0, x);
    let config = LbfgsbConfig {
        bounds: vec![Bound::upper(1.0)],
        max_iterations: 100,
        grad_tolerance: 1.0e-9,
        ..LbfgsbConfig::default()
    };
    let result = lbfgsb(&objective_fn, &gradient_fn, &[0.0], &config)
        .map_err(|error| format!("L-BFGS-B failed: {error}"))?;
    if !result.converged {
        return Err(format!("L-BFGS-B did not converge: {result:?}"));
    }
    let report = result
        .kkt
        .as_ref()
        .ok_or_else(|| "L-BFGS-B result has no KKT report".to_string())?;
    if report.diagnostics.active_upper_bounds != [0]
        || report.multipliers.upper_bounds[0] <= 1.0e-3
        || !report.norm.is_finite()
        || report.norm > 2.0e-5
    {
        return Err(format!("bound KKT report failed: {report:?}"));
    }
    runs += 1.0;

    Ok(vec![
        ("optimizer_runs", runs),
        ("active_equality_cases", 3.0),
        ("active_inequality_cases", 3.0),
        ("inactive_inequality_cases", 3.0),
        ("active_bound_cases", 1.0),
        ("multidimensional_runs", multidimensional_runs),
        ("multidimensional_constraint_cases", 9.0),
        ("multidimensional_bound_cases", 1.0),
        ("max_independent_kkt_norm", max_kkt),
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
            result = result.elapsed_ms(started.elapsed().as_millis() as u64);
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
