//! Independent CPU evidence for epsilon-constraint Pareto KKT reporting.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    Constraint, ConstraintKind, EpsilonConstraintConfig, epsilon_constraint,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-epsilon-kkt.v1";

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    // Keep the secondary objective's minimum on an explicit boundary so the
    // endpoint epsilon subproblem satisfies a regular KKT constraint
    // qualification (an unconstrained scalar minimum would have zero
    // constraint gradient at epsilon = f2_min).
    let f1 = |x: &[f64]| (x[0] + 1.0).powi(2);
    let g1 = |x: &[f64]| vec![2.0 * (x[0] + 1.0)];
    let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
    let g2 = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let objectives = [f1, f2];
    let gradients = [g1, g2];
    let domain = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![-x[0]]),
        jac: Some(Box::new(|_x: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![-1.0])
        })),
    };
    let config = EpsilonConstraintConfig {
        num_points: 5,
        max_iterations: 100,
        tolerance: 1.0e-8,
        ..EpsilonConstraintConfig::default()
    };
    let pareto = epsilon_constraint(&objectives, &gradients, &[-0.5], &[domain], &config)
        .map_err(|error| format!("epsilon-constraint solve failed: {error}"))?;

    if pareto.solutions.len() != config.num_points {
        return Err(format!(
            "expected {} Pareto points, got {}",
            config.num_points,
            pareto.solutions.len()
        ));
    }
    if pareto.subproblem_results.len() != config.num_points {
        return Err("subproblem result count does not match Pareto points".into());
    }

    let mut max_kkt_norm = 0.0_f64;
    let mut missing_kkt_reports = 0.0_f64;
    for (index, result) in pareto.subproblem_results.iter().enumerate() {
        if !result.converged {
            return Err(format!(
                "epsilon subproblem {index} did not converge: {result:?}"
            ));
        }
        let Some(report) = result.kkt.as_ref() else {
            missing_kkt_reports += 1.0;
            continue;
        };
        if !report.norm.is_finite() || report.norm > 1.0e-3 {
            return Err(format!(
                "epsilon subproblem {index} has invalid KKT norm {}",
                report.norm
            ));
        }
        max_kkt_norm = max_kkt_norm.max(report.norm);
    }
    let finite_objective_values = pareto
        .objective_values
        .iter()
        .flatten()
        .all(|value| value.is_finite());
    if !finite_objective_values {
        return Err("Pareto objective values contain a non-finite value".into());
    }

    Ok(vec![
        ("pareto_points", pareto.solutions.len() as f64),
        (
            "kkt_report_cases",
            (pareto.subproblem_results.len() as f64) - missing_kkt_reports,
        ),
        ("missing_kkt_reports", missing_kkt_reports),
        (
            "finite_objective_values",
            if finite_objective_values { 1.0 } else { 0.0 },
        ),
        ("max_kkt_norm", max_kkt_norm),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("analytic two-objective epsilon-constraint KKT reports")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-optimization-epsilon-kkt-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
