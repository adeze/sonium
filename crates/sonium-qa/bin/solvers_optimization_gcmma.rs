//! Independent CPU QA for the production GCMMA constrained optimizer.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{Bound, Constraint, ConstraintKind, MmaConfig, MmaVariant, mma};
use sonium_solvers_core::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-gcmma.v1";
const TOLERANCE: f64 = 2.0e-3;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let objective = |x: &[f64]| 0.5 * (x[0] * x[0] + x[1] * x[1]);
    let gradient = |x: &[f64]| vec![x[0], x[1]];
    let constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
        jac: Some(Box::new(|_: &[f64]| {
            DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
        })),
    };
    let config = MmaConfig {
        variant: MmaVariant::Gcmma,
        max_iterations: 300,
        max_conservative_iterations: 20,
        kkt_tolerance: 1.0e-7,
        bounds: vec![Bound::both(0.01, 2.0), Bound::both(0.01, 2.0)],
        ..MmaConfig::default()
    };
    let result = mma(&objective, &gradient, &[0.1, 0.1], &[constraint], &config)
        .map_err(|error| format!("GCMMA failed: {error:?}"))?;
    let solution_error = (result.x[0] - 0.5).abs().max((result.x[1] - 0.5).abs());
    let feasibility_error = (1.0 - result.x[0] - result.x[1]).max(0.0);
    let objective_error = (result.fun - 0.25).abs();
    let convergence_error = if result.converged { 0.0 } else { 1.0 };
    let finite_error = if result.x.iter().all(|value| value.is_finite())
        && result.fun.is_finite()
        && result.grad_norm.is_finite()
    {
        0.0
    } else {
        1.0
    };
    let metrics = vec![
        ("solution_error", solution_error),
        ("feasibility_error", feasibility_error),
        ("objective_error", objective_error),
        ("convergence_error", convergence_error),
        ("finite_result_error", finite_error),
    ];
    if metrics.iter().any(|(_, value)| *value > TOLERANCE) {
        return Err(format!(
            "GCMMA analytic mismatch: {metrics:?}, result={result:?}"
        ));
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
                .reference("analytic minimum of 0.5*||x||^2 subject to x+y>=1")
                .input_checksum("analytic/deterministic-fixture:gcmma-linear-constraint-v1")
                .reference_checksum("analytic/deterministic-fixture:gcmma-linear-constraint-v1")
                .formulation("GCMMA conservative approximation with box bounds")
                .dimension("two design variables; one analytic inequality")
                .tolerance("solution/objective/feasibility errors <= 2e-3 and finite convergence");
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            result = result
                .seed("solvers-optimization-gcmma:cpu:v1")
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
