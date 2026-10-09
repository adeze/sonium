//! Release-facing KKT gates for constrained optimization results.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    Bound, Constraint, ConstraintKind, LbfgsbConfig, SqpConfig, lbfgsb, sqp,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-kkt-reference.v1";

fn run() -> Result<(f64, f64, f64, f64), String> {
    let objective = |x: &[f64]| (x[0] - 2.0).powi(2);
    let gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
        jac: Some(Box::new(|_| {
            sonium_solvers::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    let constrained = sqp(
        &objective,
        &gradient,
        &[0.0],
        &constraints,
        &SqpConfig::default(),
    )
    .map_err(|error| format!("SQP: {error}"))?;
    let constrained_kkt = constrained
        .kkt
        .as_ref()
        .ok_or_else(|| "SQP returned no KKT report".to_string())?;

    let bounded = lbfgsb(
        &objective,
        &gradient,
        &[0.0],
        &LbfgsbConfig {
            bounds: vec![Bound::upper(1.0)],
            ..LbfgsbConfig::default()
        },
    )
    .map_err(|error| format!("L-BFGS-B: {error}"))?;
    let bounded_kkt = bounded
        .kkt
        .as_ref()
        .ok_or_else(|| "L-BFGS-B returned no KKT report".to_string())?;
    if !constrained.converged || !bounded.converged {
        return Err(format!(
            "optimizer did not converge: SQP={}, L-BFGS-B={}",
            constrained.converged, bounded.converged
        ));
    }
    let constrained_feasibility = constrained_kkt.diagnostics.primal_feasibility;
    let constrained_stationarity = constrained_kkt.diagnostics.stationarity;
    let bounded_norm = bounded_kkt.norm;
    let active_upper = bounded_kkt.diagnostics.active_upper_bounds.len() as f64;
    if constrained_feasibility > 1.0e-4
        || constrained_stationarity > 1.0e-3
        || bounded_norm > 1.0e-3
        || active_upper != 1.0
    {
        return Err(format!(
            "KKT gate failed: feasibility={constrained_feasibility}, stationarity={constrained_stationarity}, bounded_norm={bounded_norm}, active_upper={active_upper}"
        ));
    }
    Ok((
        constrained_feasibility,
        constrained_stationarity,
        bounded_norm,
        active_upper,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((feasibility, stationarity, bounded_norm, active_upper)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("equality_primal_feasibility", feasibility)
            .metric("equality_stationarity", stationarity)
            .metric("bound_kkt_norm", bounded_norm)
            .metric("active_upper_bound_count", active_upper)
            .maximum("equality_primal_feasibility", 1.0e-4)
            .maximum("equality_stationarity", 1.0e-3)
            .maximum("bound_kkt_norm", 1.0e-3)
            .minimum("active_upper_bound_count", 1.0)
            .reference_checksum("analytic/deterministic-fixture:solvers-optimization-kkt-v1")
            .dimension("one-variable equality-constrained SQP and upper-bound L-BFGS-B")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    if let Ok(json) = result.to_json_line() {
        println!("QA_RESULT: {json}");
    }
    if matches!(&result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
