//! Independent CPU evidence for constrained-optimizer KKT success gating.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    AdvancedSqpConfig, AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind,
    HessianRegularisation, MmaConfig, OptimResult, PenaltyConfig, SlsqpConfig, SqpConfig,
    TrustRegionConstrainedConfig, augmented_lagrangian, mma, penalty_method, slsqp, sqp,
    sqp_advanced, trust_region_constrained,
};
use sonium_solvers_core::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-kkt-gate-family.v1";

fn equality_constraint() -> Constraint<f64> {
    Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
        jac: Some(Box::new(|_| {
            DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
        })),
    }
}

fn assert_gated(result: &OptimResult<f64>, method: &str) -> Result<f64, String> {
    if !result.converged {
        return Err(format!("{method} did not converge: {result:?}"));
    }
    let report = result
        .kkt
        .as_ref()
        .ok_or_else(|| format!("{method} returned converged without a KKT report"))?;
    if !report.norm.is_finite() || report.norm > 1.0e-2 {
        return Err(format!(
            "{method} returned an invalid KKT norm: {}",
            report.norm
        ));
    }
    Ok(report.norm)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let objective = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
    let gradient = |x: &[f64]| x.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
    let x0 = [0.0, 0.0];
    let mut max_kkt_norm = 0.0_f64;

    let penalty = penalty_method(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &PenaltyConfig::default(),
    )
    .map_err(|error| format!("penalty method failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&penalty, "penalty")?);

    let augmented = augmented_lagrangian(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &AugmentedLagrangianConfig::default(),
    )
    .map_err(|error| format!("augmented Lagrangian failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&augmented, "augmented Lagrangian")?);

    let trust = trust_region_constrained(
        &objective,
        &gradient,
        &|_x: &[f64], v: &[f64]| v.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
        &x0,
        &[equality_constraint()],
        &TrustRegionConstrainedConfig::default(),
    )
    .map_err(|error| format!("trust-region method failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&trust, "trust-region")?);

    let sqp_result = sqp(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &SqpConfig::default(),
    )
    .map_err(|error| format!("SQP failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&sqp_result, "SQP")?);

    let slsqp_result = slsqp(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &SlsqpConfig::default(),
    )
    .map_err(|error| format!("SLSQP failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&slsqp_result, "SLSQP")?);

    let hessian = |_x: &[f64], _lambda: &[f64]| DenseMatrix::from_diag(&[2.0, 2.0]);
    let advanced_sqp_result = sqp_advanced(
        &objective,
        &gradient,
        &hessian,
        &x0,
        &[equality_constraint()],
        &AdvancedSqpConfig {
            hessian_regularisation: HessianRegularisation::ModifiedCholesky,
            ..AdvancedSqpConfig::default()
        },
    )
    .map_err(|error| format!("advanced SQP failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&advanced_sqp_result, "advanced SQP")?);

    let mma_objective = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
    let mma_gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)];
    let mma_constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![2.0 - x[0] - x[1]]),
        jac: Some(Box::new(|_| {
            DenseMatrix::from_row_major_vec(1, 2, vec![-1.0, -1.0])
        })),
    };
    let mma = mma(
        &mma_objective,
        &mma_gradient,
        &x0,
        &[mma_constraint],
        &MmaConfig {
            bounds: vec![Bound::both(0.0, 3.0), Bound::both(0.0, 3.0)],
            ..MmaConfig::default()
        },
    )
    .map_err(|error| format!("MMA failed: {error}"))?;
    max_kkt_norm = max_kkt_norm.max(assert_gated(&mma, "MMA")?);

    Ok(vec![
        ("family_cases", 7.0),
        ("kkt_report_cases", 7.0),
        ("missing_kkt_reports", 0.0),
        ("nongated_successes", 0.0),
        ("max_kkt_norm", max_kkt_norm),
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
                .reference("independent constrained optimizer KKT success-gate checks")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-optimization-kkt-gate-family-v1",
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
