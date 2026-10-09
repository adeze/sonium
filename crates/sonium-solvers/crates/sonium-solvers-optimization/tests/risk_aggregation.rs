use sonium_solvers_core::traits::SolverResultContract;
use sonium_solvers_core::traits::{AccuracyStatus, SolverErrorKind, SolverFailure, SolverStatus};
use sonium_solvers_optimization::{
    BfgsConfig, RiskAggregation, RiskAggregationError, RiskObjective, RiskObjectiveEvaluation,
    ScenarioEvaluation, bfgs_risk, chance_constraint_satisfied, empirical_cvar,
    empirical_cvar_with_gradients, ks_aggregate, ks_aggregate_with_gradients,
};
use std::cell::Cell;

#[test]
fn cvar_uses_fractional_tail_weight_and_aligned_gradient() {
    let losses = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = empirical_cvar(&losses, 0.6).expect("valid CVaR");
    assert!((result.value - 4.5).abs() < 1.0e-14);
    assert!((result.threshold - 3.0).abs() < 1.0e-14);
    assert!((result.tail_mass - 2.0).abs() < 1.0e-14);
    assert_eq!(result.weights, vec![0.0, 0.0, 0.0, 0.5, 0.5]);

    let gradients = vec![
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![4.0, 1.0],
        vec![5.0, 2.0],
    ];
    let (result, gradient) =
        empirical_cvar_with_gradients(&losses, &gradients, 0.6).expect("valid CVaR gradient");
    assert!((result.value - 4.5).abs() < 1.0e-14);
    assert!((gradient[0] - 4.5).abs() < 1.0e-14);
    assert!((gradient[1] - 1.5).abs() < 1.0e-14);
    assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
}

#[test]
fn ks_is_stable_for_large_values_and_preserves_gradient_weights() {
    let values = vec![1000.0_f64, 999.0, 997.0];
    let result = ks_aggregate(&values, 10.0).expect("valid KS aggregate");
    assert!(result.value.is_finite());
    assert!(result.value >= 1000.0);
    assert!(result.value <= 1000.0 + (3.0_f64.ln() / 10.0));
    assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
    assert!(result.weights[0] > result.weights[1]);

    let gradients = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 3.0]];
    let (result, gradient) =
        ks_aggregate_with_gradients(&values, &gradients, 10.0).expect("valid KS gradient");
    assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
    assert!((gradient[0] - (result.weights[0] + 2.0 * result.weights[2])).abs() < 1.0e-14);
    assert!((gradient[1] - (result.weights[1] + 3.0 * result.weights[2])).abs() < 1.0e-14);
}

#[test]
fn chance_constraint_counts_failures_at_the_boundary() {
    let losses = [0.0_f64, 1.0, 1.0, 2.0];
    assert!(chance_constraint_satisfied(&losses, 1.0, 0.25).expect("valid chance constraint"));
    assert!(!chance_constraint_satisfied(&losses, 1.0, 0.24).expect("valid chance constraint"));
    assert!(chance_constraint_satisfied(&losses, 2.0, 0.0).expect("valid chance constraint"));
}

#[test]
fn risk_helpers_reject_invalid_inputs_before_math() {
    assert_eq!(
        empirical_cvar(&[1.0_f64, 2.0], 0.0).expect_err("zero confidence"),
        RiskAggregationError::InvalidConfidence
    );
    assert_eq!(
        ks_aggregate(&[1.0_f64], 0.0).expect_err("zero sharpness"),
        RiskAggregationError::InvalidSharpness
    );
    assert_eq!(
        empirical_cvar(&[1.0_f64, f64::NAN], 0.5).expect_err("NaN loss"),
        RiskAggregationError::NonFiniteInput
    );
    assert_eq!(
        empirical_cvar_with_gradients(&[1.0_f64, 2.0], &[vec![1.0]], 0.5)
            .expect_err("gradient row mismatch"),
        RiskAggregationError::DimensionMismatch {
            expected: 2,
            got: 1,
        }
    );
    assert_eq!(
        empirical_cvar_with_gradients(&[1.0_f64], &[vec![f64::NAN]], 0.5)
            .expect_err("non-finite gradient"),
        RiskAggregationError::NonFiniteInput
    );
    assert_eq!(
        chance_constraint_satisfied(&[1.0_f64], 1.0, 1.1).expect_err("invalid probability"),
        RiskAggregationError::InvalidProbability
    );
}

fn quadratic_scenarios(x: &[f64]) -> Result<ScenarioEvaluation<f64>, RiskAggregationError> {
    let delta = x
        .first()
        .copied()
        .ok_or(RiskAggregationError::DimensionMismatch {
            expected: 1,
            got: 0,
        })?
        - 1.0;
    let base = delta * delta;
    Ok(ScenarioEvaluation::new(
        vec![base, base + 0.5, base + 1.0],
        vec![vec![2.0 * delta]; 3],
    ))
}

#[test]
fn risk_objective_caches_value_and_gradient_and_propagates_weights() {
    let calls = Cell::new(0_usize);
    let objective = RiskObjective::new(
        |x: &[f64]| {
            calls.set(calls.get() + 1);
            quadratic_scenarios(x)
        },
        RiskAggregation::Cvar { confidence: 0.5 },
    );
    let evaluation = objective.evaluate(&[0.0]).expect("valid risk objective");
    assert!((evaluation.value - (11.0 / 6.0)).abs() < 1.0e-14);
    assert!((evaluation.gradient[0] + 2.0).abs() < 1.0e-14);
    assert!((evaluation.weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
    assert_eq!(objective.value(&[0.0]).unwrap(), evaluation.value);
    assert_eq!(objective.gradient(&[0.0]).unwrap(), evaluation.gradient);
    assert_eq!(
        calls.get(),
        1,
        "value and gradient should share one scenario run"
    );
}

#[test]
fn bfgs_risk_minimizes_ks_aggregated_scenarios() {
    let result = bfgs_risk(
        quadratic_scenarios,
        &[3.0],
        &BfgsConfig::default(),
        RiskAggregation::Ks { sharpness: 8.0 },
    )
    .expect("KS risk BFGS should converge");
    assert!(result.converged);
    assert!((result.x[0] - 1.0).abs() < 1.0e-6);
    assert!(result.fun.is_finite());
    result.validate_contract().expect("valid optimizer result");
}

#[test]
fn bfgs_risk_maps_scenario_shape_failures_to_optimizer_errors() {
    let error = bfgs_risk(
        |_| Ok(ScenarioEvaluation::new(vec![1.0], Vec::new())),
        &[0.0_f64],
        &BfgsConfig::default(),
        RiskAggregation::Cvar { confidence: 0.5 },
    )
    .expect_err("malformed scenario gradients must fail before iteration");
    assert!(matches!(
        error,
        sonium_solvers_optimization::OptimError::DimensionMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn risk_objective_rejects_design_gradient_dimension_mismatch() {
    let objective = RiskObjective::new(
        |_x: &[f64]| {
            Ok(ScenarioEvaluation::new(
                vec![1.0, 2.0],
                vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            ))
        },
        RiskAggregation::Cvar { confidence: 0.5 },
    );

    assert_eq!(
        objective
            .evaluate(&[0.0])
            .expect_err("wrong design gradient width"),
        RiskAggregationError::DesignDimensionMismatch {
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn risk_objective_contains_callback_panics() {
    let objective = RiskObjective::new(
        |_: &[f64]| -> Result<ScenarioEvaluation<f64>, RiskAggregationError> {
            panic!("scenario callback failed")
        },
        RiskAggregation::Cvar { confidence: 0.5 },
    );
    assert_eq!(
        objective.evaluate(&[0.0]).expect_err("panic must be typed"),
        RiskAggregationError::CallbackPanicked
    );
}

#[test]
fn risk_failures_expose_shared_solver_failure_contract() {
    let dimension = RiskAggregationError::DimensionMismatch {
        expected: 3,
        got: 2,
    };
    assert_eq!(dimension.error_kind(), SolverErrorKind::DimensionMismatch);
    assert_eq!(dimension.status(), SolverStatus::NumericalError);
    assert_eq!(dimension.accuracy_status(), AccuracyStatus::NotChecked);

    let design_dimension = RiskAggregationError::DesignDimensionMismatch {
        expected: 3,
        got: 2,
    };
    assert_eq!(
        design_dimension.error_kind(),
        SolverErrorKind::DimensionMismatch
    );
    assert_eq!(design_dimension.status(), SolverStatus::NumericalError);
    assert_eq!(
        design_dimension.accuracy_status(),
        AccuracyStatus::NotChecked
    );

    for error in [
        RiskAggregationError::EmptyInput,
        RiskAggregationError::InvalidConfidence,
        RiskAggregationError::InvalidSharpness,
        RiskAggregationError::InvalidProbability,
        RiskAggregationError::NonFiniteInput,
        RiskAggregationError::CallbackPanicked,
    ] {
        assert_eq!(error.error_kind(), SolverErrorKind::OperationFailed);
        assert_eq!(error.status(), SolverStatus::NumericalError);
        assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
    }
}
#[test]
fn risk_result_contracts_accept_valid_and_reject_malformed_packets() {
    let cvar = empirical_cvar(&[1.0_f64, 2.0, 3.0], 0.5).expect("valid CVaR");
    assert_eq!(cvar.solver_status(), SolverStatus::Converged);
    assert_eq!(cvar.accuracy_status(), AccuracyStatus::Attained);
    cvar.validate_contract().expect("CVaR contract");

    let mut bad_cvar = cvar.clone();
    bad_cvar.weights[0] = -0.1;
    assert!(bad_cvar.validate_contract().is_err());
    assert_eq!(bad_cvar.solver_status(), SolverStatus::NumericalError);

    let ks = ks_aggregate(&[1.0_f64, 2.0, 3.0], 4.0).expect("valid KS");
    ks.validate_contract().expect("KS contract");
    let mut bad_ks = ks.clone();
    bad_ks.weights[0] = f64::NAN;
    assert!(bad_ks.validate_contract().is_err());

    let evaluation = RiskObjectiveEvaluation {
        value: cvar.value,
        gradient: vec![1.0, -2.0],
        weights: cvar.weights.clone(),
    };
    evaluation.validate_contract().expect("evaluation contract");
    let mut bad_evaluation = evaluation.clone();
    bad_evaluation.gradient[1] = f64::INFINITY;
    assert!(bad_evaluation.validate_contract().is_err());
}
#[test]
fn risk_aggregators_reject_non_finite_generated_outputs() {
    assert_eq!(
        empirical_cvar(&[f64::MAX, f64::MAX], 0.1).expect_err("overflowing CVaR"),
        RiskAggregationError::NonFiniteInput
    );
    assert_eq!(
        ks_aggregate_with_gradients(&[0.0_f64, 0.0], &[vec![f64::NAN], vec![f64::MAX]], 1.0,)
            .expect_err("non-finite KS gradient"),
        RiskAggregationError::NonFiniteInput
    );
    assert_eq!(
        ks_aggregate(&[f64::MAX, f64::MAX], 1.0e-308).expect_err("overflowing KS value"),
        RiskAggregationError::NonFiniteInput
    );
}
#[test]
fn risk_result_contract_supports_f32() {
    let cvar = empirical_cvar(&[1.0_f32, 2.0, 3.0], 0.5).expect("valid f32 CVaR");
    cvar.validate_contract().expect("f32 CVaR contract");
    let ks = ks_aggregate(&[1.0_f32, 2.0, 3.0], 4.0).expect("valid f32 KS");
    ks.validate_contract().expect("f32 KS contract");
}
