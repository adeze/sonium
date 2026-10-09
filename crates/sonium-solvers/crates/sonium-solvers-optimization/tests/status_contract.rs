use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_optimization::{BfgsConfig, OptimError, OptimResult, OptimStatus, bfgs};

fn assert_failure_contract<E: SolverFailure>(
    error: &E,
    expected_status: SolverStatus,
    expected_accuracy: AccuracyStatus,
) {
    assert_eq!(error.status(), expected_status);
    assert_eq!(error.accuracy_status(), expected_accuracy);
}

#[test]
fn optimization_errors_implement_shared_failure_contract() {
    let not_converged = OptimError::NotConverged {
        iterations: 3,
        grad_norm: 1.0e-3,
    };
    assert_failure_contract(
        &not_converged,
        SolverStatus::MaxIterationsReached,
        AccuracyStatus::NotAttained,
    );

    let infeasible = OptimError::Infeasible;
    assert_failure_contract(
        &infeasible,
        SolverStatus::Stagnated,
        AccuracyStatus::NotAttained,
    );
}

fn finite_result(status: OptimStatus, converged: bool) -> OptimResult<f64> {
    OptimResult {
        x: vec![0.0],
        fun: 0.0,
        grad: Some(vec![0.0]),
        iterations: 1,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 0.0,
        converged,
        status,
        kkt: None,
    }
}

#[test]
fn optimizer_results_normalize_success_and_failure_statuses() {
    for status in [
        OptimStatus::Converged,
        OptimStatus::FunctionTolerance,
        OptimStatus::StepTolerance,
    ] {
        let result = finite_result(status, true);
        assert_eq!(result.status(), SolverStatus::Converged);
        assert_eq!(result.attainable_accuracy(), AccuracyStatus::Attained);
    }

    let capped = finite_result(OptimStatus::MaxIterationsReached, false);
    assert_eq!(capped.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(capped.attainable_accuracy(), AccuracyStatus::NotAttained);

    for status in [
        OptimStatus::LineSearchFailed,
        OptimStatus::TrustRegionCollapsed,
        OptimStatus::KktNotSatisfied,
        OptimStatus::RestorationFailed,
    ] {
        let result = finite_result(status, false);
        assert_eq!(result.status(), SolverStatus::Stagnated);
        assert_eq!(result.attainable_accuracy(), AccuracyStatus::NotAttained);
    }

    let mut malformed = finite_result(OptimStatus::Converged, true);
    malformed.fun = f64::NAN;
    assert_eq!(malformed.status(), SolverStatus::NumericalError);
    assert_eq!(malformed.attainable_accuracy(), AccuracyStatus::NotChecked);
}

#[test]
fn optimizer_errors_normalize_typed_failures() {
    let not_converged = OptimError::NotConverged {
        iterations: 3,
        grad_norm: 1.0,
    };
    assert_eq!(not_converged.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(
        not_converged.attainable_accuracy(),
        AccuracyStatus::NotAttained
    );

    let callback = OptimError::CallbackPanicked {
        iteration: 0,
        stage: "objective".into(),
    };
    assert_eq!(callback.status(), SolverStatus::NumericalError);
    assert_eq!(callback.attainable_accuracy(), AccuracyStatus::NotChecked);

    let infeasible = OptimError::Infeasible;
    assert_eq!(infeasible.status(), SolverStatus::Stagnated);
    assert_eq!(
        infeasible.attainable_accuracy(),
        AccuracyStatus::NotAttained
    );
}

#[test]
fn bfgs_quadratic_fixture_exposes_canonical_result_metadata() {
    let result = bfgs(
        &|x: &[f64]| 0.5 * x[0] * x[0],
        &|x: &[f64]| vec![x[0]],
        &[2.0],
        &BfgsConfig::default(),
    )
    .expect("quadratic BFGS fixture should converge");

    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    assert!(result.x[0].abs() < 1.0e-6);

    let error = bfgs(
        &|_: &[f64]| f64::NAN,
        &|_: &[f64]| vec![0.0],
        &[0.0],
        &BfgsConfig::default(),
    )
    .expect_err("non-finite objective must remain a typed failure");
    assert!(matches!(error, OptimError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.attainable_accuracy(), AccuracyStatus::NotChecked);
}
