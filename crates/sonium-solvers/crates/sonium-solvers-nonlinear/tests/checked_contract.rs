use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_nonlinear::{
    AndersonConfig, AndersonError, NewtonConfig, NewtonError, anderson_acceleration, newton_raphson,
};

fn assert_failure_contract<E: SolverFailure>(
    error: &E,
    expected_status: SolverStatus,
    expected_accuracy: AccuracyStatus,
) {
    assert_eq!(error.status(), expected_status);
    assert_eq!(error.accuracy_status(), expected_accuracy);
}

#[test]
fn nonlinear_errors_implement_shared_failure_contract() {
    let anderson = AndersonError::NotConverged(3, 1.0e-3);
    assert_failure_contract(
        &anderson,
        SolverStatus::MaxIterationsReached,
        AccuracyStatus::NotAttained,
    );

    let newton = NewtonError::LineSearchFailed {
        iteration: 2,
        halvings: 8,
        residual: 1.0e-3,
    };
    assert_failure_contract(
        &newton,
        SolverStatus::Stagnated,
        AccuracyStatus::NotAttained,
    );
}

#[test]
fn nonlinear_fixtures_report_canonical_success_metadata() {
    let fixed_point = anderson_acceleration(
        &[0.0_f64],
        |x| vec![0.5 * x[0] + 1.0],
        &AndersonConfig {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .expect("scalar contraction should converge");
    assert_eq!(fixed_point.status, SolverStatus::Converged);
    assert_eq!(fixed_point.attainable_accuracy, AccuracyStatus::Attained);
    assert!((fixed_point.x[0] - 2.0).abs() < 1.0e-10);

    let newton = newton_raphson(
        &[0.0_f64],
        |x| vec![x[0] - 2.0],
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig::default(),
    )
    .expect("scalar Newton fixture should converge");
    assert_eq!(newton.status, SolverStatus::Converged);
    assert_eq!(newton.attainable_accuracy, AccuracyStatus::Attained);
    assert!((newton.x[0] - 2.0).abs() < 1.0e-10);
}

#[test]
fn anderson_checked_boundary_rejects_invalid_callback_contracts() {
    let invalid_config = anderson_acceleration(
        &[0.0_f64],
        |x| x.to_vec(),
        &AndersonConfig {
            mixing: f64::NAN,
            ..Default::default()
        },
    )
    .expect_err("invalid configuration must be rejected");
    assert_eq!(invalid_config.status(), SolverStatus::NumericalError);
    assert_eq!(
        invalid_config.attainable_accuracy(),
        AccuracyStatus::NotChecked
    );

    let wrong_length =
        anderson_acceleration(&[0.0_f64], |_| vec![0.0, 1.0], &AndersonConfig::default())
            .expect_err("wrong callback length must be typed");
    assert!(matches!(
        wrong_length,
        AndersonError::DimensionMismatch { .. }
    ));

    let non_finite =
        anderson_acceleration(&[0.0_f64], |_| vec![f64::NAN], &AndersonConfig::default())
            .expect_err("non-finite callback output must be typed");
    assert!(matches!(non_finite, AndersonError::NonFinite { .. }));

    let panic = anderson_acceleration(
        &[0.0_f64],
        |_| panic!("fixed-point callback failure"),
        &AndersonConfig::default(),
    )
    .expect_err("callback panic must be contained");
    assert!(matches!(panic, AndersonError::CallbackPanicked { .. }));
}

#[test]
fn newton_checked_boundary_contains_callback_panics() {
    let error = newton_raphson(
        &[0.0_f64],
        |_| panic!("residual callback failure"),
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig::default(),
    )
    .expect_err("Newton callback panic must be converted to a typed failure");
    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.attainable_accuracy(), AccuracyStatus::NotChecked);
}
