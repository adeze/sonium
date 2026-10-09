use sonium_solvers_core::traits::{
    AccuracyStatus, SolverError, SolverResultContract, SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_time_integration::{
    ExplicitConfig, GeneralizedAlphaConfig, NewmarkConfig, NewmarkInitial, NewmarkResult,
    RevolveSchedule, try_explicit_integrate, try_generalized_alpha_integrate,
    try_newmark_integrate,
};

fn identity() -> CsrMatrix<f64> {
    CsrMatrix::identity(1)
}

fn initial() -> NewmarkInitial<f64> {
    NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    }
}

#[test]
fn checked_time_integrators_match_independent_one_dof_fixture() {
    let stiffness = identity();
    let config = NewmarkConfig::average_acceleration(0.01, 10);
    let result = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial(),
        |rhs: &[f64]| rhs.to_vec(),
        &config,
    )
    .expect("Newmark checked fixture should complete");
    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.attainable_accuracy(), AccuracyStatus::NotChecked);
    SolverResultContract::validate_contract(&result)
        .expect("finite Newmark trajectory should satisfy shared contract");
    assert!((result.final_time - 0.1).abs() < 1.0e-14);
    assert!((result.displacement[0] - (0.1_f64).cos()).abs() < 1.0e-5);

    let generalized = try_generalized_alpha_integrate(
        &stiffness,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial(),
        |rhs: &[f64]| rhs.to_vec(),
        &GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.01, 10),
    )
    .expect("generalized-alpha checked fixture should complete");
    assert_eq!(generalized.solver_status(), SolverStatus::Converged);
    assert!(generalized.displacement[0].is_finite());

    let explicit = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _time| vec![0.0],
        |rhs: &[f64]| rhs.to_vec(),
        &initial(),
        &ExplicitConfig::new(0.01, 10),
    )
    .expect("explicit checked fixture should complete");
    assert_eq!(explicit.status(), SolverStatus::Converged);
    assert!(explicit.velocity[0].is_finite());
}

#[test]
fn checked_time_integrators_reject_dimensions_and_contain_callbacks() {
    let bad_stiffness = CsrMatrix::identity(2);
    let dimension = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &bad_stiffness,
        |_time| vec![0.0],
        &initial(),
        |rhs: &[f64]| rhs.to_vec(),
        &NewmarkConfig::average_acceleration(0.1, 1),
    )
    .expect_err("operator dimension mismatch must be typed");
    assert!(matches!(dimension, SolverError::DimensionMismatch { .. }));

    let panic = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &CsrMatrix::identity(1),
        |_time| panic!("force callback failure"),
        &initial(),
        |rhs: &[f64]| rhs.to_vec(),
        &NewmarkConfig::average_acceleration(0.1, 1),
    )
    .expect_err("callback panic must be contained");
    assert!(matches!(panic, SolverError::OperationFailed { .. }));

    let explicit_initial = initial();
    let generated_dimension = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _time| Vec::new(),
        |rhs: &[f64]| rhs.to_vec(),
        &explicit_initial,
        &ExplicitConfig::new(0.1, 1),
    )
    .expect_err("generated force dimension must be typed");
    assert!(matches!(
        generated_dimension,
        SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        }
    ));

    let generated_force = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _time| vec![f64::NAN],
        |rhs: &[f64]| rhs.to_vec(),
        &explicit_initial,
        &ExplicitConfig::new(0.1, 1),
    )
    .expect_err("generated force non-finite value must be typed");
    assert!(matches!(
        generated_force,
        SolverError::OperationFailed { .. }
    ));

    let generated_mass = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _time| vec![0.0],
        |_rhs: &[f64]| vec![f64::INFINITY],
        &explicit_initial,
        &ExplicitConfig::new(0.1, 1),
    )
    .expect_err("generated mass non-finite value must be typed");
    assert!(matches!(
        generated_mass,
        SolverError::OperationFailed { .. }
    ));

    let invalid = ExplicitConfig::new(f64::INFINITY, 1).validate();
    assert!(matches!(invalid, Err(SolverError::OperationFailed { .. })));

    let schedule = RevolveSchedule::new(1, 0).expect_err("checkpoint budget must be typed");
    assert_eq!(schedule.status(), SolverStatus::NumericalError);
    assert_eq!(schedule.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn checked_implicit_callbacks_report_typed_failures() {
    let stiffness = identity();
    let initial = initial();
    let newmark_config = NewmarkConfig::average_acceleration(0.1, 1);

    let newmark_force_dimension = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| Vec::new(),
        &initial,
        |rhs: &[f64]| rhs.to_vec(),
        &newmark_config,
    )
    .expect_err("Newmark force dimension must be typed");
    assert!(matches!(
        newmark_force_dimension,
        SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        }
    ));

    let newmark_solve_nonfinite = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial,
        |_rhs: &[f64]| vec![f64::NAN],
        &newmark_config,
    )
    .expect_err("Newmark solve output must be typed");
    assert!(matches!(
        newmark_solve_nonfinite,
        SolverError::OperationFailed {
            operation: "Newmark integration linear-solve callback",
            ..
        }
    ));

    let generalized_config = GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.1, 1);
    let generalized_force_nonfinite = try_generalized_alpha_integrate(
        &stiffness,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![f64::INFINITY],
        &initial,
        |rhs: &[f64]| rhs.to_vec(),
        &generalized_config,
    )
    .expect_err("generalized-alpha force output must be typed");
    assert!(matches!(
        generalized_force_nonfinite,
        SolverError::OperationFailed {
            operation: "generalized-alpha integration force callback",
            ..
        }
    ));

    let generalized_solve_dimension = try_generalized_alpha_integrate(
        &stiffness,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial,
        |_rhs: &[f64]| Vec::new(),
        &generalized_config,
    )
    .expect_err("generalized-alpha solve dimension must be typed");
    assert!(matches!(
        generalized_solve_dimension,
        SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn checked_generalized_and_explicit_results_round_trip_checkpoints() {
    let stiffness = identity();
    let initial = initial();
    let generalized = try_generalized_alpha_integrate(
        &stiffness,
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial,
        |rhs: &[f64]| rhs.to_vec(),
        &GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.01, 3).with_history(),
    )
    .expect("generalized-alpha checkpoint fixture");
    let explicit = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _time| vec![0.0],
        |rhs: &[f64]| rhs.to_vec(),
        &initial,
        &ExplicitConfig::new(0.01, 3).with_history(),
    )
    .expect("explicit checkpoint fixture");

    for result in [&generalized, &explicit] {
        let encoded = result.to_checkpoint_bytes().expect("encode trajectory");
        let restored = NewmarkResult::from_checkpoint_bytes(&encoded).expect("decode trajectory");
        assert_eq!(restored.displacement, result.displacement);
        assert_eq!(restored.velocity, result.velocity);
        assert_eq!(restored.acceleration, result.acceleration);
        assert_eq!(restored.displacement_history, result.displacement_history);
        assert_eq!(restored.velocity_history, result.velocity_history);
        assert_eq!(restored.acceleration_history, result.acceleration_history);
        assert_eq!(restored.steps_completed, result.steps_completed);
        assert_eq!(restored.final_time, result.final_time);
        assert_eq!(restored.status(), result.status());
        SolverResultContract::validate_contract(&restored)
            .expect("restored trajectory should satisfy result contract");
    }
}

#[test]
fn newmark_result_contract_rejects_incomplete_or_partial_histories() {
    let base = || NewmarkResult {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
        displacement_history: Some(vec![vec![1.0], vec![0.99]]),
        velocity_history: Some(vec![vec![0.0], vec![-0.1]]),
        acceleration_history: Some(vec![vec![-1.0], vec![-0.99]]),
        steps_completed: 1,
        final_time: 0.01,
    };

    let complete = base();
    assert_eq!(complete.solver_status(), SolverStatus::Converged);
    complete
        .validate_contract()
        .expect("complete trajectory should satisfy the result contract");

    let mut truncated = base();
    truncated
        .displacement_history
        .as_mut()
        .expect("history present")
        .pop();
    assert_eq!(truncated.solver_status(), SolverStatus::NumericalError);
    assert!(matches!(
        truncated.validate_contract(),
        Err(SolverError::OperationFailed { .. })
    ));

    let mut partial = base();
    partial.velocity_history = None;
    assert_eq!(partial.solver_status(), SolverStatus::NumericalError);
    assert!(matches!(
        partial.validate_contract(),
        Err(SolverError::OperationFailed { .. })
    ));
}
