use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_nonlinear::{
    InexactNewtonConfig, LineSearch, NewtonConfig, NewtonError, newton_raphson,
    newton_raphson_continuation,
};
use std::cell::Cell;

#[test]
fn newton_rejects_nonfinite_final_residual_as_numerical_error() {
    let calls = Cell::new(0usize);
    let config = NewtonConfig {
        max_iterations: 1,
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..NewtonConfig::default()
    };
    let error = newton_raphson(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 0 { vec![1.0] } else { vec![f64::NAN] }
        },
        |_x, rhs| Ok(rhs.to_vec()),
        &config,
    )
    .expect_err("non-finite final residual must fail closed");

    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn newton_rejects_malformed_no_line_search_trial_residual() {
    let calls = Cell::new(0usize);
    let error = newton_raphson(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            match call {
                0 => vec![1.0],
                1 => Vec::new(),
                _ => vec![0.25],
            }
        },
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig {
            max_iterations: 2,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..NewtonConfig::default()
        },
    )
    .expect_err("malformed no-line-search trial residual must fail closed");

    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
}

#[test]
fn continuation_rejects_nonfinite_final_residual_as_numerical_error() {
    let calls = Cell::new(0usize);
    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            line_search: LineSearch::None,
            ..NewtonConfig::default()
        },
        ..InexactNewtonConfig::default()
    };
    let error = newton_raphson_continuation(
        &[0.0_f64],
        |_x| {
            let call = calls.get();
            calls.set(call + 1);
            match call {
                0 => vec![1.0],
                1 => vec![0.5],
                _ => vec![f64::NAN],
            }
        },
        |_x, v| v.to_vec(),
        &config,
    )
    .expect_err("continuation non-finite final residual must fail closed");

    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn continuation_rejects_malformed_jacobian_matvec_output() {
    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            line_search: LineSearch::None,
            ..NewtonConfig::default()
        },
        ..InexactNewtonConfig::default()
    };
    let error =
        newton_raphson_continuation(&[0.0_f64], |_x| vec![1.0], |_x, _v| Vec::new(), &config)
            .expect_err("malformed Jacobian matvec output must fail closed");

    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn continuation_rejects_nonfinite_jacobian_matvec_output() {
    let config = InexactNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            line_search: LineSearch::None,
            ..NewtonConfig::default()
        },
        ..InexactNewtonConfig::default()
    };
    let error =
        newton_raphson_continuation(&[0.0_f64], |_x| vec![1.0], |_x, _v| vec![f64::NAN], &config)
            .expect_err("non-finite Jacobian matvec output must fail closed");

    assert!(matches!(error, NewtonError::NumericalError { .. }));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}
