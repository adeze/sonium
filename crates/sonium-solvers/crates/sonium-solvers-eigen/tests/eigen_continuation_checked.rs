use sonium_solvers_core::traits::SolverError;
use sonium_solvers_eigen::{
    EigenContinuationConfig, EigenContinuationEvent, EigenContinuationPoint,
    continue_generalized_eigenpair, continue_generalized_eigenpair_pseudo_arclength,
    predict_generalized_eigenpair,
};
use sonium_solvers_sparse::CsrMatrix;
use std::cell::RefCell;

fn diagonal(first: f64, second: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(2, 2, vec![(0, 0, first), (1, 1, second)])
}

fn point() -> EigenContinuationPoint<f64> {
    EigenContinuationPoint {
        parameter: 2.0,
        eigenvalue: 3.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    }
}

#[test]
fn predictor_matches_analytic_diagonal_branch() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let prediction = predict_generalized_eigenpair(
        &point(),
        0.25,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |rhs| {
            assert!(rhs.iter().all(|value| value.abs() < 1.0e-14));
            Ok(vec![0.0; rhs.len()])
        },
    )
    .expect("diagonal eigenpair predictor should converge");

    assert!((prediction.parameter - 2.25).abs() < 1.0e-14);
    assert!((prediction.eigenvalue - 3.25).abs() < 1.0e-14);
    assert!((prediction.eigenvalue_sensitivity - 1.0).abs() < 1.0e-14);
    assert!(prediction.right_eigenvector[0] == 1.0);
    assert!(prediction.right_eigenvector[1].abs() < 1.0e-14);
}

#[test]
fn predictor_includes_mass_derivative_in_tangent() {
    let mass = diagonal(2.0, 3.0);
    let mass_derivative = diagonal(1.0, 0.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let prediction = predict_generalized_eigenpair(
        &point(),
        -0.5,
        &mass,
        Some(&mass_derivative),
        &operator_derivative,
        |rhs| {
            assert!(rhs.iter().all(|value| value.abs() < 1.0e-14));
            Ok(vec![0.0; rhs.len()])
        },
    )
    .expect("mass-dependent eigenpair predictor should converge");

    assert!((prediction.eigenvalue_sensitivity + 0.5).abs() < 1.0e-14);
    assert!((prediction.eigenvalue - 3.25).abs() < 1.0e-14);
}

#[test]
fn predictor_rejects_invalid_step_before_singular_callback() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let error = predict_generalized_eigenpair(
        &point(),
        0.0,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| panic!("invalid step must be rejected before the callback"),
    )
    .expect_err("zero continuation step must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn predictor_rejects_malformed_eigenpair_dimensions() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let malformed = EigenContinuationPoint {
        parameter: 2.0,
        eigenvalue: 3.0,
        right_eigenvector: vec![1.0],
        left_eigenvector: vec![1.0],
    };
    let error = predict_generalized_eigenpair(
        &malformed,
        0.25,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0]),
    )
    .expect_err("malformed eigenpair dimensions must be rejected");
    assert!(matches!(error, SolverError::DimensionMismatch { .. }));
}

#[test]
fn adaptive_corrector_clamps_and_shrinks_signed_step() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let attempts = RefCell::new(Vec::new());
    let config = EigenContinuationConfig {
        min_step_size: 0.1,
        max_step_size: 0.5,
        shrink_factor: 0.5,
        max_retries: 3,
    };
    let step = continue_generalized_eigenpair(
        &point(),
        -0.8,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0; 2]),
        &config,
        |prediction| {
            attempts.borrow_mut().push(prediction.step_size);
            if prediction.step_size.abs() > 0.26 {
                return Err(SolverError::OperationFailed {
                    operation: "test corrector",
                    reason: "synthetic large-step failure".into(),
                });
            }
            Ok(EigenContinuationPoint {
                parameter: prediction.parameter,
                eigenvalue: prediction.eigenvalue,
                right_eigenvector: prediction.right_eigenvector.clone(),
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .expect("adaptive corrector should recover after one retry");

    assert_eq!(step.retries, 1);
    assert_eq!(attempts.into_inner(), vec![-0.5, -0.25]);
    assert!((step.prediction.step_size + 0.25).abs() < 1.0e-14);
    assert!((step.corrected_point.parameter - 1.75).abs() < 1.0e-14);
}

#[test]
fn adaptive_corrector_rejects_invalid_config_before_callbacks() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let config = EigenContinuationConfig {
        min_step_size: 0.0,
        ..EigenContinuationConfig::default()
    };
    let error = continue_generalized_eigenpair(
        &point(),
        0.25,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| panic!("invalid configuration must be rejected first"),
        &config,
        |_| panic!("invalid configuration must be rejected first"),
    )
    .expect_err("invalid adaptive configuration must fail closed");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn adaptive_corrector_contains_panics_and_exhausts_retries() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let config = EigenContinuationConfig {
        min_step_size: 0.1,
        max_step_size: 0.5,
        shrink_factor: 0.5,
        max_retries: 1,
    };
    let error = continue_generalized_eigenpair(
        &point(),
        0.5,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0; 2]),
        &config,
        |_| panic!("synthetic corrector panic"),
    )
    .expect_err("exhausted panicking corrector must return a typed error");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn adaptive_corrector_rejects_malformed_or_mismatched_points() {
    let mass = diagonal(2.0, 3.0);
    let operator_derivative = diagonal(2.0, 0.0);
    let config = EigenContinuationConfig {
        max_retries: 0,
        ..EigenContinuationConfig::default()
    };
    let malformed = continue_generalized_eigenpair(
        &point(),
        0.25,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0; 2]),
        &config,
        |_| {
            Ok(EigenContinuationPoint {
                parameter: 2.25,
                eigenvalue: 3.25,
                right_eigenvector: vec![1.0],
                left_eigenvector: vec![1.0],
            })
        },
    )
    .expect_err("malformed corrected vectors must be rejected");
    assert!(matches!(malformed, SolverError::DimensionMismatch { .. }));

    let mismatched = continue_generalized_eigenpair(
        &point(),
        0.25,
        &mass,
        None::<&CsrMatrix<f64>>,
        &operator_derivative,
        |_| Ok(vec![0.0; 2]),
        &config,
        |prediction| {
            Ok(EigenContinuationPoint {
                parameter: prediction.parameter + 1.0,
                eigenvalue: prediction.eigenvalue,
                right_eigenvector: vec![1.0, 0.0],
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .expect_err("corrected parameter must match predicted parameter");
    assert!(matches!(mismatched, SolverError::OperationFailed { .. }));
}
#[test]
fn pseudo_arclength_predictor_crosses_turning_point() {
    let mass = diagonal(2.0, 3.0);
    let previous = EigenContinuationPoint {
        parameter: 0.0,
        eigenvalue: 1.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let current = EigenContinuationPoint {
        parameter: 1.0,
        eigenvalue: 2.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let config = EigenContinuationConfig {
        min_step_size: 0.1,
        max_step_size: 2.0,
        shrink_factor: 0.5,
        max_retries: 2,
    };
    let step = continue_generalized_eigenpair_pseudo_arclength(
        &previous,
        &current,
        2.0_f64.sqrt(),
        &mass,
        &config,
        |prediction| {
            Ok(EigenContinuationPoint {
                parameter: 0.5,
                eigenvalue: prediction.eigenvalue,
                right_eigenvector: prediction.right_eigenvector.clone(),
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .expect("pseudo-arclength corrector should accept a fold");
    assert_eq!(step.retries, 0);
    assert_eq!(step.event, EigenContinuationEvent::TurningPoint);
    assert!((step.prediction.parameter - 2.0).abs() < 1.0e-14);
}

#[test]
fn pseudo_arclength_retries_and_classifies_branch_reversal() {
    let mass = diagonal(2.0, 3.0);
    let previous = EigenContinuationPoint {
        parameter: 0.0,
        eigenvalue: 1.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let current = EigenContinuationPoint {
        parameter: 1.0,
        eigenvalue: 2.0,
        right_eigenvector: vec![1.0, 0.0],
        left_eigenvector: vec![1.0, 0.0],
    };
    let config = EigenContinuationConfig {
        min_step_size: 0.1,
        max_step_size: 1.0,
        shrink_factor: 0.5,
        max_retries: 2,
    };
    let attempts = RefCell::new(0usize);
    let step = continue_generalized_eigenpair_pseudo_arclength(
        &previous,
        &current,
        1.0,
        &mass,
        &config,
        |prediction| {
            let attempt = {
                let mut attempts = attempts.borrow_mut();
                *attempts += 1;
                *attempts
            };
            if attempt == 1 {
                return Err(SolverError::OperationFailed {
                    operation: "test pseudo-arclength corrector",
                    reason: "synthetic failure".into(),
                });
            }
            Ok(EigenContinuationPoint {
                parameter: 0.0,
                eigenvalue: 1.0,
                right_eigenvector: prediction.right_eigenvector.clone(),
                left_eigenvector: vec![1.0, 0.0],
            })
        },
    )
    .expect("a shrunken pseudo-arclength step should recover");
    assert_eq!(step.retries, 1);
    assert_eq!(*attempts.borrow(), 2);
    assert_eq!(step.event, EigenContinuationEvent::BranchReversal);
}

#[test]
fn pseudo_arclength_rejects_degenerate_secant() {
    let mass = diagonal(2.0, 3.0);
    let current = point();
    let error = continue_generalized_eigenpair_pseudo_arclength(
        &current,
        &current,
        0.25,
        &mass,
        &EigenContinuationConfig::default(),
        |_| panic!("degenerate secant must fail before callback"),
    )
    .expect_err("identical continuation points must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}
