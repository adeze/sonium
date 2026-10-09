use sonium_solvers_optimization::{
    AdvancedSqpConfig, AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind,
    KktMultipliers, LbfgsbConfig, MmaConfig, OptimError, OptimResult, OptimStatus, PenaltyConfig,
    SlsqpConfig, SqpConfig, TrustRegionConstrainedConfig, augmented_lagrangian, lbfgsb, mma,
    penalty_method, slsqp, sqp, sqp_advanced, trust_region_constrained,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn attached_kkt_report_is_a_success_gate() {
    let mut result = OptimResult {
        x: vec![0.0_f64],
        fun: 0.0,
        grad: Some(vec![1.0]),
        iterations: 1,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 1.0,
        converged: true,
        status: OptimStatus::Converged,
        kkt: None,
    };
    result
        .attach_kkt(
            &[],
            &[Bound::lower(0.0)],
            KktMultipliers {
                constraints: Vec::new(),
                lower_bounds: vec![1.0],
                upper_bounds: vec![0.0],
            },
        )
        .unwrap();
    assert_eq!(result.kkt.as_ref().unwrap().norm, 0.0);

    result.gate_kkt(1.0e-12);
    assert!(result.converged);
    assert_eq!(result.status, OptimStatus::Converged);

    let mut failing = result.clone();
    failing.kkt = None;
    failing.gate_kkt(1.0e-12);
    assert!(!failing.converged);
    assert_eq!(failing.status, OptimStatus::KktNotSatisfied);
}

#[test]
fn kkt_gate_rejects_invalid_tolerance_and_attachment_rejects_non_finite_multipliers() {
    let mut result = OptimResult {
        x: vec![0.0_f64],
        fun: 0.0,
        grad: Some(vec![0.0]),
        iterations: 0,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 0.0,
        converged: true,
        status: OptimStatus::Converged,
        kkt: None,
    };
    result
        .attach_kkt(
            &[],
            &[Bound::lower(0.0)],
            KktMultipliers {
                constraints: Vec::new(),
                lower_bounds: vec![f64::NAN],
                upper_bounds: vec![0.0],
            },
        )
        .unwrap_err();

    result.gate_kkt(f64::NAN);
    assert!(!result.converged);
    assert_eq!(result.status, OptimStatus::KktNotSatisfied);
}

#[test]
fn kkt_diagnostics_reject_non_finite_constraint_values() {
    let result = OptimResult {
        x: vec![0.0_f64],
        fun: 0.0,
        grad: Some(vec![0.0]),
        iterations: 0,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 0.0,
        converged: false,
        status: OptimStatus::MaxFunctionEvals,
        kkt: None,
    };
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|_| vec![f64::NAN]),
        jac: Some(Box::new(|_| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    assert!(matches!(
        result.kkt_diagnostics(&constraints, &[], None),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn kkt_report_covers_equality_and_active_inactive_inequality_rows() {
    let mut result = OptimResult {
        x: vec![1.0_f64, 2.0],
        fun: 0.0,
        grad: Some(vec![6.0, 0.0]),
        iterations: 1,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 6.0,
        converged: true,
        status: OptimStatus::Converged,
        kkt: None,
    };
    let constraints = vec![
        Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 0.0])
            })),
        },
        Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[1] - 1.0]),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![0.0, 1.0])
            })),
        },
        Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 0.0])
            })),
        },
    ];
    result
        .attach_kkt(
            &constraints,
            &[],
            KktMultipliers {
                constraints: vec![6.0, 0.0, 0.0],
                lower_bounds: Vec::new(),
                upper_bounds: Vec::new(),
            },
        )
        .unwrap();
    let report = result.kkt.as_ref().unwrap();
    assert_eq!(report.diagnostics.primal_feasibility, 0.0);
    assert_eq!(report.diagnostics.dual_feasibility, 0.0);
    assert_eq!(report.diagnostics.complementarity, 0.0);
    assert_eq!(report.diagnostics.stationarity, 0.0);
    result.gate_kkt(1.0e-12);
    assert!(result.converged);
}

#[test]
fn sqp_public_result_carries_original_problem_kkt_report() {
    let objective = |x: &[f64]| (x[0] - 2.0).powi(2);
    let gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![x[0] - 1.0]),
        jac: Some(Box::new(|_| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    let result = sqp(
        &objective,
        &gradient,
        &[0.0],
        &constraints,
        &SqpConfig::default(),
    )
    .unwrap();
    let report = result.kkt.as_ref().expect("SQP must attach KKT report");
    assert!(report.diagnostics.primal_feasibility < 1.0e-4);
    assert!(report.diagnostics.stationarity < 1.0e-3);
    assert!(report.norm.is_finite());
}

#[test]
fn constrained_solver_rejects_malformed_bounds_before_callbacks() {
    let objective_calls = AtomicUsize::new(0);
    let objective = |_: &[f64]| {
        objective_calls.fetch_add(1, Ordering::Relaxed);
        0.0
    };
    let gradient = |_: &[f64]| vec![0.0, 0.0];
    let config = SqpConfig {
        bounds: vec![Bound::both(1.0, 0.0)],
        ..SqpConfig::default()
    };
    assert!(matches!(
        sqp(&objective, &gradient, &[0.0, 0.0], &[], &config),
        Err(OptimError::InvalidBounds { .. }) | Err(OptimError::DimensionMismatch { .. })
    ));
    assert_eq!(objective_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn sqp_returns_typed_failure_for_non_finite_objective_callback() {
    let objective = |_x: &[f64]| f64::NAN;
    let gradient = |_x: &[f64]| vec![0.0];

    assert!(matches!(
        sqp(&objective, &gradient, &[0.0], &[], &SqpConfig::default()),
        Err(OptimError::NumericalError { iteration: 0, .. })
    ));
}

#[test]
fn advanced_sqp_returns_typed_failure_for_non_finite_objective_callback() {
    let objective_calls = AtomicUsize::new(0);
    let objective = |_: &[f64]| {
        if objective_calls.fetch_add(1, Ordering::Relaxed) == 0 {
            0.0
        } else {
            f64::NAN
        }
    };
    let gradient = |_x: &[f64]| vec![0.0];
    let hessian =
        |_x: &[f64], _lambda: &[f64]| sonium_solvers_core::dense::DenseMatrix::from_diag(&[1.0]);
    let config = AdvancedSqpConfig {
        max_iterations: 1,
        ..AdvancedSqpConfig::default()
    };

    assert!(matches!(
        sqp_advanced(&objective, &gradient, &hessian, &[0.0], &[], &config),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn advanced_sqp_rejects_non_finite_generated_hessian() {
    let hessian_calls = AtomicUsize::new(0);
    let objective = |x: &[f64]| (x[0] - 1.0).powi(2);
    let gradient = |x: &[f64]| vec![2.0 * (x[0] - 1.0)];
    let hessian = |_: &[f64], _: &[f64]| {
        if hessian_calls.fetch_add(1, Ordering::Relaxed) == 0 {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0])
        } else {
            sonium_solvers_core::dense::DenseMatrix::from_diag(&[f64::NAN])
        }
    };

    assert!(matches!(
        sqp_advanced(
            &objective,
            &gradient,
            &hessian,
            &[0.0],
            &[],
            &AdvancedSqpConfig {
                max_iterations: 1,
                ..AdvancedSqpConfig::default()
            },
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn sqp_contains_callback_panics_and_bad_gradient_dimensions() {
    let panicking_objective = |_x: &[f64]| -> f64 { panic!("objective callback failure") };
    let gradient = |_x: &[f64]| vec![0.0];
    assert!(matches!(
        sqp(
            &panicking_objective,
            &gradient,
            &[0.0],
            &[],
            &SqpConfig::default()
        ),
        Err(OptimError::CallbackPanicked { .. })
    ));

    let objective = |_x: &[f64]| 0.0;
    let bad_gradient = |_x: &[f64]| Vec::<f64>::new();
    assert!(matches!(
        sqp(
            &objective,
            &bad_gradient,
            &[0.0],
            &[],
            &SqpConfig::default()
        ),
        Err(OptimError::DimensionMismatch {
            expected: 1,
            got: 0
        })
    ));

    let panicking_constraint = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(|_: &[f64]| -> Vec<f64> { panic!("constraint callback failure") }),
        jac: None,
    }];
    assert!(matches!(
        sqp(
            &objective,
            &gradient,
            &[0.0],
            &panicking_constraint,
            &SqpConfig::default()
        ),
        Err(OptimError::CallbackPanicked { .. })
    ));
}

#[test]
fn lbfgsb_attaches_bound_kkt_report() {
    let objective = |x: &[f64]| (x[0] - 2.0).powi(2);
    let gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
    let result = lbfgsb(
        &objective,
        &gradient,
        &[0.0],
        &LbfgsbConfig {
            bounds: vec![Bound::upper(1.0)],
            ..LbfgsbConfig::default()
        },
    )
    .unwrap();
    let report = result
        .kkt
        .as_ref()
        .expect("L-BFGS-B must attach KKT report");
    assert_eq!(report.diagnostics.active_upper_bounds, vec![0]);
    assert!(report.norm < 1.0e-3);
}

#[test]
fn lbfgsb_rejects_non_finite_generated_callback_values() {
    let objective_calls = AtomicUsize::new(0);
    let objective = |x: &[f64]| {
        if objective_calls.fetch_add(1, Ordering::Relaxed) < 2 {
            (x[0] - 1.0).powi(2)
        } else {
            f64::NAN
        }
    };
    let gradient = |_x: &[f64]| vec![1.0];

    assert!(matches!(
        lbfgsb(
            &objective,
            &gradient,
            &[1.0],
            &LbfgsbConfig {
                max_iterations: 1,
                ..LbfgsbConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn mma_rejects_non_finite_generated_objective() {
    let objective_calls = AtomicUsize::new(0);
    let objective = |x: &[f64]| {
        if objective_calls.fetch_add(1, Ordering::Relaxed) < 2 {
            x[0] * x[0]
        } else {
            f64::NAN
        }
    };
    let gradient = |_x: &[f64]| vec![0.0];

    assert!(matches!(
        mma(
            &objective,
            &gradient,
            &[0.0],
            &[],
            &MmaConfig {
                max_iterations: 1,
                ..MmaConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn trust_region_rejects_non_finite_generated_callback_values() {
    let objective_calls = AtomicUsize::new(0);
    let objective = |x: &[f64]| {
        if objective_calls.fetch_add(1, Ordering::Relaxed) < 2 {
            x[0] * x[0]
        } else {
            f64::NAN
        }
    };
    let gradient = |_x: &[f64]| vec![0.0];
    let hess_vec = |_x: &[f64], _v: &[f64]| vec![0.0];

    assert!(matches!(
        trust_region_constrained(
            &objective,
            &gradient,
            &hess_vec,
            &[0.0],
            &[],
            &TrustRegionConstrainedConfig {
                max_iterations: 1,
                ..TrustRegionConstrainedConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn slsqp_rejects_non_finite_generated_constraint_values() {
    let constraint_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&constraint_calls);
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(move |x: &[f64]| {
            if callback_calls.fetch_add(1, Ordering::Relaxed) < 5 {
                vec![x[0] - 1.0]
            } else {
                vec![f64::NAN]
            }
        }),
        jac: Some(Box::new(|_x: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    let objective = |x: &[f64]| x[0] * x[0];
    let gradient = |x: &[f64]| vec![2.0 * x[0]];

    assert!(matches!(
        slsqp(
            &objective,
            &gradient,
            &[0.0],
            &constraints,
            &SlsqpConfig {
                max_iterations: 1,
                ..SlsqpConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn augmented_lagrangian_rejects_non_finite_generated_constraint_values() {
    let constraint_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&constraint_calls);
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(move |x: &[f64]| {
            if callback_calls.fetch_add(1, Ordering::Relaxed) < 2 {
                vec![x[0] - 1.0]
            } else {
                vec![f64::NAN]
            }
        }),
        jac: Some(Box::new(|_x: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    let objective = |x: &[f64]| x[0] * x[0];
    let gradient = |x: &[f64]| vec![2.0 * x[0]];

    assert!(matches!(
        augmented_lagrangian(
            &objective,
            &gradient,
            &[0.0],
            &constraints,
            &AugmentedLagrangianConfig {
                max_outer_iterations: 1,
                ..AugmentedLagrangianConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn penalty_rejects_non_finite_generated_constraint_values() {
    let constraint_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&constraint_calls);
    let constraints = vec![Constraint {
        kind: ConstraintKind::Equality,
        dim: 1,
        fun: Box::new(move |x: &[f64]| {
            if callback_calls.fetch_add(1, Ordering::Relaxed) < 2 {
                vec![x[0] - 1.0]
            } else {
                vec![f64::NAN]
            }
        }),
        jac: Some(Box::new(|_x: &[f64]| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 1, vec![1.0])
        })),
    }];
    let objective = |x: &[f64]| x[0] * x[0];
    let gradient = |x: &[f64]| vec![2.0 * x[0]];

    assert!(matches!(
        penalty_method(
            &objective,
            &gradient,
            &[0.0],
            &constraints,
            &PenaltyConfig {
                max_outer_iterations: 1,
                ..PenaltyConfig::default()
            }
        ),
        Err(OptimError::NumericalError { .. })
    ));
}

#[test]
fn constrained_success_paths_attach_and_gate_kkt_reports() {
    fn objective(x: &[f64]) -> f64 {
        x.iter().map(|value| value * value).sum()
    }

    fn gradient(x: &[f64]) -> Vec<f64> {
        x.iter().map(|value| 2.0 * value).collect()
    }

    fn equality_constraint() -> Constraint<f64> {
        Constraint {
            kind: ConstraintKind::Equality,
            dim: 1,
            fun: Box::new(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
            jac: Some(Box::new(|_| {
                sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![1.0, 1.0])
            })),
        }
    }

    fn assert_gated(result: &OptimResult<f64>, method: &str) {
        assert!(result.converged, "{method} did not converge: {result:?}");
        assert!(
            matches!(result.status, OptimStatus::Converged),
            "{method} returned an unexpected status: {:?}",
            result.status
        );
        let report = result
            .kkt
            .as_ref()
            .unwrap_or_else(|| panic!("{method} returned no KKT report"));
        assert!(report.norm.is_finite(), "{method} KKT norm is non-finite");
        assert!(
            report.norm <= 1.0e-2,
            "{method} KKT norm too large: {}",
            report.norm
        );
    }

    let x0 = [0.0, 0.0];
    let penalty = penalty_method(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &PenaltyConfig::default(),
    )
    .expect("penalty method should return a result");
    assert_gated(&penalty, "penalty");

    let augmented = augmented_lagrangian(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &AugmentedLagrangianConfig::default(),
    )
    .expect("augmented Lagrangian should return a result");
    assert_gated(&augmented, "augmented Lagrangian");

    let trust = trust_region_constrained(
        &objective,
        &gradient,
        &|_x: &[f64], v: &[f64]| v.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
        &x0,
        &[equality_constraint()],
        &TrustRegionConstrainedConfig::default(),
    )
    .expect("trust-region method should return a result");
    assert_gated(&trust, "trust-region");

    let slsqp = slsqp(
        &objective,
        &gradient,
        &x0,
        &[equality_constraint()],
        &SlsqpConfig::default(),
    )
    .expect("SLSQP should return a result");
    assert_gated(&slsqp, "SLSQP");

    let hessian = |_x: &[f64], _lambda: &[f64]| {
        sonium_solvers_core::dense::DenseMatrix::from_diag(&[2.0, 2.0])
    };
    let advanced_sqp = sqp_advanced(
        &objective,
        &gradient,
        &hessian,
        &x0,
        &[equality_constraint()],
        &AdvancedSqpConfig::default(),
    )
    .expect("advanced SQP should return a result");
    assert_gated(&advanced_sqp, "advanced SQP");

    let mma_constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new(|x: &[f64]| vec![2.0 - x[0] - x[1]]),
        jac: Some(Box::new(|_| {
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(1, 2, vec![-1.0, -1.0])
        })),
    };
    let mma_objective = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
    let mma_gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)];
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
    .expect("MMA should return a result");
    assert_gated(&mma, "MMA");
}
