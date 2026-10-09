use sonium_solvers::FgmresConfig;
use sonium_solvers::mixed_precision::{mixed_precision_solve, try_mixed_precision_solve};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{LinearOperator, Preconditioner, SolverError, SolverStatus};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn checked_mixed_precision_wrapper_reports_original_operator_residual() {
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0_f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
    );
    let preconditioner = DiagonalPreconditioner::from_diagonal(&[4.0_f32, 4.0, 4.0]);
    let result = try_mixed_precision_solve::<f64, f32, _>(
        &matrix,
        &[1.0, 2.0, 3.0],
        Box::new(preconditioner),
        &FgmresConfig {
            max_iterations: 8,
            restart: 4,
            tolerance: 1e-12,
            // The mixed-precision wrapper must promote the default final-only
            // policy to an exact candidate check so the rejected first
            // projected iterate restarts instead of being reported converged.
            exact_residual_check_interval: 0,
            ..FgmresConfig::default()
        },
    )
    .expect("checked mixed-precision solve");
    assert!(result.converged);
    assert!(result.residual.is_finite());
    assert!(result.residual < 1e-12);
    assert!(
        result.op_counts.exact_residual_checks > 1,
        "mixed-precision wrapper must propagate interval checks, got {}",
        result.op_counts.exact_residual_checks,
    );
}

#[test]
fn checked_mixed_precision_aborts_before_operator_after_low_precision_failure() {
    struct CountingIdentity {
        calls: AtomicUsize,
    }

    impl LinearOperator<f64> for CountingIdentity {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            y.copy_from_slice(x);
        }

        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            self.apply_into(x, y);
        }
    }

    struct PoisoningPreconditioner;
    impl Preconditioner<f32> for PoisoningPreconditioner {
        fn apply_into(&self, _r: &[f32], y: &mut [f32]) {
            y.fill(f32::NAN);
        }
    }

    let operator = CountingIdentity {
        calls: AtomicUsize::new(0),
    };
    let error = try_mixed_precision_solve::<f64, f32, _>(
        &operator,
        &[1.0],
        Box::new(PoisoningPreconditioner),
        &FgmresConfig {
            max_iterations: 8,
            restart: 1,
            ..FgmresConfig::default()
        },
    )
    .expect_err("low-precision failure must abort the checked solve");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "mixed-precision preconditioner",
            ..
        }
    ));
    assert_eq!(
        operator.calls.load(Ordering::SeqCst),
        1,
        "the failed low-precision correction must not reach another operator application"
    );
}

#[test]
fn checked_mixed_precision_wrapper_rejects_non_finite_preconditioner_output() {
    struct NonFinitePreconditioner;

    impl Preconditioner<f32> for NonFinitePreconditioner {
        fn apply(&self, r: &[f32]) -> Vec<f32> {
            vec![f32::NAN; r.len()]
        }

        fn apply_into(&self, _r: &[f32], y: &mut [f32]) {
            y.fill(f32::NAN);
        }
    }

    let matrix = CsrMatrix::<f64>::identity(2);
    let error = try_mixed_precision_solve::<f64, f32, _>(
        &matrix,
        &[1.0, 2.0],
        Box::new(NonFinitePreconditioner),
        &FgmresConfig::default(),
    )
    .expect_err("non-finite inner preconditioner output must fail");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn checked_mixed_precision_wrapper_restarts_after_promoted_gate_rejection() {
    struct FirstProjectedCandidateOperator {
        calls: AtomicUsize,
    }

    impl LinearOperator<f64> for FirstProjectedCandidateOperator {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            // Let the initial residual and first Arnoldi action look like a
            // nearly-identity operator, then switch to 2I for the promoted
            // exact gate and all later calls. The first projected candidate
            // is therefore rejected; the restarted cycle must correct it to
            // x=b/2.
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < 2 {
                let epsilon = 1.0e-10;
                y[0] = x[0] + epsilon * x[1];
                y[1] = x[1];
            } else {
                for (out, &value) in y.iter_mut().zip(x) {
                    *out = 2.0 * value;
                }
            }
        }

        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            self.apply_into(x, y);
        }
    }

    let result = try_mixed_precision_solve::<f64, f32, _>(
        &FirstProjectedCandidateOperator {
            calls: AtomicUsize::new(0),
        },
        &[1.0, 1.0],
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &FgmresConfig {
            max_iterations: 2,
            restart: 1,
            tolerance: 1.0e-12,
            exact_residual_check_interval: 1,
            ..FgmresConfig::default()
        },
    )
    .expect("the restarted mixed-precision solve should converge");

    assert!(
        result.converged,
        "status={:?}, residual={}, estimated={}, iterations={}, restarts={}",
        result.status,
        result.residual,
        result.estimated_residual,
        result.iterations,
        result.restarts
    );
    assert_eq!(result.restarts, 1);
    assert_eq!(result.iterations, 2);
    assert!(result.residual < 1.0e-12, "residual={}", result.residual);
    assert!(result.estimated_residual.is_finite());
}

#[test]
fn legacy_mixed_precision_wrapper_maps_callback_panic_to_numerical_status() {
    struct PanickingPreconditioner;

    impl Preconditioner<f32> for PanickingPreconditioner {
        fn apply(&self, _r: &[f32]) -> Vec<f32> {
            panic!("injected low-precision preconditioner failure");
        }

        fn apply_into(&self, _r: &[f32], _y: &mut [f32]) {
            panic!("injected low-precision preconditioner failure");
        }
    }

    let result = mixed_precision_solve::<f64, f32, _>(
        &CsrMatrix::<f64>::identity(2),
        &[1.0, 2.0],
        Box::new(PanickingPreconditioner),
        &FgmresConfig::default(),
    );

    assert_eq!(result.status, SolverStatus::NumericalError);
    assert!(!result.converged);
    assert!(!result.residual.is_finite());
}

#[test]
fn checked_mixed_precision_wrapper_rejects_nonfinite_low_precision_cast() {
    struct SanitizingPreconditioner;

    impl Preconditioner<f32> for SanitizingPreconditioner {
        fn apply_into(&self, _r: &[f32], y: &mut [f32]) {
            // Deliberately hide a non-finite low-precision factor output. The
            // wrapper must report the failure rather than treating this as a
            // valid zero correction and returning a breakdown/max-iteration
            // result.
            y.fill(f32::NAN);
        }
    }

    let error = try_mixed_precision_solve::<f64, f32, _>(
        &CsrMatrix::<f64>::identity(1),
        &[f64::MAX],
        Box::new(SanitizingPreconditioner),
        &FgmresConfig {
            max_iterations: 4,
            restart: 2,
            ..FgmresConfig::default()
        },
    )
    .expect_err("low-precision cast overflow must be a typed failure");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "mixed-precision preconditioner",
            ..
        }
    ));
}
