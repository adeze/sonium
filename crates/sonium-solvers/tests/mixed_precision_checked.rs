use num_complex::{Complex32, Complex64};
use sonium_solvers::direct::SparseLuError;
use sonium_solvers::preconditioners::{DiagonalPreconditioner, IluConfig};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::SolverError;
use sonium_solvers::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers::{
    FgmresConfig, gmres_ir_ilu, gmres_ir_ilu_with_guess, gmres_ir_lu, gmres_ir_lu_with_guess,
    try_gmres_ir_ilu, try_gmres_ir_ilu_with_guess, try_gmres_ir_lu, try_gmres_ir_lu_with_guess,
    try_mixed_precision_solve_with_guess,
};

#[test]
fn checked_gmres_ir_ilu_rejects_malformed_high_precision_csr() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    matrix.row_ptrs[2] = 1;

    let error = try_gmres_ir_ilu::<f64, f32>(
        &matrix,
        &[1.0, 2.0],
        &IluConfig::default(),
        &FgmresConfig::default(),
    )
    .expect_err("checked GMRES-IR must reject malformed CSR before casting");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn checked_gmres_ir_lu_rejects_nonfinite_rhs_before_factorization() {
    let matrix = CsrMatrix::<f64>::identity(2);
    let error = try_gmres_ir_lu::<f64, f32>(&matrix, &[f64::NAN, 1.0], &FgmresConfig::default())
        .expect_err("checked GMRES-IR LU must reject a non-finite RHS");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn checked_gmres_ir_lu_reports_original_operator_residual() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0_f64, 1.0, 1.0, 3.0],
    );
    let result = try_gmres_ir_lu::<f64, f32>(
        &matrix,
        &[6.0, 7.0],
        &FgmresConfig {
            tolerance: 1.0e-10,
            ..FgmresConfig::default()
        },
    )
    .expect("checked GMRES-IR LU should solve a well-pivoted system");
    assert!(result.converged);
    assert!(result.residual <= 1.0e-9, "residual={}", result.residual);
    assert!(result.estimated_residual.is_finite());
    assert!(result.backward_error.is_finite());
    assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
}

#[test]
fn gmres_ir_lu_preserves_projected_estimate_on_max_iterations() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0000001_f64, 0.25, 0.3333333, 2.0000003],
    );
    let config = FgmresConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1.0e-20,
        exact_residual_check_interval: 0,
        ..FgmresConfig::default()
    };

    let checked = try_gmres_ir_lu::<f64, f32>(&matrix, &[1.0, 2.0], &config)
        .expect("checked GMRES-IR LU should return its max-iteration result");
    let legacy = gmres_ir_lu::<f64, f32>(&matrix, &[1.0, 2.0], &config)
        .expect("legacy GMRES-IR LU should return its max-iteration result");

    for result in [checked, legacy] {
        assert!(!result.converged);
        assert_eq!(result.status, SolverStatus::MaxIterationsReached);
        assert!(result.residual.is_finite());
        assert!(result.estimated_residual.is_finite());
        assert_ne!(
            result.residual.to_bits(),
            result.estimated_residual.to_bits(),
            "max-iteration result must retain the projected estimate separately"
        );
    }
}

#[test]
fn gmres_ir_ilu_preserves_projected_estimate_on_max_iterations() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0000001_f64, 0.25, 0.3333333, 2.0000003],
    );
    let config = FgmresConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1.0e-20,
        exact_residual_check_interval: 0,
        ..FgmresConfig::default()
    };

    let result = try_gmres_ir_ilu::<f64, f32>(&matrix, &[1.0, 2.0], &IluConfig::default(), &config)
        .expect("checked GMRES-IR ILU return max-iteration result");

    assert!(!result.converged);
    assert_eq!(result.status, SolverStatus::MaxIterationsReached);
    assert!(result.residual.is_finite());
    assert!(result.estimated_residual.is_finite());
    assert_ne!(
        result.residual.to_bits(),
        result.estimated_residual.to_bits(),
        "max-iteration result should retain projected estimate separately"
    );
}

#[test]
fn legacy_gmres_ir_ilu_maps_setup_failure_to_numerical_status() {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    matrix.row_ptrs[2] = 1;

    let result = gmres_ir_ilu::<f64, f32>(
        &matrix,
        &[1.0, 2.0],
        &IluConfig::default(),
        &FgmresConfig::default(),
    );
    assert!(!result.converged);
    assert_eq!(result.status, SolverStatus::NumericalError);
    assert!(!result.residual.is_finite());
}

#[test]
fn checked_gmres_ir_lu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);

    let error = try_gmres_ir_lu::<f64, f32>(&matrix, &[1.0], &FgmresConfig::default())
        .expect_err("finite f64 values overflowing f32 must fail at the cast boundary");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES-IR LU cast",
            ..
        }
    ));
}

#[test]
fn checked_gmres_ir_ilu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);

    let error = try_gmres_ir_ilu::<f64, f32>(
        &matrix,
        &[1.0],
        &IluConfig::default(),
        &FgmresConfig::default(),
    )
    .expect_err("finite f64 values overflowing f32 must fail at the cast boundary");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES-IR ILU cast",
            ..
        }
    ));
}

#[test]
fn legacy_gmres_ir_lu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);

    let error = gmres_ir_lu::<f64, f32>(&matrix, &[1.0], &FgmresConfig::default())
        .expect_err("legacy GMRES-IR LU must reject a cast overflow");
    assert!(matches!(error, SparseLuError::IterativeSolve(reason) if reason.contains("cast")));
}

#[test]
fn legacy_gmres_ir_ilu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);

    let result = gmres_ir_ilu::<f64, f32>(
        &matrix,
        &[1.0],
        &IluConfig::default(),
        &FgmresConfig::default(),
    );
    assert_eq!(result.status, SolverStatus::NumericalError);
    assert!(!result.converged);
    assert!(!result.residual.is_finite());
}

#[test]
fn legacy_complex_gmres_ir_lu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(
        1,
        1,
        vec![0, 1],
        vec![0],
        vec![Complex64::new(f64::MAX, 0.0)],
    );

    let error = gmres_ir_lu::<Complex64, Complex32>(
        &matrix,
        &[Complex64::new(1.0, 0.0)],
        &FgmresConfig::default(),
    )
    .expect_err("legacy complex GMRES-IR LU must reject a cast overflow");
    assert!(matches!(error, SparseLuError::IterativeSolve(reason) if reason.contains("cast")));
}

#[test]
fn legacy_complex_gmres_ir_ilu_rejects_nonfinite_low_precision_cast() {
    let matrix = CsrMatrix::from_raw_parts(
        1,
        1,
        vec![0, 1],
        vec![0],
        vec![Complex64::new(f64::MAX, 0.0)],
    );

    let result = gmres_ir_ilu::<Complex64, Complex32>(
        &matrix,
        &[Complex64::new(1.0, 0.0)],
        &IluConfig::default(),
        &FgmresConfig::default(),
    );
    assert_eq!(result.status, SolverStatus::NumericalError);
    assert!(!result.converged);
    assert!(!result.residual.is_finite());
}

#[test]
fn legacy_gmres_ir_ilu_rejects_invalid_pivot_tolerance() {
    let matrix = CsrMatrix::<f64>::identity(2);
    let config = IluConfig {
        pivot_tolerance: f64::NAN,
        ..IluConfig::default()
    };

    let result = gmres_ir_ilu::<f64, f32>(&matrix, &[1.0, 2.0], &config, &FgmresConfig::default());

    assert_eq!(result.status, SolverStatus::NumericalError);
    assert!(!result.converged);
    assert!(!result.residual.is_finite());
}

#[test]
fn gmres_ir_continuation_entry_points_normalize_residual_metadata() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0_f64, -1.0, -2.0, 3.0],
    );
    let b = [3.0, 1.0];
    let initial_guess = [0.5, 0.25];
    let config = FgmresConfig {
        max_iterations: 20,
        restart: 2,
        tolerance: 1.0e-12,
        exact_residual_check_interval: 1,
        ..FgmresConfig::default()
    };
    let ilu_config = IluConfig::default();

    let checked_lu =
        try_gmres_ir_lu_with_guess::<f64, f32>(&matrix, &b, Some(&initial_guess), &config)
            .expect("checked GMRES-IR LU continuation");
    let checked_ilu = try_gmres_ir_ilu_with_guess::<f64, f32>(
        &matrix,
        &b,
        Some(&initial_guess),
        &ilu_config,
        &config,
    )
    .expect("checked GMRES-IR ILU continuation");
    let legacy_lu = gmres_ir_lu_with_guess::<f64, f32>(&matrix, &b, Some(&initial_guess), &config)
        .expect("legacy GMRES-IR LU continuation");
    let legacy_ilu = gmres_ir_ilu_with_guess::<f64, f32>(
        &matrix,
        &b,
        Some(&initial_guess),
        &ilu_config,
        &config,
    );

    for result in [&checked_lu, &checked_ilu, &legacy_lu, &legacy_ilu] {
        assert!(result.residual.is_finite());
        assert!(result.estimated_residual.is_finite());
        assert!(result.backward_error.is_finite());
        assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
        assert!(result.converged);

        let ax = matrix.matvec(&result.x);
        let residual_norm = b
            .iter()
            .zip(ax.iter())
            .map(|(rhs, applied)| (rhs - applied).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((result.residual - residual_norm / b_norm).abs() < 1.0e-12);
    }
}

#[test]
fn mixed_precision_continuation_rejects_initial_guess_before_setup() {
    let singular = CsrMatrix::from_raw_parts(2, 2, vec![0, 0, 0], vec![], vec![]);
    let rhs = [1.0, 2.0];
    let config = FgmresConfig::default();
    let guess = [0.0];

    for error in [
        try_gmres_ir_lu_with_guess::<f64, f32>(&singular, &rhs, Some(&guess), &config)
            .expect_err("LU must reject a malformed continuation before factorization"),
        try_gmres_ir_ilu_with_guess::<f64, f32>(
            &singular,
            &rhs,
            Some(&guess),
            &IluConfig::default(),
            &config,
        )
        .expect_err("ILU must reject a malformed continuation before factorization"),
    ] {
        assert!(matches!(
            error,
            SolverError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }
}

#[test]
fn mixed_precision_continuation_rejects_nonfinite_initial_guess() {
    let matrix = CsrMatrix::<f64>::identity(2);
    let guess = [f64::NAN, 0.0];
    let error = try_mixed_precision_solve_with_guess::<f64, f32, _>(
        &matrix,
        &[1.0, 2.0],
        Some(&guess),
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &FgmresConfig::default(),
    )
    .expect_err("mixed-precision FGMRES must reject a non-finite continuation");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "mixed-precision FGMRES",
            ref reason,
        } if reason.contains("initial guess")
    ));
}
