//! Deterministic metamorphic contracts for checked iterative solver paths.

use num_complex::Complex64;
use num_traits::Float;
use sonium_qa::QaResult;
use sonium_solvers::preconditioners::DiagonalPreconditioner;
#[allow(deprecated)]
use sonium_solvers::{
    AccuracyStatus, AdjointSolve, AndersonConfig, AndersonError, BiCgstabConfig, BlockGmresConfig,
    CgConfig, CgsConfig, CsrBuilder, CsrMatrix, DifferentiableLinearSolve, FgmresConfig,
    FgmresSolution, GmresConfig, GmresDrConfig, IdrsConfig, IdrsStabConfig, IluConfig,
    IluPreconditioner, LinearOperator, LinearSolveConfig, LinearSolverType, NewtonConfig,
    NewtonError, OptimError, Preconditioner, RecycledSubspace, SolverError, SolverFailure,
    SolverStatus, SpmvPlan, SpmvPlanConfig, SpmvPlanPreference, anderson_acceleration,
    block_gmres_preconditioned_with_guess, dense::DenseMatrix, gmres, newton_raphson, solve_real,
    try_bicgstab, try_bicgstab_pipelined, try_block_gmres, try_cg, try_cgs, try_fgmres, try_gmres,
    try_gmres_dr_preconditioned, try_gmres_ir_complex32, try_gmres_ir_f32, try_gmres_ir_ilu,
    try_gmres_ir_ilu_with_guess, try_gmres_ir_lu_with_guess, try_gmres_recycled, try_idrs,
    try_idrs_stab, try_mixed_precision_solve_with_guess,
};

fn shared_failure_contract_ok<E: SolverFailure>(
    error: &E,
    expected_status: SolverStatus,
    expected_accuracy: AccuracyStatus,
) -> bool {
    error.status() == expected_status && error.accuracy_status() == expected_accuracy
}

fn matrix(scale: f64, permuted: bool) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(2, 2);
    let rows = if permuted {
        [[3.0, 1.0], [1.0, 4.0]]
    } else {
        [[4.0, 1.0], [1.0, 3.0]]
    };
    for values in rows {
        builder.add_row_entries([(0, scale * values[0]), (1, scale * values[1])].into_iter());
    }
    builder.finish()
}

fn solve(matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Result<Vec<f64>, String> {
    let result = gmres(
        matrix,
        rhs,
        &GmresConfig {
            max_iterations: 16,
            restart: 4,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
    );
    if !result.converged || result.residual > 1e-10 {
        return Err(format!("GMRES did not converge: {result:?}"));
    }
    Ok(result.x)
}

fn checked_facade_result_contracts() -> Result<f64, String> {
    let mut checks = 0.0;
    let identity = CsrMatrix::identity(2);
    let linear = solve_real(
        &identity,
        &[1.0, -2.0],
        &LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .map_err(|error| format!("linear facade solve failed: {error}"))?;
    if linear.status != SolverStatus::Converged
        || linear.attainable_accuracy != AccuracyStatus::Attained
        || !linear.residual.is_finite()
        || !linear.backward_error.is_finite()
    {
        return Err(format!(
            "linear facade metadata invalid: status={:?}, accuracy={:?}, residual={}, backward_error={}",
            linear.status, linear.attainable_accuracy, linear.residual, linear.backward_error
        ));
    }
    checks += 1.0;

    let anderson = anderson_acceleration(
        &[0.0_f64],
        |x| vec![0.5 * x[0] + 1.0],
        &AndersonConfig {
            absolute_tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .map_err(|error| format!("Anderson facade solve failed: {error}"))?;
    if anderson.status != SolverStatus::Converged
        || anderson.attainable_accuracy != AccuracyStatus::Attained
        || !anderson.residual.is_finite()
    {
        return Err(format!("Anderson metadata invalid: {anderson:?}"));
    }
    checks += 1.0;

    let newton = newton_raphson(
        &[1.0_f64],
        |x| vec![x[0] * x[0] - 4.0],
        |x, rhs| Ok(vec![rhs[0] / (2.0 * x[0])]),
        &NewtonConfig {
            absolute_tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .map_err(|error| format!("Newton facade solve failed: {error}"))?;
    if newton.status != SolverStatus::Converged
        || newton.attainable_accuracy != AccuracyStatus::Attained
        || !newton.residual.is_finite()
    {
        return Err(format!("Newton metadata invalid: {newton:?}"));
    }
    checks += 1.0;

    let anderson_error = AndersonError::NotConverged(2, 1.0);
    if !shared_failure_contract_ok(
        &anderson_error,
        SolverStatus::MaxIterationsReached,
        AccuracyStatus::NotAttained,
    ) {
        return Err("Anderson failure metadata mapping is invalid".into());
    }
    checks += 1.0;

    let newton_error = NewtonError::InvalidConfig("invalid QA configuration".into());
    if !shared_failure_contract_ok(
        &newton_error,
        SolverStatus::NumericalError,
        AccuracyStatus::NotChecked,
    ) {
        return Err("Newton failure metadata mapping is invalid".into());
    }
    checks += 1.0;

    let optimization_error = OptimError::Infeasible;
    if !shared_failure_contract_ok(
        &optimization_error,
        SolverStatus::Stagnated,
        AccuracyStatus::NotAttained,
    ) {
        return Err("optimization failure metadata mapping invalid".into());
    }
    checks += 1.0;

    Ok(checks)
}

fn complex_matrix() -> CsrMatrix<Complex64> {
    let mut builder = CsrBuilder::new(2, 2);
    builder.add_row_entries(
        [
            (0, Complex64::new(2.0, 1.0)),
            (1, Complex64::new(1.0, -0.5)),
        ]
        .into_iter(),
    );
    builder.add_row_entries(
        [
            (0, Complex64::new(0.25, 0.75)),
            (1, Complex64::new(3.0, -1.0)),
        ]
        .into_iter(),
    );
    builder.finish()
}

fn hermitian_inner(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .fold(Complex64::new(0.0, 0.0), |sum, (lhs, rhs)| {
            sum + lhs.conj() * rhs
        })
}

fn transpose_inner(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .fold(Complex64::new(0.0, 0.0), |sum, (lhs, rhs)| sum + lhs * rhs)
}

fn relative_residual(
    matrix: &CsrMatrix<Complex64>,
    x: &[Complex64],
    rhs: &[Complex64],
) -> Result<f64, String> {
    let applied = matrix
        .try_matvec(x)
        .map_err(|error| format!("forward SpMV failed: {error}"))?;
    let residual = applied
        .iter()
        .zip(rhs)
        .map(|(applied, rhs)| (applied - rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    Ok(residual / rhs_norm.max(f64::MIN_POSITIVE))
}

fn adjoint_residual(
    matrix: &CsrMatrix<Complex64>,
    x: &[Complex64],
    rhs: &[Complex64],
    transpose: bool,
) -> Result<f64, String> {
    let applied = if transpose {
        matrix
            .try_matvec_transpose(x)
            .map_err(|error| format!("transpose SpMV failed: {error}"))?
    } else {
        matrix
            .try_matvec_hermitian(x)
            .map_err(|error| format!("Hermitian SpMV failed: {error}"))?
    };
    let residual = applied
        .iter()
        .zip(rhs)
        .map(|(applied, rhs)| (applied - rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    Ok(residual / rhs_norm.max(f64::MIN_POSITIVE))
}

fn checked_complex_adjoint_contracts() -> Result<(f64, f64, f64, f64), String> {
    let matrix = complex_matrix();
    let u = [Complex64::new(0.75, -0.25), Complex64::new(-0.5, 0.4)];
    let v = [Complex64::new(-0.2, 0.6), Complex64::new(0.9, -0.1)];
    let av = matrix
        .try_matvec(&v)
        .map_err(|error| format!("operator application failed: {error}"))?;
    let at_u = matrix
        .try_matvec_transpose(&u)
        .map_err(|error| format!("operator transpose failed: {error}"))?;
    let ah_u = matrix
        .try_matvec_hermitian(&u)
        .map_err(|error| format!("operator Hermitian transpose failed: {error}"))?;
    let transpose_error = (transpose_inner(&u, &av) - transpose_inner(&at_u, &v)).norm();
    let hermitian_error = (hermitian_inner(&u, &av) - hermitian_inner(&ah_u, &v)).norm();

    let preconditioner = IluPreconditioner::from_csr(&matrix);
    let p_v = preconditioner
        .try_apply(&v)
        .map_err(|error| format!("preconditioner application failed: {error}"))?;
    let pt_u = preconditioner
        .try_apply_transpose(&u)
        .map_err(|error| format!("preconditioner transpose failed: {error}"))?;
    let ph_u = preconditioner
        .try_apply_hermitian(&u)
        .map_err(|error| format!("preconditioner Hermitian transpose failed: {error}"))?;
    let preconditioner_transpose_error =
        (transpose_inner(&u, &p_v) - transpose_inner(&pt_u, &v)).norm();
    let preconditioner_hermitian_error =
        (hermitian_inner(&u, &p_v) - hermitian_inner(&ph_u, &v)).norm();
    if p_v
        .iter()
        .chain(pt_u.iter())
        .chain(ph_u.iter())
        .any(|value| !value.norm().is_finite())
    {
        return Err("preconditioner adjoint produced a non-finite value".into());
    }

    let mut config = GmresConfig::default();
    config.max_iterations = 64;
    config.restart = 8;
    config.tolerance = 1.0e-12;
    config.convergence_check_interval = 1;
    config.exact_residual_check_interval = 1;
    let solve = DifferentiableLinearSolve::new(&matrix, &preconditioner, config)
        .map_err(|error| format!("differentiable solve setup failed: {error}"))?;
    let rhs = [Complex64::new(1.2, -0.4), Complex64::new(-0.3, 0.8)];
    let forward = solve
        .solve_forward(&rhs)
        .map_err(|error| format!("forward solve failed: {error}"))?;
    let transpose = solve
        .solve_adjoint(&rhs, AdjointSolve::Transpose)
        .map_err(|error| format!("transpose solve failed: {error}"))?;
    let hermitian = solve
        .solve_adjoint(&rhs, AdjointSolve::Hermitian)
        .map_err(|error| format!("Hermitian solve failed: {error}"))?;
    if !forward.converged || !transpose.converged || !hermitian.converged {
        return Err(format!(
            "adjoint solve did not converge: forward={:?}, transpose={:?}, Hermitian={:?}",
            forward.status, transpose.status, hermitian.status
        ));
    }
    let forward_residual = relative_residual(&matrix, &forward.x, &rhs)?;
    let transpose_residual = adjoint_residual(&matrix, &transpose.x, &rhs, true)?;
    let hermitian_residual = adjoint_residual(&matrix, &hermitian.x, &rhs, false)?;
    let invalid_rhs = solve.solve_adjoint(&rhs[..1], AdjointSolve::Hermitian);
    if invalid_rhs.is_ok() {
        return Err("adjoint dimension mismatch was not rejected".into());
    }
    println!(
        "adjoint=transpose,hermitian operator_error=({transpose_error:.3e},{hermitian_error:.3e}) preconditioner_error=({preconditioner_transpose_error:.3e},{preconditioner_hermitian_error:.3e}) residual=({forward_residual:.3e},{transpose_residual:.3e},{hermitian_residual:.3e})"
    );
    Ok((
        transpose_error
            .max(hermitian_error)
            .max(preconditioner_transpose_error)
            .max(preconditioner_hermitian_error),
        forward_residual,
        transpose_residual.max(hermitian_residual),
        1.0,
    ))
}

fn checked_spmv_plan_contracts() -> Result<f64, String> {
    let triplets = (0..8)
        .flat_map(|row| [(row, row, 3.0 + row as f64), (row, (row + 1) % 8, -0.25)])
        .collect::<Vec<_>>();
    let matrix = CsrMatrix::from_triplets(8, 8, triplets);
    let input = (0..8)
        .map(|index| 1.0 + index as f64 * 0.125)
        .collect::<Vec<_>>();
    let preferences = [
        SpmvPlanPreference::Csr,
        SpmvPlanPreference::CompactCsrU32,
        SpmvPlanPreference::NumaCsr,
        SpmvPlanPreference::RowBucketedCsr,
        SpmvPlanPreference::SellCSigma,
        SpmvPlanPreference::Bsr2,
        SpmvPlanPreference::Bsr3,
        SpmvPlanPreference::Bsr4,
    ];

    for preference in preferences {
        let plan = SpmvPlan::try_from_csr(
            &matrix,
            SpmvPlanConfig {
                preference,
                numa_partitions: Some(2),
                sell_c_sigma_chunk_size: 4,
                sell_c_sigma_window: 8,
                ..Default::default()
            },
        )
        .map_err(|error| format!("{preference:?} plan construction failed: {error}"))?;
        plan.try_apply(&input)
            .map_err(|error| format!("{preference:?} checked forward failed: {error}"))?;
        plan.try_apply_transpose(&input)
            .map_err(|error| format!("{preference:?} checked transpose failed: {error}"))?;
        plan.try_apply_hermitian(&input)
            .map_err(|error| format!("{preference:?} checked Hermitian failed: {error}"))?;
        if plan.try_apply(&input[..7]).is_ok() {
            return Err(format!(
                "{preference:?} accepted a dimension-mismatched input"
            ));
        }
        if plan.try_apply(&[f64::NAN; 8]).is_ok() {
            return Err(format!("{preference:?} accepted a non-finite input"));
        }
        if plan.try_apply_transpose(&input[..7]).is_ok() {
            return Err(format!(
                "{preference:?} accepted a transpose dimension mismatch"
            ));
        }
        if plan.try_apply_hermitian(&[f64::INFINITY; 8]).is_ok() {
            return Err(format!(
                "{preference:?} accepted a Hermitian non-finite input"
            ));
        }
    }
    Ok(preferences.len() as f64)
}

#[derive(Debug, Default)]
struct CheckedFamilyMetrics {
    converged: f64,
    dimension_rejections: f64,
    non_finite_rejections: f64,
    panic_rejections: f64,
    breakdown_status_checks: f64,
}

#[derive(Debug, Default)]
struct CheckedBlockFamilyMetrics {
    block_converged: f64,
    block_dimension_rejections: f64,
    block_non_finite_rejections: f64,
    block_panic_rejections: f64,
    block_status_checks: f64,
    block_column_checks: f64,
    block_warm_start_checks: f64,
    gmres_dr_converged: f64,
    gmres_dr_failure_rejections: f64,
    gmres_dr_status_checks: f64,
    recycled_converged: f64,
    recycled_failure_rejections: f64,
    recycled_status_checks: f64,
}

#[derive(Debug, Default)]
struct CheckedMixedPrecisionMetrics {
    converged: f64,
    metadata_checks: f64,
    dimension_rejections: f64,
    non_finite_rejections: f64,
    cast_rejections: f64,
    initial_guess_rejections: f64,
}

fn expect_dimension_rejection<T>(
    operation: &str,
    result: Result<T, SolverError>,
) -> Result<f64, String> {
    match result {
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        }) => Ok(1.0),
        Err(error) => Err(format!(
            "{operation} family dimension fixture returned the wrong error: {error}"
        )),
        Ok(_) => Err(format!(
            "{operation} family dimension fixture unexpectedly succeeded"
        )),
    }
}

fn expect_rhs_rejection<T>(operation: &str, result: Result<T, SolverError>) -> Result<f64, String> {
    match result {
        Err(SolverError::OperationFailed {
            operation: actual,
            reason,
        }) if actual == operation && reason.contains("right-hand side") => Ok(1.0),
        Err(error) => Err(format!(
            "{operation} family non-finite fixture returned the wrong error: {error}"
        )),
        Ok(_) => Err(format!(
            "{operation} family non-finite fixture unexpectedly succeeded"
        )),
    }
}

fn expect_panic_rejection<T>(
    operation: &str,
    result: Result<T, SolverError>,
) -> Result<f64, String> {
    match result {
        Err(SolverError::OperationFailed {
            operation: actual,
            reason,
        }) if actual == operation && !reason.is_empty() => Ok(1.0),
        Err(error) => Err(format!(
            "{operation} family panic fixture returned the wrong error: {error}"
        )),
        Ok(_) => Err(format!(
            "{operation} family panic fixture unexpectedly succeeded"
        )),
    }
}

fn expect_cast_rejection<T>(
    operation: &str,
    result: Result<T, SolverError>,
) -> Result<f64, String> {
    match result {
        Err(SolverError::OperationFailed {
            operation: actual,
            reason,
        }) if actual == operation && !reason.is_empty() => Ok(1.0),
        Err(error) => Err(format!(
            "{operation} mixed-precision cast fixture returned wrong error: {error}"
        )),
        Ok(_) => Err(format!(
            "{operation} mixed-precision cast fixture unexpectedly succeeded"
        )),
    }
}

fn expect_initial_guess_rejection<T>(
    operation: &str,
    result: Result<T, SolverError>,
) -> Result<f64, String> {
    match result {
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        }) => Ok(1.0),
        Err(SolverError::OperationFailed {
            operation: actual,
            reason,
        }) if actual == operation && reason.contains("initial guess") => Ok(1.0),
        Err(error) => Err(format!(
            "{operation} initial-guess fixture returned the wrong error: {error}"
        )),
        Ok(_) => Err(format!(
            "{operation} initial-guess fixture unexpectedly succeeded"
        )),
    }
}

fn validate_mixed_solution<T: sonium_solvers::ComplexField>(
    label: &str,
    result: Result<FgmresSolution<T>, SolverError>,
    metrics: &mut CheckedMixedPrecisionMetrics,
) -> Result<(), String> {
    let solution = result
        .map_err(|error| format!("{label} convergence fixture returned an error: {error}"))?;
    if !solution.converged
        || solution.status != SolverStatus::Converged
        || !solution.residual.is_finite()
        || !solution.estimated_residual.is_finite()
        || !solution.backward_error.is_finite()
    {
        return Err(format!(
            "{label} returned invalid mixed-precision metadata: {solution:?}"
        ));
    }
    metrics.converged += 1.0;
    metrics.metadata_checks += 1.0;
    Ok(())
}

fn checked_mixed_precision_contracts() -> Result<CheckedMixedPrecisionMetrics, String> {
    let matrix: CsrMatrix<f64> = CsrMatrix::identity(2);
    let rhs = [1.0, -2.0];
    let complex_matrix: CsrMatrix<Complex64> = CsrMatrix::identity(2);
    let complex_rhs = [Complex64::new(1.0, 0.25), Complex64::new(-2.0, 0.5)];
    let config = FgmresConfig {
        max_iterations: 16,
        restart: 4,
        tolerance: 1.0e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        ..FgmresConfig::default()
    };
    let ilu_config = IluConfig::default();
    let mut metrics = CheckedMixedPrecisionMetrics::default();

    validate_mixed_solution(
        "GMRES-IR f32 LU",
        try_gmres_ir_f32(&matrix, &rhs, &config),
        &mut metrics,
    )?;
    validate_mixed_solution(
        "GMRES-IR Complex32 LU",
        try_gmres_ir_complex32(&complex_matrix, &complex_rhs, &config),
        &mut metrics,
    )?;
    validate_mixed_solution(
        "GMRES-IR f32 ILU",
        try_gmres_ir_ilu::<f64, f32>(&matrix, &rhs, &ilu_config, &config),
        &mut metrics,
    )?;

    metrics.dimension_rejections =
        expect_dimension_rejection("GMRES-IR", try_gmres_ir_f32(&matrix, &[1.0], &config))?;
    metrics.non_finite_rejections = [
        expect_rhs_rejection(
            "GMRES-IR LU",
            try_gmres_ir_f32(&matrix, &[f64::NAN, -2.0], &config),
        )?,
        expect_rhs_rejection(
            "GMRES-IR LU",
            try_gmres_ir_complex32(
                &complex_matrix,
                &[Complex64::new(f64::NAN, 0.0), complex_rhs[1]],
                &config,
            ),
        )?,
    ]
    .into_iter()
    .sum();

    let overflowing_real_matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, f64::MAX)]);
    let overflowing_complex_matrix =
        CsrMatrix::from_triplets(1, 1, vec![(0, 0, Complex64::new(f64::MAX, 0.0))]);
    metrics.cast_rejections = [
        expect_cast_rejection(
            "GMRES-IR LU cast",
            try_gmres_ir_f32(&overflowing_real_matrix, &[1.0], &config),
        )?,
        expect_cast_rejection(
            "GMRES-IR LU cast",
            try_gmres_ir_complex32(
                &overflowing_complex_matrix,
                &[Complex64::new(1.0, 0.0)],
                &config,
            ),
        )?,
    ]
    .into_iter()
    .sum();

    let singular = CsrMatrix::from_raw_parts(2, 2, vec![0, 0, 0], vec![], vec![]);
    let malformed_guess = [0.0];
    metrics.initial_guess_rejections = [
        expect_initial_guess_rejection(
            "GMRES-IR LU",
            try_gmres_ir_lu_with_guess::<f64, f32>(
                &singular,
                &rhs,
                Some(&malformed_guess),
                &config,
            ),
        )?,
        expect_initial_guess_rejection(
            "GMRES-IR ILU",
            try_gmres_ir_ilu_with_guess::<f64, f32>(
                &singular,
                &rhs,
                Some(&malformed_guess),
                &ilu_config,
                &config,
            ),
        )?,
        expect_initial_guess_rejection(
            "mixed-precision FGMRES",
            try_mixed_precision_solve_with_guess::<f64, f32, _>(
                &singular,
                &rhs,
                Some(&malformed_guess),
                Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
                &config,
            ),
        )?,
        expect_initial_guess_rejection(
            "mixed-precision FGMRES",
            try_mixed_precision_solve_with_guess::<f64, f32, _>(
                &matrix,
                &rhs,
                Some(&[f64::NAN, 0.0]),
                Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
                &config,
            ),
        )?,
    ]
    .into_iter()
    .sum();

    Ok(metrics)
}

#[allow(deprecated)]
fn checked_block_family_contracts() -> Result<CheckedBlockFamilyMetrics, String> {
    let matrix: CsrMatrix<f64> = CsrMatrix::identity(2);
    let rhs = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 3.0, -2.0, 4.0]);
    let config = BlockGmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let mut metrics = CheckedBlockFamilyMetrics::default();

    let block_solution = try_block_gmres(&matrix, &rhs, &config)
        .map_err(|error| format!("block GMRES convergence fixture returned an error: {error}"))?;
    if !block_solution.converged
        || block_solution.x.shape() != (2, 2)
        || block_solution.residuals.len() != 2
        || block_solution.true_residuals.len() != 2
        || block_solution.backward_errors.len() != 2
        || block_solution.attainable_accuracy.len() != 2
        || block_solution
            .residuals
            .iter()
            .chain(block_solution.true_residuals.iter())
            .chain(block_solution.backward_errors.iter())
            .any(|value| !value.is_finite() || *value > 1.0e-10)
    {
        return Err(format!(
            "block GMRES convergence fixture returned invalid per-RHS metadata: {block_solution:?}"
        ));
    }
    metrics.block_converged = block_solution.true_residuals.len() as f64;

    let scalar_config = GmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 1,
    };
    for (column, expected) in [vec![1.0, -2.0], vec![3.0, 4.0]].into_iter().enumerate() {
        let scalar = try_gmres(&matrix, &expected, &scalar_config)
            .map_err(|error| format!("scalar GMRES column fixture returned an error: {error}"))?;
        let actual = block_solution.x.column_to_vec(column);
        if !scalar.converged
            || actual
                .iter()
                .zip(scalar.x.iter())
                .any(|(block, scalar)| (*block - *scalar).abs() > 1.0e-10)
        {
            return Err(format!(
                "block-vs-column GMRES mismatch in column {column}: block={actual:?}, scalar={:?}",
                scalar.x
            ));
        }
    }
    metrics.block_column_checks = 2.0;

    let identity = sonium_solvers::traits::IdentityPreconditioner;
    let warm = block_gmres_preconditioned_with_guess(
        &matrix,
        &rhs,
        &identity,
        Some(&block_solution.x),
        &config,
    );
    if !warm.converged
        || warm
            .true_residuals
            .iter()
            .zip(block_solution.true_residuals.iter())
            .any(|(warm, cold)| !warm.is_finite() || *warm > *cold + 1.0e-12)
    {
        return Err(format!(
            "warm-start block GMRES regressed: cold={:?}, warm={:?}",
            block_solution.true_residuals, warm.true_residuals
        ));
    }
    metrics.block_warm_start_checks = 2.0;

    let mismatched_rhs = DenseMatrix::from_row_major_vec(1, 2, vec![1.0, -2.0]);
    metrics.block_dimension_rejections = match try_block_gmres(&matrix, &mismatched_rhs, &config) {
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        }) => 1.0,
        Err(error) => {
            return Err(format!(
                "block GMRES dimension fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "block GMRES dimension fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };

    let non_finite_rhs = DenseMatrix::from_row_major_vec(2, 2, vec![f64::NAN, 3.0, -2.0, 4.0]);
    metrics.block_non_finite_rejections = expect_rhs_rejection(
        "block GMRES",
        try_block_gmres(&matrix, &non_finite_rhs, &config),
    )?;

    struct PanicOperator;
    impl LinearOperator<f64> for PanicOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA block/recycled operator panic");
        }

        fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA block/recycled transpose panic");
        }
    }
    let panic_rhs = DenseMatrix::from_row_major_vec(1, 1, vec![1.0]);
    metrics.block_panic_rejections = expect_panic_rejection(
        "block GMRES",
        try_block_gmres(&PanicOperator, &panic_rhs, &config),
    )?;

    let zero_matrix: CsrMatrix<f64> =
        CsrMatrix::from_triplets(2, 2, Vec::<(usize, usize, f64)>::new());
    let zero_solution = try_block_gmres(&zero_matrix, &rhs, &config)
        .map_err(|error| format!("block GMRES status fixture returned an error: {error}"))?;
    if zero_solution.converged
        || zero_solution.true_residuals.len() != 2
        || zero_solution
            .true_residuals
            .iter()
            .chain(zero_solution.backward_errors.iter())
            .any(|value| !value.is_finite())
    {
        return Err(format!(
            "block GMRES status fixture returned an invalid result: {zero_solution:?}"
        ));
    }
    metrics.block_status_checks = 1.0;

    let identity = sonium_solvers::traits::IdentityPreconditioner;
    let dr_config = GmresDrConfig {
        gmres: GmresConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
        num_deflation_vectors: 0,
    };
    let dr_solution =
        try_gmres_dr_preconditioned(&matrix, &identity, &[-2.0, 4.0], None, &dr_config)
            .map_err(|error| format!("GMRES-DR convergence fixture returned an error: {error}"))?;
    if !dr_solution.converged
        || dr_solution.status != SolverStatus::Converged
        || !dr_solution.residual.is_finite()
        || !dr_solution.backward_error.is_finite()
    {
        return Err(format!(
            "GMRES-DR convergence fixture returned invalid metadata: {dr_solution:?}"
        ));
    }
    metrics.gmres_dr_converged = 1.0;
    metrics.gmres_dr_failure_rejections = [
        expect_dimension_rejection(
            "GMRES-DR",
            try_gmres_dr_preconditioned(&matrix, &identity, &[1.0], None, &dr_config),
        )?,
        expect_rhs_rejection(
            "GMRES-DR",
            try_gmres_dr_preconditioned(&matrix, &identity, &[f64::NAN, 1.0], None, &dr_config),
        )?,
        expect_panic_rejection(
            "GMRES-DR",
            try_gmres_dr_preconditioned(&PanicOperator, &identity, &[1.0], None, &dr_config),
        )?,
    ]
    .into_iter()
    .sum();
    let dr_zero_solution =
        try_gmres_dr_preconditioned(&zero_matrix, &identity, &[-2.0, 4.0], None, &dr_config)
            .map_err(|error| format!("GMRES-DR status fixture returned an error: {error}"))?;
    if dr_zero_solution.converged
        || dr_zero_solution.status == SolverStatus::Converged
        || !dr_zero_solution.residual.is_finite()
    {
        return Err(format!(
            "GMRES-DR status fixture returned an invalid result: {dr_zero_solution:?}"
        ));
    }
    metrics.gmres_dr_status_checks = 1.0;

    let mut recycled_subspace = RecycledSubspace::new(1);
    let recycled_solution = try_gmres_recycled(
        &matrix,
        &[-2.0, 4.0],
        None,
        &mut recycled_subspace,
        &dr_config.gmres,
    )
    .map_err(|error| format!("recycled GMRES convergence fixture returned an error: {error}"))?;
    if !recycled_solution.converged
        || recycled_solution.status != SolverStatus::Converged
        || !recycled_solution.residual.is_finite()
        || !recycled_solution.backward_error.is_finite()
    {
        return Err(format!(
            "recycled GMRES convergence fixture returned invalid metadata: {recycled_solution:?}"
        ));
    }
    metrics.recycled_converged = 1.0;

    let mut recycled_dimension_subspace = RecycledSubspace::new(1);
    let mut recycled_non_finite_subspace = RecycledSubspace::new(1);
    let mut recycled_panic_subspace = RecycledSubspace::new(1);
    metrics.recycled_failure_rejections = [
        expect_dimension_rejection(
            "recycled GMRES",
            try_gmres_recycled(
                &matrix,
                &[1.0],
                None,
                &mut recycled_dimension_subspace,
                &dr_config.gmres,
            ),
        )?,
        expect_rhs_rejection(
            "recycled GMRES",
            try_gmres_recycled(
                &matrix,
                &[f64::NAN, 1.0],
                None,
                &mut recycled_non_finite_subspace,
                &dr_config.gmres,
            ),
        )?,
        expect_panic_rejection(
            "recycled GMRES",
            try_gmres_recycled(
                &PanicOperator,
                &[1.0],
                None,
                &mut recycled_panic_subspace,
                &dr_config.gmres,
            ),
        )?,
    ]
    .into_iter()
    .sum();
    let mut recycled_zero_subspace = RecycledSubspace::new(1);
    let recycled_zero_solution = try_gmres_recycled(
        &zero_matrix,
        &[-2.0, 4.0],
        None,
        &mut recycled_zero_subspace,
        &dr_config.gmres,
    )
    .map_err(|error| format!("recycled GMRES status fixture returned an error: {error}"))?;
    if recycled_zero_solution.converged
        || recycled_zero_solution.status == SolverStatus::Converged
        || !recycled_zero_solution.residual.is_finite()
    {
        return Err(format!(
            "recycled GMRES status fixture returned an invalid result: {recycled_zero_solution:?}"
        ));
    }
    metrics.recycled_status_checks = 1.0;

    Ok(metrics)
}

fn checked_solver_failure_contracts()
-> Result<(f64, f64, f64, f64, f64, f64, f64, CheckedFamilyMetrics), String> {
    let gmres_matrix = CsrMatrix::identity(2);
    let gmres_config = GmresConfig::default();
    let gmres_dimension_rejection = match try_gmres(&gmres_matrix, &[1.0], &gmres_config) {
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        }) => 1.0,
        Err(error) => {
            return Err(format!(
                "GMRES dimension fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "GMRES dimension fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };
    let gmres_non_finite_rejection = match try_gmres(&gmres_matrix, &[f64::NAN, 1.0], &gmres_config)
    {
        Err(SolverError::OperationFailed { operation, reason })
            if operation == "GMRES" && reason.contains("right-hand side") =>
        {
            1.0
        }
        Err(error) => {
            return Err(format!(
                "GMRES non-finite fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "GMRES non-finite fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };

    struct PanicOperator;
    impl LinearOperator<f64> for PanicOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA GMRES operator panic");
        }

        fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA GMRES transpose operator panic");
        }
    }
    let gmres_panic_rejection = match try_gmres(&PanicOperator, &[1.0], &gmres_config) {
        Err(SolverError::OperationFailed { operation, reason })
            if operation == "GMRES" && reason.contains("QA GMRES operator panic") =>
        {
            1.0
        }
        Err(error) => {
            return Err(format!(
                "GMRES panic fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "GMRES panic fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };
    let gmres_zero = CsrMatrix::from_triplets(2, 2, Vec::<(usize, usize, f64)>::new());
    let gmres_status = try_gmres(&gmres_zero, &[1.0, -2.0], &gmres_config)
        .map_err(|error| format!("GMRES status fixture returned an error: {error}"))?;
    if gmres_status.converged
        || gmres_status.status == SolverStatus::Converged
        || !gmres_status.residual.is_finite()
    {
        return Err(format!(
            "GMRES status fixture returned an invalid result: {gmres_status:?}"
        ));
    }

    let matrix = CsrMatrix::identity(2);
    let config = FgmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
    };

    let dimension_rejection = match try_fgmres(&matrix, &[1.0], &config) {
        Err(SolverError::DimensionMismatch {
            expected: 2,
            got: 1,
        }) => 1.0,
        Err(error) => {
            return Err(format!(
                "dimension fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "dimension fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };

    let non_finite_rejection = match try_fgmres(&matrix, &[f64::NAN, 1.0], &config) {
        Err(SolverError::OperationFailed { operation, reason })
            if operation == "FGMRES" && reason.contains("right-hand side") =>
        {
            1.0
        }
        Err(error) => {
            return Err(format!(
                "non-finite fixture returned the wrong error: {error}"
            ));
        }
        Ok(result) => {
            return Err(format!(
                "non-finite fixture unexpectedly succeeded: {result:?}"
            ));
        }
    };

    let zero_matrix = CsrMatrix::from_triplets(2, 2, Vec::<(usize, usize, f64)>::new());
    let breakdown = try_fgmres(&zero_matrix, &[1.0, -2.0], &config)
        .map_err(|error| format!("breakdown fixture returned an error: {error}"))?;
    if breakdown.converged
        || breakdown.status != SolverStatus::Breakdown
        || !breakdown.residual.is_finite()
    {
        return Err(format!(
            "breakdown fixture returned an invalid result: {breakdown:?}"
        ));
    }

    let family_matrix: CsrMatrix<f64> = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let family_rhs = [1.0, -2.0];
    let bicgstab_config = BiCgstabConfig {
        max_iterations: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
    };
    let cg_config = CgConfig {
        max_iterations: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
    };
    let cgs_config = CgsConfig {
        max_iterations: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
    };
    let idrs_config = IdrsConfig {
        s: 2,
        max_iterations: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
    };
    let idrs_stab_config = IdrsStabConfig {
        s: 2,
        ell: 1,
        max_iterations: 16,
        tolerance: 1.0e-12,
        print_interval: 0,
    };

    let family_solutions = [
        (
            "BiCGSTAB",
            try_bicgstab(&family_matrix, &family_rhs, &bicgstab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "pipelined BiCGSTAB",
            try_bicgstab_pipelined(&family_matrix, &family_rhs, &bicgstab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "CG",
            try_cg(&family_matrix, &family_rhs, &cg_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "CGS",
            try_cgs(&family_matrix, &family_rhs, &cgs_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "IDR(s)",
            try_idrs(&family_matrix, &family_rhs, &idrs_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "IDR(s)stab",
            try_idrs_stab(&family_matrix, &family_rhs, &idrs_stab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
    ];
    let mut family_metrics = CheckedFamilyMetrics::default();
    for (operation, result) in family_solutions {
        let (status, converged, residual) = result.map_err(|error| {
            format!("{operation} checked convergence fixture returned an error: {error}")
        })?;
        if !converged || status != SolverStatus::Converged || !residual.is_finite() {
            return Err(format!(
                "{operation} checked convergence fixture returned status={status:?}, converged={converged}, residual={residual}"
            ));
        }
        family_metrics.converged += 1.0;
    }

    family_metrics.dimension_rejections = [
        expect_dimension_rejection(
            "BiCGSTAB",
            try_bicgstab(&family_matrix, &[1.0], &bicgstab_config),
        )?,
        expect_dimension_rejection(
            "pipelined BiCGSTAB",
            try_bicgstab_pipelined(&family_matrix, &[1.0], &bicgstab_config),
        )?,
        expect_dimension_rejection("CG", try_cg(&family_matrix, &[1.0], &cg_config))?,
        expect_dimension_rejection("CGS", try_cgs(&family_matrix, &[1.0], &cgs_config))?,
        expect_dimension_rejection("IDR(s)", try_idrs(&family_matrix, &[1.0], &idrs_config))?,
        expect_dimension_rejection(
            "IDR(s)stab",
            try_idrs_stab(&family_matrix, &[1.0], &idrs_stab_config),
        )?,
    ]
    .into_iter()
    .sum();
    family_metrics.non_finite_rejections = [
        expect_rhs_rejection(
            "BiCGSTAB",
            try_bicgstab(&family_matrix, &[f64::NAN, 1.0], &bicgstab_config),
        )?,
        expect_rhs_rejection(
            "pipelined BiCGSTAB",
            try_bicgstab_pipelined(&family_matrix, &[f64::NAN, 1.0], &bicgstab_config),
        )?,
        expect_rhs_rejection("CG", try_cg(&family_matrix, &[f64::NAN, 1.0], &cg_config))?,
        expect_rhs_rejection(
            "CGS",
            try_cgs(&family_matrix, &[f64::NAN, 1.0], &cgs_config),
        )?,
        expect_rhs_rejection(
            "IDR(s)",
            try_idrs(&family_matrix, &[f64::NAN, 1.0], &idrs_config),
        )?,
        expect_rhs_rejection(
            "IDR(s)stab",
            try_idrs_stab(&family_matrix, &[f64::NAN, 1.0], &idrs_stab_config),
        )?,
    ]
    .into_iter()
    .sum();

    struct FamilyPanicOperator;
    impl LinearOperator<f64> for FamilyPanicOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA iterative family operator panic");
        }

        fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("QA iterative family transpose panic");
        }
    }
    let panic_operator = FamilyPanicOperator;
    family_metrics.panic_rejections = [
        expect_panic_rejection(
            "BiCGSTAB",
            try_bicgstab(&panic_operator, &[1.0], &bicgstab_config),
        )?,
        expect_panic_rejection(
            "pipelined BiCGSTAB",
            try_bicgstab_pipelined(&panic_operator, &[1.0], &bicgstab_config),
        )?,
        expect_panic_rejection("CG", try_cg(&panic_operator, &[1.0], &cg_config))?,
        expect_panic_rejection("CGS", try_cgs(&panic_operator, &[1.0], &cgs_config))?,
        expect_panic_rejection("IDR(s)", try_idrs(&panic_operator, &[1.0], &idrs_config))?,
        expect_panic_rejection(
            "IDR(s)stab",
            try_idrs_stab(&panic_operator, &[1.0], &idrs_stab_config),
        )?,
    ]
    .into_iter()
    .sum();

    let zero_family_matrix: CsrMatrix<f64> =
        CsrMatrix::from_triplets(2, 2, Vec::<(usize, usize, f64)>::new());
    let family_breakdowns = [
        (
            "BiCGSTAB",
            try_bicgstab(&zero_family_matrix, &family_rhs, &bicgstab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "pipelined BiCGSTAB",
            try_bicgstab_pipelined(&zero_family_matrix, &family_rhs, &bicgstab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "CG",
            try_cg(&zero_family_matrix, &family_rhs, &cg_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "CGS",
            try_cgs(&zero_family_matrix, &family_rhs, &cgs_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "IDR(s)",
            try_idrs(&zero_family_matrix, &family_rhs, &idrs_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
        (
            "IDR(s)stab",
            try_idrs_stab(&zero_family_matrix, &family_rhs, &idrs_stab_config)
                .map(|solution| (solution.status, solution.converged, solution.residual)),
        ),
    ];
    for (operation, result) in family_breakdowns {
        let (status, converged, residual) = result.map_err(|error| {
            format!("{operation} checked breakdown fixture returned an error: {error}")
        })?;
        if status != SolverStatus::Breakdown || converged || !residual.is_finite() {
            return Err(format!(
                "{operation} checked breakdown fixture returned status={status:?}, converged={converged}, residual={residual}"
            ));
        }
        family_metrics.breakdown_status_checks += 1.0;
    }

    Ok((
        gmres_dimension_rejection,
        gmres_non_finite_rejection,
        gmres_panic_rejection,
        1.0,
        dimension_rejection,
        non_finite_rejection,
        1.0,
        family_metrics,
    ))
}

fn main() {
    match run() {
        Ok((
            max_error,
            adjoint_error,
            forward_residual,
            adjoint_residual,
            rejected_inputs,
            spmv_format_count,
            dimension_rejections,
            non_finite_rejections,
            breakdown_status_checks,
            gmres_dimension_rejections,
            gmres_non_finite_rejections,
            gmres_panic_rejections,
            gmres_status_checks,
            checked_family_metrics,
            checked_block_family_metrics,
            checked_mixed_precision_metrics,
            facade_result_checks,
        )) => {
            let record = QaResult::passed("solvers.robustness-metamorphic.v1")
                .requested_backend("cpu")
            .actual_backend("cpu")
            .seed("seed=20260819")
            .metric("replay_seed", 20260819.0)
                .metric("max_solution_error", max_error)
                .metric("max_adjoint_identity_error", adjoint_error)
                .metric("forward_residual", forward_residual)
                .metric("adjoint_residual", adjoint_residual)
                .metric("rejected_adjoint_inputs", rejected_inputs)
                .metric("spmv_format_count", spmv_format_count)
                .metric("dimension_rejections", dimension_rejections)
                .metric("non_finite_rejections", non_finite_rejections)
                .metric("breakdown_status_checks", breakdown_status_checks)
                .metric("gmres_dimension_rejections", gmres_dimension_rejections)
                .metric("gmres_non_finite_rejections", gmres_non_finite_rejections)
                .metric("gmres_panic_rejections", gmres_panic_rejections)
                .metric("gmres_status_checks", gmres_status_checks)
                .metric(
                    "checked_family_converged",
                    checked_family_metrics.converged,
                )
                .metric(
                    "checked_family_dimension_rejections",
                    checked_family_metrics.dimension_rejections,
                )
                .metric(
                    "checked_family_non_finite_rejections",
                    checked_family_metrics.non_finite_rejections,
                )
                .metric(
                    "checked_family_panic_rejections",
                    checked_family_metrics.panic_rejections,
                )
                .metric(
                    "checked_family_breakdown_status_checks",
                    checked_family_metrics.breakdown_status_checks,
                )
                .metric(
                    "block_converged_rhs",
                    checked_block_family_metrics.block_converged,
                )
                .metric(
                    "block_dimension_rejections",
                    checked_block_family_metrics.block_dimension_rejections,
                )
                .metric(
                    "block_non_finite_rejections",
                    checked_block_family_metrics.block_non_finite_rejections,
                )
                .metric(
                    "block_panic_rejections",
                    checked_block_family_metrics.block_panic_rejections,
                )
        .metric(
            "block_status_checks",
            checked_block_family_metrics.block_status_checks,
        )
        .metric(
            "block_column_checks",
            checked_block_family_metrics.block_column_checks,
        )
        .metric(
            "block_warm_start_checks",
            checked_block_family_metrics.block_warm_start_checks,
        )
                .metric(
                    "gmres_dr_converged",
                    checked_block_family_metrics.gmres_dr_converged,
                )
                .metric(
                    "gmres_dr_failure_rejections",
                    checked_block_family_metrics.gmres_dr_failure_rejections,
                )
                .metric(
                    "gmres_dr_status_checks",
                    checked_block_family_metrics.gmres_dr_status_checks,
                )
                .metric(
                    "recycled_converged",
                    checked_block_family_metrics.recycled_converged,
                )
                .metric(
                    "recycled_failure_rejections",
                    checked_block_family_metrics.recycled_failure_rejections,
                )
                .metric(
                    "recycled_status_checks",
                    checked_block_family_metrics.recycled_status_checks,
                )
                .metric(
                    "mixed_precision_converged",
                    checked_mixed_precision_metrics.converged,
                )
                .metric(
                    "mixed_precision_metadata_checks",
                    checked_mixed_precision_metrics.metadata_checks,
                )
                .metric(
                    "mixed_precision_dimension_rejections",
                    checked_mixed_precision_metrics.dimension_rejections,
                )
                .metric(
                    "mixed_precision_non_finite_rejections",
                    checked_mixed_precision_metrics.non_finite_rejections,
                )
                .metric(
                    "mixed_precision_cast_rejections",
                    checked_mixed_precision_metrics.cast_rejections,
                )
                .metric(
                    "mixed_precision_initial_guess_rejections",
                    checked_mixed_precision_metrics.initial_guess_rejections,
                )
                .metric("facade_result_checks", facade_result_checks)
                .maximum("max_solution_error", 1.0e-10)
                .maximum("max_adjoint_identity_error", 1.0e-10)
                .maximum("forward_residual", 1.0e-8)
                .maximum("adjoint_residual", 1.0e-8)
                .minimum("rejected_adjoint_inputs", 1.0)
                .minimum("spmv_format_count", 8.0)
                .minimum("dimension_rejections", 1.0)
                .minimum("non_finite_rejections", 1.0)
                .minimum("breakdown_status_checks", 1.0)
                .minimum("gmres_dimension_rejections", 1.0)
                .minimum("gmres_non_finite_rejections", 1.0)
                .minimum("gmres_panic_rejections", 1.0)
                .minimum("gmres_status_checks", 1.0)
                .minimum("checked_family_converged", 6.0)
                .minimum("checked_family_dimension_rejections", 6.0)
                .minimum("checked_family_non_finite_rejections", 6.0)
                .minimum("checked_family_panic_rejections", 6.0)
                .minimum("checked_family_breakdown_status_checks", 6.0)
                .minimum("block_converged_rhs", 2.0)
                .minimum("block_dimension_rejections", 1.0)
                .minimum("block_non_finite_rejections", 1.0)
                .minimum("block_panic_rejections", 1.0)
        .minimum("block_status_checks", 1.0)
        .minimum("block_column_checks", 2.0)
        .minimum("block_warm_start_checks", 2.0)
                .minimum("gmres_dr_converged", 1.0)
                .minimum("gmres_dr_failure_rejections", 3.0)
                .minimum("gmres_dr_status_checks", 1.0)
                .minimum("recycled_converged", 1.0)
                .minimum("recycled_failure_rejections", 3.0)
                .minimum("recycled_status_checks", 1.0)
                .minimum("mixed_precision_converged", 3.0)
                .minimum("mixed_precision_metadata_checks", 3.0)
                .minimum("mixed_precision_dimension_rejections", 1.0)
                .minimum("mixed_precision_non_finite_rejections", 2.0)
                .minimum("mixed_precision_cast_rejections", 2.0)
                .minimum("mixed_precision_initial_guess_rejections", 4.0)
                .minimum("facade_result_checks", 5.0)
                .reference(
                    "complex inner-product identities, true residuals, checked plan rejection, typed scalar/block/GMRES-DR/recycled/mixed-precision GMRES plus BiCGSTAB/pipelined BiCGSTAB/CG/CGS/IDR(s)/IDR(s)stab failure status, and high-level linear/Anderson/Newton result status and accuracy contracts",
                )
                .reference_checksum("analytic/deterministic-fixture")
                .dimension(
                    "2 DOF Complex64 adjoint fixture; 8x8 real CSR plan formats; 2 DOF checked scalar/block/deflated/recycled/mixed-precision GMRES and six-family iterative fixtures",
                )
                .formulation(
                    "2x2 complex nonsymmetric CSR with ILU/GMRES adjoints; checked 8x8 SpMV plan formats; scalar, multi-RHS block, GMRES-DR/recycled, and f64-to-f32/Complex32 mixed-precision GMRES plus BiCGSTAB/pipelined BiCGSTAB/CG/CGS/IDR(s)/IDR(s)stab dimension/non-finite/panic/status fixtures; high-level linear facade and nonlinear Anderson/Newton metadata fixtures",
                )
                .tolerance(
                    "all identity errors <= 1e-10; residuals <= 1e-8; eight plan formats reject invalid inputs; scalar/block/GMRES-DR/recycled/mixed-precision GMRES and six iterative families reject invalid inputs and retain finite non-convergent/breakdown statuses; five high-level linear/nonlinear status and accuracy checks pass",
                );
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("solvers.robustness-metamorphic.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("seed=20260819")
                .metric("replay_seed", 20260819.0)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("SOLVER ROBUSTNESS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<
    (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        CheckedFamilyMetrics,
        CheckedBlockFamilyMetrics,
        CheckedMixedPrecisionMetrics,
        f64,
    ),
    String,
> {
    let x = solve(&matrix(1.0, false), &[9.0, 8.0])?;
    let scaled = solve(&matrix(7.0, false), &[63.0, 56.0])?;
    // P A Pᵀ and P b have solution P x; unpermute before comparison.
    let permutation = solve(&matrix(1.0, true), &[8.0, 9.0])?;
    let unpermuted = [permutation[1], permutation[0]];
    let expected = [19.0 / 11.0, 23.0 / 11.0];
    let mut max_error = 0.0_f64;
    for candidate in [&x[..], &scaled[..], &unpermuted] {
        if candidate.iter().zip(expected).any(|(got, expected)| {
            let error = (got - expected).abs();
            max_error = max_error.max(error);
            error > 1e-10
        }) {
            return Err(format!("metamorphic solution mismatch: {candidate:?}"));
        }
    }
    println!("metamorphic=scale,permutation solution={x:?}, max_error={max_error:.3e}");
    let (adjoint_error, forward_residual, adjoint_residual, rejected_inputs) =
        checked_complex_adjoint_contracts()?;
    let spmv_format_count = checked_spmv_plan_contracts()?;
    let (
        gmres_dimension_rejections,
        gmres_non_finite_rejections,
        gmres_panic_rejections,
        gmres_status_checks,
        dimension_rejections,
        non_finite_rejections,
        breakdown_status_checks,
        checked_family_metrics,
    ) = checked_solver_failure_contracts()?;
    let checked_block_family_metrics = checked_block_family_contracts()?;
    let checked_mixed_precision_metrics = checked_mixed_precision_contracts()?;
    let facade_result_checks = checked_facade_result_contracts()?;
    Ok((
        max_error,
        adjoint_error,
        forward_residual,
        adjoint_residual,
        rejected_inputs,
        spmv_format_count,
        dimension_rejections,
        non_finite_rejections,
        breakdown_status_checks,
        gmres_dimension_rejections,
        gmres_non_finite_rejections,
        gmres_panic_rejections,
        gmres_status_checks,
        checked_family_metrics,
        checked_block_family_metrics,
        checked_mixed_precision_metrics,
        facade_result_checks,
    ))
}
