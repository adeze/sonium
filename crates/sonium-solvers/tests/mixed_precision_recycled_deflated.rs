use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
#[allow(deprecated)]
use sonium_solvers::{
    DeflationSubspace, GcroDrConfig, GcroDrSubspace, GmresConfig, RecycledSubspace, SolverStatus,
    try_mixed_precision_deflated_solve, try_mixed_precision_gcro_dr_solve_with_guess,
    try_mixed_precision_recycled_solve, try_mixed_precision_recycled_solve_with_guess,
};

fn diagonal_operator() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0])
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 2,
        restart: 1,
        tolerance: 1.0e-20,
        exact_residual_check_interval: 0,
        ..GmresConfig::default()
    }
}

fn gcro_config() -> GcroDrConfig<f64> {
    GcroDrConfig {
        max_iterations: 2,
        restart: 1,
        tolerance: 1.0e-20,
        print_interval: 0,
    }
}

#[test]
fn mixed_precision_deflated_forces_high_precision_candidate_gate() {
    let operator = diagonal_operator();
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0]], &operator)
        .expect("valid deflation subspace");

    let result = try_mixed_precision_deflated_solve::<f64, f32, _>(
        &operator,
        &deflation,
        &[1.0, 1.0],
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &config(),
    )
    .expect("mixed-precision deflated solve should complete");

    assert!(result.op_counts.exact_residual_checks > 0);
    assert!(result.residual.is_finite());
    assert_eq!(result.status, SolverStatus::Converged);
    assert!(result.converged);
}

#[test]
#[allow(deprecated)]
fn mixed_precision_recycled_continuation_accepts_exact_iterate() {
    let operator = diagonal_operator();
    let mut subspace = RecycledSubspace::new(1);

    let first = try_mixed_precision_recycled_solve::<f64, f32, _>(
        &operator,
        &[1.0, 1.0],
        &mut subspace,
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &config(),
    )
    .expect("initial mixed-precision recycled solve should complete");
    assert!(first.op_counts.exact_residual_checks > 0);

    let resumed = try_mixed_precision_recycled_solve_with_guess::<f64, f32, _>(
        &operator,
        &[1.0, 1.0],
        Some(&[0.5, 1.0 / 3.0]),
        &mut subspace,
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &config(),
    )
    .expect("exact mixed-precision recycled continuation should complete");

    assert!(resumed.converged);
    assert_eq!(resumed.iterations, 0);
    assert!(resumed.residual < 1.0e-14);
}

#[test]
fn mixed_precision_gcro_dr_continuation_accepts_exact_iterate() {
    let operator = diagonal_operator();
    let mut subspace = GcroDrSubspace::new(1);
    let resumed = try_mixed_precision_gcro_dr_solve_with_guess::<f64, f32, _>(
        &operator,
        &[1.0, 1.0],
        Some(&[0.5, 1.0 / 3.0]),
        &mut subspace,
        Box::new(DiagonalPreconditioner::from_diagonal(&[1.0_f32, 1.0])),
        &gcro_config(),
    )
    .expect("exact mixed-precision GCRO-DR continuation should complete");

    assert!(resumed.converged);
    assert_eq!(resumed.iterations, 0);
    assert!(resumed.residual < 1.0e-14);
}
