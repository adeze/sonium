use num_traits::Zero;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    AccuracyStatus, AdjointSolve, BlockFgmresConfig, BlockFgmresSolution, CsrMatrix,
    DifferentiableBlockSolveResult, DifferentiableLinearSolve, DifferentiableSolveContract,
    GmresConfig, IdentityPreconditioner, KrylovOpCounts, SolverError, SolverResultContract,
};

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ],
    )
}

fn block(columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    let mut result = DenseMatrix::from_elem(3, 2, f64::zero());
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            result[[row, column]] = *value;
        }
    }
    result
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn block_config() -> BlockFgmresConfig<f64> {
    BlockFgmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-11,
        true_residual_check: true,
        exact_residual_check_interval: 0,
    }
}

fn block_solution(rows: usize) -> BlockFgmresSolution<f64> {
    BlockFgmresSolution {
        x: DenseMatrix::zeros(rows, 1),
        iterations: 1,
        residuals: vec![0.0],
        true_residuals: vec![0.0],
        backward_errors: vec![0.0],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    }
}

#[test]
fn differentiable_block_promotes_projected_only_configuration() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config())
        .expect("matrix is square");
    let rhs = block(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let mut projected_only = block_config();
    projected_only.true_residual_check = false;
    projected_only.exact_residual_check_interval = 0;

    let result = handle
        .solve_forward_block(&rhs, &projected_only)
        .expect("differentiable block solve should complete");
    assert!(result.converged);
    assert!(result.op_counts.exact_residual_checks > 0);
    assert!(
        result
            .true_residuals
            .iter()
            .all(|residual| *residual <= 1.0e-11)
    );
}

#[test]
fn differentiable_block_forward_and_hermitian_adjoint_share_contract() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let forward_rhs = block(&[[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]]);
    let adjoint_rhs = block(&[[3.0, -2.0, 1.0], [-1.0, 4.0, 2.0]]);
    let result = handle
        .solve_block(
            &forward_rhs,
            Some((&adjoint_rhs, AdjointSolve::Hermitian)),
            &block_config(),
        )
        .unwrap();
    assert!(result.forward.converged);
    assert!(
        result
            .adjoint
            .as_ref()
            .is_some_and(|solution| solution.converged)
    );
    assert!(
        result
            .forward
            .true_residuals
            .iter()
            .chain(result.adjoint.as_ref().unwrap().true_residuals.iter())
            .all(|residual| *residual < 1.0e-9)
    );
    result.validate_contract().unwrap();
    assert_eq!(result.forward.x.ncols(), 2);
}

#[test]
fn differentiable_block_continuation_reuses_forward_and_adjoint_iterates() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let forward_rhs = block(&[[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]]);
    let adjoint_rhs = block(&[[3.0, -2.0, 1.0], [-1.0, 4.0, 2.0]]);
    let block_config = block_config();

    let cold_forward = handle
        .solve_forward_block(&forward_rhs, &block_config)
        .unwrap();
    assert!(cold_forward.converged);
    let warm_forward = handle
        .solve_forward_block_with_guess(&forward_rhs, Some(&cold_forward.x), &block_config)
        .unwrap();
    assert!(warm_forward.converged);
    assert_eq!(warm_forward.iterations, 0);
    assert!(
        warm_forward
            .true_residuals
            .iter()
            .all(|residual| *residual <= 1.0e-11)
    );

    let cold_adjoint = handle
        .solve_adjoint_block(&adjoint_rhs, AdjointSolve::Hermitian, &block_config)
        .unwrap();
    assert!(cold_adjoint.converged);
    let warm_adjoint = handle
        .solve_adjoint_block_with_guess(
            &adjoint_rhs,
            AdjointSolve::Hermitian,
            Some(&cold_adjoint.x),
            &block_config,
        )
        .unwrap();
    assert!(warm_adjoint.converged);
    assert_eq!(warm_adjoint.iterations, 0);
    assert!(
        warm_adjoint
            .true_residuals
            .iter()
            .all(|residual| *residual <= 1.0e-11)
    );
}

#[test]
fn differentiable_block_contract_rejects_forward_adjoint_dimension_mismatch() {
    let result = DifferentiableBlockSolveResult {
        forward: block_solution(2),
        adjoint: Some(block_solution(3)),
    };

    let error = result
        .validate_contract()
        .expect_err("forward and adjoint block row dimensions must agree");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::DimensionMismatch { .. }
    ));
}

#[test]
fn differentiable_block_rejects_nonfinite_and_dimension_mismatch() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let malformed = DenseMatrix::from_elem(2, 1, f64::NAN);
    assert!(matches!(
        handle.solve_forward_block(&malformed, &block_config()),
        Err(SolverError::DimensionMismatch { .. })
    ));
    let nonfinite = DenseMatrix::from_elem(3, 1, f64::NAN);
    assert!(matches!(
        handle.solve_forward_block(&nonfinite, &block_config()),
        Err(SolverError::OperationFailed { .. })
    ));

    let rhs = block(&[[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]]);
    let bad_shape = DenseMatrix::zeros(2, 2);
    assert!(matches!(
        handle.solve_forward_block_with_guess(&rhs, Some(&bad_shape), &block_config()),
        Err(SolverError::DimensionMismatch { .. })
    ));
    let bad_values = DenseMatrix::from_elem(3, 2, f64::NAN);
    assert!(matches!(
        handle.solve_forward_block_with_guess(&rhs, Some(&bad_values), &block_config()),
        Err(SolverError::OperationFailed { .. })
    ));
}

#[test]
fn differentiable_block_result_contract_rejects_bad_metadata() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let rhs = block(&[[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]]);
    let mut forward = handle.solve_forward_block(&rhs, &block_config()).unwrap();
    forward.true_residuals[0] = f64::NAN;
    let invalid = DifferentiableBlockSolveResult {
        forward,
        adjoint: None,
    };
    assert!(invalid.validate_contract().is_err());
}

#[test]
fn steady_differentiable_contract_rejects_stale_forward_state_before_adjoint() {
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let mut forward = handle.solve_forward(&[1.0, 2.0, -1.0]).unwrap();
    forward.x[0] = f64::NAN;
    let seed = sonium_solvers::SteadyAdjointSeed::new(vec![1.0, 0.0, 0.5], AdjointSolve::Hermitian);

    let error = DifferentiableSolveContract::solve_adjoint(&handle, &forward, &seed)
        .expect_err("tampered retained state must fail before adjoint iteration");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(error.to_string().contains("steady forward state"));

    let valid_forward = handle.solve_forward(&[1.0, 2.0, -1.0]).unwrap();
    let bad_seed = sonium_solvers::SteadyAdjointSeed::new(vec![1.0, 0.0], AdjointSolve::Hermitian);
    let error = DifferentiableSolveContract::solve_adjoint(&handle, &valid_forward, &bad_seed)
        .expect_err("adjoint seed dimension must match the retained forward state");
    assert!(matches!(error, SolverError::DimensionMismatch { .. }));
}
