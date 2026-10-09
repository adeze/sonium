//! Public saddle-point compatibility and result-contract checks.

use sonium_solvers::{
    BlockPreconditionerConfig, CsrBuilder, CsrMatrix, LinearSolveConfig, LinearSolveError,
    LinearSolverType, SaddlePointConfig, SchurStrategy, solve_saddle_point,
};

fn saddle_matrix() -> CsrMatrix<f64> {
    let rows = [
        vec![(0, 4.0), (1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (1, 3.0), (3, 1.0)],
        vec![(0, 1.0), (2, 2.0)],
        vec![(1, 1.0), (3, 1.5)],
    ];
    let mut builder = CsrBuilder::new(4, 4);
    for row in rows {
        builder.add_row_entries(row.into_iter());
    }
    builder.finish()
}

fn solve_config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 80,
        tolerance: 1.0e-10,
        ..Default::default()
    }
}

fn saddle_config() -> SaddlePointConfig {
    SaddlePointConfig {
        block1_size: 2,
        schur_strategy: SchurStrategy::External,
        block2_config: BlockPreconditionerConfig::Diagonal,
        ..Default::default()
    }
}

fn schur_approximation() -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(2, 2);
    builder.add_row_entries([(0, 2.0)].into_iter());
    builder.add_row_entries([(1, 1.5)].into_iter());
    builder.finish()
}

fn rhs() -> Vec<f64> {
    saddle_matrix().matvec(&[1.0, -0.5, 0.75, 1.5])
}

#[test]
fn public_saddle_point_solve_reports_residual_and_assumptions() {
    let matrix = saddle_matrix();
    let result = solve_saddle_point(
        &matrix,
        &rhs(),
        &solve_config(),
        saddle_config(),
        Some(&schur_approximation()),
    )
    .expect("valid public saddle-point solve");

    assert!(result.converged);
    assert!(result.residual <= 1.0e-10);
    assert!(result.validate_contract().is_ok());
    let assumptions = result
        .saddle_assumptions
        .expect("saddle result must retain assumption evidence");
    assert_eq!(
        assumptions.block1_provenance,
        sonium_solvers::AssumptionProvenance::IndependentlyValidated
    );
    assert_eq!(
        assumptions.block2_provenance,
        sonium_solvers::AssumptionProvenance::NotRequired
    );
}

#[test]
fn public_saddle_point_rejects_malformed_rhs_and_guess() {
    let matrix = saddle_matrix();
    let config = solve_config();
    let schur = schur_approximation();

    let rhs_error = solve_saddle_point(
        &matrix,
        &[1.0, 2.0, 3.0],
        &config,
        saddle_config(),
        Some(&schur),
    )
    .expect_err("short saddle RHS must be rejected");
    assert!(matches!(
        rhs_error,
        LinearSolveError::DimensionMismatch {
            operation: "saddle-point rhs",
            expected: 4,
            actual: 3,
        }
    ));

    let guess_error = sonium_solvers::solve_saddle_point_with_guess(
        &matrix,
        &rhs(),
        &[0.0, 0.0],
        &config,
        saddle_config(),
        Some(&schur),
    )
    .expect_err("short saddle initial guess must be rejected");
    assert!(matches!(
        guess_error,
        LinearSolveError::DimensionMismatch {
            operation: "saddle-point initial guess",
            expected: 4,
            actual: 2,
        }
    ));
}

#[test]
fn public_saddle_point_rejects_unsupported_and_incomplete_configuration() {
    let matrix = saddle_matrix();
    let rhs = rhs();

    let unsupported = solve_saddle_point(
        &matrix,
        &rhs,
        &LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..solve_config()
        },
        saddle_config(),
        Some(&schur_approximation()),
    )
    .expect_err("CG must not dispatch to a saddle-point solve");
    assert!(matches!(unsupported, LinearSolveError::InvalidConfig(_)));

    let missing_schur = solve_saddle_point(&matrix, &rhs, &solve_config(), saddle_config(), None)
        .expect_err("external Schur strategy requires an approximation");
    assert!(matches!(missing_schur, LinearSolveError::InvalidConfig(_)));
}
