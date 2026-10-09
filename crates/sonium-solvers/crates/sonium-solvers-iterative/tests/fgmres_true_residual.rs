use sonium_solvers_core::traits::{IdentityPreconditioner, LinearOperator};
use sonium_solvers_iterative::{FgmresConfig, fgmres_preconditioned};
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn infallible_fgmres_reports_original_operator_residual_on_max_iterations() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0_f64), (1, 1, 3.0_f64)]);
    let rhs = [1.0, 1.0];
    let config = FgmresConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1e-14,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
        ..Default::default()
    };

    let solution = fgmres_preconditioned(&matrix, &IdentityPreconditioner, &rhs, None, &config);
    let ax = matrix.apply(&solution.x);
    let residual_norm = ax
        .iter()
        .zip(rhs.iter())
        .map(|(&actual, &expected)| {
            let difference = expected - actual;
            difference * difference
        })
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    let expected_relative_residual = residual_norm / rhs_norm;

    assert!((solution.residual - expected_relative_residual).abs() < 1e-14);
    assert_eq!(
        solution.converged,
        expected_relative_residual <= config.tolerance
    );
}

#[test]
fn infallible_fgmres_marks_all_residual_metrics_on_operator_failure() {
    struct NonFiniteOperator;

    impl LinearOperator<f64> for NonFiniteOperator {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
            y[0] = f64::NAN;
        }

        fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
            y[0] = f64::NAN;
        }
    }

    let solution = fgmres_preconditioned(
        &NonFiniteOperator,
        &IdentityPreconditioner,
        &[1.0],
        None,
        &FgmresConfig::default(),
    );

    assert_eq!(
        solution.status,
        sonium_solvers_core::traits::SolverStatus::NumericalError
    );
    assert!(!solution.converged);
    assert!(solution.residual.is_infinite());
    assert!(solution.estimated_residual.is_infinite());
    assert!(solution.backward_error.is_infinite());
}
