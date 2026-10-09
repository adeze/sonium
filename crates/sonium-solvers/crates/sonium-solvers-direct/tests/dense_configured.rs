use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::SolverResultContract;
use sonium_solvers_direct::{
    DenseDirectConfig, DenseDirectError, DenseDirectMethod, DirectFactorization, MatrixClass,
    factorize_dense, factorize_dense_configured,
};

fn general_config() -> DenseDirectConfig {
    DenseDirectConfig {
        matrix_class: MatrixClass::General,
        method: DenseDirectMethod::Lu,
        ..DenseDirectConfig::default()
    }
}

#[test]
fn configured_factor_retains_condition_estimate_and_result_diagnostics() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]);
    let config = DenseDirectConfig {
        estimate_condition: true,
        iterative_refinement_steps: 2,
        ..general_config()
    };
    let factor = factorize_dense_configured(&matrix, &config).unwrap();

    let estimate = factor.condition_estimate().unwrap();
    assert!(
        (estimate - 3.0).abs() < 1.0e-14,
        "condition estimate={estimate}"
    );

    let rhs = [6.0_f64, 8.0];
    let result = factor.solve_with_result(&rhs).unwrap();
    result.validate_contract().unwrap();
    assert_eq!(result.x.len(), 2);
    assert!(result.backward_error.unwrap().is_finite());
    assert!(result.residual.unwrap().is_finite());
}

#[test]
fn configured_refinement_respects_iteration_cap_and_tolerance_validation() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![0.1_f64, 0.2, 0.3, 0.7]);
    let config = DenseDirectConfig {
        iterative_refinement_steps: 2,
        ..general_config()
    };
    let factor = factorize_dense_configured(&matrix, &config).unwrap();
    let result = factor
        .solve_refined_with_tolerance(&[0.123456789, 0.987654321], 0.0, 2)
        .unwrap();
    assert!(result.iterations > 0 && result.iterations <= 2);
    assert!(result.backward_error.is_finite());

    assert!(matches!(
        factor.solve_refined_with_tolerance(&[0.123456789, 0.987654321], f64::NAN, 1),
        Err(DenseDirectError::InvalidRefinementTolerance)
    ));
}

#[test]
fn configured_condition_estimate_covers_complex_scalar_family() {
    let matrix = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    );
    let config = DenseDirectConfig {
        estimate_condition: true,
        ..general_config()
    };
    let factor = factorize_dense_configured(&matrix, &config).unwrap();
    assert!((factor.condition_estimate().unwrap() - 3.0).abs() < 1.0e-14);
    let result = factor
        .solve_with_result(&[Complex64::new(6.0, 0.0), Complex64::new(8.0, 0.0)])
        .unwrap();
    assert!((result.x[0].norm() - 1.0).abs() < 1.0e-14);
    assert!((result.x[1].norm() - 2.0).abs() < 1.0e-14);
}

#[test]
fn configured_api_rejects_unsafe_rectangular_refinement_and_legacy_flags() {
    let rectangular = DenseMatrix::from_row_major_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let config = DenseDirectConfig {
        iterative_refinement_steps: 1,
        matrix_class: MatrixClass::RectangularFullRank,
        method: DenseDirectMethod::Qr,
        ..DenseDirectConfig::default()
    };
    assert!(matches!(
        factorize_dense_configured(&rectangular, &config),
        Err(DenseDirectError::OptionUnavailable { .. })
    ));

    let square = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]);
    let legacy_config = DenseDirectConfig {
        estimate_condition: true,
        ..general_config()
    };
    assert!(matches!(
        factorize_dense(&square, &legacy_config),
        Err(DenseDirectError::OptionUnavailable {
            option: "condition estimation"
        })
    ));
}
