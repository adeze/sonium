use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_direct::{
    DenseDirectConfig, DenseDirectMethod, MatrixClass, factorize_dense_configured,
};

#[test]
fn dense_refinement_result_exposes_checked_status_and_accuracy() {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]);
    let config = DenseDirectConfig {
        matrix_class: MatrixClass::General,
        method: DenseDirectMethod::Lu,
        iterative_refinement_steps: 2,
        ..DenseDirectConfig::default()
    };
    let factor = factorize_dense_configured(&matrix, &config).expect("configured dense factor");
    let result = factor
        .solve_refined_with_tolerance(&[1.0, 2.0], 1.0e-12, 2)
        .expect("dense refinement");

    result
        .validate_contract()
        .expect("valid refinement contract");
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);

    let mut contradictory = result.clone();
    contradictory.converged = true;
    contradictory.backward_error = 1.0;
    contradictory.tolerance = 1.0e-12;
    assert!(contradictory.validate_contract().is_err());

    let mut non_finite = result;
    non_finite.x[0] = f64::NAN;
    assert_eq!(non_finite.solver_status(), SolverStatus::NumericalError);
    assert_eq!(non_finite.accuracy_status(), AccuracyStatus::NotChecked);
    assert!(non_finite.validate_contract().is_err());
}
