use sonium_solvers_core::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers_direct::{
    BlrError, IrConfig, IrError, IterativeRefinement, MixedPrecisionRefinement,
    MultifrontalBlrFactorization,
};
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn checked_constructor_validates_dimensions_matrix_and_configuration() {
    let matrix = CsrMatrix::<f64>::identity(2);
    let factor = MultifrontalBlrFactorization::factorize(&matrix).unwrap();

    assert!(matches!(
        IterativeRefinement::try_new(
            &factor,
            &matrix,
            IrConfig {
                tolerance: f64::NAN,
                ..IrConfig::default()
            },
        ),
        Err(IrError::InvalidConfiguration { field: "tolerance" })
    ));

    let smaller = CsrMatrix::<f64>::identity(1);
    assert!(matches!(
        IterativeRefinement::try_new(&factor, &smaller, IrConfig::default()),
        Err(IrError::FactorDimensionMismatch {
            factor: 2,
            matrix: 1,
        })
    ));

    let mut non_finite = matrix.clone();
    non_finite.values[0] = f64::INFINITY;
    assert!(matches!(
        IterativeRefinement::try_new(&factor, &non_finite, IrConfig::default()),
        Err(IrError::NonFiniteMatrix(0))
    ));
}

#[test]
fn checked_solve_rejects_bad_rhs_and_accepts_large_finite_norm() {
    let matrix = CsrMatrix::<f64>::identity(1);
    let factor = MultifrontalBlrFactorization::factorize(&matrix).unwrap();
    let refinement = IterativeRefinement::try_new(&factor, &matrix, IrConfig::default()).unwrap();

    assert!(matches!(
        refinement.try_solve(&[]),
        Err(IrError::RhsDimensionMismatch {
            expected: 1,
            actual: 0,
        })
    ));
    assert!(matches!(
        refinement.try_solve(&[f64::NAN]),
        Err(IrError::NonFiniteRhs(0))
    ));

    let result = refinement.try_solve(&[f64::MAX]).unwrap();
    assert_eq!(result.x, vec![f64::MAX]);
    assert_eq!(result.final_residual, 0.0);
    assert!(result.converged);
    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.attainable_accuracy(), AccuracyStatus::Attained);
}

#[test]
fn mixed_precision_refinement_reports_downcast_overflow() {
    let matrix_low = CsrMatrix::<f32>::identity(1);
    let factor_low = MultifrontalBlrFactorization::factorize(&matrix_low).unwrap();
    let matrix_high = CsrMatrix::<f64>::identity(1);
    let refinement = MixedPrecisionRefinement::<f64, f32>::try_new(
        &factor_low,
        &matrix_high,
        IrConfig::default(),
    )
    .unwrap();

    assert!(matches!(
        refinement.try_solve(&[f64::MAX]),
        Err(IrError::PrecisionConversion {
            stage: "right-hand-side downcast",
            index: 0,
        })
    ));
}

#[test]
fn refinement_errors_expose_canonical_status_and_accuracy() {
    let input = IrError::RhsDimensionMismatch {
        expected: 2,
        actual: 1,
    };
    assert_eq!(input.status(), SolverStatus::NumericalError);
    assert_eq!(input.attainable_accuracy(), AccuracyStatus::NotChecked);

    let breakdown = IrError::NumericalBreakdown {
        stage: "correction accumulation",
        index: 0,
    };
    assert_eq!(breakdown.status(), SolverStatus::Breakdown);
    assert_eq!(breakdown.attainable_accuracy(), AccuracyStatus::NotChecked);

    let singular = BlrError::SingularFront(0);
    assert_eq!(singular.status(), SolverStatus::Breakdown);
    assert_eq!(singular.attainable_accuracy(), AccuracyStatus::NotChecked);

    let propagated = IrError::DirectSolve(singular);
    assert_eq!(propagated.status(), SolverStatus::Breakdown);
    assert_eq!(propagated.attainable_accuracy(), AccuracyStatus::NotChecked);
}
