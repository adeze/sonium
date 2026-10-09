use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn diagonal(value: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(1, 1, vec![(0, 0, value)])
}

#[test]
fn factorization_rejects_non_finite_pivot_thresholds() {
    let matrix = diagonal(1.0);

    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            SparseLuFactorization::factorize_with_pivot_threshold(&matrix, threshold),
            Err(SparseLuError::InvalidPivotThreshold)
        ));
        assert!(matches!(
            SparseLuFactorization::factorize_supernodal_with_pivot_threshold(&matrix, threshold),
            Err(SparseLuError::InvalidPivotThreshold)
        ));
    }
}

#[test]
fn factorization_reports_generated_overflow() {
    let max = f64::MAX;
    let matrix: CsrMatrix<f64> = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, max), (0, 1, max), (1, 0, max), (1, 1, -max)],
    );

    assert!(matches!(
        SparseLuFactorization::factorize(&matrix),
        Err(SparseLuError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        SparseLuFactorization::factorize_supernodal(&matrix),
        Err(SparseLuError::NumericalBreakdown { .. })
    ));
}

#[test]
fn solve_reports_finite_rhs_overflow() {
    for factor in [
        SparseLuFactorization::factorize(&diagonal(1.0e-20)).unwrap(),
        SparseLuFactorization::factorize_supernodal(&diagonal(1.0e-20)).unwrap(),
    ] {
        assert!(matches!(
            factor.solve(&[f64::MAX]),
            Err(SparseLuError::NumericalBreakdown {
                stage: "backward substitution",
                index: 0,
            })
        ));
    }
}

#[test]
fn checked_solve_succeeds_for_both_backends() {
    let matrix: CsrMatrix<f64> = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)],
    );
    for factor in [
        SparseLuFactorization::factorize(&matrix).unwrap(),
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
    ] {
        let solution = factor.solve(&[6.0, 8.0]).unwrap();
        assert!((solution[0] - 1.0).abs() < 1.0e-12);
        assert!((solution[1] - 2.0).abs() < 1.0e-12);
    }
}

#[test]
fn solve_time_singular_pivot_reports_first_bad_column_everywhere() {
    // 1e-31·I passes the factorize-time eps·n·‖A‖ pivot guard but trips the
    // solve-time 1e-30 absolute floor: factorization must succeed and every
    // solve entry point must report the same first-bad-column error (this
    // pins the cached-diagonal validation shared by the single, adjoint,
    // and block paths).
    let triplets: Vec<(usize, usize, f64)> = (0..3).map(|i| (i, i, 1e-31)).collect();
    let matrix = CsrMatrix::from_triplets(3, 3, triplets);
    let factor = SparseLuFactorization::factorize(&matrix).expect("tiny-scale factorization");
    let rhs = vec![1.0; 3];
    assert!(matches!(
        factor.solve(&rhs),
        Err(SparseLuError::SingularMatrix(0))
    ));
    assert!(matches!(
        factor.solve_transpose(&rhs),
        Err(SparseLuError::SingularMatrix(0))
    ));
    let block = vec![1.0; 3 * 2];
    let mut output = vec![0.0; 3 * 2];
    assert!(matches!(
        factor.solve_many_into(&block, &mut output, 2),
        Err(SparseLuError::SingularMatrix(0))
    ));
    assert!(matches!(
        factor.solve_transpose_many(&block, 2),
        Err(SparseLuError::SingularMatrix(0))
    ));
}
