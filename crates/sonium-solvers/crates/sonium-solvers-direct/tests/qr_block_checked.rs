use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{
    DenseDirectConfig, DenseDirectError, MatrixClass, QrError, QrFactorization, factorize_dense,
};

#[test]
fn qr_multi_rhs_matches_single_rhs_and_preserves_destination_on_rejection() {
    let matrix =
        DenseMatrix::from_row_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let factor = QrFactorization::factorize(&matrix).expect("full-rank QR factorization");
    let rhs = vec![
        2.0, 1.0, // row 0
        -3.0, 4.0, // row 1
        -1.0, 5.0, // row 2
        7.0, -2.0, // row 3
    ];
    let mut solution = vec![99.0; 4];
    factor
        .solve_least_squares_many_into(&rhs, &mut solution, 2)
        .expect("two-RHS least-squares solve");

    let first = factor
        .solve_least_squares(&[2.0, -3.0, -1.0, 7.0])
        .expect("first RHS");
    let second = factor
        .solve_least_squares(&[1.0, 4.0, 5.0, -2.0])
        .expect("second RHS");
    assert_eq!(solution, vec![first[0], second[0], first[1], second[1]],);

    let mut rejected = vec![7.0; 4];
    assert!(matches!(
        factor.solve_least_squares_many_into(&[1.0; 3], &mut rejected, 2),
        Err(QrError::RhsDimension {
            expected: 8,
            actual: 3,
        })
    ));
    assert_eq!(rejected, vec![7.0; 4]);
    assert!(matches!(
        factor.solve_least_squares_many_into(&[], &mut rejected, usize::MAX),
        Err(QrError::SizeOverflow { .. })
    ));
    assert_eq!(rejected, vec![7.0; 4]);
}

#[test]
fn dense_facade_routes_rectangular_multi_rhs_through_checked_qr() {
    let matrix =
        DenseMatrix::from_row_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let factor = factorize_dense(
        &matrix,
        &DenseDirectConfig {
            matrix_class: MatrixClass::RectangularFullRank,
            ..Default::default()
        },
    )
    .expect("rectangular QR factorization");
    let rhs = vec![2.0, 1.0, -3.0, 4.0, -1.0, 5.0, 7.0, -2.0];
    let mut solution = vec![0.0; 4];
    factor
        .solve_many_into(&rhs, &mut solution, 2)
        .expect("facade QR multi-RHS solve");
    assert!(solution.iter().all(|value| value.is_finite()));

    let mut rejected = vec![5.0; 4];
    assert!(matches!(
        factor.solve_many_into(&[1.0; 3], &mut rejected, 2),
        Err(DenseDirectError::RhsDimension {
            expected: 8,
            actual: 3,
        })
    ));
    assert_eq!(rejected, vec![5.0; 4]);
}
