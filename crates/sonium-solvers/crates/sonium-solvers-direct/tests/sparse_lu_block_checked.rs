use num_complex::Complex64;
use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn system() -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 1, 2.0),
            (0, 2, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 2.0),
        ],
    );
    let first = vec![1.0, -2.0, 0.5];
    (matrix.clone(), matrix.matvec(&first), first)
}

#[test]
fn sparse_lu_many_rhs_matches_single_rhs_and_preserves_destination_on_rejection() {
    let (matrix, first_rhs, first_solution) = system();
    let second_solution = vec![2.0, 1.0, -1.0];
    let second_rhs = matrix.matvec(&second_solution);
    let b = vec![
        first_rhs[0],
        second_rhs[0],
        first_rhs[1],
        second_rhs[1],
        first_rhs[2],
        second_rhs[2],
    ];
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");

    let mut x = vec![9.0; 6];
    factor
        .solve_many_into(&b, &mut x, 2)
        .expect("two RHS solve");
    assert_eq!(x, vec![1.0, 2.0, -2.0, 1.0, 0.5, -1.0]);
    assert_eq!(factor.solve(&first_rhs).unwrap(), first_solution);

    let before = x.clone();
    let mut non_finite_rhs = b.clone();
    non_finite_rhs[4] = f64::NAN;
    assert!(matches!(
        factor.solve_many_into(&non_finite_rhs, &mut x, 2),
        Err(SparseLuError::NonFiniteRhs(4))
    ));
    assert_eq!(x, before);
}

#[test]
fn sparse_lu_many_rhs_rejects_shape_mismatch_and_overflow() {
    let (matrix, _, _) = system();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    assert!(matches!(
        factor.solve_many(&[1.0, 2.0], 1),
        Err(SparseLuError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        factor.solve_many(&[], usize::MAX),
        Err(SparseLuError::SizeOverflow {
            dimension: 3,
            nrhs: usize::MAX
        })
    ));
}

#[test]
fn sparse_lu_many_rhs_preserves_destination_when_later_solve_breaks_down() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0e-15), (1, 1, 2.0)]);
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");

    // The first RHS is benign. The second is finite at the API boundary but
    // overflows while dividing by the tiny first pivot, forcing a typed
    // numerical-breakdown return after the first column has been solved.
    let rhs = vec![1.0, f64::MAX, 2.0, 0.0];
    let mut destination = vec![41.0; 4];
    let before = destination.clone();
    assert!(matches!(
        factor.solve_many_into(&rhs, &mut destination, 2),
        Err(SparseLuError::NumericalBreakdown { .. })
    ));
    assert_eq!(destination, before);
}

#[test]
fn sparse_lu_adjoint_many_rhs_matches_single_rhs_and_preserves_destination() {
    let entries = vec![
        (0, 0, Complex64::new(4.0, 1.0)),
        (0, 1, Complex64::new(1.0, -2.0)),
        (0, 2, Complex64::new(0.5, 0.25)),
        (1, 0, Complex64::new(2.0, 0.5)),
        (1, 1, Complex64::new(5.0, -1.0)),
        (1, 2, Complex64::new(-1.0, 0.75)),
        (2, 0, Complex64::new(0.25, -0.5)),
        (2, 1, Complex64::new(-1.0, 2.0)),
        (2, 2, Complex64::new(3.0, 0.25)),
    ];
    let matrix = CsrMatrix::from_triplets(3, 3, entries.clone());
    let first = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.75, 2.0),
    ];
    let second = vec![
        Complex64::new(-0.25, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(1.25, 0.0),
    ];
    let make_rhs = |solution: &[Complex64], hermitian: bool| {
        let mut rhs = vec![Complex64::new(0.0, 0.0); 3];
        for &(row, col, value) in &entries {
            rhs[col] += if hermitian { value.conj() } else { value } * solution[row];
        }
        rhs
    };
    let transpose_first = make_rhs(&first, false);
    let transpose_second = make_rhs(&second, false);
    let hermitian_first = make_rhs(&first, true);
    let hermitian_second = make_rhs(&second, true);
    let transpose_rhs = vec![
        transpose_first[0],
        transpose_second[0],
        transpose_first[1],
        transpose_second[1],
        transpose_first[2],
        transpose_second[2],
    ];
    let hermitian_rhs = vec![
        hermitian_first[0],
        hermitian_second[0],
        hermitian_first[1],
        hermitian_second[1],
        hermitian_first[2],
        hermitian_second[2],
    ];

    for factor in [
        SparseLuFactorization::factorize(&matrix).unwrap(),
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
    ] {
        let mut transpose = vec![Complex64::new(9.0, -3.0); 6];
        factor
            .solve_transpose_many_into(&transpose_rhs, &mut transpose, 2)
            .unwrap();
        let mut hermitian = vec![Complex64::new(9.0, -3.0); 6];
        factor
            .solve_hermitian_many_into(&hermitian_rhs, &mut hermitian, 2)
            .unwrap();
        for row in 0..3 {
            assert!((transpose[row * 2] - first[row]).norm() < 1.0e-10);
            assert!((transpose[row * 2 + 1] - second[row]).norm() < 1.0e-10);
            assert!((hermitian[row * 2] - first[row]).norm() < 1.0e-10);
            assert!((hermitian[row * 2 + 1] - second[row]).norm() < 1.0e-10);
        }

        let mut rejected = vec![Complex64::new(7.0, 2.0); 6];
        let mut non_finite_rhs = transpose_rhs.clone();
        non_finite_rhs[4] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(
            factor.solve_transpose_many_into(&non_finite_rhs, &mut rejected, 2),
            Err(SparseLuError::NonFiniteRhs(4))
        ));
        assert!(
            rejected
                .iter()
                .all(|value| *value == Complex64::new(7.0, 2.0))
        );

        let mut hermitian_non_finite_rhs = hermitian_rhs.clone();
        hermitian_non_finite_rhs[1] = Complex64::new(0.0, f64::NAN);
        let mut hermitian_rejected = vec![Complex64::new(8.0, -4.0); 6];
        assert!(matches!(
            factor
                .solve_hermitian_many_into(&hermitian_non_finite_rhs, &mut hermitian_rejected, 2,),
            Err(SparseLuError::NonFiniteRhs(1))
        ));
        assert!(
            hermitian_rejected
                .iter()
                .all(|value| *value == Complex64::new(8.0, -4.0))
        );

        assert_eq!(
            factor.solve_transpose_many(&transpose_rhs, 2).unwrap(),
            transpose
        );
        assert_eq!(
            factor.solve_hermitian_many(&hermitian_rhs, 2).unwrap(),
            hermitian
        );
    }
}
