use num_complex::Complex64;
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use sonium_solvers_direct::{SparseLuError, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn pivoted_system() -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
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
    let expected = vec![1.0, -2.0, 0.5];
    let rhs = matrix.matvec(&expected);
    (matrix, rhs, expected)
}

#[test]
fn simplicial_sparse_lu_reconstructs_pivoted_system() {
    let (matrix, rhs, expected) = pivoted_system();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    let solution = factor.solve(&rhs).expect("solve");

    for (actual, expected) in solution.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-10);
    }
    assert_eq!(factor.dimension(), 3);
    assert!(factor.factor_nonzeros() > 0);
}

#[test]
fn supernodal_sparse_lu_and_preprocessing_match_reference() {
    let (matrix, rhs, expected) = pivoted_system();
    for factor in [
        SparseLuFactorization::factorize_supernodal(&matrix).expect("supernodal factorization"),
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
            .expect("preprocessed factorization"),
    ] {
        let solution = factor.solve(&rhs).expect("solve");
        for (actual, expected) in solution.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}

#[test]
fn sparse_lu_reports_dimension_and_singularity_failures() {
    let (matrix, _rhs, _) = pivoted_system();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    assert!(matches!(
        factor.solve(&[1.0, 2.0]),
        Err(SparseLuError::DimensionMismatch { .. })
    ));

    let singular = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)],
    );
    assert!(matches!(
        SparseLuFactorization::factorize(&singular),
        Err(SparseLuError::SingularMatrix(_))
    ));

    let error = factor
        .solve_refined(&matrix, &[1.0, 2.0, 3.0], f64::NAN, 2)
        .expect_err("non-finite refinement tolerance must be rejected");
    assert!(matches!(
        error,
        SparseLuError::InvalidRefinementTolerance(_)
    ));
}

#[test]
fn preconditioner_mode_accepts_and_solves_tiny_f64_pivots() {
    // `factorize_for_preconditioner` deliberately relaxes the direct-solver
    // pivot guard so an outer iterative method can certify the result.  The
    // solve path must preserve that admission instead of applying the direct
    // solver's absolute 1e-30 diagonal floor.
    let pivot = 1.0e-31_f64;
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, pivot)]);
    let factor = SparseLuFactorization::factorize_for_preconditioner(&matrix)
        .expect("preconditioner factorization should admit tiny pivot");

    let solution = factor
        .solve(&[1.0])
        .expect("preconditioner solve should retain tiny admitted pivot");
    assert!((solution[0] - 1.0 / pivot).abs() / (1.0 / pivot) < 1.0e-12);
}

#[test]
fn sparse_lu_rejects_non_finite_rhs_before_solving() {
    let (matrix, _, _) = pivoted_system();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factorization");
    let error = factor
        .solve(&[1.0, f64::NAN, 3.0])
        .expect_err("non-finite RHS must be rejected");
    assert!(matches!(error, SparseLuError::NonFiniteRhs(1)));
}

#[test]
fn sparse_lu_transpose_and_hermitian_reuse_all_factor_classes() {
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
    let expected = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.75, 2.0),
    ];
    let action = |hermitian: bool| {
        let mut rhs = vec![Complex64::new(0.0, 0.0); 3];
        for &(row, col, value) in &entries {
            rhs[col] += if hermitian { value.conj() } else { value } * expected[row];
        }
        rhs
    };
    let transpose_rhs = action(false);
    let hermitian_rhs = action(true);

    let factors = vec![
        SparseLuFactorization::factorize(&matrix).unwrap(),
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
        SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true).unwrap(),
    ];
    for factor in factors {
        assert_eq!(factor.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(factor.hermitian_capability(), AdjointCapability::Exact);
        let transpose = factor.solve_transpose(&transpose_rhs).unwrap();
        let hermitian = factor.solve_hermitian(&hermitian_rhs).unwrap();
        let transpose_checked = factor.try_apply_transpose(&transpose_rhs).unwrap();
        let hermitian_checked = factor.try_apply_hermitian(&hermitian_rhs).unwrap();
        for i in 0..expected.len() {
            assert!((transpose[i] - expected[i]).norm() < 1e-10);
            assert!((transpose_checked[i] - expected[i]).norm() < 1e-10);
            assert!((hermitian[i] - expected[i]).norm() < 1e-10);
            assert!((hermitian_checked[i] - expected[i]).norm() < 1e-10);
        }
    }
}

fn transpose_matvec(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; matrix.num_cols];
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            y[matrix.col_indices[index]] += matrix.values[index] * x[row];
        }
    }
    y
}

#[test]
fn preprocessed_sparse_lu_transpose_and_hermitian_match_reference() {
    // The diagonal scaling spans six orders of magnitude, forcing the
    // equilibration path to exercise both row/column scaling and the COLAMD
    // permutations.  The transpose solve must undo those transforms in the
    // opposite order; forward-only coverage cannot detect a swapped scale.
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0e-3, -1.0e3, 4.0, -1.0e-3, -1.0e3, 4.0],
    );
    let expected = [1.2, -0.7, 2.3];
    let rhs = transpose_matvec(&matrix, &expected);
    let residual = |actual: &[f64]| {
        transpose_matvec(&matrix, actual)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    };

    let simplicial = SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true)
        .expect("preprocessed simplicial factorization");
    let supernodal =
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
            .expect("preprocessed supernodal factorization");

    for factor in [&simplicial, &supernodal] {
        let transpose = factor.solve_transpose(&rhs).expect("transpose solve");
        let hermitian = factor.solve_hermitian(&rhs).expect("Hermitian solve");
        assert!(
            residual(&transpose) < 1.0e-10,
            "transpose residual: {}",
            residual(&transpose)
        );
        assert!(
            residual(&hermitian) < 1.0e-10,
            "Hermitian residual: {}",
            residual(&hermitian)
        );
        for (actual, expected) in transpose.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in hermitian.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}
