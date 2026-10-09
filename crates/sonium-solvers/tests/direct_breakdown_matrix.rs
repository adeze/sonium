//! Direct-factor breakdown and failure-exit matrix.

use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, SparseDirectConfig,
    SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn assert_rejected_without_panic(label: &str, operation: impl FnOnce() -> bool) {
    let result = catch_unwind(AssertUnwindSafe(operation));
    assert!(result.is_ok(), "{label}: breakdown path panicked");
    assert!(
        result.expect("checked above"),
        "{label}: invalid factorization succeeded"
    );
}

#[test]
fn singular_and_non_spd_dense_and_sparse_factors_reject() {
    let singular_dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    assert_rejected_without_panic("dense singular LU", || {
        match factorize_dense(
            &singular_dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        ) {
            Err(_) => true,
            Ok(factor) => factor.solve(&[1.0, 2.0]).is_err(),
        }
    });

    let non_spd_dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
    assert_rejected_without_panic("dense non-SPD Cholesky", || {
        factorize_dense(
            &non_spd_dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                method: DenseDirectMethod::Cholesky,
                ..Default::default()
            },
        )
        .is_err()
    });
    let rank_deficient_qr =
        DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_rejected_without_panic("dense rank-deficient QR", || {
        factorize_dense(
            &rank_deficient_qr,
            &DenseDirectConfig {
                matrix_class: MatrixClass::RectangularFullRank,
                method: DenseDirectMethod::Qr,
                ..Default::default()
            },
        )
        .is_err()
    });

    let singular_sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)],
    );
    assert_rejected_without_panic("sparse singular LU", || {
        match factorize_sparse(
            &singular_sparse,
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ..Default::default()
            },
            None,
        ) {
            Err(_) => true,
            Ok(factor) => factor.solve(&[1.0, 2.0]).is_err(),
        }
    });

    let non_spd_sparse = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)],
    );
    assert_rejected_without_panic("sparse non-SPD Cholesky", || {
        factorize_sparse(
            &non_spd_sparse,
            &SparseDirectConfig {
                matrix_class: MatrixClass::SymmetricPositiveDefinite,
                method: SparseDirectMethod::SimplicialCholesky,
                ..Default::default()
            },
            None,
        )
        .is_err()
    });
}
