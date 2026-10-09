//! Parity between the optimizer-local unblocked Cholesky
//! (`util::dense_linalg::cholesky`) and the blocked, scale-aware
//! `sonium-solvers-direct` factorization.
//!
//! The local kernel exists because the optimizer works on `RealScalar`
//! (f32/f64, no complex) Hessians; this test pins the two implementations
//! together so their `Option`/`NotPositiveDefinite` semantics cannot drift
//! apart silently.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::cholesky_factorize;
use sonium_solvers_optimization::util::dense_linalg::cholesky;

/// SPD fixture: diagonally dominant with fixed off-diagonal pattern.
fn spd_fixture(n: usize) -> DenseMatrix<f64> {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = if i == j {
                4.0 + (i as f64) * 0.5
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            };
        }
    }
    DenseMatrix::from_row_major_vec(n, n, data)
}

#[test]
fn local_cholesky_matches_direct_factorization() {
    for n in [1, 2, 5, 16] {
        let a = spd_fixture(n);
        let local = cholesky(&a).expect("SPD fixture must factorise locally");
        let direct = cholesky_factorize(&a).expect("SPD fixture must factorise directly");
        let reference = direct.factor();
        for i in 0..n {
            for j in 0..=i {
                assert!(
                    (local[[i, j]] - reference[[i, j]]).abs() < 1.0e-12,
                    "L[{i},{j}] drift at n={n}: {} vs {}",
                    local[[i, j]],
                    reference[[i, j]]
                );
            }
        }
    }
}

#[test]
fn local_and_direct_cholesky_agree_on_indefiniteness() {
    // Indefinite (eigenvalues -1 and 3): both must refuse.
    let a = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
    assert!(cholesky(&a).is_none());
    assert!(cholesky_factorize(&a).is_err());
    // Non-square: both must refuse (None vs typed error, never a panic).
    let wide = DenseMatrix::from_row_major_vec(2, 3, vec![1.0; 6]);
    assert!(cholesky(&wide).is_none());
    assert!(cholesky_factorize(&wide).is_err());
}
