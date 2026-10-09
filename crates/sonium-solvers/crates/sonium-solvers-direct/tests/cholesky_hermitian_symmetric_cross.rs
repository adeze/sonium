//! Hermitian vs complex-symmetric Cholesky differential test.
//!
//! The Cholesky factor/TRSM/trailing-update/solve paths thread a
//! `hermitian: bool` flag whose only effect is conjugation. A single missed
//! `.conj()` still yields finite output, so finiteness checks cannot catch
//! the mix-up; only a differential test against dense LU on inputs where the
//! two factorizations genuinely differ can.

use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{cholesky_factorize, cholesky_factorize_hermitian, lu_factorize};

const N: usize = 4;

/// Fixed nonsymmetric complex generator with nonzero imaginary parts on the
/// `A^T A` diagonal (needed for the Hermitian-rejects-symmetric control).
fn generator() -> DenseMatrix<Complex64> {
    DenseMatrix::from_row_major_vec(
        N,
        N,
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.5, -1.0),
            Complex64::new(-1.5, 0.25),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.75, -0.5),
            Complex64::new(-1.0, 1.5),
            Complex64::new(0.25, 1.25),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.5, -1.5),
            Complex64::new(0.5, 0.5),
            Complex64::new(1.75, -0.75),
            Complex64::new(0.0, -2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.5, 1.0),
        ],
    )
}

fn mat_mul_conj_transpose(a: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
    let mut out = DenseMatrix::zeros(N, N);
    for i in 0..N {
        for j in 0..N {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..N {
                acc += a[[k, i]].conj() * a[[k, j]];
            }
            out[[i, j]] = acc
                + if i == j {
                    Complex64::new(N as f64, 0.0)
                } else {
                    acc * 0.0
                };
        }
    }
    out
}

fn mat_mul_transpose(a: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
    let mut out = DenseMatrix::zeros(N, N);
    for i in 0..N {
        for j in 0..N {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..N {
                acc += a[[k, i]] * a[[k, j]];
            }
            out[[i, j]] = acc
                + if i == j {
                    Complex64::new(N as f64, 0.0)
                } else {
                    acc * 0.0
                };
        }
    }
    out
}

fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn hermitian_cholesky_matches_lu_on_hermitian_input() {
    let hermitian = mat_mul_conj_transpose(&generator());
    for i in 0..N {
        for j in 0..N {
            let expected = hermitian[[j, i]].conj();
            assert!(
                (hermitian[[i, j]] - expected).norm() < 1e-12,
                "fixture must be Hermitian at ({i},{j})"
            );
        }
    }
    let rhs = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.25, -1.0),
        Complex64::new(3.0, 0.0),
    ];
    let via_cholesky = cholesky_factorize_hermitian(&hermitian)
        .expect("Hermitian input must factorize")
        .solve(&rhs)
        .expect("Hermitian solve must succeed");
    let via_lu = lu_factorize(&hermitian)
        .expect("LU must factorize Hermitian input")
        .solve(&rhs)
        .expect("LU solve must succeed");
    assert!(
        max_abs_diff(&via_cholesky, &via_lu) < 1e-10,
        "Hermitian Cholesky disagrees with LU: {via_cholesky:?} vs {via_lu:?}"
    );
}

#[test]
fn symmetric_cholesky_matches_lu_on_complex_symmetric_input() {
    let symmetric = mat_mul_transpose(&generator());
    for i in 0..N {
        for j in 0..N {
            assert!(
                (symmetric[[i, j]] - symmetric[[j, i]]).norm() < 1e-12,
                "fixture must be complex-symmetric at ({i},{j})"
            );
        }
    }
    assert!(
        (symmetric[[0, 1]] - symmetric[[1, 0]].conj()).norm() > 1e-6,
        "fixture must NOT be Hermitian or the differential is vacuous"
    );
    let rhs = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(0.0, 2.0),
        Complex64::new(-0.25, 1.0),
        Complex64::new(2.0, 1.0),
    ];
    let via_cholesky = cholesky_factorize(&symmetric)
        .expect("complex-symmetric input must factorize")
        .solve(&rhs)
        .expect("symmetric solve must succeed");
    let via_lu = lu_factorize(&symmetric)
        .expect("LU must factorize symmetric input")
        .solve(&rhs)
        .expect("LU solve must succeed");
    assert!(
        max_abs_diff(&via_cholesky, &via_lu) < 1e-10,
        "symmetric Cholesky disagrees with LU: {via_cholesky:?} vs {via_lu:?}"
    );
}

#[test]
fn hermitian_factorization_rejects_complex_symmetric_input() {
    // The (0,0) diagonal entry of A^T A + nI has nonzero imaginary part, so
    // the Hermitian diagonal screen must fire: silently accepting would mean
    // the `hermitian` flag no longer discriminates the two code paths.
    let symmetric = mat_mul_transpose(&generator());
    assert!(
        symmetric[[0, 0]].im.abs() > 1e-6,
        "fixture diagonal must carry imaginary part"
    );
    let rejected = cholesky_factorize_hermitian(&symmetric).is_err();
    assert!(
        rejected,
        "Hermitian factorization must reject complex-symmetric input"
    );
}
