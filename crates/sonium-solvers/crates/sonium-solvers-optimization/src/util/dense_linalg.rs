//! Small dense linear algebra helpers for optimisation.
//!
//! Cholesky factorisation, symmetric positive-definite solve, and modified
//! Cholesky (Gill-Murray-Wright). These operate on `sonium_solvers_core::dense::DenseMatrix<T>` and are used
//! by BFGS, SQP, and trust-region methods where the matrix dimension equals
//! the optimisation variable count.

use crate::types::RealScalar;

/// Cholesky factorisation of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L` such that `A = L * L^T`.
/// Returns `None` if the matrix is not positive definite, including when it
/// is not square (a non-square input cannot be positive definite, so both
/// failure modes share the `None` case by design).
pub fn cholesky<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
) -> Option<sonium_solvers_core::dense::DenseMatrix<T>> {
    let n = a.nrows();
    if n != a.ncols() {
        return None;
    }
    let mut l = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);

    for j in 0..n {
        let mut sum = T::zero();
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= T::zero() {
            return None;
        }
        l[[j, j]] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum = T::zero();
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
        }
    }
    Some(l)
}

/// Solve `L * y = b` where `L` is lower-triangular (forward substitution).
pub fn solve_lower_triangular<T: RealScalar>(
    l: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &[T],
) -> Vec<T> {
    let n = b.len();
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..i {
            sum += l[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / l[[i, i]];
    }
    y
}

/// Solve `L^T * x = y` where `L` is lower-triangular (back substitution).
pub fn solve_lower_triangular_transpose<T: RealScalar>(
    l: &sonium_solvers_core::dense::DenseMatrix<T>,
    y: &[T],
) -> Vec<T> {
    let n = y.len();
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }
    x
}

/// Solve `A * x = b` for symmetric positive-definite `A` via Cholesky.
///
/// Returns `None` if `A` is not positive definite.
pub fn solve_spd<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &[T],
) -> Option<Vec<T>> {
    let l = cholesky(a)?;
    let y = solve_lower_triangular(&l, b);
    Some(solve_lower_triangular_transpose(&l, &y))
}

/// Modified Cholesky factorisation (Gill-Murray-Wright).
///
/// Computes `L` such that `A + E = L * L^T` where `E` is a non-negative
/// diagonal correction that makes the modified matrix positive definite.
/// Returns `Some((L, diagonal_correction))`, or `None` if `A` is not square
/// (a non-square input cannot be factorised, mirroring [`cholesky`]).
pub fn modified_cholesky<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
) -> Option<(sonium_solvers_core::dense::DenseMatrix<T>, Vec<T>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return None;
    }

    // Estimate: delta = max(eps * ||A||, eps)
    let eps = T::machine_epsilon();
    let mut a_norm = T::zero();
    for i in 0..n {
        for j in 0..n {
            let v = a[[i, j]].abs();
            if v > a_norm {
                a_norm = v;
            }
        }
    }
    let delta = eps * T::max(a_norm, T::one());
    let beta_sq = T::max(a_norm, T::one());

    let mut c = a.clone();
    let mut l = sonium_solvers_core::dense::DenseMatrix::zeros(n, n);
    let mut e = vec![T::zero(); n];

    for j in 0..n {
        // Compute column j of L
        let mut sum = T::zero();
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let mut cjj = c[[j, j]] - sum;

        // Find maximum off-diagonal magnitude in column j
        let mut theta = T::zero();
        for i in (j + 1)..n {
            let mut s = T::zero();
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            c[[i, j]] = a[[i, j]] - s;
            let v = c[[i, j]].abs();
            if v > theta {
                theta = v;
            }
        }

        // Ensure positive definiteness
        let correction = T::max(T::zero(), T::max(delta, theta * theta / beta_sq) - cjj);
        cjj += correction;
        // Guard against floating-point rounding producing a tiny negative value
        let guard_extra = T::max(T::zero(), T::machine_epsilon() - cjj);
        cjj += guard_extra;
        e[j] = correction + guard_extra;

        l[[j, j]] = cjj.sqrt();

        for i in (j + 1)..n {
            l[[i, j]] = c[[i, j]] / l[[j, j]];
        }
    }

    Some((l, e))
}

/// Solve `(A + E) * x = b` using modified Cholesky.
///
/// Guarantees a solution even when `A` is indefinite or singular. Returns
/// `None` if `A` is not square or `b` has the wrong length.
pub fn solve_modified_cholesky<T: RealScalar>(
    a: &sonium_solvers_core::dense::DenseMatrix<T>,
    b: &[T],
) -> Option<(Vec<T>, Vec<T>)> {
    let (l, e) = modified_cholesky(a)?;
    if b.len() != a.nrows() {
        return None;
    }
    let y = solve_lower_triangular(&l, b);
    let x = solve_lower_triangular_transpose(&l, &y);
    Some((x, e))
}

/// Symmetric rank-1 update: `A += alpha * v * v^T` (in-place).
pub fn symmetric_rank1_update<T: RealScalar>(
    a: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    alpha: T,
    v: &[T],
) {
    let n = v.len();
    for i in 0..n {
        for j in 0..=i {
            let update = alpha * v[i] * v[j];
            a[[i, j]] += update;
            if i != j {
                a[[j, i]] += update;
            }
        }
    }
}

/// Symmetric rank-2 update: `A += alpha * (u * v^T + v * u^T)` (in-place).
pub fn symmetric_rank2_update<T: RealScalar>(
    a: &mut sonium_solvers_core::dense::DenseMatrix<T>,
    alpha: T,
    u: &[T],
    v: &[T],
) {
    let n = u.len();
    for i in 0..n {
        for j in 0..=i {
            let update = alpha * (u[i] * v[j] + v[i] * u[j]);
            a[[i, j]] += update;
            if i != j {
                a[[j, i]] += update;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cholesky_2x2() {
        let a = sonium_solvers_core::array![[4.0, 2.0], [2.0, 5.0]];
        let l = cholesky(&a).unwrap();
        // Verify L * L^T == A
        let lt = l.t();
        let reconstructed = l.dot(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_cholesky_not_pd() {
        let a = sonium_solvers_core::array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(cholesky(&a).is_none());
    }

    #[test]
    fn test_solve_spd() {
        let a = sonium_solvers_core::array![[4.0, 2.0], [2.0, 5.0]];
        let b = sonium_solvers_core::array![1.0, 2.0];
        let x = solve_spd(&a, &b).unwrap();
        // Verify A * x == b
        let ax = a.dot(&x);
        assert_relative_eq!(ax[0], b[0], epsilon = 1e-12);
        assert_relative_eq!(ax[1], b[1], epsilon = 1e-12);
    }

    #[test]
    fn test_modified_cholesky_pd() {
        // Already PD: correction should be zero or near-zero
        let a = sonium_solvers_core::array![[4.0, 1.0], [1.0, 3.0]];
        let (l, e) = modified_cholesky(&a).expect("PD fixture must factorise");
        let lt = l.t();
        let reconstructed = l.dot(&lt);
        // A + E == L * L^T
        for i in 0..2 {
            for j in 0..2 {
                let aij = a[[i, j]] + if i == j { e[i] } else { 0.0 };
                assert_relative_eq!(reconstructed[[i, j]], aij, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_modified_cholesky_indefinite() {
        // Indefinite matrix: eigenvalues of different signs
        let a = sonium_solvers_core::array![[1.0_f64, 2.0], [2.0, 1.0]];
        let (l, e) = modified_cholesky(&a).expect("GMW must regularise indefinite input");
        // Verify the factorisation succeeds and L is lower-triangular
        assert!(l[[0, 0]] > 0.0);
        assert!(l[[1, 1]] > 0.0);
        assert!(l[[0, 1]].abs() < 1e-15);
        // Verify (A + diag(e)) = L * L^T
        let lt = l.t();
        let llt = l.dot(&lt);
        for i in 0..2 {
            for j in 0..2 {
                let aij = a[[i, j]] + if i == j { e[i] } else { 0.0 };
                assert_relative_eq!(llt[[i, j]], aij, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_modified_cholesky_rejects_non_square() {
        // Non-square inputs return None instead of panicking (the old
        // `assert_eq!(n, ncols)` aborted the process on caller shape bugs).
        let wide = sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(2, 3, vec![1.0; 6]);
        assert!(modified_cholesky(&wide).is_none());
        assert!(cholesky(&wide).is_none());
        assert!(
            solve_modified_cholesky(&wide, &[1.0, 2.0]).is_none(),
            "non-square system must not solve"
        );
        let square =
            sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(2, 2, vec![1.0; 4]);
        assert!(
            solve_modified_cholesky(&square, &[1.0]).is_none(),
            "wrong-length RHS must not solve"
        );
    }

    #[test]
    fn test_symmetric_rank1_update() {
        let mut a = sonium_solvers_core::dense::DenseMatrix::<f64>::eye(3);
        let v = sonium_solvers_core::array![1.0, 2.0, 3.0];
        symmetric_rank1_update(&mut a, 1.0, &v);
        // A should be I + v * v^T
        assert_relative_eq!(a[[0, 0]], 2.0, epsilon = 1e-12);
        assert_relative_eq!(a[[1, 1]], 5.0, epsilon = 1e-12);
        assert_relative_eq!(a[[2, 2]], 10.0, epsilon = 1e-12);
        assert_relative_eq!(a[[0, 1]], 2.0, epsilon = 1e-12);
        assert_relative_eq!(a[[1, 0]], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cholesky_3x3() {
        let a =
            sonium_solvers_core::array![[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]];
        let l = cholesky(&a).unwrap();
        let llt = l.dot(&l.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(llt[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_lower_triangular() {
        // L = [[2,0],[3,4]], b = [2, 11]
        // Forward substitution: y0 = 2/2 = 1, y1 = (11 - 3*1)/4 = 8/4 = 2
        let l = sonium_solvers_core::array![[2.0, 0.0], [3.0, 4.0]];
        let b = sonium_solvers_core::array![2.0, 11.0];
        let y = solve_lower_triangular(&l, &b);
        assert_relative_eq!(y[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(y[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_solve_lower_triangular_transpose() {
        // L = [[2,0],[3,4]], so L^T = [[2,3],[0,4]]
        // Solve L^T x = y where y = [1, 2]
        // Back substitution: x1 = 2/4 = 0.5, x0 = (1 - 3*0.5)/2 = -0.5/2 = -0.25
        let l = sonium_solvers_core::array![[2.0, 0.0], [3.0, 4.0]];
        let y = sonium_solvers_core::array![1.0, 2.0];
        let x = solve_lower_triangular_transpose(&l, &y);
        assert_relative_eq!(x[0], -0.25, epsilon = 1e-12);
        assert_relative_eq!(x[1], 0.5, epsilon = 1e-12);
        // Verify L^T * x == y
        let lt = l.t().to_owned();
        let ltx = lt.dot(&x);
        assert_relative_eq!(ltx[0], y[0], epsilon = 1e-12);
        assert_relative_eq!(ltx[1], y[1], epsilon = 1e-12);
    }

    #[test]
    fn test_symmetric_rank2_update() {
        // A = I (2x2), u = [1,0], v = [0,1], alpha = 1
        // update = alpha * (u*v^T + v*u^T) = [[0,1],[1,0]]
        // A_new = I + [[0,1],[1,0]] = [[1,1],[1,1]]
        let mut a = sonium_solvers_core::dense::DenseMatrix::<f64>::eye(2);
        let u = sonium_solvers_core::array![1.0, 0.0];
        let v = sonium_solvers_core::array![0.0, 1.0];
        symmetric_rank2_update(&mut a, 1.0, &u, &v);
        assert_relative_eq!(a[[0, 0]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[[0, 1]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[[1, 0]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[[1, 1]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_solve_modified_cholesky() {
        // Indefinite matrix: [[1,2],[2,1]] (eigenvalues -1 and 3)
        // solve_modified_cholesky guarantees a solution even for indefinite A.
        let a = sonium_solvers_core::array![[1.0_f64, 2.0], [2.0, 1.0]];
        let b = sonium_solvers_core::array![1.0, 2.0];
        let (x, e) = solve_modified_cholesky(&a, &b).expect("indefinite fixture must solve");
        // The correction should be non-trivial since A is indefinite
        assert!(
            e.iter().any(|&ei| ei > 0.0),
            "Expected non-zero correction for indefinite matrix"
        );
        // x should be finite and non-zero (a meaningful solution was produced)
        assert!(
            x.iter().all(|&xi| xi.is_finite()),
            "Solution contains non-finite values"
        );
        assert!(
            x.iter().any(|&xi| xi.abs() > 1e-15),
            "Solution is trivially zero"
        );
        // Verify the factorisation itself is consistent: (A+E) = L*L^T
        let (l, e2) = modified_cholesky(&a).expect("test fixture is square");
        let llt = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                let aij = a[[i, j]] + if i == j { e2[i] } else { 0.0 };
                assert_relative_eq!(llt[[i, j]], aij, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_2x2_f32() {
        let a: sonium_solvers_core::dense::DenseMatrix<f32> =
            sonium_solvers_core::array![[4.0_f32, 2.0_f32], [2.0_f32, 5.0_f32]];
        let l = cholesky(&a).unwrap();
        let lt = l.t();
        let reconstructed = l.dot(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-4,
                    "mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_solve_spd_f32() {
        let a: sonium_solvers_core::dense::DenseMatrix<f32> =
            sonium_solvers_core::array![[4.0_f32, 2.0_f32], [2.0_f32, 5.0_f32]];
        let b: Vec<f32> = sonium_solvers_core::array![1.0_f32, 2.0_f32];
        let x = solve_spd(&a, &b).unwrap();
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-4, "residual[0]={}", ax[0] - b[0]);
        assert!((ax[1] - b[1]).abs() < 1e-4, "residual[1]={}", ax[1] - b[1]);
    }
}
