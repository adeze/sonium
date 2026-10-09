//! ε-rank truncation for off-diagonal frontal-matrix blocks.
//!
//! Randomized range finding with a robust RRQR fallback. For a block `B` of shape `(m × n)`,
//! returns factors `(U, V)` with `B ≈ U · V^T`, where `U` is `(m × r)` and
//! `V` is `(n × r)`. The numerical rank `r` is the smallest integer such that
//! `|R_{kk}| ≤ ε · |R_{00}|` for all `k ≥ r`.
//!
//! Large compression candidates use an adaptive randomized range finder with
//! one power iteration and deterministic sampling. Its geometric sample-size
//! growth costs O(m·n·r) for numerical rank r. Small blocks and exact
//! (`epsilon = 0`) factorizations retain standard column-pivoted Householder
//! QR with an explicit
//! `v_head` cache so we can rebuild `Q[:, :r]` cleanly after truncation. The
//! column squared-norms used for pivot selection are recomputed each step
//! (O(m·n²) total) — RRQR isn't on the asymptotic critical path of the
//! multifrontal walk, dense GEMM on the front bodies is.

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;

/// Output of a column-pivoted Householder QR with ε-rank truncation.
pub struct TruncatedQr<T: ComplexField> {
    /// Left factor `U`, shape `(m × r)`.
    pub u: DenseMatrix<T>,
    /// Right factor `V`, shape `(n × r)`, satisfying `B ≈ U · V^T`.
    pub v: DenseMatrix<T>,
    /// Numerical rank `r` (may be 0 for an all-zero block).
    pub rank: usize,
    /// RRQR diagonal after pivoting, or the final randomized residual norm.
    pub r_diag: Vec<T::Real>,
}

/// Adaptive randomized range finder with one power iteration.
///
/// Sampling is deterministic so factorization and tests are reproducible.
/// The returned basis is explicitly checked against the requested relative
/// Frobenius-norm tolerance; sample width doubles geometrically until the
/// check passes or the full numerical dimension is reached.
pub fn truncate_randomized<T: ComplexField>(
    block: &DenseMatrix<T>,
    epsilon: T::Real,
) -> TruncatedQr<T> {
    let m = block.nrows();
    let n = block.ncols();
    let max_rank = m.min(n);
    if m == 0 || n == 0 {
        return TruncatedQr {
            u: DenseMatrix::zeros(m, 0),
            v: DenseMatrix::zeros(n, 0),
            rank: 0,
            r_diag: Vec::new(),
        };
    }
    if epsilon <= T::Real::zero() {
        return truncate_rrqr(block, epsilon);
    }

    let norm_sq = block
        .as_slice()
        .iter()
        .fold(T::Real::zero(), |sum, value| sum + value.norm_sqr());
    let norm = norm_sq.sqrt();
    if norm == T::Real::zero() {
        return TruncatedQr {
            u: DenseMatrix::zeros(m, 0),
            v: DenseMatrix::zeros(n, 0),
            rank: 0,
            r_diag: Vec::new(),
        };
    }

    let adjoint = DenseMatrix::from_fn(n, m, |i, j| block[[j, i]].conj());
    let mut sample_width = 8.min(max_rank);
    loop {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64
            ^ (m as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ (n as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
            ^ sample_width as u64;
        let omega = DenseMatrix::from_fn(n, sample_width, |_, _| {
            let re = next_signed_unit(&mut state);
            let im = next_signed_unit(&mut state);
            T::from_re_im(
                T::Real::from_f64(re).unwrap(),
                T::Real::from_f64(im).unwrap(),
            )
        });

        // Y = (A Aᴴ) A Ω: one power iteration improves separation when
        // singular values decay slowly, while every product stays BLAS-3.
        let y0 = block.matmul(&omega);
        let z = adjoint.matmul(&y0);
        let y = block.matmul(&z);
        let q = orthonormal_columns(&y);
        let qh = DenseMatrix::from_fn(q.ncols(), q.nrows(), |i, j| q[[j, i]].conj());
        let coefficients = qh.matmul(block);
        let approximation = q.matmul(&coefficients);
        let residual_sq = block
            .as_slice()
            .iter()
            .zip(approximation.as_slice())
            .fold(T::Real::zero(), |sum, (value, approx)| {
                sum + (*value - *approx).norm_sqr()
            });
        let residual = residual_sq.sqrt();

        if residual <= epsilon * norm {
            return TruncatedQr {
                rank: q.ncols(),
                u: q,
                v: coefficients.transposed(),
                r_diag: vec![residual],
            };
        }
        if sample_width == max_rank {
            // The powered sample can numerically erase a very small singular
            // direction. Never claim the requested tolerance merely because
            // sampling reached full width; fall back to deterministic RRQR.
            return truncate_rrqr(block, epsilon);
        }
        sample_width = (sample_width * 2).min(max_rank);
    }
}

fn next_signed_unit(state: &mut u64) -> f64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
    (bits as f64) * (2.0 / ((1_u64 << 53) as f64)) - 1.0
}

fn orthonormal_columns<T: ComplexField>(input: &DenseMatrix<T>) -> DenseMatrix<T> {
    let m = input.nrows();
    let dependency_tol = T::Real::epsilon().sqrt();
    let mut columns: Vec<Vec<T>> = Vec::with_capacity(input.ncols());
    let mut first_norm = T::Real::zero();
    for j in 0..input.ncols() {
        let mut candidate = input.column_to_vec(j);
        // Twice-applied modified Gram-Schmidt is stable enough for the small
        // sampled range and avoids an O(n²m) pivoted QR.
        for _ in 0..2 {
            for basis in &columns {
                let coefficient = T::vec_dot(basis, &candidate);
                T::vec_axpy(-coefficient, basis, &mut candidate);
            }
        }
        let candidate_norm = T::vec_norm(&candidate);
        if first_norm == T::Real::zero() {
            first_norm = candidate_norm;
        }
        if candidate_norm <= dependency_tol * first_norm {
            continue;
        }
        T::vec_scale(
            &mut candidate,
            T::from_real(T::Real::one() / candidate_norm),
        );
        columns.push(candidate);
    }
    DenseMatrix::from_fn(m, columns.len(), |i, j| columns[j][i])
}

/// Column-pivoted Householder QR with ε-rank truncation.
///
/// `epsilon` is the relative tolerance: the rank cutoff is the smallest `r`
/// such that `‖x_r‖ ≤ ε · max(|R_{kk}|)` where `x_r` is the residual k-th
/// column. Pass `0.0` to keep the full numerical rank (no truncation).
pub fn truncate_rrqr<T: ComplexField>(block: &DenseMatrix<T>, epsilon: T::Real) -> TruncatedQr<T> {
    let m = block.nrows();
    let n = block.ncols();
    let k_max = m.min(n);

    if m == 0 || n == 0 {
        return TruncatedQr {
            u: DenseMatrix::<T>::zeros(m, 0),
            v: DenseMatrix::<T>::zeros(n, 0),
            rank: 0,
            r_diag: Vec::new(),
        };
    }

    // Column-major working copy of B.
    let mut a = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            a[j * m + i] = block[[i, j]];
        }
    }
    let mut col_perm: Vec<usize> = (0..n).collect();

    // Reflector storage: for each step k, we keep
    //   v_heads[k] = v_k  (full complex value)
    //   v_tails[k][..]   = v[k+1..m]  (the column k of `a` below the diagonal,
    //                                 saved before later steps could touch it)
    // Together with taus[k], this fully describes H_k = I − τ_k v_k v_k^H.
    let mut taus: Vec<T> = vec![T::zero(); k_max];
    let mut v_heads: Vec<T> = vec![T::zero(); k_max];
    let mut v_tails: Vec<Vec<T>> = vec![Vec::new(); k_max];

    let zero_real = T::Real::zero();
    let one_real = T::Real::one();
    let two_real = one_real + one_real;

    // Initial column squared-norms.
    let mut col_sqnorms = vec![T::Real::zero(); n];
    for j in 0..n {
        let mut s = zero_real;
        for i in 0..m {
            s += a[j * m + i].norm_sqr();
        }
        col_sqnorms[j] = s;
    }

    let mut r_diag: Vec<T::Real> = Vec::with_capacity(k_max);
    let mut max_r0 = zero_real;
    let mut realised_rank = k_max;

    for k in 0..k_max {
        // 1. Pick pivot column.
        let mut piv = k;
        let mut piv_val = col_sqnorms[k];
        for (j, &sqnorm) in col_sqnorms.iter().enumerate().skip(k + 1) {
            if sqnorm > piv_val {
                piv_val = sqnorm;
                piv = j;
            }
        }
        if piv != k {
            for i in 0..m {
                a.swap(k * m + i, piv * m + i);
            }
            col_perm.swap(k, piv);
            col_sqnorms.swap(k, piv);
        }

        // 2. Compute ||x|| where x = a[k.., k].
        let mut sigma = zero_real;
        for i in k..m {
            sigma += a[k * m + i].norm_sqr();
        }
        let norm_x = sigma.sqrt();
        if k == 0 {
            max_r0 = norm_x;
        }
        if epsilon > zero_real && norm_x <= epsilon * max_r0 {
            realised_rank = k;
            break;
        }

        // 3. Build reflector v = x - α e_k with α = -phase(x_k) · ||x||.
        let xk = a[k * m + k];
        let xk_norm = xk.norm();
        let sign_factor = if xk_norm > T::Real::from_f64(1e-300).unwrap() {
            xk * T::from_real(one_real / xk_norm)
        } else {
            T::one()
        };
        let alpha = -sign_factor * T::from_real(norm_x);
        let v_head = xk - alpha;

        let mut vtv = v_head.norm_sqr();
        for i in (k + 1)..m {
            vtv += a[k * m + i].norm_sqr();
        }
        let tau = if vtv > T::Real::from_f64(1e-300).unwrap() {
            T::from_real(two_real / vtv)
        } else {
            T::zero()
        };

        // Save reflector tail BEFORE applying H to subsequent columns. The
        // tail values are a[k*m + (k+1)..k*m + m] — these will not change in
        // later iterations (later H_j only touches columns j > k), so we
        // could read them later too, but copying once is clearer.
        let mut tail = Vec::with_capacity(m - k - 1);
        for i in (k + 1)..m {
            tail.push(a[k * m + i]);
        }

        // 4. Apply H_k to columns j > k: a[:, j] -= τ · v · (v^H · a[:, j]).
        for j in (k + 1)..n {
            let mut dot = v_head.conj() * a[j * m + k];
            for i in (k + 1)..m {
                dot += a[k * m + i].conj() * a[j * m + i];
            }
            let coef = tau * dot;
            a[j * m + k] -= coef * v_head;
            for i in (k + 1)..m {
                let v_i = a[k * m + i];
                a[j * m + i] -= coef * v_i;
            }
            // Refresh column squared-norm for next pivot search.
            let mut s = zero_real;
            for i in (k + 1)..m {
                s += a[j * m + i].norm_sqr();
            }
            col_sqnorms[j] = s;
        }

        // 5. Save R[k, k] and reflector data; clear column-k below-diagonal.
        a[k * m + k] = alpha;
        for i in (k + 1)..m {
            a[k * m + i] = T::zero();
        }
        taus[k] = tau;
        v_heads[k] = v_head;
        v_tails[k] = tail;
        r_diag.push(alpha.norm());
        col_sqnorms[k] = zero_real;
        if alpha.norm() > max_r0 {
            max_r0 = alpha.norm();
        }
    }

    // 6. Reconstruct U = Q[:, :r] by applying stored reflectors to I_{m × r}.
    let r = realised_rank;
    let mut u = DenseMatrix::<T>::zeros(m, r);
    for j in 0..r {
        u[[j, j]] = T::one();
    }
    for k in (0..r).rev() {
        let tau = taus[k];
        if tau.norm() == zero_real {
            continue;
        }
        let v_head = v_heads[k];
        let v_tail = &v_tails[k];
        for j in k..r {
            let mut dot = v_head.conj() * u[[k, j]];
            for (offset, v_i) in v_tail.iter().enumerate() {
                let i = k + 1 + offset;
                dot += v_i.conj() * u[[i, j]];
            }
            let coef = tau * dot;
            u[[k, j]] -= coef * v_head;
            for (offset, v_i) in v_tail.iter().enumerate() {
                let i = k + 1 + offset;
                u[[i, j]] -= coef * *v_i;
            }
        }
    }

    // 7. Build V so that U · V^T = B.
    //
    // After pivoting, column j of `a` is column `col_perm[j]` of the original
    // B reduced by Q^H. So R[i, j] = a[j*m + i] for i ≤ j and i < r.
    // The factorisation is B[:, col_perm[j]] = Q · R[:, j], i.e.
    // B = Q · R · P^T where P is the permutation matrix with P_{col_perm[j], j} = 1.
    // We want U · V^T = B with U = Q[:, :r], so V^T = R[:r, :] · P^T,
    // i.e. V[col_perm[j], i] = R[i, j].
    let mut v_mat = DenseMatrix::<T>::zeros(n, r);
    for j in 0..n {
        let orig = col_perm[j];
        let i_max = r.min(j + 1);
        for i in 0..i_max {
            v_mat[[orig, i]] = a[j * m + i];
        }
    }

    TruncatedQr {
        u,
        v: v_mat,
        rank: r,
        r_diag,
    }
}

/// Compress a dense block to low-rank `(U, V)` only if it actually saves
/// storage. Returns `None` to signal "keep dense".
pub fn maybe_compress<T: ComplexField>(
    block: &DenseMatrix<T>,
    epsilon: T::Real,
) -> Option<(DenseMatrix<T>, DenseMatrix<T>)> {
    let m = block.nrows();
    let n = block.ncols();
    if m == 0 || n == 0 {
        return None;
    }
    // Benchmarked crossover for rank-8 square blocks is around 256 on Apple
    // silicon; below it the allocation and power-iteration overhead outweigh
    // BLAS-3 sampling.
    let qr = if m.min(n) >= 256 && epsilon > T::Real::zero() {
        truncate_randomized(block, epsilon)
    } else {
        truncate_rrqr(block, epsilon)
    };
    if qr.rank * (m + n) < m * n {
        Some((qr.u, qr.v))
    } else {
        None
    }
}

// Silence the unused-Float warning when generic bounds drag it in indirectly.
#[allow(dead_code)]
fn _touch_float<F: Float + FromPrimitive>(_: F) {}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn fro_err_f64(a: &DenseMatrix<f64>, b: &DenseMatrix<f64>) -> f64 {
        assert_eq!(a.shape(), b.shape());
        let mut s = 0.0;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let d = a[[i, j]] - b[[i, j]];
                s += d * d;
            }
        }
        s.sqrt()
    }

    fn fro_norm_f64(a: &DenseMatrix<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                s += a[[i, j]] * a[[i, j]];
            }
        }
        s.sqrt()
    }

    #[test]
    fn rrqr_full_rank_recovers_block() {
        let data: Vec<f64> = (0..24).map(|k| ((k * 7 + 3) % 11) as f64 - 5.0).collect();
        let b = DenseMatrix::<f64>::from_row_major_vec(6, 4, data);
        let qr = truncate_rrqr(&b, 0.0);
        assert_eq!(qr.rank, 4);
        let reconstructed = qr.u.matmul(&qr.v.t());
        let err = fro_err_f64(&b, &reconstructed);
        let nrm = fro_norm_f64(&b);
        assert!(err / nrm < 1e-10, "rel err {:.3e}", err / nrm);
    }

    #[test]
    fn rrqr_low_rank_block_truncates() {
        let m = 6;
        let n = 5;
        let u1: Vec<f64> = (0..m).map(|i| (i + 1) as f64).collect();
        let v1: Vec<f64> = (0..n).map(|j| (j as f64 - 2.0).cos()).collect();
        let u2: Vec<f64> = (0..m).map(|i| (i as f64).sin()).collect();
        let v2: Vec<f64> = (0..n).map(|j| j as f64 + 0.5).collect();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = u1[i] * v1[j] + u2[i] * v2[j];
            }
        }
        let b = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let qr = truncate_rrqr(&b, 1e-10);
        assert_eq!(qr.rank, 2, "rank was {}", qr.rank);
        let reconstructed = qr.u.matmul(&qr.v.t());
        let err = fro_err_f64(&b, &reconstructed);
        let nrm = fro_norm_f64(&b);
        assert!(err / nrm < 1e-8, "rel err {:.3e}", err / nrm);
    }

    #[test]
    fn maybe_compress_keeps_dense_when_not_worth_it() {
        let data: Vec<f64> = (0..16).map(|k| (k as f64 + 0.7).sin()).collect();
        let b = DenseMatrix::<f64>::from_row_major_vec(4, 4, data);
        let out = maybe_compress(&b, 0.0);
        assert!(out.is_none());
    }

    #[test]
    fn rrqr_complex_low_rank() {
        let m = 5;
        let n = 4;
        let mut data = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let u = Complex64::new((i as f64).cos(), (i as f64).sin());
                let v = Complex64::new((j as f64 + 1.0).sqrt(), -(j as f64).cos());
                data[i * n + j] = u * v;
            }
        }
        let b = DenseMatrix::<Complex64>::from_row_major_vec(m, n, data);
        let qr = truncate_rrqr(&b, 1e-10);
        assert_eq!(qr.rank, 1, "rank was {}", qr.rank);
        let reconstructed = qr.u.matmul(&qr.v.t());
        let mut err_sq = 0.0;
        let mut nrm_sq = 0.0;
        for i in 0..m {
            for j in 0..n {
                let d = b[[i, j]] - reconstructed[[i, j]];
                err_sq += d.norm_sqr();
                nrm_sq += b[[i, j]].norm_sqr();
            }
        }
        let rel = (err_sq / nrm_sq).sqrt();
        assert!(rel < 1e-8, "rel err {:.3e}", rel);
    }

    #[test]
    fn randomized_range_finder_is_deterministic_and_accurate() {
        let (m, n, rank) = (96, 80, 5);
        let u = DenseMatrix::from_fn(m, rank, |i, k| {
            (((i * 17 + k * 31 + i * k * 7 + 3) % 101) as f64 - 50.0) / 51.0
        });
        let v = DenseMatrix::from_fn(n, rank, |j, k| {
            (((j * 29 + k * 11 + j * k * 13 + 5) % 103) as f64 - 51.0) / 52.0
        });
        let block = u.matmul(&v.transposed());
        let first = truncate_randomized(&block, 1e-10);
        let second = truncate_randomized(&block, 1e-10);
        assert_eq!(first.rank, rank);
        assert_eq!(second.rank, rank);
        assert_eq!(first.u.as_slice(), second.u.as_slice());
        assert_eq!(first.v.as_slice(), second.v.as_slice());
        let reconstructed = first.u.matmul(&first.v.transposed());
        let relative = fro_err_f64(&block, &reconstructed) / fro_norm_f64(&block);
        assert!(relative < 1e-10, "relative error {relative:e}");
    }

    #[test]
    fn randomized_range_finder_respects_noisy_error_bound() {
        let (m, n, rank) = (72, 64, 4);
        let block = DenseMatrix::from_fn(m, n, |i, j| {
            let low_rank = (0..rank)
                .map(|k| ((i + 3 * k) as f64 * 0.11).sin() * ((j + 5 * k) as f64 * 0.09).cos())
                .sum::<f64>();
            low_rank + 1e-8 * (((i * 43 + j * 19) % 101) as f64 - 50.0)
        });
        let truncated = truncate_randomized(&block, 1e-5);
        assert!(truncated.rank < 16, "rank was {}", truncated.rank);
        let reconstructed = truncated.u.matmul(&truncated.v.transposed());
        let relative = fro_err_f64(&block, &reconstructed) / fro_norm_f64(&block);
        assert!(relative <= 1e-5, "relative error {relative:e}");
    }

    #[test]
    fn randomized_range_finder_handles_complex_rectangular_block() {
        let (m, n, rank) = (65, 41, 3);
        let u = DenseMatrix::from_fn(m, rank, |i, k| {
            Complex64::new(
                ((i + k * 7) as f64 * 0.13).sin(),
                ((i * 3 + k) as f64 * 0.07).cos(),
            )
        });
        let v = DenseMatrix::from_fn(n, rank, |j, k| {
            Complex64::new(
                ((j + k * 5) as f64 * 0.17).cos(),
                ((j * 2 + k) as f64 * 0.19).sin(),
            )
        });
        let block = u.matmul(&v.transposed());
        let truncated = truncate_randomized(&block, 1e-9);
        assert_eq!(truncated.rank, rank);
        let reconstructed = truncated.u.matmul(&truncated.v.transposed());
        let (error_sq, norm_sq) = block.as_slice().iter().zip(reconstructed.as_slice()).fold(
            (0.0, 0.0),
            |(error, norm), (value, approx)| {
                (
                    error + (*value - *approx).norm_sqr(),
                    norm + value.norm_sqr(),
                )
            },
        );
        assert!((error_sq / norm_sq).sqrt() < 1e-9);
    }

    #[test]
    fn randomized_range_finder_zero_block_has_zero_rank() {
        let block = DenseMatrix::<f64>::zeros(64, 48);
        let truncated = truncate_randomized(&block, 1e-8);
        assert_eq!(truncated.rank, 0);
        assert_eq!(truncated.u.shape(), (64, 0));
        assert_eq!(truncated.v.shape(), (48, 0));
    }
}
