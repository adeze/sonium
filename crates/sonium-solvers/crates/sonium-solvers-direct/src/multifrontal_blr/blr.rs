//! Block-low-rank (BLR) representation of off-diagonal frontal-matrix panels.
//!
//! A `BlrPanel<T>` is a (block-row × block-col) grid where each block is
//! either stored as a dense matrix or as a low-rank factor pair `(U, V)` with
//! `block ≈ U · V^T`. The block grid uses a uniform block size `bs`, with the
//! last row/column block possibly smaller.
//!
//! The `apply` and `apply_transpose` helpers do BLR-aware matvecs and matmuls
//! used during the multifrontal forward/backward solve and the Schur-update
//! step. We deliberately keep the operations simple: BLR×dense → dense; we do
//! not maintain low-rank products in the Schur update (that's a follow-up).

use half::f16;
use num_traits::{FromPrimitive as _, ToPrimitive as _, Zero as _};
use rayon::prelude::*;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;
use std::any::TypeId;

use super::scalar_is_finite;

use super::truncation::maybe_compress;

/// Precision used for the stored low-rank factors.
///
/// `Full` stores `(U, V)` at `T`'s precision (e.g. `f64` or `Complex64`).
/// `Half` stores `(U, V)` as `half::f16` with per-block Higham-style
/// scaling, halving the memory cost at the price of ≈ 10⁻³ relative
/// error per block. Only real-valued `T` (`f32`, `f64`) are supported
/// for half storage — complex blocks silently fall back to `Full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LowRankPrecision {
    #[default]
    Full,
    Half,
}

/// One block in a `BlrPanel`.
#[derive(Debug, Clone)]
pub enum BlrBlock<T: ComplexField> {
    /// Full dense storage. Shape `(rows × cols)`.
    Dense(DenseMatrix<T>),
    /// Low-rank factor pair at `T`'s precision.
    /// `u.shape() = (rows × r)`, `v.shape() = (cols × r)`.
    LowRank {
        u: DenseMatrix<T>,
        v: DenseMatrix<T>,
    },
    /// Half-precision low-rank factor pair with per-block Higham scaling.
    /// The flat row-major `u_half[i * r + l]` and `v_half[j * r + l]` are
    /// normalised so each factor's max-magnitude entry is `≤ 1`. The
    /// reconstruction is `block[i, j] ≈ (scale_u · scale_v) · Σ_l u·v`.
    /// Only constructed when `T` is `f32` or `f64`.
    LowRankHalf {
        u_half: Vec<f16>,
        v_half: Vec<f16>,
        rows: usize,
        cols: usize,
        rank: usize,
        scale_u: f64,
        scale_v: f64,
    },
}

/// `out[r0 + i, c0 + j] -= (lhs · rhs)[i, j]` for two BLR sub-blocks of
/// compatible inner dimension. Picks the cheapest reduction based on the
/// rank structure of `lhs` and `rhs`.
fn block_matmul_sub_dense<T: ComplexField>(
    lhs: &BlrBlock<T>,
    rhs: &BlrBlock<T>,
    out: &mut DenseMatrix<T>,
    r0: usize,
    c0: usize,
) {
    let m = lhs.rows();
    let n = rhs.cols();
    let k = lhs.cols();
    debug_assert_eq!(k, rhs.rows());
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let product = match (lhs, rhs) {
        (BlrBlock::LowRank { u: ua, v: va }, BlrBlock::LowRank { u: ub, v: vb }) => {
            // (Ua Va^T) · (Ub Vb^T) = Ua · (Va^T · Ub) · Vb^T.
            let ra = ua.ncols();
            let rb = ub.ncols();
            if ra == 0 || rb == 0 {
                return;
            }
            let inner = va.transposed().matmul(ub);
            ua.matmul(&inner).matmul(&vb.transposed())
        }
        (BlrBlock::LowRank { u: ua, v: va }, _) => {
            let rhs_dense = rhs.to_dense();
            if ua.ncols() == 0 {
                return;
            }
            ua.matmul(&va.transposed().matmul(&rhs_dense))
        }
        (_, BlrBlock::LowRank { u: ub, v: vb }) => {
            let lhs_dense = lhs.to_dense();
            if ub.ncols() == 0 {
                return;
            }
            lhs_dense.matmul(ub).matmul(&vb.transposed())
        }
        _ => {
            // Dense × Dense (or half storage falls through to dense via to_dense).
            lhs.to_dense().matmul(&rhs.to_dense())
        }
    };
    for i in 0..m {
        for j in 0..n {
            out[[r0 + i, c0 + j]] -= product[[i, j]];
        }
    }
}

/// Decide whether half-precision storage is supported for `T`.
fn supports_half<T: ComplexField>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Convert a `DenseMatrix<T>` to a row-major `Vec<f16>` with per-matrix
/// Higham scaling. Returns `(half_flat, scale)` where `scale` is the
/// original max-magnitude such that decoded[i, j] ≈ scale · half[i, j].
fn matrix_to_half<T: ComplexField>(m: &DenseMatrix<T>) -> (Vec<f16>, f64) {
    let rows = m.nrows();
    let cols = m.ncols();
    if rows == 0 || cols == 0 {
        return (Vec::new(), 1.0);
    }
    let mut max_mag = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let mag = m[[i, j]].norm().to_f64().unwrap_or(0.0);
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }
    let scale = if max_mag > 0.0 { max_mag } else { 1.0 };
    let inv = 1.0 / scale;
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            // Cast T entry to f64 via re() (real-only path enforced upstream).
            let v_real = m[[i, j]].re().to_f64().unwrap_or(0.0);
            let normalised = v_real * inv;
            out.push(f16::from_f64(normalised));
        }
    }
    (out, scale)
}

impl<T: ComplexField> BlrBlock<T> {
    pub(crate) fn is_finite(&self) -> bool {
        match self {
            BlrBlock::Dense(matrix) => matrix.as_slice().iter().copied().all(scalar_is_finite),
            BlrBlock::LowRank { u, v } => {
                u.as_slice().iter().copied().all(scalar_is_finite)
                    && v.as_slice().iter().copied().all(scalar_is_finite)
            }
            BlrBlock::LowRankHalf {
                u_half,
                v_half,
                scale_u,
                scale_v,
                ..
            } => {
                scale_u.is_finite()
                    && scale_v.is_finite()
                    && u_half.iter().all(|value| value.is_finite())
                    && v_half.iter().all(|value| value.is_finite())
            }
        }
    }

    /// Number of rows in this block.
    pub fn rows(&self) -> usize {
        match self {
            BlrBlock::Dense(m) => m.nrows(),
            BlrBlock::LowRank { u, .. } => u.nrows(),
            BlrBlock::LowRankHalf { rows, .. } => *rows,
        }
    }

    /// Number of columns in this block.
    pub fn cols(&self) -> usize {
        match self {
            BlrBlock::Dense(m) => m.ncols(),
            BlrBlock::LowRank { v, .. } => v.nrows(),
            BlrBlock::LowRankHalf { cols, .. } => *cols,
        }
    }

    /// Numerical rank.
    pub fn rank(&self) -> usize {
        match self {
            BlrBlock::Dense(m) => m.nrows().min(m.ncols()),
            BlrBlock::LowRank { u, .. } => u.ncols(),
            BlrBlock::LowRankHalf { rank, .. } => *rank,
        }
    }

    /// Stored entry count, in units of `T`-sized entries. Half-precision
    /// blocks store `(rows + cols) · rank` `f16` payloads (2 bytes each);
    /// we report `ceil(2·(rows + cols)·rank / size_of::<T>())` so that
    /// `entries · size_of::<T>()` accurately reflects on-disk bytes for
    /// any `T` (4-byte `f32` ⇒ ½ of `Full`, 8-byte `f64` ⇒ ¼ of `Full`).
    pub fn stored_entries(&self) -> usize {
        match self {
            BlrBlock::Dense(m) => m.nrows() * m.ncols(),
            BlrBlock::LowRank { u, v } => (u.nrows() + v.nrows()) * u.ncols(),
            BlrBlock::LowRankHalf {
                rows, cols, rank, ..
            } => {
                let half_bytes = (rows + cols) * rank * std::mem::size_of::<f16>();
                let t_size = std::mem::size_of::<T>().max(1);
                half_bytes.div_ceil(t_size) + 1
            }
        }
    }

    /// `y += alpha · block · x`.
    pub fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            BlrBlock::Dense(m) => {
                debug_assert_eq!(x.len(), m.ncols());
                debug_assert_eq!(y.len(), m.nrows());
                for i in 0..m.nrows() {
                    let mut s = T::zero();
                    for j in 0..m.ncols() {
                        s += m[[i, j]] * x[j];
                    }
                    y[i] += alpha * s;
                }
            }
            BlrBlock::LowRank { u, v } => {
                let r = u.ncols();
                let mut tmp = vec![T::zero(); r];
                for l in 0..r {
                    let mut s = T::zero();
                    for j in 0..v.nrows() {
                        s += v[[j, l]] * x[j];
                    }
                    tmp[l] = s;
                }
                for i in 0..u.nrows() {
                    let mut s = T::zero();
                    for l in 0..r {
                        s += u[[i, l]] * tmp[l];
                    }
                    y[i] += alpha * s;
                }
            }
            BlrBlock::LowRankHalf {
                u_half,
                v_half,
                rows,
                cols,
                rank,
                scale_u,
                scale_v,
            } => {
                debug_assert_eq!(x.len(), *cols);
                debug_assert_eq!(y.len(), *rows);
                let r = *rank;
                // tmp[l] = Σ_j (v_half[j*r + l] as f64) · (x[j] as f64)
                let mut tmp = vec![0.0_f64; r];
                for l in 0..r {
                    let mut s = 0.0_f64;
                    for j in 0..*cols {
                        let v = v_half[j * r + l].to_f64();
                        let xj = x[j].re().to_f64().unwrap_or(0.0);
                        s += v * xj;
                    }
                    tmp[l] = s * *scale_v;
                }
                let alpha_re = alpha.re().to_f64().unwrap_or(0.0);
                let pre = *scale_u * alpha_re;
                for i in 0..*rows {
                    let mut s = 0.0_f64;
                    for l in 0..r {
                        let u = u_half[i * r + l].to_f64();
                        s += u * tmp[l];
                    }
                    let inc = T::from_real(T::Real::from_f64(pre * s).unwrap_or(T::Real::zero()));
                    y[i] += inc;
                }
            }
        }
    }

    /// `y += alpha · block^T · x`.
    pub fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            BlrBlock::Dense(m) => {
                debug_assert_eq!(x.len(), m.nrows());
                debug_assert_eq!(y.len(), m.ncols());
                for j in 0..m.ncols() {
                    let mut s = T::zero();
                    for i in 0..m.nrows() {
                        s += m[[i, j]] * x[i];
                    }
                    y[j] += alpha * s;
                }
            }
            BlrBlock::LowRank { u, v } => {
                let r = u.ncols();
                let mut tmp = vec![T::zero(); r];
                for l in 0..r {
                    let mut s = T::zero();
                    for i in 0..u.nrows() {
                        s += u[[i, l]] * x[i];
                    }
                    tmp[l] = s;
                }
                for j in 0..v.nrows() {
                    let mut s = T::zero();
                    for l in 0..r {
                        s += v[[j, l]] * tmp[l];
                    }
                    y[j] += alpha * s;
                }
            }
            BlrBlock::LowRankHalf {
                u_half,
                v_half,
                rows,
                cols,
                rank,
                scale_u,
                scale_v,
            } => {
                debug_assert_eq!(x.len(), *rows);
                debug_assert_eq!(y.len(), *cols);
                let r = *rank;
                let mut tmp = vec![0.0_f64; r];
                for l in 0..r {
                    let mut s = 0.0_f64;
                    for i in 0..*rows {
                        let u = u_half[i * r + l].to_f64();
                        let xi = x[i].re().to_f64().unwrap_or(0.0);
                        s += u * xi;
                    }
                    tmp[l] = s * *scale_u;
                }
                let alpha_re = alpha.re().to_f64().unwrap_or(0.0);
                let pre = *scale_v * alpha_re;
                for j in 0..*cols {
                    let mut s = 0.0_f64;
                    for l in 0..r {
                        let v = v_half[j * r + l].to_f64();
                        s += v * tmp[l];
                    }
                    let inc = T::from_real(T::Real::from_f64(pre * s).unwrap_or(T::Real::zero()));
                    y[j] += inc;
                }
            }
        }
    }

    /// Materialise into a dense matrix.
    pub fn to_dense(&self) -> DenseMatrix<T> {
        match self {
            BlrBlock::Dense(m) => m.clone(),
            BlrBlock::LowRank { u, v } => u.matmul(&v.t()),
            BlrBlock::LowRankHalf {
                u_half,
                v_half,
                rows,
                cols,
                rank,
                scale_u,
                scale_v,
            } => {
                let mut out = DenseMatrix::<T>::zeros(*rows, *cols);
                let pre = scale_u * scale_v;
                for i in 0..*rows {
                    for j in 0..*cols {
                        let mut s = 0.0_f64;
                        for l in 0..*rank {
                            s += u_half[i * rank + l].to_f64() * v_half[j * rank + l].to_f64();
                        }
                        out[[i, j]] =
                            T::from_real(T::Real::from_f64(pre * s).unwrap_or(T::Real::zero()));
                    }
                }
                out
            }
        }
    }
}

/// A panel of a frontal matrix stored as a uniform block grid.
///
/// `blocks[bi][bj]` covers global rows `[bi·bs .. (bi+1)·bs)` (clipped at
/// `n_rows`) and global cols `[bj·bs .. (bj+1)·bs)` (clipped at `n_cols`).
#[derive(Debug, Clone)]
pub struct BlrPanel<T: ComplexField> {
    pub n_rows: usize,
    pub n_cols: usize,
    pub bs: usize,
    pub blocks: Vec<Vec<BlrBlock<T>>>,
}

impl<T: ComplexField> BlrPanel<T> {
    pub(crate) fn is_finite(&self) -> bool {
        self.blocks.iter().flatten().all(BlrBlock::is_finite)
    }

    /// Build a panel from a row-major dense source. Equivalent to
    /// `from_dense_with_precision(source, bs, epsilon, LowRankPrecision::Full)`.
    pub fn from_dense(source: &DenseMatrix<T>, bs: usize, epsilon: T::Real) -> Self {
        Self::from_dense_with_precision(source, bs, epsilon, LowRankPrecision::Full)
    }

    /// Build a panel from a row-major dense source at the requested
    /// low-rank storage precision.
    ///
    /// `LowRankPrecision::Half` halves storage for blocks that pass the
    /// rank-truncation test, at the cost of ≈ `1e-3` relative error per
    /// block. Falls back to `Full` for complex-valued `T`.
    pub fn from_dense_with_precision(
        source: &DenseMatrix<T>,
        bs: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
    ) -> Self {
        Self::from_dense_with_precision_maybe_parallel(source, bs, epsilon, precision, false)
    }

    pub(crate) fn from_dense_with_precision_maybe_parallel(
        source: &DenseMatrix<T>,
        bs: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Self {
        let n_rows = source.nrows();
        let n_cols = source.ncols();
        if n_rows == 0 || n_cols == 0 {
            return Self {
                n_rows,
                n_cols,
                bs: bs.max(1),
                blocks: Vec::new(),
            };
        }
        let bs = bs.max(1);
        let nbr = n_rows.div_ceil(bs);
        let nbc = n_cols.div_ceil(bs);
        let half_ok = supports_half::<T>() && precision == LowRankPrecision::Half;

        let build_row = |bi: usize| {
            let r0 = bi * bs;
            let r1 = ((bi + 1) * bs).min(n_rows);
            let mut row = Vec::with_capacity(nbc);
            for bj in 0..nbc {
                let c0 = bj * bs;
                let c1 = ((bj + 1) * bs).min(n_cols);
                let mut block_dense = DenseMatrix::<T>::zeros(r1 - r0, c1 - c0);
                for i in r0..r1 {
                    for j in c0..c1 {
                        block_dense[[i - r0, j - c0]] = source[[i, j]];
                    }
                }
                row.push(Self::compress_block(block_dense, epsilon, half_ok));
            }
            row
        };

        let blocks: Vec<Vec<BlrBlock<T>>> = if use_parallel && nbr > 1 {
            (0..nbr).into_par_iter().map(build_row).collect()
        } else {
            (0..nbr).map(build_row).collect()
        };

        Self {
            n_rows,
            n_cols,
            bs,
            blocks,
        }
    }

    /// Owned-source variant of
    /// [`from_dense_with_precision_maybe_parallel`](Self::from_dense_with_precision_maybe_parallel).
    /// Bit-identical output, but avoids copying the source twice:
    /// - single-block panels (`nbr == nbc == 1`) move the source `Vec`
    ///   straight into `BlrBlock::Dense` (no allocate-zero-copy cycle);
    /// - multi-block panels copy row segments with `extend_from_slice`
    ///   (memcpy) into capacity-exact buffers instead of allocating zeros
    ///   and storing element by element through `Index`.
    pub(crate) fn from_dense_owned_with_precision_maybe_parallel(
        source: DenseMatrix<T>,
        bs: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Self {
        let n_rows = source.nrows();
        let n_cols = source.ncols();
        if n_rows == 0 || n_cols == 0 {
            return Self {
                n_rows,
                n_cols,
                bs: bs.max(1),
                blocks: Vec::new(),
            };
        }
        let bs = bs.max(1);
        let nbr = n_rows.div_ceil(bs);
        let nbc = n_cols.div_ceil(bs);
        let half_ok = supports_half::<T>() && precision == LowRankPrecision::Half;

        if nbr == 1 && nbc == 1 {
            // Whole panel is one block: the source storage already has the
            // exact shape and layout `BlrBlock::Dense` needs.
            return Self {
                n_rows,
                n_cols,
                bs,
                blocks: vec![vec![Self::compress_block(source, epsilon, half_ok)]],
            };
        }

        let build_row = |bi: usize| {
            let r0 = bi * bs;
            let r1 = ((bi + 1) * bs).min(n_rows);
            let mut row = Vec::with_capacity(nbc);
            for bj in 0..nbc {
                let c0 = bj * bs;
                let c1 = ((bj + 1) * bs).min(n_cols);
                let mut data = Vec::with_capacity((r1 - r0) * (c1 - c0));
                for i in r0..r1 {
                    data.extend_from_slice(&source.row(i)[c0..c1]);
                }
                let block_dense = DenseMatrix::<T>::from_row_major_vec(r1 - r0, c1 - c0, data);
                row.push(Self::compress_block(block_dense, epsilon, half_ok));
            }
            row
        };

        let blocks: Vec<Vec<BlrBlock<T>>> = if use_parallel && nbr > 1 {
            (0..nbr).into_par_iter().map(build_row).collect()
        } else {
            (0..nbr).map(build_row).collect()
        };

        Self {
            n_rows,
            n_cols,
            bs,
            blocks,
        }
    }

    fn compress_block(block_dense: DenseMatrix<T>, epsilon: T::Real, half_ok: bool) -> BlrBlock<T> {
        // Exact mode (epsilon == 0): rank analysis can only conclude "keep
        // dense" for a nonzero block, so skip the RRQR entirely. All-zero
        // blocks keep the rank-0 LowRank encoding on the non-half path
        // (half mode stores zeros dense), exactly matching the analysis
        // outcome without running it.
        if epsilon <= T::Real::zero() {
            if half_ok || block_dense.as_slice().iter().any(|v| *v != T::zero()) {
                return BlrBlock::Dense(block_dense);
            }
            let (m, n) = (block_dense.nrows(), block_dense.ncols());
            return BlrBlock::LowRank {
                u: DenseMatrix::zeros(m, 0),
                v: DenseMatrix::zeros(n, 0),
            };
        }
        match maybe_compress(&block_dense, epsilon) {
            Some((u, v)) if half_ok => {
                let (u_half, scale_u) = matrix_to_half::<T>(&u);
                let (v_half, scale_v) = matrix_to_half::<T>(&v);
                let rows = u.nrows();
                let cols = v.nrows();
                let rank = u.ncols();
                if rank == 0 {
                    BlrBlock::Dense(block_dense)
                } else {
                    BlrBlock::LowRankHalf {
                        u_half,
                        v_half,
                        rows,
                        cols,
                        rank,
                        scale_u,
                        scale_v,
                    }
                }
            }
            Some((u, v)) => BlrBlock::LowRank { u, v },
            None => BlrBlock::Dense(block_dense),
        }
    }

    /// `y += alpha · panel · x` (panel as an `n_rows × n_cols` matrix).
    pub fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_cols);
        debug_assert_eq!(y.len(), self.n_rows);
        for (bi, row) in self.blocks.iter().enumerate() {
            let r0 = bi * self.bs;
            for (bj, block) in row.iter().enumerate() {
                let c0 = bj * self.bs;
                let x_slice = &x[c0..c0 + block.cols()];
                let y_slice = &mut y[r0..r0 + block.rows()];
                block.gemv_into(alpha, x_slice, y_slice);
            }
        }
    }

    /// `y += alpha · panel^T · x`.
    pub fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_rows);
        debug_assert_eq!(y.len(), self.n_cols);
        for (bi, row) in self.blocks.iter().enumerate() {
            let r0 = bi * self.bs;
            for (bj, block) in row.iter().enumerate() {
                let c0 = bj * self.bs;
                let x_slice = &x[r0..r0 + block.rows()];
                let y_slice = &mut y[c0..c0 + block.cols()];
                block.gemv_transpose_into(alpha, x_slice, y_slice);
            }
        }
    }

    /// Total stored entries across all blocks.
    pub fn stored_entries(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(BlrBlock::stored_entries)
            .sum()
    }

    /// Fraction of blocks stored as low-rank.
    pub fn low_rank_fraction(&self) -> f64 {
        let mut total = 0usize;
        let mut lr = 0usize;
        for row in &self.blocks {
            for b in row {
                total += 1;
                if matches!(b, BlrBlock::LowRank { .. } | BlrBlock::LowRankHalf { .. }) {
                    lr += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            lr as f64 / total as f64
        }
    }

    /// Average numerical rank across low-rank blocks (`0.0` if none).
    pub fn average_low_rank(&self) -> f64 {
        let mut sum = 0usize;
        let mut count = 0usize;
        for row in &self.blocks {
            for b in row {
                match b {
                    BlrBlock::LowRank { u, .. } => {
                        sum += u.ncols();
                        count += 1;
                    }
                    BlrBlock::LowRankHalf { rank, .. } => {
                        sum += rank;
                        count += 1;
                    }
                    BlrBlock::Dense(_) => {}
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64
        }
    }

    /// Block-wise BLR×BLR matmul, accumulating the product into a dense
    /// output: `out -= self · rhs`.
    ///
    /// When both operand blocks are low-rank, computes
    /// `U_a · (V_a^T · U_b) · V_b^T` (cheap rank·rank inner product
    /// followed by two thin matmuls). When only one is low-rank, exploits
    /// the rank-r factor to skip the full dense GEMM. When both are dense,
    /// falls back to a triple loop.
    ///
    /// `self.bs` and `rhs.bs` must match; this is the natural case when
    /// both panels are built from the same front (uniform block grid).
    pub fn matmul_sub_into_dense(&self, rhs: &Self, out: &mut DenseMatrix<T>) {
        assert_eq!(self.n_cols, rhs.n_rows, "BLR matmul shape mismatch");
        assert_eq!(out.nrows(), self.n_rows);
        assert_eq!(out.ncols(), rhs.n_cols);
        debug_assert_eq!(self.bs, rhs.bs);
        let bs = self.bs;
        let nbr = self.blocks.len();
        let nbk = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].len()
        };
        let nbc = if rhs.blocks.is_empty() {
            0
        } else {
            rhs.blocks[0].len()
        };

        for bi in 0..nbr {
            let r0 = bi * bs;
            for bj in 0..nbc {
                let c0 = bj * bs;
                for bk in 0..nbk {
                    let lhs_block = &self.blocks[bi][bk];
                    let rhs_block = &rhs.blocks[bk][bj];
                    block_matmul_sub_dense(lhs_block, rhs_block, out, r0, c0);
                }
            }
        }
    }

    /// Materialise the entire panel as a dense matrix (debug / verification).
    pub fn to_dense(&self) -> DenseMatrix<T> {
        let mut out = DenseMatrix::<T>::zeros(self.n_rows, self.n_cols);
        for (bi, row) in self.blocks.iter().enumerate() {
            let r0 = bi * self.bs;
            for (bj, block) in row.iter().enumerate() {
                let c0 = bj * self.bs;
                let dense = block.to_dense();
                for i in 0..block.rows() {
                    for j in 0..block.cols() {
                        out[[r0 + i, c0 + j]] = dense[[i, j]];
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn panel_round_trip_full_rank() {
        let data: Vec<f64> = (0..400).map(|k| (k as f64).sin() + 0.5).collect();
        let m = DenseMatrix::<f64>::from_row_major_vec(20, 20, data);
        let panel = BlrPanel::from_dense(&m, 8, 0.0);
        let back = panel.to_dense();
        for i in 0..20 {
            for j in 0..20 {
                assert!((m[[i, j]] - back[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn panel_gemv_matches_dense() {
        let data: Vec<f64> = (0..200).map(|k| (k as f64).cos()).collect();
        let m = DenseMatrix::<f64>::from_row_major_vec(10, 20, data);
        let panel = BlrPanel::from_dense(&m, 4, 0.0);
        let x: Vec<f64> = (0..20).map(|j| (j as f64 + 0.3).sin()).collect();
        let mut y_panel = vec![0.0_f64; 10];
        let mut y_dense = [0.0_f64; 10];
        panel.gemv_into(1.0, &x, &mut y_panel);
        for i in 0..10 {
            let mut s = 0.0;
            for j in 0..20 {
                s += m[[i, j]] * x[j];
            }
            y_dense[i] = s;
        }
        for i in 0..10 {
            assert!((y_panel[i] - y_dense[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn panel_low_rank_compresses() {
        // Rank-1 panel u v^T at large size — BLR should compress every block.
        let m = 32;
        let n = 32;
        let u_vec: Vec<f64> = (0..m).map(|i| ((i + 1) as f64).sqrt()).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| (j as f64 + 0.5).cos()).collect();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = u_vec[i] * v_vec[j];
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let panel = BlrPanel::from_dense(&mat, 8, 1e-10);
        let frac = panel.low_rank_fraction();
        assert!(frac > 0.5, "low-rank fraction too low: {}", frac);
        // GEMV stays accurate.
        let x: Vec<f64> = (0..n).map(|j| (j as f64).cos()).collect();
        let mut y_panel = vec![0.0_f64; m];
        let mut y_ref = vec![0.0_f64; m];
        panel.gemv_into(1.0, &x, &mut y_panel);
        for i in 0..m {
            for j in 0..n {
                y_ref[i] += mat[[i, j]] * x[j];
            }
        }
        for i in 0..m {
            assert!(
                (y_panel[i] - y_ref[i]).abs() < 1e-8,
                "i={}, diff={}",
                i,
                (y_panel[i] - y_ref[i]).abs()
            );
        }
    }

    #[test]
    fn panel_parallel_build_matches_serial_full_precision() {
        let m = 48;
        let n = 40;
        let u_vec: Vec<f64> = (0..m).map(|i| (i as f64 + 1.0).sqrt()).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| (j as f64 + 0.25).cos()).collect();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = u_vec[i] * v_vec[j];
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let serial = BlrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            8,
            1e-10,
            LowRankPrecision::Full,
            false,
        );
        let parallel = BlrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            8,
            1e-10,
            LowRankPrecision::Full,
            true,
        );

        assert_eq!(serial.stored_entries(), parallel.stored_entries());
        assert_eq!(serial.low_rank_fraction(), parallel.low_rank_fraction());
        assert_eq!(serial.average_low_rank(), parallel.average_low_rank());

        let serial_dense = serial.to_dense();
        let parallel_dense = parallel.to_dense();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (serial_dense[[i, j]] - parallel_dense[[i, j]]).abs() < 1e-10,
                    "({i}, {j}) differs: serial={} parallel={}",
                    serial_dense[[i, j]],
                    parallel_dense[[i, j]]
                );
            }
        }
    }

    #[test]
    fn panel_parallel_build_matches_serial_half_precision() {
        let m = 48;
        let n = 40;
        let u_vec: Vec<f64> = (0..m).map(|i| (i as f64 + 1.0).sin()).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| (j as f64 + 0.5).cos()).collect();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = u_vec[i] * v_vec[j];
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let serial = BlrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            8,
            1e-10,
            LowRankPrecision::Half,
            false,
        );
        let parallel = BlrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            8,
            1e-10,
            LowRankPrecision::Half,
            true,
        );

        assert_eq!(serial.stored_entries(), parallel.stored_entries());
        assert_eq!(serial.low_rank_fraction(), parallel.low_rank_fraction());
        assert_eq!(serial.average_low_rank(), parallel.average_low_rank());

        let serial_dense = serial.to_dense();
        let parallel_dense = parallel.to_dense();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (serial_dense[[i, j]] - parallel_dense[[i, j]]).abs() < 1e-10,
                    "({i}, {j}) differs: serial={} parallel={}",
                    serial_dense[[i, j]],
                    parallel_dense[[i, j]]
                );
            }
        }
    }

    #[test]
    fn panel_complex_matches_dense() {
        let m = 6;
        let n = 8;
        let mut data = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] =
                    Complex64::new((i as f64 + 0.1) * (j as f64 + 1.0).sqrt(), (j as f64).cos());
            }
        }
        let mat = DenseMatrix::<Complex64>::from_row_major_vec(m, n, data);
        let panel = BlrPanel::from_dense(&mat, 4, 0.0);
        let x: Vec<Complex64> = (0..n)
            .map(|j| Complex64::new(j as f64, -(j as f64)))
            .collect();
        let mut y = vec![Complex64::new(0.0, 0.0); m];
        panel.gemv_into(Complex64::new(1.0, 0.0), &x, &mut y);
        for i in 0..m {
            let mut s = Complex64::new(0.0, 0.0);
            for j in 0..n {
                s += mat[[i, j]] * x[j];
            }
            let d = (s - y[i]).norm();
            assert!(d < 1e-10, "i={} d={}", i, d);
        }
    }
}
