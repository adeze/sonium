//! Nested-basis global low-rank panel (**not** HSS).
//!
//! This module was formerly labelled "HSS", but the construction is *not*
//! Hierarchically Semi-Separable: the whole panel is approximated as one
//! global product `A ≈ U_root · S · V_rootᵀ` with hierarchically shared
//! bases — there are no per-sibling coupling blocks and no ULV
//! factorisation. The name [`NestedLrPanel`] (nested-basis low-rank)
//! reflects what the code actually does.
//!
//! Where [`HodlrPanel`](super::hodlr::HodlrPanel) stores an *independent*
//! `(U, V)` pair for every off-diagonal sub-block at every level of its
//! recursion, this panel shares the row / column bases across levels:
//!
//! * For each leaf row cluster `i`, a tall thin `U_i` of shape `(m_i × k_i)`
//!   spans the row range of the panel restricted to those rows.
//! * For each internal row node `τ` with children `(l, r)`, a *translation*
//!   operator `R_τ` of shape `((k_l + k_r) × k_τ)` expresses the row basis
//!   of `τ` as a linear combination of the (already-stored) row bases of
//!   its children:
//!   `U_τ = blkdiag(U_l, U_r) · R_τ`.
//! * The same construction on the columns yields leaf bases `V_j` and
//!   translation operators `W_τ`.
//! * A single dense core `S` of shape `(k_root × l_root)` carries all of
//!   the panel's interaction information, so the panel approximates as
//!   `A ≈ U_root · S · V_root^T`
//!   where neither `U_root` nor `V_root` is ever materialised.
//!
//! Asymptotic memory is `O((M + N)·k + k²·log(M/leaf))` where `k` is the
//! maximum cluster rank — better than HODLR's
//! `O((M + N)·k·log(M/leaf))` because the leaf bases are not duplicated
//! across levels. The flip side of a *global* basis is that a panel which
//! is only block-wise low-rank (independent low-rank blocks along the
//! diagonal, say) drives the nested root rank up towards `min(M, N)`. To
//! keep `from_dense` from ever returning a representation that costs more
//! than the dense panel, construction applies the same storage-gain rule
//! as the BLR `maybe_compress` guard: if the nested storage reaches
//! `M · N` entries, the panel is stored dense instead.
//!
//! The panel is rectangular and *entirely* off-diagonal in the multifrontal
//! sense (it represents `L21` / `U12` / `CB`), so there are no in-panel
//! "near-field" dense blocks to worry about; the construction reduces to a
//! single hierarchical low-rank approximation of the whole panel.
//!
//! Construction overview (`from_dense`):
//!
//! 1. Build a binary row tree (`NestedLrRowTree`) by recursively splitting the
//!    `[0, M)` range until clusters fit `leaf_size`. Same for columns.
//! 2. Row pass (bottom-up): at every node compute a column-orthonormal
//!    cluster basis. Leaves call [`truncate_rrqr`] on the dense source rows.
//!    Internal nodes compress
//!    `[U_l^H · A[l, :]; U_r^H · A[r, :]] ∈ ℝ^{(k_l + k_r) × N}`
//!    and store the resulting `Q[:, :k_τ]` as `R_τ`.
//! 3. Column pass (bottom-up): symmetric — at every node compute a
//!    column-orthonormal basis over column slabs, materialising leaf `V_j`
//!    and translation `W_τ`.
//! 4. Core: `S = U_root^H · A · V_root`, computed by re-using the same
//!    bottom-up projections (avoids materialising `U_root` or `V_root`).
//!
//! `gemv` runs in three phases: an upward column-tree contraction
//! (`x̂ = V_root^H · x`), a small dense `ŷ = S · x̂` at the root, and a
//! downward row-tree expansion (`y = U_root · ŷ`). Transpose-gemv flips
//! the roles symmetrically.

use std::any::TypeId;
use std::borrow::Cow;

use half::f16;
use num_traits::{FromPrimitive as _, ToPrimitive as _, Zero as _};

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;

use super::blr::LowRankPrecision;
use super::scalar_is_finite;
use super::truncation::truncate_rrqr;

/// A row-major dense matrix held either in full `T` precision or in
/// `f16` with a single Higham-style per-matrix scale. The gemv / matmul
/// paths call [`MaybeHalf::as_dense`] to obtain a `T`-precision view
/// before doing arithmetic — the half representation is *storage-only*,
/// decoded once per visit.
pub enum MaybeHalf<T: ComplexField> {
    Full(DenseMatrix<T>),
    Half {
        data: Vec<f16>,
        rows: usize,
        cols: usize,
        scale: f64,
    },
}

impl<T: ComplexField> MaybeHalf<T> {
    fn is_finite(&self) -> bool {
        match self {
            MaybeHalf::Full(matrix) => matrix.as_slice().iter().copied().all(scalar_is_finite),
            MaybeHalf::Half { data, scale, .. } => {
                scale.is_finite() && data.iter().all(|value| value.is_finite())
            }
        }
    }

    fn ncols(&self) -> usize {
        match self {
            MaybeHalf::Full(m) => m.ncols(),
            MaybeHalf::Half { cols, .. } => *cols,
        }
    }
    /// Storage cost in `T`-sized units, matching the convention used by
    /// [`crate::multifrontal_blr::MultifrontalBlrFactorization::factor_memory_bytes`].
    fn stored_entries(&self) -> usize {
        match self {
            MaybeHalf::Full(m) => m.nrows() * m.ncols(),
            MaybeHalf::Half { rows, cols, .. } => {
                let half_bytes = rows * cols * std::mem::size_of::<f16>();
                let t_size = std::mem::size_of::<T>().max(1);
                half_bytes.div_ceil(t_size) + 1
            }
        }
    }
    /// Decode-to-dense for the half case; cheap borrow for the full case.
    /// Cost: `O(rows · cols)` once per call when decoding.
    fn as_dense(&self) -> Cow<'_, DenseMatrix<T>> {
        match self {
            MaybeHalf::Full(m) => Cow::Borrowed(m),
            MaybeHalf::Half {
                data,
                rows,
                cols,
                scale,
            } => {
                let mut out = DenseMatrix::<T>::zeros(*rows, *cols);
                let s_real = T::Real::from_f64(*scale).unwrap_or_else(T::Real::zero);
                for i in 0..*rows {
                    for j in 0..*cols {
                        let v = data[i * cols + j].to_f64();
                        let real = T::Real::from_f64(v).unwrap_or_else(T::Real::zero) * s_real;
                        out[[i, j]] = T::from_real(real);
                    }
                }
                Cow::Owned(out)
            }
        }
    }
    /// Build a `MaybeHalf::Half` from a full-precision matrix using a
    /// single per-matrix Higham scale. Falls back to `Full` when `T` does
    /// not support a half representation (i.e. complex `T`) or the matrix
    /// has zero area.
    fn from_dense_with_precision(m: DenseMatrix<T>, precision: LowRankPrecision) -> Self {
        let half_ok = precision == LowRankPrecision::Half
            && supports_half::<T>()
            && m.nrows() * m.ncols() > 0;
        if !half_ok {
            return MaybeHalf::Full(m);
        }
        let rows = m.nrows();
        let cols = m.ncols();
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
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let v = m[[i, j]].re().to_f64().unwrap_or(0.0) * inv;
                data.push(f16::from_f64(v));
            }
        }
        MaybeHalf::Half {
            data,
            rows,
            cols,
            scale,
        }
    }
}

fn supports_half<T: ComplexField>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

/// One node of the row tree of a nested-basis panel.
pub enum NestedLrRowTree<T: ComplexField> {
    Leaf {
        /// Inclusive row start of this cluster in the panel.
        row_start: usize,
        /// Exclusive row end.
        row_end: usize,
        /// Column-orthonormal leaf basis, shape `(m_i × k_i)`. Either full
        /// `T` precision or `f16` with a per-matrix scale; see [`MaybeHalf`].
        u: MaybeHalf<T>,
    },
    Node {
        row_start: usize,
        row_end: usize,
        left: Box<NestedLrRowTree<T>>,
        right: Box<NestedLrRowTree<T>>,
        /// Translation operator, shape `((k_l + k_r) × k_τ)`.
        r: MaybeHalf<T>,
    },
}

/// One node of the column tree of a nested-basis panel.
pub enum NestedLrColTree<T: ComplexField> {
    Leaf {
        col_start: usize,
        col_end: usize,
        /// Column-orthonormal leaf basis, shape `(n_j × l_j)`.
        v: MaybeHalf<T>,
    },
    Node {
        col_start: usize,
        col_end: usize,
        left: Box<NestedLrColTree<T>>,
        right: Box<NestedLrColTree<T>>,
        /// Translation operator, shape `((l_l + l_r) × l_τ)`.
        w: MaybeHalf<T>,
    },
}

impl<T: ComplexField> NestedLrRowTree<T> {
    fn is_finite(&self) -> bool {
        match self {
            NestedLrRowTree::Leaf { u, .. } => u.is_finite(),
            NestedLrRowTree::Node { left, right, r, .. } => {
                left.is_finite() && right.is_finite() && r.is_finite()
            }
        }
    }

    fn rank(&self) -> usize {
        match self {
            NestedLrRowTree::Leaf { u, .. } => u.ncols(),
            NestedLrRowTree::Node { r, .. } => r.ncols(),
        }
    }
    fn stored_entries(&self) -> usize {
        match self {
            NestedLrRowTree::Leaf { u, .. } => u.stored_entries(),
            NestedLrRowTree::Node { left, right, r, .. } => {
                left.stored_entries() + right.stored_entries() + r.stored_entries()
            }
        }
    }
}

impl<T: ComplexField> NestedLrColTree<T> {
    fn is_finite(&self) -> bool {
        match self {
            NestedLrColTree::Leaf { v, .. } => v.is_finite(),
            NestedLrColTree::Node { left, right, w, .. } => {
                left.is_finite() && right.is_finite() && w.is_finite()
            }
        }
    }

    fn rank(&self) -> usize {
        match self {
            NestedLrColTree::Leaf { v, .. } => v.ncols(),
            NestedLrColTree::Node { w, .. } => w.ncols(),
        }
    }
    fn stored_entries(&self) -> usize {
        match self {
            NestedLrColTree::Leaf { v, .. } => v.stored_entries(),
            NestedLrColTree::Node { left, right, w, .. } => {
                left.stored_entries() + right.stored_entries() + w.stored_entries()
            }
        }
    }
}

/// A panel stored in nested-basis low-rank form, or dense when the panel
/// is too small to benefit or the nested representation would cost at
/// least as much as the dense panel (storage-gain fallback, same rule as
/// the BLR `maybe_compress` guard).
pub enum NestedLrPanel<T: ComplexField> {
    /// Plain dense storage — used when the panel fits a single leaf or
    /// when the nested-basis form would not save memory.
    Dense(DenseMatrix<T>),
    /// Nested-basis form: `A ≈ U_root · S · V_root^T` with the row and
    /// column bases described by their respective trees.
    Nested {
        n_rows: usize,
        n_cols: usize,
        row_tree: NestedLrRowTree<T>,
        col_tree: NestedLrColTree<T>,
        /// Core matrix `S`, shape `(k_root × l_root)`.
        s: DenseMatrix<T>,
    },
}

/// Deprecated alias for [`NestedLrPanel`]: the panel is a nested-basis
/// global low-rank approximation (`U_root·S·V_rootᵀ` over the whole
/// panel), not HSS.
#[deprecated(note = "renamed to NestedLrPanel — this is not HSS")]
pub type HssPanel<T> = NestedLrPanel<T>;

/// Deprecated alias for [`NestedLrRowTree`] (see [`HssPanel`]).
#[deprecated(note = "renamed to NestedLrRowTree — this is not HSS")]
pub type HssRowTree<T> = NestedLrRowTree<T>;

/// Deprecated alias for [`NestedLrColTree`] (see [`HssPanel`]).
#[deprecated(note = "renamed to NestedLrColTree — this is not HSS")]
pub type HssColTree<T> = NestedLrColTree<T>;

impl<T: ComplexField> NestedLrPanel<T> {
    pub(crate) fn is_finite(&self) -> bool {
        match self {
            NestedLrPanel::Dense(matrix) => matrix.as_slice().iter().copied().all(scalar_is_finite),
            NestedLrPanel::Nested {
                row_tree,
                col_tree,
                s,
                ..
            } => {
                row_tree.is_finite()
                    && col_tree.is_finite()
                    && s.as_slice().iter().copied().all(scalar_is_finite)
            }
        }
    }

    /// Build a nested-basis low-rank panel from a dense source at full `T` precision.
    /// Shorthand for `from_dense_with_precision(.., LowRankPrecision::Full)`.
    pub fn from_dense(source: &DenseMatrix<T>, leaf_size: usize, epsilon: T::Real) -> Self {
        Self::from_dense_with_precision(source, leaf_size, epsilon, LowRankPrecision::Full)
    }

    /// Build a nested-basis low-rank panel from a dense source.
    ///
    /// * `leaf_size`: clusters with side ≤ `leaf_size` become tree leaves.
    /// * `epsilon`: relative RRQR tolerance for both row and column cluster
    ///   bases. `0.0` keeps full numerical rank.
    /// * `precision`: storage precision for the nested basis matrices
    ///   (`U`, `V`, `R`, `W`). [`LowRankPrecision::Half`] cuts their
    ///   storage by `4×` for `f64` panels and `2×` for `f32` panels at
    ///   the cost of a ~`1e-3` per-block error floor. The root core `S`
    ///   stays at full precision because it is small and on the inner
    ///   hot path. Half is a silent no-op for complex `T`.
    ///
    /// Falls back to [`NestedLrPanel::Dense`] when `max(M, N) ≤ leaf_size`,
    /// and also when the finished nested representation would occupy at
    /// least `M · N` entries — a panel that is block-wise low-rank but not
    /// *globally* low-rank grows the nested root rank until dense is
    /// cheaper, and this constructor never silently returns a
    /// representation that costs more memory than dense.
    pub fn from_dense_with_precision(
        source: &DenseMatrix<T>,
        leaf_size: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
    ) -> Self {
        Self::from_dense_with_precision_maybe_parallel(source, leaf_size, epsilon, precision, false)
    }

    pub(crate) fn from_dense_with_precision_maybe_parallel(
        source: &DenseMatrix<T>,
        leaf_size: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Self {
        let m = source.nrows();
        let n = source.ncols();
        if m == 0 || n == 0 {
            return NestedLrPanel::Dense(DenseMatrix::<T>::zeros(m, n));
        }
        if m <= leaf_size || n <= leaf_size {
            return NestedLrPanel::Dense(source.clone());
        }

        // 1. Build row tree (bottom-up), populating leaf U and internal R
        //    operators. Each node carries with it the projected matrix
        //    `proj = U_τ^H · A[row_range(τ), :]` of shape `(k_τ × N)` so the
        //    parent can compute its R operator without re-touching A.
        let (row_tree, row_proj) =
            build_row_tree(source, 0, m, leaf_size, epsilon, precision, use_parallel);
        // 2. Column tree on `row_proj`: the row projection has shape
        //    (k_root × N), so the column tree bases of A and of row_proj
        //    span the same column ranges in the panel. We compress those
        //    rows to obtain V_j leaves and W_τ translations.
        let (col_tree, s) =
            build_col_tree(&row_proj, 0, n, leaf_size, epsilon, precision, use_parallel);

        // 3. Storage-gain guard (same rule as the BLR `maybe_compress`
        //    fallback): if the nested representation costs at least as much
        //    as the dense panel, keep the dense panel. Because the basis is
        //    *global*, a panel that is only block-wise low-rank grows the
        //    nested ranks until this triggers.
        let nested_entries =
            row_tree.stored_entries() + col_tree.stored_entries() + s.nrows() * s.ncols();
        if nested_entries >= m * n {
            return NestedLrPanel::Dense(source.clone());
        }

        NestedLrPanel::Nested {
            n_rows: m,
            n_cols: n,
            row_tree,
            col_tree,
            s,
        }
    }

    pub fn n_rows(&self) -> usize {
        match self {
            NestedLrPanel::Dense(m) => m.nrows(),
            NestedLrPanel::Nested { n_rows, .. } => *n_rows,
        }
    }

    pub fn n_cols(&self) -> usize {
        match self {
            NestedLrPanel::Dense(m) => m.ncols(),
            NestedLrPanel::Nested { n_cols, .. } => *n_cols,
        }
    }

    /// `y += alpha · panel · x` via the two-pass nested-basis algorithm.
    pub fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_cols());
        debug_assert_eq!(y.len(), self.n_rows());
        match self {
            NestedLrPanel::Dense(d) => {
                for i in 0..d.nrows() {
                    let mut s = T::zero();
                    for j in 0..d.ncols() {
                        s += d[[i, j]] * x[j];
                    }
                    y[i] += alpha * s;
                }
            }
            NestedLrPanel::Nested {
                row_tree,
                col_tree,
                s,
                ..
            } => {
                // Upward column-tree contraction: x̂ = V_root^H · x.
                let x_hat = upward_col(col_tree, x);
                // Core multiply: ŷ = S · x̂.
                let k = s.nrows();
                let l = s.ncols();
                debug_assert_eq!(x_hat.len(), l);
                let mut y_hat = vec![T::zero(); k];
                for i in 0..k {
                    let mut acc = T::zero();
                    for j in 0..l {
                        acc += s[[i, j]] * x_hat[j];
                    }
                    y_hat[i] = acc;
                }
                // Downward row-tree expansion: y += alpha · U_root · ŷ.
                downward_row(row_tree, alpha, &y_hat, y);
            }
        }
    }

    /// `y += alpha · panel^T · x` (transpose, no conjugation; matches the
    /// existing BLR / HODLR `gemv_transpose_into` convention).
    pub fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_rows());
        debug_assert_eq!(y.len(), self.n_cols());
        match self {
            NestedLrPanel::Dense(d) => {
                for j in 0..d.ncols() {
                    let mut s = T::zero();
                    for i in 0..d.nrows() {
                        s += d[[i, j]] * x[i];
                    }
                    y[j] += alpha * s;
                }
            }
            NestedLrPanel::Nested {
                row_tree,
                col_tree,
                s,
                ..
            } => {
                // Upward row-tree contraction: x̂ = U_root^T · x.
                let x_hat = upward_row(row_tree, x);
                // ŷ = S^T · x̂.
                let k = s.nrows();
                let l = s.ncols();
                debug_assert_eq!(x_hat.len(), k);
                let mut y_hat = vec![T::zero(); l];
                for j in 0..l {
                    let mut acc = T::zero();
                    for i in 0..k {
                        acc += s[[i, j]] * x_hat[i];
                    }
                    y_hat[j] = acc;
                }
                // Downward col-tree expansion: y += alpha · V_root · ŷ.
                downward_col(col_tree, alpha, &y_hat, y);
            }
        }
    }

    /// Mean cluster rank across all nodes of the row and column trees.
    /// Returns `None` for [`NestedLrPanel::Dense`] (no nested basis to average).
    ///
    /// This is the most natural single-number summary of nested-basis compression:
    /// every leaf and every internal node carries one rank (the column
    /// count of its `U`/`V`/`R`/`W` matrix), and the arithmetic mean over
    /// the union of the two trees indicates how "narrow" the typical
    /// shared basis is. For a smooth kernel sampled on a dense grid the
    /// value stays roughly constant across `M`; for full-rank panels it
    /// approaches `min(M, N) / leaf_size`.
    pub fn average_rank(&self) -> Option<f64> {
        match self {
            NestedLrPanel::Dense(_) => None,
            NestedLrPanel::Nested {
                row_tree, col_tree, ..
            } => {
                let (rs, rc) = row_rank_sum_count(row_tree);
                let (cs, cc) = col_rank_sum_count(col_tree);
                let total = rc + cc;
                if total == 0 {
                    None
                } else {
                    Some((rs + cs) as f64 / total as f64)
                }
            }
        }
    }

    /// Stored entries across the row tree, the column tree, and the core.
    pub fn stored_entries(&self) -> usize {
        match self {
            NestedLrPanel::Dense(d) => d.nrows() * d.ncols(),
            NestedLrPanel::Nested {
                row_tree,
                col_tree,
                s,
                ..
            } => row_tree.stored_entries() + col_tree.stored_entries() + s.nrows() * s.ncols(),
        }
    }

    /// NestedLr×NestedLr matmul accumulating into a dense output: `out -= self · rhs`.
    ///
    /// Uses the nested-basis identity
    ///
    /// ```text
    /// A · B = U_A · S_A · (V_A^T · U_B) · S_B · V_B^T
    ///       = U_A · core · V_B^T   with core = S_A · M · S_B,
    ///                              M    = V_A^T · U_B.
    /// ```
    ///
    /// `M` is computed by a lockstep recursion over `A`'s column tree and
    /// `B`'s row tree (both partition the contraction axis identically when
    /// the panels were built from a common front, which the multifrontal
    /// walk guarantees). When the partitions don't match — or either side
    /// is the dense fallback variant — we materialise to dense and fall
    /// back to a triple loop.
    ///
    /// Cost (matched topology): two small dense products of size
    /// `k_root × l_root`, two tree walks of work `O((M + N) · k²)`, plus
    /// the inner small GEMMs at each level. No `m × p × n` GEMM ever runs.
    pub fn matmul_sub_into_dense(&self, rhs: &Self, out: &mut DenseMatrix<T>) {
        let m_a = self.n_rows();
        let n_b = rhs.n_cols();
        assert_eq!(
            self.n_cols(),
            rhs.n_rows(),
            "nested-LR matmul shape mismatch"
        );
        assert_eq!(out.nrows(), m_a);
        assert_eq!(out.ncols(), n_b);

        // Pull out the nested pieces from each side, falling back to a
        // dense GEMM whenever we hit a `Dense` variant.
        let (s_a, row_tree_a, col_tree_a) = match self {
            NestedLrPanel::Nested {
                s,
                row_tree,
                col_tree,
                ..
            } => (s, row_tree, col_tree),
            NestedLrPanel::Dense(d) => {
                let rhs_dense = rhs.to_dense();
                dense_gemm_sub(d, &rhs_dense, out);
                return;
            }
        };
        let (s_b, row_tree_b, col_tree_b) = match rhs {
            NestedLrPanel::Nested {
                s,
                row_tree,
                col_tree,
                ..
            } => (s, row_tree, col_tree),
            NestedLrPanel::Dense(d) => {
                let lhs_dense = self.to_dense();
                dense_gemm_sub(&lhs_dense, d, out);
                return;
            }
        };

        let m = match contract_basis(col_tree_a, row_tree_b) {
            Some(m) => m,
            None => {
                // Partition mismatch — should not happen for panels built
                // from a common front; defensive fallback.
                let lhs_dense = self.to_dense();
                let rhs_dense = rhs.to_dense();
                dense_gemm_sub(&lhs_dense, &rhs_dense, out);
                return;
            }
        };

        let k_a = s_a.nrows();
        let l_a = s_a.ncols();
        let k_b = s_b.nrows();
        let l_b = s_b.ncols();
        debug_assert_eq!(m.nrows(), l_a);
        debug_assert_eq!(m.ncols(), k_b);

        // tmp = S_A · M  (k_a × k_b)
        let mut tmp = DenseMatrix::<T>::zeros(k_a, k_b);
        for i in 0..k_a {
            for j in 0..k_b {
                let mut s = T::zero();
                for kk in 0..l_a {
                    s += s_a[[i, kk]] * m[[kk, j]];
                }
                tmp[[i, j]] = s;
            }
        }
        // core = tmp · S_B  (k_a × l_b)
        let mut core = DenseMatrix::<T>::zeros(k_a, l_b);
        for i in 0..k_a {
            for j in 0..l_b {
                let mut s = T::zero();
                for kk in 0..k_b {
                    s += tmp[[i, kk]] * s_b[[kk, j]];
                }
                core[[i, j]] = s;
            }
        }

        // Y = U_A · core  via one downward row-tree expansion per column
        // of core. Y has shape (m_a × l_b), staying small in the column
        // dimension since l_b is the nested root rank of B.
        let mut y = DenseMatrix::<T>::zeros(m_a, l_b);
        let mut col_buf = vec![T::zero(); k_a];
        let mut y_col = vec![T::zero(); m_a];
        for c in 0..l_b {
            for i in 0..k_a {
                col_buf[i] = core[[i, c]];
            }
            for yi in y_col.iter_mut() {
                *yi = T::zero();
            }
            downward_row(row_tree_a, T::one(), &col_buf, &mut y_col);
            for i in 0..m_a {
                y[[i, c]] = y_col[i];
            }
        }

        // out -= Y · V_B^T  via one downward col-tree expansion per row of Y.
        let mut row_buf = vec![T::zero(); l_b];
        let mut out_row = vec![T::zero(); n_b];
        for r in 0..m_a {
            for c in 0..l_b {
                row_buf[c] = y[[r, c]];
            }
            for oi in out_row.iter_mut() {
                *oi = T::zero();
            }
            downward_col(col_tree_b, T::one(), &row_buf, &mut out_row);
            for c in 0..n_b {
                out[[r, c]] -= out_row[c];
            }
        }
    }

    /// Materialise the panel as a dense matrix (debug / verification path).
    pub fn to_dense(&self) -> DenseMatrix<T> {
        let m = self.n_rows();
        let n = self.n_cols();
        match self {
            NestedLrPanel::Dense(d) => d.clone(),
            NestedLrPanel::Nested {
                row_tree,
                col_tree,
                s,
                ..
            } => {
                let mut out = DenseMatrix::<T>::zeros(m, n);
                // For each unit basis vector e_j (j = 0..n), compute
                // y = panel · e_j and write it as column j. Cheap enough
                // for a debug/round-trip path and avoids materialising
                // U_root / V_root.
                let mut e = vec![T::zero(); n];
                let mut y = vec![T::zero(); m];
                for j in 0..n {
                    e[j] = T::one();
                    for yi in y.iter_mut() {
                        *yi = T::zero();
                    }
                    // re-implement the gemv path with the locally-borrowed pieces.
                    let x_hat = upward_col(col_tree, &e);
                    let k = s.nrows();
                    let l = s.ncols();
                    let mut y_hat = vec![T::zero(); k];
                    for i in 0..k {
                        let mut acc = T::zero();
                        for jj in 0..l {
                            acc += s[[i, jj]] * x_hat[jj];
                        }
                        y_hat[i] = acc;
                    }
                    downward_row(row_tree, T::one(), &y_hat, &mut y);
                    for i in 0..m {
                        out[[i, j]] = y[i];
                    }
                    e[j] = T::zero();
                }
                out
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------

/// Build the row tree over rows `[row_start, row_end)` of `source` and
/// return it together with the projection `U_τ^H · A[row_range(τ), :]` of
/// shape `(k_τ × N)` at the root.
///
/// We recurse bottom-up. Each leaf computes its `U_i` directly via RRQR of
/// the dense source rows. Each internal node compresses the stacked
/// projections of its children — at no point do we have to materialise
/// `U_τ` for an internal node.
fn build_row_tree<T: ComplexField>(
    source: &DenseMatrix<T>,
    row_start: usize,
    row_end: usize,
    leaf_size: usize,
    epsilon: T::Real,
    precision: LowRankPrecision,
    use_parallel: bool,
) -> (NestedLrRowTree<T>, DenseMatrix<T>) {
    debug_assert!(row_start < row_end);
    let n = source.ncols();
    let m_local = row_end - row_start;

    if m_local <= leaf_size {
        // Leaf: U_i := Q from column-pivoted QR of A[row_range, :].
        let mut leaf_block = DenseMatrix::<T>::zeros(m_local, n);
        for i in 0..m_local {
            for j in 0..n {
                leaf_block[[i, j]] = source[[row_start + i, j]];
            }
        }
        let qr = truncate_rrqr(&leaf_block, epsilon);
        let u = qr.u;
        // proj = U^H · A[row_range, :] of shape (k × N). Since QR gives
        // A_leaf ≈ U · V^T with V of shape (N × k), U^H · A_leaf = V^T.
        let k = u.ncols();
        let mut proj = DenseMatrix::<T>::zeros(k, n);
        for kk in 0..k {
            for j in 0..n {
                proj[[kk, j]] = qr.v[[j, kk]];
            }
        }
        // Reconstruction guard: if the RRQR returned k = 0 (all-zero rows),
        // we still need a valid empty leaf. The translation/core code
        // handles k = 0 transparently.
        return (
            NestedLrRowTree::Leaf {
                row_start,
                row_end,
                u: MaybeHalf::from_dense_with_precision(u, precision),
            },
            proj,
        );
    }

    // Internal node: split in half. The translation operator R_τ is the
    // orthonormal compression of the stacked child projections.
    let split = row_start + m_local / 2;
    let build_left = || {
        build_row_tree(
            source,
            row_start,
            split,
            leaf_size,
            epsilon,
            precision,
            use_parallel,
        )
    };
    let build_right = || {
        build_row_tree(
            source,
            split,
            row_end,
            leaf_size,
            epsilon,
            precision,
            use_parallel,
        )
    };
    let ((left, proj_l), (right, proj_r)) = if use_parallel && m_local > leaf_size * 2 {
        rayon::join(build_left, build_right)
    } else {
        (build_left(), build_right())
    };
    let k_l = proj_l.nrows();
    let k_r = proj_r.nrows();
    debug_assert_eq!(proj_l.ncols(), n);
    debug_assert_eq!(proj_r.ncols(), n);

    // stacked = [proj_l; proj_r] of shape ((k_l + k_r) × N).
    let mut stacked = DenseMatrix::<T>::zeros(k_l + k_r, n);
    for i in 0..k_l {
        for j in 0..n {
            stacked[[i, j]] = proj_l[[i, j]];
        }
    }
    for i in 0..k_r {
        for j in 0..n {
            stacked[[k_l + i, j]] = proj_r[[i, j]];
        }
    }

    let qr = truncate_rrqr(&stacked, epsilon);
    let r_op = qr.u; // shape ((k_l + k_r) × k_τ), column-orthonormal.
    let k_tau = r_op.ncols();
    // proj_τ = R_τ^H · stacked = V^T (by the same identity as in the leaf path).
    let mut proj_tau = DenseMatrix::<T>::zeros(k_tau, n);
    for kk in 0..k_tau {
        for j in 0..n {
            proj_tau[[kk, j]] = qr.v[[j, kk]];
        }
    }

    (
        NestedLrRowTree::Node {
            row_start,
            row_end,
            left: Box::new(left),
            right: Box::new(right),
            r: MaybeHalf::from_dense_with_precision(r_op, precision),
        },
        proj_tau,
    )
}

/// Build the column tree over columns `[col_start, col_end)` of `row_proj`
/// (which has shape `(k_root × N)`). Symmetric to `build_row_tree`,
/// operating on columns of `row_proj`. The returned matrix is `S =
/// row_proj · V_root` of shape `(k_root × l_root)` — i.e. the panel's
/// panel core.
fn build_col_tree<T: ComplexField>(
    row_proj: &DenseMatrix<T>,
    col_start: usize,
    col_end: usize,
    leaf_size: usize,
    epsilon: T::Real,
    precision: LowRankPrecision,
    use_parallel: bool,
) -> (NestedLrColTree<T>, DenseMatrix<T>) {
    debug_assert!(col_start < col_end);
    let k_root = row_proj.nrows();
    let n_local = col_end - col_start;

    if n_local <= leaf_size {
        // Leaf: V_j := Q from RRQR of (row_proj[:, col_range])^T treated
        // as the "transposed" problem, then transposed back. Equivalent:
        // do RRQR on the rectangle (n_local × k_root) so the "U" output
        // *is* V_j directly.
        let mut leaf_block_t = DenseMatrix::<T>::zeros(n_local, k_root);
        for i in 0..n_local {
            for kk in 0..k_root {
                leaf_block_t[[i, kk]] = row_proj[[kk, col_start + i]];
            }
        }
        let qr = truncate_rrqr(&leaf_block_t, epsilon);
        let v = qr.u; // shape (n_local × l_j), the column-orthonormal leaf basis V_j.
        // RRQR gives `leaf_block_t = qr.u · qr.v^T = V_j · qr.v^T` exactly
        // (modulo truncation). Transposing both sides:
        //     row_proj[:, col_range] = leaf_block_t^T = qr.v · V_j^T.
        // So `contrib = qr.v` is precisely the matrix that satisfies
        //     row_proj[:, col_range] = contrib · V_j^T,
        // which is the only identity the internal-node recursion needs.
        // Importantly this holds without assuming any orthogonality of
        // `V_j` under the transpose inner product, so it works for both
        // real and complex `T` (RRQR's `qr.u` is column-orthonormal
        // under the Hermitian inner product `V_j^H V_j = I`, which is
        // not the same as `V_j^T V_j = I` when `T` is complex).
        // Shape of `qr.v` is `(k_root × l_j)`.
        let contrib = qr.v;
        return (
            NestedLrColTree::Leaf {
                col_start,
                col_end,
                v: MaybeHalf::from_dense_with_precision(v, precision),
            },
            contrib,
        );
    }

    let split = col_start + n_local / 2;
    let build_left = || {
        build_col_tree(
            row_proj,
            col_start,
            split,
            leaf_size,
            epsilon,
            precision,
            use_parallel,
        )
    };
    let build_right = || {
        build_col_tree(
            row_proj,
            split,
            col_end,
            leaf_size,
            epsilon,
            precision,
            use_parallel,
        )
    };
    let ((left, contrib_l), (right, contrib_r)) = if use_parallel && n_local > leaf_size * 2 {
        rayon::join(build_left, build_right)
    } else {
        (build_left(), build_right())
    };
    let l_l = contrib_l.ncols();
    let l_r = contrib_r.ncols();
    debug_assert_eq!(contrib_l.nrows(), k_root);
    debug_assert_eq!(contrib_r.nrows(), k_root);

    // For the column tree, we need to compress the stacked V columns the
    // same way as for rows — but the stacking is "horizontal" here:
    // hcat = [contrib_l, contrib_r] of shape (k_root × (l_l + l_r)).
    // Doing RRQR on hcat^T (shape (l_l + l_r) × k_root) gives a
    // column-orthonormal W_τ via the "U" output.
    let mut hcat_t = DenseMatrix::<T>::zeros(l_l + l_r, k_root);
    for j in 0..l_l {
        for i in 0..k_root {
            hcat_t[[j, i]] = contrib_l[[i, j]];
        }
    }
    for j in 0..l_r {
        for i in 0..k_root {
            hcat_t[[l_l + j, i]] = contrib_r[[i, j]];
        }
    }
    let qr = truncate_rrqr(&hcat_t, epsilon);
    let w_op = qr.u; // shape ((l_l + l_r) × l_τ).
    // Same identity as in the leaf path: contrib_τ = hcat · W = qr.v.
    let contrib_tau = qr.v;
    (
        NestedLrColTree::Node {
            col_start,
            col_end,
            left: Box::new(left),
            right: Box::new(right),
            w: MaybeHalf::from_dense_with_precision(w_op, precision),
        },
        contrib_tau,
    )
}

// ----------------------------------------------------------------------------
// GEMV passes
// ----------------------------------------------------------------------------

/// Upward column-tree contraction: returns `x̂ = V_root^H · x`.
fn upward_col<T: ComplexField>(tree: &NestedLrColTree<T>, x: &[T]) -> Vec<T> {
    match tree {
        NestedLrColTree::Leaf {
            col_start,
            col_end,
            v,
        } => {
            let n_local = col_end - col_start;
            let v_dense = v.as_dense();
            let v_ref: &DenseMatrix<T> = &v_dense;
            let l = v_ref.ncols();
            debug_assert_eq!(v_ref.nrows(), n_local);
            let mut hat = vec![T::zero(); l];
            for ll in 0..l {
                let mut s = T::zero();
                for j in 0..n_local {
                    s += v_ref[[j, ll]] * x[col_start + j];
                }
                hat[ll] = s;
            }
            hat
        }
        NestedLrColTree::Node { left, right, w, .. } => {
            let hat_l = upward_col(left, x);
            let hat_r = upward_col(right, x);
            let l_l = hat_l.len();
            let l_r = hat_r.len();
            let w_dense = w.as_dense();
            let w_ref: &DenseMatrix<T> = &w_dense;
            let l_tau = w_ref.ncols();
            debug_assert_eq!(w_ref.nrows(), l_l + l_r);
            let mut hat = vec![T::zero(); l_tau];
            for ll in 0..l_tau {
                let mut s = T::zero();
                for i in 0..l_l {
                    s += w_ref[[i, ll]] * hat_l[i];
                }
                for i in 0..l_r {
                    s += w_ref[[l_l + i, ll]] * hat_r[i];
                }
                hat[ll] = s;
            }
            hat
        }
    }
}

/// Downward row-tree expansion: `y += alpha · U_root · y_hat_root`.
fn downward_row<T: ComplexField>(tree: &NestedLrRowTree<T>, alpha: T, y_hat: &[T], y: &mut [T]) {
    match tree {
        NestedLrRowTree::Leaf {
            row_start,
            row_end,
            u,
        } => {
            let m_local = row_end - row_start;
            let u_dense = u.as_dense();
            let u_ref: &DenseMatrix<T> = &u_dense;
            let k = u_ref.ncols();
            debug_assert_eq!(u_ref.nrows(), m_local);
            debug_assert_eq!(y_hat.len(), k);
            for i in 0..m_local {
                let mut s = T::zero();
                for kk in 0..k {
                    s += u_ref[[i, kk]] * y_hat[kk];
                }
                y[row_start + i] += alpha * s;
            }
        }
        NestedLrRowTree::Node { left, right, r, .. } => {
            let k_l = left.rank();
            let k_r = right.rank();
            let r_dense = r.as_dense();
            let r_ref: &DenseMatrix<T> = &r_dense;
            let k_tau = r_ref.ncols();
            debug_assert_eq!(r_ref.nrows(), k_l + k_r);
            debug_assert_eq!(y_hat.len(), k_tau);
            let mut y_hat_l = vec![T::zero(); k_l];
            let mut y_hat_r = vec![T::zero(); k_r];
            for i in 0..k_l {
                let mut s = T::zero();
                for kk in 0..k_tau {
                    s += r_ref[[i, kk]] * y_hat[kk];
                }
                y_hat_l[i] = s;
            }
            for i in 0..k_r {
                let mut s = T::zero();
                for kk in 0..k_tau {
                    s += r_ref[[k_l + i, kk]] * y_hat[kk];
                }
                y_hat_r[i] = s;
            }
            downward_row(left, alpha, &y_hat_l, y);
            downward_row(right, alpha, &y_hat_r, y);
        }
    }
}

/// Upward row-tree contraction (transpose path): returns `x̂ = U_root^T · x`.
fn upward_row<T: ComplexField>(tree: &NestedLrRowTree<T>, x: &[T]) -> Vec<T> {
    match tree {
        NestedLrRowTree::Leaf {
            row_start,
            row_end,
            u,
        } => {
            let m_local = row_end - row_start;
            let u_dense = u.as_dense();
            let u_ref: &DenseMatrix<T> = &u_dense;
            let k = u_ref.ncols();
            debug_assert_eq!(u_ref.nrows(), m_local);
            let mut hat = vec![T::zero(); k];
            for kk in 0..k {
                let mut s = T::zero();
                for i in 0..m_local {
                    s += u_ref[[i, kk]] * x[row_start + i];
                }
                hat[kk] = s;
            }
            hat
        }
        NestedLrRowTree::Node { left, right, r, .. } => {
            let hat_l = upward_row(left, x);
            let hat_r = upward_row(right, x);
            let k_l = hat_l.len();
            let k_r = hat_r.len();
            let r_dense = r.as_dense();
            let r_ref: &DenseMatrix<T> = &r_dense;
            let k_tau = r_ref.ncols();
            debug_assert_eq!(r_ref.nrows(), k_l + k_r);
            let mut hat = vec![T::zero(); k_tau];
            for kk in 0..k_tau {
                let mut s = T::zero();
                for i in 0..k_l {
                    s += r_ref[[i, kk]] * hat_l[i];
                }
                for i in 0..k_r {
                    s += r_ref[[k_l + i, kk]] * hat_r[i];
                }
                hat[kk] = s;
            }
            hat
        }
    }
}

/// Downward column-tree expansion (transpose path): `y += alpha · V_root · y_hat`.
fn downward_col<T: ComplexField>(tree: &NestedLrColTree<T>, alpha: T, y_hat: &[T], y: &mut [T]) {
    match tree {
        NestedLrColTree::Leaf {
            col_start,
            col_end,
            v,
        } => {
            let n_local = col_end - col_start;
            let v_dense = v.as_dense();
            let v_ref: &DenseMatrix<T> = &v_dense;
            let l = v_ref.ncols();
            debug_assert_eq!(v_ref.nrows(), n_local);
            debug_assert_eq!(y_hat.len(), l);
            for j in 0..n_local {
                let mut s = T::zero();
                for ll in 0..l {
                    s += v_ref[[j, ll]] * y_hat[ll];
                }
                y[col_start + j] += alpha * s;
            }
        }
        NestedLrColTree::Node { left, right, w, .. } => {
            let l_l = left.rank();
            let l_r = right.rank();
            let w_dense = w.as_dense();
            let w_ref: &DenseMatrix<T> = &w_dense;
            let l_tau = w_ref.ncols();
            debug_assert_eq!(w_ref.nrows(), l_l + l_r);
            debug_assert_eq!(y_hat.len(), l_tau);
            let mut y_hat_l = vec![T::zero(); l_l];
            let mut y_hat_r = vec![T::zero(); l_r];
            for i in 0..l_l {
                let mut s = T::zero();
                for ll in 0..l_tau {
                    s += w_ref[[i, ll]] * y_hat[ll];
                }
                y_hat_l[i] = s;
            }
            for i in 0..l_r {
                let mut s = T::zero();
                for ll in 0..l_tau {
                    s += w_ref[[l_l + i, ll]] * y_hat[ll];
                }
                y_hat_r[i] = s;
            }
            downward_col(left, alpha, &y_hat_l, y);
            downward_col(right, alpha, &y_hat_r, y);
        }
    }
}

// ----------------------------------------------------------------------------
// NestedLr×NestedLr matmul helpers
// ----------------------------------------------------------------------------

/// Computes `M = V_A^T · U_B` by walking `A`'s column tree and `B`'s row
/// tree in lockstep, using the recursive identity
///
/// ```text
/// V_τ_A^T · U_τ_B = W_A^T · blkdiag(M_l, M_r) · R_B,
/// ```
///
/// so neither `V_A` nor `U_B` is ever materialised. Returns `None` when
/// the two trees do not partition the contraction axis identically — the
/// caller falls back to a dense matmul in that case.
fn contract_basis<T: ComplexField>(
    col_tree_a: &NestedLrColTree<T>,
    row_tree_b: &NestedLrRowTree<T>,
) -> Option<DenseMatrix<T>> {
    match (col_tree_a, row_tree_b) {
        (
            NestedLrColTree::Leaf {
                col_start: cs_a,
                col_end: ce_a,
                v: v_a,
            },
            NestedLrRowTree::Leaf {
                row_start: rs_b,
                row_end: re_b,
                u: u_b,
            },
        ) => {
            if ce_a - cs_a != re_b - rs_b {
                return None;
            }
            let v_dense = v_a.as_dense();
            let u_dense = u_b.as_dense();
            let v_ref: &DenseMatrix<T> = &v_dense;
            let u_ref: &DenseMatrix<T> = &u_dense;
            let n = v_ref.nrows();
            if n != u_ref.nrows() {
                return None;
            }
            let l = v_ref.ncols();
            let k = u_ref.ncols();
            let mut m = DenseMatrix::<T>::zeros(l, k);
            for ll in 0..l {
                for kk in 0..k {
                    let mut s = T::zero();
                    for ii in 0..n {
                        s += v_ref[[ii, ll]] * u_ref[[ii, kk]];
                    }
                    m[[ll, kk]] = s;
                }
            }
            Some(m)
        }
        (
            NestedLrColTree::Node {
                left: ca_l,
                right: ca_r,
                w: w_a,
                ..
            },
            NestedLrRowTree::Node {
                left: rb_l,
                right: rb_r,
                r: r_b,
                ..
            },
        ) => {
            let m_l = contract_basis(ca_l, rb_l)?;
            let m_r = contract_basis(ca_r, rb_r)?;
            let l_l = m_l.nrows();
            let l_r = m_r.nrows();
            let k_l = m_l.ncols();
            let k_r = m_r.ncols();
            let w_dense = w_a.as_dense();
            let r_dense = r_b.as_dense();
            let w_ref: &DenseMatrix<T> = &w_dense;
            let r_ref: &DenseMatrix<T> = &r_dense;
            let l_tau_a = w_ref.ncols();
            let k_tau_b = r_ref.ncols();
            if w_ref.nrows() != l_l + l_r || r_ref.nrows() != k_l + k_r {
                return None;
            }

            // mid = blkdiag(m_l, m_r) · R_B  of shape ((l_l + l_r) × k_tau_b).
            let mut mid = DenseMatrix::<T>::zeros(l_l + l_r, k_tau_b);
            for i in 0..l_l {
                for kk in 0..k_tau_b {
                    let mut s = T::zero();
                    for j in 0..k_l {
                        s += m_l[[i, j]] * r_ref[[j, kk]];
                    }
                    mid[[i, kk]] = s;
                }
            }
            for i in 0..l_r {
                for kk in 0..k_tau_b {
                    let mut s = T::zero();
                    for j in 0..k_r {
                        s += m_r[[i, j]] * r_ref[[k_l + j, kk]];
                    }
                    mid[[l_l + i, kk]] = s;
                }
            }

            // result = W_A^T · mid  of shape (l_tau_a × k_tau_b).
            let mut result = DenseMatrix::<T>::zeros(l_tau_a, k_tau_b);
            for i in 0..l_tau_a {
                for j in 0..k_tau_b {
                    let mut s = T::zero();
                    for kk in 0..(l_l + l_r) {
                        s += w_ref[[kk, i]] * mid[[kk, j]];
                    }
                    result[[i, j]] = s;
                }
            }
            Some(result)
        }
        _ => None,
    }
}

/// Sum of cluster ranks and node count across a nested row tree.
fn row_rank_sum_count<T: ComplexField>(tree: &NestedLrRowTree<T>) -> (usize, usize) {
    match tree {
        NestedLrRowTree::Leaf { u, .. } => (u.ncols(), 1),
        NestedLrRowTree::Node { left, right, r, .. } => {
            let (ls, lc) = row_rank_sum_count(left);
            let (rs, rc) = row_rank_sum_count(right);
            (ls + rs + r.ncols(), lc + rc + 1)
        }
    }
}

/// Sum of cluster ranks and node count across a nested column tree.
fn col_rank_sum_count<T: ComplexField>(tree: &NestedLrColTree<T>) -> (usize, usize) {
    match tree {
        NestedLrColTree::Leaf { v, .. } => (v.ncols(), 1),
        NestedLrColTree::Node { left, right, w, .. } => {
            let (ls, lc) = col_rank_sum_count(left);
            let (rs, rc) = col_rank_sum_count(right);
            (ls + rs + w.ncols(), lc + rc + 1)
        }
    }
}

/// `out -= a · b`, BLAS-3 dense GEMM. Used by the NestedLr×NestedLr path
/// whenever either operand is the dense fallback variant.
fn dense_gemm_sub<T: ComplexField>(
    a: &DenseMatrix<T>,
    b: &DenseMatrix<T>,
    out: &mut DenseMatrix<T>,
) {
    let m = a.nrows();
    let p = a.ncols();
    let n = b.ncols();
    debug_assert_eq!(b.nrows(), p);
    debug_assert_eq!(out.nrows(), m);
    debug_assert_eq!(out.ncols(), n);
    let product = a.matmul(b);
    for (value, update) in out.as_mut_slice().iter_mut().zip(product.as_slice()) {
        *value -= *update;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn random_smooth_panel(m: usize, n: usize) -> DenseMatrix<f64> {
        // A smooth kernel sampled on a regular grid is the canonical
        // low-rank-friendly case for nested bases. Use 1 / (1 + ‖x_i − y_j‖²).
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let xi = (i as f64) / (m as f64);
                let yj = 2.0 + (j as f64) / (n as f64); // separated supports
                data[i * n + j] = 1.0 / (1.0 + (xi - yj).powi(2));
            }
        }
        DenseMatrix::<f64>::from_row_major_vec(m, n, data)
    }

    fn dense_gemv(m: &DenseMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; m.nrows()];
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                y[i] += m[[i, j]] * x[j];
            }
        }
        y
    }

    #[test]
    fn dense_fallback_for_small_panel() {
        // A 6 × 6 panel with leaf_size 8 must fall back to Dense — too
        // small to benefit from the nested-basis machinery.
        let mat = random_smooth_panel(6, 6);
        let h = NestedLrPanel::from_dense(&mat, 8, 1e-10);
        match &h {
            NestedLrPanel::Dense(d) => assert_eq!(d.nrows() * d.ncols(), 36),
            _ => panic!("expected dense fallback"),
        }
        let stored = h.stored_entries();
        assert_eq!(stored, 36);
    }

    #[test]
    fn round_trip_dense_eps_zero() {
        // ε = 0 ⇒ no truncation; the nested panel must reproduce the source
        // exactly (up to floating-point round-off) via to_dense.
        let m = 32;
        let n = 24;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense(&mat, 4, 0.0);
        let reconstructed = h.to_dense();
        assert_eq!(reconstructed.nrows(), m);
        assert_eq!(reconstructed.ncols(), n);
        for i in 0..m {
            for j in 0..n {
                let diff = (mat[[i, j]] - reconstructed[[i, j]]).abs();
                assert!(diff < 1e-10, "[{},{}] diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn parallel_build_matches_serial() {
        let m = 48;
        let n = 40;
        let mat = random_smooth_panel(m, n);
        let serial = NestedLrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            4,
            0.0,
            LowRankPrecision::Full,
            false,
        );
        let parallel = NestedLrPanel::from_dense_with_precision_maybe_parallel(
            &mat,
            4,
            0.0,
            LowRankPrecision::Full,
            true,
        );

        assert_eq!(serial.stored_entries(), parallel.stored_entries());
        assert_eq!(serial.average_rank(), parallel.average_rank());
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
    fn gemv_matches_dense() {
        let m = 40;
        let n = 32;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense(&mat, 4, 1e-12);
        let x: Vec<f64> = (0..n).map(|j| (j as f64 + 0.3).sin()).collect();
        let y_ref = dense_gemv(&mat, &x);
        let mut y_h = vec![0.0; m];
        h.gemv_into(1.0, &x, &mut y_h);
        for i in 0..m {
            let diff = (y_h[i] - y_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "gemv [{}]: nlr={} dense={} diff={}",
                i,
                y_h[i],
                y_ref[i],
                diff
            );
        }
    }

    #[test]
    fn gemv_transpose_matches_dense() {
        let m = 40;
        let n = 32;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense(&mat, 4, 1e-12);
        let x: Vec<f64> = (0..m).map(|i| (i as f64 - 0.7).cos()).collect();
        let mut y_ref = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                y_ref[j] += mat[[i, j]] * x[i];
            }
        }
        let mut y_h = vec![0.0; n];
        h.gemv_transpose_into(1.0, &x, &mut y_h);
        for j in 0..n {
            let diff = (y_h[j] - y_ref[j]).abs();
            assert!(
                diff < 1e-9,
                "gemv^T [{}]: nlr={} dense={} diff={}",
                j,
                y_h[j],
                y_ref[j],
                diff
            );
        }
    }

    #[test]
    fn smooth_kernel_compresses_well() {
        // 64 × 64 smooth kernel has rapidly-decaying singular values. With
        // ε = 1e-8 nested storage should be a small fraction of dense.
        let m = 64;
        let n = 64;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense(&mat, 4, 1e-8);
        let stored = h.stored_entries();
        let dense = m * n;
        assert!(
            stored < dense / 2,
            "nested panel not compressing: stored {} vs dense {}",
            stored,
            dense
        );

        // Round-trip accuracy: A · x recoverable to ≈ 1e-6.
        let x: Vec<f64> = (0..n).map(|j| (j as f64 * 0.41).sin()).collect();
        let y_ref = dense_gemv(&mat, &x);
        let mut y_h = vec![0.0; m];
        h.gemv_into(1.0, &x, &mut y_h);
        let mut err_sq = 0.0;
        let mut ref_sq = 0.0;
        for i in 0..m {
            err_sq += (y_h[i] - y_ref[i]).powi(2);
            ref_sq += y_ref[i].powi(2);
        }
        let rel = (err_sq / ref_sq).sqrt();
        assert!(rel < 1e-6, "rel error {} too large", rel);
    }

    #[test]
    fn half_precision_round_trip_within_tolerance() {
        // The defining accuracy claim for nested half storage: gemv via a
        // half-precision panel matches dense reference to better than the
        // ~1e-3 per-block f16 error floor times the tree depth.
        let m = 64;
        let n = 64;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense_with_precision(&mat, 8, 1e-8, LowRankPrecision::Half);
        let x: Vec<f64> = (0..n).map(|j| (j as f64 + 0.5).sin()).collect();
        let y_ref = dense_gemv(&mat, &x);
        let mut y_h = vec![0.0; m];
        h.gemv_into(1.0, &x, &mut y_h);
        let mut err_sq = 0.0;
        let mut ref_sq = 0.0;
        for i in 0..m {
            err_sq += (y_h[i] - y_ref[i]).powi(2);
            ref_sq += y_ref[i].powi(2);
        }
        let rel = (err_sq / ref_sq).sqrt();
        assert!(
            rel < 5e-2,
            "half-precision relative error {} above tolerance",
            rel
        );
    }

    #[test]
    fn half_precision_cuts_storage_versus_full_on_f64() {
        // For `f64`, an `f16`-stored block is 1/4 the size of full. With
        // most of a nested-basis panel's memory in the leaf U/V and translation
        // operators (everything except the small core S), Half should
        // visibly shrink the panel.
        let m = 128;
        let n = 128;
        let mat = random_smooth_panel(m, n);
        let full = NestedLrPanel::from_dense_with_precision(&mat, 8, 1e-8, LowRankPrecision::Full);
        let half = NestedLrPanel::from_dense_with_precision(&mat, 8, 1e-8, LowRankPrecision::Half);
        let full_e = full.stored_entries();
        let half_e = half.stored_entries();
        assert!(
            half_e < full_e,
            "Half should reduce nested storage: full={} half={}",
            full_e,
            half_e
        );
    }

    #[test]
    fn half_precision_silently_falls_back_for_complex() {
        // f16 only supports real values; complex `T` must silently keep
        // full precision. We don't have a programmatic introspection
        // helper, so verify indirectly: the gemv on a Complex64 panel
        // built with Half must still match dense to round-off, which only
        // holds if the Full path was used.
        let m = 16;
        let n = 16;
        let mut data = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let r = 1.0 / (1.0 + ((i as f64) - (j as f64) - 8.0).powi(2));
                let im = 0.3 * (((i + j) as f64) * 0.2).sin();
                data[i * n + j] = Complex64::new(r, im);
            }
        }
        let mat = DenseMatrix::<Complex64>::from_row_major_vec(m, n, data);
        let h = NestedLrPanel::from_dense_with_precision(&mat, 4, 1e-10, LowRankPrecision::Half);
        let x: Vec<Complex64> = (0..n)
            .map(|j| Complex64::new((j as f64).sin(), (j as f64).cos()))
            .collect();
        let mut y_ref = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..m {
            for j in 0..n {
                y_ref[i] += mat[[i, j]] * x[j];
            }
        }
        let mut y_h = vec![Complex64::new(0.0, 0.0); m];
        h.gemv_into(Complex64::new(1.0, 0.0), &x, &mut y_h);
        for i in 0..m {
            let diff = (y_h[i] - y_ref[i]).norm();
            assert!(
                diff < 1e-8,
                "complex Half should silently use Full: [{}] diff={}",
                i,
                diff
            );
        }
    }

    #[test]
    fn average_rank_none_for_dense_fallback() {
        // A panel that fell back to Dense has no nested basis to summarise.
        let mat = random_smooth_panel(6, 6);
        let h = NestedLrPanel::from_dense(&mat, 8, 1e-10);
        assert!(matches!(h, NestedLrPanel::Dense(_)));
        assert!(h.average_rank().is_none());
    }

    #[test]
    fn average_rank_bounded_for_smooth_kernel() {
        // For a smooth, low-numerical-rank kernel the average cluster rank
        // is small — far below the leaf size — because every level
        // compresses heavily.
        let m = 128;
        let n = 128;
        let mat = random_smooth_panel(m, n);
        let h = NestedLrPanel::from_dense(&mat, 8, 1e-8);
        let avg = h.average_rank().expect("nested panel");
        assert!(
            avg > 0.0 && avg < 8.0,
            "smooth-kernel avg rank {} should be a small constant",
            avg
        );
    }

    #[test]
    fn full_rank_panel_falls_back_to_dense_storage() {
        // A genuinely full-rank panel (hash-mixed noise) has no global
        // low-rank structure: the nested root rank grows until the nested
        // representation costs at least as much as dense. The storage-gain
        // guard must fall back to Dense, so `stored_entries` never exceeds
        // `m * n` — `from_dense` must never silently return a
        // representation that costs more memory than dense.
        let m = 64;
        let n = 64;
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let h = ((i as u64).wrapping_mul(2654435761))
                    .wrapping_add((j as u64).wrapping_mul(4294967291))
                    .wrapping_mul(0x9E3779B97F4A7C15);
                data[i * n + j] = ((h >> 11) as f64) / ((1u64 << 53) as f64);
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);

        let h = NestedLrPanel::from_dense(&mat, 8, 1e-8);
        assert!(
            matches!(h, NestedLrPanel::Dense(_)),
            "full-rank panel should fall back to dense storage"
        );
        assert!(h.stored_entries() <= m * n);
        assert!(h.average_rank().is_none());

        // The smooth kernel, by contrast, is globally low-rank and must
        // stay in the nested form with strictly smaller storage.
        let smooth = random_smooth_panel(m, n);
        let h_smooth = NestedLrPanel::from_dense(&smooth, 8, 1e-8);
        assert!(matches!(h_smooth, NestedLrPanel::Nested { .. }));
        assert!(h_smooth.stored_entries() < m * n);
    }

    #[test]
    fn block_low_rank_but_not_globally_low_rank_falls_back() {
        // Block-diagonal panel: every diagonal 8×8 block has rank 4, so a
        // block-wise (BLR) view is highly compressible — but the blocks
        // are independent, so the *global* rank is 4 per block and the
        // nested root rank grows linearly with the block count until the
        // nested representation costs more than dense. The storage-gain
        // guard must kick in.
        let m = 64;
        let n = 64;
        let mut data = vec![0.0; m * n];
        for blk in 0..8 {
            for p in 0..8 {
                for q in 0..8 {
                    // Rank-4 block: sum of 4 outer products.
                    let mut v = 0.0;
                    for r in 0..4 {
                        v += (((p + 1) * (r + 1) + blk) as f64).sin()
                            * (((q + 1) * (r + 2) + blk) as f64 * 0.7).cos();
                    }
                    data[(blk * 8 + p) * n + (blk * 8 + q)] = v;
                }
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);

        let h = NestedLrPanel::from_dense(&mat, 8, 1e-8);
        assert!(
            matches!(h, NestedLrPanel::Dense(_)),
            "block-low-rank but globally high-rank panel should fall back to dense"
        );
        assert!(
            h.stored_entries() <= m * n,
            "stored {} must not exceed dense {}",
            h.stored_entries(),
            m * n
        );

        // The dense fallback reproduces the panel exactly.
        let round_trip = h.to_dense();
        for i in 0..m {
            for j in 0..n {
                assert_eq!(round_trip[[i, j]], mat[[i, j]]);
            }
        }
    }

    #[test]
    fn nested_lr_beats_hodlr_on_smooth_kernel_memory() {
        // The defining motivation: at fixed accuracy, the nested basis stores fewer
        // entries than the non-nested HODLR for a low-rank-friendly kernel
        // because leaf bases aren't replicated across levels.
        use super::super::hodlr::HodlrPanel;
        let m = 128;
        let n = 128;
        let mat = random_smooth_panel(m, n);
        let nlr = NestedLrPanel::from_dense(&mat, 8, 1e-8);
        let hodlr = HodlrPanel::from_dense(&mat, 8, 1e-8);
        let nlr_bytes = nlr.stored_entries();
        let hodlr_bytes = hodlr.stored_entries();
        assert!(
            nlr_bytes <= hodlr_bytes,
            "nested-LR should beat HODLR on smooth kernel: nlr={} hodlr={}",
            nlr_bytes,
            hodlr_bytes
        );
    }

    #[test]
    fn matmul_sub_matches_dense() {
        // Build two nested panels A (32×24) and B (24×28), then verify that
        // `out -= A · B` agrees with the dense triple-loop reference to
        // round-off. Both panels use the same block_size on the matched
        // axis (24), so the lockstep contract_basis path is exercised.
        let m_a = 32;
        let p = 24;
        let n_b = 28;
        let mat_a = random_smooth_panel(m_a, p);
        // For B, build a different smooth kernel with the contraction
        // axis as the row axis.
        let mut data_b = vec![0.0; p * n_b];
        for i in 0..p {
            for j in 0..n_b {
                let xi = (i as f64) / (p as f64);
                let yj = 3.0 + (j as f64) / (n_b as f64);
                data_b[i * n_b + j] = 1.0 / (1.0 + (xi - yj).powi(2));
            }
        }
        let mat_b = DenseMatrix::<f64>::from_row_major_vec(p, n_b, data_b);

        let leaf = 4;
        let a = NestedLrPanel::from_dense(&mat_a, leaf, 1e-12);
        let b = NestedLrPanel::from_dense(&mat_b, leaf, 1e-12);

        // Start `out` from a fixed nonzero state to make sure the routine
        // really *subtracts* and doesn't overwrite.
        let mut out = DenseMatrix::<f64>::zeros(m_a, n_b);
        for i in 0..m_a {
            for j in 0..n_b {
                out[[i, j]] = ((i + j) as f64) * 0.01;
            }
        }
        let mut out_ref = out.clone();
        a.matmul_sub_into_dense(&b, &mut out);

        // Reference: dense GEMM subtraction.
        for i in 0..m_a {
            for j in 0..n_b {
                let mut s = 0.0;
                for kk in 0..p {
                    s += mat_a[[i, kk]] * mat_b[[kk, j]];
                }
                out_ref[[i, j]] -= s;
            }
        }

        for i in 0..m_a {
            for j in 0..n_b {
                let diff = (out[[i, j]] - out_ref[[i, j]]).abs();
                assert!(
                    diff < 1e-8,
                    "matmul_sub [{},{}]: nlr={} ref={} diff={}",
                    i,
                    j,
                    out[[i, j]],
                    out_ref[[i, j]],
                    diff
                );
            }
        }
    }

    #[test]
    fn matmul_sub_matches_dense_complex() {
        // Same identity as `matmul_sub_matches_dense`, but with Complex64
        // operands. The codebase uses plain-transpose semantics for the nested panel
        // (`A ≈ U_root · S · V_root^T`, no conjugation), and the leaf
        // construction relies on `leaf_block_t = qr.u · qr.v^T` rather
        // than any `V^T V = I` orthogonality identity. This test pins
        // down that the complex path actually closes.
        let m_a = 24;
        let p = 16;
        let n_b = 20;
        // Two complex smooth kernels with separated supports (so the
        // panels are low-rank-friendly and the test is fast).
        let mut data_a = vec![Complex64::new(0.0, 0.0); m_a * p];
        for i in 0..m_a {
            for j in 0..p {
                let xi = (i as f64) / (m_a as f64);
                let yj = 2.0 + (j as f64) / (p as f64);
                let re = 1.0 / (1.0 + (xi - yj).powi(2));
                let im = 0.3 * ((xi + yj) * 0.7).sin();
                data_a[i * p + j] = Complex64::new(re, im);
            }
        }
        let mat_a = DenseMatrix::<Complex64>::from_row_major_vec(m_a, p, data_a);
        let mut data_b = vec![Complex64::new(0.0, 0.0); p * n_b];
        for i in 0..p {
            for j in 0..n_b {
                let xi = (i as f64) / (p as f64);
                let yj = 3.0 + (j as f64) / (n_b as f64);
                let re = 1.0 / (1.0 + (xi - yj).powi(2));
                let im = 0.2 * ((xi - yj) * 1.3).cos();
                data_b[i * n_b + j] = Complex64::new(re, im);
            }
        }
        let mat_b = DenseMatrix::<Complex64>::from_row_major_vec(p, n_b, data_b);

        let leaf = 4;
        let a = NestedLrPanel::from_dense(&mat_a, leaf, 1e-12);
        let b = NestedLrPanel::from_dense(&mat_b, leaf, 1e-12);

        let mut out = DenseMatrix::<Complex64>::zeros(m_a, n_b);
        for i in 0..m_a {
            for j in 0..n_b {
                out[[i, j]] = Complex64::new(((i + j) as f64) * 0.01, 0.0);
            }
        }
        let mut out_ref = out.clone();
        a.matmul_sub_into_dense(&b, &mut out);

        for i in 0..m_a {
            for j in 0..n_b {
                let mut s = Complex64::new(0.0, 0.0);
                for kk in 0..p {
                    s += mat_a[[i, kk]] * mat_b[[kk, j]];
                }
                out_ref[[i, j]] -= s;
            }
        }

        for i in 0..m_a {
            for j in 0..n_b {
                let diff = (out[[i, j]] - out_ref[[i, j]]).norm();
                assert!(
                    diff < 1e-8,
                    "complex matmul_sub [{},{}]: diff = {} (nlr={:?} ref={:?})",
                    i,
                    j,
                    diff,
                    out[[i, j]],
                    out_ref[[i, j]]
                );
            }
        }
    }

    #[test]
    fn matmul_sub_dense_fallback_when_lhs_small() {
        // A small panel ⇒ Dense variant (m ≤ leaf); rhs is a larger smooth
        // panel that stays Nested. The mixed Dense×Nested matmul must
        // still match the dense reference via the fallback path.
        let m_a = 5;
        let p = 64;
        let n_b = 48;
        let mat_a = random_smooth_panel(m_a, p);
        let mat_b = random_smooth_panel(p, n_b);

        let a = NestedLrPanel::from_dense(&mat_a, 8, 1e-12); // Dense (m ≤ leaf)
        let b = NestedLrPanel::from_dense(&mat_b, 4, 1e-12); // Nested
        assert!(matches!(a, NestedLrPanel::Dense(_)));
        assert!(matches!(b, NestedLrPanel::Nested { .. }));

        let mut out = DenseMatrix::<f64>::zeros(m_a, n_b);
        a.matmul_sub_into_dense(&b, &mut out);

        let mut out_ref = DenseMatrix::<f64>::zeros(m_a, n_b);
        for i in 0..m_a {
            for j in 0..n_b {
                let mut s = 0.0;
                for kk in 0..p {
                    s += mat_a[[i, kk]] * mat_b[[kk, j]];
                }
                out_ref[[i, j]] -= s;
            }
        }
        for i in 0..m_a {
            for j in 0..n_b {
                assert!((out[[i, j]] - out_ref[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn complex_panel_round_trip() {
        // Verify the Complex64 path is intact (the trait bounds are
        // exercised, but RRQR uses the conjugate-correct identity that
        // makes the leaf shortcut `proj = qr.v^T` work).
        let m = 16;
        let n = 16;
        let mut data = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let r = 1.0 / (1.0 + ((i as f64) - (j as f64) - 8.0).powi(2));
                let im = 0.3 * (((i + j) as f64) * 0.2).sin();
                data[i * n + j] = Complex64::new(r, im);
            }
        }
        let mat = DenseMatrix::<Complex64>::from_row_major_vec(m, n, data);
        let h = NestedLrPanel::from_dense(&mat, 4, 1e-10);
        let x: Vec<Complex64> = (0..n)
            .map(|j| Complex64::new((j as f64).sin(), (j as f64).cos()))
            .collect();
        let mut y_ref = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..m {
            for j in 0..n {
                y_ref[i] += mat[[i, j]] * x[j];
            }
        }
        let mut y_h = vec![Complex64::new(0.0, 0.0); m];
        h.gemv_into(Complex64::new(1.0, 0.0), &x, &mut y_h);
        for i in 0..m {
            let diff = (y_h[i] - y_ref[i]).norm();
            assert!(diff < 1e-8, "complex gemv [{}] diff={}", i, diff);
        }
    }
}
