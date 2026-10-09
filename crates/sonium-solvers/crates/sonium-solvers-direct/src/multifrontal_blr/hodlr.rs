//! Hierarchical Off-Diagonal Low-Rank (HODLR) panel representation.
//!
//! Where [`BlrPanel`](super::blr::BlrPanel) stores a *flat* grid of small
//! independently-compressed blocks, [`HodlrPanel`] partitions the panel
//! recursively into a 2×2 split:
//!
//! ```text
//!   ┌───────┬───────┐
//!   │  A11  │  A12  │
//!   ├───────┼───────┤
//!   │  A21  │  A22  │
//!   └───────┴───────┘
//! ```
//!
//! The diagonal blocks `A11, A22` are themselves [`HodlrPanel`]s recursively;
//! the off-diagonal blocks `A12, A21` are wrapped in [`OffDiagBlock`] which
//! is *either* a single low-rank pair `(U, V)` or a dense matrix — see the
//! `from_dense` constructor for the storage-saving threshold. Recursion
//! bottoms out at a configurable leaf size where the panel is stored dense.
//!
//! Compared to BLR:
//! - HODLR exploits a *single* large low-rank approximation of each
//!   off-diagonal region instead of many small ones. For panels with smooth
//!   underlying kernels (typical for 3D PDEs) this can compress more
//!   aggressively and store fewer entries.
//! - HODLR is more sensitive to the rank growth of A12/A21 at coarse
//!   levels; BLR handles partially-low-rank panels more gracefully.
//!
//! Storage discipline: every off-diagonal block goes through
//! [`maybe_compress`](super::truncation::maybe_compress) — when low-rank
//! storage would not be smaller than dense (`r·(m+n) ≥ m·n`), the block
//! falls back to [`OffDiagBlock::Dense`]. This guarantees
//! `HodlrPanel::stored_entries() ≤ m·n` even at `ε = 0`.
//!
//! This implementation gives a *non-nested-basis* HODLR — every level uses
//! its own off-diagonal factors. The fully-nested HSS variant (where
//! children's U / V columns are linear combinations of the parent's) is
//! deferred.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;

use super::scalar_is_finite;
use super::truncation::maybe_compress;

/// Storage for one off-diagonal block in a HODLR node.
pub enum OffDiagBlock<T: ComplexField> {
    /// Full dense storage. Used when low-rank wouldn't save entries.
    Dense(DenseMatrix<T>),
    /// Low-rank: `block ≈ U · V^T` with `U: m × r`, `V: n × r`.
    LowRank {
        u: DenseMatrix<T>,
        v: DenseMatrix<T>,
    },
}

impl<T: ComplexField> OffDiagBlock<T> {
    fn is_finite(&self) -> bool {
        match self {
            OffDiagBlock::Dense(matrix) => matrix.as_slice().iter().copied().all(scalar_is_finite),
            OffDiagBlock::LowRank { u, v } => {
                u.as_slice().iter().copied().all(scalar_is_finite)
                    && v.as_slice().iter().copied().all(scalar_is_finite)
            }
        }
    }

    fn from_dense_compressed(block: DenseMatrix<T>, epsilon: T::Real) -> Self {
        match maybe_compress(&block, epsilon) {
            Some((u, v)) => OffDiagBlock::LowRank { u, v },
            None => OffDiagBlock::Dense(block),
        }
    }

    #[allow(dead_code)]
    fn rows(&self) -> usize {
        match self {
            OffDiagBlock::Dense(m) => m.nrows(),
            OffDiagBlock::LowRank { u, .. } => u.nrows(),
        }
    }

    #[allow(dead_code)]
    fn cols(&self) -> usize {
        match self {
            OffDiagBlock::Dense(m) => m.ncols(),
            OffDiagBlock::LowRank { v, .. } => v.nrows(),
        }
    }

    fn stored_entries(&self) -> usize {
        match self {
            OffDiagBlock::Dense(m) => m.nrows() * m.ncols(),
            OffDiagBlock::LowRank { u, v } => (u.nrows() + v.nrows()) * u.ncols(),
        }
    }

    /// `y += alpha · block · x` where `x.len() = cols`, `y.len() = rows`.
    fn gemv_add(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            OffDiagBlock::Dense(m) => {
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
            OffDiagBlock::LowRank { u, v } => {
                if u.ncols() == 0 {
                    return;
                }
                let r = u.ncols();
                debug_assert_eq!(v.nrows(), x.len());
                debug_assert_eq!(u.nrows(), y.len());
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
        }
    }

    /// `y += alpha · block^T · x` where `x.len() = rows`, `y.len() = cols`.
    fn gemv_transpose_add(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            OffDiagBlock::Dense(m) => {
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
            OffDiagBlock::LowRank { u, v } => {
                if u.ncols() == 0 {
                    return;
                }
                let r = u.ncols();
                debug_assert_eq!(u.nrows(), x.len());
                debug_assert_eq!(v.nrows(), y.len());
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
        }
    }

    fn write_into(&self, out: &mut DenseMatrix<T>, row_off: usize, col_off: usize) {
        match self {
            OffDiagBlock::Dense(m) => {
                for i in 0..m.nrows() {
                    for j in 0..m.ncols() {
                        out[[row_off + i, col_off + j]] = m[[i, j]];
                    }
                }
            }
            OffDiagBlock::LowRank { u, v } => {
                let m = u.nrows();
                let n = v.nrows();
                let r = u.ncols();
                for i in 0..m {
                    for j in 0..n {
                        let mut s = T::zero();
                        for l in 0..r {
                            s += u[[i, l]] * v[[j, l]];
                        }
                        out[[row_off + i, col_off + j]] = s;
                    }
                }
            }
        }
    }
}

/// Recursive HODLR representation of an `n_rows × n_cols` panel.
pub enum HodlrPanel<T: ComplexField> {
    /// Leaf: store the panel densely.
    Leaf(DenseMatrix<T>),
    /// Hierarchical 2×2 split.
    ///
    /// `a11` covers rows `[0..m_split)` × cols `[0..n_split)`;
    /// `a22` covers `[m_split..n_rows)` × `[n_split..n_cols)`;
    /// `a12` and `a21` are off-diagonal blocks stored as either low-rank or
    /// dense via [`OffDiagBlock`].
    Node {
        n_rows: usize,
        n_cols: usize,
        m_split: usize,
        n_split: usize,
        a11: Box<HodlrPanel<T>>,
        a12: OffDiagBlock<T>,
        a21: OffDiagBlock<T>,
        a22: Box<HodlrPanel<T>>,
    },
}

impl<T: ComplexField> HodlrPanel<T> {
    pub(crate) fn is_finite(&self) -> bool {
        match self {
            HodlrPanel::Leaf(matrix) => matrix.as_slice().iter().copied().all(scalar_is_finite),
            HodlrPanel::Node {
                a11, a12, a21, a22, ..
            } => a11.is_finite() && a12.is_finite() && a21.is_finite() && a22.is_finite(),
        }
    }

    /// Build a HODLR panel from a dense source.
    ///
    /// - `leaf_size`: blocks with `max(rows, cols) ≤ leaf_size` are stored dense.
    /// - `epsilon`: relative tolerance for RRQR compression of each off-diagonal
    ///   sub-block. Each off-diagonal falls back to [`OffDiagBlock::Dense`]
    ///   when low-rank storage would not reduce entries.
    pub fn from_dense(source: &DenseMatrix<T>, leaf_size: usize, epsilon: T::Real) -> Self {
        Self::from_dense_maybe_parallel(source, leaf_size, epsilon, false)
    }

    pub(crate) fn from_dense_maybe_parallel(
        source: &DenseMatrix<T>,
        leaf_size: usize,
        epsilon: T::Real,
        use_parallel: bool,
    ) -> Self {
        let m = source.nrows();
        let n = source.ncols();

        if m == 0 || n == 0 {
            return HodlrPanel::Leaf(DenseMatrix::<T>::zeros(m, n));
        }
        if m <= leaf_size || n <= leaf_size {
            return HodlrPanel::Leaf(source.clone());
        }

        let m_split = m / 2;
        let n_split = n / 2;

        let build_a11 = || {
            let a11_dense = extract_block(source, 0, m_split, 0, n_split);
            Box::new(HodlrPanel::from_dense_maybe_parallel(
                &a11_dense,
                leaf_size,
                epsilon,
                use_parallel,
            ))
        };
        let build_a12 = || {
            let a12_dense = extract_block(source, 0, m_split, n_split, n);
            OffDiagBlock::from_dense_compressed(a12_dense, epsilon)
        };
        let build_a21 = || {
            let a21_dense = extract_block(source, m_split, m, 0, n_split);
            OffDiagBlock::from_dense_compressed(a21_dense, epsilon)
        };
        let build_a22 = || {
            let a22_dense = extract_block(source, m_split, m, n_split, n);
            Box::new(HodlrPanel::from_dense_maybe_parallel(
                &a22_dense,
                leaf_size,
                epsilon,
                use_parallel,
            ))
        };

        let split_parallel = use_parallel && m > leaf_size * 2 && n > leaf_size * 2;
        let ((a11, a12), (a21, a22)) = if split_parallel {
            rayon::join(
                || rayon::join(build_a11, build_a12),
                || rayon::join(build_a21, build_a22),
            )
        } else {
            ((build_a11(), build_a12()), (build_a21(), build_a22()))
        };

        HodlrPanel::Node {
            n_rows: m,
            n_cols: n,
            m_split,
            n_split,
            a11,
            a12,
            a21,
            a22,
        }
    }

    /// Number of rows in the panel.
    pub fn n_rows(&self) -> usize {
        match self {
            HodlrPanel::Leaf(m) => m.nrows(),
            HodlrPanel::Node { n_rows, .. } => *n_rows,
        }
    }

    /// Number of columns in the panel.
    pub fn n_cols(&self) -> usize {
        match self {
            HodlrPanel::Leaf(m) => m.ncols(),
            HodlrPanel::Node { n_cols, .. } => *n_cols,
        }
    }

    /// `y += alpha · panel · x`.
    pub fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_cols());
        debug_assert_eq!(y.len(), self.n_rows());
        match self {
            HodlrPanel::Leaf(m) => {
                for i in 0..m.nrows() {
                    let mut s = T::zero();
                    for j in 0..m.ncols() {
                        s += m[[i, j]] * x[j];
                    }
                    y[i] += alpha * s;
                }
            }
            HodlrPanel::Node {
                m_split,
                n_split,
                a11,
                a12,
                a21,
                a22,
                ..
            } => {
                let (x1, x2) = x.split_at(*n_split);
                let (y1, y2) = y.split_at_mut(*m_split);
                a11.gemv_into(alpha, x1, y1);
                a22.gemv_into(alpha, x2, y2);
                a12.gemv_add(alpha, x2, y1);
                a21.gemv_add(alpha, x1, y2);
            }
        }
    }

    /// `y += alpha · panel^T · x`.
    pub fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n_rows());
        debug_assert_eq!(y.len(), self.n_cols());
        match self {
            HodlrPanel::Leaf(m) => {
                for j in 0..m.ncols() {
                    let mut s = T::zero();
                    for i in 0..m.nrows() {
                        s += m[[i, j]] * x[i];
                    }
                    y[j] += alpha * s;
                }
            }
            HodlrPanel::Node {
                m_split,
                n_split,
                a11,
                a12,
                a21,
                a22,
                ..
            } => {
                let (x1, x2) = x.split_at(*m_split);
                let (y1, y2) = y.split_at_mut(*n_split);
                a11.gemv_transpose_into(alpha, x1, y1);
                a22.gemv_transpose_into(alpha, x2, y2);
                // A12 is at (top, right) → A12^T contributes from x1 (top
                // rows) into y2 (right cols).
                a12.gemv_transpose_add(alpha, x1, y2);
                // A21 is at (bottom, left) → A21^T contributes from x2 into y1.
                a21.gemv_transpose_add(alpha, x2, y1);
            }
        }
    }

    /// Stored entries across all levels.
    pub fn stored_entries(&self) -> usize {
        match self {
            HodlrPanel::Leaf(m) => m.nrows() * m.ncols(),
            HodlrPanel::Node {
                a11, a12, a21, a22, ..
            } => {
                a11.stored_entries()
                    + a22.stored_entries()
                    + a12.stored_entries()
                    + a21.stored_entries()
            }
        }
    }

    /// Materialise the panel as a dense matrix (debug / verification).
    pub fn to_dense(&self) -> DenseMatrix<T> {
        let m = self.n_rows();
        let n = self.n_cols();
        let mut out = DenseMatrix::<T>::zeros(m, n);
        self.write_dense_into(&mut out, 0, 0);
        out
    }

    /// HODLR-aware matmul: `out -= self · rhs`.
    ///
    /// Both panels are recursively decomposed; the four sub-block products
    /// of the 2×2 split fall out as:
    ///
    /// ```text
    /// [A11 A12]   [B11 B12]   [A11·B11 + A12·B21   A11·B12 + A12·B22]
    /// [A21 A22] · [B21 B22] = [A21·B11 + A22·B21   A21·B12 + A22·B22]
    /// ```
    ///
    /// When both operand sub-blocks are leaves, this becomes a dense GEMM.
    /// When one is `OffDiagBlock::LowRank`, the rank-r factor is exploited
    /// to avoid an `m·k·n` GEMM. Leaf×Leaf at the recursion boundary falls
    /// back to dense.
    pub fn matmul_sub_into_dense(&self, rhs: &Self, out: &mut DenseMatrix<T>) {
        assert_eq!(self.n_cols(), rhs.n_rows(), "HODLR matmul shape mismatch");
        assert_eq!(out.nrows(), self.n_rows());
        assert_eq!(out.ncols(), rhs.n_cols());
        hodlr_matmul_sub_recursive(self, rhs, out, 0, 0);
    }

    fn write_dense_into(&self, out: &mut DenseMatrix<T>, row_off: usize, col_off: usize) {
        match self {
            HodlrPanel::Leaf(d) => {
                for i in 0..d.nrows() {
                    for j in 0..d.ncols() {
                        out[[row_off + i, col_off + j]] = d[[i, j]];
                    }
                }
            }
            HodlrPanel::Node {
                m_split,
                n_split,
                a11,
                a12,
                a21,
                a22,
                ..
            } => {
                a11.write_dense_into(out, row_off, col_off);
                a22.write_dense_into(out, row_off + m_split, col_off + n_split);
                a12.write_into(out, row_off, col_off + n_split);
                a21.write_into(out, row_off + m_split, col_off);
            }
        }
    }
}

/// Recursive HODLR matmul: contribute `-self · rhs` into `out` at offset
/// `(row_off, col_off)`. Both operands must have matching inner dimensions.
fn hodlr_matmul_sub_recursive<T: ComplexField>(
    lhs: &HodlrPanel<T>,
    rhs: &HodlrPanel<T>,
    out: &mut DenseMatrix<T>,
    row_off: usize,
    col_off: usize,
) {
    let m = lhs.n_rows();
    let n = rhs.n_cols();
    let k = lhs.n_cols();
    debug_assert_eq!(k, rhs.n_rows());
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // If either operand is a Leaf, materialise both as dense and do a
    // straightforward GEMM. This is the recursion floor.
    let lhs_leaf = matches!(lhs, HodlrPanel::Leaf(_));
    let rhs_leaf = matches!(rhs, HodlrPanel::Leaf(_));
    if lhs_leaf || rhs_leaf {
        let lhs_dense = lhs.to_dense();
        let rhs_dense = rhs.to_dense();
        for i in 0..m {
            for j in 0..n {
                let mut s = T::zero();
                for p in 0..k {
                    s += lhs_dense[[i, p]] * rhs_dense[[p, j]];
                }
                out[[row_off + i, col_off + j]] -= s;
            }
        }
        return;
    }

    // Both lhs and rhs are Node: do the 2×2 block matmul on the children.
    let (l_m_split, l_n_split, l_a11, l_a12, l_a21, l_a22) = match lhs {
        HodlrPanel::Node {
            m_split,
            n_split,
            a11,
            a12,
            a21,
            a22,
            ..
        } => (*m_split, *n_split, a11.as_ref(), a12, a21, a22.as_ref()),
        HodlrPanel::Leaf(_) => unreachable!(),
    };
    let (r_m_split, r_n_split, r_a11, r_a12, r_a21, r_a22) = match rhs {
        HodlrPanel::Node {
            m_split,
            n_split,
            a11,
            a12,
            a21,
            a22,
            ..
        } => (*m_split, *n_split, a11.as_ref(), a12, a21, a22.as_ref()),
        HodlrPanel::Leaf(_) => unreachable!(),
    };

    // Inner split must match for the block products to align. If LHS's
    // n_split differs from RHS's m_split, fall back to dense — this happens
    // only for asymmetric panel shapes at recursion boundaries.
    if l_n_split != r_m_split {
        let lhs_dense = lhs.to_dense();
        let rhs_dense = rhs.to_dense();
        for i in 0..m {
            for j in 0..n {
                let mut s = T::zero();
                for p in 0..k {
                    s += lhs_dense[[i, p]] * rhs_dense[[p, j]];
                }
                out[[row_off + i, col_off + j]] -= s;
            }
        }
        return;
    }

    // Top-left of the result: A11·B11 + A12·B21.
    hodlr_matmul_sub_recursive(l_a11, r_a11, out, row_off, col_off);
    offdiag_offdiag_matmul_sub(l_a12, r_a21, out, row_off, col_off);

    // Top-right: A11·B12 + A12·B22.
    offdiag_dense_matmul_sub(l_a11, r_a12, out, row_off, col_off + r_n_split, false);
    offdiag_dense_matmul_sub(r_a22, l_a12, out, row_off, col_off + r_n_split, true);

    // Bottom-left: A21·B11 + A22·B21.
    offdiag_dense_matmul_sub(r_a11, l_a21, out, row_off + l_m_split, col_off, true);
    offdiag_dense_matmul_sub(l_a22, r_a21, out, row_off + l_m_split, col_off, false);

    // Bottom-right: A21·B12 + A22·B22.
    offdiag_offdiag_matmul_sub(l_a21, r_a12, out, row_off + l_m_split, col_off + r_n_split);
    hodlr_matmul_sub_recursive(l_a22, r_a22, out, row_off + l_m_split, col_off + r_n_split);
}

/// `out -= off_a · off_b` where both are off-diagonal blocks (Dense or
/// LowRank). Exploits rank structure on either side.
fn offdiag_offdiag_matmul_sub<T: ComplexField>(
    a: &OffDiagBlock<T>,
    b: &OffDiagBlock<T>,
    out: &mut DenseMatrix<T>,
    row_off: usize,
    col_off: usize,
) {
    use OffDiagBlock::*;
    match (a, b) {
        (LowRank { u: ua, v: va }, LowRank { u: ub, v: vb }) => {
            // (Ua·Va^T) · (Ub·Vb^T) = Ua · (Va^T·Ub) · Vb^T.
            let ra = ua.ncols();
            let rb = ub.ncols();
            if ra == 0 || rb == 0 {
                return;
            }
            let inner = va.nrows();
            let mut m_inner = DenseMatrix::<T>::zeros(ra, rb);
            for l in 0..ra {
                for r in 0..rb {
                    let mut s = T::zero();
                    for p in 0..inner {
                        s += va[[p, l]] * ub[[p, r]];
                    }
                    m_inner[[l, r]] = s;
                }
            }
            let mut tmp = DenseMatrix::<T>::zeros(ua.nrows(), rb);
            for i in 0..ua.nrows() {
                for r in 0..rb {
                    let mut s = T::zero();
                    for l in 0..ra {
                        s += ua[[i, l]] * m_inner[[l, r]];
                    }
                    tmp[[i, r]] = s;
                }
            }
            for i in 0..ua.nrows() {
                for j in 0..vb.nrows() {
                    let mut s = T::zero();
                    for r in 0..rb {
                        s += tmp[[i, r]] * vb[[j, r]];
                    }
                    out[[row_off + i, col_off + j]] -= s;
                }
            }
        }
        _ => {
            // Mixed dense / low-rank fall-through: materialise both.
            let ad = match a {
                Dense(m) => m.clone(),
                LowRank { u, v } => u.matmul(&v.t()),
            };
            let bd = match b {
                Dense(m) => m.clone(),
                LowRank { u, v } => u.matmul(&v.t()),
            };
            for i in 0..ad.nrows() {
                for j in 0..bd.ncols() {
                    let mut s = T::zero();
                    for p in 0..ad.ncols() {
                        s += ad[[i, p]] * bd[[p, j]];
                    }
                    out[[row_off + i, col_off + j]] -= s;
                }
            }
        }
    }
}

/// `out -= panel · off_b` (or `off_b · panel` if `swap`), with the panel
/// materialised dense at this contraction step. Used at the top-right /
/// bottom-left corners of the recursive HODLR×HODLR product where one
/// operand is a recursive HodlrPanel (diagonal block) and the other is an
/// off-diagonal block. We just densify the diagonal sub-panel — recursive
/// HODLR matmul against a thin (m × r) or (r × n) low-rank pair has
/// limited compression benefit and the recursion floor lands on dense
/// soon anyway.
fn offdiag_dense_matmul_sub<T: ComplexField>(
    diag_panel: &HodlrPanel<T>,
    off_block: &OffDiagBlock<T>,
    out: &mut DenseMatrix<T>,
    row_off: usize,
    col_off: usize,
    swap: bool,
) {
    let diag_dense = diag_panel.to_dense();
    let off_dense = match off_block {
        OffDiagBlock::Dense(m) => m.clone(),
        OffDiagBlock::LowRank { u, v } => u.matmul(&v.t()),
    };
    let (a, b) = if swap {
        (&off_dense, &diag_dense)
    } else {
        (&diag_dense, &off_dense)
    };
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();
    for i in 0..m {
        for j in 0..n {
            let mut s = T::zero();
            for p in 0..k {
                s += a[[i, p]] * b[[p, j]];
            }
            out[[row_off + i, col_off + j]] -= s;
        }
    }
}

fn extract_block<T: ComplexField>(
    source: &DenseMatrix<T>,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) -> DenseMatrix<T> {
    let mut out = DenseMatrix::<T>::zeros(r1 - r0, c1 - c0);
    for i in r0..r1 {
        for j in c0..c1 {
            out[[i - r0, j - c0]] = source[[i, j]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_dense_eps_zero() {
        let m = 16;
        let n = 24;
        let data: Vec<f64> = (0..(m * n))
            .map(|k| ((k * 13 + 5) % 19) as f64 - 9.0)
            .collect();
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let h = HodlrPanel::from_dense(&mat, 4, 0.0);
        let back = h.to_dense();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (mat[[i, j]] - back[[i, j]]).abs() < 1e-9,
                    "diff at ({},{}): {} vs {}",
                    i,
                    j,
                    mat[[i, j]],
                    back[[i, j]]
                );
            }
        }
    }

    #[test]
    fn gemv_matches_dense() {
        let m = 20;
        let n = 16;
        let data: Vec<f64> = (0..(m * n)).map(|k| (k as f64 * 0.7).cos()).collect();
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let h = HodlrPanel::from_dense(&mat, 4, 0.0);

        let x: Vec<f64> = (0..n).map(|j| (j as f64 + 0.1).sin()).collect();
        let mut y_h = vec![0.0; m];
        h.gemv_into(1.0, &x, &mut y_h);
        let mut y_ref = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                y_ref[i] += mat[[i, j]] * x[j];
            }
        }
        for i in 0..m {
            assert!((y_h[i] - y_ref[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn rank_one_panel_compresses_aggressively() {
        // Pure rank-1 panel: HODLR should drop to (m+n)·1 per level off-diag.
        let m = 32;
        let n = 32;
        let u: Vec<f64> = (0..m).map(|i| ((i + 1) as f64).sqrt()).collect();
        let v: Vec<f64> = (0..n).map(|j| (j as f64).cos() + 1.5).collect();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = u[i] * v[j];
            }
        }
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let h = HodlrPanel::from_dense(&mat, 4, 1e-10);
        let stored = h.stored_entries();
        assert!(stored < 400, "stored entries {} not compressing", stored);

        let x: Vec<f64> = (0..n).map(|j| (j as f64).sin()).collect();
        let mut y_h = vec![0.0; m];
        let mut y_ref = vec![0.0; m];
        h.gemv_into(1.0, &x, &mut y_h);
        for i in 0..m {
            for j in 0..n {
                y_ref[i] += mat[[i, j]] * x[j];
            }
        }
        for i in 0..m {
            assert!((y_h[i] - y_ref[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn parallel_build_matches_serial() {
        let m = 48;
        let n = 40;
        let data: Vec<f64> = (0..(m * n))
            .map(|k| ((k as f64 * 0.37).sin()) + 0.25 * ((k as f64 * 0.91).cos()))
            .collect();
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let serial = HodlrPanel::from_dense_maybe_parallel(&mat, 4, 0.0, false);
        let parallel = HodlrPanel::from_dense_maybe_parallel(&mat, 4, 0.0, true);

        assert_eq!(serial.stored_entries(), parallel.stored_entries());
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
    fn full_rank_panel_never_grows_storage() {
        // A 32×32 random panel is essentially full rank. At ε = 0 every
        // off-diagonal block should fall back to Dense via maybe_compress
        // so the total storage equals dense (m × n). This is the guard
        // against the bug the reviewer flagged: HODLR must not balloon
        // memory for non-compressible blocks.
        let m = 32;
        let n = 32;
        let data: Vec<f64> = (0..(m * n))
            .map(|k| ((k as f64 * 0.913).sin()) + 0.5 * ((k as f64 * 1.731).cos()))
            .collect();
        let mat = DenseMatrix::<f64>::from_row_major_vec(m, n, data);
        let h = HodlrPanel::from_dense(&mat, 4, 0.0);
        let stored = h.stored_entries();
        assert!(
            stored <= m * n,
            "HODLR grew storage from {} to {}",
            m * n,
            stored
        );
    }
}
