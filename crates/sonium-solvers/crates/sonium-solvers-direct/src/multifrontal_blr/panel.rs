//! Polymorphic compressed-panel wrapper.
//!
//! A frontal-matrix off-diagonal panel can be stored in one of three
//! formats: flat block-low-rank (BLR), hierarchical off-diagonal low-rank
//! (HODLR), or nested-basis global low-rank (`NestedLr`). [`Panel`] is a
//! small enum that unifies the operations the multifrontal walk needs
//! (build, GEMV, transpose-GEMV, dense materialisation, entry counts) so
//! [`crate::multifrontal_blr::front::FactoredFront`] doesn't have
//! to be generic over the panel kind.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;

use super::blr::{BlrPanel, LowRankPrecision};
use super::hodlr::HodlrPanel;
use super::nested_lr::NestedLrPanel;

/// Compression scheme to use for the L21 / U12 / CB panels of every front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanelKind {
    /// Flat block-low-rank with a uniform block grid. Default.
    #[default]
    Blr,
    /// Hierarchical off-diagonal low-rank with recursive 2×2 splits and
    /// *independent* `(U, V)` factors at every level.
    Hodlr,
    /// Nested-basis global low-rank: leaf row / column bases shared across
    /// all levels via per-node translation operators, with a single global
    /// core — the whole panel is approximated as one product
    /// `U_root · S · V_rootᵀ`. Asymptotically smaller than
    /// [`PanelKind::Hodlr`] for smooth kernels at the cost of higher build
    /// cost and more bookkeeping.
    ///
    /// NOTE: despite the historical name this is **not** true HSS — there
    /// are no per-sibling coupling blocks and no ULV factorisation. When
    /// the global nested rank grows to the point where the representation
    /// costs as much as the dense panel, construction falls back to dense
    /// storage (same storage-gain rule as the BLR `maybe_compress` guard).
    NestedLr,
}

impl PanelKind {
    /// Backward-compatible alias for [`PanelKind::NestedLr`]. Kept so
    /// existing callers (and the `SparseDirectMethod::MultifrontalHss`
    /// mapping) keep compiling; the panel was never true HSS — see the
    /// [`PanelKind::NestedLr`] docs.
    #[allow(non_upper_case_globals)]
    pub const Hss: PanelKind = PanelKind::NestedLr;
}

/// A compressed panel in one of the supported formats.
pub enum Panel<T: ComplexField> {
    Blr(BlrPanel<T>),
    Hodlr(HodlrPanel<T>),
    NestedLr(NestedLrPanel<T>),
}

impl<T: ComplexField> Panel<T> {
    pub(crate) fn is_finite(&self) -> bool {
        match self {
            Panel::Blr(panel) => panel.is_finite(),
            Panel::Hodlr(panel) => panel.is_finite(),
            Panel::NestedLr(panel) => panel.is_finite(),
        }
    }

    /// Build a panel of the chosen kind from a dense source. Uses
    /// `LowRankPrecision::Full` for BLR; HODLR ignores the precision.
    pub fn from_dense(
        source: &DenseMatrix<T>,
        kind: PanelKind,
        block_size: usize,
        epsilon: T::Real,
    ) -> Self {
        Self::from_dense_with_precision(source, kind, block_size, epsilon, LowRankPrecision::Full)
    }

    /// Like `from_dense` but with explicit low-rank storage precision.
    /// Half-precision only applies to BLR for real-valued `T`; HODLR and
    /// complex `T` silently use full precision.
    pub fn from_dense_with_precision(
        source: &DenseMatrix<T>,
        kind: PanelKind,
        block_size: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
    ) -> Self {
        Self::from_dense_with_precision_maybe_parallel(
            source, kind, block_size, epsilon, precision, false,
        )
    }

    pub(crate) fn from_dense_with_precision_maybe_parallel(
        source: &DenseMatrix<T>,
        kind: PanelKind,
        block_size: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Self {
        match kind {
            PanelKind::Blr => Panel::Blr(BlrPanel::from_dense_with_precision_maybe_parallel(
                source,
                block_size,
                epsilon,
                precision,
                use_parallel,
            )),
            PanelKind::Hodlr => Panel::Hodlr(HodlrPanel::from_dense_maybe_parallel(
                source,
                block_size,
                epsilon,
                use_parallel,
            )),
            PanelKind::NestedLr => {
                Panel::NestedLr(NestedLrPanel::from_dense_with_precision_maybe_parallel(
                    source,
                    block_size,
                    epsilon,
                    precision,
                    use_parallel,
                ))
            }
        }
    }

    /// Owned-source variant of
    /// [`from_dense_with_precision_maybe_parallel`](Self::from_dense_with_precision_maybe_parallel).
    /// The BLR path moves single-block panel storage instead of copying it;
    /// the other kinds borrow the source and drop it afterwards (same work
    /// as the borrowed entry).
    pub(crate) fn from_dense_owned_with_precision_maybe_parallel(
        source: DenseMatrix<T>,
        kind: PanelKind,
        block_size: usize,
        epsilon: T::Real,
        precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Self {
        match kind {
            PanelKind::Blr => Panel::Blr(BlrPanel::from_dense_owned_with_precision_maybe_parallel(
                source,
                block_size,
                epsilon,
                precision,
                use_parallel,
            )),
            _ => Self::from_dense_with_precision_maybe_parallel(
                &source,
                kind,
                block_size,
                epsilon,
                precision,
                use_parallel,
            ),
        }
    }

    pub fn n_rows(&self) -> usize {
        match self {
            Panel::Blr(p) => p.n_rows,
            Panel::Hodlr(p) => p.n_rows(),
            Panel::NestedLr(p) => p.n_rows(),
        }
    }

    pub fn n_cols(&self) -> usize {
        match self {
            Panel::Blr(p) => p.n_cols,
            Panel::Hodlr(p) => p.n_cols(),
            Panel::NestedLr(p) => p.n_cols(),
        }
    }

    pub fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            Panel::Blr(p) => p.gemv_into(alpha, x, y),
            Panel::Hodlr(p) => p.gemv_into(alpha, x, y),
            Panel::NestedLr(p) => p.gemv_into(alpha, x, y),
        }
    }

    pub fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            Panel::Blr(p) => p.gemv_transpose_into(alpha, x, y),
            Panel::Hodlr(p) => p.gemv_transpose_into(alpha, x, y),
            Panel::NestedLr(p) => p.gemv_transpose_into(alpha, x, y),
        }
    }

    pub fn stored_entries(&self) -> usize {
        match self {
            Panel::Blr(p) => p.stored_entries(),
            Panel::Hodlr(p) => p.stored_entries(),
            Panel::NestedLr(p) => p.stored_entries(),
        }
    }

    pub fn to_dense(&self) -> DenseMatrix<T> {
        match self {
            Panel::Blr(p) => p.to_dense(),
            Panel::Hodlr(p) => p.to_dense(),
            Panel::NestedLr(p) => p.to_dense(),
        }
    }

    /// `out -= self · rhs`. Same-kind operands stay in compressed form:
    /// BLR×BLR uses block-wise low-rank-aware products; HODLR×HODLR
    /// exploits rank at each level; NestedLr×NestedLr uses the
    /// nested-basis `U_A · S_A · M · S_B · V_B^T` identity to avoid every
    /// `m × p × n` GEMM. Mixed kinds fall back to a dense intermediate.
    pub fn matmul_sub_into_dense(&self, rhs: &Self, out: &mut DenseMatrix<T>) {
        match (self, rhs) {
            (Panel::Blr(a), Panel::Blr(b)) => a.matmul_sub_into_dense(b, out),
            (Panel::Hodlr(a), Panel::Hodlr(b)) => a.matmul_sub_into_dense(b, out),
            (Panel::NestedLr(a), Panel::NestedLr(b)) => a.matmul_sub_into_dense(b, out),
            _ => {
                let a = self.to_dense();
                let b = rhs.to_dense();
                let product = a.matmul(&b);
                for (value, update) in out.as_mut_slice().iter_mut().zip(product.as_slice()) {
                    *value -= *update;
                }
            }
        }
    }

    /// Reports `Some(avg_rank)` summarising compression of the panel.
    ///
    /// * BLR: mean rank of low-rank sub-blocks across the uniform grid.
    /// * NestedLr: mean cluster rank across both the row and column
    ///   trees — the most natural single-number summary of nested-basis
    ///   sharing.
    /// * HODLR: returns `None` since per-level off-diagonal ranks vary
    ///   independently and a single average is not informative.
    /// * Dense fallback variants of NestedLr return `None`.
    pub fn average_low_rank(&self) -> Option<f64> {
        match self {
            Panel::Blr(p) => Some(p.average_low_rank()),
            Panel::Hodlr(_) => None,
            Panel::NestedLr(p) => p.average_rank(),
        }
    }
}
