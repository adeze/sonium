//! Multifrontal sparse LDLᵀ / LDLᴴ for symmetric / Hermitian indefinite matrices.
//!
//! This is the symmetric counterpart of [`super::multifrontal_blr`]: same
//! nested-dissection assembly tree, same supernode amalgamation, same
//! MUMPS-style delayed pivoting, same level-parallel tree traversal — but the
//! fronts are *symmetric*. Concretely:
//!
//! 1. **One-triangle storage.** Only the lower triangle of every symmetric
//!    block is assembled, factorised and stored: the FS factor `F11` is a
//!    packed lower triangle (unit `L` below the diagonal, 1×1 / 2×2 `D`
//!    blocks on it), the CB panel is a packed lower triangle, and the
//!    off-diagonal panel is a single `L21` (vs `L21` + `U12` for LU). That
//!    halves the factor memory compared to routing the same matrix through
//!    the multifrontal LU path.
//! 2. **Bunch–Kaufman pivoting.** Symmetric 1×1 / 2×2 pivot selection with
//!    the classical `α = (1 + √17) / 8` criterion (the same criterion
//!    [`super::multifrontal_blr::front`] uses for its LU fronts). Pivot
//!    swaps are symmetric row+column interchanges confined to the FS block.
//!    Rejected pivots are postponed to the parent front via the same
//!    scale-aware tolerance and dense delayed-pivot payload as the LU path;
//!    the delayed payload is symmetric too (`dd` packed lower + `dc`, with
//!    `cd = (dc)ᴴ` implied).
//! 3. **Hermitian vs complex-symmetric.** With `hermitian = true` the
//!    transposes in `L·D·Lᴴ` conjugate (A = Aᴴ). With `hermitian = false`
//!    the factorisation is a true LDLᵀ for complex-symmetric (A = Aᵀ)
//!    matrices — no conjugation anywhere. For real `T` the two coincide.
//!
//! Fronts are dense by default. Setting [`LdltConfig::blr`] enables BLR
//! compression of the off-diagonal panels of every front, reusing the panel
//! machinery of [`super::multifrontal_blr`]: the below-diagonal `L21` panel
//! and the symmetric CB contribution block are compressed after each front's
//! factorisation. Only the lower triangle is ever assembled — the compressed
//! `L21` is the single below-diagonal panel (vs `L21` + `U12` for LU), and
//! the CB panel is compressed from its materialised dense symmetric form and
//! consumed through decompression at extend-add, exactly like the LU path's
//! `cb_panel`. Solve passes apply the compressed panels through their
//! BLR-aware GEMV / transpose-GEMV. Tree-level rayon parallelism is
//! inherited from the assembly-tree level structure, and the within-front
//! symmetric rank-k update parallelises over rows for large fronts
//! ([`LdltConfig::parallel_front_threshold`]).

use std::collections::BTreeSet;

use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::prelude::*;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

use super::multifrontal_blr::factorize::{build_column_incoming, permute_matrix};
use super::multifrontal_blr::front::{BlockInverseError, PivotKind, invert_2x2};
use super::multifrontal_blr::ordering::{AssemblyTree, amalgamate_tree, build_assembly_tree};
use super::multifrontal_blr::panel::{Panel, PanelKind};
use super::multifrontal_blr::{
    BlrError, FrontError, LowRankPrecision, first_non_finite, scalar_is_finite,
};

fn map_front_error(front: usize, error: FrontError) -> BlrError {
    match error {
        FrontError::Invalid(reason) => BlrError::InvalidFactorization { front, reason },
        FrontError::Numerical { stage, index } => BlrError::FrontNumericalBreakdown {
            front,
            stage,
            index,
        },
    }
}

/// Configuration knobs for [`MultifrontalLdltFactorization`].
#[derive(Debug, Clone)]
pub struct LdltConfig {
    /// Minimum dissection-block size: subdomains smaller than this are
    /// treated as leaves. Default `128`.
    pub min_block_size: usize,
    /// Maximum dissection depth (auto if `None`).
    pub max_levels: Option<usize>,
    /// Supernode amalgamation threshold, same semantics as
    /// [`super::multifrontal_blr::BlrConfig::amalgamation_threshold`].
    /// Default `256`; `0` disables amalgamation.
    pub amalgamation_threshold: usize,
    /// Front-internal parallelism threshold for the symmetric rank-k Schur
    /// update. Fronts whose total size exceeds this value distribute the
    /// trailing-row updates across threads. Default `1024`; `0` disables.
    pub parallel_front_threshold: usize,
    /// Multiplier for the scale-aware pivot rejection tolerance
    /// `pivot_tol_multiplier · ε_machine · nfs · max|F11|`, the same
    /// convention as the LU fronts. Default `1.0`.
    pub pivot_tol_multiplier: f64,
    /// `true` for Hermitian matrices (`A = Aᴴ`, factorisation `L·D·Lᴴ`);
    /// `false` for complex-symmetric matrices (`A = Aᵀ`, factorisation
    /// `L·D·Lᵀ` without conjugation). Ignored (identical) for real `T`.
    /// Default `true`.
    pub hermitian: bool,
    /// BLR compression of the off-diagonal panels of every front. `None`
    /// (default) keeps the exact dense `L21` and packed-triangle CB panels;
    /// `Some` compresses both after each front's factorisation with the
    /// [`super::multifrontal_blr`] panel machinery. See [`LdltBlrConfig`].
    pub blr: Option<LdltBlrConfig>,
}

impl Default for LdltConfig {
    fn default() -> Self {
        Self {
            min_block_size: 128,
            max_levels: None,
            amalgamation_threshold: 256,
            parallel_front_threshold: 1024,
            pivot_tol_multiplier: 1.0,
            hermitian: true,
            blr: None,
        }
    }
}

/// BLR compression knobs for the off-diagonal panels (`L21` and the CB
/// contribution block) of the LDLᵀ fronts. Field names mirror
/// [`super::multifrontal_blr::BlrConfig`]; the symmetric path has no
/// `U12` panel and no pivot-strategy / update-order knobs, so only the
/// compression settings are carried over.
#[derive(Debug, Clone)]
pub struct LdltBlrConfig {
    /// Relative ε-rank tolerance for BLR compression of off-diagonal
    /// panels. `0.0` keeps the full numerical rank (exact round-trip; the
    /// panels are still block-partitioned). Default `1e-8`.
    pub epsilon: f64,
    /// Block size of the BLR grid on each panel. Default `128`.
    pub block_size: usize,
    /// Compressed-panel format for the off-diagonal panels of every front.
    /// Default [`PanelKind::Blr`].
    pub panel_kind: PanelKind,
}

impl Default for LdltBlrConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            block_size: 128,
            panel_kind: PanelKind::Blr,
        }
    }
}

/// A symmetric front, fully assembled but not yet factorised. Only the lower
/// triangle of `matrix` is populated and read.
struct AssembledSymFront<T: ComplexField> {
    /// Size of the fully-summed block (own FS plus absorbed delayed pivots).
    nfs: usize,
    /// Size of the contribution block.
    ncb: usize,
    /// Dense `(nfs + ncb)²` buffer; only the lower triangle is meaningful.
    matrix: DenseMatrix<T>,
    /// Elimination-order index of every assembled FS position (length `nfs`).
    fs_elim_indices: Vec<usize>,
    /// Elimination-order indices of the CB rows/cols (length `ncb`).
    cb_indices: Vec<usize>,
}

/// Dense symmetric delayed-pivot payload handed from a front to its parent.
///
/// `dd` is the packed lower triangle of the updated `F[D, D]` block; `dc` is
/// the updated `F[D, C]` block. The `F[C, D]` block is `(dc)ᴴ` (or `(dc)ᵀ`
/// for complex-symmetric) and is not stored.
struct DelayedSymBlocks<T: ComplexField> {
    /// Packed lower triangle of the trailing `F[D, D]` block, length
    /// `ndelayed · (ndelayed + 1) / 2`, rows/cols in `delayed_row_indices`
    /// order.
    dd_lower: Vec<T>,
    /// Updated `F[D, C]` block, shape `(ndelayed × ncb)`; columns in
    /// `cb_indices` order.
    dc: DenseMatrix<T>,
}

/// Storage for the below-diagonal off-diagonal panel (`L21`) of a factored
/// symmetric front: dense by default, BLR-compressed when
/// [`LdltConfig::blr`] is enabled.
pub enum L21Panel<T: ComplexField> {
    /// Dense `(ncb × nfs)` row-major panel.
    Dense(DenseMatrix<T>),
    /// Compressed panel (flat BLR, HODLR, or nested-basis, per
    /// [`LdltBlrConfig::panel_kind`]).
    Blr(Panel<T>),
}

impl<T: ComplexField> L21Panel<T> {
    fn dimensions(&self) -> (usize, usize) {
        match self {
            L21Panel::Dense(panel) => (panel.nrows(), panel.ncols()),
            L21Panel::Blr(panel) => (panel.n_rows(), panel.n_cols()),
        }
    }

    fn is_finite(&self) -> bool {
        match self {
            L21Panel::Dense(panel) => panel.as_slice().iter().copied().all(scalar_is_finite),
            L21Panel::Blr(panel) => panel.is_finite(),
        }
    }

    /// `y += alpha · L21 · x` (`x` length `nfs`, `y` length `ncb`).
    fn gemv_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            L21Panel::Dense(l21) => {
                debug_assert_eq!(x.len(), l21.ncols());
                debug_assert_eq!(y.len(), l21.nrows());
                for (c, yc) in y.iter_mut().enumerate() {
                    let mut s = T::zero();
                    for p in 0..l21.ncols() {
                        s += l21[[c, p]] * x[p];
                    }
                    *yc += alpha * s;
                }
            }
            L21Panel::Blr(panel) => panel.gemv_into(alpha, x, y),
        }
    }

    /// `y += alpha · L21ᵀ · x` — the plain transpose, *no* conjugation
    /// (`x` length `ncb`, `y` length `nfs`). Hermitian callers get `L21ᴴ`
    /// via the conjugate sandwich `L21ᴴ·x = conj(L21ᵀ·conj(x))`.
    fn gemv_transpose_into(&self, alpha: T, x: &[T], y: &mut [T]) {
        match self {
            L21Panel::Dense(l21) => {
                debug_assert_eq!(x.len(), l21.nrows());
                debug_assert_eq!(y.len(), l21.ncols());
                for (p, yp) in y.iter_mut().enumerate() {
                    let mut s = T::zero();
                    for c in 0..l21.nrows() {
                        s += l21[[c, p]] * x[c];
                    }
                    *yp += alpha * s;
                }
            }
            L21Panel::Blr(panel) => panel.gemv_transpose_into(alpha, x, y),
        }
    }

    /// Total stored entries.
    fn stored_entries(&self) -> usize {
        match self {
            L21Panel::Dense(l21) => l21.nrows() * l21.ncols(),
            L21Panel::Blr(panel) => panel.stored_entries(),
        }
    }
}

/// Storage for the symmetric CB contribution panel of a factored front:
/// packed lower triangle by default, BLR-compressed (from the materialised
/// dense symmetric panel) when [`LdltConfig::blr`] is enabled.
pub enum CbPanel<T: ComplexField> {
    /// Packed lower triangle, length `ncb · (ncb + 1) / 2`.
    PackedLower(Vec<T>),
    /// Compressed full symmetric panel. Only its lower triangle is read
    /// back (extend-add stores one triangle).
    Blr(Panel<T>),
}

impl<T: ComplexField> CbPanel<T> {
    fn is_finite(&self) -> bool {
        match self {
            CbPanel::PackedLower(values) => values.iter().copied().all(scalar_is_finite),
            CbPanel::Blr(panel) => panel.is_finite(),
        }
    }

    /// Total stored entries.
    fn stored_entries(&self) -> usize {
        match self {
            CbPanel::PackedLower(cb_lower) => cb_lower.len(),
            CbPanel::Blr(panel) => panel.stored_entries(),
        }
    }
}

/// Factorised symmetric front: packed-triangle `L·D·Lᴴ` factors plus the
/// data the parent needs for extend-add.
pub struct FactoredLdltFront<T: ComplexField> {
    /// Number of FS pivots actually eliminated in this front.
    pub nfs: usize,
    pub ncb: usize,
    /// Elimination-order index of every *assembled* FS position, indexed by
    /// front-local FS position (length = assembled FS size ≥ `nfs`).
    pub fs_elim_indices: Vec<usize>,
    /// Packed lower triangle of the eliminated FS block, length
    /// `nfs · (nfs + 1) / 2`. Below-diagonal entries are the unit-`L`
    /// multipliers; the diagonal holds the 1×1 `D` entries and the 2×2 `D`
    /// blocks (`[k, k]`, `[k+1, k]`, `[k+1, k+1]`).
    pub f11_ldl: Vec<T>,
    /// Per-step pivot descriptor (length `nfs`).
    pub pivot_kinds: Vec<PivotKind>,
    /// Symmetric pivot map (length `nfs`): assembled-FS position eliminated
    /// at each step, in factored order. Row and column pivots coincide.
    pub pivots: Vec<usize>,
    /// `L21` panel, dense `(ncb × nfs)` or BLR-compressed.
    pub l21: L21Panel<T>,
    /// Elimination-order indices of the CB rows/cols (delayed indices stay
    /// in this set; the parent absorbs them into its enlarged FS set).
    pub cb_indices: Vec<usize>,
    /// Schur complement of the front: packed lower triangle of length
    /// `ncb · (ncb + 1) / 2`, or BLR-compressed. The parent extend-adds it
    /// into its own dense storage.
    pub cb_panel: CbPanel<T>,
    /// Number of pivots postponed to the parent front (0 in the common case).
    pub ndelayed: usize,
    /// Elimination-order indices of the delayed rows/columns (length
    /// `ndelayed`, in factored order). Rows and columns coincide under
    /// symmetric pivoting.
    pub delayed_row_indices: Vec<usize>,
    /// Dense updated trailing blocks involving the delayed indices; `None`
    /// iff `ndelayed == 0`.
    delayed: Option<DelayedSymBlocks<T>>,
}

/// Packed lower-triangle index: element `(i, j)` with `i ≥ j`.
#[inline(always)]
fn pl(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

/// Conjugate when Hermitian, identity for complex-symmetric (LDLᵀ).
#[inline(always)]
fn cj<T: ComplexField>(v: T, hermitian: bool) -> T {
    if hermitian { v.conj() } else { v }
}

/// Outcome of the symmetric Bunch–Kaufman pivot selection at step `k`.
enum SymBkPivot {
    /// 1×1 pivot on the diagonal position `row` (swap `k ↔ row` first).
    One { row: usize },
    /// 2×2 pivot on `{k, row}` (swap `k + 1 ↔ row` first).
    Two { row: usize },
}

/// Symmetric Bunch–Kaufman 1×1 / 2×2 pivot selection on the active FS
/// submatrix `[k..nfs, k..nfs]` of a lower-triangle-stored front.
///
/// This is the same criterion as
/// [`super::multifrontal_blr::front`]'s BK selection (α = (1+√17)/8, λ/σ
/// tests), specialised to symmetric pivoting: 1×1 pivots are always diagonal,
/// 2×2 pivots are symmetric index pairs, and all reads go through the
/// Hermitian-aware lower-triangle accessor.
fn find_sym_bk_pivot<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    k: usize,
    nfs: usize,
    pivot_tol: T::Real,
    hermitian: bool,
) -> Option<SymBkPivot> {
    let get = |i: usize, j: usize| -> T {
        if i >= j {
            matrix[[i, j]]
        } else {
            cj(matrix[[j, i]], hermitian)
        }
    };
    // Single remaining position: only the diagonal is a candidate.
    if k + 1 >= nfs {
        return if get(k, k).norm() >= pivot_tol {
            Some(SymBkPivot::One { row: k })
        } else {
            None
        };
    }

    let alpha = T::Real::from_f64((1.0 + 17.0_f64.sqrt()) / 8.0).unwrap();

    // λ = largest magnitude in column k strictly below the diagonal.
    let mut lambda = T::Real::zero();
    let mut r = k + 1;
    for i in (k + 1)..nfs {
        let v = get(i, k).norm();
        if v > lambda {
            lambda = v;
            r = i;
        }
    }
    let diag_k = get(k, k).norm();

    // Whole column k vanishes ⇒ no pivot this step (delayed pivoting).
    if diag_k < pivot_tol && lambda < pivot_tol {
        return None;
    }

    // Test 1: diagonal already dominates column ⇒ 1×1 at k.
    if diag_k >= pivot_tol && diag_k >= alpha * lambda {
        return Some(SymBkPivot::One { row: k });
    }

    // σ = strongest entry in column/row r outside the diagonal.
    let mut sigma = T::Real::zero();
    for i in k..nfs {
        if i == r {
            continue;
        }
        let v = get(i, r).norm();
        if v > sigma {
            sigma = v;
        }
    }

    // Test 2: |A[k,k]|·σ ≥ α·λ² ⇒ 1×1 at k is still safe.
    if diag_k >= pivot_tol && diag_k * sigma >= alpha * lambda * lambda {
        return Some(SymBkPivot::One { row: k });
    }

    // Test 3: row r has a strong diagonal ⇒ swap k ↔ r, take 1×1.
    let diag_r = get(r, r).norm();
    if diag_r >= pivot_tol && diag_r >= alpha * sigma {
        return Some(SymBkPivot::One { row: r });
    }

    // Otherwise: 2×2 block on rows/columns {k, r}.
    Some(SymBkPivot::Two { row: r })
}

/// Symmetric interchange of indices `k` and `r` (`k < r`) on the
/// lower-triangle-stored Hermitian matrix: `M ← P·M·Pᵀ` with `P` swapping
/// `k ↔ r` across the full front (FS and CB parts alike).
fn sym_swap<T: ComplexField>(
    matrix: &mut DenseMatrix<T>,
    size: usize,
    k: usize,
    r: usize,
    hermitian: bool,
) {
    debug_assert!(k < r && r < size);
    let get = |m: &DenseMatrix<T>, i: usize, j: usize| -> T {
        if i >= j {
            m[[i, j]]
        } else {
            cj(m[[j, i]], hermitian)
        }
    };
    // Off-diagonal pairs (i, k) ↔ (i, r) for i ∉ {k, r}: reads happen before
    // the writes of the same pair and pairs are disjoint, so every read sees
    // the original Hermitian matrix.
    for i in 0..size {
        if i == k || i == r {
            continue;
        }
        let a = get(matrix, i, k);
        let b = get(matrix, i, r);
        if i >= k {
            matrix[[i, k]] = b;
        } else {
            matrix[[k, i]] = cj(b, hermitian);
        }
        if i >= r {
            matrix[[i, r]] = a;
        } else {
            matrix[[r, i]] = cj(a, hermitian);
        }
    }
    // Diagonal swap and the (k, r) cross term: M''[r, k] = M[k, r] = cj(M[r, k]).
    let kk = matrix[[k, k]];
    let rr = matrix[[r, r]];
    matrix[[k, k]] = rr;
    matrix[[r, r]] = kk;
    let kr = matrix[[r, k]];
    matrix[[r, k]] = cj(kr, hermitian);
}

/// Extract the lower triangle of `matrix` (one entry per unordered pair,
/// row ≥ col) into a fresh CSR matrix.
///
/// The input may store the full symmetric matrix, only the lower triangle,
/// or only the upper triangle — an upper entry `(i, j)` whose transpose
/// `(j, i)` is absent is reflected (with conjugation when Hermitian) so all
/// three storage conventions are accepted without double counting.
fn lower_triangle_csr<T: ComplexField>(matrix: &CsrMatrix<T>, hermitian: bool) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(matrix.values.len() / 2 + n);
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            let v = matrix.values[idx];
            if j <= i {
                triplets.push((i, j, v));
            } else {
                // Upper entry: keep only if the lower counterpart is missing.
                let rs = matrix.row_ptrs[j];
                let re = matrix.row_ptrs[j + 1];
                if matrix.col_indices[rs..re].binary_search(&i).is_err() {
                    triplets.push((j, i, cj(v, hermitian)));
                }
            }
        }
    }
    triplets.sort_by_key(|&(r, c, _)| (r, c));

    let mut row_ptrs = vec![0usize; n + 1];
    let mut col_indices: Vec<usize> = Vec::with_capacity(triplets.len());
    let mut values: Vec<T> = Vec::with_capacity(triplets.len());
    let mut cursor = 0usize;
    for row in 0..n {
        while cursor < triplets.len() && triplets[cursor].0 == row {
            let (_, col, v) = triplets[cursor];
            // Merge duplicate (row, col) entries defensively (the duplicate
            // must belong to the current row).
            if col_indices.len() > row_ptrs[row]
                && col_indices.last() == Some(&col)
                && let Some(last) = values.last_mut()
            {
                *last += v;
            } else {
                col_indices.push(col);
                values.push(v);
            }
            cursor += 1;
        }
        row_ptrs[row + 1] = col_indices.len();
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
}

/// Assemble one symmetric front: fill the lower triangle of its dense
/// `(nfs + ncb)²` matrix with the permuted-A entries plus extend-add of each
/// child's packed CB triangle and each child's delayed-pivot blocks.
fn assemble_front<T: ComplexField>(
    a_lower: &CsrMatrix<T>,
    fs_start: usize,
    fs_end: usize,
    cb_indices: Vec<usize>,
    children_factored: &[&FactoredLdltFront<T>],
    hermitian: bool,
    front: usize,
) -> Result<AssembledSymFront<T>, BlrError> {
    if a_lower.num_rows != a_lower.num_cols {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front matrix is not square",
        });
    }
    if fs_start > fs_end || fs_end > a_lower.num_rows {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front fully-summed range is outside the matrix",
        });
    }
    if cb_indices
        .iter()
        .any(|&g| g >= a_lower.num_rows || g < fs_end)
        || cb_indices.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front contribution indices are invalid",
        });
    }

    let packed_len = |n: usize| {
        n.checked_add(1)
            .and_then(|next| n.checked_mul(next))
            .and_then(|value| value.checked_div(2))
    };
    let mut delayed_indices = BTreeSet::new();
    for child in children_factored {
        let child_metadata_valid = child.ncb == child.cb_indices.len()
            && child.cb_indices.iter().all(|&g| g < a_lower.num_rows)
            && child.cb_indices.windows(2).all(|pair| pair[0] < pair[1])
            && child.ndelayed == child.delayed_row_indices.len()
            && child
                .delayed_row_indices
                .iter()
                .all(|&g| g < a_lower.num_rows && g < fs_start)
            && child
                .delayed_row_indices
                .iter()
                .all(|&g| delayed_indices.insert(g));
        if !child_metadata_valid {
            return Err(BlrError::InvalidFactorization {
                front,
                reason: "child front index metadata is inconsistent",
            });
        }

        let Some(expected_cb_len) = packed_len(child.ncb) else {
            return Err(BlrError::InvalidFactorization {
                front,
                reason: "child contribution size overflows usize",
            });
        };
        let cb_panel_valid = match &child.cb_panel {
            CbPanel::PackedLower(values) => values.len() == expected_cb_len,
            CbPanel::Blr(panel) => panel.n_rows() == child.ncb && panel.n_cols() == child.ncb,
        };
        if !cb_panel_valid {
            return Err(BlrError::InvalidFactorization {
                front,
                reason: "child contribution panel dimensions are inconsistent",
            });
        }

        if child.ndelayed == 0 {
            if child.delayed.is_some() {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "child delayed-pivot metadata is inconsistent",
                });
            }
        } else {
            let Some(blocks) = &child.delayed else {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "child delayed-pivot blocks are missing",
                });
            };
            let Some(expected_dd_len) = packed_len(child.ndelayed) else {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "child delayed-pivot size overflows usize",
                });
            };
            if blocks.dd_lower.len() != expected_dd_len
                || blocks.dc.nrows() != child.ndelayed
                || blocks.dc.ncols() != child.ncb
            {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "child delayed-pivot block dimensions are inconsistent",
                });
            }
        }
    }

    let own_nfs = fs_end - fs_start;
    let mut delayed_set: BTreeSet<usize> = BTreeSet::new();
    for child in children_factored {
        for &g in &child.delayed_row_indices {
            debug_assert!(
                g < fs_start,
                "delayed index {g} must lie in a descendant's fs range (< {fs_start})"
            );
            delayed_set.insert(g);
        }
    }
    let delayed: Vec<usize> = delayed_set.into_iter().collect();
    let Some(nfs) = own_nfs.checked_add(delayed.len()) else {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front fully-summed size overflows usize",
        });
    };
    let ncb = cb_indices.len();
    let Some(size) = nfs.checked_add(ncb) else {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front total size overflows usize",
        });
    };

    let mut matrix = DenseMatrix::<T>::zeros(size, size);

    let fs_pos = |g: usize| -> Option<usize> {
        if g >= fs_start && g < fs_end {
            Some(g - fs_start)
        } else {
            delayed.binary_search(&g).ok().map(|r| own_nfs + r)
        }
    };
    let cb_pos = |g: usize| -> Option<usize> { cb_indices.binary_search(&g).ok().map(|p| nfs + p) };
    let fs_or_cb_pos = |g: usize| -> Result<usize, BlrError> {
        fs_pos(g)
            .or_else(|| cb_pos(g))
            .ok_or(BlrError::InvalidFactorization {
                front,
                reason: "child CB index missing from parent front index set",
            })
    };
    // Store value `v` of the logical entry (gr, gc) into the lower triangle.
    let store_lower = |matrix: &mut DenseMatrix<T>, pa: usize, pb: usize, v: T| {
        if pa >= pb {
            matrix[[pa, pb]] += v;
        } else {
            matrix[[pb, pa]] += cj(v, hermitian);
        }
    };

    // (1) Fill from the lower-triangle CSR. Entry (i, j), i ≥ j:
    //   - FS rows: j ∈ own FS → F11. (j ≥ fs_end impossible since j ≤ i;
    //     j < fs_start belongs to a descendant or arrives via delayed blocks.)
    //   - CB rows: j ∈ own FS → F21. CB×CB entries are owned by an ancestor
    //     and skipped, exactly as in the LU assembly.
    for i in fs_start..fs_end {
        let row_local = i - fs_start;
        for idx in a_lower.row_ptrs[i]..a_lower.row_ptrs[i + 1] {
            let j = a_lower.col_indices[idx];
            if j >= fs_start && j < fs_end {
                matrix[[row_local, j - fs_start]] += a_lower.values[idx];
            }
        }
    }
    for (k, &g) in cb_indices.iter().enumerate() {
        let row_local = nfs + k;
        for idx in a_lower.row_ptrs[g]..a_lower.row_ptrs[g + 1] {
            let j = a_lower.col_indices[idx];
            if j >= fs_start && j < fs_end {
                matrix[[row_local, j - fs_start]] += a_lower.values[idx];
            }
        }
    }

    // (2) Extend-add each child's CB contribution. Packed triangles are
    // scattered entry by entry; a compressed panel is decompressed on the
    // fly and only its lower triangle is scattered (the parent's
    // lower-triangle storage would discard the rest anyway).
    for child in children_factored {
        let pos: Vec<usize> = child
            .cb_indices
            .iter()
            .map(|&g| fs_or_cb_pos(g))
            .collect::<Result<_, _>>()?;
        match &child.cb_panel {
            CbPanel::PackedLower(cb_lower) => {
                for a in 0..child.ncb {
                    for b in 0..=a {
                        let v = cb_lower[pl(a, b)];
                        store_lower(&mut matrix, pos[a], pos[b], v);
                    }
                }
            }
            CbPanel::Blr(panel) => {
                let dense = panel.to_dense();
                for a in 0..child.ncb {
                    for b in 0..=a {
                        let v = dense[[a, b]];
                        store_lower(&mut matrix, pos[a], pos[b], v);
                    }
                }
            }
        }
    }

    // (3) Extend-add each child's delayed-pivot blocks. `dd` (packed lower)
    // lands inside the enlarged FS block; `dc` covers the delayed × CB
    // cross terms — the CB × delayed block is its (conjugate) transpose and
    // is handled implicitly by `store_lower`.
    for child in children_factored {
        let Some(blocks) = &child.delayed else {
            continue;
        };
        let nd = child.ndelayed;
        let dpos: Vec<usize> = child
            .delayed_row_indices
            .iter()
            .map(|&g| {
                fs_pos(g).ok_or(BlrError::InvalidFactorization {
                    front,
                    reason: "delayed index missing from parent front index set",
                })
            })
            .collect::<Result<_, _>>()?;
        for a in 0..nd {
            for b in 0..=a {
                let v = blocks.dd_lower[pl(a, b)];
                store_lower(&mut matrix, dpos[a], dpos[b], v);
            }
            for (c, &g) in child.cb_indices.iter().enumerate() {
                let v = blocks.dc[[a, c]];
                store_lower(&mut matrix, dpos[a], fs_or_cb_pos(g)?, v);
            }
        }
    }

    let mut fs_elim_indices: Vec<usize> = (fs_start..fs_end).collect();
    fs_elim_indices.extend_from_slice(&delayed);

    Ok(AssembledSymFront {
        nfs,
        ncb,
        matrix,
        fs_elim_indices,
        cb_indices,
    })
}

/// Compute the CB index set of a symmetric front. With a lower-triangle CSR,
/// FS→CB edges `(i, g)` with `g > i` are stored at row `g`, so the set is
/// found through the column-incoming pattern of the FS columns (the matrix
/// is structurally symmetric, so incoming rows of FS columns are exactly the
/// FS→CB neighbours) plus children's CB indices propagated up.
fn compute_cb_indices(
    a_lower_in: &[Vec<usize>],
    fs_start: usize,
    fs_end: usize,
    children_cb: &[&[usize]],
) -> Vec<usize> {
    let mut cb: BTreeSet<usize> = BTreeSet::new();
    for col in &a_lower_in[fs_start..fs_end] {
        for &i in col {
            if i >= fs_end {
                cb.insert(i);
            }
        }
    }
    for &child_cb in children_cb {
        for &g in child_cb {
            if g >= fs_end {
                cb.insert(g);
            }
        }
    }
    cb.into_iter().collect()
}

/// Factorise one assembled symmetric front in place: Bunch–Kaufman LDLᴴ of
/// the FS block with the Schur complement formed in place over the whole
/// trailing submatrix (CB×CB becomes the packed CB panel; any un-eliminable
/// trailing FS block becomes the delayed payload).
fn factor_sym_front<T: ComplexField + FromPrimitive>(
    mut front: AssembledSymFront<T>,
    parallel_front_threshold: usize,
    pivot_tol_rel: T::Real,
    hermitian: bool,
    blr: Option<&LdltBlrConfig>,
) -> Result<FactoredLdltFront<T>, FrontError> {
    let nfs = front.nfs;
    let ncb = front.ncb;
    let size = nfs + ncb;
    let use_parallel = parallel_front_threshold > 0 && size > parallel_front_threshold;
    let matrix = &mut front.matrix;
    if let Some(index) = first_non_finite(matrix.as_slice()) {
        return Err(FrontError::numerical("symmetric front assembly", index));
    }

    // Scale-aware pivot rejection threshold (same convention as the LU fronts).
    let mut max_abs = T::Real::zero();
    for i in 0..nfs {
        for j in 0..=i {
            let v = matrix[[i, j]].norm();
            if v > max_abs {
                max_abs = v;
            }
        }
    }
    let pivot_tol = pivot_tol_rel
        * T::Real::epsilon()
        * T::Real::from_usize(nfs.max(1)).unwrap_or_else(T::Real::one)
        * max_abs.max(T::Real::min_positive_value());
    if !pivot_tol.is_finite() {
        return Err(FrontError::numerical("symmetric pivot tolerance", 0));
    }

    let mut pivots: Vec<usize> = (0..nfs).collect();
    let mut pivot_kinds: Vec<PivotKind> = vec![PivotKind::OneByOne; nfs];

    let mut k = 0usize;
    while k < nfs {
        let Some(choice) = find_sym_bk_pivot(matrix, k, nfs, pivot_tol, hermitian) else {
            break;
        };
        match choice {
            SymBkPivot::One { row } => {
                if row != k {
                    sym_swap(matrix, size, k, row, hermitian);
                    pivots.swap(k, row);
                }
                let d = matrix[[k, k]];
                let d_inv = d.inv();
                // L column: scale below-diagonal entries of column k.
                for i in (k + 1)..size {
                    matrix[[i, k]] *= d_inv;
                }
                // Symmetric rank-1 update of the trailing lower triangle:
                // S[i, j] -= L[i, k] · d · cj(L[j, k]).
                let lcol: Vec<T> = ((k + 1)..size).map(|i| matrix[[i, k]]).collect();
                let row0 = k + 1;
                let rows = &mut matrix.as_mut_slice()[row0 * size..];
                let update = |(idx, row): (usize, &mut [T])| {
                    let i = row0 + idx;
                    let ld = lcol[idx] * d;
                    for j in row0..=i {
                        row[j] -= ld * cj(lcol[j - row0], hermitian);
                    }
                };
                if use_parallel && size - row0 > 1 {
                    rows.par_chunks_mut(size).enumerate().for_each(update);
                } else {
                    rows.chunks_mut(size).enumerate().for_each(update);
                }
                pivot_kinds[k] = PivotKind::OneByOne;
                k += 1;
            }
            SymBkPivot::Two { row } => {
                if row != k + 1 {
                    sym_swap(matrix, size, k + 1, row, hermitian);
                    pivots.swap(k + 1, row);
                }
                let a = matrix[[k, k]];
                let b = matrix[[k + 1, k]];
                let c = matrix[[k + 1, k + 1]];
                // D = [[a, cj(b)], [b, c]] (Hermitian or complex-symmetric).
                let d_inv = match invert_2x2::<T>(a, cj(b, hermitian), b, c, pivot_tol) {
                    Ok(inverse) => inverse,
                    Err(BlockInverseError::Singular) => {
                        // 2×2 block numerically singular: postpone to the parent.
                        break;
                    }
                    Err(BlockInverseError::NonFinite) => {
                        return Err(FrontError::numerical("symmetric 2x2 pivot inversion", k));
                    }
                };
                // L column pair: L[i, K] = F[i, K] · D⁻¹ for i ≥ k+2.
                for i in (k + 2)..size {
                    let f0 = matrix[[i, k]];
                    let f1 = matrix[[i, k + 1]];
                    matrix[[i, k]] = f0 * d_inv[0][0] + f1 * d_inv[1][0];
                    matrix[[i, k + 1]] = f0 * d_inv[0][1] + f1 * d_inv[1][1];
                }
                let l0: Vec<T> = ((k + 2)..size).map(|i| matrix[[i, k]]).collect();
                let l1: Vec<T> = ((k + 2)..size).map(|i| matrix[[i, k + 1]]).collect();
                // M = L · D per row, then S[i, j] -= M[i, 0]·cj(L[j, 0]) +
                // M[i, 1]·cj(L[j, 1]).
                let cjb = cj(b, hermitian);
                let m0: Vec<T> = (0..size - (k + 2)).map(|t| l0[t] * a + l1[t] * b).collect();
                let m1: Vec<T> = (0..size - (k + 2))
                    .map(|t| l0[t] * cjb + l1[t] * c)
                    .collect();
                let row0 = k + 2;
                let rows = &mut matrix.as_mut_slice()[row0 * size..];
                let update = |(idx, row): (usize, &mut [T])| {
                    let i = row0 + idx;
                    let mi0 = m0[idx];
                    let mi1 = m1[idx];
                    for (t, slot) in row[row0..=i].iter_mut().enumerate() {
                        *slot -= mi0 * cj(l0[t], hermitian) + mi1 * cj(l1[t], hermitian);
                    }
                };
                if use_parallel && size - row0 > 1 {
                    rows.par_chunks_mut(size).enumerate().for_each(update);
                } else {
                    rows.chunks_mut(size).enumerate().for_each(update);
                }
                pivot_kinds[k] = PivotKind::TwoByTwoStart;
                pivot_kinds[k + 1] = PivotKind::TwoByTwoCont;
                k += 2;
            }
        }
    }
    if let Some(index) = first_non_finite(matrix.as_slice()) {
        return Err(FrontError::numerical(
            "symmetric frontal factorization",
            index,
        ));
    }
    let k_elim = k;
    let nd = nfs - k_elim;

    // Pack the factors: F11 lower triangle (L below diag, D on it) and the
    // off-diagonal panels. The panels stay dense / packed-triangle by
    // default; with BLR enabled they go through the same compression
    // machinery as the LU path (right-look style: factorise dense, then
    // compress the finished panels).
    let mut f11_ldl = Vec::with_capacity(k_elim * (k_elim + 1) / 2);
    for i in 0..k_elim {
        for j in 0..=i {
            f11_ldl.push(matrix[[i, j]]);
        }
    }
    let mut l21_dense = DenseMatrix::<T>::zeros(ncb, k_elim);
    for c in 0..ncb {
        for p in 0..k_elim {
            l21_dense[[c, p]] = matrix[[nfs + c, p]];
        }
    }
    let blr_params = blr.map(|cfg| {
        (
            cfg.panel_kind,
            cfg.block_size.max(1),
            T::Real::from_f64(cfg.epsilon.max(0.0)).unwrap_or(T::Real::zero()),
        )
    });
    let l21 = match blr_params {
        Some((kind, block_size, epsilon)) => {
            let panel = Panel::from_dense_with_precision_maybe_parallel(
                &l21_dense,
                kind,
                block_size,
                epsilon,
                LowRankPrecision::Full,
                use_parallel,
            );
            // Storage-gain guard (same rule as `maybe_compress`): keep the
            // dense panel when compression does not pay.
            if panel.stored_entries() < l21_dense.nrows() * l21_dense.ncols() {
                L21Panel::Blr(panel)
            } else {
                L21Panel::Dense(l21_dense)
            }
        }
        None => L21Panel::Dense(l21_dense),
    };
    let cb_panel = match blr_params {
        Some((kind, block_size, epsilon)) => {
            // The panel machinery works on full dense matrices, so
            // materialise the symmetric CB block; only its lower triangle
            // is ever read back at extend-add.
            let mut cb_lower = Vec::with_capacity(ncb * (ncb + 1) / 2);
            let mut cb_dense = DenseMatrix::<T>::zeros(ncb, ncb);
            for a in 0..ncb {
                for b in 0..=a {
                    let v = matrix[[nfs + a, nfs + b]];
                    cb_lower.push(v);
                    cb_dense[[a, b]] = v;
                    cb_dense[[b, a]] = cj(v, hermitian);
                }
            }
            let panel = Panel::from_dense_with_precision_maybe_parallel(
                &cb_dense,
                kind,
                block_size,
                epsilon,
                LowRankPrecision::Full,
                use_parallel,
            );
            // Storage-gain guard against the *packed* triangle: the full
            // symmetric panel starts at 2× the packed storage, so modest
            // ranks are not enough — fall back when compression loses.
            if panel.stored_entries() < cb_lower.len() {
                CbPanel::Blr(panel)
            } else {
                CbPanel::PackedLower(cb_lower)
            }
        }
        None => {
            let mut cb_lower = Vec::with_capacity(ncb * (ncb + 1) / 2);
            for a in 0..ncb {
                for b in 0..=a {
                    cb_lower.push(matrix[[nfs + a, nfs + b]]);
                }
            }
            CbPanel::PackedLower(cb_lower)
        }
    };

    // Delayed payload: the trailing FS block and the delayed × CB cross
    // block were already updated in place by the rank-k updates above.
    let delayed = if nd > 0 {
        let mut dd_lower = Vec::with_capacity(nd * (nd + 1) / 2);
        for a in 0..nd {
            for b in 0..=a {
                dd_lower.push(matrix[[k_elim + a, k_elim + b]]);
            }
        }
        let mut dc = DenseMatrix::<T>::zeros(nd, ncb);
        for a in 0..nd {
            for c in 0..ncb {
                // F[d, cb] sits above the diagonal in the front layout
                // (delayed rows precede the CB columns), so the stored lower
                // entry is its (conjugate) transpose at (cb, d).
                dc[[a, c]] = cj(matrix[[nfs + c, k_elim + a]], hermitian);
            }
        }
        Some(DelayedSymBlocks { dd_lower, dc })
    } else {
        None
    };

    let delayed_row_indices: Vec<usize> = pivots[k_elim..]
        .iter()
        .map(|&p| front.fs_elim_indices[p])
        .collect();
    pivots.truncate(k_elim);
    pivot_kinds.truncate(k_elim);

    let factored = FactoredLdltFront {
        nfs: k_elim,
        ncb,
        fs_elim_indices: front.fs_elim_indices,
        f11_ldl,
        pivot_kinds,
        pivots,
        l21,
        cb_indices: front.cb_indices,
        cb_panel,
        ndelayed: nd,
        delayed_row_indices,
        delayed,
    };
    factored.validate()?;
    Ok(factored)
}

/// Run the numeric multifrontal LDLᴴ factorisation over the assembly tree.
/// Same level-parallel post-order traversal as the LU path; the error return
/// carries the root front index when postponed pivots survive to the root
/// (numerically singular matrix).
fn numeric_factorise<T: ComplexField + FromPrimitive>(
    a_lower: &CsrMatrix<T>,
    tree: &AssemblyTree,
    parallel_front_threshold: usize,
    pivot_tol_rel: T::Real,
    hermitian: bool,
    blr: Option<&LdltBlrConfig>,
) -> Result<Vec<FactoredLdltFront<T>>, BlrError> {
    if a_lower.num_rows != a_lower.num_cols || a_lower.validate_structure().is_err() {
        return Err(BlrError::InvalidFactorization {
            front: 0,
            reason: "front matrix CSR structure is invalid",
        });
    }
    let n_fronts = tree.fronts.len();
    let mut factored: Vec<Option<FactoredLdltFront<T>>> = (0..n_fronts).map(|_| None).collect();

    // Column-incoming pattern of the lower-triangle CSR, used by
    // `compute_cb_indices` to find FS→CB edges (stored at the CB row).
    let a_lower_in = build_column_incoming(a_lower);

    for level_fronts in &tree.levels {
        let mut inputs: Vec<(usize, AssembledSymFront<T>)> = Vec::with_capacity(level_fronts.len());
        for &front_idx in level_fronts {
            let f = &tree.fronts[front_idx];
            let mut children_factored: Vec<&FactoredLdltFront<T>> =
                Vec::with_capacity(f.children.len());
            for &child in &f.children {
                let child_front =
                    factored[child]
                        .as_ref()
                        .ok_or(BlrError::InvalidFactorization {
                            front: front_idx,
                            reason: "assembly tree visited a parent before its child",
                        })?;
                children_factored.push(child_front);
            }
            let children_cb_slices: Vec<&[usize]> = children_factored
                .iter()
                .map(|c| c.cb_indices.as_slice())
                .collect();
            let cb_indices =
                compute_cb_indices(&a_lower_in, f.fs_start, f.fs_end, &children_cb_slices);
            let assembled = assemble_front(
                a_lower,
                f.fs_start,
                f.fs_end,
                cb_indices,
                &children_factored,
                hermitian,
                front_idx,
            )?;
            inputs.push((front_idx, assembled));
        }

        let results: Vec<(usize, Result<FactoredLdltFront<T>, FrontError>)> = inputs
            .into_par_iter()
            .map(|(idx, assembled)| {
                (
                    idx,
                    factor_sym_front(
                        assembled,
                        parallel_front_threshold,
                        pivot_tol_rel,
                        hermitian,
                        blr,
                    ),
                )
            })
            .collect();

        for (idx, result) in results {
            let f = result.map_err(|error| map_front_error(idx, error))?;
            if tree.fronts[idx].parent.is_none() && f.ndelayed > 0 {
                return Err(BlrError::SingularFront(idx));
            }
            factored[idx] = Some(f);
        }
    }

    let mut fronts = Vec::with_capacity(n_fronts);
    for (front, factor) in factored.into_iter().enumerate() {
        fronts.push(factor.ok_or(BlrError::InvalidFactorization {
            front,
            reason: "assembly tree did not produce a factor for every front",
        })?);
    }
    Ok(fronts)
}

impl<T: ComplexField + FromPrimitive> FactoredLdltFront<T> {
    fn validate(&self) -> Result<(), FrontError> {
        let packed_len = self
            .nfs
            .checked_add(1)
            .and_then(|next| self.nfs.checked_mul(next))
            .and_then(|value| value.checked_div(2))
            .ok_or(FrontError::Invalid("packed F11 size overflows usize"))?;
        if self.f11_ldl.len() != packed_len {
            return Err(FrontError::Invalid("packed F11 length does not match nfs"));
        }
        if self.pivot_kinds.len() != self.nfs || self.pivots.len() != self.nfs {
            return Err(FrontError::Invalid("pivot metadata length mismatch"));
        }
        if self.fs_elim_indices.len() != self.nfs + self.ndelayed
            || self.cb_indices.len() != self.ncb
        {
            return Err(FrontError::Invalid("front index metadata length mismatch"));
        }
        if self
            .pivots
            .iter()
            .any(|&pivot| pivot >= self.fs_elim_indices.len())
        {
            return Err(FrontError::Invalid("pivot index is out of range"));
        }
        let mut pivot = 0usize;
        while pivot < self.nfs {
            match self.pivot_kinds[pivot] {
                PivotKind::OneByOne => pivot += 1,
                PivotKind::TwoByTwoStart
                    if pivot + 1 < self.nfs
                        && self.pivot_kinds[pivot + 1] == PivotKind::TwoByTwoCont =>
                {
                    pivot += 2;
                }
                _ => return Err(FrontError::Invalid("malformed 2x2 pivot metadata")),
            }
        }
        if self.l21.dimensions() != (self.ncb, self.nfs) {
            return Err(FrontError::Invalid("L21 panel dimensions are inconsistent"));
        }
        match &self.cb_panel {
            CbPanel::PackedLower(values) => {
                let expected = self
                    .ncb
                    .checked_add(1)
                    .and_then(|next| self.ncb.checked_mul(next))
                    .and_then(|value| value.checked_div(2))
                    .ok_or(FrontError::Invalid("packed CB size overflows usize"))?;
                if values.len() != expected {
                    return Err(FrontError::Invalid("packed CB length is inconsistent"));
                }
            }
            CbPanel::Blr(panel) if panel.n_rows() != self.ncb || panel.n_cols() != self.ncb => {
                return Err(FrontError::Invalid(
                    "compressed CB dimensions are inconsistent",
                ));
            }
            CbPanel::Blr(_) => {}
        }
        if self.ndelayed != self.delayed_row_indices.len()
            || (self.ndelayed == 0) != self.delayed.is_none()
        {
            return Err(FrontError::Invalid(
                "delayed-pivot metadata is inconsistent",
            ));
        }
        if let Some(delayed) = &self.delayed {
            let expected = self
                .ndelayed
                .checked_add(1)
                .and_then(|next| self.ndelayed.checked_mul(next))
                .and_then(|value| value.checked_div(2))
                .ok_or(FrontError::Invalid("packed delayed size overflows usize"))?;
            if delayed.dd_lower.len() != expected
                || delayed.dc.nrows() != self.ndelayed
                || delayed.dc.ncols() != self.ncb
            {
                return Err(FrontError::Invalid(
                    "delayed-pivot block dimensions are inconsistent",
                ));
            }
            if let Some(index) = first_non_finite(&delayed.dd_lower) {
                return Err(FrontError::numerical("delayed symmetric DD block", index));
            }
            if let Some(index) = first_non_finite(delayed.dc.as_slice()) {
                return Err(FrontError::numerical("delayed symmetric DC block", index));
            }
        }
        if let Some(index) = first_non_finite(&self.f11_ldl) {
            return Err(FrontError::numerical("stored LDL factor", index));
        }
        if !self.l21.is_finite() || !self.cb_panel.is_finite() {
            return Err(FrontError::numerical(
                "compressed symmetric panel storage",
                0,
            ));
        }
        Ok(())
    }

    /// In-place forward apply (unit-`L` part only) on a front-local vector of
    /// length `nfs + ncb`: `b[0..nfs] := L₁₁⁻¹ · b[0..nfs]` (block-aware),
    /// then `b[nfs..] -= L21 · b[0..nfs]`. The FS slice must already be
    /// gathered in factored (pivoted) order.
    fn forward_apply(&self, b: &mut [T]) -> Result<(), FrontError> {
        self.validate()?;
        let nfs = self.nfs;
        if b.len() != nfs + self.ncb {
            return Err(FrontError::Invalid("front vector length mismatch"));
        }
        for i in 0..nfs {
            let mut s = b[i];
            #[allow(clippy::needless_range_loop)]
            for p in 0..i {
                // Within-block slot of a 2×2 cont row holds D[1, 0], not L.
                if self.pivot_kinds[i] == PivotKind::TwoByTwoCont && p == i - 1 {
                    continue;
                }
                s -= self.f11_ldl[pl(i, p)] * b[p];
            }
            b[i] = s;
            if !scalar_is_finite(s) {
                return Err(FrontError::numerical(
                    "symmetric front forward substitution",
                    i,
                ));
            }
        }
        if self.ncb > 0 {
            let (b_fs, b_cb) = b.split_at_mut(nfs);
            self.l21.gemv_into(-T::one(), b_fs, b_cb);
            if let Some(index) = first_non_finite(b_cb) {
                return Err(FrontError::numerical("symmetric front L21 update", index));
            }
        }
        Ok(())
    }

    /// Apply `D⁻¹` (1×1 and 2×2 blocks) to the FS slice in factored order.
    fn diagonal_apply(&self, z: &mut [T], hermitian: bool) -> Result<(), FrontError> {
        self.validate()?;
        let nfs = self.nfs;
        if z.len() != nfs {
            return Err(FrontError::Invalid("diagonal vector length mismatch"));
        }
        let mut k = 0usize;
        while k < nfs {
            match self.pivot_kinds[k] {
                PivotKind::OneByOne => {
                    z[k] *= self.f11_ldl[pl(k, k)].inv();
                    if !scalar_is_finite(z[k]) {
                        return Err(FrontError::numerical("D one-by-one solve", k));
                    }
                    k += 1;
                }
                PivotKind::TwoByTwoStart => {
                    let a = self.f11_ldl[pl(k, k)];
                    let b = self.f11_ldl[pl(k + 1, k)];
                    let c = self.f11_ldl[pl(k + 1, k + 1)];
                    let d_inv = invert_2x2::<T>(a, cj(b, hermitian), b, c, T::Real::zero())
                        .map_err(|_| FrontError::numerical("D 2x2 block solve", k))?;
                    let z0 = z[k];
                    let z1 = z[k + 1];
                    z[k] = d_inv[0][0] * z0 + d_inv[0][1] * z1;
                    z[k + 1] = d_inv[1][0] * z0 + d_inv[1][1] * z1;
                    if !scalar_is_finite(z[k]) || !scalar_is_finite(z[k + 1]) {
                        return Err(FrontError::numerical("D 2x2 block solve", k));
                    }
                    k += 2;
                }
                PivotKind::TwoByTwoCont => {
                    return Err(FrontError::Invalid(
                        "TwoByTwoCont without preceding TwoByTwoStart",
                    ));
                }
            }
        }
        Ok(())
    }

    /// In-place backward apply: `b[0..nfs] := L₁₁⁻ᴴ · (b[0..nfs] − L21ᴴ ·
    /// b[nfs..])`, block-aware. The FS slice arrives in factored order
    /// holding `D⁻¹ · z`; the result is scattered with the same pivot map.
    fn backward_apply(&self, b: &mut [T], hermitian: bool) -> Result<(), FrontError> {
        self.validate()?;
        let nfs = self.nfs;
        if b.len() != nfs + self.ncb {
            return Err(FrontError::Invalid("front vector length mismatch"));
        }
        // t = y − L21ᴴ · w_cb. The panel transpose-GEMV applies the plain
        // transpose without conjugation, so the Hermitian case sandwiches
        // it: L21ᴴ·w = conj(L21ᵀ·conj(w)).
        if self.ncb > 0 {
            let (b_fs, b_cb) = b.split_at_mut(nfs);
            if hermitian {
                let x: Vec<T> = b_cb.iter().map(|v| v.conj()).collect();
                let mut tmp = vec![T::zero(); nfs];
                self.l21.gemv_transpose_into(T::one(), &x, &mut tmp);
                for (bp, t) in b_fs.iter_mut().zip(tmp) {
                    *bp -= t.conj();
                }
            } else {
                self.l21.gemv_transpose_into(-T::one(), b_cb, b_fs);
            }
            if let Some(index) = first_non_finite(b_fs) {
                return Err(FrontError::numerical(
                    "symmetric front transpose update",
                    index,
                ));
            }
        }
        // L₁₁ᴴ is unit upper triangular (with unit 2×2 block diagonal):
        // solve from the bottom up.
        let mut i = nfs;
        while i > 0 {
            if self.pivot_kinds[i - 1] == PivotKind::TwoByTwoCont {
                let lo = i - 2;
                let mut s0 = b[lo];
                let mut s1 = b[lo + 1];
                #[allow(clippy::needless_range_loop)]
                for j in (lo + 2)..nfs {
                    s0 -= cj(self.f11_ldl[pl(j, lo)], hermitian) * b[j];
                    s1 -= cj(self.f11_ldl[pl(j, lo + 1)], hermitian) * b[j];
                }
                b[lo] = s0;
                b[lo + 1] = s1;
                if !scalar_is_finite(s0) || !scalar_is_finite(s1) {
                    return Err(FrontError::numerical(
                        "symmetric front block back substitution",
                        lo,
                    ));
                }
                i -= 2;
            } else {
                let k = i - 1;
                let mut s = b[k];
                #[allow(clippy::needless_range_loop)]
                for j in (k + 1)..nfs {
                    s -= cj(self.f11_ldl[pl(j, k)], hermitian) * b[j];
                }
                b[k] = s;
                if !scalar_is_finite(s) {
                    return Err(FrontError::numerical(
                        "symmetric front back substitution",
                        k,
                    ));
                }
                i -= 1;
            }
        }
        Ok(())
    }

    /// Total stored entries (packed F11 + L21 panel + CB panel + dense
    /// delayed blocks still held for the parent).
    fn stored_entries(&self) -> usize {
        let delayed_entries = self
            .delayed
            .as_ref()
            .map(|d| d.dd_lower.len() + d.dc.nrows() * d.dc.ncols())
            .unwrap_or(0);
        self.f11_ldl.len()
            + self.l21.stored_entries()
            + self.cb_panel.stored_entries()
            + delayed_entries
    }
}

struct ForwardFrontResult<T> {
    fs_elim: Vec<usize>,
    fs_values: Vec<T>,
    cb_deltas: Vec<(usize, T)>,
}

struct BackwardFrontResult<T> {
    fs_elim: Vec<usize>,
    fs_values: Vec<T>,
}

/// Forward (L), diagonal (D) and backward (Lᴴ) solves against the factored
/// assembly tree. Mirrors the LU path's traversal: post-order forward with
/// per-front deltas reduced deterministically, a tree-local diagonal phase,
/// then a pre-order backward pass.
fn forward_backward_solve<T: ComplexField + FromPrimitive>(
    tree: &AssemblyTree,
    fronts: &[FactoredLdltFront<T>],
    b_perm: &[T],
    hermitian: bool,
) -> Result<Vec<T>, BlrError> {
    let n = b_perm.len();
    if tree.fronts.len() != fronts.len() {
        return Err(BlrError::InvalidFactorization {
            front: 0,
            reason: "assembly-tree and numeric-front counts differ",
        });
    }

    let mut z_global = vec![T::zero(); n];
    let mut cb_global = vec![T::zero(); n];
    cb_global[..n].copy_from_slice(&b_perm[..n]);

    for level in &tree.levels {
        let results: Vec<(usize, Result<ForwardFrontResult<T>, FrontError>)> = if level.len() > 1 {
            level
                .par_iter()
                .map(|&k| (k, forward_front(fronts, &cb_global, k)))
                .collect()
        } else {
            level
                .iter()
                .map(|&k| (k, forward_front(fronts, &cb_global, k)))
                .collect()
        };
        for (front, result) in results {
            let result = result.map_err(|error| map_front_error(front, error))?;
            for (g, value) in result.fs_elim.into_iter().zip(result.fs_values) {
                if g >= n {
                    return Err(BlrError::InvalidFactorization {
                        front,
                        reason: "eliminated index is out of global range",
                    });
                }
                z_global[g] = value;
                cb_global[g] = T::zero();
            }
            for (g, delta) in result.cb_deltas {
                if g >= n {
                    return Err(BlrError::InvalidFactorization {
                        front,
                        reason: "contribution index is out of global range",
                    });
                }
                cb_global[g] += delta;
                if !scalar_is_finite(cb_global[g]) {
                    return Err(BlrError::FrontNumericalBreakdown {
                        front,
                        stage: "symmetric contribution accumulation",
                        index: g,
                    });
                }
            }
        }
    }

    // Diagonal phase: y = D⁻¹ · z, block-diagonal per front.
    for (front, ff) in fronts.iter().enumerate() {
        if ff.nfs == 0 {
            continue;
        }
        ff.validate()
            .map_err(|error| map_front_error(front, error))?;
        let mut z = Vec::with_capacity(ff.nfs);
        for i in 0..ff.nfs {
            let global = ff.fs_elim_indices[ff.pivots[i]];
            if global >= n {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "diagonal index is out of global range",
                });
            }
            z.push(z_global[global]);
        }
        ff.diagonal_apply(&mut z, hermitian)
            .map_err(|error| map_front_error(front, error))?;
        for (i, value) in z.into_iter().enumerate() {
            let global = ff.fs_elim_indices[ff.pivots[i]];
            if global >= n {
                return Err(BlrError::InvalidFactorization {
                    front,
                    reason: "diagonal index is out of global range",
                });
            }
            z_global[global] = value;
        }
    }

    let mut w_global = vec![T::zero(); n];
    for level in tree.levels.iter().rev() {
        let results: Vec<(usize, Result<BackwardFrontResult<T>, FrontError>)> = if level.len() > 1 {
            level
                .par_iter()
                .map(|&k| {
                    (
                        k,
                        backward_front(fronts, &z_global, &w_global, k, hermitian),
                    )
                })
                .collect()
        } else {
            level
                .iter()
                .map(|&k| {
                    (
                        k,
                        backward_front(fronts, &z_global, &w_global, k, hermitian),
                    )
                })
                .collect()
        };
        for (front, result) in results {
            let result = result.map_err(|error| map_front_error(front, error))?;
            for (g, value) in result.fs_elim.into_iter().zip(result.fs_values) {
                if g >= n {
                    return Err(BlrError::InvalidFactorization {
                        front,
                        reason: "solution index is out of global range",
                    });
                }
                w_global[g] = value;
            }
        }
    }

    Ok(w_global)
}

fn forward_front<T: ComplexField + FromPrimitive>(
    fronts: &[FactoredLdltFront<T>],
    cb_global: &[T],
    k: usize,
) -> Result<ForwardFrontResult<T>, FrontError> {
    let ff = fronts
        .get(k)
        .ok_or(FrontError::Invalid("front index is out of range"))?;
    ff.validate()?;
    let nfs = ff.nfs;
    let ncb = ff.ncb;

    let mut b_local = vec![T::zero(); nfs + ncb];
    let mut fs_elim = Vec::with_capacity(nfs);
    for (i, slot) in b_local[..nfs].iter_mut().enumerate() {
        let g = ff.fs_elim_indices[ff.pivots[i]];
        if g >= cb_global.len() {
            return Err(FrontError::Invalid(
                "eliminated index is out of global range",
            ));
        }
        *slot = cb_global[g];
        fs_elim.push(g);
    }

    let mut cb_before = Vec::with_capacity(ncb);
    for (j, &g) in ff.cb_indices.iter().enumerate() {
        if g >= cb_global.len() {
            return Err(FrontError::Invalid(
                "contribution index is out of global range",
            ));
        }
        let value = cb_global[g];
        b_local[nfs + j] = value;
        cb_before.push(value);
    }

    ff.forward_apply(&mut b_local)?;

    let fs_values = b_local[..nfs].to_vec();
    let cb_deltas = ff
        .cb_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(j, g)| (g, b_local[nfs + j] - cb_before[j]))
        .collect();

    Ok(ForwardFrontResult {
        fs_elim,
        fs_values,
        cb_deltas,
    })
}

fn backward_front<T: ComplexField + FromPrimitive>(
    fronts: &[FactoredLdltFront<T>],
    y_global: &[T],
    w_global: &[T],
    k: usize,
    hermitian: bool,
) -> Result<BackwardFrontResult<T>, FrontError> {
    let ff = fronts
        .get(k)
        .ok_or(FrontError::Invalid("front index is out of range"))?;
    ff.validate()?;
    let nfs = ff.nfs;
    let ncb = ff.ncb;

    let mut b_local = vec![T::zero(); nfs + ncb];
    for (i, slot) in b_local[..nfs].iter_mut().enumerate() {
        let global = ff.fs_elim_indices[ff.pivots[i]];
        if global >= y_global.len() {
            return Err(FrontError::Invalid(
                "eliminated index is out of global range",
            ));
        }
        *slot = y_global[global];
    }
    for (j, &g) in ff.cb_indices.iter().enumerate() {
        if g >= w_global.len() {
            return Err(FrontError::Invalid(
                "contribution index is out of global range",
            ));
        }
        b_local[nfs + j] = w_global[g];
    }

    ff.backward_apply(&mut b_local, hermitian)?;

    let mut fs_elim = Vec::with_capacity(nfs);
    let mut fs_values = Vec::with_capacity(nfs);
    for (i, &value) in b_local[..nfs].iter().enumerate() {
        fs_elim.push(ff.fs_elim_indices[ff.pivots[i]]);
        fs_values.push(value);
    }

    Ok(BackwardFrontResult { fs_elim, fs_values })
}

/// Multifrontal sparse LDLᵀ / LDLᴴ factorisation for symmetric / Hermitian
/// indefinite matrices. See the [module documentation](self) for the
/// algorithm; shares the assembly tree, amalgamation, delayed pivoting and
/// error conventions with [`super::multifrontal_blr`].
pub struct MultifrontalLdltFactorization<T: ComplexField> {
    n: usize,
    tree: AssemblyTree,
    fronts: Vec<FactoredLdltFront<T>>,
    hermitian: bool,
}

/// Sign counts of the eigenvalues of a real symmetric matrix.
///
/// For a nonsingular factorisation `zero` is normally zero. It is retained
/// in the public contract because a numerically rank-deficient 2x2 pivot can
/// expose a zero eigenvalue at the factorisation tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixInertia {
    /// Number of strictly positive eigenvalues.
    pub positive: usize,
    /// Number of strictly negative eigenvalues.
    pub negative: usize,
    /// Number of numerically zero eigenvalues.
    pub zero: usize,
}

impl<T: ComplexField> MultifrontalLdltFactorization<T> {
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }
}

impl<T: ComplexField + FromPrimitive> MultifrontalLdltFactorization<T> {
    /// Factorise with default configuration (Hermitian).
    pub fn factorize(matrix: &CsrMatrix<T>) -> Result<Self, BlrError> {
        Self::factorize_with_config(matrix, &LdltConfig::default())
    }

    /// Factorise with a custom configuration.
    ///
    /// The matrix may store both triangles, only the lower, or only the
    /// upper triangle of the symmetric/Hermitian pattern; one triangle is
    /// extracted internally.
    pub fn factorize_with_config(
        matrix: &CsrMatrix<T>,
        cfg: &LdltConfig,
    ) -> Result<Self, BlrError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(BlrError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        if let Some(index) = first_non_finite(&matrix.values) {
            return Err(BlrError::NonFiniteMatrix { index });
        }
        matrix
            .validate_structure()
            .map_err(|error| BlrError::InvalidCsrStructure(error.to_string()))?;
        if !cfg.pivot_tol_multiplier.is_finite() || cfg.pivot_tol_multiplier < 0.0 {
            return Err(BlrError::InvalidConfiguration {
                field: "pivot_tol_multiplier",
            });
        }
        if let Some(blr) = &cfg.blr
            && (!blr.epsilon.is_finite() || blr.epsilon < 0.0)
        {
            return Err(BlrError::InvalidConfiguration {
                field: "blr.epsilon",
            });
        }
        if T::Real::from_f64(cfg.pivot_tol_multiplier)
            .filter(|value| value.is_finite())
            .is_none()
        {
            return Err(BlrError::InvalidConfiguration {
                field: "pivot_tol_multiplier",
            });
        }
        if let Some(blr) = &cfg.blr
            && T::Real::from_f64(blr.epsilon)
                .filter(|value| value.is_finite())
                .is_none()
        {
            return Err(BlrError::InvalidConfiguration {
                field: "blr.epsilon",
            });
        }
        if n == 0 {
            return Ok(Self {
                n: 0,
                tree: AssemblyTree {
                    perm: Vec::new(),
                    perm_inv: Vec::new(),
                    fronts: Vec::new(),
                    levels: Vec::new(),
                },
                fronts: Vec::new(),
                hermitian: cfg.hermitian,
            });
        }

        let raw_tree = build_assembly_tree(matrix, cfg.min_block_size.max(1), cfg.max_levels);
        let tree = if cfg.amalgamation_threshold > 0 {
            amalgamate_tree(&raw_tree, cfg.amalgamation_threshold)
        } else {
            raw_tree
        };
        let a_perm = permute_matrix(matrix, &tree.perm_inv);
        let a_lower = lower_triangle_csr(&a_perm, cfg.hermitian);
        let pivot_tol_rel =
            T::Real::from_f64(cfg.pivot_tol_multiplier.max(0.0)).unwrap_or(T::Real::one());
        let fronts = numeric_factorise(
            &a_lower,
            &tree,
            cfg.parallel_front_threshold,
            pivot_tol_rel,
            cfg.hermitian,
            cfg.blr.as_ref(),
        )?;

        Ok(Self {
            n,
            tree,
            fronts,
            hermitian: cfg.hermitian,
        })
    }

    /// Solve `A · x = b` using the pre-computed factorisation.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        match self.try_solve(b) {
            Ok(solution) => solution,
            Err(error) => panic!("multifrontal LDLT solve failed: {error}"),
        }
    }

    /// Checked solve that rejects malformed/non-finite input and generated
    /// numerical breakdowns without panicking or publishing a partial result.
    pub fn try_solve(&self, b: &[T]) -> Result<Vec<T>, BlrError> {
        if b.len() != self.n {
            return Err(BlrError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(BlrError::NonFiniteRhs { index });
        }
        if self.n == 0 {
            return Ok(Vec::new());
        }

        let mut b_perm = vec![T::zero(); self.n];
        for i in 0..self.n {
            b_perm[self.tree.perm_inv[i]] = b[i];
        }

        let w_perm = forward_backward_solve(&self.tree, &self.fronts, &b_perm, self.hermitian)?;

        let mut x = vec![T::zero(); self.n];
        for g in 0..self.n {
            x[g] = w_perm[self.tree.perm_inv[g]];
        }
        if let Some(index) = first_non_finite(&x) {
            return Err(BlrError::NumericalBreakdown {
                stage: "final symmetric solution permutation",
                index,
            });
        }
        Ok(x)
    }

    /// Total stored entries across all fronts (packed F11 + L21 panel + CB
    /// panel + dense delayed blocks), times the scalar size. Compressed
    /// panels report their compressed entry counts.
    pub fn factor_memory_bytes(&self) -> usize {
        let entries = self.fronts.iter().fold(0usize, |total, front| {
            total.saturating_add(front.stored_entries())
        });
        entries.saturating_mul(std::mem::size_of::<T>())
    }

    /// Number of fronts in the assembly tree.
    pub fn num_fronts(&self) -> usize {
        self.fronts.len()
    }

    /// Total number of pivots postponed to a parent front across the tree
    /// (0 when no delayed pivoting was needed).
    pub fn total_delayed_pivots(&self) -> usize {
        self.fronts.iter().map(|f| f.ndelayed).sum()
    }

    /// Number of 2×2 Bunch–Kaufman pivot blocks across all fronts.
    pub fn two_by_two_pivot_count(&self) -> usize {
        self.fronts
            .iter()
            .flat_map(|f| f.pivot_kinds.iter())
            .filter(|&&kind| kind == PivotKind::TwoByTwoStart)
            .count()
    }
}

impl MultifrontalLdltFactorization<f64> {
    /// Return the inertia of the factored real symmetric matrix.
    ///
    /// Sylvester's law of inertia makes the sign counts of the block diagonal
    /// `D` factor identical to those of the original matrix. The method is
    /// therefore independent of fill-reducing permutations and delayed
    /// Bunch-Kaufman pivots.
    pub fn inertia(&self) -> MatrixInertia {
        let mut inertia = MatrixInertia {
            positive: 0,
            negative: 0,
            zero: 0,
        };

        for front in &self.fronts {
            let mut pivot = 0;
            while pivot < front.nfs {
                match front.pivot_kinds[pivot] {
                    PivotKind::OneByOne => {
                        classify_sign(front.f11_ldl[pl(pivot, pivot)], &mut inertia);
                        pivot += 1;
                    }
                    PivotKind::TwoByTwoStart => {
                        let a = front.f11_ldl[pl(pivot, pivot)];
                        let b = front.f11_ldl[pl(pivot + 1, pivot)];
                        let c = front.f11_ldl[pl(pivot + 1, pivot + 1)];
                        classify_symmetric_2x2(a, b, c, &mut inertia);
                        pivot += 2;
                    }
                    PivotKind::TwoByTwoCont => {
                        unreachable!("TwoByTwoCont without preceding TwoByTwoStart")
                    }
                }
            }
        }

        debug_assert_eq!(inertia.positive + inertia.negative + inertia.zero, self.n);
        inertia
    }
}

fn classify_sign(value: f64, inertia: &mut MatrixInertia) {
    if value > 0.0 {
        inertia.positive += 1;
    } else if value < 0.0 {
        inertia.negative += 1;
    } else {
        inertia.zero += 1;
    }
}

fn classify_symmetric_2x2(a: f64, b: f64, c: f64, inertia: &mut MatrixInertia) {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 {
        inertia.zero += 2;
        return;
    }

    let a_scaled = a / scale;
    let b_scaled = b / scale;
    let c_scaled = c / scale;
    let product = a_scaled * c_scaled;
    let square = b_scaled * b_scaled;
    let determinant = product - square;
    let determinant_tolerance = 64.0 * f64::EPSILON * (product.abs() + square).max(1.0);

    if determinant.abs() <= determinant_tolerance {
        inertia.zero += 1;
        classify_sign(a_scaled + c_scaled, inertia);
    } else if determinant < 0.0 {
        inertia.positive += 1;
        inertia.negative += 1;
    } else if a_scaled + c_scaled > 0.0 {
        inertia.positive += 2;
    } else {
        inertia.negative += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multifrontal_blr::{
        BlrConfig, MultifrontalBlrFactorization,
        ordering::{AssemblyTree, FrontInfo},
    };
    use num_complex::Complex64;
    use sonium_solvers_core::dense::DenseMatrix;

    fn rel_residual<T: ComplexField>(m: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let n = b.len();
        let mut res_sq = T::Real::zero();
        let mut b_sq = T::Real::zero();
        for i in 0..n {
            let mut s = T::zero();
            for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
                s += m.values[idx] * x[m.col_indices[idx]];
            }
            let r = b[i] - s;
            res_sq += r.norm_sqr();
            b_sq += b[i].norm_sqr();
        }
        (res_sq / b_sq).sqrt()
    }

    fn shifted_laplacian_2d(nx: usize, sigma: f64) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0 - sigma;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn shifted_laplacian_3d(nx: usize, sigma: f64) -> CsrMatrix<f64> {
        let n = nx * nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iz in 0..nx {
            for iy in 0..nx {
                for ix in 0..nx {
                    let i = iz * nx * nx + iy * nx + ix;
                    dense[[i, i]] = 6.0 - sigma;
                    if ix > 0 {
                        dense[[i, i - 1]] = -1.0;
                    }
                    if ix + 1 < nx {
                        dense[[i, i + 1]] = -1.0;
                    }
                    if iy > 0 {
                        dense[[i, i - nx]] = -1.0;
                    }
                    if iy + 1 < nx {
                        dense[[i, i + nx]] = -1.0;
                    }
                    if iz > 0 {
                        dense[[i, i - nx * nx]] = -1.0;
                    }
                    if iz + 1 < nx {
                        dense[[i, i + nx * nx]] = -1.0;
                    }
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn test_config() -> LdltConfig {
        LdltConfig {
            min_block_size: 16,
            // Below the matrix size so the memory test exercises a real tree.
            amalgamation_threshold: 64,
            ..LdltConfig::default()
        }
    }

    fn blr_lu_config() -> BlrConfig {
        BlrConfig {
            epsilon: 0.0,
            min_block_size: 16,
            amalgamation_threshold: 64,
            ..BlrConfig::default()
        }
    }

    #[test]
    fn shifted_laplacian_2d_indefinite_matches_lu_path() {
        // σ = 3.5 sits inside the (0, 8) spectrum of the 2D Laplacian:
        // K − σM is symmetric indefinite.
        let m = shifted_laplacian_2d(12, 3.5);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(&m, &test_config())
            .expect("LDLT factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "LDLT residual {res:e} too large");

        let lu = MultifrontalBlrFactorization::factorize_with_config(&m, &blr_lu_config())
            .expect("LU factorise");
        let x_lu = lu.solve(&b);
        for i in 0..n {
            assert!(
                (x[i] - x_lu[i]).abs() < 1e-9,
                "LDLT vs multifrontal LU diverge at {i}: {} vs {}",
                x[i],
                x_lu[i]
            );
        }
    }

    #[test]
    fn shifted_laplacian_3d_indefinite_solves_and_uses_half_the_memory() {
        // σ = 5.5 sits inside the (0, 12) spectrum of the 3D Laplacian
        // (σ = 5.0 would be an exact eigenvalue of the 6³ grid).
        let m = shifted_laplacian_3d(6, 5.5);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.13).cos()).collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(&m, &test_config())
            .expect("LDLT factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "LDLT residual {res:e} too large");
        assert!(ldlt.num_fronts() > 1, "test must exercise a real tree");

        let lu = MultifrontalBlrFactorization::factorize_with_config(&m, &blr_lu_config())
            .expect("LU factorise");
        let x_lu = lu.solve(&b);
        let res_lu = rel_residual(&m, &x_lu, &b);
        assert!(res_lu < 1e-10, "LU residual {res_lu:e} too large");

        let ldlt_bytes = ldlt.factor_memory_bytes();
        let lu_bytes = lu.factor_memory_bytes();
        eprintln!(
            "memory: LDLT = {ldlt_bytes} B, multifrontal LU = {lu_bytes} B, \
             ratio = {:.3}",
            ldlt_bytes as f64 / lu_bytes as f64
        );
        assert!(
            ldlt_bytes * 5 < lu_bytes * 3,
            "LDLT ({ldlt_bytes}) should use ≤ ~60% of the LU path memory ({lu_bytes})"
        );
    }

    #[test]
    fn saddle_point_block_solves() {
        // [K Bᵀ; B 0] with K SPD tridiagonal and B full row rank.
        let nk = 4;
        let n = nk + 2;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..nk {
            dense[[i, i]] = 2.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
                dense[[i - 1, i]] = -1.0;
            }
        }
        dense[[0, nk]] = 1.0;
        dense[[nk, 0]] = 1.0;
        dense[[2, nk]] = 1.0;
        dense[[nk, 2]] = 1.0;
        dense[[1, nk + 1]] = 1.0;
        dense[[nk + 1, 1]] = 1.0;
        dense[[3, nk + 1]] = 1.0;
        dense[[nk + 1, 3]] = 1.0;
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).cos()).collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 2,
                ..LdltConfig::default()
            },
        )
        .expect("LDLT factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "saddle-point residual {res:e} too large");
    }

    /// `k` disjoint strong `[[0, 1], [1, 0]]` pairs on a weak ring: every
    /// diagonal is exactly zero, so no 1×1 pivot is acceptable anywhere and
    /// Bunch–Kaufman must emit 2×2 blocks.
    fn zero_diag_pairs(k: usize) -> CsrMatrix<f64> {
        let n = 2 * k;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..k {
            dense[[2 * i, 2 * i + 1]] = 1.0;
            dense[[2 * i + 1, 2 * i]] = 1.0;
        }
        for i in 0..n {
            if i + 2 < n {
                dense[[i, i + 2]] = 0.05;
                dense[[i + 2, i]] = 0.05;
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn zero_diagonal_forces_2x2_pivots() {
        let m = zero_diag_pairs(12);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).sin()).collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 4,
                ..LdltConfig::default()
            },
        )
        .expect("LDLT factorise");
        assert!(
            ldlt.two_by_two_pivot_count() > 0,
            "zero-diagonal matrix must produce 2×2 Bunch–Kaufman pivots"
        );
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "2×2-pivot residual {res:e} too large");
    }

    /// Matching graph: `k` disjoint pairs `(v_i, u_i)` with coupling 1,
    /// diagonal 2 on `u_i` and *no* diagonal on `v_i` (same construction as
    /// the LU delayed-pivoting tests).
    fn matching_matrix(k: usize, drop_first_coupling: bool) -> CsrMatrix<f64> {
        let n = 2 * k;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..k {
            if drop_first_coupling && i == 0 {
                continue;
            }
            dense[[i, k + i]] = 1.0;
            dense[[k + i, i]] = 1.0;
        }
        for i in 0..k {
            dense[[k + i, k + i]] = 2.0;
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    /// Two-front tree: front 0 owns the `v`s (zero F11 → every pivot is
    /// postponed), front 1 (root) owns the `u`s and absorbs the delayed set.
    fn two_front_tree(k: usize) -> AssemblyTree {
        let n = 2 * k;
        AssemblyTree {
            perm: (0..n).collect(),
            perm_inv: (0..n).collect(),
            fronts: vec![
                FrontInfo {
                    fs_start: 0,
                    fs_end: k,
                    parent: Some(1),
                    children: vec![],
                    level: 0,
                },
                FrontInfo {
                    fs_start: k,
                    fs_end: n,
                    parent: None,
                    children: vec![0],
                    level: 1,
                },
            ],
            levels: vec![vec![0], vec![1]],
        }
    }

    #[test]
    fn delayed_pivots_absorbed_by_ancestor_front() {
        let k = 4;
        let n = 2 * k;
        let m = matching_matrix(k, false);
        let tree = two_front_tree(k);
        let a_lower = lower_triangle_csr(&m, true);
        let fronts = numeric_factorise(&a_lower, &tree, 0, 1.0, true, None)
            .expect("factorise with delayed pivoting");

        assert_eq!(fronts[0].nfs, 0, "leaf must postpone every pivot");
        assert_eq!(fronts[0].ndelayed, k);
        assert_eq!(fronts[1].ndelayed, 0, "root must absorb the delayed set");

        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let x = forward_backward_solve(&tree, &fronts, &b, true).unwrap();
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "delayed-pivot residual {res:e} too large");
    }

    #[test]
    fn singular_root_reports_error() {
        // Drop the v_0—u_0 coupling: row/column v_0 is entirely zero, the
        // matrix is genuinely singular, and the root cannot pivot on it.
        let k = 4;
        let m = matching_matrix(k, true);
        let tree = two_front_tree(k);
        let a_lower = lower_triangle_csr(&m, true);
        match numeric_factorise(&a_lower, &tree, 0, 1.0, true, None) {
            Err(BlrError::SingularFront(front)) => {
                assert_eq!(front, 1, "failure must be reported at the root front")
            }
            Err(error) => panic!("unexpected factorization error: {error}"),
            Ok(_) => panic!("singular matrix must fail at the root front"),
        }
    }

    /// 2D grid with complex Hermitian couplings, shifted by σ to be
    /// indefinite (σ inside the spectral range) or left at 0 (HPD).
    fn hermitian_grid(nx: usize, sigma: f64) -> CsrMatrix<Complex64> {
        let n = nx * nx;
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        let h = Complex64::new(1.0, 0.3);
        let v = Complex64::new(0.8, -0.2);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = Complex64::new(4.0 - sigma, 0.0);
                if ix > 0 {
                    dense[[i, i - 1]] = -h;
                    dense[[i - 1, i]] = -h.conj();
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -v;
                    dense[[i - nx, i]] = -v.conj();
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn assert_close_complex(a: &[Complex64], b: &[Complex64], tol: f64, label: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((*x - *y).norm() < tol, "{label} diverge at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn hermitian_indefinite_complex_matches_lu_path() {
        let m = hermitian_grid(6, 3.0);
        let n = m.num_rows;
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 0.5).sin(), (i as f64).cos()))
            .collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 8,
                hermitian: true,
                ..LdltConfig::default()
            },
        )
        .expect("LDLH factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "LDLH residual {res:e} too large");

        let lu = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                min_block_size: 8,
                ..BlrConfig::default()
            },
        )
        .expect("LU factorise");
        assert_close_complex(&x, &lu.solve(&b), 1e-9, "LDLH vs multifrontal LU");
    }

    #[test]
    fn hpd_complex_solves() {
        let m = hermitian_grid(6, 0.0);
        let n = m.num_rows;
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 0.2).cos(), (i as f64 + 1.0).sin()))
            .collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 8,
                hermitian: true,
                ..LdltConfig::default()
            },
        )
        .expect("LDLH factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "HPD LDLH residual {res:e} too large");
    }

    #[test]
    fn complex_symmetric_ldlt_no_conjugation() {
        // Complex-*symmetric* (A = Aᵀ, not Aᴴ): tiny imaginary diagonals and
        // symmetric complex couplings — needs 2×2 pivots and the non-
        // conjugating transpose (hermitian = false).
        let nx = 5;
        let n = nx * nx;
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        let coupling = Complex64::new(1.0, 0.4);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = Complex64::new(0.05, 0.1);
                if ix > 0 {
                    dense[[i, i - 1]] = coupling;
                    dense[[i - 1, i]] = coupling;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = coupling;
                    dense[[i - nx, i]] = coupling;
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 0.7).sin(), (i as f64 + 0.1).cos()))
            .collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 6,
                hermitian: false,
                ..LdltConfig::default()
            },
        )
        .expect("LDLT factorise");
        assert!(
            ldlt.two_by_two_pivot_count() > 0,
            "tiny-diagonal complex-symmetric matrix must produce 2×2 pivots"
        );
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-10, "complex-symmetric residual {res:e} too large");

        let lu = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                min_block_size: 6,
                ..BlrConfig::default()
            },
        )
        .expect("LU factorise");
        assert_close_complex(&x, &lu.solve(&b), 1e-8, "complex-symmetric vs LU");
    }

    #[test]
    fn upper_triangle_only_input_solves() {
        // Storage-convention robustness: only the upper triangle stored.
        let full = shifted_laplacian_2d(8, 3.0);
        let n = full.num_rows;
        let mut triplets = Vec::new();
        for i in 0..n {
            for idx in full.row_ptrs[i]..full.row_ptrs[i + 1] {
                let j = full.col_indices[idx];
                if j >= i {
                    triplets.push((i, j, full.values[idx]));
                }
            }
        }
        let m = CsrMatrix::from_triplets(n, n, triplets);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.9).sin()).collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 8,
                ..LdltConfig::default()
            },
        )
        .expect("LDLT factorise");
        let x = ldlt.solve(&b);
        // Residual against the *full* matrix.
        let res = rel_residual(&full, &x, &b);
        assert!(res < 1e-10, "upper-only residual {res:e} too large");
    }

    #[test]
    fn parallel_front_threshold_matches_serial() {
        let m = shifted_laplacian_3d(5, 5.5);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.31).sin()).collect();

        let mk = |threshold| {
            MultifrontalLdltFactorization::factorize_with_config(
                &m,
                &LdltConfig {
                    min_block_size: 16,
                    amalgamation_threshold: 0,
                    parallel_front_threshold: threshold,
                    ..LdltConfig::default()
                },
            )
            .expect("factorise")
        };
        let serial = mk(usize::MAX);
        let parallel = mk(16);

        let x_serial = serial.solve(&b);
        let x_parallel = parallel.solve(&b);
        for i in 0..n {
            assert!(
                (x_serial[i] - x_parallel[i]).abs() < 1e-13,
                "intra-front parallelism diverges at {i}: {} vs {}",
                x_serial[i],
                x_parallel[i]
            );
        }
        let res_s = rel_residual(&m, &x_serial, &b);
        let res_p = rel_residual(&m, &x_parallel, &b);
        assert!(res_s < 1e-10 && res_p < 1e-10);
    }

    fn blr_test_config(epsilon: f64, block_size: usize) -> LdltConfig {
        LdltConfig {
            min_block_size: 16,
            amalgamation_threshold: 64,
            blr: Some(LdltBlrConfig {
                epsilon,
                block_size,
                panel_kind: PanelKind::Blr,
            }),
            ..LdltConfig::default()
        }
    }

    #[test]
    fn blr_epsilon_zero_reproduces_dense_panels() {
        // With compression enabled but ε = 0 the panels are merely
        // block-partitioned (full numerical rank), so the solve must
        // reproduce the dense-panel factorisation bit-near.
        let m = shifted_laplacian_2d(12, 3.5);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();

        let dense = MultifrontalLdltFactorization::factorize_with_config(&m, &test_config())
            .expect("dense-panel LDLT factorise");
        let blr =
            MultifrontalLdltFactorization::factorize_with_config(&m, &blr_test_config(0.0, 16))
                .expect("BLR ε=0 LDLT factorise");

        let x_dense = dense.solve(&b);
        let x_blr = blr.solve(&b);
        for i in 0..n {
            assert!(
                (x_blr[i] - x_dense[i]).abs() < 1e-10,
                "ε=0 BLR diverges from dense panels at {i}: {} vs {}",
                x_blr[i],
                x_dense[i]
            );
        }
        let res = rel_residual(&m, &x_blr, &b);
        assert!(res < 1e-10, "ε=0 BLR residual {res:e} too large");
    }

    #[test]
    fn blr_compression_shrinks_panels_on_2d_poisson() {
        // SPD 2D Poisson. The solver applies its own nested-dissection
        // ordering (the good ordering BLR needs — there is no external
        // AMD/COLAMD hook in the multifrontal path). At ε = 1e-8 the
        // off-diagonal panels of the large fronts are numerically
        // low-rank: the solve must stay within ~1e-6 relative while the
        // compressed-panel factor storage drops well below the
        // dense-panel storage.
        let m = shifted_laplacian_2d(64, 0.0);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.25).cos()).collect();

        // Amalgamate up to 256-DOF FS blocks: the large fronts this
        // produces have wide L21 / CB panels whose off-diagonal blocks are
        // genuinely low-rank.
        let mk = |blr: Option<LdltBlrConfig>| {
            MultifrontalLdltFactorization::factorize_with_config(
                &m,
                &LdltConfig {
                    min_block_size: 16,
                    blr,
                    ..LdltConfig::default()
                },
            )
            .expect("LDLT factorise")
        };
        let dense = mk(None);
        let blr = mk(Some(LdltBlrConfig {
            epsilon: 1e-8,
            block_size: 64,
            panel_kind: PanelKind::Blr,
        }));

        let x = blr.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-6, "BLR residual {res:e} too large (ε = 1e-8)");

        // Compare off-diagonal panel storage only (the packed F11 factors
        // are dense in both configurations and dominate the total).
        let panel_entries = |fac: &MultifrontalLdltFactorization<f64>| -> usize {
            fac.fronts
                .iter()
                .map(|f| f.l21.stored_entries() + f.cb_panel.stored_entries())
                .sum()
        };
        let dense_panels = panel_entries(&dense) * std::mem::size_of::<f64>();
        let blr_panels = panel_entries(&blr) * std::mem::size_of::<f64>();
        let dense_bytes = dense.factor_memory_bytes();
        let blr_bytes = blr.factor_memory_bytes();
        eprintln!(
            "BLR panel storage: dense panels = {dense_panels} B, compressed panels = \
             {blr_panels} B, panel ratio = {:.3}; total factor: dense = {dense_bytes} B, \
             compressed = {blr_bytes} B, ratio = {:.3}",
            blr_panels as f64 / dense_panels as f64,
            blr_bytes as f64 / dense_bytes as f64
        );
        assert!(
            blr_panels * 4 < dense_panels * 3,
            "compressed panels ({blr_panels}) should use ≤ ~75% of dense panels \
             ({dense_panels})"
        );
    }

    #[test]
    fn blr_hermitian_complex_solves() {
        // Complex Hermitian indefinite grid with BLR panels: exercises the
        // conjugate-sandwich transpose apply of the compressed L21 in the
        // backward substitution.
        let m = hermitian_grid(8, 3.0);
        let n = m.num_rows;
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 0.5).sin(), (i as f64).cos()))
            .collect();

        let ldlt = MultifrontalLdltFactorization::factorize_with_config(
            &m,
            &LdltConfig {
                min_block_size: 8,
                hermitian: true,
                blr: Some(LdltBlrConfig {
                    epsilon: 1e-8,
                    block_size: 8,
                    panel_kind: PanelKind::Blr,
                }),
                ..LdltConfig::default()
            },
        )
        .expect("BLR LDLH factorise");
        let x = ldlt.solve(&b);
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-6, "BLR LDLH residual {res:e} too large");
    }

    #[test]
    fn inertia_counts_diagonal_signs_across_fronts() {
        let matrix = CsrMatrix::from_triplets(
            5,
            5,
            vec![
                (0, 0, -7.0),
                (1, 1, 2.0),
                (2, 2, -3.0),
                (3, 3, 11.0),
                (4, 4, 5.0),
            ],
        );
        let factor = MultifrontalLdltFactorization::factorize_with_config(
            &matrix,
            &LdltConfig {
                min_block_size: 1,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            factor.inertia(),
            MatrixInertia {
                positive: 3,
                negative: 2,
                zero: 0,
            }
        );
    }

    #[test]
    fn inertia_counts_indefinite_two_by_two_pivot() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 1, 2.0),
                (1, 0, 2.0),
                (1, 2, 0.25),
                (2, 1, 0.25),
                (2, 2, 4.0),
            ],
        );
        let factor = MultifrontalLdltFactorization::factorize(&matrix).unwrap();

        assert!(factor.two_by_two_pivot_count() >= 1);
        assert_eq!(
            factor.inertia(),
            MatrixInertia {
                positive: 2,
                negative: 1,
                zero: 0,
            }
        );
    }

    #[test]
    fn inertia_of_empty_factorization_is_empty() {
        let matrix = CsrMatrix::<f64>::from_triplets(0, 0, Vec::new());
        let factor = MultifrontalLdltFactorization::factorize(&matrix).unwrap();
        assert_eq!(
            factor.inertia(),
            MatrixInertia {
                positive: 0,
                negative: 0,
                zero: 0,
            }
        );
    }

    #[test]
    fn multifrontal_ldlt_rejects_malformed_csr_before_factorization() {
        let mut matrix =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]);
        matrix.col_indices[0] = 2;
        assert!(MultifrontalLdltFactorization::factorize(&matrix).is_err());
    }

    #[test]
    fn malformed_child_assembly_is_reported_without_panicking() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let child = FactoredLdltFront {
            nfs: 0,
            ncb: 1,
            fs_elim_indices: Vec::new(),
            f11_ldl: Vec::new(),
            pivot_kinds: Vec::new(),
            pivots: Vec::new(),
            l21: L21Panel::Dense(DenseMatrix::zeros(1, 0)),
            cb_indices: vec![1],
            cb_panel: CbPanel::PackedLower(vec![0.0]),
            ndelayed: 0,
            delayed_row_indices: Vec::new(),
            delayed: None,
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assemble_front(&matrix, 0, 1, Vec::new(), &[&child], true, 9)
        }))
        .expect("malformed assembly must not panic");
        assert!(matches!(
            result,
            Err(BlrError::InvalidFactorization { front: 9, .. })
        ));
    }

    #[test]
    fn malformed_delayed_blocks_are_rejected_before_indexing() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let child = FactoredLdltFront {
            nfs: 0,
            ncb: 0,
            fs_elim_indices: vec![0],
            f11_ldl: Vec::new(),
            pivot_kinds: Vec::new(),
            pivots: Vec::new(),
            l21: L21Panel::Dense(DenseMatrix::zeros(0, 0)),
            cb_indices: Vec::new(),
            cb_panel: CbPanel::PackedLower(Vec::new()),
            ndelayed: 1,
            delayed_row_indices: vec![0],
            delayed: Some(DelayedSymBlocks {
                dd_lower: Vec::new(),
                dc: DenseMatrix::zeros(1, 0),
            }),
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assemble_front(&matrix, 1, 2, Vec::new(), &[&child], true, 11)
        }))
        .expect("malformed delayed assembly must not panic");
        assert!(matches!(
            result,
            Err(BlrError::InvalidFactorization { front: 11, .. })
        ));
    }
}
