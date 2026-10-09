//! Frontal-matrix data structures and per-front numeric factorisation.
//!
//! A front sits between a node's fully-summed (FS) block and its
//! contribution block (CB). The FS block is dense and gets factorised with
//! either partial pivoting (row swaps only) or rook pivoting (alternating
//! row and column swaps within the FS block). The off-diagonal panels
//! (L21, U12) and the Schur complement that becomes the parent's
//! contribution are BLR-compressed before being stored.
//!
//! Pivot rejection is *scale-aware*: a pivot candidate below
//! `pivot_tol_rel · ε_machine · nfs · max|F11|` is treated as zero,
//! matching the crate's sparse-LU `ε·n·max|A|` convention. Rejection is
//! not fatal: elimination stops at that point and the surviving FS rows /
//! columns are handed to the parent front as dense [`DelayedBlocks`]
//! (MUMPS-style delayed pivoting), where pivoting is retried against a
//! larger fully-summed set. Only a root front that still cannot pivot
//! makes the factorisation fail.
//!
//! Index conventions inside a front: rows / cols indexed `0..size` where
//! `size = nfs + ncb`, with the FS block at `[0..nfs, 0..nfs]` and the CB
//! at `[nfs..size, nfs..size]`. Rook column pivots permute columns *within*
//! `[0..nfs, 0..nfs]` only — they never touch the CB block.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;

use super::blr::LowRankPrecision;
use super::panel::{Panel, PanelKind};
use super::{FrontError, first_non_finite, scalar_is_finite};
use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::prelude::*;

/// Within-front update order for the Schur step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateOrder {
    /// Right-look (default, MUMPS-BLR-style): compute dense L21 and U12
    /// after the F11 LU factorisation, then form the Schur complement as a
    /// dense `F22 - L21 · U12` subtraction and BLR-compress the result.
    #[default]
    RightLook,
    /// Left-look BLR: compute dense L21/U12, compress them to BLR
    /// immediately, then form the Schur as a BLR×BLR matmul into a dense
    /// F22 accumulator. Cheaper when L21/U12 contain many low-rank blocks
    /// (rank-aware contraction) at the cost of an extra compression pass.
    LeftLook,
}

/// Pivoting strategy for the FS dense block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotStrategy {
    /// Classical partial pivoting: at each step, pick the row in column `k`
    /// (rows ≥ k) with the largest magnitude as pivot. Row swaps only.
    Partial,
    /// Rook pivoting: alternate between picking the largest in the current
    /// column and the largest in the current row until a (row, col) is the
    /// max of both. More robust on indefinite / near-singular blocks at the
    /// cost of extra search work. Row + column swaps within FS.
    Rook,
    /// Bunch-Kaufman-style pivoting. The current BLR front implementation
    /// stores only 1×1 pivots, so this selects the same robust rook pivot as
    /// the 1×1 fallback of a BK scheme.
    BunchKaufman,
}

/// A front, fully assembled but not yet factorised.
pub struct AssembledFront<T: ComplexField> {
    /// Size of the fully-summed block (own FS plus any delayed pivots
    /// absorbed from children).
    pub nfs: usize,
    /// Size of the contribution block.
    pub ncb: usize,
    /// Dense storage of the entire front `(nfs + ncb) × (nfs + ncb)`.
    pub matrix: DenseMatrix<T>,
    /// Elimination-order index of every assembled FS position (length
    /// `nfs`): own FS positions map to `fs_start + p`, positions beyond
    /// the own FS range are delayed pivots absorbed from children.
    pub fs_elim_indices: Vec<usize>,
    /// Elimination-order indices of the CB rows/cols (length `ncb`).
    pub cb_indices: Vec<usize>,
}

/// Per-step pivot block descriptor.
///
/// For `PivotStrategy::Partial` and `Rook`, every step is `OneByOne`. For
/// `BunchKaufman`, a step can mark the start of a 2×2 block pivot — in that
/// case position `k` carries `TwoByTwoStart` and position `k+1` carries
/// `TwoByTwoCont`. The factorised storage at `f11_lu[[k, k]]`,
/// `f11_lu[[k, k+1]]`, `f11_lu[[k+1, k]]`, `f11_lu[[k+1, k+1]]` holds the
/// original 2×2 block `D` (so the solve can apply `D^{-1}` per block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotKind {
    OneByOne,
    TwoByTwoStart,
    TwoByTwoCont,
}

/// Dense delayed-pivot payload handed from a front to its parent.
///
/// All three blocks already carry the full Schur update of the pivots that
/// *were* eliminated in the child, i.e. they are the restriction of
/// `F[S, S] - F[S, E] · F[E, E]⁻¹ · F[E, S]` (E = eliminated, S = delayed
/// ∪ CB) to the blocks below, so the parent can extend-add them verbatim
/// into its enlarged fully-summed block. They are always stored dense —
/// MUMPS-style delayed rows/columns simply stay dense inside the parent's
/// F11 and are never BLR-compressed.
pub struct DelayedBlocks<T: ComplexField> {
    /// Trailing `F[D, D]` block, shape `(ndelayed × ndelayed)`; rows in
    /// `delayed_row_indices` order, columns in `delayed_col_indices` order.
    pub dd: DenseMatrix<T>,
    /// Updated `F[D, C]` block, shape `(ndelayed × ncb)`; columns in
    /// `cb_indices` order.
    pub dc: DenseMatrix<T>,
    /// Updated `F[C, D]` block, shape `(ncb × ndelayed)`; rows in
    /// `cb_indices` order, columns in `delayed_col_indices` order.
    pub cd: DenseMatrix<T>,
}

/// Factorised front: stores everything needed for the forward + back solve
/// and for the parent's extend-add of the CB.
pub struct FactoredFront<T: ComplexField> {
    /// Number of FS pivots actually eliminated in this front. When delayed
    /// pivoting postpones part of the assembled FS block this is smaller
    /// than `fs_elim_indices.len()`; the postponed rows/columns are
    /// described by the `delayed_*` fields and retried at the parent.
    pub nfs: usize,
    pub ncb: usize,
    /// Elimination-order index of every *assembled* FS position, indexed by
    /// front-local FS position (length = assembled FS size ≥ `nfs`).
    /// Needed because delayed pivots absorbed from children are appended
    /// after the own FS range, breaking the
    /// "position ↔ fs_start + position" identity.
    pub fs_elim_indices: Vec<usize>,
    /// LU factors of the (nfs × nfs) eliminated FS block. Convention:
    /// - 1×1 pivot at k: `f11_lu[k, k]` = U[k, k]; below-diagonal column k
    ///   holds L[k+1:, k]; above-diagonal row k holds U[k, k+1:].
    /// - 2×2 pivot starting at k: `f11_lu[[k:k+2, k:k+2]]` holds the
    ///   original 2×2 block `D`; below-block column pair holds the
    ///   L-multipliers; above-block row pair holds U.
    pub f11_lu: DenseMatrix<T>,
    /// Per-step pivot descriptor (length `nfs`).
    pub pivot_kinds: Vec<PivotKind>,
    /// Row-pivot map (length `nfs`): assembled-FS position of the row
    /// eliminated at each step, in factored order. The solve gathers
    /// `b_FS[pivots[i]]` before [`Self::forward_apply`].
    pub pivots: Vec<usize>,
    /// Column-pivot map (length `nfs`), same convention for the eliminated
    /// columns. Identity under partial pivoting; may permute under rook /
    /// Bunch–Kaufman pivoting.
    pub col_pivots: Vec<usize>,
    /// `L21 = F[nfs..size, 0..nfs] · P_c · U^{-1}` (block-aware), BLR.
    pub l21: Panel<T>,
    /// `U12 = L^{-1} · P_r · F[0..nfs, nfs..size]` (block-aware), BLR.
    pub u12: Panel<T>,
    /// Elimination-order indices of CB rows/cols. Delayed indices stay in
    /// this set (the parent absorbs them into its enlarged FS set), so the
    /// CB panel keeps covering the full Schur complement.
    pub cb_indices: Vec<usize>,
    /// Schur complement of the front, compressed. The parent extend-adds
    /// this into its own dense storage.
    pub cb_panel: Panel<T>,
    /// Number of pivots postponed to the parent front (0 in the common
    /// case). Non-zero only on fronts whose FS block had no acceptable
    /// pivot for the trailing rows/columns.
    pub ndelayed: usize,
    /// Elimination-order indices of the delayed rows (length `ndelayed`,
    /// in factored order). Subset of this front's assembled FS range.
    pub delayed_row_indices: Vec<usize>,
    /// Elimination-order indices of the delayed columns. Equal to
    /// `delayed_row_indices` under partial pivoting; may differ under
    /// rook / Bunch–Kaufman column pivoting.
    pub delayed_col_indices: Vec<usize>,
    /// Dense updated trailing blocks involving the delayed indices;
    /// `None` iff `ndelayed == 0`. Consumed by the parent's extend-add.
    pub delayed: Option<DelayedBlocks<T>>,
}

/// Find a rook-pivot for step `k`: alternate row-max and col-max searches
/// over `[k..nfs, k..nfs]` until a (row, col) is the magnitude maximum in
/// both its column slice and its row slice. Returns `(row, col)` as global
/// matrix indices, or `None` if the magnitude stays below `pivot_tol`.
fn find_rook_pivot<T: ComplexField>(
    matrix: &DenseMatrix<T>,
    k: usize,
    nfs: usize,
    pivot_tol: T::Real,
) -> Option<(usize, usize)> {
    // Initial column is k. Find the row in column k (rows ≥ k) with largest |.|.
    let mut col = k;
    let mut best_row = k;
    let mut best_val = matrix[[k, k]].norm();
    for i in (k + 1)..nfs {
        let v = matrix[[i, col]].norm();
        if v > best_val {
            best_val = v;
            best_row = i;
        }
    }
    if best_val < pivot_tol {
        return None;
    }

    // Alternate: find the col in row best_row (cols ≥ k) with the largest
    // magnitude; if it changes the column, find the row in that new column
    // again; repeat until stable. Capped at a few iterations — in practice
    // rook converges in ≤ 3 swaps but we guard against degenerate cases.
    for _ in 0..8 {
        let row = best_row;
        let mut new_col = col;
        let mut row_max = matrix[[row, col]].norm();
        for j in (k + 1)..nfs {
            let v = matrix[[row, j]].norm();
            if v > row_max {
                row_max = v;
                new_col = j;
            }
        }
        if new_col == col {
            break;
        }
        col = new_col;

        // Find row max in this new column.
        let mut new_row = best_row;
        let mut col_max = matrix[[best_row, col]].norm();
        for i in k..nfs {
            let v = matrix[[i, col]].norm();
            if v > col_max {
                col_max = v;
                new_row = i;
            }
        }
        if new_row == best_row {
            break;
        }
        best_row = new_row;
    }

    Some((best_row, col))
}

/// Outcome of the Bunch-Kaufman pivot-selection routine.
enum BkPivot {
    OneByOne {
        row: usize,
        col: usize,
    },
    /// 2×2 block at positions [`k`, `k+1`] after applying the suggested
    /// row/col swaps. `swap_row1`, `swap_col1`, `swap_row2`, `swap_col2`
    /// describe the swaps needed: `(k ↔ swap_row1, k ↔ swap_col1)` then
    /// `(k+1 ↔ swap_row2, k+1 ↔ swap_col2)`.
    TwoByTwo {
        swap_row1: usize,
        swap_col1: usize,
        swap_row2: usize,
        swap_col2: usize,
    },
}

/// Bunch-Kaufman 1×1 / 2×2 pivot selection for an LU step.
///
/// Implements the standard BK partial-pivoting criterion on column `k` of
/// the active submatrix `[k..nfs, k..nfs]`:
///
/// * `λ = max_{i>k} |A[i, k]|` with argmax row `r`.
/// * If `|A[k,k]| ≥ α·λ`, take the 1×1 pivot at `(k, k)`.
/// * Else let `σ = max_{i ∈ [k..nfs], i≠r} |A[i, r]|`.
///   - If `|A[k,k]|·σ ≥ α·λ²`, the 1×1 pivot at `(k, k)` is still safe.
///   - Else if `|A[r,r]| ≥ α·σ`, swap `k ↔ r` and take a 1×1 pivot.
///   - Else emit a 2×2 block on rows/columns `{k, r}`.
///
/// The matrix is general (we are doing LU, not LDLᵀ); this is the
/// symmetric-style BK criterion applied as a heuristic — it does not
/// guarantee bounded growth on non-symmetric matrices but it does
/// actually exercise the 2×2 path on indefinite / zero-diagonal blocks.
fn find_bk_pivot<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    k: usize,
    nfs: usize,
    pivot_tol: T::Real,
) -> Option<BkPivot> {
    // Need at least 2 rows/cols left for a 2×2 candidate; otherwise fall back.
    if k + 1 >= nfs {
        let (r, c) = find_rook_pivot(matrix, k, nfs, pivot_tol)?;
        return Some(BkPivot::OneByOne { row: r, col: c });
    }

    let alpha = T::Real::from_f64((1.0 + 17.0_f64.sqrt()) / 8.0).unwrap();

    // λ = largest magnitude in column k, strictly below the diagonal.
    let mut lambda = T::Real::zero();
    let mut r_row = k + 1;
    for i in (k + 1)..nfs {
        let v = matrix[[i, k]].norm();
        if v > lambda {
            lambda = v;
            r_row = i;
        }
    }
    let diag_k = matrix[[k, k]].norm();

    // Whole column k vanishes (including the diagonal) ⇒ singular front.
    if diag_k < pivot_tol && lambda < pivot_tol {
        return None;
    }

    // Test 1: diagonal already dominates column ⇒ 1×1 at (k, k).
    if diag_k >= alpha * lambda {
        return Some(BkPivot::OneByOne { row: k, col: k });
    }

    // σ = strongest entry in column r outside row r (rows in [k..nfs]).
    let mut sigma = T::Real::zero();
    for i in k..nfs {
        if i == r_row {
            continue;
        }
        let v = matrix[[i, r_row]].norm();
        if v > sigma {
            sigma = v;
        }
    }

    // Test 2: |A[k,k]|·σ ≥ α·λ² ⇒ 1×1 at (k, k) is still safe.
    if diag_k * sigma >= alpha * lambda * lambda {
        return Some(BkPivot::OneByOne { row: k, col: k });
    }

    // Test 3: row r has a strong diagonal ⇒ swap k ↔ r, take 1×1.
    let diag_r = matrix[[r_row, r_row]].norm();
    if diag_r >= alpha * sigma {
        return Some(BkPivot::OneByOne {
            row: r_row,
            col: r_row,
        });
    }

    // Otherwise: 2×2 block on rows/cols {k, r}. `r_row > k` is guaranteed
    // (the λ-search runs over `i > k`), so this never produces overlapping
    // swap targets.
    debug_assert!(r_row > k);
    Some(BkPivot::TwoByTwo {
        swap_row1: k,
        swap_col1: k,
        swap_row2: r_row,
        swap_col2: r_row,
    })
}

/// Apply a row swap k ↔ r across the entire front (all `size` columns).
fn swap_rows_full<T: ComplexField>(matrix: &mut DenseMatrix<T>, size: usize, k: usize, r: usize) {
    if k == r {
        return;
    }
    for j in 0..size {
        let tmp = matrix[[k, j]];
        matrix[[k, j]] = matrix[[r, j]];
        matrix[[r, j]] = tmp;
    }
}

/// Apply a column swap k ↔ c across the entire front (all `size` rows).
fn swap_cols_full<T: ComplexField>(matrix: &mut DenseMatrix<T>, size: usize, k: usize, c: usize) {
    if k == c {
        return;
    }
    for i in 0..size {
        let tmp = matrix[[i, k]];
        matrix[[i, k]] = matrix[[i, c]];
        matrix[[i, c]] = tmp;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockInverseError {
    Singular,
    NonFinite,
}

/// Invert a 2×2 block with an explicit singular/non-finite distinction.
pub(crate) fn invert_2x2<T: ComplexField + FromPrimitive>(
    a: T,
    b: T,
    c: T,
    d: T,
    pivot_tol: T::Real,
) -> Result<[[T; 2]; 2], BlockInverseError> {
    if !scalar_is_finite(a)
        || !scalar_is_finite(b)
        || !scalar_is_finite(c)
        || !scalar_is_finite(d)
        || !pivot_tol.is_finite()
    {
        return Err(BlockInverseError::NonFinite);
    }
    let det = a * d - b * c;
    if !scalar_is_finite(det) {
        return Err(BlockInverseError::NonFinite);
    }
    let det_magnitude = det.re().abs().max(det.im().abs());
    if det_magnitude <= pivot_tol {
        return Err(BlockInverseError::Singular);
    }
    let inv_det = det.inv();
    let inverse = [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]];
    if inverse.iter().flatten().copied().all(scalar_is_finite) {
        Ok(inverse)
    } else {
        Err(BlockInverseError::NonFinite)
    }
}

#[allow(clippy::too_many_arguments)]
fn select_pivot<T: ComplexField + FromPrimitive>(
    matrix: &DenseMatrix<T>,
    k: usize,
    nfs: usize,
    pivot_tol: T::Real,
    pivot_strategy: PivotStrategy,
) -> Option<BkPivot> {
    match pivot_strategy {
        PivotStrategy::Partial => {
            let mut max_val = matrix[[k, k]].norm();
            let mut max_row = k;
            for i in (k + 1)..nfs {
                let v = matrix[[i, k]].norm();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val < pivot_tol {
                return None;
            }
            Some(BkPivot::OneByOne {
                row: max_row,
                col: k,
            })
        }
        PivotStrategy::Rook => {
            let (r, c) = find_rook_pivot(matrix, k, nfs, pivot_tol)?;
            Some(BkPivot::OneByOne { row: r, col: c })
        }
        PivotStrategy::BunchKaufman => find_bk_pivot(matrix, k, nfs, pivot_tol),
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_one_by_one_pivot<T: ComplexField + FromPrimitive>(
    matrix: &mut DenseMatrix<T>,
    nfs: usize,
    size: usize,
    pivots: &mut [usize],
    col_pivots: &mut [usize],
    pivot_kinds: &mut [PivotKind],
    k: usize,
    pivot_row: usize,
    pivot_col: usize,
) {
    if pivot_row != k {
        swap_rows_full(matrix, size, k, pivot_row);
        pivots.swap(k, pivot_row);
    }
    if pivot_col != k {
        swap_cols_full(matrix, size, k, pivot_col);
        col_pivots.swap(k, pivot_col);
    }

    let pivot = matrix[[k, k]];
    let pivot_inv = pivot.inv();
    // NOTE (vec_axpy rank-1 attempt, reverted 2026-09-25): routing rows
    // with len >= 32 through T::vec_axpy (cblas1::axpy_f64) measured ~-7%
    // on the pivot bucket (medians 61 -> 54ms) but the residual moved
    // 4.56e-14 -> 4.53e-14: the SIMD kernel is NOT bit-identical to the
    // scalar loop (FMA fusion: one rounding vs two). Same-class gain
    // inside tenant noise is a bad trade against bit-identity — scalar
    // loop stands.
    for i in (k + 1)..nfs {
        let l_ik = matrix[[i, k]] * pivot_inv;
        matrix[[i, k]] = l_ik;
        for j in (k + 1)..nfs {
            let u_kj = matrix[[k, j]];
            let a = matrix[[i, j]];
            matrix[[i, j]] = a - l_ik * u_kj;
        }
    }
    pivot_kinds[k] = PivotKind::OneByOne;
}

#[allow(clippy::too_many_arguments)]
fn apply_two_by_two_pivot<T: ComplexField + FromPrimitive>(
    matrix: &mut DenseMatrix<T>,
    nfs: usize,
    size: usize,
    pivots: &mut [usize],
    col_pivots: &mut [usize],
    pivot_kinds: &mut [PivotKind],
    k: usize,
    swap_row1: usize,
    swap_col1: usize,
    swap_row2: usize,
    swap_col2: usize,
    pivot_tol: T::Real,
) -> Result<Option<usize>, FrontError> {
    // Bring (swap_row1, swap_col1) to (k, k).
    if swap_row1 != k {
        swap_rows_full(matrix, size, k, swap_row1);
        pivots.swap(k, swap_row1);
    }
    if swap_col1 != k {
        swap_cols_full(matrix, size, k, swap_col1);
        col_pivots.swap(k, swap_col1);
    }
    // Then bring (swap_row2, swap_col2) to (k+1, k+1).
    if swap_row2 != k + 1 {
        swap_rows_full(matrix, size, k + 1, swap_row2);
        pivots.swap(k + 1, swap_row2);
    }
    if swap_col2 != k + 1 {
        swap_cols_full(matrix, size, k + 1, swap_col2);
        col_pivots.swap(k + 1, swap_col2);
    }

    let a = matrix[[k, k]];
    let b = matrix[[k, k + 1]];
    let c = matrix[[k + 1, k]];
    let d = matrix[[k + 1, k + 1]];
    let d_inv = match invert_2x2::<T>(a, b, c, d, pivot_tol) {
        Ok(inverse) => inverse,
        Err(BlockInverseError::Singular) => {
            // 2×2 too singular — fall back to 1×1 on a.
            if a.norm() < pivot_tol {
                return Ok(None);
            }
            let pivot_inv = a.inv();
            for i in (k + 1)..nfs {
                let l_ik = matrix[[i, k]] * pivot_inv;
                matrix[[i, k]] = l_ik;
                for j in (k + 1)..nfs {
                    let u_kj = matrix[[k, j]];
                    let av = matrix[[i, j]];
                    matrix[[i, j]] = av - l_ik * u_kj;
                }
            }
            pivot_kinds[k] = PivotKind::OneByOne;
            return Ok(Some(1));
        }
        Err(BlockInverseError::NonFinite) => {
            return Err(FrontError::numerical("2x2 pivot inversion", k));
        }
    };

    // Compute L_{i, [k, k+1]} = F[i, [k, k+1]] · D^{-1} for i ≥ k+2.
    // Then eliminate F[i, j ≥ k+2] -= L_{i, [k,k+1]} · F[[k,k+1], j].
    for i in (k + 2)..nfs {
        let f_i0 = matrix[[i, k]];
        let f_i1 = matrix[[i, k + 1]];
        let l_i0 = f_i0 * d_inv[0][0] + f_i1 * d_inv[1][0];
        let l_i1 = f_i0 * d_inv[0][1] + f_i1 * d_inv[1][1];
        matrix[[i, k]] = l_i0;
        matrix[[i, k + 1]] = l_i1;
        for j in (k + 2)..nfs {
            let u_kj = matrix[[k, j]];
            let u_k1j = matrix[[k + 1, j]];
            let av = matrix[[i, j]];
            matrix[[i, j]] = av - l_i0 * u_kj - l_i1 * u_k1j;
        }
    }
    pivot_kinds[k] = PivotKind::TwoByTwoStart;
    pivot_kinds[k + 1] = PivotKind::TwoByTwoCont;
    Ok(Some(2))
}

/// Pivot outcome of [`factor_frontal_block`]: row pivot map, column pivot
/// map, per-step pivot descriptors, and `k_elim` (positions `k_elim..nfs`
/// are the delayed rows/columns handed to the parent front).
type FrontalPivotOutcome = (Vec<usize>, Vec<usize>, Vec<PivotKind>, usize);

/// Dense LU factorisation of the FS block with the chosen pivot strategy.
///
/// Returns the pivot maps (full-length permutations of the assembled FS
/// positions), the per-step pivot descriptors, and `k_elim`, the number of
/// pivots actually eliminated. When no acceptable pivot exists at step
/// `k`, elimination stops instead of failing: positions `k_elim..nfs` are
/// the delayed rows/columns handed to the parent front (MUMPS-style
/// delayed pivoting).
#[allow(clippy::too_many_arguments)]
fn factor_frontal_block<T: ComplexField + FromPrimitive>(
    matrix: &mut DenseMatrix<T>,
    nfs: usize,
    size: usize,
    pivot_tol: T::Real,
    pivot_strategy: PivotStrategy,
) -> Result<FrontalPivotOutcome, FrontError> {
    let mut pivots: Vec<usize> = (0..nfs).collect();
    let mut col_pivots: Vec<usize> = (0..nfs).collect();
    let mut pivot_kinds: Vec<PivotKind> = vec![PivotKind::OneByOne; nfs];

    let mut k = 0usize;
    while k < nfs {
        let Some(bk_choice) = select_pivot(matrix, k, nfs, pivot_tol, pivot_strategy) else {
            break;
        };
        match bk_choice {
            BkPivot::OneByOne { row, col } => {
                apply_one_by_one_pivot(
                    matrix,
                    nfs,
                    size,
                    &mut pivots,
                    &mut col_pivots,
                    &mut pivot_kinds,
                    k,
                    row,
                    col,
                );
                k += 1;
            }
            BkPivot::TwoByTwo {
                swap_row1,
                swap_col1,
                swap_row2,
                swap_col2,
            } => {
                let Some(advance) = apply_two_by_two_pivot(
                    matrix,
                    nfs,
                    size,
                    &mut pivots,
                    &mut col_pivots,
                    &mut pivot_kinds,
                    k,
                    swap_row1,
                    swap_col1,
                    swap_row2,
                    swap_col2,
                    pivot_tol,
                )?
                else {
                    break;
                };
                k += advance;
            }
        }
    }

    Ok((pivots, col_pivots, pivot_kinds, k))
}

#[allow(clippy::too_many_arguments)]
fn compute_u12<T: ComplexField + FromPrimitive>(
    front_matrix: &DenseMatrix<T>,
    nfs: usize,
    ncb: usize,
    use_parallel: bool,
    pivot_kinds: &[PivotKind],
) -> DenseMatrix<T> {
    let compute_u12_column = |j: usize, col: &mut [T]| {
        for (i, slot) in col.iter_mut().enumerate() {
            *slot = front_matrix[[i, nfs + j]];
        }
        for i in 0..nfs {
            let mut s = col[i];
            #[allow(clippy::needless_range_loop)]
            for prev in 0..i {
                // Skip within-block L entry for the row of a 2×2 cont (we
                // stored D[1,0] there, not L).
                if pivot_kinds[i] == PivotKind::TwoByTwoCont && prev == i - 1 {
                    continue;
                }
                s -= front_matrix[[i, prev]] * col[prev];
            }
            col[i] = s;
        }
    };
    // Bit-identical 4-column interleave: each lane executes exactly its
    // scalar FMA sequence (same values, same order), while the four
    // independent chains overlap to hide FMA latency. The 2×2-Cont skip is
    // column-independent (same i/prev/kinds for all lanes), so one hoisted
    // branch covers the quad.
    // NOTE (falsified 2026-09-25): an unchecked-pointer variant of this
    // quad (raw lane pointers + one unsafe block, same op order) showed no
    // separable win — 4 reps trsm 67-83ms vs 71-82ms indexed, inside the
    // throttled-machine variance band — so the safe indexed form stands.
    let u12_quad = |j0: usize, c0: &mut [T], c1: &mut [T], c2: &mut [T], c3: &mut [T]| {
        for i in 0..nfs {
            c0[i] = front_matrix[[i, nfs + j0]];
            c1[i] = front_matrix[[i, nfs + j0 + 1]];
            c2[i] = front_matrix[[i, nfs + j0 + 2]];
            c3[i] = front_matrix[[i, nfs + j0 + 3]];
        }
        for i in 0..nfs {
            let (mut s0, mut s1, mut s2, mut s3) = (c0[i], c1[i], c2[i], c3[i]);
            let cont = pivot_kinds[i] == PivotKind::TwoByTwoCont;
            #[allow(clippy::needless_range_loop)]
            for prev in 0..i {
                if cont && prev == i - 1 {
                    continue;
                }
                let l = front_matrix[[i, prev]];
                s0 -= l * c0[prev];
                s1 -= l * c1[prev];
                s2 -= l * c2[prev];
                s3 -= l * c3[prev];
            }
            c0[i] = s0;
            c1[i] = s1;
            c2[i] = s2;
            c3[i] = s3;
        }
    };
    let mut u12_dense = DenseMatrix::<T>::zeros(nfs, ncb);
    if use_parallel && ncb > 1 {
        let mut u12_col_major = vec![T::zero(); nfs * ncb];
        let nquads = ncb / 4;
        let (quad_part, rest_part) = u12_col_major.split_at_mut(nquads * 4 * nfs);
        quad_part
            .par_chunks_mut(4 * nfs)
            .enumerate()
            .for_each(|(q, chunk)| {
                let (c0, r) = chunk.split_at_mut(nfs);
                let (c1, r) = r.split_at_mut(nfs);
                let (c2, c3) = r.split_at_mut(nfs);
                u12_quad(4 * q, c0, c1, c2, c3);
            });
        rest_part
            .par_chunks_mut(nfs)
            .enumerate()
            .for_each(|(r, col)| compute_u12_column(nquads * 4 + r, col));
        for j in 0..ncb {
            for i in 0..nfs {
                u12_dense[[i, j]] = u12_col_major[j * nfs + i];
            }
        }
    } else {
        let mut cols = [
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
        ];
        let nquads = ncb / 4;
        for q in 0..nquads {
            // Disjoint lane borrows via split_at_mut (indexing would alias).
            let (c0, rest) = cols.split_at_mut(1);
            let (c1, rest) = rest.split_at_mut(1);
            let (c2, c3) = rest.split_at_mut(1);
            u12_quad(4 * q, &mut c0[0], &mut c1[0], &mut c2[0], &mut c3[0]);
            for (k, lane) in [&c0[0], &c1[0], &c2[0], &c3[0]].iter().enumerate() {
                for i in 0..nfs {
                    u12_dense[[i, 4 * q + k]] = lane[i];
                }
            }
        }
        let mut col = vec![T::zero(); nfs];
        for j in (nquads * 4)..ncb {
            compute_u12_column(j, &mut col);
            for i in 0..nfs {
                u12_dense[[i, j]] = col[i];
            }
        }
    }
    u12_dense
}

#[allow(clippy::too_many_arguments)]
fn compute_l21<T: ComplexField + FromPrimitive>(
    front_matrix: &DenseMatrix<T>,
    nfs: usize,
    ncb: usize,
    use_parallel: bool,
    pivot_kinds: &[PivotKind],
    pivot_tol: T::Real,
) -> Result<DenseMatrix<T>, FrontError> {
    let compute_l21_row = |i: usize, row: &mut [T]| -> Result<(), FrontError> {
        let r = nfs + i;
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = front_matrix[[r, j]];
        }
        let mut j = 0usize;
        while j < nfs {
            match pivot_kinds[j] {
                PivotKind::OneByOne => {
                    let mut s = row[j];
                    #[allow(clippy::needless_range_loop)]
                    for prev in 0..j {
                        s -= row[prev] * front_matrix[[prev, j]];
                    }
                    let u_jj = front_matrix[[j, j]];
                    row[j] = s * u_jj.inv();
                    if !scalar_is_finite(row[j]) {
                        return Err(FrontError::numerical("L21 one-by-one solve", i * nfs + j));
                    }
                    j += 1;
                }
                PivotKind::TwoByTwoStart => {
                    let mut s0 = row[j];
                    let mut s1 = row[j + 1];
                    #[allow(clippy::needless_range_loop)]
                    for prev in 0..j {
                        s0 -= row[prev] * front_matrix[[prev, j]];
                        s1 -= row[prev] * front_matrix[[prev, j + 1]];
                    }
                    let a = front_matrix[[j, j]];
                    let b = front_matrix[[j, j + 1]];
                    let c = front_matrix[[j + 1, j]];
                    let d = front_matrix[[j + 1, j + 1]];
                    let d_inv = invert_2x2::<T>(a, b, c, d, pivot_tol)
                        .map_err(|_| FrontError::numerical("L21 2x2 block solve", i * nfs + j))?;
                    row[j] = s0 * d_inv[0][0] + s1 * d_inv[1][0];
                    row[j + 1] = s0 * d_inv[0][1] + s1 * d_inv[1][1];
                    if !scalar_is_finite(row[j]) || !scalar_is_finite(row[j + 1]) {
                        return Err(FrontError::numerical("L21 2x2 block solve", i * nfs + j));
                    }
                    j += 2;
                }
                PivotKind::TwoByTwoCont => {
                    return Err(FrontError::Invalid(
                        "TwoByTwoCont without preceding TwoByTwoStart",
                    ));
                }
            }
        }
        Ok(())
    };
    // Bit-identical 4-row interleave (mirror of the U12 quad): the pivot
    // stream is row-independent, so all four lanes advance in lockstep
    // with one shared `j` cursor. Per-`j` reciprocals (`u_jj.inv`,
    // `invert_2x2`) are hoisted across lanes — same values, so each
    // lane's FMA sequence is unchanged. Error stages/indices keep the
    // lane's own `i`; the defensive Cont arm (unreachable on valid pivot
    // streams) reports lane 0.
    let l21_quad = |i0: usize,
                    r0: &mut [T],
                    r1: &mut [T],
                    r2: &mut [T],
                    r3: &mut [T]|
     -> Result<(), FrontError> {
        for j in 0..nfs {
            r0[j] = front_matrix[[nfs + i0, j]];
            r1[j] = front_matrix[[nfs + i0 + 1, j]];
            r2[j] = front_matrix[[nfs + i0 + 2, j]];
            r3[j] = front_matrix[[nfs + i0 + 3, j]];
        }
        let mut j = 0usize;
        while j < nfs {
            match pivot_kinds[j] {
                PivotKind::OneByOne => {
                    let inv = front_matrix[[j, j]].inv();
                    let mut s = [r0[j], r1[j], r2[j], r3[j]];
                    #[allow(clippy::needless_range_loop)]
                    for prev in 0..j {
                        let u = front_matrix[[prev, j]];
                        s[0] -= r0[prev] * u;
                        s[1] -= r1[prev] * u;
                        s[2] -= r2[prev] * u;
                        s[3] -= r3[prev] * u;
                    }
                    let rows = [&mut *r0, &mut *r1, &mut *r2, &mut *r3];
                    for (k, sk) in s.iter().enumerate() {
                        rows[k][j] = *sk * inv;
                        if !scalar_is_finite(rows[k][j]) {
                            return Err(FrontError::numerical(
                                "L21 one-by-one solve",
                                (i0 + k) * nfs + j,
                            ));
                        }
                    }
                    j += 1;
                }
                PivotKind::TwoByTwoStart => {
                    let a = front_matrix[[j, j]];
                    let b = front_matrix[[j, j + 1]];
                    let c = front_matrix[[j + 1, j]];
                    let d = front_matrix[[j + 1, j + 1]];
                    let d_inv = invert_2x2::<T>(a, b, c, d, pivot_tol)
                        .map_err(|_| FrontError::numerical("L21 2x2 block solve", i0 * nfs + j))?;
                    let mut s0 = [r0[j], r1[j], r2[j], r3[j]];
                    let mut s1 = [r0[j + 1], r1[j + 1], r2[j + 1], r3[j + 1]];
                    #[allow(clippy::needless_range_loop)]
                    for prev in 0..j {
                        let uj = front_matrix[[prev, j]];
                        let uj1 = front_matrix[[prev, j + 1]];
                        s0[0] -= r0[prev] * uj;
                        s0[1] -= r1[prev] * uj;
                        s0[2] -= r2[prev] * uj;
                        s0[3] -= r3[prev] * uj;
                        s1[0] -= r0[prev] * uj1;
                        s1[1] -= r1[prev] * uj1;
                        s1[2] -= r2[prev] * uj1;
                        s1[3] -= r3[prev] * uj1;
                    }
                    let rows = [&mut *r0, &mut *r1, &mut *r2, &mut *r3];
                    for k in 0..4 {
                        rows[k][j] = s0[k] * d_inv[0][0] + s1[k] * d_inv[1][0];
                        rows[k][j + 1] = s0[k] * d_inv[0][1] + s1[k] * d_inv[1][1];
                        if !scalar_is_finite(rows[k][j]) || !scalar_is_finite(rows[k][j + 1]) {
                            return Err(FrontError::numerical(
                                "L21 2x2 block solve",
                                (i0 + k) * nfs + j,
                            ));
                        }
                    }
                    j += 2;
                }
                PivotKind::TwoByTwoCont => {
                    return Err(FrontError::Invalid(
                        "TwoByTwoCont without preceding TwoByTwoStart",
                    ));
                }
            }
        }
        Ok(())
    };
    let mut l21_dense = DenseMatrix::<T>::zeros(ncb, nfs);
    if use_parallel && ncb > 1 {
        let nquads = ncb / 4;
        let (quad_part, rest_part) = l21_dense.as_mut_slice().split_at_mut(nquads * 4 * nfs);
        quad_part
            .par_chunks_mut(4 * nfs)
            .enumerate()
            .try_for_each(|(q, chunk)| {
                let (r0, r) = chunk.split_at_mut(nfs);
                let (r1, r) = r.split_at_mut(nfs);
                let (r2, r3) = r.split_at_mut(nfs);
                l21_quad(4 * q, r0, r1, r2, r3)
            })?;
        rest_part
            .par_chunks_mut(nfs)
            .enumerate()
            .try_for_each(|(r, row)| compute_l21_row(nquads * 4 + r, row))?;
    } else {
        let mut rows = [
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
            vec![T::zero(); nfs],
        ];
        let nquads = ncb / 4;
        for q in 0..nquads {
            let (r0, rest) = rows.split_at_mut(1);
            let (r1, rest) = rest.split_at_mut(1);
            let (r2, r3) = rest.split_at_mut(1);
            l21_quad(4 * q, &mut r0[0], &mut r1[0], &mut r2[0], &mut r3[0])?;
            for (k, lane) in [&r0[0], &r1[0], &r2[0], &r3[0]].iter().enumerate() {
                for j in 0..nfs {
                    l21_dense[[4 * q + k, j]] = lane[j];
                }
            }
        }
        for i in (nquads * 4)..ncb {
            compute_l21_row(i, l21_dense.row_mut(i))?;
        }
    }
    Ok(l21_dense)
}

/// Form the Schur complement `F22 - L21 · U12` over the CB block.
///
/// `nfs_elim` is the number of eliminated pivots (the contraction
/// dimension of `L21 · U12`); `nfs_total` is the assembled FS size, used
/// to locate the F22 block inside `front_matrix`. The two differ when
/// delayed pivoting postponed part of the FS block.
#[allow(clippy::too_many_arguments)]
/// Either an already-packed panel or its still-dense source, handed from
/// the Schur update to the pack step so the pack can move (not copy)
/// single-block panel storage.
enum DenseOrPanel<T: ComplexField> {
    Dense(DenseMatrix<T>),
    Panel(Panel<T>),
}

#[expect(
    clippy::too_many_arguments,
    reason = "numeric kernel: 12 scalar/config operands; grouping them would churn the hot Schur-update call sites for no behavioral gain"
)]
fn compute_schur_update<T: ComplexField + FromPrimitive>(
    l21_dense: DenseMatrix<T>,
    u12_dense: DenseMatrix<T>,
    front_matrix: &DenseMatrix<T>,
    nfs_elim: usize,
    nfs_total: usize,
    ncb: usize,
    use_parallel: bool,
    update_order: UpdateOrder,
    panel_kind: PanelKind,
    block_size: usize,
    epsilon: T::Real,
    low_rank_precision: LowRankPrecision,
) -> (DenseMatrix<T>, DenseOrPanel<T>, DenseOrPanel<T>) {
    // Row-segment memcpy: every destination cell is overwritten, so no
    // zeroed allocation is needed.
    let mut cb_data = Vec::with_capacity(ncb * ncb);
    for i in 0..ncb {
        cb_data.extend_from_slice(&front_matrix.row(nfs_total + i)[nfs_total..nfs_total + ncb]);
    }
    let mut cb_dense = DenseMatrix::<T>::from_row_major_vec(ncb, ncb, cb_data);
    let (l21_for_pack, u12_for_pack) = match update_order {
        UpdateOrder::RightLook => {
            // Dense subtraction `F22 -= L21 · U12`. Keep the front matrix
            // separate from the product so the O(ncb²*nfs) work is one
            // contiguous BLAS-3 call (SME2-backed where available).
            let mut update = vec![T::zero(); ncb * ncb];
            T::mat_mul_nn(
                ncb,
                ncb,
                nfs_elim,
                l21_dense.as_slice(),
                u12_dense.as_slice(),
                &mut update,
            );
            for (value, product) in cb_dense.as_mut_slice().iter_mut().zip(update) {
                *value -= product;
            }
            (
                DenseOrPanel::Dense(l21_dense),
                DenseOrPanel::Dense(u12_dense),
            )
        }
        UpdateOrder::LeftLook => {
            // Compress L21/U12 into the active panel kind first, then
            // subtract the (kind-aware) product into the dense `cb_dense`
            // accumulator. Keep those panels for packing so LeftLook does
            // not pay for the same compression twice.
            let build_l21 = || {
                Panel::from_dense_owned_with_precision_maybe_parallel(
                    l21_dense,
                    panel_kind,
                    block_size,
                    epsilon,
                    low_rank_precision,
                    use_parallel,
                )
            };
            let build_u12 = || {
                Panel::from_dense_owned_with_precision_maybe_parallel(
                    u12_dense,
                    panel_kind,
                    block_size,
                    epsilon,
                    low_rank_precision,
                    use_parallel,
                )
            };
            let (l21_panel, u12_panel) = if use_parallel {
                rayon::join(build_l21, build_u12)
            } else {
                (build_l21(), build_u12())
            };
            l21_panel.matmul_sub_into_dense(&u12_panel, &mut cb_dense);
            (
                DenseOrPanel::Panel(l21_panel),
                DenseOrPanel::Panel(u12_panel),
            )
        }
    };
    (cb_dense, l21_for_pack, u12_for_pack)
}

#[allow(clippy::too_many_arguments)]
fn pack_factored_front<T: ComplexField + FromPrimitive>(
    nfs: usize,
    ncb: usize,
    fs_elim_indices: Vec<usize>,
    cb_indices: Vec<usize>,
    f11_lu: DenseMatrix<T>,
    pivots: Vec<usize>,
    col_pivots: Vec<usize>,
    pivot_kinds: Vec<PivotKind>,
    l21_dense: DenseOrPanel<T>,
    u12_dense: DenseOrPanel<T>,
    cb_dense: DenseMatrix<T>,
    panel_kind: PanelKind,
    block_size: usize,
    epsilon: T::Real,
    low_rank_precision: LowRankPrecision,
    use_parallel: bool,
    delayed_row_indices: Vec<usize>,
    delayed_col_indices: Vec<usize>,
    delayed: Option<DelayedBlocks<T>>,
) -> FactoredFront<T> {
    fn finish_panel<T: ComplexField + FromPrimitive>(
        src: DenseOrPanel<T>,
        panel_kind: PanelKind,
        block_size: usize,
        epsilon: T::Real,
        low_rank_precision: LowRankPrecision,
        use_parallel: bool,
    ) -> Panel<T> {
        match src {
            DenseOrPanel::Panel(panel) => panel,
            DenseOrPanel::Dense(dense) => Panel::from_dense_owned_with_precision_maybe_parallel(
                dense,
                panel_kind,
                block_size,
                epsilon,
                low_rank_precision,
                use_parallel,
            ),
        }
    }
    let build_l21 = || {
        finish_panel(
            l21_dense,
            panel_kind,
            block_size,
            epsilon,
            low_rank_precision,
            use_parallel,
        )
    };
    let build_u12 = || {
        finish_panel(
            u12_dense,
            panel_kind,
            block_size,
            epsilon,
            low_rank_precision,
            use_parallel,
        )
    };
    let build_cb = || {
        Panel::from_dense_owned_with_precision_maybe_parallel(
            cb_dense,
            panel_kind,
            block_size,
            epsilon,
            low_rank_precision,
            use_parallel,
        )
    };
    let (l21, u12, cb_panel) = if use_parallel {
        let (l21, (u12, cb_panel)) = rayon::join(build_l21, || rayon::join(build_u12, build_cb));
        (l21, u12, cb_panel)
    } else {
        (build_l21(), build_u12(), build_cb())
    };

    let ndelayed = delayed_row_indices.len();
    FactoredFront {
        nfs,
        ncb,
        fs_elim_indices,
        f11_lu,
        pivot_kinds,
        pivots,
        col_pivots,
        l21,
        u12,
        cb_indices,
        cb_panel,
        ndelayed,
        delayed_row_indices,
        delayed_col_indices,
        delayed,
    }
}

/// Factorise an `AssembledFront` in place.
///
/// Performs:
/// 1. Dense LU on F[0..nfs, 0..nfs] with the chosen pivot strategy. Row
///    pivots are propagated across the *entire* front so F[0..nfs, nfs..]
///    moves with them. Column pivots are *only* applied inside F11 (and to
///    F[nfs.., 0..nfs] so the Schur formation stays consistent).
/// 2. Forward-substitute to get `U12 = L^{-1} · F[0..k, nfs..]` for the
///    `k` eliminated pivots.
/// 3. Back-substitute to get `L21 = F[nfs.., 0..k] · U^{-1}` (with the
///    F21 cols already permuted by the FS column pivot).
/// 4. Schur: `F22_new = F[nfs.., nfs..] - L21 · U12`.
/// 5. BLR-compress L21, U12, and F22.
///
/// Pivot rejection is scale-aware: the rejection threshold is
/// `pivot_tol_rel · ε_machine · nfs · max|F11|`, the front-local analogue
/// of the sparse-LU `ε·n·max|A|` convention (`pivot_tol_rel` defaults to
/// 1, see [`crate::multifrontal_blr::BlrConfig::pivot_tol_multiplier`]).
///
/// When no acceptable pivot exists at some step `k < nfs` the front does
/// not fail. Elimination stops and the surviving `nfs - k` FS rows/cols
/// are packed as dense [`DelayedBlocks`] for the parent front:
/// - `dd`: the FS trailing block (already rank-k updated in place),
/// - `dc`: the delayed F12 rows, updated by `L·U12`,
/// - `cd`: the delayed F21 columns, updated by `L21·U`.
///
/// The parent extend-adds all three into its enlarged fully-summed block
/// and retries pivoting there. The caller reports singularity only if the
/// *root* front is left with delayed pivots.
#[allow(clippy::too_many_arguments)]
pub fn factor_front<T: ComplexField + FromPrimitive>(
    mut front: AssembledFront<T>,
    block_size: usize,
    epsilon: T::Real,
    pivot_strategy: PivotStrategy,
    panel_kind: PanelKind,
    low_rank_precision: LowRankPrecision,
    update_order: UpdateOrder,
    parallel_front_threshold: usize,
    pivot_tol_rel: T::Real,
) -> Result<FactoredFront<T>, FrontError> {
    let nfs = front.nfs;
    let ncb = front.ncb;
    let size = nfs + ncb;
    // Front-internal parallelism kicks in for fronts where the heavy
    // O((nfs + ncb)·ncb·nfs) Schur update has enough work to amortise
    // rayon task-spawn overhead. `parallel_front_threshold = 0`
    // disables it everywhere; large values force every front to run
    // single-threaded inside the per-level pool.
    let use_parallel = parallel_front_threshold > 0 && size > parallel_front_threshold;
    if let Some(index) = first_non_finite(front.matrix.as_slice()) {
        return Err(FrontError::numerical("front assembly", index));
    }

    // Scale-aware pivot rejection threshold, matching the sparse-LU
    // `ε·n·max|A|` convention (see direct/sparse_lu.rs). A pivot candidate
    // below this magnitude is treated as zero and delayed to the parent.
    let mut max_abs = T::Real::zero();
    for i in 0..nfs {
        for j in 0..nfs {
            let v = front.matrix[[i, j]].norm();
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
        return Err(FrontError::numerical("pivot tolerance", 0));
    }

    let (pivots_full, col_pivots_full, pivot_kinds_full, k_elim) =
        factor_frontal_block(&mut front.matrix, nfs, size, pivot_tol, pivot_strategy)?;
    if let Some(index) = first_non_finite(front.matrix.as_slice()) {
        return Err(FrontError::numerical("frontal factorization", index));
    }
    let nd = nfs - k_elim;

    let u12_dense = compute_u12(&front.matrix, k_elim, ncb, use_parallel, &pivot_kinds_full);
    if let Some(index) = first_non_finite(u12_dense.as_slice()) {
        return Err(FrontError::numerical("U12 solve", index));
    }
    let l21_dense = compute_l21(
        &front.matrix,
        k_elim,
        ncb,
        use_parallel,
        &pivot_kinds_full,
        pivot_tol,
    )?;
    // Delayed-pivot payload: the restriction of the updated trailing
    // submatrix `F[S,S] - F[S,E]·F[E,E]⁻¹·F[E,S]` to the blocks that
    // involve at least one delayed index (the CB×CB part ships separately
    // as `cb_panel`). The FS trailing block `dd` was already rank-k
    // updated in place by the elimination; `dc` / `cd` need the explicit
    // `L·U12` / `L21·U` correction.
    let delayed = if nd > 0 {
        let mut dd = DenseMatrix::<T>::zeros(nd, nd);
        for a in 0..nd {
            for b in 0..nd {
                dd[[a, b]] = front.matrix[[k_elim + a, k_elim + b]];
            }
        }
        let mut dc = DenseMatrix::<T>::zeros(nd, ncb);
        for a in 0..nd {
            for c in 0..ncb {
                let mut s = front.matrix[[k_elim + a, nfs + c]];
                for p in 0..k_elim {
                    s -= front.matrix[[k_elim + a, p]] * u12_dense[[p, c]];
                }
                dc[[a, c]] = s;
            }
        }
        let mut cd = DenseMatrix::<T>::zeros(ncb, nd);
        for c in 0..ncb {
            for b in 0..nd {
                let mut s = front.matrix[[nfs + c, k_elim + b]];
                for p in 0..k_elim {
                    s -= l21_dense[[c, p]] * front.matrix[[p, k_elim + b]];
                }
                cd[[c, b]] = s;
            }
        }
        Some(DelayedBlocks { dd, dc, cd })
    } else {
        None
    };

    // Map the surviving (delayed) assembled-FS positions to
    // elimination-order indices via the pivot permutations.
    let delayed_row_indices: Vec<usize> = pivots_full[k_elim..]
        .iter()
        .map(|&p| front.fs_elim_indices[p])
        .collect();
    let delayed_col_indices: Vec<usize> = col_pivots_full[k_elim..]
        .iter()
        .map(|&p| front.fs_elim_indices[p])
        .collect();

    let mut pivots = pivots_full;
    pivots.truncate(k_elim);
    let mut col_pivots = col_pivots_full;
    col_pivots.truncate(k_elim);
    let mut pivot_kinds = pivot_kinds_full;
    pivot_kinds.truncate(k_elim);

    let (cb_dense, l21_for_pack, u12_for_pack) = compute_schur_update(
        l21_dense,
        u12_dense,
        &front.matrix,
        k_elim,
        nfs,
        ncb,
        use_parallel,
        update_order,
        panel_kind,
        block_size,
        epsilon,
        low_rank_precision,
    );
    if let Some(index) = first_non_finite(cb_dense.as_slice()) {
        return Err(FrontError::numerical("Schur update", index));
    }
    // Row-segment memcpy: every destination cell is overwritten, so no
    // zeroed allocation is needed.
    let mut f11_data = Vec::with_capacity(k_elim * k_elim);
    for i in 0..k_elim {
        f11_data.extend_from_slice(&front.matrix.row(i)[..k_elim]);
    }
    let f11_lu = DenseMatrix::<T>::from_row_major_vec(k_elim, k_elim, f11_data);

    let factored = pack_factored_front(
        k_elim,
        ncb,
        front.fs_elim_indices,
        front.cb_indices,
        f11_lu,
        pivots,
        col_pivots,
        pivot_kinds,
        l21_for_pack,
        u12_for_pack,
        cb_dense,
        panel_kind,
        block_size,
        epsilon,
        low_rank_precision,
        use_parallel,
        delayed_row_indices,
        delayed_col_indices,
        delayed,
    );
    factored.validate()?;
    Ok(factored)
}

impl<T: ComplexField + FromPrimitive> FactoredFront<T> {
    pub(crate) fn validate(&self) -> Result<(), FrontError> {
        if self.f11_lu.nrows() != self.nfs || self.f11_lu.ncols() != self.nfs {
            return Err(FrontError::Invalid("F11 dimensions do not match nfs"));
        }
        if self.pivot_kinds.len() != self.nfs
            || self.pivots.len() != self.nfs
            || self.col_pivots.len() != self.nfs
        {
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
            .chain(&self.col_pivots)
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
        if self.l21.n_rows() != self.ncb
            || self.l21.n_cols() != self.nfs
            || self.u12.n_rows() != self.nfs
            || self.u12.n_cols() != self.ncb
            || self.cb_panel.n_rows() != self.ncb
            || self.cb_panel.n_cols() != self.ncb
        {
            return Err(FrontError::Invalid(
                "compressed panel dimensions are inconsistent",
            ));
        }
        if self.ndelayed != self.delayed_row_indices.len()
            || self.ndelayed != self.delayed_col_indices.len()
            || (self.ndelayed == 0) != self.delayed.is_none()
        {
            return Err(FrontError::Invalid(
                "delayed-pivot metadata is inconsistent",
            ));
        }
        if let Some(delayed) = &self.delayed {
            if delayed.dd.nrows() != self.ndelayed
                || delayed.dd.ncols() != self.ndelayed
                || delayed.dc.nrows() != self.ndelayed
                || delayed.dc.ncols() != self.ncb
                || delayed.cd.nrows() != self.ncb
                || delayed.cd.ncols() != self.ndelayed
            {
                return Err(FrontError::Invalid(
                    "delayed-pivot block dimensions are inconsistent",
                ));
            }
            for (stage, values) in [
                ("delayed DD block", delayed.dd.as_slice()),
                ("delayed DC block", delayed.dc.as_slice()),
                ("delayed CD block", delayed.cd.as_slice()),
            ] {
                if let Some(index) = first_non_finite(values) {
                    return Err(FrontError::numerical(stage, index));
                }
            }
        }
        if let Some(index) = first_non_finite(self.f11_lu.as_slice()) {
            return Err(FrontError::numerical("stored F11 factor", index));
        }
        if !self.l21.is_finite() || !self.u12.is_finite() || !self.cb_panel.is_finite() {
            return Err(FrontError::numerical("compressed panel storage", 0));
        }
        Ok(())
    }

    /// In-place forward apply on a front-local vector of length `nfs + ncb`:
    /// - `b[0..nfs] := L^{-1} · b[0..nfs]` (block-aware)
    /// - `b[nfs..]  -= L21 · b[0..nfs]`
    ///
    /// The caller must pass `b[0..nfs]` already gathered into factored row
    /// order, i.e. `b[i] = rhs_FS[pivots[i]]` where `pivots` maps each
    /// eliminated step to its assembled-FS row position (this is the
    /// row-pivot permutation `P_r`, applied at gather time so fronts with
    /// delayed pivots — whose eliminated set is a non-contiguous subset of
    /// the assembled FS — work unchanged).
    pub fn forward_apply(&self, b: &mut [T]) {
        if let Err(error) = self.forward_apply_checked(b) {
            panic!("invalid multifrontal front forward apply: {error:?}");
        }
    }

    /// Checked counterpart to [`Self::forward_apply`].
    ///
    /// The legacy method remains available for compatibility, but callers
    /// handling untrusted or externally supplied factor state should use this
    /// boundary so malformed metadata and vector dimensions become typed
    /// [`FrontError`] values instead of panics.
    pub fn try_forward_apply(&self, b: &mut [T]) -> Result<(), FrontError> {
        self.forward_apply_checked(b)
    }

    pub(crate) fn forward_apply_checked(&self, b: &mut [T]) -> Result<(), FrontError> {
        self.validate()?;
        self.forward_apply_assume_validated(b)
    }

    /// [`forward_apply_checked`](Self::forward_apply_checked) without the
    /// metadata validation pass, for callers holding fronts validated at
    /// pack time (the struct is immutable after `factored.validate()`, and
    /// the solve path only shares `&` references, so re-validation per
    /// front per solve is redundant). The `b` length check stays: it guards
    /// the caller-built vector, not front metadata. Bit-identical output.
    pub(crate) fn forward_apply_assume_validated(&self, b: &mut [T]) -> Result<(), FrontError> {
        let nfs = self.nfs;
        if b.len() != nfs + self.ncb {
            return Err(FrontError::Invalid("front vector length mismatch"));
        }
        // 1. Forward sub with block-aware unit L.
        for i in 0..nfs {
            let mut s = b[i];
            #[allow(clippy::needless_range_loop)]
            for prev in 0..i {
                if self.pivot_kinds[i] == PivotKind::TwoByTwoCont && prev == i - 1 {
                    // Within-block entry is D[1,0], not L.
                    continue;
                }
                s -= self.f11_lu[[i, prev]] * b[prev];
            }
            b[i] = s;
            if !scalar_is_finite(s) {
                return Err(FrontError::numerical("front forward substitution", i));
            }
        }
        // 2. CB update.
        if self.ncb > 0 {
            let (b_fs, b_cb) = b.split_at_mut(nfs);
            let mut tmp = vec![T::zero(); self.ncb];
            self.l21.gemv_into(T::one(), b_fs, &mut tmp);
            for i in 0..self.ncb {
                b_cb[i] -= tmp[i];
                if !scalar_is_finite(b_cb[i]) {
                    return Err(FrontError::numerical("front L21 update", i));
                }
            }
        }
        Ok(())
    }

    /// In-place backward apply:
    /// - `b[0..nfs] := U^{-1} · (b[0..nfs] - U12 · b[nfs..])` (block-aware)
    ///
    /// The result is left in *factored column order*: the caller scatters
    /// step `k` to assembled-FS position `col_pivots[k]` (the column-pivot
    /// permutation `P_c`, applied at scatter time for the same reason as
    /// the row-pivot gather in [`Self::forward_apply`]).
    pub fn backward_apply(&self, b: &mut [T]) {
        if let Err(error) = self.backward_apply_checked(b) {
            panic!("invalid multifrontal front backward apply: {error:?}");
        }
    }

    /// Checked counterpart to [`Self::backward_apply`].
    pub fn try_backward_apply(&self, b: &mut [T]) -> Result<(), FrontError> {
        self.backward_apply_checked(b)
    }

    pub(crate) fn backward_apply_checked(&self, b: &mut [T]) -> Result<(), FrontError> {
        self.validate()?;
        self.backward_apply_assume_validated(b)
    }

    /// [`backward_apply_checked`](Self::backward_apply_checked) without the
    /// metadata validation pass. Same trust argument as
    /// [`forward_apply_assume_validated`](Self::forward_apply_assume_validated).
    pub(crate) fn backward_apply_assume_validated(&self, b: &mut [T]) -> Result<(), FrontError> {
        let nfs = self.nfs;
        if b.len() != nfs + self.ncb {
            return Err(FrontError::Invalid("front vector length mismatch"));
        }
        // 1. Subtract U12 · b[nfs..].
        if self.ncb > 0 {
            let (b_fs, b_cb) = b.split_at_mut(nfs);
            let mut tmp = vec![T::zero(); nfs];
            self.u12.gemv_into(T::one(), b_cb, &mut tmp);
            for i in 0..nfs {
                b_fs[i] -= tmp[i];
                if !scalar_is_finite(b_fs[i]) {
                    return Err(FrontError::numerical("front U12 update", i));
                }
            }
        }
        // 2. Block-aware back sub.
        //
        // Process columns from high to low. For a 1×1 pivot at j, divide by
        // U[j, j]. For a 2×2 block starting at j (which means kind[j] =
        // TwoByTwoStart and kind[j+1] = TwoByTwoCont), first subtract all U
        // contributions from columns > j+1 on both b[j] and b[j+1], then
        // solve D · [b[j], b[j+1]]^T = [b[j], b[j+1]]^T via D^{-1}.
        //
        // Every 2×2 block stored here passed the factor-time invertibility
        // check, so the solve-time inversion uses a zero tolerance (an
        // absolute threshold would mis-fire on badly scaled matrices).
        let pivot_tol = T::Real::zero();
        let mut j = nfs;
        while j > 0 {
            let kind_jm1 = self.pivot_kinds[j - 1];
            if kind_jm1 == PivotKind::TwoByTwoCont {
                // 2×2 block at [j-2, j-1].
                let lo = j - 2;
                let mut s0 = b[lo];
                let mut s1 = b[lo + 1];
                for (hi, &b_hi) in b.iter().enumerate().take(nfs).skip(lo + 2) {
                    s0 -= self.f11_lu[[lo, hi]] * b_hi;
                    s1 -= self.f11_lu[[lo + 1, hi]] * b_hi;
                }
                let a = self.f11_lu[[lo, lo]];
                let b_e = self.f11_lu[[lo, lo + 1]];
                let c = self.f11_lu[[lo + 1, lo]];
                let d = self.f11_lu[[lo + 1, lo + 1]];
                let d_inv = invert_2x2::<T>(a, b_e, c, d, pivot_tol)
                    .map_err(|_| FrontError::numerical("front 2x2 back substitution", lo))?;
                b[lo] = d_inv[0][0] * s0 + d_inv[0][1] * s1;
                b[lo + 1] = d_inv[1][0] * s0 + d_inv[1][1] * s1;
                if !scalar_is_finite(b[lo]) || !scalar_is_finite(b[lo + 1]) {
                    return Err(FrontError::numerical("front 2x2 back substitution", lo));
                }
                j -= 2;
            } else {
                let i = j - 1;
                let mut s = b[i];
                #[allow(clippy::needless_range_loop)]
                for hi in (i + 1)..nfs {
                    s -= self.f11_lu[[i, hi]] * b[hi];
                }
                b[i] = s * self.f11_lu[[i, i]].inv();
                if !scalar_is_finite(b[i]) {
                    return Err(FrontError::numerical("front back substitution", i));
                }
                j -= 1;
            }
        }
        Ok(())
    }

    /// Total stored entries (eliminated FS dense + L21 + U12 + CB panel +
    /// dense delayed-pivot blocks still held for the parent).
    pub fn stored_entries(&self) -> usize {
        let delayed_entries = self
            .delayed
            .as_ref()
            .map(|d| {
                d.dd.nrows() * d.dd.ncols()
                    + d.dc.nrows() * d.dc.ncols()
                    + d.cd.nrows() * d.cd.ncols()
            })
            .unwrap_or(0);
        self.nfs * self.nfs
            + self.l21.stored_entries()
            + self.u12.stored_entries()
            + self.cb_panel.stored_entries()
            + delayed_entries
    }
}
