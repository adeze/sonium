//! Numeric multifrontal factorisation.
//!
//! Walks the assembly tree in post-order: at each front, the FS row/col set
//! plus the union of children's CB sets defines the front's full index list.
//! We assemble a dense front, extend-add child CBs, factor with partial
//! pivoting, and store the BLR-compressed factors. The child's `cb_panel`
//! is consumed at this point and dropped.

use std::collections::BTreeSet;

use num_traits::FromPrimitive;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

use super::blr::{BlrBlock, LowRankPrecision};
use super::front::{AssembledFront, FactoredFront, PivotStrategy, UpdateOrder, factor_front};
use super::ordering::AssemblyTree;
use super::panel::{Panel, PanelKind};
use super::{BlrError, FrontError};

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

/// Extend-add a child's CB panel into the parent frontal matrix without
/// materializing the panel dense: each block adds straight into its mapped
/// rows/cols (rank-0 LowRank blocks contribute nothing and are skipped).
/// Each target cell is touched by exactly one block cell, so traversal
/// order is irrelevant and the result matches the `to_dense` + add path
/// bit-for-bit on valid panels (all-zero rank-0 blocks add +0.0; see the
/// -0.0 note in the test below).
fn extend_add_panel<T: ComplexField>(panel: &Panel<T>, map: &[usize], matrix: &mut DenseMatrix<T>) {
    let Panel::Blr(blr) = panel else {
        // Non-BLR panels keep the materialized path.
        let dense = panel.to_dense();
        for (i, &r) in map.iter().enumerate() {
            for (j, &c) in map.iter().enumerate() {
                matrix[[r, c]] += dense[[i, j]];
            }
        }
        return;
    };
    let bs = blr.bs;
    for (bi, row) in blr.blocks.iter().enumerate() {
        for (bj, block) in row.iter().enumerate() {
            match block {
                BlrBlock::Dense(m) => {
                    for (di, r) in (bi * bs..bi * bs + m.nrows()).enumerate() {
                        let row_local = map[r];
                        for (dj, c) in (bj * bs..bj * bs + m.ncols()).enumerate() {
                            matrix[[row_local, map[c]]] += m[[di, dj]];
                        }
                    }
                }
                BlrBlock::LowRank { u, .. } if u.ncols() == 0 => {
                    // Rank-0 factor pair encodes an all-zero block: nothing
                    // to add (u·v over an empty sum is +0.0).
                }
                other => {
                    // Generic (compressible or half-precision) path: keep
                    // the materialized semantics via to_dense.
                    let dense = other.to_dense();
                    for (di, r) in (bi * bs..bi * bs + dense.nrows()).enumerate() {
                        let row_local = map[r];
                        for (dj, c) in (bj * bs..bj * bs + dense.ncols()).enumerate() {
                            matrix[[row_local, map[c]]] += dense[[di, dj]];
                        }
                    }
                }
            }
        }
    }
}

/// Structural symmetry prepass: every off-diagonal entry needs its
/// transpose present. CSR rows are sorted, so one binary search per
/// upper-triangle entry: O(nnz log deg), run once per factorisation.
fn is_structurally_symmetric<T: ComplexField>(a_perm: &CsrMatrix<T>) -> bool {
    if a_perm.num_rows != a_perm.num_cols {
        return false;
    }
    for i in 0..a_perm.num_rows {
        for idx in a_perm.row_ptrs[i]..a_perm.row_ptrs[i + 1] {
            let j = a_perm.col_indices[idx];
            if j <= i {
                continue;
            }
            let row_j = &a_perm.col_indices[a_perm.row_ptrs[j]..a_perm.row_ptrs[j + 1]];
            if row_j.binary_search(&i).is_err() {
                return false;
            }
        }
    }
    true
}

/// Build the front's full index list (FS indices first, then CB indices).
///
/// CB indices are the union of
/// 1. Elimination-order indices `j ≥ fs_end` appearing in `A_perm[i, j]`
///    for some `i ∈ [fs_start, fs_end)` (outgoing edges from FS rows).
/// 2. Elimination-order indices `i ≥ fs_end` appearing in `A_perm[i, j]`
///    for some `j ∈ [fs_start, fs_end)` (incoming edges into FS columns).
///    This is what `a_perm_in` covers — without it, structurally
///    nonsymmetric matrices where `A[i, j] ≠ 0` but `A[j, i] = 0` lose
///    the F21 contribution.
/// 3. Children's CB indices `> fs_end` (those `≤ fs_end` have already
///    been eliminated).
fn compute_cb_indices<T: ComplexField>(
    a_perm: &CsrMatrix<T>,
    // `None` for structurally symmetric matrices: part (2) below is
    // redundant there (every incoming edge's transpose is an outgoing edge
    // of an FS row, already pushed by part (1)), so the incoming buckets
    // are neither built nor consulted. Output is the identical
    // sorted-unique set either way.
    a_perm_in: Option<&[Vec<usize>]>,
    fs_start: usize,
    fs_end: usize,
    children_cb: &[&[usize]],
) -> Vec<usize> {
    // Bulk sort+dedup instead of per-edge BTreeSet inserts: identical
    // sorted-unique output, no per-element node allocation. (Was the
    // dominant per-front fixed cost: every edge re-inserted at each of
    // ~levels fronts on the path to the root.)
    let mut reserve = 0usize;
    for i in fs_start..fs_end {
        reserve += a_perm.row_ptrs[i + 1] - a_perm.row_ptrs[i];
    }
    if let Some(buckets) = a_perm_in {
        for rows in buckets.iter().take(fs_end).skip(fs_start) {
            reserve += rows.len();
        }
    }
    for &child_cb in children_cb {
        reserve += child_cb.len();
    }
    let mut cb: Vec<usize> = Vec::with_capacity(reserve);
    // (1) Outgoing edges from FS rows.
    for i in fs_start..fs_end {
        for idx in a_perm.row_ptrs[i]..a_perm.row_ptrs[i + 1] {
            let j = a_perm.col_indices[idx];
            if j >= fs_end {
                cb.push(j);
            }
        }
    }
    // (2) Incoming edges into FS columns — needed for structurally
    // nonsymmetric matrices so F21-only entries land in the front.
    // Skipped when `a_perm_in` is `None` (symmetric: redundant, see above).
    if let Some(buckets) = a_perm_in {
        for rows in buckets.iter().take(fs_end).skip(fs_start) {
            for &row in rows {
                if row >= fs_end {
                    cb.push(row);
                }
            }
        }
    }
    // (3) Children's CB indices propagated up.
    for &child_cb in children_cb {
        for &g in child_cb {
            if g >= fs_end {
                cb.push(g);
            }
        }
    }
    cb.sort_unstable();
    cb.dedup();
    cb
}

/// Build the column-incoming structural pattern of `a_perm`:
/// `out[c]` is a sorted list of rows `i` such that `a_perm[i, c] ≠ 0`.
pub(crate) fn build_column_incoming<T: ComplexField>(a_perm: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = a_perm.num_cols;
    // Two-pass bucket sort.
    let mut counts = vec![0usize; n];
    for i in 0..a_perm.num_rows {
        for idx in a_perm.row_ptrs[i]..a_perm.row_ptrs[i + 1] {
            counts[a_perm.col_indices[idx]] += 1;
        }
    }
    let mut out: Vec<Vec<usize>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
    for i in 0..a_perm.num_rows {
        for idx in a_perm.row_ptrs[i]..a_perm.row_ptrs[i + 1] {
            out[a_perm.col_indices[idx]].push(i);
        }
    }
    // Row ordering inside CSR is sorted; the column-incoming bucket is
    // naturally sorted by row by construction of the outer loop.
    out
}

/// Assemble a single front: fill its dense `(nfs + ncb)²` matrix with the
/// permuted-A entries plus extend-add of each child's CB panel and each
/// child's dense delayed-pivot blocks.
///
/// The front's fully-summed set is its own FS range `[fs_start, fs_end)`
/// *plus* every pivot the children postponed (delayed pivoting): the
/// delayed indices are appended after the own FS rows/cols and pivoting is
/// retried on the enlarged block.
fn assemble_front<T: ComplexField>(
    a_perm: &CsrMatrix<T>,
    fs_start: usize,
    fs_end: usize,
    cb_indices: Vec<usize>,
    children_factored: &[&FactoredFront<T>],
    front: usize,
) -> Result<AssembledFront<T>, BlrError> {
    if fs_start > fs_end || fs_end > a_perm.num_rows || a_perm.num_rows != a_perm.num_cols {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front fully-summed range is outside the matrix",
        });
    }
    if cb_indices.iter().any(|&g| g >= a_perm.num_rows)
        || cb_indices.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "front contribution indices are invalid",
        });
    }
    if children_factored.iter().any(|child| {
        child.ncb != child.cb_indices.len()
            || child.cb_indices.iter().any(|&g| g >= a_perm.num_rows)
            || child.cb_indices.windows(2).any(|pair| pair[0] >= pair[1])
            || child.ndelayed != child.delayed_row_indices.len()
            || child.ndelayed != child.delayed_col_indices.len()
    }) {
        return Err(BlrError::InvalidFactorization {
            front,
            reason: "child front index metadata is inconsistent",
        });
    }
    let own_nfs = fs_end - fs_start;
    // Delayed pivots postponed by the children join this front's
    // fully-summed set (sorted, appended after the own FS block).
    let mut delayed_set: BTreeSet<usize> = BTreeSet::new();
    for child in children_factored {
        for &g in child
            .delayed_row_indices
            .iter()
            .chain(child.delayed_col_indices.iter())
        {
            debug_assert!(
                g < fs_start,
                "delayed index {g} must lie in a descendant's fs range (< {fs_start})"
            );
            delayed_set.insert(g);
        }
    }
    let delayed: Vec<usize> = delayed_set.into_iter().collect();
    let nfs = own_nfs + delayed.len();
    let ncb = cb_indices.len();
    let size = nfs + ncb;

    let mut matrix = DenseMatrix::<T>::zeros(size, size);

    // Front-local indexing: own FS elim indices map to `g - fs_start`;
    // absorbed delayed indices map to `own_nfs + rank` in the sorted
    // delayed list; CB elim indices map to `nfs + position in cb_indices`
    // (binary search — O(log ncb), plenty fast for prototype sizes).
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

    // (1) Fill from A_perm. An entry A_perm[i, j] belongs to this front iff
    // `min(i, j) ∈ [fs_start, fs_end)` — i.e. the smaller-indexed endpoint
    // is fully-summed here. That excludes CB×CB entries (their LCA in the
    // assembly tree is some ancestor, which owns those entries).
    //
    // Three placement cases:
    //   FS rows  → any column in FS ∪ CB (entry goes to F11 or F12).
    //   CB rows  → only FS columns (entry goes to F21).
    // CB-CB entries are intentionally skipped.
    //
    // Entries touching a *delayed* index are never filled here: the child
    // that postponed the index already assembled them (it owns
    // `min(i, j)`), and they arrive via its dense delayed blocks below.
    for i in fs_start..fs_end {
        let row_local = i - fs_start;
        for idx in a_perm.row_ptrs[i]..a_perm.row_ptrs[i + 1] {
            let j = a_perm.col_indices[idx];
            let v = a_perm.values[idx];
            let col_local = if j >= fs_start && j < fs_end {
                Some(j - fs_start)
            } else if j >= fs_end {
                cb_pos(j)
            } else {
                None // j < fs_start: eliminated by a descendant, or a
                // delayed index whose entries arrive via the child's
                // delayed blocks — never refilled from A here.
            };
            if let Some(c) = col_local {
                matrix[[row_local, c]] += v;
            }
        }
    }
    for (k, &g) in cb_indices.iter().enumerate() {
        let row_local = nfs + k;
        for idx in a_perm.row_ptrs[g]..a_perm.row_ptrs[g + 1] {
            let j = a_perm.col_indices[idx];
            // CB row: only own-FS columns contribute (entry goes to F21).
            // Columns below fs_start (eliminated or delayed) arrive via
            // the children's panels, not from A.
            if j >= fs_start && j < fs_end {
                let v = a_perm.values[idx];
                let col_local = j - fs_start;
                matrix[[row_local, col_local]] += v;
            }
        }
    }

    // (2) Extend-add: for each child, look up where its CB indices map in
    // this front (own FS, absorbed delayed slot, or CB) and add the
    // child's CB panel values.
    for child in children_factored {
        let cb_pos_for_child: Vec<usize> = child
            .cb_indices
            .iter()
            .map(|&g| fs_or_cb_pos(g))
            .collect::<Result<_, _>>()?;
        extend_add_panel(&child.cb_panel, &cb_pos_for_child, &mut matrix);
    }

    // (3) Extend-add each child's dense delayed-pivot blocks: the updated
    // `F[D,D]`, `F[D,C]` and `F[C,D]` blocks for the postponed indices.
    // These land in the enlarged FS block (or at the FS/CB boundary) and
    // are retried by this front's pivot search.
    for child in children_factored {
        let Some(blocks) = &child.delayed else {
            continue;
        };
        let nd = child.ndelayed;
        for a in 0..nd {
            let row_local =
                fs_pos(child.delayed_row_indices[a]).ok_or(BlrError::InvalidFactorization {
                    front,
                    reason: "delayed row index missing from parent front index set",
                })?;
            for b in 0..nd {
                let col_local =
                    fs_pos(child.delayed_col_indices[b]).ok_or(BlrError::InvalidFactorization {
                        front,
                        reason: "delayed column index missing from parent front index set",
                    })?;
                matrix[[row_local, col_local]] += blocks.dd[[a, b]];
            }
            for (c, &g) in child.cb_indices.iter().enumerate() {
                matrix[[row_local, fs_or_cb_pos(g)?]] += blocks.dc[[a, c]];
            }
        }
        for (c, &g) in child.cb_indices.iter().enumerate() {
            let row_local = fs_or_cb_pos(g)?;
            for b in 0..nd {
                let col_local =
                    fs_pos(child.delayed_col_indices[b]).ok_or(BlrError::InvalidFactorization {
                        front,
                        reason: "delayed column index missing from parent front index set",
                    })?;
                matrix[[row_local, col_local]] += blocks.cd[[c, b]];
            }
        }
    }

    // Elimination-order index of every assembled FS position: own FS first
    // (`fs_start + p`), then the absorbed delayed indices.
    let mut fs_elim_indices: Vec<usize> = (fs_start..fs_end).collect();
    fs_elim_indices.extend_from_slice(&delayed);

    Ok(AssembledFront {
        nfs,
        ncb,
        matrix,
        fs_elim_indices,
        cb_indices,
    })
}

/// Permute the input matrix from global ordering to elimination ordering.
///
/// Returns `B = P · A · P^T` where `P` is the permutation matrix defined by
/// `perm_inv`: `B[perm_inv[i], perm_inv[j]] = A[i, j]`.
pub fn permute_matrix<T: ComplexField>(matrix: &CsrMatrix<T>, perm_inv: &[usize]) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let nnz = matrix.values.len();

    // Build (new_row, new_col, val) triplets then bucket-sort into CSR.
    let mut row_starts = vec![0usize; n + 1];
    for (i, &new_row) in perm_inv.iter().take(n).enumerate() {
        let count = matrix.row_ptrs[i + 1] - matrix.row_ptrs[i];
        row_starts[new_row + 1] += count;
    }
    for i in 0..n {
        row_starts[i + 1] += row_starts[i];
    }
    debug_assert_eq!(row_starts[n], nnz);

    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut cursors = row_starts.clone();

    for i in 0..n {
        let new_row = perm_inv[i];
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let dst = cursors[new_row];
            cursors[new_row] += 1;
            col_indices[dst] = perm_inv[matrix.col_indices[idx]];
            values[dst] = matrix.values[idx];
        }
    }

    // Sort each new row by column index (stable, in-place).
    for r in 0..n {
        let s = row_starts[r];
        let e = row_starts[r + 1];
        if e - s > 1 {
            let mut pairs: Vec<(usize, T)> = (s..e).map(|k| (col_indices[k], values[k])).collect();
            pairs.sort_by_key(|p| p.0);
            for (k, (c, v)) in pairs.into_iter().enumerate() {
                col_indices[s + k] = c;
                values[s + k] = v;
            }
        }
    }

    CsrMatrix::from_raw_parts(n, n, row_starts, col_indices, values)
}

/// Run the numeric multifrontal factorisation. Returns one `FactoredFront`
/// per assembly-tree node, in factorisation (post-order) order.
///
/// Fronts that cannot eliminate their whole fully-summed block postpone
/// the surviving pivots to their parent (delayed pivoting); the error
/// return fires only when the *root* front is left with postponed pivots,
/// i.e. the matrix is numerically singular.
#[allow(clippy::too_many_arguments)]
pub fn numeric_factorise<T: ComplexField + FromPrimitive>(
    a_perm: &CsrMatrix<T>,
    tree: &AssemblyTree,
    block_size: usize,
    epsilon: T::Real,
    pivot_strategy: PivotStrategy,
    panel_kind: PanelKind,
    low_rank_precision: LowRankPrecision,
    update_order: UpdateOrder,
    parallel_front_threshold: usize,
    pivot_tol_rel: T::Real,
) -> Result<Vec<FactoredFront<T>>, BlrError> {
    let n_fronts = tree.fronts.len();
    let mut factored: Vec<Option<FactoredFront<T>>> = (0..n_fronts).map(|_| None).collect();

    // Column-incoming structural pattern of `a_perm`, used by
    // `compute_cb_indices` so structurally nonsymmetric matrices contribute
    // F21-only entries correctly. Built once, shared across all fronts —
    // unless `a_perm` is structurally symmetric, in which case part (1) of
    // `compute_cb_indices` already covers every edge and the buckets are
    // skipped entirely (same sorted-unique output, ~half the pushes).
    let symmetric = is_structurally_symmetric(a_perm);
    let a_perm_in = if symmetric {
        Vec::new()
    } else {
        build_column_incoming(a_perm)
    };
    let a_perm_in_opt = (!symmetric).then_some(a_perm_in.as_slice());

    // Factorise in post-order (the natural order of `tree.fronts`).
    // Tree-level parallelism via rayon: fronts at the same level have no
    // mutual dependency. Within a level, we process them in parallel; across
    // levels we synchronise. NOTE: a child can sit at a lower level than its
    // parent's other branch; the `tree.levels` grouping is by dissection-tree
    // level so parent is always at a strictly higher level than its children.
    use rayon::prelude::*;
    for level_fronts in &tree.levels {
        // Sequentially read children's CBs, build inputs, then process this
        // level. We cannot mutate `factored` from multiple threads, so we
        // gather the dense fronts first (read-only access to children), then
        // run the factorisation in parallel and collect results.
        // NOTE (parallel-assembly attempt, reverted 2026-09-25): per-level
        // par_map over fronts was bit-identical (disjoint writes,
        // order-preserving collect) and faster cold (11.2 vs 19.6ms rep1:
        // concurrent first-touch), but slower warm (9.4-9.9 vs 8.0-8.5ms:
        // top levels have 1-4 fronts so dispatch is pure overhead). The
        // library hot path (repeated factorization) is warm — keep serial.
        let mut inputs: Vec<(usize, AssembledFront<T>)> = Vec::with_capacity(level_fronts.len());
        for &front_idx in level_fronts {
            let f = &tree.fronts[front_idx];
            let mut children_factored: Vec<&FactoredFront<T>> =
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
            let cb_indices = compute_cb_indices(
                a_perm,
                a_perm_in_opt,
                f.fs_start,
                f.fs_end,
                &children_cb_slices,
            );
            let assembled = assemble_front(
                a_perm,
                f.fs_start,
                f.fs_end,
                cb_indices,
                &children_factored,
                front_idx,
            )?;
            inputs.push((front_idx, assembled));
        }
        // Now factor each in parallel (independent — they only read from
        // `inputs`, which is owned data).
        let results: Vec<(usize, Result<FactoredFront<T>, FrontError>)> = inputs
            .into_par_iter()
            .map(|(idx, assembled)| {
                (
                    idx,
                    factor_front(
                        assembled,
                        block_size,
                        epsilon,
                        pivot_strategy,
                        panel_kind,
                        low_rank_precision,
                        update_order,
                        parallel_front_threshold,
                        pivot_tol_rel,
                    ),
                )
            })
            .collect();

        for (idx, result) in results {
            let f = result.map_err(|error| map_front_error(idx, error))?;
            // Delayed pivots with no parent to absorb them: the matrix is
            // numerically singular.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multifrontal_blr::blr::BlrPanel;
    use crate::multifrontal_blr::ordering::{AssemblyTree, FrontInfo};
    use crate::multifrontal_blr::panel::Panel;
    use crate::multifrontal_blr::solve::forward_backward_solve;

    #[test]
    fn structural_symmetry_prepass() {
        // Symmetric tridiagonal: detected, so the incoming-bucket path is
        // skipped downstream.
        let mut dense = DenseMatrix::<f64>::zeros(4, 4);
        for i in 0..4 {
            dense[[i, i]] = 2.0;
            if i + 1 < 4 {
                dense[[i, i + 1]] = -1.0;
                dense[[i + 1, i]] = -1.0;
            }
        }
        assert!(is_structurally_symmetric(&CsrMatrix::from_dense(
            &dense, 1e-15
        )));
        // One missing transpose entry: structurally nonsymmetric.
        dense[[3, 2]] = 0.0;
        assert!(!is_structurally_symmetric(&CsrMatrix::from_dense(
            &dense, 1e-15
        )));
    }

    /// Matching graph: `k` disjoint pairs `(v_i, u_i)` with coupling 1 and
    /// diagonal 2 on `u_i`; `v_i` has *no* diagonal and no coupling to any
    /// other `v`. Elim order puts all `v`s in `[0, k)` and all `u`s in
    /// `[k, 2k)`, so the leaf front owning the `v`s has an entirely zero
    /// F11 block — no acceptable pivot at any step — while the parent can
    /// absorb the postponed pivots (each delayed `v_i` couples to `u_i`).
    /// This is a small saddle-point-style indefinite matrix; it previously
    /// aborted with `SingularFront`.
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

    /// Two-front tree: front 0 owns the `v`s (zero F11), front 1 (root)
    /// owns the `u`s and absorbs the delayed pivots.
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

    fn rel_residual(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut res_sq = 0.0;
        let mut b_sq = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                s += matrix.values[idx] * x[matrix.col_indices[idx]];
            }
            res_sq += (b[i] - s) * (b[i] - s);
            b_sq += b[i] * b[i];
        }
        (res_sq / b_sq).sqrt()
    }

    fn factorise_matching(
        k: usize,
        drop_first_coupling: bool,
        pivot_strategy: PivotStrategy,
    ) -> Result<Vec<FactoredFront<f64>>, BlrError> {
        let m = matching_matrix(k, drop_first_coupling);
        let tree = two_front_tree(k);
        numeric_factorise(
            &m,
            &tree,
            8,
            0.0,
            pivot_strategy,
            PanelKind::Blr,
            LowRankPrecision::Full,
            UpdateOrder::RightLook,
            0,
            1.0,
        )
    }

    #[test]
    fn delayed_pivots_absorbed_by_parent() {
        for strategy in [PivotStrategy::Partial, PivotStrategy::Rook] {
            let k = 4;
            let n = 2 * k;
            let m = matching_matrix(k, false);
            let tree = two_front_tree(k);
            let fronts = numeric_factorise(
                &m,
                &tree,
                8,
                0.0,
                strategy,
                PanelKind::Blr,
                LowRankPrecision::Full,
                UpdateOrder::RightLook,
                0,
                1.0,
            )
            .unwrap_or_else(|idx| panic!("factorise failed at front {idx} ({strategy:?})"));

            // The leaf front eliminated nothing: its whole F11 block is
            // zero, so all 4 pivots were postponed.
            assert_eq!(fronts[0].nfs, 0, "{strategy:?}: leaf eliminated pivots");
            assert_eq!(fronts[0].ndelayed, k, "{strategy:?}: wrong delayed count");
            assert_eq!(fronts[0].delayed_row_indices, vec![0, 1, 2, 3]);
            // The root absorbed the delayed set and eliminated everything.
            assert_eq!(fronts[1].nfs, n, "{strategy:?}: root did not absorb");
            assert_eq!(fronts[1].ndelayed, 0);

            // The factorisation must actually solve the system.
            let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
            let x = forward_backward_solve(&tree, &fronts, &b).unwrap();
            let res = rel_residual(&m, &x, &b);
            assert!(
                res < 1e-10,
                "{strategy:?}: residual {res} too large after delayed pivoting"
            );
        }
    }

    #[test]
    fn delayed_pivots_propagate_through_left_look_and_compression() {
        // Same matching problem with BLR compression enabled and the
        // LeftLook update order: the delayed payload is dense and immune
        // to compression, but the surrounding panels go through the
        // compressed path.
        let k = 4;
        let n = 2 * k;
        let m = matching_matrix(k, false);
        let tree = two_front_tree(k);
        let fronts = numeric_factorise(
            &m,
            &tree,
            4,
            1e-12,
            PivotStrategy::Partial,
            PanelKind::Blr,
            LowRankPrecision::Full,
            UpdateOrder::LeftLook,
            0,
            1.0,
        )
        .expect("factorise with LeftLook");
        assert_eq!(fronts[0].ndelayed, k);
        assert_eq!(fronts[1].ndelayed, 0);

        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).cos()).collect();
        let x = forward_backward_solve(&tree, &fronts, &b).unwrap();
        let res = rel_residual(&m, &x, &b);
        assert!(res < 1e-9, "LeftLook residual {res} too large");
    }

    #[test]
    fn singular_root_still_fails() {
        // Drop the v_0—u_0 coupling: row/column v_0 is entirely zero, so
        // the matrix is genuinely singular. The leaf postpones v_0 as
        // before, but the root cannot pivot on it either — only now must
        // the factorisation fail.
        match factorise_matching(4, true, PivotStrategy::Partial) {
            Err(BlrError::SingularFront(front)) => {
                assert_eq!(front, 1, "failure must be reported at the root front")
            }
            Err(error) => panic!("unexpected factorization error: {error}"),
            Ok(_) => panic!("singular matrix must fail at the root front"),
        }
    }

    #[test]
    fn extend_add_panel_matches_materialized_path() {
        // Mixed 4x4 panel, bs=2: Dense | rank-0 LowRank (zeros);
        // rank-1 LowRank | Dense. Map permutes rows/cols.
        let dense_tl = DenseMatrix::from_fn(2, 2, |i, j| (i * 2 + j) as f64 + 1.0);
        let dense_br = DenseMatrix::from_fn(2, 2, |i, j| -((i * 2 + j) as f64) - 0.5);
        let u = DenseMatrix::from_fn(2, 1, |i, _| (i + 1) as f64);
        let v = DenseMatrix::from_fn(2, 1, |j, _| (j + 3) as f64);
        let panel = Panel::Blr(BlrPanel {
            n_rows: 4,
            n_cols: 4,
            bs: 2,
            blocks: vec![
                vec![
                    BlrBlock::Dense(dense_tl),
                    BlrBlock::LowRank {
                        u: DenseMatrix::zeros(2, 0),
                        v: DenseMatrix::zeros(2, 0),
                    },
                ],
                vec![BlrBlock::LowRank { u, v }, BlrBlock::Dense(dense_br)],
            ],
        });
        let map = vec![2usize, 0, 3, 1];
        // Reference: the old materialize-then-add path.
        let reference = panel.to_dense();
        let mut want = DenseMatrix::<f64>::zeros(4, 4);
        for (i, &r) in map.iter().enumerate() {
            for (j, &c) in map.iter().enumerate() {
                want[[r, c]] += reference[[i, j]];
            }
        }
        // Candidate, starting from nonzero fill so skips are observable.
        let mut got = DenseMatrix::from_fn(4, 4, |i, j| (i * 4 + j) as f64 * 0.25);
        let mut want_shifted = want.clone();
        for i in 0..4 {
            for j in 0..4 {
                want_shifted[[i, j]] += (i * 4 + j) as f64 * 0.25;
            }
        }
        extend_add_panel(&panel, &map, &mut got);
        assert_eq!(
            got.as_slice(),
            want_shifted.as_slice(),
            "extend_add must match materialized extend-add bit-for-bit"
        );
        // -0.0 disclosure: a rank-0-skipped cell holding -0.0 keeps -0.0
        // instead of becoming +0.0 (-0.0 + 0.0). Numerically identical
        // (-0.0 == +0.0, same norms/residuals), not bitwise identical.
        // Panel cell (0, 2) sits in the rank-0 block and maps to
        // matrix cell (map[0], map[2]).
        let mut got_neg = DenseMatrix::<f64>::zeros(4, 4);
        got_neg[[map[0], map[2]]] = -0.0;
        extend_add_panel(&panel, &map, &mut got_neg);
        assert_eq!(got_neg[[map[0], map[2]]], 0.0);
        assert!(got_neg[[map[0], map[2]]].is_sign_negative());
    }

    #[test]
    fn malformed_child_index_is_reported_without_panicking() {
        let matrix = CsrMatrix::identity(1);
        let zero_panel = DenseMatrix::<f64>::zeros(1, 1);
        let child = FactoredFront {
            nfs: 0,
            ncb: 1,
            ndelayed: 0,
            fs_elim_indices: Vec::new(),
            f11_lu: DenseMatrix::zeros(0, 0),
            pivot_kinds: Vec::new(),
            pivots: Vec::new(),
            col_pivots: Vec::new(),
            l21: Panel::from_dense(&zero_panel, PanelKind::Blr, 1, 0.0),
            u12: Panel::from_dense(&zero_panel, PanelKind::Blr, 1, 0.0),
            cb_indices: vec![1],
            cb_panel: Panel::from_dense(&zero_panel, PanelKind::Blr, 1, 0.0),
            delayed_row_indices: Vec::new(),
            delayed_col_indices: Vec::new(),
            delayed: None,
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assemble_front(&matrix, 0, 1, Vec::new(), &[&child], 7)
        }))
        .expect("malformed assembly must not panic");
        assert!(matches!(
            result,
            Err(BlrError::InvalidFactorization { front: 7, .. })
        ));
    }
}
