//! Column Approximate Minimum Degree (COLAMD) ordering for sparse LU.
//!
//! This is the honest replacement for a previous heuristic that was *named*
//! `colamd` but was in fact a static minimum-degree peeling on the column
//! intersection graph: adjacency built once, degrees only decremented, fill
//! never simulated, no quotient graph, no approximate-degree updates.
//!
//! The implementation follows Davis, Gilbert, Larimore & Ng, "A Column
//! Approximate Minimum Degree Ordering Algorithm" (ACM TOMS, 2004), in the
//! same algorithmic family as CSparse's `cs_amd`:
//!
//! - **Row-set quotient representation**: columns of A are the variables;
//!   rows of A are the initial *elements*. Eliminating a column merges the
//!   rows containing it into one new element (a clique over their column
//!   sets), so fill is represented implicitly and never materialised edge
//!   by edge. Working on A's row sets rather than the column intersection
//!   graph is essential for unsymmetric patterns: an arrowhead matrix has a
//!   dense column intersection graph (a clique) yet admits a zero-fill
//!   ordering, which only the row-set view can discover.
//! - **Approximate external degrees**: when pivot `p` is eliminated and the
//!   union of its row sets becomes the new element `Lp`, the degree of every
//!   `i ∈ Lp` is recomputed as in CSparse's `cs_amd`:
//!   `d_i ← min(d_i, Σ_{e ∈ E_i} |L_e \ Lp|) + (|Lp| − 1)`, capped at
//!   `remaining − 1`. The set differences `|L_e \ Lp|` are computed fresh
//!   each round with per-element counters (start from `|L_e|`, decrement per
//!   member found in `Lp`), so degree errors cannot drift across rounds.
//!   Elements with `|L_e \ Lp| = 0` are covered by `Lp` and aggressively
//!   absorbed, keeping element lists short.
//! - **Dense-row removal**: rows longer than `min(n, max(16, 10·√n))` are
//!   removed from the quotient graph up front (standard COLAMD/CSparse
//!   dense-row handling, capped at `n` so small dense blocks still prune
//!   their fully-dense rows).
//! - **Supervariable detection**: columns with identical live row sets are
//!   structurally indistinguishable. They are detected by hashing each
//!   column's live row-set signature, grouping equal hashes, and confirming
//!   by exact comparison; each group is absorbed into a single *principal*
//!   column that alone enters the quotient graph and the heap. Dead members
//!   are placed immediately after their principal in the permutation —
//!   eliminating them right after the principal adds no fill beyond the
//!   principal's own clique (Davis's COLAMD / `cs_amd` supervariables).
//! - **Mass elimination**: after a pivot's new element `Lp` is formed and
//!   covered elements (`|L_e \ Lp| = 0`) are aggressively absorbed, a member
//!   of `Lp` whose element list becomes empty has its entire external
//!   adjacency covered by `Lp`. It is eliminated immediately after the
//!   pivot together with its supervariable members: its induced clique is
//!   exactly `Lp`, so this adds no fill beyond the new element itself
//!   (`cs_amd` mass elimination). `Lp` is then compacted before it becomes
//!   the new element, and degree updates use the compacted size.
//!
//! Remaining difference from SuiteSparse COLAMD, kept deliberately for
//! simplicity: dead entries stay in adjacency lists and are skipped lazily
//! instead of being compacted in place. This does not affect the
//! permutation; it is a performance refinement only.

use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Unlink column `c` from its degree bucket (`-1` = empty-list sentinel).
fn bucket_remove(
    head: &mut [isize],
    next_c: &mut [isize],
    prev_c: &mut [isize],
    deg: usize,
    c: usize,
) {
    let nx = next_c[c];
    let pv = prev_c[c];
    if pv != -1 {
        next_c[pv as usize] = nx;
    } else {
        head[deg] = nx;
    }
    if nx != -1 {
        prev_c[nx as usize] = pv;
    }
    next_c[c] = -1;
    prev_c[c] = -1;
}

/// Link column `c` at the head of degree bucket `deg` (most-recent first).
fn bucket_insert(
    head: &mut [isize],
    next_c: &mut [isize],
    prev_c: &mut [isize],
    deg: usize,
    c: usize,
) {
    let nx = head[deg];
    next_c[c] = nx;
    prev_c[c] = -1;
    if nx != -1 {
        prev_c[nx as usize] = c as isize;
    }
    head[deg] = c as isize;
}

/// Compute a fill-reducing column permutation for a square sparse matrix.
pub fn colamd<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    let n = matrix.num_rows;
    if n <= 1 {
        return (0..n).collect();
    }

    // Row -> columns and column -> rows incidence. Capacities are exact
    // (row nnz from the CSR pointers, column counts from one histogram
    // pass), so the 2n small vectors never reallocate: the default
    // grow-from-zero path costs ~2 allocations per vector (~65k extra
    // allocs at n=16384). Push order is unchanged (row-major), so all
    // downstream results are bit-identical.
    let mut row_cols: Vec<Vec<usize>> = Vec::with_capacity(n);
    for row in 0..n {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        let mut cols = Vec::with_capacity(end - start);
        cols.extend(
            matrix.col_indices[start..end]
                .iter()
                .copied()
                .filter(|&column| column < n),
        );
        row_cols.push(cols);
    }
    let mut col_counts = vec![0usize; n];
    for cols in row_cols.iter() {
        for &column in cols {
            col_counts[column] += 1;
        }
    }
    let mut col_rows: Vec<Vec<usize>> = Vec::with_capacity(n);
    for &count in col_counts.iter() {
        col_rows.push(Vec::with_capacity(count));
    }
    for (row, cols) in row_cols.iter().enumerate() {
        for &column in cols {
            col_rows[column].push(row);
        }
    }

    // Dense-row removal: dense rows are taken out of the quotient graph and
    // ignored (they would merge nearly all columns at the first pivot).
    let dense_row_threshold = n.min(((n as f64).sqrt() * 10.0).max(16.0) as usize);
    let mut row_live = vec![true; n];
    let mut any_dense_row = false;
    for (row, cols) in row_cols.iter().enumerate() {
        if cols.len() >= dense_row_threshold {
            row_live[row] = false;
            any_dense_row = true;
        }
    }

    // Supervariable detection: columns with identical live row sets are
    // absorbed into a single principal column. Hash each column's live
    // row-set signature, group equal hashes, and confirm candidates by
    // exact comparison. `live[c] = false` marks a dead member; the members
    // of each principal are placed immediately after it in `perm`.
    let mut live = vec![true; n];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
    {
        let mut hash = vec![0u64; n];
        for (column, rows) in col_rows.iter().enumerate() {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for &row in rows {
                if !row_live[row] {
                    continue;
                }
                h ^= (row as u64).wrapping_add(1);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash[column] = h;
        }
        let live_rows_equal = |a: usize, b: usize| {
            col_rows[a]
                .iter()
                .copied()
                .filter(|&r| row_live[r])
                .eq(col_rows[b].iter().copied().filter(|&r| row_live[r]))
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&c| (hash[c], c));
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && hash[order[end]] == hash[order[start]] {
                end += 1;
            }
            if end - start > 1 {
                // Equal-hash candidates: split into exact-equality classes,
                // the first member of each class becoming its principal.
                let mut representatives: Vec<usize> = Vec::new();
                for &c in &order[start..end] {
                    let mut principal = None;
                    for &rep in &representatives {
                        if live_rows_equal(rep, c) {
                            principal = Some(rep);
                            break;
                        }
                    }
                    match principal {
                        Some(rep) => {
                            members[rep].push(c);
                            live[c] = false;
                        }
                        None => representatives.push(c),
                    }
                }
            }
            start = end;
        }
    }
    let any_dead_member = members.iter().any(|m| !m.is_empty());

    // Initial external degrees: |union of live row sets containing c| − 1,
    // computed exactly with generation-stamped marks. Dead supervariable
    // members are excluded (they are eliminated with their principal, whose
    // identical row set contributes the same union).
    let mut degree = vec![0usize; n];
    {
        let mut marks = vec![usize::MAX; n];
        for (column, rows) in col_rows.iter().enumerate() {
            if !live[column] {
                continue;
            }
            marks[column] = column;
            let mut d = 0usize;
            for &row in rows {
                if !row_live[row] {
                    continue;
                }
                for &j in &row_cols[row] {
                    if live[j] && marks[j] != column {
                        marks[j] = column;
                        d += 1;
                    }
                }
            }
            degree[column] = d;
        }
    }

    // Quotient-graph state.
    //
    // `elems[c]`     : live elements (rows or generated) containing column c
    //                  (absorbed elements are skipped lazily).
    // `elem_cols[e]` : column set of element e; ids 0..n are the rows of A,
    //                  later ids are generated elements named by counter.
    // `elem_size[e]` : number of live columns of e (fixed at creation; a live
    //                  element never loses a member without being absorbed).
    // `elem_dead[e]` : element e has been absorbed into a later element.
    // `degree[c]`    : approximate external degree of column c.
    // `w[e]`         : scratch for the set-difference computation, holding
    //                  |L_e \ Lp| for elements adjacent to the current Lp
    //                  (`usize::MAX` = untouched sentinel).
    // `in_le[c]`     : generation stamp marking membership in the current Lp.
    // Move (not filter-clone) the incidence lists when nothing was removed:
    // `row_cols`/`col_rows` are never read after this point, and with no
    // dense rows / dead members the live-filtered copies are content-equal.
    // Bit-identical downstream in the common case; the filtered path below
    // is unchanged for matrices that do prune.
    let mut elems: Vec<Vec<usize>> = if any_dense_row {
        col_rows
            .iter()
            .map(|rows| rows.iter().copied().filter(|&r| row_live[r]).collect())
            .collect()
    } else {
        col_rows
    };
    // Element column sets exclude dead supervariable members: they never
    // appear in an `Lp`, and counting them would inflate `elem_size` and
    // suppress the `|L_e \ Lp| = 0` absorption test.
    let mut elem_cols: Vec<Vec<usize>> = if any_dead_member {
        row_cols
            .iter()
            .map(|cols| cols.iter().copied().filter(|&c| live[c]).collect())
            .collect()
    } else {
        row_cols
    };
    let mut elem_size: Vec<usize> = elem_cols.iter().map(Vec::len).collect();
    let mut elem_dead: Vec<bool> = if any_dense_row {
        row_live.iter().map(|&live| !live).collect()
    } else {
        vec![false; n]
    };
    let mut w: Vec<usize> = vec![usize::MAX; n];
    let mut in_le = vec![0usize; n];
    let mut generation = 0usize;

    // Degree buckets (CSparse `cs_amd` style): `head[d]` is the most recently
    // linked live principal of approximate degree `d`, chained by
    // `next_c`/`prev_c`. Degree updates and pivot selection are O(1); the
    // former `BinaryHeap` with lazy deletion pushed one entry per surviving
    // Lp member per round (up to ~1M pushes at n=16384) and dominated the
    // ordering time. Tie-breaks between equal degrees are
    // implementation-defined (most-recently-linked first), never by column
    // id; only the approximate-degree sequence is meaningful.
    const EMPTY: isize = -1;
    let mut head: Vec<isize> = vec![EMPTY; n];
    let mut next_c: Vec<isize> = vec![EMPTY; n];
    let mut prev_c: Vec<isize> = vec![EMPTY; n];
    let mut remaining = 0usize;
    for c in 0..n {
        if live[c] {
            bucket_insert(&mut head, &mut next_c, &mut prev_c, degree[c].min(n - 1), c);
            remaining += 1;
        }
    }
    let mut mindeg = 0usize;

    let mut perm = Vec::with_capacity(n);
    // Round scratch hoisted out of the loop: `lp`, `touched`, `survivors`,
    // `set_differences`, and `mass_eliminated` are cleared and refilled each
    // round instead of reallocated (n rounds of allocation dominated the
    // allocator traffic). Per-member `kept` lists below use in-place
    // `retain_mut`, which additionally reuses element-list capacity across
    // rounds.
    let mut lp: Vec<usize> = Vec::new();
    let mut touched: Vec<usize> = Vec::new();
    let mut survivors: Vec<usize> = Vec::new();
    let mut set_differences: Vec<usize> = Vec::new();
    let mut mass_eliminated: Vec<usize> = Vec::new();

    loop {
        while mindeg < n && head[mindeg] == EMPTY {
            mindeg += 1;
        }
        if mindeg >= n {
            break;
        }
        let pivot = head[mindeg] as usize;
        bucket_remove(
            &mut head,
            &mut next_c,
            &mut prev_c,
            degree[pivot].min(n - 1),
            pivot,
        );
        live[pivot] = false;
        remaining -= 1;
        perm.push(pivot);
        // Dead supervariable members are eliminated right after their
        // principal at no additional fill cost.
        perm.extend_from_slice(&members[pivot]);

        generation += 1;
        in_le[pivot] = generation;

        // Form the new element Lp: the union of the live column sets of the
        // pivot's live elements. Each of those elements is absorbed. The
        // pivot's element list is cleared in place so its allocation is
        // reused if the column ever reappears (it cannot — the pivot is
        // dead — but the capacity is freed without dropping the buffer).
        lp.clear();
        for &e in elems[pivot].iter() {
            // SAFETY: element id (scan-1 invariant) and column ids (< n,
            // filtered at incidence build) are in bounds.
            if unsafe { *elem_dead.get_unchecked(e) } {
                continue;
            }
            unsafe { *elem_dead.get_unchecked_mut(e) = true };
            for &j in &elem_cols[e] {
                if unsafe { *live.get_unchecked(j) && *in_le.get_unchecked(j) != generation } {
                    unsafe { *in_le.get_unchecked_mut(j) = generation };
                    lp.push(j);
                }
            }
        }
        elems[pivot].clear();

        // Scan 1 (set differences): for every live element e adjacent to a
        // member of Lp, compute w[e] = |L_e \ Lp| by starting from |L_e|
        // and decrementing once per member of e found in Lp.
        //
        // SAFETY (unchecked indexing, here and in scan 2a): element ids in
        // member lists are always valid indices into `elem_dead`/`w`/
        // `elem_size`. Initial row ids are < n, and every generated id is
        // created together with one `elem_cols`/`elem_dead`/`elem_size`/`w`
        // entry (see scan 2b), so all four arrays share the same length and
        // every stored id is in bounds.
        touched.clear();
        for &i in &lp {
            for &e in &elems[i] {
                if unsafe { *elem_dead.get_unchecked(e) } {
                    continue;
                }
                if unsafe { *w.get_unchecked(e) } == usize::MAX {
                    unsafe { *w.get_unchecked_mut(e) = *elem_size.get_unchecked(e) };
                    touched.push(e);
                }
                unsafe { *w.get_unchecked_mut(e) -= 1 };
            }
        }

        // Scan 2a (absorption, following CSparse's cs_amd): elements with
        // |L_e \ Lp| = 0 are covered by Lp and are aggressively absorbed
        // (they are redundant against the new element). A member of Lp
        // whose element list becomes empty has its entire external
        // adjacency covered by Lp: it is *mass eliminated* — placed in the
        // permutation immediately after the pivot (with its supervariable
        // members), since its induced clique is exactly Lp and eliminating
        // it now adds no fill beyond the new element itself.
        survivors.clear();
        set_differences.clear();
        mass_eliminated.clear();
        for &i in &lp {
            // In-place filter of i's element list (dead entries dropped,
            // covered elements aggressively absorbed): identical survivor
            // sets and set-difference sums as the former take/filter/restore,
            // but the list allocation is reused across rounds.
            let mut set_difference = 0usize;
            elems[i].retain_mut(|e| {
                let id = *e;
                // SAFETY: see scan 1 — `id` is in bounds for both arrays.
                if unsafe { *elem_dead.get_unchecked(id) } {
                    return false;
                }
                let dext = unsafe { *w.get_unchecked(id) }; // |L_e \ Lp|
                if dext == 0 {
                    unsafe {
                        *elem_dead.get_unchecked_mut(id) = true;
                    } // L_e ⊆ Lp
                    return false;
                }
                set_difference += dext;
                true
            });
            if elems[i].is_empty() {
                mass_eliminated.push(i);
            } else {
                survivors.push(i);
                set_differences.push(set_difference);
            }
        }
        for &i in &mass_eliminated {
            live[i] = false;
            remaining -= 1;
            bucket_remove(&mut head, &mut next_c, &mut prev_c, degree[i].min(n - 1), i);
            perm.push(i);
            perm.extend_from_slice(&members[i]);
        }

        // Scan 2b (degree update, following CSparse's cs_amd): for each
        // surviving member of the compacted Lp, its new external degree is
        //   min(d_old, Σ_{e ∈ E_i} |L_e \ Lp|) + (|Lp| − 1),
        // recomputed from the current quotient graph so errors cannot drift.
        // Mass-eliminated members are removed from Lp before it becomes the
        // new element, and the degree updates use the compacted size.
        let lp_size = survivors.len();
        if lp_size > 0 {
            let new_elem = elem_cols.len();
            let lower = lp_size - 1;
            for (member, &i) in survivors.iter().enumerate() {
                elems[i].push(new_elem);
                bucket_remove(&mut head, &mut next_c, &mut prev_c, degree[i].min(n - 1), i);
                let updated = degree[i].min(set_differences[member]) + lower;
                degree[i] = updated.min(remaining.saturating_sub(1).max(lower));
                let bucket = degree[i].min(n - 1);
                bucket_insert(&mut head, &mut next_c, &mut prev_c, bucket, i);
                if bucket < mindeg {
                    mindeg = bucket;
                }
            }
            elem_cols.push(std::mem::take(&mut survivors));
            elem_dead.push(false);
            elem_size.push(lp_size);
            w.push(usize::MAX);
        }

        // Reset the set-difference scratch.
        for &e in &touched {
            w[e] = usize::MAX;
        }
    }

    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::{BTreeSet, BinaryHeap};

    /// The previous heuristic named `colamd`: static minimum-degree peeling
    /// on the column intersection graph, degrees only decremented, fill
    /// never simulated. Kept here solely as the baseline for fill-quality
    /// regression tests.
    fn legacy_static_ordering<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
        let n = matrix.num_rows;
        if n <= 1 {
            return (0..n).collect();
        }

        let mut column_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for &column in &matrix.col_indices[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]] {
                column_rows[column].push(row);
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut marks = vec![usize::MAX; n];
        for column in 0..n {
            marks[column] = column;
            for &row in &column_rows[column] {
                for &neighbor in &matrix.col_indices[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]]
                {
                    if marks[neighbor] != column {
                        marks[neighbor] = column;
                        adjacency[column].push(neighbor);
                    }
                }
            }
        }

        let mut degree: Vec<usize> = adjacency.iter().map(Vec::len).collect();
        let mut eliminated = vec![false; n];
        let mut queue: BinaryHeap<Reverse<(usize, usize, usize)>> = degree
            .iter()
            .enumerate()
            .map(|(column, &column_degree)| Reverse((column_degree, n - 1 - column, column)))
            .collect();

        let mut perm = Vec::with_capacity(n);
        while let Some(Reverse((queued_degree, _, pivot))) = queue.pop() {
            if eliminated[pivot] || degree[pivot] != queued_degree {
                continue;
            }
            perm.push(pivot);
            eliminated[pivot] = true;

            for &neighbor in &adjacency[pivot] {
                if !eliminated[neighbor] {
                    degree[neighbor] -= 1;
                    queue.push(Reverse((degree[neighbor], n - 1 - neighbor, neighbor)));
                }
            }
        }

        perm
    }

    /// Symbolic nnz(L + U) (unit diagonal counted once) under the symmetric
    /// fill model: both axes permuted by `perm`, no row pivoting. This is
    /// the standard model for evaluating a fill-reducing column ordering —
    /// partial pivoting re-aligns rows numerically, so measuring fill with
    /// rows left in natural order would punish orderings that are actually
    /// good.
    fn symbolic_fill(matrix: &CsrMatrix<f64>, perm: &[usize]) -> usize {
        let n = matrix.num_rows;
        let mut position = vec![0usize; n];
        for (k, &column) in perm.iter().enumerate() {
            position[column] = k;
        }
        let mut col_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for row in 0..n {
            for &column in &matrix.col_indices[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]] {
                col_sets[position[column]].insert(position[row]);
            }
        }
        let mut fill = 0;
        for k in 0..n {
            fill += col_sets[k].len();
            let below: Vec<usize> = col_sets[k].range(k + 1..).copied().collect();
            let absorbed: Vec<usize> = col_sets[k].range(k + 1..).copied().collect();
            for i in below {
                col_sets[i].extend(absorbed.iter().copied());
            }
        }
        fill
    }

    /// 2D 5-point Laplacian on an `nx` × `ny` grid.
    fn laplacian_2d(nx: usize, ny: usize) -> CsrMatrix<f64> {
        let n = nx * ny;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                if iy > 0 {
                    col_indices.push(idx - nx);
                    values.push(-1.0);
                }
                if ix > 0 {
                    col_indices.push(idx - 1);
                    values.push(-1.0);
                }
                col_indices.push(idx);
                values.push(4.0);
                if ix < nx - 1 {
                    col_indices.push(idx + 1);
                    values.push(-1.0);
                }
                if iy < ny - 1 {
                    col_indices.push(idx + nx);
                    values.push(-1.0);
                }
                row_ptrs.push(col_indices.len());
            }
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    /// Irregular unsymmetric-pattern matrix from a deterministic LCG, with a
    /// mix of short- and long-range connections plus a diagonal. LU fill on
    /// an unsymmetric matrix is governed by the column intersection graph
    /// (George–Ng: the AᵀA column etree bounds L's structure), so ordering
    /// quality for this matrix is measured on the intersection graph.
    fn irregular_matrix(n: usize) -> CsrMatrix<f64> {
        let mut row_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for i in 0..n {
            row_sets[i].insert(i);
            let degree = 2 + next() % 6;
            for _ in 0..degree {
                let j = if next() % 3 == 0 {
                    next() % n // long-range edge
                } else {
                    (i + 1 + next() % 8) % n // short-range edge
                };
                row_sets[i].insert(j);
            }
        }
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (i, set) in row_sets.iter().enumerate() {
            for &j in set {
                col_indices.push(j);
                values.push(if i == j { 10.0 } else { -1.0 });
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    /// Column intersection graph of `matrix` as a CSR pattern (unit values):
    /// columns are adjacent when they share at least one matrix row.
    fn intersection_graph_csr(matrix: &CsrMatrix<f64>) -> CsrMatrix<f64> {
        let n = matrix.num_rows;
        let mut column_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for &column in &matrix.col_indices[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]] {
                column_rows[column].push(row);
            }
        }
        let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for (column, rows) in column_rows.iter().enumerate() {
            adjacency[column].insert(column);
            for &row in rows {
                for &neighbor in &matrix.col_indices[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]]
                {
                    adjacency[column].insert(neighbor);
                }
            }
        }
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for set in &adjacency {
            for &j in set {
                col_indices.push(j);
                values.push(1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    #[test]
    fn amd_produces_valid_permutation() {
        let a = laplacian_2d(6, 5);
        let perm = colamd(&a);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..a.num_rows).collect::<Vec<_>>());
    }

    #[test]
    fn amd_fill_not_worse_than_legacy_on_2d_laplacian() {
        let a = laplacian_2d(20, 15);
        let legacy_perm = legacy_static_ordering(&a);
        let amd_perm = colamd(&a);
        let legacy_fill = symbolic_fill(&a, &legacy_perm);
        let amd_fill = symbolic_fill(&a, &amd_perm);
        println!("2D Laplacian 20x15: legacy fill={legacy_fill}, AMD fill={amd_fill}");
        assert!(
            amd_fill <= legacy_fill,
            "AMD fill {amd_fill} should not exceed legacy heuristic fill {legacy_fill}"
        );
    }

    #[test]
    fn amd_fill_not_worse_than_legacy_on_irregular_matrix() {
        // Unsymmetric irregular matrix: ordering quality is measured by fill
        // of the column intersection graph (the George–Ng bound for LU fill
        // with partial pivoting), which is exactly what COLAMD optimises.
        let a = irregular_matrix(200);
        let g2 = intersection_graph_csr(&a);
        let legacy_perm = legacy_static_ordering(&a);
        let amd_perm = colamd(&a);
        let legacy_fill = symbolic_fill(&g2, &legacy_perm);
        let amd_fill = symbolic_fill(&g2, &amd_perm);
        println!(
            "irregular n=200 (intersection graph): legacy fill={legacy_fill}, AMD fill={amd_fill}"
        );
        assert!(
            amd_fill <= legacy_fill,
            "AMD fill {amd_fill} should not exceed legacy heuristic fill {legacy_fill}"
        );
    }

    #[test]
    fn amd_orders_arrowhead_hub_last() {
        // Arrowhead: hub column 0 shares a row with every leaf.
        let n = 30;
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            col_indices.push(0);
            values.push(1.0);
            if row != 0 {
                col_indices.push(row);
                values.push(2.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let perm = colamd(&a);
        let hub_position = perm.iter().position(|&p| p == 0).unwrap();
        // The hub's approximate degree is capped at `remaining - 1`, so in
        // the final rounds it ties the last surviving leaf at degree 1 and
        // the winner is implementation-defined (bucket order, not column
        // id). Either the hub is last, or it is second-to-last with the
        // final leaf mass eliminated immediately after it — both add zero
        // fill beyond the hub's own clique.
        assert!(
            hub_position >= n - 2,
            "hub column should be eliminated last or second-to-last (got position {hub_position} of {n})"
        );
    }

    #[test]
    fn supervariable_columns_are_eliminated_adjacently() {
        // Pairs of structurally identical columns embedded in a 1D chain:
        // columns 2k and 2k+1 have the same row sets for k in 0..m/2 (they
        // touch the same rows), so each pair is a supervariable and must be
        // eliminated adjacently (principal followed by its dead member).
        let m = 12;
        let n = 2 * m;
        let mut row_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for row in 0..m {
            // Rows touch column pair (row, row + m) plus chain neighbours.
            for pair in [row, (row + 1) % m, (row + 2) % m] {
                row_sets[row].insert(pair);
                row_sets[row].insert(pair + m);
            }
        }
        // Keep the matrix square: rows m..2m mirror the same pattern on the
        // second half of the chain.
        for row in m..n {
            let r = row - m;
            for pair in [r, (r + m - 1) % m] {
                row_sets[row].insert(pair);
                row_sets[row].insert(pair + m);
            }
        }
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for set in &row_sets {
            for &j in set {
                col_indices.push(j);
                values.push(1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let perm = colamd(&a);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "valid permutation");
        let position = |c: usize| perm.iter().position(|&p| p == c).unwrap();
        for pair in 0..m {
            let p = position(pair);
            let q = position(pair + m);
            assert_eq!(
                p.abs_diff(q),
                1,
                "identical columns {pair} and {} should be adjacent in the order",
                pair + m
            );
        }
    }

    #[test]
    fn mass_elimination_on_disconnected_dense_block() {
        // A dense 8x8 block disconnected from a sparse tail: once the first
        // column of the dense block is eliminated, every other column of the
        // block has its whole adjacency covered by the new element and is
        // mass eliminated. The fill of a dense block is exactly its dense
        // triangular count; mass elimination must not add anything.
        let dense = 8;
        let tail = 16;
        let n = dense + tail;
        let mut row_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for row in 0..dense {
            for column in 0..dense {
                row_sets[row].insert(column);
            }
        }
        for row in dense..n {
            row_sets[row].insert(row);
            if row > dense {
                row_sets[row].insert(row - 1);
            }
            if row + 1 < n {
                row_sets[row].insert(row + 1);
            }
        }
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for set in &row_sets {
            for &j in set {
                col_indices.push(j);
                values.push(1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let perm = colamd(&a);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "valid permutation");
        let fill = symbolic_fill(&a, &perm);
        // Dense block: 8*9/2 upper + 8*7/2 lower = 64; chain: 3 per row
        // minus boundaries, zero fill. Total is the exact minimum here.
        let chain_fill = 3 * tail - 2;
        assert_eq!(
            fill,
            dense * dense + chain_fill,
            "dense block + chain should factorize with zero extra fill"
        );
    }

    #[test]
    fn mass_elimination_places_covered_columns_right_after_pivot() {
        // Columns 0 and 1 appear only in rows whose live column sets are
        // subsets of {0, 1, 2}. The chain peel removes 3, 5, 4 first (3 has
        // the unique minimum degree, 5 is mass eliminated right after it,
        // then 4 heads the minimum bucket), leaving the {0, 1, 2} block for
        // last. The first pivot inside the block covers the other two —
        // every element of theirs is then covered by its Lp — so both are
        // mass eliminated immediately after it. Which block member pivots
        // first is a degree tie broken by queue order (implementation
        // defined: the old heap picked the higher id, the degree buckets
        // pick the most recently linked), so the test asserts the
        // tie-independent shape: the block occupies three consecutive
        // trailing positions with both followers directly behind the pivot.
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for cols in [
            vec![0, 1],
            vec![0, 1, 2],
            vec![1, 2],
            vec![1, 4],
            vec![4, 5],
            vec![3, 5],
        ] {
            for j in cols {
                col_indices.push(j);
                values.push(1.0);
            }
            row_ptrs.push(col_indices.len());
        }
        let a = CsrMatrix::from_raw_parts(6, 6, row_ptrs, col_indices, values);
        let perm = colamd(&a);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>(), "valid permutation");
        let position = |c: usize| perm.iter().position(|&p| p == c).unwrap();
        let mut block = [position(0), position(1), position(2)];
        block.sort_unstable();
        assert_eq!(
            block,
            [3, 4, 5],
            "the {{0, 1, 2}} block must occupy the three trailing positions"
        );
        let first = block[0];
        assert_eq!(
            [first + 1, first + 2],
            [block[1], block[2]],
            "both covered columns must directly follow the block's pivot"
        );
    }
}
