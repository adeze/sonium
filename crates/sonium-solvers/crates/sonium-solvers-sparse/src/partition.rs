//! Graph partitioning helpers for nested dissection.
//!
//! Lifted from `preconditioners::ilu_hierarchical` so the H-ILU preconditioner
//! and the multifrontal-BLR direct solver can share one implementation. The
//! partitioner is matrix-pattern-only — values are read from the CSR but only
//! to skim the sparsity (`build_adjacency` ignores values entirely).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::ops::Range;

use crate::CsrMatrix;
use sonium_solvers_core::traits::{ComplexField, SolverError};

/// Row-pointer element types accepted by [`nnz_balanced_ranges`]
/// (`usize` for [`CsrMatrix`], `u32` for compact CSR).
pub trait RowPtrElem: Copy {
    fn to_usize(self) -> usize;
}

impl RowPtrElem for usize {
    #[inline]
    fn to_usize(self) -> usize {
        self
    }
}

impl RowPtrElem for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

/// Partition CSR rows into `nparts` contiguous ranges with ≈ equal nonzero
/// counts, via binary search over the `row_ptrs` prefix sums.
///
/// This is the load-balancing strategy used by PETSc/MKL for parallel SpMV:
/// an equal-*rows* split is straggler-bound on skewed matrices (row lengths
/// varying by orders of magnitude), whereas an nnz-balanced split gives every
/// thread the same amount of multiply-accumulate work.
///
/// Returns exactly `nparts` ranges (some may be empty when a single row is
/// heavier than the per-part target). `nparts == 0` yields no ranges.
pub fn nnz_balanced_ranges<I: RowPtrElem>(row_ptrs: &[I], nparts: usize) -> Vec<Range<usize>> {
    let nrows = row_ptrs.len().saturating_sub(1);
    if nparts == 0 || nrows == 0 {
        return Vec::new();
    }
    let total_nnz = row_ptrs[nrows].to_usize();
    let mut ranges = Vec::with_capacity(nparts);
    let mut row_lo = 0;
    for part in 1..nparts {
        let target = total_nnz * part / nparts;
        // First row whose cumulative nnz reaches/exceeds the target. Ties
        // (multiple rows with equal cumulative nnz, e.g. empty rows) resolve
        // to the lowest row, keeping ranges non-overlapping and maximal.
        let row_hi = row_ptrs.partition_point(|&p| p.to_usize() < target);
        ranges.push(row_lo..row_hi.max(row_lo));
        row_lo = row_hi.max(row_lo);
    }
    ranges.push(row_lo..nrows);
    ranges
}

/// Node in the nested-dissection tree.
///
/// Leaves carry the DOF list for a domain; interior nodes carry the separator
/// that bisects their two children. The tree is post-ordered by construction
/// (children pushed before parents).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DissectionNode {
    /// Global DOF indices owned by this node.
    pub global_indices: Vec<usize>,
    /// Left child index in the flat `nodes` vector (None at leaves).
    pub left: Option<usize>,
    /// Right child index in the flat `nodes` vector (None at leaves).
    pub right: Option<usize>,
    /// Level in the tree (0 = leaves; root has the maximum level).
    pub level: usize,
}

/// Build symmetric adjacency list from CSR sparsity pattern.
///
/// Self-loops are dropped; off-diagonal entries are recorded as neighbours of
/// the row index. Caller-provided symmetry is assumed (no transposition).
#[allow(clippy::needless_range_loop)]
pub fn build_adjacency<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if j != i && j < n {
                adj[i].push(j);
                // Symmetrize: nested dissection / FM refinement operate on the
                // undirected graph. Inputs storing only one triangle (or a
                // structurally nonsymmetric pattern) must still yield a
                // symmetric adjacency, otherwise per-vertex gain bookkeeping
                // in FM drifts out of range.
                adj[j].push(i);
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    adj
}

/// Find a pseudo-peripheral node via two BFS passes.
fn pseudo_peripheral(adjacency: &[Vec<usize>], indices: &[usize]) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let mut in_set = vec![false; adjacency.len()];
    for &i in indices {
        in_set[i] = true;
    }
    let start = indices[0];
    let far = bfs_farthest(adjacency, start, &in_set);
    bfs_farthest(adjacency, far, &in_set)
}

fn bfs_farthest(adjacency: &[Vec<usize>], start: usize, in_set: &[bool]) -> usize {
    let mut visited = vec![false; adjacency.len()];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    let mut last = start;
    while let Some(node) = queue.pop_front() {
        last = node;
        for &nbr in &adjacency[node] {
            if in_set[nbr] && !visited[nbr] {
                visited[nbr] = true;
                queue.push_back(nbr);
            }
        }
    }
    last
}

/// BFS-based bisection: split `indices` into `(left, right, separator)`.
///
/// Runs BFS from a pseudo-peripheral node; the first half of the BFS ordering
/// goes left, the rest right. Separator = nodes with at least one neighbour on
/// the opposite side.
pub fn bfs_bisect(
    adjacency: &[Vec<usize>],
    indices: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = indices.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut in_set = vec![false; adjacency.len()];
    for &i in indices {
        in_set[i] = true;
    }

    let start = pseudo_peripheral(adjacency, indices);
    let mut visited = vec![false; adjacency.len()];
    let mut order = Vec::with_capacity(n);
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &nbr in &adjacency[node] {
            if in_set[nbr] && !visited[nbr] {
                visited[nbr] = true;
                queue.push_back(nbr);
            }
        }
    }
    for &i in indices {
        if !visited[i] {
            order.push(i);
        }
    }

    let mid = n / 2;
    let mut side = vec![0u8; adjacency.len()];
    for &node in &order[..mid] {
        side[node] = 1;
    }
    for &node in &order[mid..] {
        side[node] = 2;
    }

    let mut is_separator = vec![false; adjacency.len()];
    for &node in indices {
        let my_side = side[node];
        for &nbr in &adjacency[node] {
            if in_set[nbr] && side[nbr] != 0 && side[nbr] != my_side {
                is_separator[node] = true;
                is_separator[nbr] = true;
            }
        }
    }

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut separator = Vec::new();
    for &node in indices {
        if is_separator[node] {
            separator.push(node);
        } else if side[node] == 1 {
            left.push(node);
        } else {
            right.push(node);
        }
    }

    (left, right, separator)
}

/// Bisection algorithm used at each level of nested dissection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BisectionMethod {
    /// Legacy BFS level-set bisection. Kept for A/B comparison and as a
    /// fallback for degenerate subgraphs.
    Bfs,
    /// METIS-style multilevel bisection: heavy-edge-matching coarsening,
    /// BFS initial bisection at the coarsest level, and Fiduccia-Mattheyses
    /// refinement at each uncoarsening level. Deterministic.
    #[default]
    Multilevel,
}

/// Coarsening stops once the coarse graph has at most this many vertices.
const COARSEN_TO: usize = 200;
/// Stop coarsening when a round removes at most this fraction of vertices.
const COARSEN_STALL: f64 = 0.10;
/// Safety cap on coarsening depth for pathological graphs.
const MAX_COARSEN_LEVELS: usize = 20;
/// Heaviest allowed partition side, as a fraction of the total vertex weight.
const MAX_PART_FRACTION: f64 = 0.58;
/// Maximum FM passes per (un)coarsening level.
const MAX_FM_PASSES: usize = 8;
/// Graphs smaller than this skip FM polish (the coarsened initial bisection
/// stands): 82% of FM passes run on <128-vertex graphs where separators
/// are a few vertices and polish moves little cut mass.
const FM_MIN_REFINE: usize = 128;
/// Subgraphs at least this large are bisected with both nested-dissection
/// subtrees computed in parallel (rayon join).
// NOTE (threshold probe, reverted 2026-09-25): 2048 -> 512 kept the perm
// bit-identical but did not move recursion time (19.1 -> 18.7ms): deep
// tasks are too small for further fan-out. Keep 2048.
const ND_PARALLEL_MIN_SUBTREE: usize = 2048;

/// Edge-weighted graph used inside the multilevel pipeline.
#[derive(Debug, Clone)]
struct CoarseGraph {
    /// Adjacency as `(neighbour, weight)`, sorted by neighbour id.
    adj: Vec<Vec<(usize, i64)>>,
    /// Vertex weights (1 at the finest level).
    vwgt: Vec<i64>,
}

/// Restrict `adjacency` to the subgraph induced by `indices`, remapped to
/// dense local ids `0..indices.len()`. Duplicate neighbour entries are merged.
fn local_graph(adjacency: &[Vec<usize>], indices: &[usize]) -> CoarseGraph {
    let m = indices.len();
    let mut g2l = vec![usize::MAX; adjacency.len()];
    for (l, &g) in indices.iter().enumerate() {
        g2l[g] = l;
    }
    let mut adj = vec![Vec::new(); m];
    for (l, &g) in indices.iter().enumerate() {
        let mut nbrs: Vec<(usize, i64)> = Vec::with_capacity(adjacency[g].len());
        nbrs.extend(adjacency[g].iter().filter_map(|&h| {
            let lu = g2l[h];
            (lu != usize::MAX).then_some((lu, 1))
        }));
        nbrs.sort_unstable();
        nbrs.dedup();
        adj[l] = nbrs;
    }
    CoarseGraph {
        adj,
        vwgt: vec![1; m],
    }
}

/// One round of greedy heavy-edge matching. Returns the coarse graph and the
/// fine-to-coarse vertex map, or `None` when no edge could be matched.
///
/// Deterministic: vertices are visited in ascending id order; each unmatched
/// vertex is matched with its unmatched neighbour maximising
/// `(edge_weight, neighbour_degree, -neighbour_id)`.
fn coarsen_once(g: &CoarseGraph) -> Option<(CoarseGraph, Vec<usize>)> {
    let n = g.adj.len();
    let mut matched = vec![usize::MAX; n];
    for v in 0..n {
        if matched[v] != usize::MAX {
            continue;
        }
        let mut best = usize::MAX;
        let mut best_w = i64::MIN;
        let mut best_deg = 0usize;
        for &(u, w) in &g.adj[v] {
            if u == v || matched[u] != usize::MAX {
                continue;
            }
            let deg = g.adj[u].len();
            let better = best == usize::MAX
                || w > best_w
                || (w == best_w && (deg > best_deg || (deg == best_deg && u < best)));
            if better {
                best = u;
                best_w = w;
                best_deg = deg;
            }
        }
        if best != usize::MAX {
            matched[v] = best;
            matched[best] = v;
        }
    }

    let mut cmap = vec![usize::MAX; n];
    let mut nc = 0;
    for v in 0..n {
        if cmap[v] != usize::MAX {
            continue;
        }
        cmap[v] = nc;
        let u = matched[v];
        if u != usize::MAX {
            cmap[u] = nc;
        }
        nc += 1;
    }
    if nc == n {
        return None;
    }

    let mut vwgt = vec![0i64; nc];
    for v in 0..n {
        vwgt[cmap[v]] += g.vwgt[v];
    }

    // Aggregate parallel edges between coarse vertices per coarse vertex:
    // group fine vertices by coarse id (counting sort), then gather, sort
    // by neighbour, and merge runs within each group. Same sorted-by-
    // neighbour output as the former global (cv, cu) sort, but O(E + Σ
    // d·log d) over small groups instead of O(E·log E) — the global sort
    // dominated ordering time on large graphs.
    let mut counts = vec![0usize; nc];
    for &c in cmap.iter() {
        counts[c] += 1;
    }
    let mut starts = vec![0usize; nc + 1];
    for c in 0..nc {
        starts[c + 1] = starts[c] + counts[c];
    }
    let mut members = vec![0usize; n];
    let mut next = starts[..nc].to_vec();
    for (v, &c) in cmap.iter().enumerate() {
        members[next[c]] = v;
        next[c] += 1;
    }
    let mut adj: Vec<Vec<(usize, i64)>> = Vec::with_capacity(nc);
    let mut buf: Vec<(usize, i64)> = Vec::new();
    for c in 0..nc {
        buf.clear();
        for &v in &members[starts[c]..starts[c + 1]] {
            for &(u, w) in &g.adj[v] {
                let cu = cmap[u];
                if cu != c {
                    buf.push((cu, w));
                }
            }
        }
        buf.sort_unstable_by_key(|&(cu, _)| cu);
        let mut nbrs: Vec<(usize, i64)> = Vec::with_capacity(buf.len());
        for &(cu, w) in &buf {
            if let Some(last) = nbrs.last_mut()
                && last.0 == cu
            {
                last.1 += w;
            } else {
                nbrs.push((cu, w));
            }
        }
        adj.push(nbrs);
    }
    Some((CoarseGraph { adj, vwgt }, cmap))
}

/// Farthest vertex from `start` on a weighted coarse graph (BFS).
fn coarse_bfs_farthest(adj: &[Vec<(usize, i64)>], start: usize) -> usize {
    let mut visited = vec![false; adj.len()];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    let mut last = start;
    while let Some(v) = queue.pop_front() {
        last = v;
        for &(u, _) in &adj[v] {
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    last
}

/// Weighted BFS growing bisection at the coarsest level. Vertices are added
/// to side 0 in BFS order from a pseudo-peripheral node until side 0 reaches
/// half the total weight (never exceeding `limit`). Disconnected vertices are
/// swept up afterwards under the same rule.
fn initial_bisect(g: &CoarseGraph, limit: i64) -> Vec<u8> {
    let n = g.adj.len();
    let total: i64 = g.vwgt.iter().sum();
    let mut side = vec![1u8; n];
    if n == 0 {
        return side;
    }
    let far = coarse_bfs_farthest(&g.adj, 0);
    let start = coarse_bfs_farthest(&g.adj, far);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    let mut w0 = 0i64;
    while let Some(v) = queue.pop_front() {
        if w0 < total - w0 && w0 + g.vwgt[v] <= limit {
            side[v] = 0;
            w0 += g.vwgt[v];
        }
        for &(u, _) in &g.adj[v] {
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    for v in 0..n {
        if !visited[v] && w0 < total - w0 && w0 + g.vwgt[v] <= limit {
            side[v] = 0;
            w0 += g.vwgt[v];
        }
    }
    side
}

/// May vertex `v` move to the opposite side without worsening the balance?
/// Allowed when the destination stays within `limit`, or when the move
/// strictly reduces the part-weight imbalance (escape from an overweight side).
fn fm_move_allowed(side_v: u8, part_w: &[i64; 2], w: i64, limit: i64) -> bool {
    let from = side_v as usize;
    let to = 1 - from;
    let new_to = part_w[to] + w;
    if new_to <= limit {
        return true;
    }
    let new_from = part_w[from] - w;
    (part_w[to] - part_w[from]).abs() > (new_to - new_from).abs()
}

/// Boundary-only Fiduccia-Mattheyses refinement of a 2-way partition.
///
/// Each pass moves the highest-gain unlocked boundary vertex whose move is
/// balance-feasible (ties broken by ascending vertex id), locking moved
/// vertices, and rolls back to the best balanced prefix seen. Passes repeat
/// until no strict cut improvement or `MAX_FM_PASSES` is reached.
///
/// Candidate gains live in one lazy global `BinaryHeap<(gain, Reverse(id))>`
/// instead of per-gain bucket sets: no per-vertex tree node is ever
/// allocated (the heap array is reserved once per pass), removals are lazy
/// (stale entries are skipped by comparing against the live `gain[]`, which
/// is ground truth at pop time — including ABA gain excursions), and the pop
/// order (highest gain, then ascending id) reproduces the bucket scan
/// exactly, so the refinement stays deterministic. Balance-disallowed pops
/// wait in a side buffer and re-enter after each move, mirroring the old
/// re-examination of disallowed bucket members on every pick.
fn fm_refine(g: &CoarseGraph, side: &mut [u8]) {
    let n = g.adj.len();
    if n < FM_MIN_REFINE {
        return;
    }
    let total: i64 = g.vwgt.iter().sum();
    let limit = ((total as f64) * MAX_PART_FRACTION).ceil() as i64;
    let mut part_w = [0i64; 2];
    for (v, &s) in side.iter().enumerate() {
        part_w[s as usize] += g.vwgt[v];
    }

    for _ in 0..MAX_FM_PASSES {
        let mut gain = vec![0i64; n];
        let mut ext = vec![0i64; n];
        let mut cut = 0i64;
        // Lazy global candidate heap (highest gain, then ascending id —
        // the old bucket-scan order, without per-vertex tree nodes).
        let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
        for v in 0..n {
            let mut e = 0i64;
            let mut int = 0i64;
            for &(u, w) in &g.adj[v] {
                if side[u] != side[v] {
                    e += w;
                } else {
                    int += w;
                }
            }
            ext[v] = e;
            gain[v] = e - int;
            if e > 0 {
                cut += e;
                heap.push((gain[v], Reverse(v)));
            }
        }
        cut /= 2;

        let mut locked = vec![false; n];
        let mut moves: Vec<usize> = Vec::new();
        let mut best_idx = 0usize;
        let mut best_cut = cut;
        // Balance-disallowed pops wait here and re-enter the heap after
        // each move (partition weights may have shifted), mirroring the
        // old re-examination of disallowed bucket members on every pick.
        let mut waiting: Vec<(i64, Reverse<usize>)> = Vec::new();

        loop {
            // Highest-gain unlocked live boundary vertex whose move is
            // balance-feasible; ties broken by ascending vertex id. Stale
            // heap entries (superseded gains, locked or interior vertices)
            // are discarded: `gain[]`/`locked[]`/`ext[]` are ground truth.
            let mut pick = usize::MAX;
            while let Some(&(gv, Reverse(v))) = heap.peek() {
                if locked[v] || ext[v] == 0 || gain[v] != gv {
                    heap.pop();
                    continue;
                }
                if fm_move_allowed(side[v], &part_w, g.vwgt[v], limit) {
                    pick = v;
                    heap.pop();
                    break;
                }
                waiting.push((gv, Reverse(v)));
                heap.pop();
            }
            if pick == usize::MAX {
                break;
            }
            let v = pick;

            let from = side[v] as usize;
            side[v] = (1 - from) as u8;
            part_w[from] -= g.vwgt[v];
            part_w[1 - from] += g.vwgt[v];
            cut -= gain[v];
            locked[v] = true;
            for entry in waiting.drain(..) {
                heap.push(entry);
            }
            moves.push(v);
            if part_w[0].max(part_w[1]) <= limit && cut < best_cut {
                best_cut = cut;
                best_idx = moves.len();
            }

            // Incremental gain update: v crossed the cut, so each incident
            // edge flips internal<->external for the neighbour at its end.
            // The new gain is pushed lazily; the superseded heap entry (if
            // still queued) is skipped at pop time by the gain comparison.
            for &(u, w) in &g.adj[v] {
                if locked[u] {
                    continue;
                }
                if side[u] == from as u8 {
                    // Edge was internal to `from`; it now crosses the cut.
                    ext[u] += w;
                    gain[u] += 2 * w;
                } else {
                    // Edge crossed the cut; it is now internal.
                    ext[u] -= w;
                    gain[u] -= 2 * w;
                }
                if ext[u] > 0 {
                    heap.push((gain[u], Reverse(u)));
                }
            }
        }

        for &v in moves[best_idx..].iter().rev() {
            let from = side[v] as usize;
            side[v] = (1 - from) as u8;
            part_w[from] -= g.vwgt[v];
            part_w[1 - from] += g.vwgt[v];
        }
        if best_idx == 0 {
            break;
        }
    }
}

/// Full multilevel pipeline: coarsen, initial bisection at the coarsest
/// level, then uncoarsen with FM refinement at every level. Takes ownership
/// of the finest graph (avoiding a full clone) and returns it alongside the
/// side (0/1) of each of its vertices.
fn multilevel_partition(mut g: CoarseGraph) -> (CoarseGraph, Vec<u8>) {
    let mut stack: Vec<(CoarseGraph, Vec<usize>)> = Vec::new();
    while g.adj.len() > COARSEN_TO && stack.len() < MAX_COARSEN_LEVELS {
        match coarsen_once(&g) {
            Some((coarse, cmap)) => {
                let n_fine = g.adj.len();
                let stalled = (n_fine - coarse.adj.len()) as f64 <= COARSEN_STALL * n_fine as f64;
                stack.push((std::mem::replace(&mut g, coarse), cmap));
                if stalled {
                    break;
                }
            }
            None => break,
        }
    }

    let total: i64 = g.vwgt.iter().sum();
    let limit = ((total as f64) * MAX_PART_FRACTION).ceil() as i64;
    let mut side = initial_bisect(&g, limit);
    fm_refine(&g, &mut side);
    while let Some((finer, cmap)) = stack.pop() {
        let mut fine_side = vec![0u8; finer.adj.len()];
        for (v, &c) in cmap.iter().enumerate() {
            fine_side[v] = side[c];
        }
        fm_refine(&finer, &mut fine_side);
        side = fine_side;
        g = finer;
    }
    (g, side)
}

/// Turn a 2-way partition into `(left, right, separator)`: the separator is
/// the smaller of the two boundary sets; ties favour better balance, then the
/// side-0 boundary (deterministic).
fn extract_separator(g: &CoarseGraph, side: &[u8]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = g.adj.len();
    let mut crossing = vec![false; n];
    let mut b0 = 0usize;
    let mut b1 = 0usize;
    let mut s0 = 0usize;
    let mut s1 = 0usize;
    for v in 0..n {
        let crosses = g.adj[v].iter().any(|&(u, _)| side[u] != side[v]);
        crossing[v] = crosses;
        if side[v] == 0 {
            s0 += 1;
            if crosses {
                b0 += 1;
            }
        } else {
            s1 += 1;
            if crosses {
                b1 += 1;
            }
        }
    }
    // Candidate A: separator = side-0 boundary. Candidate B: side-1 boundary.
    //
    // NOTE (separator-trim attempt, reverted 2026-09-25): stripping each
    // boundary to minimality (move S vertices with no remainder-neighbours
    // across, ascending fixpoint, 0.55 balance guard) changed nothing on
    // the target grid — a TEMP census showed every separator already at
    // its cut minimum (root 128, halves 64, ...), and the UMFPACK-GIVEN
    // oracle stayed at 290M flops. The 5x gap to METIS-ND is tree-shape /
    // amalgamation (separator chaining across levels), not cut fatness.
    // Revisit only with unstructured-mesh evidence of fat boundaries.
    let a_max = (s0 - b0).max(s1);
    let b_max = s0.max(s1 - b1);
    let sep_is_b0 = b0 < b1 || (b0 == b1 && a_max <= b_max);

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut separator = Vec::new();
    for v in 0..n {
        let is_sep = crossing[v] && (sep_is_b0 == (side[v] == 0));
        if is_sep {
            separator.push(v);
        } else if side[v] == 0 {
            left.push(v);
        } else {
            right.push(v);
        }
    }
    (left, right, separator)
}

/// Multilevel bisection of the subgraph induced by `indices`.
///
/// METIS-style pipeline: heavy-edge-matching coarsening down to a few hundred
/// vertices, BFS initial bisection at the coarsest level, and
/// Fiduccia-Mattheyses refinement during uncoarsening (balance constraint
/// 55/45). Fully deterministic. Falls back to [`bfs_bisect`] when the
/// multilevel path yields a degenerate (empty) side.
pub fn multilevel_bisect(
    adjacency: &[Vec<usize>],
    indices: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // NOTE (measured): routing subgraphs < 256 (even < 64) to `bfs_bisect`
    // saves ordering time but costs +12% (+3%) fill — BFS level-set
    // separators aggregate badly across thousands of leaves. Rejected;
    // fill-neutrality rules ordering changes.
    let (left, right, separator) = if indices.len() < 2 {
        bfs_bisect(adjacency, indices)
    } else {
        let g = local_graph(adjacency, indices);
        let (g, side) = multilevel_partition(g);
        let (l, r, s) = extract_separator(&g, &side);
        (
            l.into_iter().map(|v| indices[v]).collect(),
            r.into_iter().map(|v| indices[v]).collect(),
            s.into_iter().map(|v| indices[v]).collect(),
        )
    };
    if !left.is_empty() && !right.is_empty() {
        (left, right, separator)
    } else {
        bfs_bisect(adjacency, indices)
    }
}

/// Recursive nested dissection. Returns the index of the root node in `nodes`.
///
/// Each separator is appended after its two subtrees, so `nodes` is in
/// post-order: a node's children always appear at smaller indices than the
/// node itself. The multifrontal assembly walk relies on this.
pub fn nested_dissection(
    adjacency: &[Vec<usize>],
    indices: &[usize],
    min_block_size: usize,
    max_levels: usize,
    current_level: usize,
    nodes: &mut Vec<DissectionNode>,
) -> usize {
    nested_dissection_with_method(
        adjacency,
        indices,
        min_block_size,
        max_levels,
        current_level,
        nodes,
        BisectionMethod::Multilevel,
    )
}

/// [`nested_dissection`] with an explicit per-level bisection algorithm.
///
/// `BisectionMethod::Multilevel` (the default used by [`nested_dissection`])
/// gives METIS-style separator quality; `BisectionMethod::Bfs` reproduces the
/// legacy level-set orderings for A/B comparison.
#[allow(clippy::too_many_arguments)]
pub fn nested_dissection_with_method(
    adjacency: &[Vec<usize>],
    indices: &[usize],
    min_block_size: usize,
    max_levels: usize,
    current_level: usize,
    nodes: &mut Vec<DissectionNode>,
    method: BisectionMethod,
) -> usize {
    if indices.len() <= min_block_size || current_level >= max_levels {
        let idx = nodes.len();
        nodes.push(DissectionNode {
            global_indices: indices.to_vec(),
            left: None,
            right: None,
            level: 0,
        });
        return idx;
    }

    let (left_indices, right_indices, separator) = match method {
        BisectionMethod::Bfs => bfs_bisect(adjacency, indices),
        BisectionMethod::Multilevel => multilevel_bisect(adjacency, indices),
    };
    if left_indices.is_empty() || right_indices.is_empty() {
        let idx = nodes.len();
        nodes.push(DissectionNode {
            global_indices: indices.to_vec(),
            left: None,
            right: None,
            level: 0,
        });
        return idx;
    }

    // The two subtrees are fully independent: bisect them in parallel when
    // the subgraph is large enough to amortize the rayon dispatch. Each
    // branch builds its own post-order node list; merging left-then-right
    // (with child links rebased) reproduces the sequential tree layout
    // bit-for-bit, so the ordering stays deterministic.
    let parallel = indices.len() >= ND_PARALLEL_MIN_SUBTREE && rayon::current_num_threads() > 1;
    let (left_node, right_node) = if parallel {
        let mut left_nodes = Vec::new();
        let mut right_nodes = Vec::new();
        let (left_local, right_local) = rayon::join(
            || {
                nested_dissection_with_method(
                    adjacency,
                    &left_indices,
                    min_block_size,
                    max_levels,
                    current_level + 1,
                    &mut left_nodes,
                    method,
                )
            },
            || {
                nested_dissection_with_method(
                    adjacency,
                    &right_indices,
                    min_block_size,
                    max_levels,
                    current_level + 1,
                    &mut right_nodes,
                    method,
                )
            },
        );
        let base = nodes.len();
        for node in &mut left_nodes {
            node.left = node.left.map(|i| i + base);
            node.right = node.right.map(|i| i + base);
        }
        let right_base = base + left_nodes.len();
        for node in &mut right_nodes {
            node.left = node.left.map(|i| i + right_base);
            node.right = node.right.map(|i| i + right_base);
        }
        nodes.append(&mut left_nodes);
        nodes.append(&mut right_nodes);
        (base + left_local, right_base + right_local)
    } else {
        (
            nested_dissection_with_method(
                adjacency,
                &left_indices,
                min_block_size,
                max_levels,
                current_level + 1,
                nodes,
                method,
            ),
            nested_dissection_with_method(
                adjacency,
                &right_indices,
                min_block_size,
                max_levels,
                current_level + 1,
                nodes,
                method,
            ),
        )
    };

    let child_max_level = nodes[left_node].level.max(nodes[right_node].level);
    let sep_idx = nodes.len();
    nodes.push(DissectionNode {
        global_indices: separator,
        left: Some(left_node),
        right: Some(right_node),
        level: child_max_level + 1,
    });
    sep_idx
}

/// Extract the submatrix corresponding to the given global row/column indices.
///
/// `global_indices` defines both the row set and the column set (square sub).
/// Entries outside the set are silently dropped.
///
/// This infallible compatibility wrapper preserves the historical panic
/// behavior for invalid input. New callers that need typed rejection should
/// use [`try_extract_submatrix`] instead.
pub fn extract_submatrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    global_indices: &[usize],
) -> CsrMatrix<T> {
    try_extract_submatrix(matrix, global_indices)
        .unwrap_or_else(|error| panic!("invalid sparse submatrix extraction: {error}"))
}

/// Checked extraction of a square CSR submatrix.
///
/// The selected rows may be supplied in any order. Local columns are sorted
/// and duplicate entries are accumulated, so the returned matrix is a valid
/// canonical CSR representation with the same additive SpMV semantics as the
/// source matrix. Input structure, dimensions, finite values, out-of-range
/// indices, and duplicate selections are rejected before any unchecked indexing
/// occurs.
pub fn try_extract_submatrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    global_indices: &[usize],
) -> Result<CsrMatrix<T>, SolverError> {
    matrix.validate_structure()?;
    if matrix.num_rows != matrix.num_cols {
        return Err(SolverError::OperationFailed {
            operation: "sparse submatrix extraction",
            reason: format!(
                "source matrix must be square, got {}x{}",
                matrix.num_rows, matrix.num_cols
            ),
        });
    }

    let mut global_to_local = vec![usize::MAX; matrix.num_rows];
    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
        if global_idx >= matrix.num_rows {
            return Err(SolverError::OperationFailed {
                operation: "sparse submatrix extraction",
                reason: format!(
                    "selected global index {global_idx} is outside 0..{}",
                    matrix.num_rows
                ),
            });
        }
        if global_to_local[global_idx] != usize::MAX {
            return Err(SolverError::OperationFailed {
                operation: "sparse submatrix extraction",
                reason: format!("selected global index {global_idx} appears more than once"),
            });
        }
        global_to_local[global_idx] = local_idx;
    }

    let local_n = global_indices.len();
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = Vec::with_capacity(local_n + 1);
    row_ptrs.push(0);

    for &global_row in global_indices {
        let mut row_entries = Vec::new();
        for idx in matrix.row_ptrs[global_row]..matrix.row_ptrs[global_row + 1] {
            let global_col = matrix.col_indices[idx];
            let local_col = global_to_local[global_col];
            if local_col != usize::MAX {
                row_entries.push((local_col, matrix.values[idx]));
            }
        }
        row_entries.sort_unstable_by_key(|(column, _)| *column);
        let row_start = col_indices.len();
        for (column, value) in row_entries {
            if col_indices.len() > row_start {
                let last = col_indices.len() - 1;
                if col_indices[last] == column {
                    values[last] += value;
                    continue;
                }
            }
            col_indices.push(column);
            values.push(value);
        }
        row_ptrs.push(values.len());
    }

    CsrMatrix::try_from_raw_parts(local_n, local_n, row_ptrs, col_indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnz_balanced_ranges_cover_all_rows() {
        // Skewed row lengths 1..=1000, power-law-ish cumulative profile.
        let nrows = 4_096usize;
        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        row_ptrs.push(0usize);
        for i in 0..nrows {
            let deg = 1 + (i * 7) % 1000;
            row_ptrs.push(row_ptrs[i] + deg);
        }
        let nparts = 4;
        let ranges = nnz_balanced_ranges(&row_ptrs, nparts);

        assert_eq!(ranges.len(), nparts);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[nparts - 1].end, nrows);
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start, "ranges must be contiguous");
        }
        // Balance: no range should hold more than target + max row nnz.
        let target = row_ptrs[nrows] / nparts;
        for range in &ranges {
            let nnz = row_ptrs[range.end] - row_ptrs[range.start];
            assert!(
                nnz <= target + 1000,
                "range {range:?} holds {nnz} nnz, target {target}"
            );
        }
    }

    #[test]
    fn test_nnz_balanced_ranges_edge_cases() {
        // Empty matrix / zero parts.
        assert!(nnz_balanced_ranges(&[0usize], 4).is_empty());
        assert!(nnz_balanced_ranges(&[0usize, 10], 0).is_empty());

        // Single part covers everything.
        let row_ptrs = [0usize, 3, 3, 10];
        assert_eq!(nnz_balanced_ranges(&row_ptrs, 1), vec![0..3]);

        // One heavy row dominates: empty ranges are allowed, coverage holds.
        let row_ptrs = [0usize, 1000, 1001, 1002];
        let ranges = nnz_balanced_ranges(&row_ptrs, 4);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[3].end, 3);

        // u32 row pointers (compact CSR) work through the same helper.
        let row_ptrs_u32 = [0u32, 5, 10, 15, 20];
        let ranges = nnz_balanced_ranges(&row_ptrs_u32, 2);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[1].end, 4);
    }

    /// 5-point stencil adjacency on an `nx` x `ny` grid (row-major ids).
    fn grid2d_adjacency(nx: usize, ny: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let v = y * nx + x;
                if x + 1 < nx {
                    adj[v].push(v + 1);
                    adj[v + 1].push(v);
                }
                if y + 1 < ny {
                    adj[v].push(v + nx);
                    adj[v + nx].push(v);
                }
            }
        }
        adj
    }

    /// 7-point stencil adjacency on an `nx` x `ny` x `nz` grid.
    fn grid3d_adjacency(nx: usize, ny: usize, nz: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let v = (z * ny + y) * nx + x;
                    if x + 1 < nx {
                        adj[v].push(v + 1);
                        adj[v + 1].push(v);
                    }
                    if y + 1 < ny {
                        adj[v].push(v + nx);
                        adj[v + nx].push(v);
                    }
                    if z + 1 < nz {
                        adj[v].push(v + nx * ny);
                        adj[v + nx * ny].push(v);
                    }
                }
            }
        }
        adj
    }

    /// Unstructured-mesh-like graph: triangulated `nx` x `ny` grid whose cell
    /// diagonals are oriented by a deterministic LCG (mimics a Delaunay mesh
    /// without external dependencies).
    fn triangulated_adjacency(nx: usize, ny: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); nx * ny];
        let mut state = seed;
        let mut next_bit = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) & 1
        };
        let add_edge = |adj: &mut Vec<Vec<usize>>, a: usize, b: usize| {
            adj[a].push(b);
            adj[b].push(a);
        };
        for y in 0..ny {
            for x in 0..nx {
                let v = y * nx + x;
                if x + 1 < nx {
                    add_edge(&mut adj, v, v + 1);
                }
                if y + 1 < ny {
                    add_edge(&mut adj, v, v + nx);
                }
                if x + 1 < nx && y + 1 < ny {
                    if next_bit() == 0 {
                        add_edge(&mut adj, v, v + nx + 1);
                    } else {
                        add_edge(&mut adj, v + 1, v + nx);
                    }
                }
            }
        }
        adj
    }

    /// Every edge must touch the separator or stay inside one side.
    fn assert_valid_bisection(
        adjacency: &[Vec<usize>],
        left: &[usize],
        right: &[usize],
        separator: &[usize],
    ) {
        let mut part = vec![u8::MAX; adjacency.len()];
        for &v in left {
            part[v] = 0;
        }
        for &v in right {
            part[v] = 1;
        }
        for &v in separator {
            part[v] = 2;
        }
        for v in 0..adjacency.len() {
            if part[v] == u8::MAX {
                continue;
            }
            for &u in &adjacency[v] {
                if part[u] == u8::MAX {
                    continue;
                }
                assert!(
                    part[v] == part[u] || part[v] == 2 || part[u] == 2,
                    "edge {v}-{u} crosses left/right without a separator endpoint"
                );
            }
        }
    }

    fn assert_balanced(n: usize, left: &[usize], right: &[usize]) {
        let limit = (n as f64 * MAX_PART_FRACTION).ceil() as usize;
        assert!(
            left.len() <= limit && right.len() <= limit,
            "unbalanced bisection: |left|={}, |right|={}, limit={limit} (n={n})",
            left.len(),
            right.len()
        );
    }

    #[test]
    fn test_multilevel_bisect_beats_bfs_on_2d_grid() {
        let nx = 64;
        let adjacency = grid2d_adjacency(nx, nx);
        let indices: Vec<usize> = (0..nx * nx).collect();

        let (bl, br, bs) = bfs_bisect(&adjacency, &indices);
        let (ml, mr, ms) = multilevel_bisect(&adjacency, &indices);

        assert_valid_bisection(&adjacency, &ml, &mr, &ms);
        assert_balanced(indices.len(), &ml, &mr);
        assert!(!ml.is_empty() && !mr.is_empty());
        assert!(
            ms.len() * 5 <= bs.len() * 4,
            "expected >=20% separator reduction: multilevel={}, bfs={}",
            ms.len(),
            bs.len()
        );
        // Sanity on the BFS path itself.
        assert_valid_bisection(&adjacency, &bl, &br, &bs);
    }

    #[test]
    fn test_multilevel_bisect_beats_bfs_on_3d_grid() {
        let n = 16;
        let adjacency = grid3d_adjacency(n, n, n);
        let indices: Vec<usize> = (0..n * n * n).collect();

        let (_, _, bs) = bfs_bisect(&adjacency, &indices);
        let (ml, mr, ms) = multilevel_bisect(&adjacency, &indices);

        assert_valid_bisection(&adjacency, &ml, &mr, &ms);
        assert_balanced(indices.len(), &ml, &mr);
        assert!(
            ms.len() < bs.len(),
            "expected smaller separator: multilevel={}, bfs={}",
            ms.len(),
            bs.len()
        );
    }

    #[test]
    fn test_multilevel_bisect_unstructured_mesh() {
        let nx = 40;
        let adjacency = triangulated_adjacency(nx, nx, 0x9e3779b97f4a7c15);
        let indices: Vec<usize> = (0..nx * nx).collect();

        let (_, _, bs) = bfs_bisect(&adjacency, &indices);
        let (ml, mr, ms) = multilevel_bisect(&adjacency, &indices);

        assert_valid_bisection(&adjacency, &ml, &mr, &ms);
        assert_balanced(indices.len(), &ml, &mr);
        assert!(
            ms.len() * 10 <= bs.len() * 9,
            "expected >=10% separator reduction: multilevel={}, bfs={}",
            ms.len(),
            bs.len()
        );
    }

    #[test]
    fn test_nested_dissection_deterministic() {
        let adjacency = grid2d_adjacency(32, 32);
        let indices: Vec<usize> = (0..32 * 32).collect();
        let mut nodes_a = Vec::new();
        let mut nodes_b = Vec::new();
        nested_dissection(&adjacency, &indices, 8, 64, 0, &mut nodes_a);
        nested_dissection(&adjacency, &indices, 8, 64, 0, &mut nodes_b);
        assert_eq!(nodes_a, nodes_b);
    }

    #[test]
    fn test_nested_dissection_parallel_deterministic_and_complete() {
        // 48x48 = 2304 nodes exceeds ND_PARALLEL_MIN_SUBTREE, so the rayon
        // subtree-join path (node-list merge + child-link rebase) is taken.
        let adjacency = grid2d_adjacency(48, 48);
        let indices: Vec<usize> = (0..48 * 48).collect();
        let mut nodes_a = Vec::new();
        let mut nodes_b = Vec::new();
        nested_dissection(&adjacency, &indices, 8, 64, 0, &mut nodes_a);
        nested_dissection(&adjacency, &indices, 8, 64, 0, &mut nodes_b);
        assert_eq!(nodes_a, nodes_b, "parallel ND must be deterministic");

        // Every DOF appears exactly once; child links are in-range and
        // post-ordered (children before parents).
        let mut seen = vec![false; indices.len()];
        for (idx, node) in nodes_a.iter().enumerate() {
            for &g in &node.global_indices {
                assert!(!seen[g], "dof {g} assigned twice");
                seen[g] = true;
            }
            if let Some(left) = node.left {
                assert!(left < idx, "left child must precede its parent");
            }
            if let Some(right) = node.right {
                assert!(right < idx, "right child must precede its parent");
            }
        }
        assert!(seen.iter().all(|&s| s), "some dofs missing from tree");
    }

    #[test]
    fn test_nested_dissection_partitions_all_dofs() {
        let adjacency = grid2d_adjacency(24, 24);
        let indices: Vec<usize> = (0..24 * 24).collect();
        let mut nodes = Vec::new();
        nested_dissection(&adjacency, &indices, 8, 64, 0, &mut nodes);

        let mut seen = vec![false; indices.len()];
        for node in &nodes {
            for &g in &node.global_indices {
                assert!(!seen[g], "dof {g} assigned twice");
                seen[g] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some dofs missing from tree");
    }

    #[test]
    fn test_nested_dissection_bfs_method_still_available() {
        let adjacency = grid2d_adjacency(16, 16);
        let indices: Vec<usize> = (0..16 * 16).collect();
        let mut nodes = Vec::new();
        let root = nested_dissection_with_method(
            &adjacency,
            &indices,
            8,
            64,
            0,
            &mut nodes,
            BisectionMethod::Bfs,
        );
        assert_eq!(root, nodes.len() - 1);
        assert!(nodes.len() > 1);
    }

    #[test]
    fn checked_submatrix_canonicalizes_order_and_duplicate_entries() {
        let matrix = CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 3, 4, 6],
            vec![0, 2, 2, 1, 0, 2],
            vec![4.0, 1.0, 2.0, 5.0, 3.0, 6.0],
        );

        let submatrix = try_extract_submatrix(&matrix, &[2, 0]).expect("valid submatrix");
        assert_eq!(submatrix.row_ptrs, vec![0, 2, 4]);
        assert_eq!(submatrix.col_indices, vec![0, 1, 0, 1]);
        assert_eq!(submatrix.values, vec![6.0, 3.0, 3.0, 4.0]);
        assert!(submatrix.validate_canonical_structure().is_ok());
    }

    #[test]
    fn checked_submatrix_rejects_invalid_selection_and_source_state() {
        let matrix = CsrMatrix::<f64>::identity(3);

        let out_of_range =
            try_extract_submatrix(&matrix, &[3]).expect_err("index must be rejected");
        assert!(matches!(
            out_of_range,
            SolverError::OperationFailed {
                operation: "sparse submatrix extraction",
                ..
            }
        ));

        let duplicate =
            try_extract_submatrix(&matrix, &[1, 1]).expect_err("duplicate must be rejected");
        assert!(matches!(
            duplicate,
            SolverError::OperationFailed {
                operation: "sparse submatrix extraction",
                ..
            }
        ));

        let rectangular = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0]);
        let shape_error =
            try_extract_submatrix(&rectangular, &[0]).expect_err("shape must be rejected");
        assert!(matches!(
            shape_error,
            SolverError::OperationFailed {
                operation: "sparse submatrix extraction",
                ..
            }
        ));

        let mut non_finite = CsrMatrix::<f64>::identity(2);
        non_finite.values[0] = f64::NAN;
        assert!(try_extract_submatrix(&non_finite, &[0]).is_err());
    }
}
