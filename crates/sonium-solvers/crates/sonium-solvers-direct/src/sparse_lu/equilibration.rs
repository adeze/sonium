//! MC64-style preprocessing for sparse LU: maximum-weight bipartite
//! matching plus two-sided row/column scaling.
//!
//! State-of-the-art sparse LU codes (SuperLU, MUMPS) permute large-magnitude
//! entries onto the diagonal and equilibrate row/column scales *before*
//! numeric factorization, which makes partial pivoting far more reliable on
//! badly scaled matrices. This module provides the two ingredients:
//!
//! 1. **Maximum-weight matching** on the bipartite graph of |A|, with
//!    log-modulus weights `w_ij = ln |a_ij|`, so that maximising the sum of
//!    weights maximises the product of the diagonal entries after the
//!    matched columns are permuted onto the diagonal (the MC64 "max product"
//!    objective). The matching is computed by *successive shortest
//!    augmenting paths* with Johnson potentials (a sparse primal-dual
//!    assignment solver): exact, deterministic, and `O(n · nnz · log n)`
//!    worst case. This is simpler than the full MC64 algorithm (which adds
//!    strongly-connected-component compression for structurally singular
//!    matrices) but produces the same optimal matching when a perfect
//!    matching exists; when none exists the caller falls back to no
//!    preprocessing.
//!
//! 2. **Two-sided max-abs equilibration** (the SuperLU `equil` strategy
//!    rather than the exact MC64 D2 scaling): first scale each row by the
//!    inverse of its max modulus, then scale each column of the row-scaled
//!    matrix by the inverse of its max modulus. With the matched large
//!    entries on the diagonal, the scaled matrix satisfies
//!    `max_j |â_ij| ≤ 1` per row and column, with the diagonal typically
//!    the dominant entry — the practical effect of MC64 D2 (diagonal = 1,
//!    off-diagonal ≤ 1) up to one extra relaxation pass.
//!
//! The equilibrated matrix is `Â = Dr · A · Q · Dc`, where `Q` is the
//! matching column permutation. Solving `A x = b` reduces to solving
//! `Â y = Dr b` and unscaling `x = Q Dc y`.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use num_traits::{Float, One, ToPrimitive, Zero};

use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Result of [`equilibrate`]: row/column scaling factors and the matching
/// column permutation, all in the indexing documented per field.
#[derive(Debug, Clone)]
pub(crate) struct Equilibration<R> {
    /// Row scaling `Dr`: equilibrated row `i` is `row_scale[i]` times the
    /// *original* row `i` (rows are not reordered by the preprocessing).
    pub row_scale: Vec<R>,
    /// Matching permutation `Q`: equilibrated column `j` holds original
    /// column `col_perm[j]`, placing the matched entry of row `j` on the
    /// diagonal.
    pub col_perm: Vec<usize>,
    /// Column scaling `Dc` in the *permuted* column indexing.
    pub col_scale: Vec<R>,
}

/// `f64` wrapper with a total order, for use in the Dijkstra min-heap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct F64Ord(f64);

impl Eq for F64Ord {}

impl PartialOrd for F64Ord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F64Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Minimum-cost perfect matching on the bipartite graph of a square CSR
/// pattern, minimising `Σ cost[idx]` over matched edges.
///
/// Successive shortest augmenting paths: each of the `n` iterations runs
/// Dijkstra from one free row over the alternating graph (forward edges
/// `row → col` for every nonzero, backward edges `col → row` for matched
/// pairs) with non-negative reduced costs maintained by Johnson potentials.
/// Returns `match_r[i] = column matched to row i`, or `None` when no
/// perfect matching exists (structurally singular pattern).
///
/// `cost` must be non-negative; shifting all costs by a constant does not
/// change the optimum because every perfect matching uses exactly `n` edges.
fn max_weight_matching(
    n: usize,
    row_ptrs: &[usize],
    col_indices: &[usize],
    cost: &[f64],
) -> Option<Vec<usize>> {
    let mut match_r = vec![usize::MAX; n];
    let mut match_c = vec![usize::MAX; n];
    let mut match_idx = vec![usize::MAX; n]; // CSR index of the edge matching column j
    let mut pot = vec![0.0_f64; 2 * n];
    let mut dist = vec![0.0_f64; 2 * n];
    let mut parent = vec![usize::MAX; 2 * n];
    let mut parent_edge = vec![usize::MAX; 2 * n];
    let mut settled = vec![false; 2 * n];

    for start in 0..n {
        dist.fill(f64::INFINITY);
        parent.fill(usize::MAX);
        settled.fill(false);
        dist[start] = 0.0;
        let mut heap: BinaryHeap<Reverse<(F64Ord, usize)>> =
            BinaryHeap::from([Reverse((F64Ord(0.0), start))]);
        let mut target = usize::MAX;

        while let Some(Reverse((F64Ord(d), u))) = heap.pop() {
            if settled[u] {
                continue;
            }
            settled[u] = true;
            if u >= n {
                let j = u - n;
                if match_c[j] == usize::MAX {
                    // Nearest free column: end of the shortest augmenting path.
                    target = u;
                    break;
                }
                // Backward edge along the matched pair (reduced cost 0).
                let i = match_c[j];
                // Clamp: exact arithmetic keeps reduced costs non-negative
                // by dual feasibility, so a negative value is rounding
                // noise from `pot[u] - pot[i]` cancellation. Without the
                // clamp it can overwrite `parent[start]` and cycle the
                // augment chain below forever (seen on a 187x187 saddle
                // with 1e17 block-scale disparity).
                let rc = (-cost[match_idx[j]] + pot[u] - pot[i]).max(0.0);
                let nd = d + rc;
                if nd < dist[i] {
                    dist[i] = nd;
                    parent[i] = u;
                    heap.push(Reverse((F64Ord(nd), i)));
                }
            } else {
                let i = u;
                for idx in row_ptrs[i]..row_ptrs[i + 1] {
                    let j = col_indices[idx];
                    if match_c[j] == i {
                        continue; // forward copy of the matched edge is useless
                    }
                    let v = n + j;
                    // Same clamp as the backward edge: negatives are
                    // rounding noise, and Dijkstra needs non-negative
                    // reduced costs.
                    let rc = (cost[idx] + pot[u] - pot[v]).max(0.0);
                    let nd = d + rc;
                    if nd < dist[v] {
                        dist[v] = nd;
                        parent[v] = u;
                        parent_edge[v] = idx;
                        heap.push(Reverse((F64Ord(nd), v)));
                    }
                }
            }
        }

        if target == usize::MAX {
            return None; // no perfect matching (Hall violation)
        }
        for v in 0..2 * n {
            if settled[v] {
                pot[v] += dist[v];
            }
        }

        // Augment along the parent chain back to `start`. A valid chain
        // alternates column/row nodes and visits each at most once, so it
        // is at most `2n` hops; anything longer is a corrupted parent
        // chain, reported as no-matching (the caller then factorizes
        // without preprocessing) instead of hanging.
        let mut col_node = target;
        let mut arrived = false;
        for _ in 0..=2 * n {
            let i = parent[col_node];
            let j = col_node - n;
            let previous = parent[i]; // column node that led to row i
            match_r[i] = j;
            match_c[j] = i;
            match_idx[j] = parent_edge[col_node];
            if previous == usize::MAX {
                arrived = true;
                break;
            }
            col_node = previous;
        }
        if !arrived {
            return None;
        }
    }

    Some(match_r)
}

/// Build the equilibrated matrix `Â = Dr · A · Q · Dc` for a square matrix.
///
/// Returns `None` when the sparsity pattern admits no perfect matching
/// (structurally singular); the caller should then factorize without
/// preprocessing.
pub(crate) fn equilibrate<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Option<(CsrMatrix<T>, Equilibration<T::Real>)> {
    let n = matrix.num_rows;
    if n == 0 || n != matrix.num_cols {
        return None;
    }

    // Log-modulus costs: maximising Σ ln |a_i,m(i)| maximises the product
    // of the matched diagonal entries. Shifted to be non-negative.
    let mut cost = Vec::with_capacity(matrix.values.len());
    let mut cost_min = f64::INFINITY;
    for value in &matrix.values {
        let log_modulus = value
            .norm()
            .to_f64()
            .unwrap_or(0.0)
            .max(f64::MIN_POSITIVE)
            .ln();
        cost.push(-log_modulus);
        cost_min = cost_min.min(-log_modulus);
    }
    for c in &mut cost {
        *c -= cost_min;
    }

    let row_match = max_weight_matching(n, &matrix.row_ptrs, &matrix.col_indices, &cost)?;

    // Q: equilibrated column j holds original column col_perm[j], so the
    // matched entry of row j lands on the diagonal.
    let mut col_perm = vec![0usize; n];
    for (row, &col) in row_match.iter().enumerate() {
        col_perm[col] = row;
    }
    let mut col_perm_inv = vec![0usize; n];
    for (j, &c) in col_perm.iter().enumerate() {
        col_perm_inv[c] = j;
    }

    // Row max-abs scaling (rows are not reordered by the matching).
    let one = T::Real::one();
    let mut row_scale = vec![one; n];
    for (row, slot) in row_scale.iter_mut().enumerate() {
        let mut row_max = T::Real::zero();
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            row_max = row_max.max(matrix.values[idx].norm());
        }
        if row_max > T::Real::zero() {
            *slot = one / row_max;
        }
    }

    // Column max-abs scaling of Dr · A · Q. `col_scale` first accumulates
    // the maxima, then is inverted in place.
    let mut col_scale = vec![T::Real::zero(); n];
    for (row, &scale) in row_scale.iter().enumerate() {
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let j = col_perm_inv[matrix.col_indices[idx]];
            col_scale[j] = col_scale[j].max(scale * matrix.values[idx].norm());
        }
    }
    for scale in &mut col_scale {
        *scale = if *scale > T::Real::zero() {
            one / *scale
        } else {
            one
        };
    }

    // Assemble Â = Dr · A · Q · Dc with sorted column indices per row.
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(matrix.values.len());
    let mut values = Vec::with_capacity(matrix.values.len());
    let mut row_entries: Vec<(usize, T)> = Vec::new();
    row_ptrs.push(0);
    for (row, &scale) in row_scale.iter().enumerate() {
        row_entries.clear();
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let j = col_perm_inv[matrix.col_indices[idx]];
            let scaled = matrix.values[idx] * T::from_real(scale * col_scale[j]);
            row_entries.push((j, scaled));
        }
        row_entries.sort_by_key(|&(j, _)| j);
        for (j, value) in row_entries.drain(..) {
            col_indices.push(j);
            values.push(value);
        }
        row_ptrs.push(col_indices.len());
    }

    Some((
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values),
        Equilibration {
            row_scale,
            col_perm,
            col_scale,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_core::dense::DenseMatrix;

    #[test]
    fn matching_finds_large_diagonal() {
        // Identity diagonal is tiny; the max-product matching must swap.
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1e-3_f64, 1.0, 1.0, 1.0]);
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let (equilibrated, pre) = equilibrate(&matrix).expect("perfect matching exists");
        assert_eq!(pre.col_perm, vec![1, 0]);
        // Diagonal of the equilibrated matrix must be the large entries.
        let dense_eq = equilibrated.to_dense();
        assert!(dense_eq[[0, 0]].abs() > 0.5);
        assert!(dense_eq[[1, 1]].abs() > 0.5);
    }

    #[test]
    fn equilibrated_entries_are_bounded_by_one() {
        // Badly scaled rows/columns spanning 1e±6.
        let n = 8;
        let dense = DenseMatrix::from_fn(n, n, |i, j| {
            let base = if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            };
            base * 10f64.powi(i as i32 - 4) * 10f64.powi(3 - j as i32)
        });
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let (equilibrated, _) = equilibrate(&matrix).expect("perfect matching exists");
        let mut max_modulus = 0.0_f64;
        for value in &equilibrated.values {
            max_modulus = max_modulus.max(value.abs());
        }
        assert!(
            max_modulus <= 1.0 + 1e-12,
            "max-abs equilibration must bound entries by 1, got {max_modulus}"
        );
    }

    #[test]
    fn structurally_singular_pattern_returns_none() {
        // Row 1 is empty: no perfect matching exists.
        let matrix = CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 2, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        assert!(equilibrate(&matrix).is_none());
    }

    #[test]
    fn matching_is_optimal_on_small_example() {
        // Brute-force check on a 4x4 with distinct weights.
        let dense = DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                5.0_f64, 1.0, 9.0, 1.0, //
                1.0, 8.0, 1.0, 2.0, //
                9.0, 1.0, 1.0, 1.0, //
                1.0, 2.0, 1.0, 7.0,
            ],
        );
        let matrix = CsrMatrix::from_dense(&dense, 0.0);
        let (_, pre) = equilibrate(&matrix).expect("perfect matching exists");
        // Optimal assignment (checked by enumeration): rows 0..3 -> cols 2,1,0,3
        // with weight 9+8+9+7 = 33.
        let mut best = 0.0_f64;
        let mut perm = [0usize, 1, 2, 3];
        loop {
            let weight: f64 = (0..4).map(|i| dense[[i, perm[i]]]).product();
            best = best.max(weight);
            if !next_permutation(&mut perm) {
                break;
            }
        }
        let matched: f64 = (0..4).map(|j| dense[[j, pre.col_perm[j]]]).product();
        assert!(
            (matched - best).abs() <= 1e-9 * best,
            "matching product {matched} should equal brute-force optimum {best}"
        );
    }

    fn next_permutation(p: &mut [usize]) -> bool {
        let n = p.len();
        let Some(i) = (1..n).rev().find(|&i| p[i - 1] < p[i]) else {
            return false;
        };
        let j = (i..n).rev().find(|&j| p[j] > p[i - 1]).unwrap();
        p.swap(i - 1, j);
        p[i..].reverse();
        true
    }
}
