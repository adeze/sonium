//! Approximate Minimum Degree (AMD) ordering for symmetric sparsity patterns.
//!
//! Quotient-graph implementation following Amestoy, Davis & Duff, "An
//! Approximate Minimum Degree Ordering Algorithm" (SIAM J. Matrix Anal.
//! Appl., 1996), in the same family as SuiteSparse AMD:
//!
//! - **Quotient representation**: eliminated variables merge into elements
//!   (cliques over their reachable sets), so fill is represented implicitly
//!   and never materialised edge by edge. Columns of the input are the
//!   variables; the pattern is symmetrised (`A + Aᵀ`, diagonal dropped), so
//!   unsymmetric inputs order by their symmetrised structure.
//! - **Exact external degrees with a lazy heap**: degrees are recomputed
//!   exactly (union over adjacency plus live element members) whenever a
//!   heap entry is popped, and stale entries are re-queued. This is exact
//!   minimum degree on the quotient graph — quality at or above the
//!   approximate-degree update, at a higher analyze cost that amortises over
//!   reused symbolic factorizations.
//! - **Covered-element absorption**: after each pivot, elements whose live
//!   members are all covered by the new clique are absorbed, keeping
//!   element lists short.
//! - **Mass elimination**: a member of the new clique whose remaining
//!   element list is empty is simplicial — its clique is exactly the new
//!   element — so it is eliminated immediately with no new element.
//! - **Dense-variable removal**: variables with degree at least
//!   `max(16, 10·√n)` are removed up front and ordered last (same cutoff as
//!   the [`colamd`](super::colamd) dense-row handling).
//!
//! Deliberately kept simple relative to SuiteSparse AMD: no supervariable
//! detection (identical-adjacency grouping) and no approximate-degree
//! short-circuit. Both are performance refinements; the elimination order
//! quality on PDE grids matches reference AMD fill within a few percent.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Compute a fill-reducing variable permutation for a square sparse matrix.
///
/// The pattern is symmetrised, so this is the ordering to use for LU when
/// the structure is (near-)symmetric — SuiteSparse UMFPACK/CHOLMOD territory
/// where AMD beats column orderings. Returns `perm` with `perm[k]` the
/// original variable eliminated at step `k`, matching [`colamd`](super::colamd).
pub fn amd<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    let n = matrix.num_rows;
    if n <= 1 {
        return (0..n).collect();
    }

    // Symmetric adjacency (A + Aᵀ), self-loops dropped.
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for index in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[index];
            if j < n && j != i {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for neighbours in &mut adj {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    // Dense variables leave the quotient graph up front and go last.
    let dense_threshold = (16.0_f64).max(10.0 * (n as f64).sqrt()) as usize;
    let mut alive = vec![true; n];
    let mut dense = Vec::new();
    for (variable, neighbours) in adj.iter().enumerate() {
        if neighbours.len() >= dense_threshold {
            alive[variable] = false;
            dense.push(variable);
        }
    }

    let mut var_elements: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut elem_members: Vec<Vec<usize>> = Vec::new();
    let mut elem_alive: Vec<bool> = Vec::new();
    let mut seen = vec![0usize; n];
    let mut generation = 0usize;

    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for variable in 0..n {
        if !alive[variable] {
            continue;
        }
        let degree = external_degree(
            variable,
            &adj,
            &var_elements,
            &elem_members,
            &elem_alive,
            &alive,
            &mut seen,
            &mut generation,
        );
        heap.push(Reverse((degree, variable)));
    }

    let mut perm = Vec::with_capacity(n);
    while let Some(Reverse((key_degree, variable))) = heap.pop() {
        if !alive[variable] {
            continue;
        }
        let degree = external_degree(
            variable,
            &adj,
            &var_elements,
            &elem_members,
            &elem_alive,
            &alive,
            &mut seen,
            &mut generation,
        );
        if degree != key_degree {
            heap.push(Reverse((degree, variable)));
            continue;
        }

        // Eliminate the pivot: the new element is the reachable clique.
        alive[variable] = false;
        perm.push(variable);
        generation += 1;
        let marks = generation;
        seen[variable] = marks;
        let mut clique = Vec::new();
        for &neighbour in &adj[variable] {
            if alive[neighbour] && seen[neighbour] != marks {
                seen[neighbour] = marks;
                clique.push(neighbour);
            }
        }
        for &element in &var_elements[variable] {
            if !elem_alive[element] {
                continue;
            }
            for &member in &elem_members[element] {
                if alive[member] && seen[member] != marks {
                    seen[member] = marks;
                    clique.push(member);
                }
            }
            // Pivot elements merge into the new element.
            elem_alive[element] = false;
        }
        if clique.is_empty() {
            continue;
        }
        // Phase 1 (round marks valid): prune dead elements, absorb covered
        // ones, and decide mass elimination. No degree recomputation here —
        // it would bump the generation and invalidate the marks below.
        let mut simplicial = vec![false; clique.len()];
        for (position, &member) in clique.iter().enumerate() {
            let mut kept = Vec::new();
            std::mem::swap(&mut var_elements[member], &mut kept);
            for element in kept {
                if !elem_alive[element] {
                    continue;
                }
                let mut covered = true;
                for &other in &elem_members[element] {
                    if alive[other] && seen[other] != marks {
                        covered = false;
                        break;
                    }
                }
                if covered {
                    elem_alive[element] = false;
                } else {
                    var_elements[member].push(element);
                }
            }
            // Simplicial iff nothing live remains outside the new clique:
            // no uncovered elements left AND every live adjacency neighbour
            // is marked. (An empty element list alone does not suffice —
            // adjacency neighbours are not elements.)
            let mut is_simplicial = var_elements[member].is_empty();
            if is_simplicial {
                for &other in &adj[member] {
                    if alive[other] && seen[other] != marks {
                        is_simplicial = false;
                        break;
                    }
                }
            }
            simplicial[position] = is_simplicial;
        }
        // Phase 2: install the new element, eliminate simplicial members,
        // refresh surviving members (degree recomputation bumps generations
        // freely from here on).
        let new_element = elem_members.len();
        elem_members.push(clique);
        elem_alive.push(true);
        for (position, &member) in elem_members[new_element].iter().enumerate() {
            if simplicial[position] {
                alive[member] = false;
                perm.push(member);
            } else {
                var_elements[member].push(new_element);
                let degree = external_degree(
                    member,
                    &adj,
                    &var_elements,
                    &elem_members,
                    &elem_alive,
                    &alive,
                    &mut seen,
                    &mut generation,
                );
                heap.push(Reverse((degree, member)));
            }
        }
    }

    perm.extend_from_slice(&dense);
    debug_assert_eq!(perm.len(), n, "AMD must order every variable");
    perm
}

/// Exact external degree of a live variable: distinct live reachable
/// variables through adjacency and live elements, excluding itself.
#[allow(clippy::too_many_arguments)]
fn external_degree(
    variable: usize,
    adj: &[Vec<usize>],
    var_elements: &[Vec<usize>],
    elem_members: &[Vec<usize>],
    elem_alive: &[bool],
    alive: &[bool],
    seen: &mut [usize],
    generation: &mut usize,
) -> usize {
    *generation += 1;
    let marks = *generation;
    let mut degree = 0;
    for &neighbour in &adj[variable] {
        if neighbour != variable && alive[neighbour] && seen[neighbour] != marks {
            seen[neighbour] = marks;
            degree += 1;
        }
    }
    for &element in &var_elements[variable] {
        if !elem_alive[element] {
            continue;
        }
        for &member in &elem_members[element] {
            if member != variable && alive[member] && seen[member] != marks {
                seen[member] = marks;
                degree += 1;
            }
        }
    }
    degree
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CSR from an undirected edge list (symmetric pattern, unit values).
    fn graph_matrix(n: usize, edges: &[(usize, usize)]) -> CsrMatrix<f64> {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in edges {
            rows[i].push(j);
            rows[j].push(i);
        }
        let mut row_ptrs = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (i, neighbours) in rows.iter_mut().enumerate() {
            neighbours.sort_unstable();
            neighbours.dedup();
            neighbours.retain(|&j| j != i);
            for &j in neighbours.iter() {
                col_indices.push(j);
                values.push(1.0);
            }
            row_ptrs[i + 1] = col_indices.len();
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    fn is_permutation(perm: &[usize], n: usize) -> bool {
        let mut seen = vec![false; n];
        perm.len() == n
            && perm.iter().all(|&v| {
                v < n && {
                    let fresh = !seen[v];
                    seen[v] = true;
                    fresh
                }
            })
    }

    #[test]
    fn star_eliminates_center_last() {
        // Center 0 joined to leaves 1..=5: minimum degree must peel every
        // leaf before the center.
        let edges: Vec<(usize, usize)> = (1..6).map(|leaf| (0, leaf)).collect();
        let perm = amd(&graph_matrix(6, &edges));
        assert!(is_permutation(&perm, 6));
        // The center has strictly largest degree until at most one leaf
        // remains, so it goes fifth or sixth (a 1–1 tie with the last leaf
        // may break either way; both orders are fill-free).
        let center_position = perm.iter().position(|&v| v == 0).unwrap();
        assert!(
            center_position >= 4,
            "star center eliminated too early, got {perm:?}"
        );
    }

    #[test]
    fn path_eliminates_leaf_first() {
        // Every interior vertex has degree 2, endpoints degree 1.
        let edges: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 1)).collect();
        let perm = amd(&graph_matrix(6, &edges));
        assert!(is_permutation(&perm, 6));
        assert!(
            perm[0] == 0 || perm[0] == 5,
            "path elimination starts at a leaf, got {perm:?}"
        );
    }

    #[test]
    fn ordering_is_deterministic_and_total() {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
            (2, 4),
            (4, 5),
            (5, 3),
        ];
        let matrix = graph_matrix(6, &edges);
        let first = amd(&matrix);
        let second = amd(&matrix);
        assert_eq!(first, second);
        assert!(is_permutation(&first, 6));
    }

    #[test]
    fn empty_and_singleton() {
        assert_eq!(amd(&graph_matrix(0, &[])), Vec::<usize>::new());
        assert_eq!(amd(&graph_matrix(1, &[])), vec![0]);
    }
}
