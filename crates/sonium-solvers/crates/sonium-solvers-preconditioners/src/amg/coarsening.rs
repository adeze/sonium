use std::collections::VecDeque;

use super::PointType;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sys::parallel::parallel_enumerate_map;

/// Build the transpose of the strength graph: `strong_t[i] = {j : i in S_j}`,
/// i.e. the set of points that strongly depend on `i`.
fn build_strong_transpose(strong: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = strong.len();
    // Counting pass first: pre-sized rows avoid per-push reallocation,
    // which dominated this transpose on large levels.
    let mut counts = vec![0usize; n];
    for row in strong.iter() {
        for &i in row {
            counts[i] += 1;
        }
    }
    let mut strong_t: Vec<Vec<usize>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
    for (j, row) in strong.iter().enumerate() {
        for &i in row {
            strong_t[i].push(j);
        }
    }
    strong_t
}

/// Classical Ruge-Stuben first pass.
///
/// Uses the textbook *dynamic* measure update (Ruge & Stüben 1987;
/// Henson & Yang 2002, alg. "RS"): the measure
/// `lambda_i = |S_i^T ∩ U|` (undecided points that strongly depend on `i`)
/// is updated every time a point is assigned — after `i` becomes a C-point,
/// each undecided `j ∈ S_i^T` becomes an F-point and `lambda_l` is
/// incremented for every undecided `l ∈ S_j`. Selection always takes the
/// current maximum, rather than an ordering frozen at the initial `lambda`.
/// This pass is inherently sequential (each C-selection mutates the global
/// measures), which is exactly why the parallel variants (PMIS/HMIS/CLJP)
/// exist.
fn rs_first_pass(strong: &[Vec<usize>], strong_t: &[Vec<usize>]) -> Vec<PointType> {
    let n = strong.len();
    let mut point_types = vec![PointType::Undecided; n];

    // Dynamic influence measure lambda_i = |S_i^T ∩ U|.
    let mut lambda: Vec<usize> = strong_t.iter().map(|s| s.len()).collect();

    // Bucket priority queue over the small-integer measures: bucket[m]
    // holds undecided nodes last seen with lambda == m. Entries go stale
    // when lambda is incremented after the push; stale entries are
    // detected on pop by comparing against the current lambda and
    // skipped. Push is O(1) and pops scan down from the cached maximum,
    // replacing the O(log n) binary heap (the dominant cost of this
    // sequential pass on large levels). Selection order among equal
    // measures differs from the heap's largest-node tie-break, but any
    // maximal-measure choice is a valid Ruge-Stuben selection.
    let max_lam = lambda.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_lam + 1];
    for (i, &lam) in lambda.iter().enumerate() {
        buckets[lam].push(i);
    }
    let mut cur_max = max_lam;

    // Repeatedly select the undecided point with the largest *current* measure.
    loop {
        while cur_max > 0 && buckets[cur_max].is_empty() {
            cur_max -= 1;
        }
        let Some(i) = buckets[cur_max].pop() else {
            break;
        };
        let lam = cur_max;
        if point_types[i] != PointType::Undecided || lam != lambda[i] {
            // Assigned already, or stale entry superseded by a newer push.
            continue;
        }

        // Make i a C-point.
        point_types[i] = PointType::Coarse;

        // All undecided points that strongly depend on i (in S_i^T) become
        // F-points, and each of them boosts the measure of the undecided
        // points it strongly depends on.
        for &j in &strong_t[i] {
            if point_types[j] != PointType::Undecided {
                continue;
            }
            point_types[j] = PointType::Fine;
            for &l in &strong[j] {
                if point_types[l] == PointType::Undecided {
                    lambda[l] += 1;
                    if lambda[l] >= buckets.len() {
                        buckets.resize(lambda[l] + 1, Vec::new());
                    }
                    buckets[lambda[l]].push(l);
                    cur_max = cur_max.max(lambda[l]);
                }
            }
        }
    }

    point_types
}

/// Build the coarse-to-fine mapping and convert leftover undecided points
/// to C-points.
fn finalize_cf_splitting(point_types: &mut [PointType]) -> Vec<usize> {
    for pt in point_types.iter_mut() {
        if *pt == PointType::Undecided {
            *pt = PointType::Coarse;
        }
    }
    (0..point_types.len())
        .filter(|&i| point_types[i] == PointType::Coarse)
        .collect()
}

/// Classical Ruge-Stuben coarsening
///
/// Uses the standard two-pass algorithm with the transposed strength matrix
/// for efficient neighbor lookup. First pass selects C-points by influence
/// measure; second pass ensures every F-point has at least one strong C-neighbor.
///
/// See [`rs_first_pass`] for the first-pass details.
pub(super) fn coarsen_ruge_stuben<T: ComplexField>(
    _matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
) -> (Vec<PointType>, Vec<usize>) {
    let n = strong.len();
    let strong_t = build_strong_transpose(strong);
    let mut point_types = rs_first_pass(strong, &strong_t);

    // Second pass: ensure every F-point has at least one strong C-neighbor.
    // If not, promote it to C.
    for i in 0..n {
        if point_types[i] != PointType::Fine {
            continue;
        }
        let has_strong_c = strong[i]
            .iter()
            .any(|&j| point_types[j] == PointType::Coarse);
        if !has_strong_c {
            point_types[i] = PointType::Coarse;
        }
    }

    let coarse_to_fine = finalize_cf_splitting(&mut point_types);
    (point_types, coarse_to_fine)
}

/// Falgout coarsening (the hypre BoomerAMG default).
///
/// Combines the classical Ruge-Stuben first pass (dynamic influence measure,
/// see [`rs_first_pass`]) with the Falgout second pass: for each F-point `i`,
/// the candidate partners `j` are the *boundary* F-points in the union of
/// `S_i` (strong dependencies) and `S_i^T` (points strongly influenced by
/// `i`, i.e. influenced by an already-assigned C-point's F-set). If `i` and
/// `j` have no common strong C-neighbor, `j` is promoted to a C-point.
///
/// Sweeping `S_i^T` in addition to `S_i` is what distinguishes Falgout from
/// the classical RS second pass; it produces slightly larger C-sets but
/// interpolation stencils that cover both directions of strong coupling,
/// which is why BoomerAMG defaults to it.
pub(super) fn coarsen_falgout<T: ComplexField>(
    _matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
) -> (Vec<PointType>, Vec<usize>) {
    let n = strong.len();
    let strong_t = build_strong_transpose(strong);
    let mut point_types = rs_first_pass(strong, &strong_t);

    // Falgout second pass. `mark[c] == i` means "c is a strong C-neighbor of
    // the F-point i currently being examined".
    let mut mark = vec![usize::MAX; n];
    for i in 0..n {
        if point_types[i] != PointType::Fine {
            continue;
        }
        for &c in &strong[i] {
            if point_types[c] == PointType::Coarse {
                mark[c] = i;
            }
        }
        // Boundary F-points: strong F-neighbors in both the dependency (S_i)
        // and the influence (S_i^T) direction.
        for &j in strong[i].iter().chain(strong_t[i].iter()) {
            if j == i || point_types[j] != PointType::Fine {
                continue;
            }
            let has_common_c = strong[j].iter().any(|&c| mark[c] == i);
            if !has_common_c {
                // Promote j; it becomes a strong C-neighbor of i.
                point_types[j] = PointType::Coarse;
                mark[j] = i;
            }
        }
    }

    let coarse_to_fine = finalize_cf_splitting(&mut point_types);
    (point_types, coarse_to_fine)
}

/// Parallel Modified Independent Set (PMIS) coarsening
pub(super) fn coarsen_pmis<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
) -> (Vec<PointType>, Vec<usize>) {
    let n = matrix.num_rows;
    let mut point_types = vec![PointType::Undecided; n];

    // Compute weights based on number of strong connections
    let weights: Vec<f64> = (0..n)
        .map(|i| {
            // Weight = |S_i| + random tie-breaker
            strong[i].len() as f64 + (i as f64 * 0.0001) % 0.001
        })
        .collect();

    // Iterative independent set selection
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 100;

    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;

        // Parallel pass: determine new C-points and F-points
        let updates: Vec<(usize, PointType)> = parallel_enumerate_map(&point_types, |i, pt| {
            if *pt != PointType::Undecided {
                return (i, *pt);
            }

            // Check if i has maximum weight among undecided strong neighbors
            let mut is_max = true;
            for &j in &strong[i] {
                if point_types[j] == PointType::Undecided && weights[j] > weights[i] {
                    is_max = false;
                    break;
                }
            }

            // Check if any strong neighbor is already C
            let has_c_neighbor = strong[i]
                .iter()
                .any(|&j| point_types[j] == PointType::Coarse);

            if has_c_neighbor {
                (i, PointType::Fine)
            } else if is_max {
                (i, PointType::Coarse)
            } else {
                (i, PointType::Undecided)
            }
        });

        for (i, new_type) in updates {
            if point_types[i] != new_type {
                point_types[i] = new_type;
                changed = true;
            }
        }
    }

    // Any remaining undecided become coarse
    for pt in &mut point_types {
        if *pt == PointType::Undecided {
            *pt = PointType::Coarse;
        }
    }

    // Build coarse-to-fine mapping
    let coarse_to_fine: Vec<usize> = (0..n)
        .filter(|&i| point_types[i] == PointType::Coarse)
        .collect();

    (point_types, coarse_to_fine)
}

/// Hybrid MIS (HMIS) coarsening (Henson & Yang 2002)
///
/// HMIS combines the parallel scalability of PMIS with the interpolation
/// quality of classical Ruge-Stuben:
///
/// Ruge-Stuben second pass over F-F strong connections: for every F-point
/// `i` and each of its strong F-neighbors `j`, if `i` and `j` have no
/// common strong C-neighbor (no `c ∈ C` with `c ∈ S_i ∩ S_j`), then `j`
/// is promoted to a C-point. This guarantees the interpolation stencil
/// of every F-point covers its strong F-F connections.
///
/// Shared by [`coarsen_hmis`] and [`coarsen_cljp`]: without it, F-F pairs
/// whose members interpolate from *different* C-points inject stairstep
/// corrections (adjacent fine points taking values from distant coarse
/// points), which pollutes the residual with high-frequency spikes.
///
/// The pass is inherently sequential (promotions change the C-set observed
/// by later checks), mirroring textbook RS.
fn rs_cleanup_pass(point_types: &mut [PointType], strong: &[Vec<usize>]) {
    let n = strong.len();
    // `mark[c] == i` means "c is a strong C-neighbor of the F-point i
    // currently being examined".
    let mut mark = vec![usize::MAX; n];
    for i in 0..n {
        if point_types[i] != PointType::Fine {
            continue;
        }
        for &c in &strong[i] {
            if point_types[c] == PointType::Coarse {
                mark[c] = i;
            }
        }
        for &j in &strong[i] {
            if j == i || point_types[j] != PointType::Fine {
                continue;
            }
            let has_common_c = strong[j].iter().any(|&c| mark[c] == i);
            if !has_common_c {
                // Promote j; it becomes a strong C-neighbor of i.
                point_types[j] = PointType::Coarse;
                mark[j] = i;
            }
        }
    }
}

/// 1. **First pass — PMIS**: the parallel modified-independent-set selection
///    of [`coarsen_pmis`] produces an initial C/F splitting.
/// 2. **Second pass — RS cleanup**: [`rs_cleanup_pass`] is applied to the
///    remaining F-points, guaranteeing the interpolation stencil of every
///    F-point covers its strong F-F connections, which plain PMIS does not.
///
/// The second pass is inherently sequential (promotions change the C-set
/// observed by later checks), mirroring textbook RS.
pub(super) fn coarsen_hmis<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
) -> (Vec<PointType>, Vec<usize>) {
    let n = strong.len();

    // First pass: PMIS parallel selection.
    let (mut point_types, _) = coarsen_pmis(matrix, strong);

    // Second pass: Ruge-Stuben cleanup over F-F strong connections.
    rs_cleanup_pass(&mut point_types, strong);

    // Build coarse-to-fine mapping
    let coarse_to_fine: Vec<usize> = (0..n)
        .filter(|&i| point_types[i] == PointType::Coarse)
        .collect();

    (point_types, coarse_to_fine)
}

/// MIS-k aggregation coarsening
///
/// Generalized Maximal Independent Set at distance k:
/// 1. **Greedy MIS-k**: Process nodes by decreasing degree. For each unexcluded
///    node, mark as seed then BFS to distance k marking all reached as excluded.
/// 2. **BFS assignment**: Simultaneous BFS from all seeds -- each non-seed is
///    assigned to the aggregate of its nearest seed.
/// 3. **Singletons**: Any remaining unassigned nodes get their own aggregate.
///
/// Returns (aggregate_ids, num_aggregates) where aggregate_ids[i] is the
/// aggregate ID for node i.
pub(super) fn coarsen_mis_k(strong: &[Vec<usize>], k: usize) -> (Vec<usize>, usize) {
    let n = strong.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Phase 1: Greedy MIS-k seed selection
    // Process nodes by decreasing degree (number of strong connections).
    let mut is_seed = vec![false; n];
    let mut excluded = vec![false; n];

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| strong[b].len().cmp(&strong[a].len()).then(a.cmp(&b)));

    for &i in &order {
        if excluded[i] {
            continue;
        }

        // Make i a seed
        is_seed[i] = true;
        excluded[i] = true;

        // BFS from i to distance k, marking all reached nodes as excluded
        let mut queue = VecDeque::new();
        queue.push_back((i, 0usize));
        while let Some((node, dist)) = queue.pop_front() {
            if dist >= k {
                continue;
            }
            for &nbr in &strong[node] {
                if !excluded[nbr] {
                    excluded[nbr] = true;
                    queue.push_back((nbr, dist + 1));
                }
            }
        }
    }

    // Phase 2: Assign aggregate IDs to seeds
    let mut aggregate_ids = vec![usize::MAX; n];
    let mut num_aggregates = 0usize;

    for (i, &seed) in is_seed.iter().enumerate() {
        if seed {
            aggregate_ids[i] = num_aggregates;
            num_aggregates += 1;
        }
    }

    // Phase 3: Simultaneous BFS from all seeds to assign non-seed nodes
    // to the aggregate of their nearest seed.
    let mut queue = VecDeque::new();
    for (i, &seed) in is_seed.iter().enumerate() {
        if seed {
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        let agg = aggregate_ids[node];
        for &nbr in &strong[node] {
            if aggregate_ids[nbr] == usize::MAX {
                aggregate_ids[nbr] = agg;
                queue.push_back(nbr);
            }
        }
    }

    // Phase 4: Any remaining unassigned nodes get singleton aggregates
    for agg in &mut aggregate_ids {
        if *agg == usize::MAX {
            *agg = num_aggregates;
            num_aggregates += 1;
        }
    }

    (aggregate_ids, num_aggregates)
}

/// Aggregation-based coarsening for Smoothed Aggregation AMG
///
/// Thin wrapper around `coarsen_mis_k` that accepts the matrix parameter
/// for API consistency with the other coarsening functions.
///
/// Returns (aggregate_ids, num_aggregates) where aggregate_ids[i] is the
/// aggregate ID for node i.
pub(super) fn coarsen_aggregation<T: ComplexField>(
    _matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    mis_distance: usize,
) -> (Vec<usize>, usize) {
    coarsen_mis_k(strong, mis_distance)
}

/// CLJP (Cleary-Luby-Jones-Plassmann) parallel coarsening
///
/// A fully parallel C/F splitting that produces RS-quality coarse grids.
/// Based on the algorithm from HYPRE (Henson & Yang, 2002).
///
/// The algorithm:
/// 1. Compute influence measure `lambda[i] = |S_i^T|` (transposed strong connections)
/// 2. Add a deterministic hash-based tie-breaker for consistent parallel ordering
/// 3. Iterate: in parallel, mark as C any undecided point with maximum weight
///    among its undecided strong neighbours; mark as F any undecided point with
///    a strong C-neighbour. Update weights for remaining undecided points.
/// 4. Second pass: promote any F-point without a strong C-neighbour to C.
pub(super) fn coarsen_cljp<T: ComplexField>(
    _matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
) -> (Vec<PointType>, Vec<usize>) {
    let n = strong.len();
    let mut point_types = vec![PointType::Undecided; n];

    if n == 0 {
        return (point_types, Vec::new());
    }

    // Build transpose of strength matrix: strong_t[i] = {j : i ∈ S_j}
    let mut strong_t: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, row) in strong.iter().enumerate() {
        for &i in row {
            strong_t[i].push(j);
        }
    }

    // Compute weights: influence measure + deterministic hash tie-breaker.
    // The hash ensures a consistent total ordering even for nodes with equal
    // influence, which is critical for parallel convergence.
    let mut weights: Vec<f64> = (0..n)
        .map(|i| {
            let influence = strong_t[i].len() as f64;
            // Deterministic hash in (0, 1) for tie-breaking
            let hash = ((i.wrapping_mul(2654435761)) & 0xFFFFFFFF) as f64 / 4294967296.0;
            influence + hash
        })
        .collect();

    // Iterative parallel independent set selection
    let max_iter = 100;
    for _iter in 0..max_iter {
        let mut any_changed = false;

        // Phase 1: Identify new C-points (local maxima among undecided neighbours)
        // and new F-points (undecided with a strong C-neighbour).
        //
        // Snapshot point_types for consistent reads during parallel evaluation.
        let snapshot = point_types.clone();

        let updates: Vec<(usize, PointType)> = parallel_enumerate_map(&snapshot, |i, pt| {
            if *pt != PointType::Undecided {
                return (i, *pt);
            }

            // Check if any strong neighbour is already C
            let has_c_nbr = strong[i].iter().any(|&j| snapshot[j] == PointType::Coarse);

            if has_c_nbr {
                return (i, PointType::Fine);
            }

            // Check if i has the maximum weight among undecided strong neighbours
            let mut is_max = true;
            for &j in &strong[i] {
                if snapshot[j] == PointType::Undecided && weights[j] > weights[i] {
                    is_max = false;
                    break;
                }
            }
            // Also check transposed neighbours (points that strongly depend on i)
            if is_max {
                for &j in &strong_t[i] {
                    if snapshot[j] == PointType::Undecided && weights[j] > weights[i] {
                        is_max = false;
                        break;
                    }
                }
            }

            if is_max {
                (i, PointType::Coarse)
            } else {
                (i, PointType::Undecided)
            }
        });

        // Apply updates
        for (i, new_type) in updates {
            if point_types[i] != new_type {
                point_types[i] = new_type;
                any_changed = true;
            }
        }

        // Phase 2: Update weights for remaining undecided points.
        // When a neighbour j becomes C, all points that strongly depend on j
        // gain +1 influence (they now have one more strong C-neighbour nearby,
        // making them more likely to become F next iteration).
        for i in 0..n {
            if point_types[i] == PointType::Undecided {
                let new_c_nbrs = strong[i]
                    .iter()
                    .filter(|&&j| {
                        snapshot[j] == PointType::Undecided && point_types[j] == PointType::Coarse
                    })
                    .count();
                weights[i] += new_c_nbrs as f64;
            }
        }

        if !any_changed {
            break;
        }
    }

    // Second pass: RS cleanup over F-F strong connections. Without it,
    // F-F pairs whose members interpolate from different C-points inject
    // stairstep corrections (see [`rs_cleanup_pass`]).
    rs_cleanup_pass(&mut point_types, strong);

    // Third pass: ensure every F-point has at least one strong C-neighbour
    for i in 0..n {
        if point_types[i] == PointType::Fine {
            let has_strong_c = strong[i]
                .iter()
                .any(|&j| point_types[j] == PointType::Coarse);
            if !has_strong_c {
                point_types[i] = PointType::Coarse;
            }
        }
    }

    // Any remaining undecided become C-points
    for pt in &mut point_types {
        if *pt == PointType::Undecided {
            *pt = PointType::Coarse;
        }
    }

    // Build coarse-to-fine mapping
    let coarse_to_fine: Vec<usize> = (0..n)
        .filter(|&i| point_types[i] == PointType::Coarse)
        .collect();

    (point_types, coarse_to_fine)
}
