//! Lightweight Graph Neural Network (GNN) for AMG coarsening.
//!
//! Implements a message-passing network over the strong-connection graph
//! that predicts aggregate seeds from node features derived from the matrix
//! and its strength-of-connection graph.
//!
//! ## Architecture
//!
//! ```text
//! Input features per node (dim=4):
//!   [degree/max_degree, |a_ii|/max_diag, off_diag_ratio, hash(i)]
//!
//! Message passing (1 round over the symmetrized strong graph):
//!   agg_i = element-wise max of x_j over j ∈ N(i)   (GraphSAGE-max)
//!   d_i   = x_i - agg_i                              (edge-conv difference)
//!
//! Layer 1: h = ReLU(W1 * d_i + b1)   -> dim 8
//! Layer 2: p = sigmoid(W2 * h + b2)  -> dim 1
//! ```
//!
//! Total parameters: 4*8 + 8 + 8*1 + 1 = 49.
//!
//! The 4th feature is a deterministic hash of the node index. Because the
//! network sees the *difference* between a node's hash and the max hash of
//! its strong neighbors, it can (and, with the default weights, does)
//! implement "select strict local maxima of the hash" — a hash-priority
//! parallel MIS over the strong-connection graph, analogous to PMIS seed
//! selection. The resulting seeds form an independent set in the strong
//! graph: two adjacent seeds would each need a strictly larger hash than
//! the other, which is impossible.
//!
//! Nodes with p >= 0.5 are selected as seeds; remaining nodes are assigned
//! to the nearest seed via BFS (reusing the same logic as MIS-k Phase 3-4).
//!
//! ## Learned prolongation assessment (2026-07-28)
//!
//! Decision note on extending this module's ML approach from coarsening to
//! *learned prolongation* (GNN/ML AMG interpolation).
//!
//! **(a) What the literature gets from learned prolongations.**
//! - Greenfeld, Galun, Kimmel, Yavneh, Basri, "Learning to Optimize
//!   Multigrid PDE Solvers", ICML 2019 (arXiv:1902.10271): a GNN learns
//!   prolongation sparsity and weights for 2D Laplace-type problems,
//!   beating classical SA-AMG convergence factors on in-distribution grids.
//! - Luz, Galun, Maron, Basri, Yavneh, "Learning Algebraic Multigrid Using
//!   Graph Neural Networks", ICML 2020 (arXiv:2003.05744): unsupervised
//!   training against a Frobenius-norm surrogate of the two-grid error
//!   propagator; ~20-30% lower asymptotic convergence factors than
//!   classical AMG on Laplace and rotated-anisotropic diffusion, with the
//!   gain concentrated on anisotropic/stretched problems.
//! - Taghibakhshi, Nytko, Zaman, MacLachlan, Olson, West, "MG-GNN:
//!   Multigrid Graph Neural Networks for Learning Multilevel Domain
//!   Decomposition Methods", ICML 2023: learns the whole level-to-level
//!   map; similar magnitude of gain, plus setup-cost amortization
//!   (one training run, cheap inference per matrix).
//!
//! **(b) What energy-minimizing prolongation already captures for SPD.**
//! The crate's `AmgCoarsening::EnergyMinimizing` (MANA-K: MIS-k aggregation
//! plus column-wise projected CG on `p^T A p`, see
//! `interpolation::energy_minimizing_prolongation`, and Olson-Schroder-
//! Tuminaro, SISC 33(2), 2011, doi:10.1137/100805031) approximately
//! minimizes the *same* quantity the learned methods optimize: the A-energy
//! of prolongation columns, which bounds the two-grid convergence factor
//! for SPD A. For SPD problems the learned Frobenius surrogate is a proxy
//! for exactly this energy, so energy-min already captures most of the
//! attainable quality gain — plausibly the large majority (the Luz et al.
//! 20-30% edge is measured against *classical* interpolation, not against
//! energy-minimized P). What energy-min does not capture: (i) setup cost —
//! its per-column CG is O(iterations) SpMVs per coarse column, while a
//! trained GNN pays one forward pass; (ii) nonsymmetric/indefinite
//! operators, where `p^T A p` is not a norm and no clean energy objective
//! exists (there the learned surrogate and lAIR-style local ideals are the
//! only principled targets).
//!
//! **(c) Decision: implement-later.**
//! Decline a learned-prolongation implementation for SPD elliptic work
//! now: the energy-min path already sits near the same optimum, and a GNN
//! would add a training pipeline, a model-artifact dependency, and
//! generalization risk for a small quality delta. Revisit — i.e.
//! implement — only when *both* of these hold:
//! 1. The target problem class is strongly nonsymmetric or indefinite
//!    (advection-dominated transport, high-k Helmholtz) where energy-min
//!    has no valid objective and lAIR alone leaves a measured gap; and
//! 2. A benchmark on that class shows sonium AMG needing >1.25x the
//!    iterations of hypre/GAMG at equal complexity, confirming a genuine
//!    interpolation-quality gap rather than a tuning artifact.
//!
//! If only setup cost is the pain, prefer cheaper classical remedies first
//! (fewer energy-min iterations, pattern reuse across frequency sweeps).

use std::collections::VecDeque;

use num_traits::ToPrimitive;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Weights for the 2-layer message-passing coarsening network.
///
/// Layer 1: 4 -> 8 (ReLU), Layer 2: 8 -> 1 (sigmoid). The layer-1 input is
/// the edge-conv difference `d_i = x_i - max_agg(neighbors)` (see module
/// docs), so the network scores each node *relative to* its strong
/// neighbors.
///
/// The default weights are hand-tuned to produce MIS-like seeds
/// for 2D/3D elliptic PDE matrices (Poisson, diffusion). For better
/// results on specific problem classes, train weights offline in Python
/// and load them here.
#[derive(Debug, Clone)]
pub struct GnnCoarseningWeights {
    /// Layer 1 weight matrix, 4x8 row-major (4 inputs, 8 outputs).
    pub w1: [f64; 32],
    /// Layer 1 bias, 8 values.
    pub b1: [f64; 8],
    /// Layer 2 weight matrix, 8x1 (8 inputs, 1 output).
    pub w2: [f64; 8],
    /// Layer 2 bias, scalar.
    pub b2: f64,
}

impl Default for GnnCoarseningWeights {
    /// Weights implementing hash-priority parallel MIS seed selection.
    ///
    /// Neuron 0 computes `relu(1e4 * d_hash)` where `d_hash` is the
    /// difference between the node's hash and the maximum hash over its
    /// strong neighbors. It activates iff the node is a strict local
    /// maximum of the hash (the hash resolution is 1e-4, so a 1e4 gain
    /// turns the smallest positive gap into an O(1) activation). Layer 2
    /// then maps "neuron 0 active" to p >= 0.5. Remaining neurons are
    /// unused (zero weights) but keep the weight layout compatible with
    /// externally trained models that exploit the structural features.
    ///
    /// On a uniform 2D Poisson grid this selects ~1/(1+degree) of nodes as
    /// seeds (an independent set of the 5-point strong graph), producing
    /// coarsening ratios of ~4-6x, comparable to PMIS/CLJP.
    fn default() -> Self {
        Self {
            // W1: 4x8 row-major. Each column is one output neuron.
            // Rows: [degree, diag_dominance, off_diag_ratio, hash]
            #[rustfmt::skip]
            w1: [
                // feature 0 (degree) weights for neurons 0..8
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                // feature 1 (diag dominance) weights for neurons 0..8
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                // feature 2 (off-diag ratio) weights for neurons 0..8
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                // feature 3 (hash difference) — drives neuron 0
                 1.0e4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            b1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            w2: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            b2: -1.0,
        }
    }
}

/// Node features for the GCN: 4 floats per node.
struct NodeFeatures {
    /// features[i] = [degree_norm, diag_norm, offdiag_ratio, hash(i)]
    data: Vec<[f64; 4]>,
}

/// Deterministic hash of a node index to a value in [0, 1).
///
/// Uses a simple LCG-based mixing function for speed and reproducibility.
#[inline]
fn node_hash(i: usize) -> f64 {
    let mut h = i as u64;
    // Knuth's multiplicative hash
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h = (h >> 33) ^ h;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h = (h >> 33) ^ h;
    (h % 10000) as f64 / 10000.0
}

impl NodeFeatures {
    /// Extract per-node features from the matrix and strength graph.
    fn extract<T: ComplexField>(matrix: &CsrMatrix<T>, strong: &[Vec<usize>]) -> Self {
        let n = matrix.num_rows;

        // Compute raw features
        let mut degrees = vec![0usize; n];
        let mut diag_abs = vec![0.0f64; n];
        let mut offdiag_sum = vec![0.0f64; n];

        for i in 0..n {
            degrees[i] = strong[i].len();

            // Extract diagonal |a_ii|
            for idx in matrix.row_range(i) {
                if matrix.col_indices[idx] == i {
                    diag_abs[i] = matrix.values[idx].norm().to_f64().unwrap_or(0.0);
                    break;
                }
            }

            // Sum |a_ij| for j in strong neighbours
            for &j in &strong[i] {
                offdiag_sum[i] += matrix.get(i, j).norm().to_f64().unwrap_or(0.0);
            }
        }

        // Normalisation constants
        let max_degree = degrees.iter().copied().max().unwrap_or(1).max(1) as f64;
        let max_diag = diag_abs.iter().copied().fold(f64::MIN, f64::max).max(1e-15);

        let mut data = vec![[0.0f64; 4]; n];
        for i in 0..n {
            let d = diag_abs[i].max(1e-15);
            data[i] = [
                degrees[i] as f64 / max_degree,
                diag_abs[i] / max_diag,
                offdiag_sum[i] / d,
                node_hash(i),
            ];
        }

        Self { data }
    }
}

/// ReLU activation: max(0, x).
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Run the message-passing forward pass and return per-node seed probability.
///
/// One GraphSAGE-max message round over the *symmetrized* strong-connection
/// graph (edges in either direction, so seeds form an independent set with
/// respect to both `S_i` and `S_i^T`), followed by the 2-layer perceptron
/// applied to the edge-conv difference features `d_i = x_i - agg_i`.
fn gnn_forward(
    features: &NodeFeatures,
    strong: &[Vec<usize>],
    weights: &GnnCoarseningWeights,
) -> Vec<f64> {
    let n = features.data.len();

    // --- Message passing: symmetrized max-aggregation ---
    // agg[i][f] = max over j ∈ N(i) of x[j][f], where N(i) is the union of
    // strong[i] and the transposed strong neighbors {j : i ∈ S_j}.
    // The node itself is excluded (otherwise d = x - agg could never be
    // positive); nodes with no strong neighbors aggregate themselves
    // (d_i = 0).
    let mut agg = vec![[f64::NEG_INFINITY; 4]; n];
    let mut has_neighbor = vec![false; n];
    for (i, row) in strong.iter().enumerate() {
        for &j in row {
            // Edge i -> j used in both directions.
            let xj = features.data[j];
            let xi = features.data[i];
            for f in 0..4 {
                agg[i][f] = agg[i][f].max(xj[f]);
                agg[j][f] = agg[j][f].max(xi[f]);
            }
            has_neighbor[i] = true;
            has_neighbor[j] = true;
        }
    }
    for i in 0..n {
        if !has_neighbor[i] {
            agg[i] = features.data[i];
        }
    }

    // Edge-conv difference input: d = x - agg.
    // d[i][3] > 0 iff node i has a strictly larger hash than every strong
    // neighbor, i.e. it is a local maximum of the hash on the strong graph.
    let diff: Vec<[f64; 4]> = features
        .data
        .iter()
        .zip(agg.iter())
        .map(|(x, a)| {
            let mut d = [0.0f64; 4];
            for f in 0..4 {
                d[f] = x[f] - a[f];
            }
            d
        })
        .collect();

    // --- Layer 1: h = ReLU(d * W1 + b1) ---
    // W1 is 4x8 row-major: w1[f*8 + o] = weight from feature f to output o
    let mut h1 = vec![[0.0f64; 8]; n];
    for (h1_row, feat_row) in h1.iter_mut().zip(diff.iter()) {
        for (o, h_out) in h1_row.iter_mut().enumerate() {
            let mut val = weights.b1[o];
            for (f, &feat_f) in feat_row.iter().enumerate() {
                val += feat_f * weights.w1[f * 8 + o];
            }
            *h_out = relu(val);
        }
    }

    // --- Layer 2: p = sigmoid(h1 * W2 + b2) ---
    // W2 is 8x1: w2[o] = weight from hidden o to output
    let mut probs = vec![0.0f64; n];
    for (p, h1_row) in probs.iter_mut().zip(h1.iter()) {
        let mut val = weights.b2;
        for (&h_val, &w_val) in h1_row.iter().zip(weights.w2.iter()) {
            val += h_val * w_val;
        }
        *p = sigmoid(val);
    }

    probs
}

/// Test-only accessor: run feature extraction + forward pass and return the
/// raw seed mask (before the quantile fallback), so tests can verify the
/// independent-set property of the message-passing network.
#[cfg(test)]
pub(super) fn gnn_seed_mask<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    weights: &GnnCoarseningWeights,
) -> Vec<bool> {
    let features = NodeFeatures::extract(matrix, strong);
    let probs = gnn_forward(&features, strong, weights);
    probs.iter().map(|&p| p >= 0.5).collect()
}

/// GNN-driven coarsening.
///
/// 1. Extract node features from the matrix and strength graph.
/// 2. Run the message-passing forward pass to predict seed probabilities.
/// 3. Threshold at 0.5 to select seeds.
/// 4. BFS assignment from seeds to fill remaining nodes (same as MIS-k Phase 3-4).
///
/// Returns `(aggregate_ids, num_aggregates)`.
pub(super) fn gnn_coarsen<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    weights: &GnnCoarseningWeights,
) -> (Vec<usize>, usize) {
    let n = strong.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Step 1-2: feature extraction + message-passing forward pass
    let features = NodeFeatures::extract(matrix, strong);
    let probs = gnn_forward(&features, strong, weights);

    // Step 3: threshold to select seeds
    let mut is_seed = vec![false; n];
    let mut seed_count = 0usize;
    for i in 0..n {
        if probs[i] >= 0.5 {
            is_seed[i] = true;
            seed_count += 1;
        }
    }

    // Fallback: if no seeds or too many seeds were selected, use a
    // quantile-based approach targeting ~25% of nodes as seeds.
    if seed_count == 0 || seed_count >= n {
        let mut sorted_probs: Vec<f64> = probs.clone();
        sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let target = (n / 4).max(1).min(n - 1);
        let cutoff = sorted_probs[target];

        is_seed.fill(false);
        seed_count = 0;
        for i in 0..n {
            if probs[i] >= cutoff && seed_count < target + 1 {
                is_seed[i] = true;
                seed_count += 1;
            }
        }
    }

    // Step 4a: assign aggregate IDs to seeds
    let mut aggregate_ids = vec![usize::MAX; n];
    let mut num_aggregates = 0usize;

    for (i, &seed) in is_seed.iter().enumerate() {
        if seed {
            aggregate_ids[i] = num_aggregates;
            num_aggregates += 1;
        }
    }

    // Step 4b: BFS from all seeds to assign remaining nodes
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

    // Step 4c: singleton aggregates for any unassigned nodes
    for agg in &mut aggregate_ids {
        if *agg == usize::MAX {
            *agg = num_aggregates;
            num_aggregates += 1;
        }
    }

    (aggregate_ids, num_aggregates)
}
