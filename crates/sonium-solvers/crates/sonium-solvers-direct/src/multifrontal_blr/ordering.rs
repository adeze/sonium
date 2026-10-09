//! Nested-dissection ordering → assembly tree.
//!
//! Wraps `sonium_solvers_sparse::partition::nested_dissection` and produces:
//!
//! 1. A global elimination permutation `perm` such that `perm[k]` is the
//!    global DOF eliminated at step `k`. The permutation visits the
//!    dissection tree in post-order, so every node's fully-summed indices
//!    form a contiguous range `[fs_start .. fs_end)` in elimination order.
//! 2. A flat `Vec<FrontInfo>` parallel to the dissection-tree node array,
//!    carrying that range and the parent pointer.
//!
//! Down-stream consumers (symbolic, factorise, solve) need only
//! elimination-order indices and the parent map.

use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::partition::{DissectionNode, build_adjacency, nested_dissection};

/// Per-front structural data (elimination-order ranges + parent pointer).
#[derive(Debug, Clone)]
pub struct FrontInfo {
    /// Fully-summed elimination-order range `[fs_start .. fs_end)` for this front.
    pub fs_start: usize,
    pub fs_end: usize,
    /// Parent front index (`None` for the root).
    pub parent: Option<usize>,
    /// Children front indices.
    pub children: Vec<usize>,
    /// Level in the dissection tree (0 = leaves, root has the max level).
    pub level: usize,
}

impl FrontInfo {
    pub fn fs_size(&self) -> usize {
        self.fs_end - self.fs_start
    }
}

/// Output of [`build_assembly_tree`].
#[derive(Debug, Clone)]
pub struct AssemblyTree {
    /// `perm[k]` is the global DOF eliminated at step `k`.
    pub perm: Vec<usize>,
    /// `perm_inv[g]` is the elimination step for global DOF `g`.
    pub perm_inv: Vec<usize>,
    /// Per-front structural data, in factorisation order (post-order).
    pub fronts: Vec<FrontInfo>,
    /// Indices of fronts grouped by dissection-tree level (level 0 = leaves).
    /// Fronts in the same level have no parent/child relationship → safe to
    /// factorise in parallel.
    pub levels: Vec<Vec<usize>>,
}

/// Build the nested-dissection assembly tree for a square CSR matrix.
///
/// `min_block_size`: do not subdivide a domain smaller than this.
/// `max_levels`: cap the dissection depth (auto-computed if `None`).
pub fn build_assembly_tree<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    min_block_size: usize,
    max_levels: Option<usize>,
) -> AssemblyTree {
    let n = matrix.num_rows;
    debug_assert_eq!(n, matrix.num_cols);

    if n == 0 {
        return AssemblyTree {
            perm: Vec::new(),
            perm_inv: Vec::new(),
            fronts: Vec::new(),
            levels: Vec::new(),
        };
    }

    let max_levels = max_levels.unwrap_or_else(|| {
        if n <= min_block_size {
            0
        } else {
            ((n as f64) / (min_block_size as f64)).log2().ceil() as usize + 1
        }
    });

    let adjacency = build_adjacency(matrix);
    let all_indices: Vec<usize> = (0..n).collect();
    let mut nodes: Vec<DissectionNode> = Vec::new();
    nested_dissection(
        &adjacency,
        &all_indices,
        min_block_size,
        max_levels,
        0,
        &mut nodes,
    );

    // The dissection-tree `nodes` Vec is post-order by construction.
    // We can directly use it as the factorisation order.
    let num_fronts = nodes.len();

    // Build perm by concatenating each node's global_indices in node order.
    let mut perm: Vec<usize> = Vec::with_capacity(n);
    let mut fs_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_fronts);
    for node in &nodes {
        let start = perm.len();
        perm.extend_from_slice(&node.global_indices);
        fs_ranges.push((start, perm.len()));
    }
    debug_assert_eq!(perm.len(), n, "dissection didn't cover all DOFs");

    let mut perm_inv = vec![0usize; n];
    for (k, &g) in perm.iter().enumerate() {
        perm_inv[g] = k;
    }

    // Parent pointer: from each child's index, find the parent by scanning
    // for a node whose left/right field references it. Since we have ≤ n
    // nodes, the O(n²) scan is fine for the prototype.
    let mut parent = vec![None; num_fronts];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_fronts];
    for (pi, node) in nodes.iter().enumerate() {
        if let Some(l) = node.left {
            parent[l] = Some(pi);
            children[pi].push(l);
        }
        if let Some(r) = node.right {
            parent[r] = Some(pi);
            children[pi].push(r);
        }
    }

    let mut fronts: Vec<FrontInfo> = Vec::with_capacity(num_fronts);
    let mut max_level = 0usize;
    for (i, node) in nodes.iter().enumerate() {
        let (s, e) = fs_ranges[i];
        max_level = max_level.max(node.level);
        fronts.push(FrontInfo {
            fs_start: s,
            fs_end: e,
            parent: parent[i],
            children: children[i].clone(),
            level: node.level,
        });
    }

    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (idx, f) in fronts.iter().enumerate() {
        levels[f.level].push(idx);
    }

    AssemblyTree {
        perm,
        perm_inv,
        fronts,
        levels,
    }
}

/// Supernode amalgamation: fuse adjacent small fronts into larger ones so
/// the per-front `F11` GEMM has enough work to amortise BLAS overhead.
///
/// Greedy bottom-up rule: at each node `N`, sort children by their current
/// supernode size (smallest first) and absorb as many as fit under
/// `threshold`. An absorbed child's entire supernode joins `N`'s supernode;
/// a child whose supernode is already too big stays separate and remains a
/// child of `N`'s supernode in the new tree.
///
/// `threshold = 0` (or `threshold` smaller than every `fs_size`) is a no-op
/// — the returned tree is structurally equivalent to the input, modulo a
/// possible reordering of the perm to keep each supernode contiguous.
///
/// The returned tree is in factorisation post-order (each supernode's
/// descendants come before it) and its `fronts[i].fs_start..fs_end` ranges
/// are contiguous slices of the rebuilt `perm`.
pub fn amalgamate_tree(tree: &AssemblyTree, threshold: usize) -> AssemblyTree {
    let n = tree.fronts.len();
    if n == 0 {
        return tree.clone();
    }

    // ---- Pass 1: greedy bottom-up absorption decisions ------------------
    // `sn_size[i]` tracks the cumulative size of `i`'s supernode (own
    // fs_size plus the supernode sizes of every child it has already
    // absorbed). `absorbed[c] = true` means `c` was absorbed by its
    // immediate parent.
    let mut sn_size = vec![0usize; n];
    let mut absorbed = vec![false; n];
    for i in 0..n {
        let own = tree.fronts[i].fs_size();
        sn_size[i] = own;
        // Sort children by current supernode size, ascending — absorb the
        // small ones first; bigger ones may not fit.
        let mut child_order: Vec<usize> = tree.fronts[i].children.clone();
        child_order.sort_by_key(|&c| sn_size[c]);
        for c in child_order {
            let new_total = sn_size[i] + sn_size[c];
            if new_total <= threshold {
                absorbed[c] = true;
                sn_size[i] = new_total;
            }
        }
    }

    // ---- Pass 2: resolve every node's supernode root --------------------
    // `sn_root[i]` is the closest ancestor (or `i` itself) that was *not*
    // absorbed by its parent — i.e. the representative of the supernode
    // containing `i`.
    let mut sn_root = vec![0usize; n];
    for (i, root_slot) in sn_root.iter_mut().enumerate() {
        let mut cur = i;
        while absorbed[cur] {
            cur = tree.fronts[cur]
                .parent
                .expect("absorbed node must have a parent");
        }
        *root_slot = cur;
    }

    // ---- Pass 3: group original fronts by supernode root ---------------
    // Membership ordering inside each group matches the original post-order
    // (since we walk `i = 0..n` once and append). That carries through to
    // contiguous fs ranges inside each supernode's slice of the new perm.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, &root) in sn_root.iter().enumerate() {
        members[root].push(i);
    }

    // ---- Pass 4: emit the new tree in post-order ------------------------
    // Walk original fronts in post-order. When we hit a supernode root, we
    // emit one new front; otherwise we skip (the absorbed front's dofs are
    // emitted as part of its supernode root). Original post-order guarantees
    // that any supernode root's children-in-the-new-tree (themselves the
    // supernode roots of subtree branches not absorbed by this one) have
    // already been emitted and have a final `new_idx_of_sr` value.
    let mut new_idx_of_sr: Vec<Option<usize>> = vec![None; n];
    let mut new_perm: Vec<usize> = Vec::with_capacity(tree.perm.len());
    let mut new_fronts: Vec<FrontInfo> = Vec::new();

    for i in 0..n {
        if sn_root[i] != i {
            continue;
        }
        let new_idx = new_fronts.len();
        new_idx_of_sr[i] = Some(new_idx);

        let fs_start = new_perm.len();
        for &m in &members[i] {
            let f = &tree.fronts[m];
            new_perm.extend_from_slice(&tree.perm[f.fs_start..f.fs_end]);
        }
        let fs_end = new_perm.len();

        // Children of this supernode = supernode roots of any original child
        // belonging to a member of this supernode whose parent edge crosses
        // the supernode boundary (i.e., the child wasn't absorbed in).
        let mut child_indices: Vec<usize> = Vec::new();
        for &m in &members[i] {
            for &c in &tree.fronts[m].children {
                if sn_root[c] == c {
                    child_indices
                        .push(new_idx_of_sr[c].expect("child supernode emitted before its parent"));
                }
            }
        }

        new_fronts.push(FrontInfo {
            fs_start,
            fs_end,
            parent: None,
            children: child_indices.clone(),
            level: 0,
        });
        for &ci in &child_indices {
            new_fronts[ci].parent = Some(new_idx);
        }
    }

    // ---- Pass 5: recompute level (max child level + 1) and rebuild
    //              perm_inv / levels grouping. ----------------------------
    let mut max_level = 0usize;
    for i in 0..new_fronts.len() {
        let lv = if new_fronts[i].children.is_empty() {
            0
        } else {
            new_fronts[i]
                .children
                .iter()
                .map(|&c| new_fronts[c].level)
                .max()
                .unwrap()
                + 1
        };
        new_fronts[i].level = lv;
        if lv > max_level {
            max_level = lv;
        }
    }

    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (idx, f) in new_fronts.iter().enumerate() {
        levels[f.level].push(idx);
    }

    let mut perm_inv = vec![0usize; new_perm.len()];
    for (k, &g) in new_perm.iter().enumerate() {
        perm_inv[g] = k;
    }

    AssemblyTree {
        perm: new_perm,
        perm_inv,
        fronts: new_fronts,
        levels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_core::dense::DenseMatrix;
    use sonium_solvers_sparse::CsrMatrix;

    fn laplacian_2d(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
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

    fn assert_tree_is_postorder(tree: &AssemblyTree) {
        // Each front's children must appear before it in tree.fronts order.
        for (i, f) in tree.fronts.iter().enumerate() {
            for &c in &f.children {
                assert!(c < i, "child {} of front {} appears after parent", c, i);
            }
        }
    }

    fn assert_perm_is_partition(tree: &AssemblyTree) {
        // Every fs range [start, end) should tile [0, n) exactly once, and
        // perm/perm_inv should be true inverses on [0, n).
        let n = tree.perm.len();
        if n == 0 {
            return;
        }
        let mut covered = vec![false; n];
        for f in &tree.fronts {
            assert!(f.fs_start < f.fs_end);
            assert!(f.fs_end <= n);
            for k in f.fs_start..f.fs_end {
                assert!(!covered[k], "elim slot {} covered twice", k);
                covered[k] = true;
            }
        }
        for (k, c) in covered.iter().enumerate() {
            assert!(*c, "elim slot {} not covered by any front", k);
        }
        for g in 0..n {
            assert_eq!(tree.perm[tree.perm_inv[g]], g);
        }
    }

    #[test]
    fn threshold_zero_is_no_op() {
        // threshold = 0 (or any threshold below every fs_size) must leave
        // the front count unchanged. We still pass through `amalgamate_tree`
        // so check the structural invariants survive.
        let m = laplacian_2d(8);
        let raw = build_assembly_tree(&m, 8, None);
        let n_raw = raw.fronts.len();
        let amalgamated = amalgamate_tree(&raw, 0);
        assert_eq!(amalgamated.fronts.len(), n_raw);
        assert_tree_is_postorder(&amalgamated);
        assert_perm_is_partition(&amalgamated);
    }

    #[test]
    fn threshold_above_total_fuses_everything() {
        // With a threshold larger than the matrix, every front must merge
        // into a single supernode covering all DOFs.
        let m = laplacian_2d(6);
        let n = m.num_rows;
        let raw = build_assembly_tree(&m, 4, None);
        assert!(
            raw.fronts.len() > 1,
            "expected nested dissection to produce >1 front"
        );
        let amalgamated = amalgamate_tree(&raw, n * 10);
        assert_eq!(
            amalgamated.fronts.len(),
            1,
            "expected single supernode at huge threshold, got {}",
            amalgamated.fronts.len()
        );
        let only = &amalgamated.fronts[0];
        assert_eq!(only.fs_start, 0);
        assert_eq!(only.fs_end, n);
        assert!(only.parent.is_none());
        assert!(only.children.is_empty());
        assert_perm_is_partition(&amalgamated);
    }

    #[test]
    fn moderate_threshold_reduces_front_count() {
        // The interesting regime: threshold large enough to fuse leaves
        // with their separator parents, small enough to keep the top of
        // the tree intact. Front count must strictly decrease.
        let m = laplacian_2d(12);
        let raw = build_assembly_tree(&m, 8, None);
        let n_raw = raw.fronts.len();
        let amalgamated = amalgamate_tree(&raw, 64);
        assert!(
            amalgamated.fronts.len() < n_raw,
            "expected amalgamation to reduce front count: raw={} new={}",
            n_raw,
            amalgamated.fronts.len()
        );
        // Every supernode's fs_size respects the threshold OR consists of a
        // single original front that was bigger than the threshold all on
        // its own (singleton supernodes are allowed to exceed).
        // We can't easily distinguish those here without inspecting the
        // amalgamation decisions, but post-order + partition still hold.
        assert_tree_is_postorder(&amalgamated);
        assert_perm_is_partition(&amalgamated);
    }

    #[test]
    fn perm_remains_inverse_after_amalgamation() {
        let m = laplacian_2d(10);
        let raw = build_assembly_tree(&m, 6, None);
        let amalgamated = amalgamate_tree(&raw, 100);
        let n = amalgamated.perm.len();
        for g in 0..n {
            assert_eq!(g, amalgamated.perm[amalgamated.perm_inv[g]]);
            assert_eq!(g, amalgamated.perm_inv[amalgamated.perm[g]]);
        }
    }
}
