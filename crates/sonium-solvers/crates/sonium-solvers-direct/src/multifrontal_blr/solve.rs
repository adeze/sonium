//! Forward + backward substitution against a factored assembly tree.
//!
//! We walk fronts in factorisation order (post-order) for the forward pass:
//!
//!   for each front k:
//!     gather b at the front's eliminated FS rows + accumulated CB updates
//!     from children at cb(k)
//!     apply forward (L^{-1}) to the FS slice → z at the eliminated rows
//!     compute partial update at cb(k): u[cb] -= L21 · z[fs(k)]
//!     push u to parent via accumulator
//!
//! and reverse order (pre-order) for the backward pass:
//!
//!   for each front k from root down:
//!     read w at cb(k) from parent's update buffer
//!     w[fs(k)] = U^{-1} (z[fs(k)] - U12 · w[cb(k)])
//!
//! Implementation: we store per-front "fs_z" and "cb_acc" buffers. After the
//! forward pass, `z_global` holds z restricted to each front's eliminated FS
//! indices and `cb_global` holds the accumulated CB partial updates from the
//! front and its descendants. The parent extends-adds a child's cb_acc into
//! its own frontal-local index space before factorising z.
//!
//! Delayed pivoting note: a front's eliminated FS rows/cols are a
//! pivot-selected (possibly non-contiguous) subset of its assembled FS
//! block, so the gather/scatter goes through the front's pivot maps
//! (`pivots` / `col_pivots`) combined with `fs_elim_indices` rather than
//! assuming the eliminated indices form the contiguous range
//! `[fs_start, fs_start + nfs)`.

use num_traits::FromPrimitive;
use rayon::prelude::*;
use sonium_solvers_core::traits::ComplexField;

use super::front::FactoredFront;
use super::ordering::AssemblyTree;
use super::{BlrError, FrontError, scalar_is_finite};

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

struct ForwardFrontResult<T> {
    /// Elimination-order index of each entry of `fs_values`.
    fs_elim: Vec<usize>,
    fs_values: Vec<T>,
    cb_deltas: Vec<(usize, T)>,
}

struct BackwardFrontResult<T> {
    /// Elimination-order index of each entry of `fs_values`.
    fs_elim: Vec<usize>,
    fs_values: Vec<T>,
}

/// Scatter elimination-ordered RHS into per-front buffers, run forward +
/// backward, and gather back into a global elimination-ordered solution.
pub fn forward_backward_solve<T: ComplexField + FromPrimitive>(
    tree: &AssemblyTree,
    fronts: &[FactoredFront<T>],
    b_perm: &[T],
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

    // Initialise: cb_global = b_perm, z_global = 0. At each front, the
    // "input b" for the front's FS rows is cb_global at those indices (which
    // started as b_perm). The CB updates accumulate into cb_global at CB
    // indices, so by the time the parent processes its FS, those FS indices
    // (which were the parent's CB indices when this front ran) carry the
    // sum of original b + child CB contributions.
    cb_global[..n].copy_from_slice(&b_perm[..n]);

    // Forward pass: levels are ordered leaves -> root. Same-level fronts have
    // no dependencies, so their dense local solves and panel matvecs run in
    // parallel. CB writes can overlap at ancestors, so each front returns an
    // additive delta that is reduced deterministically after the parallel work.
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
                        stage: "contribution accumulation",
                        index: g,
                    });
                }
            }
        }
    }

    // Backward pass: root -> leaves. Same-level fronts read already-solved
    // ancestor CB values and write disjoint FS ranges.
    let mut w_global = vec![T::zero(); n];
    for level in tree.levels.iter().rev() {
        let results: Vec<(usize, Result<BackwardFrontResult<T>, FrontError>)> = if level.len() > 1 {
            level
                .par_iter()
                .map(|&k| (k, backward_front(fronts, &z_global, &w_global, k)))
                .collect()
        } else {
            level
                .iter()
                .map(|&k| (k, backward_front(fronts, &z_global, &w_global, k)))
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
    fronts: &[FactoredFront<T>],
    cb_global: &[T],
    k: usize,
) -> Result<ForwardFrontResult<T>, FrontError> {
    let ff = fronts
        .get(k)
        .ok_or(FrontError::Invalid("front index is out of range"))?;
    // No per-solve validate anywhere on this path: fronts were validated
    // at factorize-pack time (`factored.validate()`) and are immutable
    // afterwards, so the apply below uses the `assume_validated` entry.
    // (Saves one O(nfs)-class metadata pass per front per apply.)
    let nfs = ff.nfs;
    let ncb = ff.ncb;

    // Gather the eliminated FS rows in factored (pivoted) order:
    // step i eliminates assembled-FS position `pivots[i]`, whose
    // elimination-order index is `fs_elim_indices[pivots[i]]`.
    let mut b_local = vec![T::zero(); nfs + ncb];
    let mut fs_elim = Vec::with_capacity(nfs);
    for (i, &pivot) in ff.pivots.iter().enumerate().take(nfs) {
        let g = ff.fs_elim_indices[pivot];
        if g >= cb_global.len() {
            return Err(FrontError::Invalid(
                "eliminated index is out of global range",
            ));
        }
        b_local[i] = cb_global[g];
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

    ff.forward_apply_assume_validated(&mut b_local)?;

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
    fronts: &[FactoredFront<T>],
    z_global: &[T],
    w_global: &[T],
    k: usize,
) -> Result<BackwardFrontResult<T>, FrontError> {
    let ff = fronts
        .get(k)
        .ok_or(FrontError::Invalid("front index is out of range"))?;
    // No second validate here either (see forward_front above).
    let nfs = ff.nfs;
    let ncb = ff.ncb;

    // Gather z at the eliminated FS rows in factored order (same map as
    // the forward pass).
    let mut b_local = vec![T::zero(); nfs + ncb];
    for (i, &pivot) in ff.pivots.iter().enumerate().take(nfs) {
        let global = ff.fs_elim_indices[pivot];
        if global >= z_global.len() {
            return Err(FrontError::Invalid(
                "eliminated index is out of global range",
            ));
        }
        b_local[i] = z_global[global];
    }
    for (j, &g) in ff.cb_indices.iter().enumerate() {
        if g >= w_global.len() {
            return Err(FrontError::Invalid(
                "contribution index is out of global range",
            ));
        }
        b_local[nfs + j] = w_global[g];
    }

    ff.backward_apply_assume_validated(&mut b_local)?;

    // `backward_apply` leaves b[0..nfs] in factored column order: step i
    // holds the unknown of assembled-FS position `col_pivots[i]`, whose
    // elimination-order index is `fs_elim_indices[col_pivots[i]]`.
    let mut fs_elim = Vec::with_capacity(nfs);
    let mut fs_values = Vec::with_capacity(nfs);
    for (i, &pivot) in ff.col_pivots.iter().enumerate().take(nfs) {
        fs_elim.push(ff.fs_elim_indices[pivot]);
        fs_values.push(b_local[i]);
    }

    Ok(BackwardFrontResult { fs_elim, fs_values })
}
