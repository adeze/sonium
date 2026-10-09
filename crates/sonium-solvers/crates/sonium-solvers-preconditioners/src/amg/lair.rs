//! Local approximate ideal restriction (ℓAIR) for nonsymmetric AMG.
//!
//! Implements the restriction operator of
//!
//! > T. A. Manteuffel, J. Ruge, B. S. Southworth,
//! > "Nonsymmetric algebraic multigrid based on local approximate ideal
//! > restriction (ℓAIR)", SIAM J. Sci. Comput. 40(6), 2018.
//! > (arXiv:1708.06065)
//!
//! as shipped in hypre >= 2.19 (options AIR-1 / AIR-2).
//!
//! ## Construction
//!
//! Classical AMG uses the Galerkin restriction `R = P^T`, which is optimal
//! for SPD problems but fails on strongly nonsymmetric operators (upwinded
//! advection-diffusion, transport): the coarse operator `P^T A P` can lose
//! stability and the two-grid correction stalls or diverges.
//!
//! ℓAIR instead builds `R = [Z I]` row by row from the *ideal restriction*
//! identity. Requiring that error supported on F-points does not corrupt the
//! coarse-grid right-hand side, `R A [δe_f; 0] = 0`, gives the ideal
//! `Z = -A_cf A_ff^{-1}` (paper §2.3). That operator is dense, so ℓAIR
//! approximates each row locally: for the `i`-th C-point (the `i`-th row of
//! `R`), pick a restriction neighborhood `R_i` of nearby F-points (the
//! sparsity pattern of the row) and choose weights `z` so that a unit error
//! perturbation at any `j in R_i` leaves the restricted residual at `i`
//! unchanged (paper eq. 14):
//!
//! ```text
//! a_{i,j} + sum_{k in R_i} z_k a_{k,j} = 0   for all j in R_i
//! ```
//!
//! This is a *square* `|R_i| x |R_i|` dense system — the local `A_ff^T`
//! block `A(R_i, R_i)^T z = -A(i, R_i)^T` (paper eq. 15) — solved directly
//! with partial pivoting. Following the paper (§3.1), near-singular local
//! systems (rare, mostly on coarse levels) fall back to a ridge-regularized
//! least-squares solve, which selects an approximate minimal-norm solution.
//! Row `i` of `R` then has the unit entry at column `i` and the weights
//! `z_k` at columns `k in R_i`.
//!
//! ## Deviations from the paper / hypre
//!
//! - Neighborhoods are grown by BFS on the *symmetrized* strong-connection
//!   graph (the SOC matrix from `strength.rs`, symmetrized), distance 1 or 2
//!   (hypre's AIR-1 / AIR-2). The symmetrization makes the neighborhood
//!   independent of SOC direction for one-sided (upwind) problems.
//! - The optional `filter` drops weights with `|z_k| < filter * max_k |z_k|`
//!   (the unit entry is always kept). Rows are *not* renormalized: `R` is not
//!   an interpolation operator, so partition-of-unity renormalization would
//!   be wrong. hypre does not renormalize either.
//! - For complex matrices the fallback least-squares Gram matrix uses the
//!   conjugated form `M^H M` (matching hypre, which conjugates for Hermitian
//!   complex problems). The weights themselves are stored *unconjugated*:
//!   ideal restriction is the plain algebraic identity `R A [δe_f; 0] = 0`,
//!   not a Hermitian one.
//! - ℓAIR is built for the classical (RS/PMIS/HMIS/CLJP) coarsening paths.
//!   Aggregation-based paths keep `R = P^T`.
//! - AIR theory pairs with F-point (CF) relaxation. Set
//!   [`AmgConfig::f_point_smoothing`] to restrict pre/post smoothing to
//!   F-points (Jacobi / l1-Jacobi update F indices only; Gauss-Seidel sweeps
//!   F-point rows only). The default remains all-point smoothing, which is
//!   sufficient in practice for robust convergence (see the advection tests
//!   in `mod.rs`); [`AmgConfig::for_nonsymmetric`] enables CF relaxation.

use super::PointType;
use super::strength::symmetrize_strong;
use num_traits::{Float, FromPrimitive};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sys::parallel::parallel_map_indexed;

/// Build the ℓAIR restriction operator `R` (n_coarse x n).
///
/// - `matrix`: fine-level operator `A`.
/// - `strong`: strength-of-connection graph (from `strength.rs`).
/// - `point_types`: C/F classification of the fine points.
/// - `coarse_to_fine`: coarse index -> fine index map.
/// - `distance`: neighborhood distance in the symmetrized strong graph
///   (clamped to 1 or 2; hypre's AIR-1 / AIR-2).
/// - `filter`: optional relative drop tolerance on the off-unit weights.
pub(super) fn build_lair_restriction<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    point_types: &[PointType],
    coarse_to_fine: &[usize],
    distance: usize,
    filter: Option<f64>,
) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let n_coarse = coarse_to_fine.len();
    let distance = distance.clamp(1, 2);

    // Note: with the ℓAIR path the SOC graph is already symmetrized in
    // `compute_strength_matrix`; symmetrizing again is idempotent and keeps
    // this function correct when called with a one-sided graph directly.
    let sym_strong = symmetrize_strong(strong);

    // Each row of R is independent: it only reads A, the symmetrized SOC
    // graph, and the point classification.
    let per_row: Vec<Vec<(usize, T)>> = parallel_map_indexed(n_coarse, |ci| {
        lair_row(
            matrix,
            &sym_strong,
            point_types,
            coarse_to_fine[ci],
            distance,
            filter,
        )
    });

    let mut row_ptrs = vec![0usize; n_coarse + 1];
    for (ci, row) in per_row.iter().enumerate() {
        row_ptrs[ci + 1] = row_ptrs[ci] + row.len();
    }
    let total = row_ptrs[n_coarse];
    let mut col_indices = vec![0usize; total];
    let mut values = vec![T::zero(); total];
    let mut idx = 0;
    for row in per_row {
        for (c, v) in row {
            col_indices[idx] = c;
            values[idx] = v;
            idx += 1;
        }
    }

    CsrMatrix::from_raw_parts(n_coarse, n, row_ptrs, col_indices, values)
}

/// Compute one row of the ℓAIR restriction for C-point `i` (a fine index).
///
/// Returns sorted `(column, weight)` pairs: the unit entry at column `i`
/// plus the solved weights on the F-points of the neighborhood.
fn lair_row<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    sym_strong: &[Vec<usize>],
    point_types: &[PointType],
    i: usize,
    distance: usize,
    filter: Option<f64>,
) -> Vec<(usize, T)> {
    let n = matrix.num_rows;

    // BFS on the symmetrized strong graph up to `distance` hops.
    // `visited` marks neighborhood membership; `rpos` maps a fine index to
    // its position within `r_points` (the local unknown vector).
    let mut visited = vec![false; n];
    let mut neighborhood = vec![i];
    visited[i] = true;
    let mut frontier = vec![i];
    for _ in 0..distance {
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in &sym_strong[u] {
                if !visited[v] {
                    visited[v] = true;
                    neighborhood.push(v);
                    next.push(v);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    // Unknowns: weights on F-points in the neighborhood (the sparsity
    // pattern R_i of this row of Z).
    let mut r_points: Vec<usize> = Vec::new();
    let mut rpos = vec![usize::MAX; n];
    for &j in &neighborhood {
        if point_types[j] == PointType::Fine {
            rpos[j] = r_points.len();
            r_points.push(j);
        }
    }

    if r_points.is_empty() {
        return vec![(i, T::one())];
    }

    let p = r_points.len();

    // Square local system (paper eq. 15):
    //   sum_k A(r_k, r_j) z_k = -A(i, r_j)   for j = 0..p
    // i.e. M z = rhs with M[j][k] = A(r_k, r_j) (the local A_ff^T block).
    let mut m = vec![vec![T::zero(); p]; p];
    for (k, &rk) in r_points.iter().enumerate() {
        for (jj, v) in matrix.row_entries(rk) {
            let col = rpos[jj];
            if col != usize::MAX {
                m[col][k] = v;
            }
        }
    }
    let mut rhs = vec![T::zero(); p];
    for (jj, v) in matrix.row_entries(i) {
        let col = rpos[jj];
        if col != usize::MAX {
            rhs[col] = -v;
        }
    }

    let z = solve_local(m, rhs);

    // Filter small weights (relative to the largest weight) and assemble.
    let mut row: Vec<(usize, T)> = vec![(i, T::one())];
    if let Some(tau) = filter {
        let tau_r = T::Real::from_f64(tau).unwrap();
        let max_w = z.iter().fold(T::Real::from_f64(0.0).unwrap(), |acc, w| {
            let nrm = w.norm();
            if nrm > acc { nrm } else { acc }
        });
        let threshold = tau_r * max_w;
        for (k, &rk) in r_points.iter().enumerate() {
            if z[k].norm() >= threshold {
                row.push((rk, z[k]));
            }
        }
    } else {
        for (k, &rk) in r_points.iter().enumerate() {
            row.push((rk, z[k]));
        }
    }

    row.sort_by_key(|&(c, _)| c);
    row
}

/// Solve one small dense local system `M z = rhs`.
///
/// Tries a direct Gaussian elimination first. If the system is near-singular
/// (paper §3.1: "there have been situations on coarse levels in the hierarchy
/// where a local linear system is singular"), falls back to ridge-regularized
/// normal equations `(M^H M + eps I) z = M^H rhs`, which selects an
/// approximate minimal-norm least-squares solution.
fn solve_local<T: ComplexField>(m: Vec<Vec<T>>, rhs: Vec<T>) -> Vec<T> {
    if let Some(z) = solve_dense(m.clone(), rhs.clone()) {
        return z;
    }

    let p = rhs.len();
    let mut gram = vec![vec![T::zero(); p]; p];
    let mut grhs = vec![T::zero(); p];
    for k in 0..p {
        for (mj, &r) in m.iter().zip(rhs.iter()) {
            grhs[k] += mj[k].conj() * r;
        }
        for l in 0..=k {
            let mut g = T::zero();
            for mj in &m {
                g += mj[k].conj() * mj[l];
            }
            gram[k][l] = g;
            gram[l][k] = g.conj();
        }
    }
    let max_diag =
        gram.iter()
            .enumerate()
            .fold(T::Real::from_f64(0.0).unwrap(), |acc, (k, row)| {
                let d = row[k].norm();
                if d > acc { d } else { acc }
            });
    let ridge = max_diag * T::Real::from_f64(1e-8).unwrap() + T::Real::from_f64(1e-30).unwrap();
    for (k, row) in gram.iter_mut().enumerate() {
        row[k] += T::from_real(ridge);
    }

    // The ridge makes the Gram matrix nonsingular; if the solve still reports
    // a degenerate pivot, return the zero weights (row reduces to injection),
    // which is always a safe restriction.
    solve_dense(gram, grhs).unwrap_or_else(|| vec![T::zero(); p])
}

/// Solve a small dense system `A z = b` by Gaussian elimination with partial
/// pivoting. Returns `None` if a pivot is degenerate (near-singular matrix),
/// leaving the fallback to the caller.
///
/// The local systems are tiny (a handful to a few dozen unknowns), so an
/// O(p^3) direct solve is cheap.
fn solve_dense<T: ComplexField>(mut a: Vec<Vec<T>>, mut b: Vec<T>) -> Option<Vec<T>> {
    let p = b.len();
    let zero = T::Real::from_f64(0.0).unwrap();
    let max_diag = a
        .iter()
        .enumerate()
        .fold(zero, |acc, (k, row)| {
            let d = row[k].norm();
            if d > acc { d } else { acc }
        })
        .max(T::Real::from_f64(1e-30).unwrap());
    let pivot_tol = max_diag * T::Real::from_f64(1e-12).unwrap();

    for col in 0..p {
        // Partial pivot on magnitude.
        let mut pivot = col;
        let mut pivot_norm = a[col][col].norm();
        for (r, arow) in a.iter().enumerate().skip(col + 1) {
            let nrm = arow[col].norm();
            if nrm > pivot_norm {
                pivot = r;
                pivot_norm = nrm;
            }
        }
        if pivot_norm <= pivot_tol {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let diag = a[col][col];
        for r in (col + 1)..p {
            let factor = a[r][col] / diag;
            if factor.norm() <= pivot_tol {
                continue;
            }
            let (top, bottom) = a.split_at_mut(r);
            let pivot_row = &top[col];
            let target = &mut bottom[0];
            for cc in col..p {
                target[cc] -= factor * pivot_row[cc];
            }
            let b_col = b[col];
            b[r] -= factor * b_col;
        }
    }

    // Back substitution.
    let mut z = vec![T::zero(); p];
    for r in (0..p).rev() {
        let mut s = b[r];
        for (c, &zc) in z.iter().enumerate().skip(r + 1) {
            s -= a[r][c] * zc;
        }
        z[r] = s / a[r][r];
    }
    Some(z)
}
