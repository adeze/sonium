//! Supernodal (BLAS-3) numeric factorization for sparse Cholesky.
//!
//! Reuses the symbolic phase of [`super::sparse_cholesky`] (nested-dissection
//! ordering, elimination tree, exact nonzero pattern of `L`) and replaces the
//! up-looking, column-at-a-time numeric loop with a **right-looking**
//! supernodal factorization:
//!
//! 1. **Relaxed supernode detection** on the factor pattern. Columns are
//!    greedily amalgamated into contiguous ranges (mirroring the relaxed
//!    panels of [`super::sparse_supernodal`] for LU): a column joins the
//!    current supernode while the width stays under a cap and the structural
//!    zeros of the dense trapezoid stay within a fill budget. The panel row
//!    set is the *union* of its columns' patterns below the supernode, so
//!    any contiguous merge is correct — etree parent chains (fundamental
//!    supernodes, `struct(L(:,j)) ⊆ struct(L(:,j+1)) ∪ {j+1}`) merge at zero
//!    extra fill, and near-compatible columns are absorbed where the fill
//!    test allows.
//! 2. **Per-supernode dense panels**. Every supernode owns a dense row-major
//!    accumulator of size `(width + |rows|) × width`, initialised from the
//!    permuted upper triangle of `A`.
//! 3. **Right-looking updates**. Supernodes are processed in increasing
//!    column order; when a supernode is reached its panel already holds
//!    `A` minus every ancestor contribution. The panel (diagonal block +
//!    below-diagonal block) is factored by a blocked right-looking sweep
//!    with block size [`PANEL_BLOCK`]: each column block receives the update
//!    of all previous blocks as one [`ComplexField::mat_mul_nn`] GEMM — the
//!    SME2/Accelerate-dispatching kernel the supernodal LU uses — its small
//!    diagonal tile is factored scalar (same loop structure as
//!    [`super::cholesky`]), and the rows below the tile are solved against
//!    it. The trailing update `S = L_BD·L_BDᴴ` (or `L_BD·L_BDᵀ`) is formed
//!    with one more GEMM and scattered into the panels of the ancestor
//!    supernodes.
//! 4. The finished panel columns are written back into the simplicial CSC
//!    layout, so the triangular solves and the symbolic-reuse contract of
//!    [`super::sparse_cholesky`] apply unchanged.
//!
//! The factorization is sequential and deterministic.

use super::sparse_cholesky::{SparseCholeskyError, SparseCholeskySymbolic, UpperTriangle};
use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::traits::ComplexField;

/// Maximum columns merged into one supernode (mirrors the LU panel cap,
/// scaled up: Cholesky panels have no pivoting and wide GEMMs pay off).
const MAX_SUPERNODE_WIDTH: usize = 128;

/// Column-block width of the blocked panel factorization. Blocks at or below
/// this width stay scalar; wider panels route their trailing updates through
/// [`ComplexField::mat_mul_nn`].
const PANEL_BLOCK: usize = 32;

/// Relaxation bound: a chain link may be merged into a supernode when the
/// structural zeros the dense trapezoid introduces do not exceed
/// `RELAX_FILL_NUM / RELAX_FILL_DEN` times the supernode's actual entries.
const RELAX_FILL_NUM: usize = 1;
const RELAX_FILL_DEN: usize = 4;

/// Relaxed-supernode partition of the factor pattern.
///
/// Supernodes are contiguous column ranges covering `0..n` in order. Each
/// carries the sorted union of its columns' below-supernode pattern rows —
/// the row set of its dense trapezoidal panel.
#[derive(Debug, Clone)]
pub struct SupernodeMap {
    /// `(start, end)` column range per supernode (`end` exclusive).
    ranges: Vec<(usize, usize)>,
    /// Sorted union of rows `>= end` over the supernode's column patterns.
    rows: Vec<Vec<usize>>,
    /// Supernode index owning each column.
    sup_of_col: Vec<usize>,
}

impl SupernodeMap {
    /// Detect relaxed supernodes from the factor pattern of a symbolic
    /// analysis.
    ///
    /// Greedy amalgamation: starting from column `s`, keep absorbing the next
    /// column while the supernode stays under [`MAX_SUPERNODE_WIDTH`] and the
    /// structural zeros the dense trapezoid introduces stay within
    /// [`RELAX_FILL_NUM`] / [`RELAX_FILL_DEN`] of the actual entry count. The
    /// below-row set is the *union* of the absorbed columns' patterns, so any
    /// contiguous merge is correct — etree parent chains (fundamental
    /// supernodes) merge at zero extra fill, and near-compatible siblings are
    /// absorbed where the fill test allows.
    pub fn analyze(symbolic: &SparseCholeskySymbolic) -> Self {
        let n = symbolic.dimension();
        let ptrs = symbolic.factor_col_ptrs();
        let row_indices = symbolic.factor_row_indices();
        let count = |j: usize| ptrs[j + 1] - ptrs[j]; // includes the diagonal
        let below = |j: usize| &row_indices[ptrs[j] + 1..ptrs[j + 1]];

        let mut ranges = Vec::new();
        let mut rows = Vec::new();
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            // Rows strictly below the current `end` in any absorbed column,
            // sorted. Initialised from column `start`.
            let mut union: Vec<usize> = below(start).to_vec();
            let mut actual = count(start) - 1; // below-diagonal entries so far
            while end < n && end - start < MAX_SUPERNODE_WIDTH {
                // Candidate supernode [start, end + 1): `end` joins the
                // diagonal block, so it leaves the union, and column `end`'s
                // below rows merge in.
                let mut next_union: Vec<usize> = Vec::with_capacity(union.len() + count(end) - 1);
                let mut u = union.iter().copied().skip_while(|&r| r <= end).peekable();
                let mut b = below(end).iter().copied().peekable();
                loop {
                    match (u.peek(), b.peek()) {
                        (Some(&x), Some(&y)) => {
                            if x <= y {
                                next_union.push(x);
                                u.next();
                                if x == y {
                                    b.next();
                                }
                            } else {
                                next_union.push(y);
                                b.next();
                            }
                        }
                        (Some(_), None) => {
                            next_union.extend(u);
                            break;
                        }
                        (None, _) => {
                            next_union.extend(b);
                            break;
                        }
                    }
                }
                let width = end + 1 - start;
                let dense = width * (width - 1) / 2 + width * next_union.len();
                let next_actual = actual + count(end) - 1;
                if (dense - next_actual) * RELAX_FILL_DEN > RELAX_FILL_NUM * next_actual {
                    break;
                }
                union = next_union;
                actual = next_actual;
                end += 1;
            }
            // Shrink the union to rows below the final `end`.
            let first_below = union.partition_point(|&r| r < end);
            union.drain(..first_below);
            ranges.push((start, end));
            rows.push(union);
            start = end;
        }

        let mut sup_of_col = vec![0usize; n];
        for (index, &(start, end)) in ranges.iter().enumerate() {
            for slot in &mut sup_of_col[start..end] {
                *slot = index;
            }
        }
        Self {
            ranges,
            rows,
            sup_of_col,
        }
    }

    /// Number of supernodes.
    pub fn count(&self) -> usize {
        self.ranges.len()
    }

    /// Width (number of columns) of the widest supernode.
    pub fn max_width(&self) -> usize {
        self.ranges
            .iter()
            .map(|&(start, end)| end - start)
            .max()
            .unwrap_or(0)
    }

    /// Column range `(start, end)` of supernode `index`.
    pub fn range(&self, index: usize) -> (usize, usize) {
        self.ranges[index]
    }

    /// Width of supernode `index`.
    pub fn width(&self, index: usize) -> usize {
        let (start, end) = self.ranges[index];
        end - start
    }

    /// Below-diagonal row set of supernode `index`.
    pub fn rows(&self, index: usize) -> &[usize] {
        &self.rows[index]
    }

    /// Total entries stored across all dense panels, including the
    /// structural zeros introduced by relaxation.
    pub fn dense_panel_entries(&self) -> usize {
        self.ranges
            .iter()
            .zip(&self.rows)
            .map(|(&(start, end), rows)| {
                let width = end - start;
                (width + rows.len()) * width
            })
            .sum()
    }
}

/// Right-looking supernodal numeric Cholesky.
///
/// Fills the CSC `L` values laid out by `symbolic` (diagonal first per
/// column) from dense per-supernode panels. Reports
/// [`SparseCholeskyError::NotPositiveDefinite`] on a non-positive pivot with
/// the same scale-aware tolerance as the simplicial path.
pub(crate) fn numeric_factorize<T: ComplexField>(
    upper: &UpperTriangle<T>,
    symbolic: &SparseCholeskySymbolic,
    map: &SupernodeMap,
    hermitian: bool,
) -> Result<Vec<T>, SparseCholeskyError> {
    let n = symbolic.dimension();
    let ptrs = symbolic.factor_col_ptrs();
    let l_rows = symbolic.factor_row_indices();
    let mut l_values = vec![T::zero(); l_rows.len()];
    if n == 0 {
        return Ok(l_values);
    }

    // Pivot tolerance relative to the diagonal scale (same rule as the
    // simplicial factorization and the dense Cholesky).
    let scale = (0..n)
        .map(|k| {
            let mut diag = T::Real::zero();
            for p in upper.col_ptrs[k]..upper.col_ptrs[k + 1] {
                if upper.row_indices[p] == k {
                    diag = upper.values[p].norm();
                }
            }
            diag
        })
        .fold(T::Real::zero(), T::Real::max)
        .max(T::Real::min_positive_value());
    let tolerance =
        T::Real::epsilon() * T::Real::from_usize(n.max(1)).unwrap_or_else(T::Real::one) * scale;

    // Dense row-major panel per supernode: rows are the supernode columns
    // followed by its below-diagonal row set, one panel column per supernode
    // column. `panel[pos * width + local]` holds the accumulator entry.
    let mut panels: Vec<Vec<T>> = (0..map.count())
        .map(|index| {
            let width = map.width(index);
            vec![T::zero(); (width + map.rows(index).len()) * width]
        })
        .collect();

    // Scratch buffers reused across supernodes and panel blocks.
    let mut left: Vec<T> = Vec::new(); // panel-block GEMM operand
    let mut right: Vec<T> = Vec::new(); // panel-block GEMM operand (transposed)
    let mut product: Vec<T> = Vec::new(); // panel-block GEMM result
    let mut update_block: Vec<T> = Vec::new(); // L_BD copy
    let mut update_transposed: Vec<T> = Vec::new(); // L_BD^H (or ^T)
    let mut schur: Vec<T> = Vec::new(); // rank-k update
    let mut pos_map: Vec<usize> = Vec::new(); // row positions in a target panel

    // Scatter the permuted upper triangle of A into the panels: entry
    // (i, j) with i <= j lands at panel row j, column i of the supernode
    // owning column i (conjugated for the Hermitian form, since the panel
    // holds the lower triangle).
    for j in 0..n {
        for p in upper.col_ptrs[j]..upper.col_ptrs[j + 1] {
            let i = upper.row_indices[p];
            debug_assert!(i <= j);
            let target = map.sup_of_col[i];
            let (start, end) = map.range(target);
            let width = end - start;
            let pos = if j < end {
                j - start
            } else {
                width
                    + map
                        .rows(target)
                        .binary_search(&j)
                        .expect("panel row set covers the factor pattern")
            };
            let value = upper.values[p];
            panels[target][pos * width + (i - start)] =
                if hermitian { value.conj() } else { value };
        }
    }

    for index in 0..map.count() {
        let (start, end) = map.range(index);
        let width = end - start;
        let below_rows = map.rows(index).len();
        let panel_rows = width + below_rows;

        // Blocked right-looking factorization of the dense trapezoidal panel
        // (diagonal block + below-diagonal block). Each column block first
        // receives the accumulated update of all previous blocks as one
        // GEMM, then its b x b diagonal tile is factored scalar (unblocked,
        // same loop structure as `super::cholesky`) and the rows below the
        // tile are solved against it.
        for j0 in (0..width).step_by(PANEL_BLOCK) {
            let j1 = (j0 + PANEL_BLOCK).min(width);
            let block = j1 - j0;

            if j0 > 0 {
                // P[j0..m, j0..j1] -= L[j0..m, 0..j0] · L[j0..j1, 0..j0]^H
                // (or ^T). `mat_mul_nn` needs contiguous row-major operands,
                // so the strided panel blocks are copied out first.
                let rows = panel_rows - j0;
                left.resize(rows * j0, T::zero());
                right.resize(j0 * block, T::zero());
                product.resize(rows * block, T::zero());
                {
                    let panel = &panels[index];
                    for r in 0..rows {
                        left[r * j0..(r + 1) * j0]
                            .copy_from_slice(&panel[(j0 + r) * width..(j0 + r) * width + j0]);
                    }
                    for c in 0..block {
                        for k in 0..j0 {
                            let value = panel[(j0 + c) * width + k];
                            right[k * block + c] = if hermitian { value.conj() } else { value };
                        }
                    }
                }
                T::mat_mul_nn(rows, block, j0, &left, &right, &mut product);
                let panel = &mut panels[index];
                for r in 0..rows {
                    let row = &mut panel[(j0 + r) * width + j0..(j0 + r) * width + j1];
                    let update = &product[r * block..(r + 1) * block];
                    for (cell, delta) in row.iter_mut().zip(update) {
                        *cell -= *delta;
                    }
                }
            }

            let panel = &mut panels[index];

            // Factor the block x block diagonal tile (scalar; pivot checks
            // use the same scale-aware tolerance as the simplicial path).
            for j in j0..j1 {
                let mut diagonal = panel[j * width + j];
                for k in j0..j {
                    let ljk = panel[j * width + k];
                    diagonal -= if hermitian {
                        ljk * ljk.conj()
                    } else {
                        ljk * ljk
                    };
                }
                if !diagonal.re().is_finite() || !diagonal.im().is_finite() {
                    return Err(SparseCholeskyError::NumericalBreakdown {
                        stage: "supernodal pivot formation",
                        index: start + j,
                    });
                }
                if diagonal.im().abs() > tolerance || diagonal.re() <= tolerance {
                    return Err(SparseCholeskyError::NotPositiveDefinite { pivot: start + j });
                }
                let sqrt = T::from_real(diagonal.re().sqrt());
                panel[j * width + j] = sqrt;
                let inv = sqrt.inv();
                for i in (j + 1)..j1 {
                    let mut value = panel[i * width + j];
                    for k in j0..j {
                        let right = if hermitian {
                            panel[j * width + k].conj()
                        } else {
                            panel[j * width + k]
                        };
                        value -= panel[i * width + k] * right;
                    }
                    panel[i * width + j] = value * inv;
                }
            }

            // Solve the block column below the tile against it:
            // P[j1..m, j0..j1] = P[j1..m, j0..j1] · L_tile⁻ᴴ (or · L_tile⁻ᵀ).
            for i in j1..panel_rows {
                for j in j0..j1 {
                    let mut value = panel[i * width + j];
                    for k in j0..j {
                        let right = if hermitian {
                            panel[j * width + k].conj()
                        } else {
                            panel[j * width + k]
                        };
                        value -= panel[i * width + k] * right;
                    }
                    panel[i * width + j] = value * panel[j * width + j].inv();
                }
            }
        }

        if below_rows > 0 {
            // Rank-k trailing update S = L_BD · L_BDᴴ, scattered into the
            // ancestor panels. Large panels form S with one BLAS-3 GEMM;
            // small panels use direct dot products to avoid the GEMM call
            // and packing overhead dominating their tiny workloads.
            update_block.resize(below_rows * width, T::zero());
            update_block
                .copy_from_slice(&panels[index][width * width..(width + below_rows) * width]);
            let dense_schur = width >= 8 && below_rows >= 16;
            if dense_schur {
                update_transposed.resize(width * below_rows, T::zero());
                for i in 0..below_rows {
                    for k in 0..width {
                        let value = update_block[i * width + k];
                        update_transposed[k * below_rows + i] =
                            if hermitian { value.conj() } else { value };
                    }
                }
                schur.resize(below_rows * below_rows, T::zero());
                T::mat_mul_nn(
                    below_rows,
                    below_rows,
                    width,
                    &update_block,
                    &update_transposed,
                    &mut schur,
                );
            }

            // The targets of the update are contiguous runs of the sorted
            // row set (supernode ranges are ascending). Resolve each run's
            // row positions in its target panel once with a merge join, then
            // scatter with direct indexing.
            let self_rows = &map.rows[index];
            pos_map.resize(below_rows, 0);
            let mut run_start = 0;
            while run_start < below_rows {
                let target = map.sup_of_col[self_rows[run_start]];
                debug_assert!(target > index, "updates only travel up the etree");
                let (target_start, target_end) = map.range(target);
                let target_width = target_end - target_start;
                let target_rows = map.rows(target);
                let mut run_end = run_start + 1;
                while run_end < below_rows && map.sup_of_col[self_rows[run_end]] == target {
                    run_end += 1;
                }
                // Position of every row self_rows[run_start..] in the target
                // panel: contiguous for rows < target_end, then a forward
                // scan of the target's sorted below-rows. With amalgamated
                // (non-nested) supernodes some pairs (r, c) are structurally
                // zero — r is not in column c's pattern, hence not in the
                // target panel; their update is exactly zero and is skipped
                // (usize::MAX sentinel).
                let mut cursor = 0usize;
                for (slot, &row) in self_rows.iter().enumerate().skip(run_start) {
                    let pos = if row < target_end {
                        row - target_start
                    } else {
                        while cursor < target_rows.len() && target_rows[cursor] < row {
                            cursor += 1;
                        }
                        if cursor < target_rows.len() && target_rows[cursor] == row {
                            target_width + cursor
                        } else {
                            usize::MAX
                        }
                    };
                    pos_map[slot] = pos;
                }
                for slot in run_start..run_end {
                    let column = self_rows[slot];
                    let target_panel = &mut panels[target];
                    for row_slot in slot..below_rows {
                        let pos = pos_map[row_slot];
                        let value = if dense_schur {
                            schur[row_slot * below_rows + slot]
                        } else {
                            let mut dot = T::zero();
                            for k in 0..width {
                                let a = update_block[row_slot * width + k];
                                let b = update_block[slot * width + k];
                                dot += if hermitian { a * b.conj() } else { a * b };
                            }
                            dot
                        };
                        if pos == usize::MAX {
                            debug_assert_eq!(
                                value,
                                T::zero(),
                                "structurally zero update outside the factor pattern"
                            );
                            continue;
                        }
                        target_panel[pos * target_width + (column - target_start)] -= value;
                    }
                }
                run_start = run_end;
            }
        }

        // Write the finished columns back into the simplicial CSC layout.
        let panel = &panels[index];
        for (local, global) in (start..end).enumerate() {
            for p in ptrs[global]..ptrs[global + 1] {
                let row = l_rows[p];
                let pos = if row < end {
                    row - start
                } else {
                    width
                        + map
                            .rows(index)
                            .binary_search(&row)
                            .expect("panel row set covers the factor pattern")
                };
                l_values[p] = panel[pos * width + local];
            }
        }
    }

    Ok(l_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_cholesky::{CholeskyOrdering, SparseCholeskyFactorization};
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

    /// 5-point 2D Laplacian on an `nx × nx` grid (SPD).
    fn laplacian_2d(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut triplets = Vec::with_capacity(5 * n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, 4.0));
                if ix > 0 {
                    triplets.push((i, i - 1, -1.0));
                }
                if ix + 1 < nx {
                    triplets.push((i, i + 1, -1.0));
                }
                if iy > 0 {
                    triplets.push((i, i - nx, -1.0));
                }
                if iy + 1 < nx {
                    triplets.push((i, i + nx, -1.0));
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// 7-point 3D Laplacian on an `nx³` grid (SPD).
    fn laplacian_3d(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx * nx;
        let mut triplets = Vec::with_capacity(7 * n);
        for iz in 0..nx {
            for iy in 0..nx {
                for ix in 0..nx {
                    let i = (iz * nx + iy) * nx + ix;
                    triplets.push((i, i, 6.0));
                    if ix > 0 {
                        triplets.push((i, i - 1, -1.0));
                    }
                    if ix + 1 < nx {
                        triplets.push((i, i + 1, -1.0));
                    }
                    if iy > 0 {
                        triplets.push((i, i - nx, -1.0));
                    }
                    if iy + 1 < nx {
                        triplets.push((i, i + nx, -1.0));
                    }
                    if iz > 0 {
                        triplets.push((i, i - nx * nx, -1.0));
                    }
                    if iz + 1 < nx {
                        triplets.push((i, i + nx * nx, -1.0));
                    }
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// HPD matrix mirroring the simplicial tests: diagonal 6, Laplacian
    /// off-diagonals, `±0.5i` skew on horizontal neighbours (strictly
    /// diagonally dominant, hence HPD).
    fn hpd_matrix(nx: usize) -> CsrMatrix<Complex64> {
        let lap = laplacian_2d(nx);
        let n = lap.num_rows;
        let mut triplets = Vec::with_capacity(lap.nnz() + n);
        for r in 0..n {
            for idx in lap.row_ptrs[r]..lap.row_ptrs[r + 1] {
                let c = lap.col_indices[idx];
                let v = lap.values[idx];
                let skew = if c == r + 1 {
                    Complex64::new(0.0, 0.5)
                } else if c + 1 == r {
                    Complex64::new(0.0, -0.5)
                } else {
                    Complex64::zero()
                };
                let diagonal_boost = if c == r { 2.0 } else { 0.0 };
                triplets.push((r, c, Complex64::new(v + diagonal_boost, 0.0) + skew));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn rhs_from<T: ComplexField>(matrix: &CsrMatrix<T>) -> (Vec<T>, Vec<T>) {
        let n = matrix.num_rows;
        let seven = T::Real::from_usize(7).unwrap();
        let x_exact: Vec<T> = (0..n)
            .map(|i| {
                T::from_re_im(
                    T::Real::from_usize(i % 7 + 1).unwrap() / seven,
                    T::Real::zero(),
                )
            })
            .collect();
        let b = matrix.matvec(&x_exact);
        (x_exact, b)
    }

    fn residual_norm<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], b: &[T]) -> T::Real {
        let actual = matrix.matvec(x);
        actual
            .iter()
            .zip(b)
            .map(|(a, e)| (*a - *e).norm())
            .fold(T::Real::zero(), T::Real::max)
    }

    /// Supernodal vs simplicial agreement on one matrix.
    fn assert_matches_simplicial<T: ComplexField>(
        matrix: &CsrMatrix<T>,
        hermitian: bool,
        label: &str,
    ) where
        T::Real: std::fmt::Display,
    {
        let (_, b) = rhs_from(matrix);
        let simplicial = SparseCholeskyFactorization::factorize_with_ordering(
            matrix,
            CholeskyOrdering::NestedDissection,
            hermitian,
        )
        .unwrap();
        let supernodal = if hermitian {
            SparseCholeskyFactorization::factorize_supernodal_hermitian(matrix).unwrap()
        } else {
            SparseCholeskyFactorization::factorize_supernodal(matrix).unwrap()
        };
        assert_eq!(
            simplicial.factor_nonzeros(),
            supernodal.factor_nonzeros(),
            "{label}: factor nnz must match the shared symbolic pattern"
        );
        let x_simplicial = simplicial.solve(&b).unwrap();
        let x_supernodal = supernodal.solve(&b).unwrap();
        let tol = T::Real::from_f64(1e-12).unwrap();
        for (i, (s, u)) in x_simplicial.iter().zip(&x_supernodal).enumerate() {
            assert!(
                (*s - *u).norm() < tol,
                "{label}: solution mismatch at {i}: simplicial {s:?} vs supernodal {u:?}"
            );
        }
        let residual = residual_norm(matrix, &x_supernodal, &b);
        assert!(
            residual < T::Real::from_f64(1e-10).unwrap(),
            "{label}: residual {residual}"
        );
    }

    #[test]
    fn matches_simplicial_2d_laplacians() {
        for nx in [4, 8, 16, 24, 32] {
            assert_matches_simplicial(&laplacian_2d(nx), false, &format!("2D Laplacian {nx}x{nx}"));
        }
    }

    #[test]
    fn matches_simplicial_3d_laplacians() {
        for nx in [3, 5, 6] {
            assert_matches_simplicial(&laplacian_3d(nx), false, &format!("3D Laplacian {nx}^3"));
        }
    }

    #[test]
    fn matches_simplicial_hpd_complex() {
        for nx in [8, 12, 16] {
            assert_matches_simplicial(&hpd_matrix(nx), true, &format!("HPD {nx}x{nx}"));
        }
    }

    #[test]
    fn multi_column_supernodes_are_formed() {
        // Without at least one width >= 2 supernode the supernodal path
        // degenerates to the simplicial one and the tests prove nothing.
        for nx in [8, 16, 32] {
            let matrix = laplacian_2d(nx);
            let symbolic =
                SparseCholeskySymbolic::analyze(&matrix, CholeskyOrdering::NestedDissection)
                    .unwrap();
            let map = symbolic.supernode_map();
            let max_width = map.max_width();
            println!(
                "2D Laplacian {nx}x{nx} (n={}): {} supernodes, max width {max_width}, \
                 dense panel entries {} vs nnz(L) {}",
                matrix.num_rows,
                map.count(),
                map.dense_panel_entries(),
                symbolic.factor_nonzeros(),
            );
            assert!(map.count() < matrix.num_rows, "{nx}: expected merging");
            assert!(max_width >= 2, "{nx}: no multi-column supernode formed");
        }
        // 3D problems have denser fronts and wider supernodes.
        let matrix = laplacian_3d(5);
        let symbolic =
            SparseCholeskySymbolic::analyze(&matrix, CholeskyOrdering::NestedDissection).unwrap();
        let map = symbolic.supernode_map();
        println!(
            "3D Laplacian 5^3 (n={}): {} supernodes, max width {}",
            matrix.num_rows,
            map.count(),
            map.max_width(),
        );
        assert!(map.max_width() >= 2);
    }

    #[test]
    fn supernode_row_sets_cover_every_column_pattern() {
        let matrix = laplacian_2d(12);
        let symbolic =
            SparseCholeskySymbolic::analyze(&matrix, CholeskyOrdering::NestedDissection).unwrap();
        let map = symbolic.supernode_map();
        let ptrs = symbolic.factor_col_ptrs();
        let rows = symbolic.factor_row_indices();
        let mut covered = 0;
        for index in 0..map.count() {
            let (start, end) = map.range(index);
            assert!(start < end && end <= matrix.num_rows);
            assert_eq!(covered, start, "supernodes must be contiguous");
            covered = end;
            // Row set: sorted, all >= end, and a superset of every absorbed
            // column's below-supernode pattern.
            let panel_rows = map.rows(index);
            assert!(
                panel_rows.windows(2).all(|w| w[0] < w[1]),
                "sorted, deduped"
            );
            assert!(panel_rows.iter().all(|&r| r >= end));
            for j in start..end {
                for &r in &rows[ptrs[j] + 1..ptrs[j + 1]] {
                    if r >= end {
                        assert!(
                            panel_rows.binary_search(&r).is_ok(),
                            "row {r} of column {j} missing from supernode row set"
                        );
                    }
                }
            }
        }
        assert_eq!(covered, matrix.num_rows);
    }

    #[test]
    fn rejects_non_positive_definite() {
        let nx = 10;
        let n = nx * nx;
        let mut triplets = Vec::new();
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, if i == n / 2 { -1.0 } else { 4.0 }));
                if ix > 0 {
                    triplets.push((i, i - 1, -1.0));
                }
                if ix + 1 < nx {
                    triplets.push((i, i + 1, -1.0));
                }
                if iy > 0 {
                    triplets.push((i, i - nx, -1.0));
                }
                if iy + 1 < nx {
                    triplets.push((i, i + nx, -1.0));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, triplets);
        assert!(matches!(
            SparseCholeskyFactorization::factorize_supernodal(&matrix),
            Err(SparseCholeskyError::NotPositiveDefinite { .. })
        ));
        // Same for the Hermitian path.
        let hpd_indefinite: CsrMatrix<Complex64> = CsrMatrix::from_raw_parts(
            n,
            n,
            matrix.row_ptrs.clone(),
            matrix.col_indices.clone(),
            matrix
                .values
                .iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect(),
        );
        assert!(matches!(
            SparseCholeskyFactorization::factorize_supernodal_hermitian(&hpd_indefinite),
            Err(SparseCholeskyError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn symbolic_reuse_and_pattern_mismatch() {
        let nx = 16;
        let a1 = laplacian_2d(nx);
        let n = a1.num_rows;
        let mut triplets = Vec::with_capacity(a1.nnz());
        for r in 0..n {
            for idx in a1.row_ptrs[r]..a1.row_ptrs[r + 1] {
                let c = a1.col_indices[idx];
                triplets.push((r, c, if c == r { 8.0 } else { -2.0 }));
            }
        }
        let a2 = CsrMatrix::from_triplets(n, n, triplets);

        let symbolic =
            SparseCholeskySymbolic::analyze(&a1, CholeskyOrdering::NestedDissection).unwrap();
        let f1 =
            SparseCholeskyFactorization::factorize_supernodal_with_symbolic(&a1, &symbolic, false)
                .unwrap();
        let f2 =
            SparseCholeskyFactorization::factorize_supernodal_with_symbolic(&a2, &symbolic, false)
                .unwrap();
        let (_, b1) = rhs_from(&a1);
        let (_, b2) = rhs_from(&a2);
        let x1 = f1.solve(&b1).unwrap();
        let x2 = f2.solve(&b2).unwrap();
        assert!(residual_norm(&a1, &x1, &b1) < 1e-10);
        assert!(residual_norm(&a2, &x2, &b2) < 1e-10);

        // A different pattern of the same dimension must be rejected.
        let mut triplets = Vec::with_capacity(a1.nnz() + 2);
        for r in 0..n {
            for idx in a1.row_ptrs[r]..a1.row_ptrs[r + 1] {
                triplets.push((r, a1.col_indices[idx], a1.values[idx]));
            }
        }
        triplets.push((0, 2, -0.25));
        triplets.push((2, 0, -0.25));
        let a3 = CsrMatrix::from_triplets(n, n, triplets);
        assert!(matches!(
            SparseCholeskyFactorization::factorize_supernodal_with_symbolic(&a3, &symbolic, false),
            Err(SparseCholeskyError::PatternMismatch)
        ));
    }

    #[test]
    fn deterministic_across_runs() {
        let matrix = laplacian_2d(20);
        let (_, b) = rhs_from(&matrix);
        let first = SparseCholeskyFactorization::factorize_supernodal(&matrix).unwrap();
        let second = SparseCholeskyFactorization::factorize_supernodal(&matrix).unwrap();
        let x1 = first.solve(&b).unwrap();
        let x2 = second.solve(&b).unwrap();
        assert_eq!(x1, x2, "supernodal factorization must be deterministic");
    }

    #[test]
    fn empty_and_singleton_matrices() {
        let empty: CsrMatrix<f64> = CsrMatrix::new(0, 0);
        let factor = SparseCholeskyFactorization::factorize_supernodal(&empty).unwrap();
        assert_eq!(factor.dimension(), 0);
        assert!(factor.solve(&[]).unwrap().is_empty());

        let one = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
        let factor = SparseCholeskyFactorization::factorize_supernodal(&one).unwrap();
        let x = factor.solve(&[8.0]).unwrap();
        assert!((x[0] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn natural_and_colamd_orderings_agree() {
        let matrix = laplacian_2d(12);
        let (_, b) = rhs_from(&matrix);
        for ordering in [CholeskyOrdering::Natural, CholeskyOrdering::Colamd] {
            let factor = SparseCholeskyFactorization::factorize_supernodal_with_ordering(
                &matrix, ordering, false,
            )
            .unwrap();
            let x = factor.solve(&b).unwrap();
            assert!(
                residual_norm(&matrix, &x, &b) < 1e-10,
                "ordering {ordering:?}"
            );
        }
    }

    #[test]
    fn factor_time_simplicial_vs_supernodal() {
        for nx in [64, 96] {
            let matrix = laplacian_2d(nx);
            let (_, b) = rhs_from(&matrix);

            // End-to-end (ordering + symbolic + numeric).
            let t0 = std::time::Instant::now();
            let simplicial = SparseCholeskyFactorization::factorize(&matrix).unwrap();
            let simplicial_time = t0.elapsed();

            let t0 = std::time::Instant::now();
            let supernodal = SparseCholeskyFactorization::factorize_supernodal(&matrix).unwrap();
            let supernodal_time = t0.elapsed();

            // Numeric-only, both reusing one shared symbolic analysis — the
            // ND ordering / etree / pattern phase is identical for the two
            // backends and dominates the end-to-end time at these sizes.
            let t0 = std::time::Instant::now();
            let symbolic =
                SparseCholeskySymbolic::analyze(&matrix, CholeskyOrdering::NestedDissection)
                    .unwrap();
            let analyze_time = t0.elapsed();

            let t0 = std::time::Instant::now();
            let simplicial_num =
                SparseCholeskyFactorization::factorize_with_symbolic(&matrix, &symbolic, false)
                    .unwrap();
            let simplicial_num_time = t0.elapsed();

            let t0 = std::time::Instant::now();
            let supernodal_num = SparseCholeskyFactorization::factorize_supernodal_with_symbolic(
                &matrix, &symbolic, false,
            )
            .unwrap();
            let supernodal_num_time = t0.elapsed();

            let map = symbolic.supernode_map();

            let x = supernodal.solve(&b).unwrap();
            assert!(residual_norm(&matrix, &x, &b) < 1e-10);
            assert_eq!(simplicial.factor_nonzeros(), supernodal.factor_nonzeros());
            assert_eq!(
                simplicial_num.factor_nonzeros(),
                supernodal_num.factor_nonzeros()
            );
            // Flop distribution across the three numeric phases.
            let (mut chol_flops, mut trsm_flops, mut gemm_flops) = (0usize, 0usize, 0usize);
            for i in 0..map.count() {
                let w = map.width(i);
                let m = map.rows(i).len();
                chol_flops += w * w * w / 3;
                trsm_flops += m * w * w;
                gemm_flops += m * m * w;
            }
            println!(
                "  phases: block-chol={chol_flops} trsm={trsm_flops} rank-k gemm={gemm_flops}"
            );
            println!(
                "2D Laplacian {nx}x{nx} (n={}): nnz(L)={}, supernodes={} (max width {}), \
                 dense panel entries={}",
                matrix.num_rows,
                supernodal.factor_nonzeros(),
                map.count(),
                map.max_width(),
                map.dense_panel_entries(),
            );
            println!(
                "  symbolic (shared)={analyze_time:?}; \
                 numeric simplicial={simplicial_num_time:?} vs supernodal={supernodal_num_time:?} \
                 ({:.2}x)",
                simplicial_num_time.as_secs_f64() / supernodal_num_time.as_secs_f64().max(1e-9),
            );
            println!(
                "  end-to-end simplicial={simplicial_time:?} vs supernodal={supernodal_time:?} \
                 ({:.2}x)",
                simplicial_time.as_secs_f64() / supernodal_time.as_secs_f64().max(1e-9),
            );
        }
    }

    /// 3D Laplacians have fat fronts (separator width ~ n^{2/3}), where the
    /// BLAS-3 panel machinery pays off. Ignored by default (release-mode
    /// measurement); run with `--ignored`.
    #[test]
    #[ignore = "release-mode performance measurement"]
    fn factor_time_3d() {
        for nx in [12, 16, 20] {
            let matrix = laplacian_3d(nx);
            let (_, b) = rhs_from(&matrix);
            let symbolic =
                SparseCholeskySymbolic::analyze(&matrix, CholeskyOrdering::NestedDissection)
                    .unwrap();
            let map = symbolic.supernode_map();

            let t0 = std::time::Instant::now();
            let simplicial =
                SparseCholeskyFactorization::factorize_with_symbolic(&matrix, &symbolic, false)
                    .unwrap();
            let simplicial_time = t0.elapsed();

            let t0 = std::time::Instant::now();
            let supernodal = SparseCholeskyFactorization::factorize_supernodal_with_symbolic(
                &matrix, &symbolic, false,
            )
            .unwrap();
            let supernodal_time = t0.elapsed();

            let x = supernodal.solve(&b).unwrap();
            assert!(residual_norm(&matrix, &x, &b) < 1e-9);
            assert_eq!(simplicial.factor_nonzeros(), supernodal.factor_nonzeros());
            println!(
                "3D Laplacian {nx}^3 (n={}): nnz(L)={}, supernodes={} (max width {}), \
                 dense panel entries={}; numeric simplicial={simplicial_time:?} vs \
                 supernodal={supernodal_time:?} ({:.2}x)",
                matrix.num_rows,
                supernodal.factor_nonzeros(),
                map.count(),
                map.max_width(),
                map.dense_panel_entries(),
                simplicial_time.as_secs_f64() / supernodal_time.as_secs_f64().max(1e-9),
            );
        }
    }
}
