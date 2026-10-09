use super::AmgLevel;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use sonium_solvers_core::traits::{ComplexField, SolverError};
use sonium_solvers_sparse::CsrMatrix;

/// Below this many DOFs, util tail-ops stay sequential — rayon scheduling
/// overhead dominates the actual O(n) or O(nnz) work.
const PAR_UTIL_THRESHOLD: usize = 8_192;

/// Validate the memory-safety invariants of CSR storage without a value scan.
///
/// This mirrors the sparse crate's private hot-path check (`matvec_into_buf`
/// runs it on every call): lengths, monotone row pointers rooted at zero
/// with terminal count equal to the value count, in-bounds columns. Like
/// the hot check it accepts duplicates, unsorted rows, and non-finite
/// values (those propagate arithmetically; only indexing must be provable).
///
/// Infallible constructors call this once at setup, after which the owned
/// matrix (kept behind a private field with no post-setup mutation path)
/// may be used with `matvec_into_buf_unchecked` in apply paths.
pub(crate) fn validate_owned_structure<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Result<(), SolverError> {
    let expected_row_ptr_len =
        matrix
            .num_rows
            .checked_add(1)
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "CSR owned-structure validation",
                reason: format!("row count {} overflows row-pointer length", matrix.num_rows),
            })?;
    if matrix.row_ptrs.len() != expected_row_ptr_len {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: format!(
                "row_ptrs has length {}, expected {}",
                matrix.row_ptrs.len(),
                expected_row_ptr_len
            ),
        });
    }
    if matrix.col_indices.len() != matrix.values.len() {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: format!(
                "col_indices has length {}, values has length {}",
                matrix.col_indices.len(),
                matrix.values.len()
            ),
        });
    }
    if matrix.row_ptrs.first().copied() != Some(0) {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: "row_ptrs must start at zero".into(),
        });
    }
    if matrix.row_ptrs.last().copied() != Some(matrix.values.len()) {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: format!(
                "terminal row pointer must equal values length {}, got {:?}",
                matrix.values.len(),
                matrix.row_ptrs.last()
            ),
        });
    }
    if matrix
        .row_ptrs
        .windows(2)
        .any(|window| window[0] > window[1])
    {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: "row_ptrs must be monotone".into(),
        });
    }
    if let Some((index, &column)) = matrix
        .col_indices
        .iter()
        .enumerate()
        .find(|(_, column)| **column >= matrix.num_cols)
    {
        return Err(SolverError::OperationFailed {
            operation: "CSR owned-structure validation",
            reason: format!(
                "column index {column} at entry {index} is outside {} columns",
                matrix.num_cols
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn transpose_rejects_corrupt_public_storage_before_scatter() {
        for n in [3, PAR_UTIL_THRESHOLD] {
            let valid = CsrMatrix::from_triplets(n, n, vec![(0, 0, 1.0)]);
            let mut corrupt = valid.clone();
            // Reuse the first stored value in a second row: without structural
            // validation the histogram requests two writes to a one-slot buffer.
            corrupt.row_ptrs[2] = 0;
            assert!(std::panic::catch_unwind(|| transpose_csr(&corrupt)).is_err());
            let mut corrupt = valid.clone();
            corrupt.col_indices[0] = n;
            assert!(std::panic::catch_unwind(|| transpose_csr(&corrupt)).is_err());
            let transposed = transpose_csr(&valid);
            assert_eq!(transposed.values, vec![1.0]);
            assert_eq!(transposed.col_indices, vec![0]);
            transposed.validate_structure().unwrap();
        }
    }

    #[test]
    fn owned_structure_mirrors_hot_path_accept_set() {
        // `from_raw_parts` validates at construction, so malformed
        // storage is built by mutating the public fields afterwards —
        // the same post-construction corruption the hot path defends
        // against.
        let valid = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
        assert!(validate_owned_structure(&valid).is_ok());
        // Unsorted rows and duplicates are accepted (rows accumulate).
        let mut unsorted_dupes =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0]);
        unsorted_dupes.col_indices.swap(0, 1);
        assert!(validate_owned_structure(&unsorted_dupes).is_ok());
        // Non-finite values are accepted (arithmetic, not indexing).
        let mut non_finite = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        non_finite.values[0] = f64::NAN;
        assert!(validate_owned_structure(&non_finite).is_ok());

        // Each malformed class is rejected.
        let mut bad_row_ptr_len =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 1], vec![0], vec![1.0]);
        bad_row_ptr_len.row_ptrs.pop();
        assert!(validate_owned_structure(&bad_row_ptr_len).is_err());
        let mut bad_start = CsrMatrix::from_raw_parts(1, 1, vec![0, 0], vec![], Vec::<f64>::new());
        bad_start.row_ptrs[0] = 1;
        assert!(validate_owned_structure(&bad_start).is_err());
        let mut non_monotone =
            CsrMatrix::from_raw_parts(3, 1, vec![0, 0, 1, 1], vec![0], vec![1.0]);
        non_monotone.row_ptrs.swap(1, 2);
        assert!(validate_owned_structure(&non_monotone).is_err());
        let mut bad_terminal = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        bad_terminal.row_ptrs[1] = 2;
        assert!(validate_owned_structure(&bad_terminal).is_err());
        let mut len_mismatch = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        len_mismatch.col_indices.push(0);
        assert!(validate_owned_structure(&len_mismatch).is_err());
        let mut col_oob = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        col_oob.col_indices[0] = 1;
        assert!(validate_owned_structure(&col_oob).is_err());
    }
}

/// Transpose a CSR matrix.
///
/// Uses a two-phase parallel histogram: each thread counts its own slice
/// of input rows into a thread-local `row_counts`, the partials are
/// reduced into a global `row_counts` via a prefix sum, then each thread
/// scatters its slice into pre-assigned positions using a thread-local
/// offset. This avoids any atomic-counter contention on the hot path.
pub(super) fn transpose_csr<T: ComplexField>(matrix: &CsrMatrix<T>) -> CsrMatrix<T> {
    // Public CSR storage can be modified after construction. The histogram
    // scatter below relies on every input entry belonging to exactly one row
    // and on the row-pointer terminal count matching the allocated values.
    matrix
        .validate_structure()
        .expect("AMG transpose requires valid CSR storage");
    let m = matrix.num_rows;
    let n = matrix.num_cols;
    let nnz = matrix.nnz();

    // Route by nonzeros, not dimensions: the parallel path's per-chunk
    // histograms (n words x thread count) and reduction only pay off on
    // large products. Tall-skinny prolongation transposes (O(100k) nnz)
    // were spending ~13ms in parallel overhead that the sequential
    // counting-scatter does in under a millisecond.
    if nnz < 2_000_000 {
        return transpose_csr_sequential(matrix);
    }

    let nthreads = rayon::current_num_threads().max(1);
    let chunk_size = m.div_ceil(nthreads);
    let chunks: Vec<(usize, usize)> = (0..nthreads)
        .map(|t| (t * chunk_size, ((t + 1) * chunk_size).min(m)))
        .filter(|(s, e)| e > s)
        .collect();

    // Phase 1: per-chunk histogram of column indices.
    let local_counts: Vec<Vec<usize>> = chunks
        .par_iter()
        .map(|&(start, end)| {
            let mut local = vec![0usize; n];
            for i in start..end {
                for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                    local[matrix.col_indices[idx]] += 1;
                }
            }
            local
        })
        .collect();

    // Reduce per-chunk histograms into a global row_counts. Per-column work
    // is independent so this row-major reduction is itself parallel.
    let mut row_counts = vec![0usize; n];
    {
        let row_counts_slice = row_counts.as_mut_slice();
        row_counts_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| {
                let mut sum = 0usize;
                for local in &local_counts {
                    sum += local[j];
                }
                *slot = sum;
            });
    }

    // Phase 2: prefix-sum into row_ptrs (sequential by definition; cheap).
    let mut row_ptrs = vec![0usize; n + 1];
    for i in 0..n {
        row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
    }

    // Phase 3: each chunk computes its own per-output-row offset = global
    // row_ptr + sum of preceding chunks' counts for that column. Then it
    // scatters into a disjoint slice of `values`/`col_indices`.
    let mut values = vec![T::zero(); nnz];
    let mut col_indices = vec![0usize; nnz];

    // For chunk c, offset_for_chunk[c][j] = row_ptrs[j] + sum_{c'<c} local_counts[c'][j].
    // Store as flat (chunks × n) for parallelism.
    let mut chunk_offsets: Vec<Vec<usize>> = vec![vec![0usize; n]; chunks.len()];
    for j in 0..n {
        let mut acc = row_ptrs[j];
        for (c, local) in local_counts.iter().enumerate() {
            chunk_offsets[c][j] = acc;
            acc += local[j];
        }
    }

    // Each chunk writes to disjoint positions in `values` and `col_indices`
    // (the chunk_offsets table guarantees uniqueness), but the positions
    // interleave by column so a clean `split_at_mut` partition is awkward.
    // Pass the pointer addresses as `usize` across workers and cast back
    // inside the closure — bypasses the *mut T auto-trait machinery.
    let values_addr = values.as_mut_ptr() as usize;
    let cols_addr = col_indices.as_mut_ptr() as usize;

    chunks
        .par_iter()
        .zip(chunk_offsets.into_par_iter())
        .for_each(|(&(start, end), mut local_off)| {
            let vp = values_addr as *mut T;
            let cp = cols_addr as *mut usize;
            for i in start..end {
                for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                    let j = matrix.col_indices[idx];
                    let pos = local_off[j];
                    // SAFETY: `pos` is unique across all (chunk, i, idx)
                    // tuples by construction (chunks are disjoint row ranges
                    // and chunk_offsets[c][j] is the exact prefix sum).
                    unsafe {
                        *vp.add(pos) = matrix.values[idx];
                        *cp.add(pos) = i;
                    }
                    local_off[j] = pos + 1;
                }
            }
        });

    CsrMatrix::from_raw_parts(n, m, row_ptrs, col_indices, values)
}

fn transpose_csr_sequential<T: ComplexField>(matrix: &CsrMatrix<T>) -> CsrMatrix<T> {
    let m = matrix.num_rows;
    let n = matrix.num_cols;
    let nnz = matrix.nnz();

    let mut row_counts = vec![0usize; n];
    for &j in &matrix.col_indices {
        row_counts[j] += 1;
    }

    let mut row_ptrs = vec![0usize; n + 1];
    for i in 0..n {
        row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
    }

    let mut values = vec![T::zero(); nnz];
    let mut col_indices = vec![0usize; nnz];
    let mut offsets = row_ptrs[..n].to_vec();

    for i in 0..m {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            let pos = offsets[j];
            values[pos] = matrix.values[idx];
            col_indices[pos] = i;
            offsets[j] += 1;
        }
    }

    CsrMatrix::from_raw_parts(n, m, row_ptrs, col_indices, values)
}

/// Absolute drop tolerance for AMG-internal SpGEMM products.
///
/// The Galerkin product `R A P` and the strength-of-connection products
/// (`A^T A`, `A^2`) historically dropped accumulated entries with
/// `|c_ij| <= 1e-15` via `CsrMatrix::matmul`. That policy now lives behind
/// the explicit opt-in [`CsrMatrix::matmul_with_drop`]; AMG keeps it to
/// preserve the established coarse-operator numerics (tiny products of
/// interpolation weights carry no useful coarse-grid information).
pub(super) const AMG_MATMUL_DROP_TOL: f64 = 1e-15;

/// Compute the adjoint (conjugate transpose) of a CSR matrix: `A^H`.
///
/// This is [`transpose_csr`] with conjugated values. For real `T` it is
/// identical to the plain transpose.
pub(super) fn adjoint_csr<T: ComplexField>(matrix: &CsrMatrix<T>) -> CsrMatrix<T> {
    let mut adjoint = transpose_csr(matrix);
    for v in &mut adjoint.values {
        *v = v.conj();
    }
    adjoint
}

/// Compute Galerkin coarse grid operator: A_c = R * A * P
pub(super) fn galerkin_product<T: ComplexField>(
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
) -> CsrMatrix<T> {
    let tol = T::Real::from_f64(AMG_MATMUL_DROP_TOL).unwrap();
    let ap = a.matmul_with_drop(p, tol);
    r.matmul_with_drop(&ap, tol)
}

/// Drop small off-diagonal entries from a sparse matrix.
///
/// Keeps entry (i,j) if |a_ij| >= 0.001 * |a_ii|, plus always keeps the diagonal.
/// This controls coarse operator complexity in SA-AMG.
///
/// Each output row's filter is independent of every other row, so this
/// builds per-row `(col, val)` lists in parallel and assembles the result
/// CSR directly via prefix-summed `row_ptrs`. Avoids the sequential
/// `CsrBuilder::add_row_entries` walk that previously dominated AMG setup
/// for moderately large coarse levels.
pub(super) fn drop_small_entries<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    tol_factor: f64,
) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let num_cols = matrix.num_cols;
    let tol_factor = T::Real::from_f64(tol_factor).unwrap();

    let build_row = |i: usize| -> Vec<(usize, T)> {
        let diag_i = matrix.get(i, i).norm();
        let threshold = tol_factor * diag_i;
        matrix
            .row_entries(i)
            .filter_map(|(j, v)| {
                if i == j || v.norm() >= threshold {
                    Some((j, v))
                } else {
                    None
                }
            })
            .collect()
    };

    let per_row: Vec<Vec<(usize, T)>> = if n >= PAR_UTIL_THRESHOLD {
        (0..n).into_par_iter().map(build_row).collect()
    } else {
        (0..n).map(build_row).collect()
    };

    let mut row_ptrs = vec![0usize; n + 1];
    for i in 0..n {
        row_ptrs[i + 1] = row_ptrs[i] + per_row[i].len();
    }
    let total = row_ptrs[n];

    let mut values = vec![T::zero(); total];
    let mut col_indices = vec![0usize; total];

    if n >= PAR_UTIL_THRESHOLD {
        let mut val_rest = values.as_mut_slice();
        let mut col_rest = col_indices.as_mut_slice();
        let mut val_chunks: Vec<&mut [T]> = Vec::with_capacity(n);
        let mut col_chunks: Vec<&mut [usize]> = Vec::with_capacity(n);
        for row in &per_row {
            let len = row.len();
            let (vh, vt) = val_rest.split_at_mut(len);
            let (ch, ct) = col_rest.split_at_mut(len);
            val_chunks.push(vh);
            col_chunks.push(ch);
            val_rest = vt;
            col_rest = ct;
        }
        per_row
            .into_par_iter()
            .zip(val_chunks.into_par_iter())
            .zip(col_chunks.into_par_iter())
            .for_each(|((row, val_chunk), col_chunk)| {
                for (k, (c, v)) in row.into_iter().enumerate() {
                    val_chunk[k] = v;
                    col_chunk[k] = c;
                }
            });
    } else {
        let mut idx = 0;
        for row in per_row {
            for (c, v) in row {
                values[idx] = v;
                col_indices[idx] = c;
                idx += 1;
            }
        }
    }

    CsrMatrix::from_raw_parts(n, num_cols, row_ptrs, col_indices, values)
}

/// Compute grid and operator complexities
pub(super) fn compute_complexities<T: ComplexField>(levels: &[AmgLevel<T>]) -> (f64, f64) {
    if levels.is_empty() {
        return (1.0, 1.0);
    }

    let fine_dofs = levels[0].num_dofs as f64;
    let fine_nnz = levels[0].matrix.nnz() as f64;

    let total_dofs: f64 = levels.iter().map(|l| l.num_dofs as f64).sum();
    let total_nnz: f64 = levels.iter().map(|l| l.matrix.nnz() as f64).sum();

    let grid_complexity = total_dofs / fine_dofs;
    let operator_complexity = total_nnz / fine_nnz;

    (grid_complexity, operator_complexity)
}

/// Compute inverse diagonal for Jacobi smoothing.
///
/// Uses [`CsrMatrix::diagonal`] for a single O(nnz) pass instead of
/// per-row linear search via `get(i, i)`.
pub(super) fn compute_diag_inv<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<T> {
    let diag = matrix.diagonal();
    let tol = T::Real::from_f64(1e-15).unwrap();
    let n = diag.len();
    let inv = |&d: &T| -> T { if d.norm() > tol { d.inv() } else { T::one() } };
    let values: Vec<T> = if n >= PAR_UTIL_THRESHOLD {
        diag.par_iter().map(inv).collect()
    } else {
        diag.iter().map(inv).collect()
    };
    values
}

/// Compute L1 row-norm diagonal for L1-Jacobi smoothing: l1_diag[i] = Σ_j |A[i,j]|
pub(super) fn compute_l1_diag<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<T::Real> {
    let n = matrix.num_rows;
    let tol = T::Real::from_f64(1e-15).unwrap();
    let row_l1 = |i: usize| -> T::Real {
        let mut sum = T::Real::from_f64(0.0).unwrap();
        for (_, val) in matrix.row_entries(i) {
            sum += val.norm();
        }
        if sum > tol {
            sum
        } else {
            T::Real::from_f64(1.0).unwrap()
        }
    };
    if n >= PAR_UTIL_THRESHOLD {
        (0..n).into_par_iter().map(row_l1).collect()
    } else {
        (0..n).map(row_l1).collect()
    }
}

/// Estimate the largest eigenvalue of D^{-1}A using power iteration.
///
/// Runs 10 iterations starting from an alternating vector.
/// Returns the Rayleigh quotient estimate.
pub(crate) fn estimate_max_eigenvalue<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    diag_inv: &[T],
) -> T::Real {
    let n = matrix.num_rows;
    if n == 0 {
        return T::Real::from_f64(1.0).unwrap();
    }

    // Start with alternating +1/-1 vector
    let mut v = vec![T::zero(); n];
    for (i, slot) in v.iter_mut().enumerate() {
        if i % 2 == 0 {
            *slot = T::from_real(T::Real::from_f64(1.0).unwrap());
        } else {
            *slot = -T::from_real(T::Real::from_f64(1.0).unwrap());
        }
    }

    let mut lambda = T::Real::from_f64(1.0).unwrap();

    // Scratch reused across power iterations (the previous version
    // allocated three vectors per iteration on top of a validated SpMV).
    let mut av = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];

    for _ in 0..20 {
        // w = D^{-1} A v
        //
        // SAFETY: every in-crate caller passes a matrix whose memory-safety
        // invariants were established before this call — the AMG/Chebyshev
        // infallible constructors validate their input once via
        // `validate_owned_structure`, the fallible ones validate more
        // strongly, and Galerkin/coarse matrices are valid by
        // construction. `v`/`av` have length `num_rows == num_cols` on the
        // square inputs this estimator serves (asserted inside).
        unsafe { matrix.matvec_into_buf_unchecked(&v, &mut av) };
        for i in 0..n {
            w[i] = av[i] * diag_inv[i];
        }

        // Rayleigh quotient magnitude: lambda = |v^H w| / |v^H v|
        // We use the magnitude (not .re()) because for complex matrices,
        // eigenvalues of D^{-1}A can be complex. The Chebyshev smoother
        // needs a bound on the spectral radius, not just the real part.
        let vw = sonium_solvers_core::blas_helpers::inner_product(&v, &w);
        let vv = sonium_solvers_core::blas_helpers::inner_product(&v, &v);
        if vv.norm() > T::Real::from_f64(1e-30).unwrap() {
            lambda = vw.norm() / vv.norm();
        }

        // Normalize w
        let w_norm = sonium_solvers_core::blas_helpers::vector_norm(&w);
        if w_norm < T::Real::from_f64(1e-30).unwrap() {
            break;
        }
        let inv_norm = T::from_real(T::Real::from_f64(1.0).unwrap() / w_norm);
        for i in 0..n {
            v[i] = w[i] * inv_norm;
        }
    }

    // Apply 10% safety factor (PETSc does this to avoid Chebyshev instability
    // from underestimating the spectral radius)
    let safety = T::Real::from_f64(1.1).unwrap();
    lambda *= safety;

    // Ensure positive and at least 1.0
    let one = T::Real::from_f64(1.0).unwrap();
    if lambda < one {
        lambda = one;
    }
    lambda
}

/// Extract a nodal (block) graph from a scalar CSR matrix.
///
/// Given a matrix of size `n x n` with `block_size` DOFs per node,
/// produces a `(n/bs) x (n/bs)` nodal matrix where entry `(I, J)` is
/// the Frobenius norm of the `bs x bs` block `A[I*bs:(I+1)*bs, J*bs:(J+1)*bs]`.
///
/// This nodal matrix is used for strength-of-connection and coarsening
/// in nodal AMG, which then operates on the much smaller nodal graph.
pub(crate) fn extract_nodal_graph<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    block_size: usize,
) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let bs = block_size;
    assert_eq!(
        n % bs,
        0,
        "Matrix size {} not a multiple of block_size {}",
        n,
        bs
    );
    let nn = n / bs; // number of nodes

    // Accumulate Frobenius norms of each bs×bs block (as f64 for simplicity)
    let mut block_norms: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();

    for row in 0..n {
        let node_i = row / bs;
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        for idx in start..end {
            let col = matrix.col_indices[idx];
            let node_j = col / bs;
            let val_norm = matrix.values[idx].norm();
            let val_norm_f64: f64 = num_traits::ToPrimitive::to_f64(&val_norm).unwrap_or(0.0);
            *block_norms.entry((node_i, node_j)).or_insert(0.0) += val_norm_f64 * val_norm_f64;
        }
    }

    // Build nodal CSR
    let mut row_ptrs = vec![0usize; nn + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for node_i in 0..nn {
        // Collect all block columns for this block row, sorted
        let mut cols: Vec<(usize, f64)> = block_norms
            .iter()
            .filter(|&(&(ni, _), _)| ni == node_i)
            .map(|(&(_, nj), &norm_sq)| (nj, norm_sq))
            .collect();
        cols.sort_by_key(|&(j, _)| j);

        for (nj, norm_sq) in cols {
            let frob = norm_sq.sqrt();
            if frob > 0.0 {
                col_indices.push(nj);
                values.push(T::from_real(T::Real::from_f64(frob).unwrap()));
            }
        }
        row_ptrs[node_i + 1] = col_indices.len();
    }

    CsrMatrix::from_raw_parts(nn, nn, row_ptrs, col_indices, values)
}

/// Build a block prolongation operator from a scalar nodal prolongation.
///
/// Given a nodal prolongation P_nodal of size `nn_fine x nn_coarse` (scalar),
/// produces a block prolongation of size `(nn_fine * bs) x (nn_coarse * bs)`
/// where each scalar entry `P_nodal[i,j] = w` becomes a `bs x bs` block
/// `w * I_bs` (identity scaled by the interpolation weight).
pub(crate) fn nodal_to_block_prolongation<T: ComplexField>(
    p_nodal: &CsrMatrix<T>,
    block_size: usize,
) -> CsrMatrix<T> {
    let bs = block_size;
    let n_fine = p_nodal.num_rows * bs;
    let n_coarse = p_nodal.num_cols * bs;

    // Each nodal entry generates bs entries (diagonal block)
    let nnz_estimate = p_nodal.nnz() * bs;
    let mut row_ptrs = vec![0usize; n_fine + 1];
    let mut col_indices = Vec::with_capacity(nnz_estimate);
    let mut values = Vec::with_capacity(nnz_estimate);

    for node_i in 0..p_nodal.num_rows {
        let p_start = p_nodal.row_ptrs[node_i];
        let p_end = p_nodal.row_ptrs[node_i + 1];

        for c in 0..bs {
            let scalar_row = node_i * bs + c;
            for idx in p_start..p_end {
                let node_j = p_nodal.col_indices[idx];
                let w = p_nodal.values[idx];
                // Only the c-th component of the block: P[scalar_row, node_j*bs + c] = w
                col_indices.push(node_j * bs + c);
                values.push(w);
            }
            row_ptrs[scalar_row + 1] = col_indices.len();
        }
    }

    CsrMatrix::from_raw_parts(n_fine, n_coarse, row_ptrs, col_indices, values)
}
