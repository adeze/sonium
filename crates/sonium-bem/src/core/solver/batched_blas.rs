//! Batched BLAS operations for complex matrices
//!
//! Pre-allocated workspaces plus the in-house BLAS-2 GEMV from
//! `sonium-compute` (SIMD-optimised, parallel where worthwhile) drive the
//! SLFMM hot path. Contiguous row-major `Vec<Complex64>` storage keeps the
//! cache happy.

use crate::core::parallel::{
    BEM_BLOCKS_PER_THREAD, gated_enumerate_filter_map, gated_flat_map, gated_map,
};
use num_complex::Complex64;
use sonium_compute::Transpose;
use sonium_compute::cpu::blas2::gemv_c64;
use sonium_solvers::dense::DenseMatrix;

/// y = A * x for `DenseMatrix<Complex64>` (row-major, stride = ncols).
#[inline]
fn gemv_no_trans_dense(a: &DenseMatrix<Complex64>, x: &[Complex64], y: &mut [Complex64]) {
    let (m, n) = (a.nrows(), a.ncols());
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    gemv_c64(Transpose::NoTrans, m, n, one, a.as_slice(), n, x, zero, y);
}

/// y = A^T * x for `DenseMatrix<Complex64>` (row-major, stride = ncols).
#[inline]
fn gemv_trans_dense(a: &DenseMatrix<Complex64>, x: &[Complex64], y: &mut [Complex64]) {
    let (m, n) = (a.nrows(), a.ncols());
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    gemv_c64(Transpose::Trans, m, n, one, a.as_slice(), n, x, zero, y);
}

/// Workspace for batched SLFMM matvec operations
///
/// Pre-allocates all necessary buffers to avoid allocations in hot path.
/// Uses flat row-major `Vec<Complex64>` storage so the `gemv_c64` kernels in
/// `sonium-compute` see a contiguous `(m, n)` block per cluster.
pub struct SlfmmMatvecWorkspace {
    /// Workspace for multipole expansions: [num_clusters * num_sphere_points] (row-major)
    pub multipoles: Vec<Complex64>,
    /// Workspace for local expansions: [num_clusters * num_sphere_points] (row-major)
    pub locals: Vec<Complex64>,
    /// Number of clusters
    pub num_clusters: usize,
    /// Number of sphere integration points
    pub num_sphere_points: usize,
    /// Number of DOFs
    pub num_dofs: usize,
}

impl SlfmmMatvecWorkspace {
    /// Create a new workspace with pre-allocated buffers
    pub fn new(num_clusters: usize, num_sphere_points: usize, num_dofs: usize) -> Self {
        let total = num_clusters * num_sphere_points;
        Self {
            multipoles: vec![Complex64::new(0.0, 0.0); total],
            locals: vec![Complex64::new(0.0, 0.0); total],
            num_clusters,
            num_sphere_points,
            num_dofs,
        }
    }

    /// Clear all workspace buffers (zero out)
    pub fn clear(&mut self) {
        self.multipoles.fill(Complex64::new(0.0, 0.0));
        self.locals.fill(Complex64::new(0.0, 0.0));
    }

    /// Get a mutable slice for a single cluster's multipole row.
    #[inline]
    fn multipole_row_mut(&mut self, cluster: usize) -> &mut [Complex64] {
        let start = cluster * self.num_sphere_points;
        &mut self.multipoles[start..start + self.num_sphere_points]
    }

    /// Get a mutable slice for a single cluster's local row.
    #[inline]
    fn local_row_mut(&mut self, cluster: usize) -> &mut [Complex64] {
        let start = cluster * self.num_sphere_points;
        &mut self.locals[start..start + self.num_sphere_points]
    }

    /// Get an immutable slice for a single cluster's local row.
    #[inline]
    fn local_row(&self, cluster: usize) -> &[Complex64] {
        let start = cluster * self.num_sphere_points;
        &self.locals[start..start + self.num_sphere_points]
    }
}

/// Batched matrix-vector multiply for T-matrix application
///
/// Computes: `multipoles[c] = T[c] * x[cluster_dofs[c]]` for all clusters
pub fn batched_t_matrix_apply(
    t_matrices: &[DenseMatrix<Complex64>],
    x: &[Complex64],
    cluster_dof_indices: &[Vec<usize>],
    workspace: &mut SlfmmMatvecWorkspace,
) {
    let nsp = workspace.num_sphere_points;

    // Batched T-matrix application (rayon above the machine-adaptive gate,
    // serial for few clusters).
    let results: Vec<(usize, Vec<Complex64>)> = gated_enumerate_filter_map(
        t_matrices,
        t_matrices.len(),
        BEM_BLOCKS_PER_THREAD,
        |cluster_idx, t_mat| {
            let cluster_dofs = &cluster_dof_indices[cluster_idx];
            if cluster_dofs.is_empty() || t_mat.as_slice().is_empty() {
                return None;
            }

            // Gather x values for this cluster
            let x_local: Vec<Complex64> = cluster_dofs.iter().map(|&i| x[i]).collect();

            // Apply T-matrix: y = T * x_local
            let mut y = vec![Complex64::new(0.0, 0.0); nsp];
            gemv_no_trans_dense(t_mat, &x_local, &mut y);
            Some((cluster_idx, y))
        },
    );

    // Store results (sequential to avoid race conditions)
    for (cluster_idx, result) in results {
        workspace
            .multipole_row_mut(cluster_idx)
            .copy_from_slice(&result);
    }
}

/// Batched matrix-vector multiply for S-matrix application
///
/// Computes: `y[cluster_dofs[c]] += S[c] * locals[c]` for all clusters
pub fn batched_s_matrix_apply(
    s_matrices: &[DenseMatrix<Complex64>],
    workspace: &SlfmmMatvecWorkspace,
    cluster_dof_indices: &[Vec<usize>],
    y: &mut [Complex64],
) {
    // Batched S-matrix application (rayon above the machine-adaptive gate,
    // serial for few clusters).
    let results: Vec<Vec<(usize, Complex64)>> = gated_enumerate_filter_map(
        s_matrices,
        s_matrices.len(),
        BEM_BLOCKS_PER_THREAD,
        |cluster_idx, s_mat| {
            let cluster_dofs = &cluster_dof_indices[cluster_idx];
            if cluster_dofs.is_empty() || s_mat.as_slice().is_empty() {
                return None;
            }

            // Apply S-matrix: y_local = S * locals_row
            let locals_row = workspace.local_row(cluster_idx);
            let mut y_local = vec![Complex64::new(0.0, 0.0); cluster_dofs.len()];
            gemv_no_trans_dense(s_mat, locals_row, &mut y_local);

            // Collect contributions
            let contributions: Vec<(usize, Complex64)> = cluster_dofs
                .iter()
                .enumerate()
                .map(|(local_j, &global_j)| (global_j, y_local[local_j]))
                .collect();

            Some(contributions)
        },
    );

    // Accumulate results
    for contributions in results {
        for (idx, val) in contributions {
            y[idx] += val;
        }
    }
}

/// Batched D-matrix translation
///
/// Computes: `locals[field_cluster] += D[entry] * multipoles[source_cluster]`
/// for all D-matrix entries.
///
/// The D-matrix is diagonal, so this is element-wise multiplication.
pub fn batched_d_matrix_apply(
    d_matrices: &[super::super::assembly::slfmm::DMatrixEntry],
    workspace: &mut SlfmmMatvecWorkspace,
) {
    let nsp = workspace.num_sphere_points;

    // Batched D-matrix translations (rayon above the machine-adaptive gate,
    // serial for few entries).
    let results: Vec<(usize, Vec<Complex64>)> = gated_map(
        d_matrices,
        d_matrices.len(),
        BEM_BLOCKS_PER_THREAD,
        |d_entry| {
            let src_mult = &workspace.multipoles
                [d_entry.source_cluster * nsp..(d_entry.source_cluster + 1) * nsp];
            // Diagonal matrix-vector multiply: translated[i] = diagonal[i] * src_mult[i]
            let diag_slice = d_entry.diagonal.as_slice();
            let translated: Vec<Complex64> = diag_slice
                .iter()
                .zip(src_mult.iter())
                .map(|(&d, &s)| d * s)
                .collect();
            (d_entry.field_cluster, translated)
        },
    );

    // Accumulate into locals (sequential to avoid race)
    for (field_cluster, translated) in results {
        let row = workspace.local_row_mut(field_cluster);
        for (i, &val) in translated.iter().enumerate() {
            row[i] += val;
        }
    }
}

/// Batched near-field block application
///
/// Computes: y += N * x where N is the sparse near-field matrix
/// represented as dense blocks
pub fn batched_near_field_apply(
    near_blocks: &[super::super::assembly::slfmm::NearFieldBlock],
    x: &[Complex64],
    cluster_dof_indices: &[Vec<usize>],
    y: &mut [Complex64],
) {
    // Batched near-field block contributions (rayon above the
    // machine-adaptive gate, serial for few blocks).
    let contributions: Vec<Vec<(usize, Complex64)>> = gated_flat_map(
        near_blocks,
        near_blocks.len(),
        BEM_BLOCKS_PER_THREAD,
        |block| {
            let src_dofs = &cluster_dof_indices[block.source_cluster];
            let fld_dofs = &cluster_dof_indices[block.field_cluster];

            let mut result = Vec::new();

            // Gather x values from field cluster
            let x_local: Vec<Complex64> = fld_dofs.iter().map(|&i| x[i]).collect();

            // Apply block matrix: y_local = coefficients * x_local
            let mut y_local = vec![Complex64::new(0.0, 0.0); src_dofs.len()];
            gemv_no_trans_dense(&block.coefficients, &x_local, &mut y_local);

            // Collect contributions for source DOFs
            for (local_i, &global_i) in src_dofs.iter().enumerate() {
                result.push((global_i, y_local[local_i]));
            }

            // Handle symmetric storage
            if block.source_cluster != block.field_cluster {
                let x_src: Vec<Complex64> = src_dofs.iter().map(|&i| x[i]).collect();
                let mut y_fld = vec![Complex64::new(0.0, 0.0); fld_dofs.len()];
                gemv_trans_dense(&block.coefficients, &x_src, &mut y_fld);
                for (local_j, &global_j) in fld_dofs.iter().enumerate() {
                    result.push((global_j, y_fld[local_j]));
                }
            }

            vec![result]
        },
    );

    // Accumulate contributions
    for block_contributions in contributions {
        for (idx, val) in block_contributions {
            y[idx] += val;
        }
    }
}

/// Optimized SLFMM matvec using batched operations and pre-allocated workspace
///
/// This version avoids allocations in the hot path by using a pre-allocated workspace.
/// Call `SlfmmMatvecWorkspace::new()` once before solving, then reuse for all matvec calls.
pub fn slfmm_matvec_batched(
    system: &super::super::assembly::slfmm::SlfmmSystem,
    x: &[Complex64],
    workspace: &mut SlfmmMatvecWorkspace,
) -> Vec<Complex64> {
    // Clear workspace
    workspace.clear();

    // Initialize output vector
    let mut y = vec![Complex64::new(0.0, 0.0); system.num_dofs];

    // === Near-field contribution ===
    batched_near_field_apply(&system.near_matrix, x, &system.cluster_dof_indices, &mut y);

    // === Far-field contribution: y += [S][D][T] * x ===

    // Step 1: T-matrix application: multipoles = T * x
    batched_t_matrix_apply(
        &system.t_matrices,
        x,
        &system.cluster_dof_indices,
        workspace,
    );

    // Step 2: D-matrix translation: locals = D * multipoles
    batched_d_matrix_apply(&system.d_matrices, workspace);

    // Step 3: S-matrix application: y += S * locals
    batched_s_matrix_apply(
        &system.s_matrices,
        workspace,
        &system.cluster_dof_indices,
        &mut y,
    );

    y
}

/// Create a batched matvec closure for use with iterative solvers
///
/// Returns a closure that applies the SLFMM operator using batched BLAS operations.
/// The workspace is created once and reused for all matvec calls.
///
/// Note: Returns FnMut because the workspace is mutated on each call.
pub fn create_batched_matvec<'a>(
    system: &'a super::super::assembly::slfmm::SlfmmSystem,
) -> impl FnMut(&[Complex64]) -> Vec<Complex64> + 'a {
    // Pre-allocate workspace
    let mut workspace = SlfmmMatvecWorkspace::new(
        system.num_clusters,
        system.num_sphere_points,
        system.num_dofs,
    );

    move |x: &[Complex64]| slfmm_matvec_batched(system, x, &mut workspace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let workspace = SlfmmMatvecWorkspace::new(10, 32, 100);
        assert_eq!(workspace.num_clusters, 10);
        assert_eq!(workspace.num_sphere_points, 32);
        assert_eq!(workspace.num_dofs, 100);
        assert_eq!(workspace.multipoles.len(), 10 * 32);
        assert_eq!(workspace.locals.len(), 10 * 32);
    }

    #[test]
    fn test_workspace_clear() {
        let mut workspace = SlfmmMatvecWorkspace::new(2, 4, 8);
        workspace.multipoles[0] = Complex64::new(1.0, 2.0);
        workspace.clear();
        assert_eq!(workspace.multipoles[0], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_gemv_no_trans() {
        // 2x3 matrix
        let a = DenseMatrix::from_row_major_vec(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
        );
        let x = [
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        gemv_no_trans_dense(&a, &x, &mut y);
        assert!((y[0] - Complex64::new(6.0, 0.0)).norm() < 1e-12);
        assert!((y[1] - Complex64::new(15.0, 0.0)).norm() < 1e-12);
    }
}
