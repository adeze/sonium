//! Efficient Helmholtz matrix assembler
//!
//! Provides a mechanism to pre-assemble the sparsity pattern of the Helmholtz system
//! and efficiently update the values for different frequencies (wavenumbers).
//!
//! This avoids rebuilding the CSR topology for every frequency step, which is
//! critical for performance in frequency sweeps.

use super::mass::{MassMatrix, assemble_mass};
use super::stiffness::{StiffnessMatrix, assemble_stiffness};
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::parallelism::{POINTWISE_VALUES_PER_THREAD, collect_indexed, use_parallel};
use num_complex::Complex64;
use rayon::prelude::*;
use sonium_solvers::CsrMatrix;
use std::collections::HashMap;

/// Efficient assembler for Helmholtz problems
pub struct HelmholtzAssembler {
    /// Number of rows (DOFs)
    pub num_rows: usize,
    /// CSR row pointers (shared by K, M, and boundaries)
    pub row_ptrs: Vec<usize>,
    /// CSR column indices (shared by K, M, and boundaries)
    pub col_indices: Vec<usize>,
    /// Stiffness matrix values (aligned with col_indices)
    pub k_values: Vec<f64>,
    /// Mass matrix values (aligned with col_indices)
    pub m_values: Vec<f64>,
    /// Boundary mass matrix values (Tag -> Values, aligned with col_indices)
    pub boundary_values: HashMap<usize, Vec<f64>>,
}

impl HelmholtzAssembler {
    /// Create a new assembler from mesh and polynomial degree
    pub fn new(mesh: &Mesh, degree: PolynomialDegree) -> Self {
        let stiffness = assemble_stiffness(mesh, degree);
        let mass = assemble_mass(mesh, degree);
        // Default: no boundary matrices
        Self::from_matrices(&stiffness, &mass, &[])
    }

    /// Create assembler from existing stiffness, mass, and boundary matrices
    pub fn from_matrices(
        stiffness: &StiffnessMatrix,
        mass: &MassMatrix,
        boundaries: &[(usize, MassMatrix)],
    ) -> Self {
        assert_eq!(stiffness.dim, mass.dim);
        let num_rows = stiffness.dim;

        let total_nnz =
            stiffness.nnz() + mass.nnz() + boundaries.iter().map(|(_, m)| m.nnz()).sum::<usize>();

        // 1. Count non-zeros per row across all matrices
        let mut row_counts = vec![0usize; num_rows];
        for &r in &stiffness.rows {
            row_counts[r] += 1;
        }
        for &r in &mass.rows {
            row_counts[r] += 1;
        }
        for (_, matrix) in boundaries {
            for &r in &matrix.rows {
                row_counts[r] += 1;
            }
        }

        // 2. Compute prefix sums for scatter pointers
        let mut scatter_ptrs = vec![0usize; num_rows + 1];
        for i in 0..num_rows {
            scatter_ptrs[i + 1] = scatter_ptrs[i] + row_counts[i];
        }

        // 3. Scatter entries into pre-allocated continuous flat arrays
        let mut scattered_cols = vec![0usize; total_nnz];
        let mut scattered_sources = vec![0i32; total_nnz]; // -1:K, -2:M, >=0:BoundaryTag
        let mut scattered_values = vec![0.0f64; total_nnz];

        let mut current_ptrs = scatter_ptrs.clone();

        for i in 0..stiffness.nnz() {
            let r = stiffness.rows[i];
            let idx = current_ptrs[r];
            scattered_cols[idx] = stiffness.cols[i];
            scattered_sources[idx] = -1;
            scattered_values[idx] = stiffness.values[i];
            current_ptrs[r] += 1;
        }

        for i in 0..mass.nnz() {
            let r = mass.rows[i];
            let idx = current_ptrs[r];
            scattered_cols[idx] = mass.cols[i];
            scattered_sources[idx] = -2;
            scattered_values[idx] = mass.values[i];
            current_ptrs[r] += 1;
        }

        for (tag, matrix) in boundaries {
            let t = *tag as i32;
            for i in 0..matrix.nnz() {
                let r = matrix.rows[i];
                let idx = current_ptrs[r];
                scattered_cols[idx] = matrix.cols[i];
                scattered_sources[idx] = t;
                scattered_values[idx] = matrix.values[i];
                current_ptrs[r] += 1;
            }
        }

        // 4. Construct final CSR logic per-row
        let mut row_ptrs = vec![0usize; num_rows + 1];
        let mut col_indices = Vec::with_capacity(total_nnz);
        let mut k_values = Vec::with_capacity(total_nnz);
        let mut m_values = Vec::with_capacity(total_nnz);

        let mut boundary_values_map: HashMap<usize, Vec<f64>> = HashMap::new();
        for (tag, _) in boundaries {
            boundary_values_map.insert(*tag, Vec::with_capacity(total_nnz));
        }

        for i in 0..num_rows {
            row_ptrs[i] = col_indices.len();

            let start = scatter_ptrs[i];
            let end = scatter_ptrs[i + 1];
            if start == end {
                continue;
            }

            // Extract row entries and sort columns locally
            let len = end - start;
            let mut row_entries: Vec<(usize, i32, f64)> = Vec::with_capacity(len);
            for idx in start..end {
                row_entries.push((
                    scattered_cols[idx],
                    scattered_sources[idx],
                    scattered_values[idx],
                ));
            }
            row_entries.sort_unstable_by_key(|&(c, _, _)| c);

            let mut prev_col = usize::MAX;
            for (c, source, v) in row_entries {
                if c != prev_col {
                    col_indices.push(c);
                    k_values.push(0.0);
                    m_values.push(0.0);
                    for vec in boundary_values_map.values_mut() {
                        vec.push(0.0);
                    }
                    prev_col = c;
                }

                let last_idx = col_indices.len() - 1;
                match source {
                    -1 => k_values[last_idx] += v,
                    -2 => m_values[last_idx] += v,
                    tag => {
                        if let Some(vec) = boundary_values_map.get_mut(&(tag as usize)) {
                            vec[last_idx] += v;
                        }
                    }
                }
            }
        }
        row_ptrs[num_rows] = col_indices.len();

        Self {
            num_rows,
            row_ptrs,
            col_indices,
            k_values,
            m_values,
            boundary_values: boundary_values_map,
        }
    }

    /// Create an empty CSR matrix with this assembler's fixed sparsity pattern.
    pub fn empty_matrix(&self) -> CsrMatrix<Complex64> {
        CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            vec![Complex64::new(0.0, 0.0); self.k_values.len()],
        )
    }

    /// Assemble Helmholtz matrix A = K - k²M + Σ c_i M_boundary_i
    pub fn assemble(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let mut matrix = self.empty_matrix();
        self.assemble_into(wavenumber, boundary_coeffs, &mut matrix);
        matrix
    }

    /// Fill an existing CSR matrix with A = K - k²M + Σ c_i M_boundary_i.
    pub fn assemble_into(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        out: &mut CsrMatrix<Complex64>,
    ) {
        assert_eq!(out.num_rows, self.num_rows);
        assert_eq!(out.num_cols, self.num_rows);
        assert_eq!(out.row_ptrs, self.row_ptrs);
        assert_eq!(out.col_indices, self.col_indices);
        assert_eq!(out.values.len(), self.k_values.len());

        let k_sq = wavenumber * wavenumber;

        let active_boundaries: Vec<(&[f64], Complex64)> = boundary_coeffs
            .iter()
            .filter_map(|(tag, &coeff)| {
                self.boundary_values.get(tag).map(|v| (v.as_slice(), coeff))
            })
            .collect();

        // Pointwise fill (rayon above the machine-adaptive gate, serial for
        // small patterns).
        let fill = |i: usize, val: &mut Complex64| {
            let mut next = Complex64::new(self.k_values[i], 0.0)
                - k_sq * Complex64::new(self.m_values[i], 0.0);
            for (b_vals, coeff) in &active_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    next += coeff * Complex64::new(bv, 0.0);
                }
            }
            *val = next;
        };
        if use_parallel(out.values.len(), POINTWISE_VALUES_PER_THREAD) {
            out.values
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| fill(i, val));
        } else {
            for (i, val) in out.values.iter_mut().enumerate() {
                fill(i, val);
            }
        }
    }

    fn helmholtz_and_shift_values(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        p_shift: Complex64,
        p_boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> (Vec<Complex64>, Vec<Complex64>) {
        let k_sq = wavenumber * wavenumber;
        let nnz = self.k_values.len();

        let active_boundaries: Vec<(&[f64], Complex64)> = boundary_coeffs
            .iter()
            .filter_map(|(tag, &coeff)| {
                self.boundary_values.get(tag).map(|v| (v.as_slice(), coeff))
            })
            .collect();
        let active_p_boundaries: Vec<(&[f64], Complex64)> = p_boundary_coeffs
            .iter()
            .filter_map(|(tag, &coeff)| {
                self.boundary_values.get(tag).map(|v| (v.as_slice(), coeff))
            })
            .collect();

        collect_indexed(nnz, POINTWISE_VALUES_PER_THREAD, |i| {
            let kv = self.k_values[i];
            let mv = self.m_values[i];
            let mut a_val = Complex64::new(kv, 0.0) - k_sq * Complex64::new(mv, 0.0);
            for (b_vals, coeff) in &active_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    a_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            let mut p_val = Complex64::new(kv, 0.0) + p_shift * Complex64::new(mv, 0.0);
            for (b_vals, coeff) in &active_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    p_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            for (b_vals, coeff) in &active_p_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    p_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            (a_val, p_val)
        })
        .into_iter()
        .unzip()
    }

    /// Assemble both `A = K - k² M + Σ c_i M_b` and a shifted-Laplacian
    /// preconditioner `P = K + p_shift · M + Σ c_i M_b` in a single parallel walk over
    /// the shared CSR storage.
    ///
    /// Frequency sweeps rebuild A and P together for every `k`. Doing them as
    /// two independent par_iters walks `k_values`, `m_values`, and the
    /// boundary slices twice; this version walks them once, halving memory
    /// traffic and matching the cost of a single `assemble()` call.
    pub fn assemble_helmholtz_and_shift(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        p_shift: Complex64,
    ) -> (CsrMatrix<Complex64>, CsrMatrix<Complex64>) {
        self.assemble_helmholtz_and_shift_with_preconditioner_boundaries(
            wavenumber,
            boundary_coeffs,
            p_shift,
            &HashMap::new(),
        )
    }

    /// Assemble both the physical Helmholtz matrix and a shifted-Laplacian
    /// preconditioner. Physical boundary terms are present in both matrices;
    /// extra preconditioner-only boundary terms are added only to `P`.
    ///
    /// `p_boundary_coeffs` is intentionally not applied to `A`. It is for
    /// artificial subdomain interfaces, e.g. ORAS/HABC transmission operators.
    pub fn assemble_helmholtz_and_shift_with_preconditioner_boundaries(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        p_shift: Complex64,
        p_boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> (CsrMatrix<Complex64>, CsrMatrix<Complex64>) {
        let (a_values, p_values) = self.helmholtz_and_shift_values(
            wavenumber,
            boundary_coeffs,
            p_shift,
            p_boundary_coeffs,
        );
        let a = CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            a_values,
        );
        let p = CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            p_values,
        );
        (a, p)
    }

    /// Fill existing CSR matrices with the physical Helmholtz matrix and
    /// shifted-Laplacian preconditioner using the assembler's fixed topology.
    pub fn assemble_helmholtz_and_shift_with_preconditioner_boundaries_into(
        &self,
        wavenumber: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        p_shift: Complex64,
        p_boundary_coeffs: &HashMap<usize, Complex64>,
        a: &mut CsrMatrix<Complex64>,
        p: &mut CsrMatrix<Complex64>,
    ) {
        assert_eq!(a.num_rows, self.num_rows);
        assert_eq!(a.num_cols, self.num_rows);
        assert_eq!(a.row_ptrs, self.row_ptrs);
        assert_eq!(a.col_indices, self.col_indices);
        assert_eq!(a.values.len(), self.k_values.len());
        assert_eq!(p.num_rows, self.num_rows);
        assert_eq!(p.num_cols, self.num_rows);
        assert_eq!(p.row_ptrs, self.row_ptrs);
        assert_eq!(p.col_indices, self.col_indices);
        assert_eq!(p.values.len(), self.k_values.len());

        let k_sq = wavenumber * wavenumber;

        let active_boundaries: Vec<(&[f64], Complex64)> = boundary_coeffs
            .iter()
            .filter_map(|(tag, &coeff)| {
                self.boundary_values.get(tag).map(|v| (v.as_slice(), coeff))
            })
            .collect();
        let active_p_boundaries: Vec<(&[f64], Complex64)> = p_boundary_coeffs
            .iter()
            .filter_map(|(tag, &coeff)| {
                self.boundary_values.get(tag).map(|v| (v.as_slice(), coeff))
            })
            .collect();

        // Single-walk pointwise fill of both matrices (rayon above the
        // machine-adaptive gate, serial for small patterns).
        let fill_both = |i: usize, a_out: &mut Complex64, p_out: &mut Complex64| {
            let kv = self.k_values[i];
            let mv = self.m_values[i];
            let mut a_val = Complex64::new(kv, 0.0) - k_sq * Complex64::new(mv, 0.0);
            for (b_vals, coeff) in &active_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    a_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            let mut p_val = Complex64::new(kv, 0.0) + p_shift * Complex64::new(mv, 0.0);
            for (b_vals, coeff) in &active_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    p_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            for (b_vals, coeff) in &active_p_boundaries {
                let bv = b_vals[i];
                if bv != 0.0 {
                    p_val += coeff * Complex64::new(bv, 0.0);
                }
            }
            *a_out = a_val;
            *p_out = p_val;
        };
        if use_parallel(a.values.len(), POINTWISE_VALUES_PER_THREAD) {
            a.values
                .par_iter_mut()
                .zip(p.values.par_iter_mut())
                .enumerate()
                .for_each(|(i, (a_out, p_out))| fill_both(i, a_out, p_out));
        } else {
            for (i, (a_out, p_out)) in a.values.iter_mut().zip(p.values.iter_mut()).enumerate() {
                fill_both(i, a_out, p_out);
            }
        }
    }

    /// Assemble Shifted-Laplacian preconditioner P = K + (α + iβ)M
    ///
    /// Uses the same CSR sparsity pattern as `assemble()` but with different
    /// linear combination coefficients. The complex shift (α + iβ) transforms
    /// the indefinite Helmholtz operator to a more favorable spectrum for AMG.
    ///
    /// Reference: Erlangga et al. (2006) "A class of preconditioners for the Helmholtz equation"
    pub fn assemble_shifted_laplacian(&self, alpha: f64, beta: f64) -> CsrMatrix<Complex64> {
        let shift = Complex64::new(alpha, beta);
        let nnz = self.k_values.len();

        let values: Vec<Complex64> = collect_indexed(nnz, POINTWISE_VALUES_PER_THREAD, |i| {
            Complex64::new(self.k_values[i], 0.0) + shift * Complex64::new(self.m_values[i], 0.0)
        });

        CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            values,
        )
    }

    /// Assemble a real-valued matrix as a linear combination: result = k_coeff * K + m_coeff * M
    ///
    /// This is used by WaveHoltz to build SPD matrices like:
    /// - A_impl = M + (dt²/4) K  → k_coeff = dt²/4, m_coeff = 1
    /// - B_rhs  = 2M - (dt²/2) K → k_coeff = -dt²/2, m_coeff = 2
    pub fn assemble_real(&self, k_coeff: f64, m_coeff: f64) -> CsrMatrix<f64> {
        let nnz = self.k_values.len();

        let values: Vec<f64> = collect_indexed(nnz, POINTWISE_VALUES_PER_THREAD, |i| {
            k_coeff * self.k_values[i] + m_coeff * self.m_values[i]
        });

        CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            values,
        )
    }

    /// Assemble a real-valued matrix with boundary contributions
    ///
    /// result = k_coeff * K + m_coeff * M + Σ `boundary_coeffs[tag]` * `M_boundary[tag]`
    pub fn assemble_real_with_boundaries(
        &self,
        k_coeff: f64,
        m_coeff: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let nnz = self.k_values.len();

        let values: Vec<f64> = collect_indexed(nnz, POINTWISE_VALUES_PER_THREAD, |i| {
            let mut val = k_coeff * self.k_values[i] + m_coeff * self.m_values[i];

            if !self.boundary_values.is_empty() {
                for (tag, &coeff) in boundary_coeffs {
                    if let Some(b_vals) = self.boundary_values.get(tag)
                        && b_vals[i] != 0.0
                    {
                        val += coeff * b_vals[i];
                    }
                }
            }
            val
        });

        CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            values,
        )
    }

    /// Apply Dirichlet boundary conditions to the assembler's K, M, and boundary matrices
    ///
    /// For each Dirichlet node i:
    /// Apply Dirichlet boundary conditions by zeroing rows AND columns of
    /// constrained DOFs, setting `K[i,i] = 1`, `M[i,i] = 0`.
    ///
    /// Both rows and columns are zeroed to preserve matrix symmetry.
    /// This ensures WaveHoltz (which builds SPD matrices from K and M)
    /// correctly enforces Dirichlet BCs with CG.
    pub fn apply_dirichlet_nodes(&mut self, dirichlet_nodes: &std::collections::HashSet<usize>) {
        if dirichlet_nodes.is_empty() {
            return;
        }

        for row in 0..self.num_rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            let is_constrained_row = dirichlet_nodes.contains(&row);

            for idx in start..end {
                let col = self.col_indices[idx];

                if is_constrained_row {
                    // Zero entire row, set diagonal to K=1, M=0
                    if row == col {
                        self.k_values[idx] = 1.0;
                        self.m_values[idx] = 0.0;
                    } else {
                        self.k_values[idx] = 0.0;
                        self.m_values[idx] = 0.0;
                    }
                    for bv in self.boundary_values.values_mut() {
                        bv[idx] = 0.0;
                    }
                } else if dirichlet_nodes.contains(&col) {
                    // Zero column entries in non-constrained rows
                    self.k_values[idx] = 0.0;
                    self.m_values[idx] = 0.0;
                    for bv in self.boundary_values.values_mut() {
                        bv[idx] = 0.0;
                    }
                }
            }
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let usize_size = std::mem::size_of::<usize>();
        let f64_size = std::mem::size_of::<f64>();

        let mut mem = self.row_ptrs.len() * usize_size
            + self.col_indices.len() * usize_size
            + self.k_values.len() * f64_size
            + self.m_values.len() * f64_size;

        for v in self.boundary_values.values() {
            mem += v.len() * f64_size;
        }
        mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_assembler_simple() {
        let mesh = unit_square_triangles(2);
        let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);

        let k = Complex64::new(1.0, 0.0);
        let coeffs = HashMap::new();
        let matrix = assembler.assemble(k, &coeffs);

        assert_eq!(matrix.num_rows, mesh.num_nodes());
        assert!(matrix.nnz() > 0);
    }

    #[test]
    fn test_assembler_with_boundary() {
        use crate::assembly::mass::assemble_boundary_mass;

        let mut mesh = unit_square_triangles(2);
        mesh.detect_boundaries(); // sets marker 0 for all boundaries

        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 0);

        let boundaries = vec![(0, b_mass)];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);

        let k = Complex64::new(1.0, 0.0);
        let mut coeffs = HashMap::new();
        coeffs.insert(0, Complex64::new(0.5, 0.0)); // Add 0.5 * BoundaryMass

        let matrix = assembler.assemble(k, &coeffs);

        // Check if values are different from base K-M
        let assembler_base = HelmholtzAssembler::from_matrices(&stiffness, &mass, &[]);
        let matrix_base = assembler_base.assemble(k, &HashMap::new());

        // Sum of values should differ
        let sum: Complex64 = matrix.values.iter().sum();
        let sum_base: Complex64 = matrix_base.values.iter().sum();

        assert!((sum - sum_base).norm() > 1e-10);
    }

    #[test]
    fn fused_helmholtz_and_shift_matches_separate_calls() {
        use crate::assembly::mass::assemble_boundary_mass;

        let mut mesh = unit_square_triangles(3);
        mesh.detect_boundaries();

        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 0);
        let boundaries = vec![(0, b_mass)];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);

        let k = Complex64::new(2.5, 0.1);
        let p_shift = Complex64::new(0.5 * 2.5 * 2.5, 0.5 * 2.5);
        let mut coeffs = HashMap::new();
        coeffs.insert(0, Complex64::new(0.7, -0.2));

        let a_ref = assembler.assemble(k, &coeffs);
        let mut p_ref = assembler.assemble_shifted_laplacian(p_shift.re, p_shift.im);
        let physical_boundary = assembler.assemble(Complex64::new(0.0, 0.0), &coeffs);
        for ((p, boundary), stiffness) in p_ref
            .values
            .iter_mut()
            .zip(physical_boundary.values.iter())
            .zip(assembler.k_values.iter())
        {
            *p += *boundary - Complex64::new(*stiffness, 0.0);
        }
        let (a_fused, p_fused) = assembler.assemble_helmholtz_and_shift(k, &coeffs, p_shift);
        let mut a_reused = assembler.empty_matrix();
        let mut p_reused = assembler.empty_matrix();
        assembler.assemble_helmholtz_and_shift_with_preconditioner_boundaries_into(
            k,
            &coeffs,
            p_shift,
            &HashMap::new(),
            &mut a_reused,
            &mut p_reused,
        );

        assert_eq!(a_ref.values.len(), a_fused.values.len());
        assert_eq!(p_ref.values.len(), p_fused.values.len());
        for ((r, f), reused) in a_ref
            .values
            .iter()
            .zip(a_fused.values.iter())
            .zip(a_reused.values.iter())
        {
            assert!((r - f).norm() < 1e-12, "A mismatch: {r} vs {f}");
            assert!(
                (r - reused).norm() < 1e-12,
                "reused A mismatch: {r} vs {reused}"
            );
        }
        for ((r, f), reused) in p_ref
            .values
            .iter()
            .zip(p_fused.values.iter())
            .zip(p_reused.values.iter())
        {
            assert!((r - f).norm() < 1e-12, "P mismatch: {r} vs {f}");
            assert!(
                (r - reused).norm() < 1e-12,
                "reused P mismatch: {r} vs {reused}"
            );
        }
    }

    #[test]
    fn preconditioner_boundary_terms_only_affect_shifted_matrix() {
        use crate::assembly::mass::assemble_boundary_mass;

        let mut mesh = unit_square_triangles(2);
        mesh.detect_boundaries();

        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 0);
        let boundaries = vec![(99, b_mass)];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);

        let k = Complex64::new(2.0, 0.0);
        let p_shift = Complex64::new(2.0, 1.0);
        let mut p_coeffs = HashMap::new();
        p_coeffs.insert(99, Complex64::new(0.0, 2.0));

        let (a_without, p_without) =
            assembler.assemble_helmholtz_and_shift(k, &HashMap::new(), p_shift);
        let (a_with, p_with) = assembler
            .assemble_helmholtz_and_shift_with_preconditioner_boundaries(
                k,
                &HashMap::new(),
                p_shift,
                &p_coeffs,
            );

        assert_eq!(a_without.values, a_with.values);
        assert!(
            p_without
                .values
                .iter()
                .zip(p_with.values.iter())
                .any(|(a, b)| (a - b).norm() > 1e-12)
        );
    }

    #[test]
    fn physical_boundary_terms_affect_both_operator_and_preconditioner() {
        use crate::assembly::mass::assemble_boundary_mass;

        let mut mesh = unit_square_triangles(2);
        mesh.detect_boundaries();

        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 0);
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &[(7, b_mass)]);
        let mut physical_coeffs = HashMap::new();
        physical_coeffs.insert(7, Complex64::new(0.1, -0.7));

        let (a_without, p_without) = assembler.assemble_helmholtz_and_shift(
            Complex64::new(2.0, 0.1),
            &HashMap::new(),
            Complex64::new(2.0, 1.0),
        );
        let (a_with, p_with) = assembler.assemble_helmholtz_and_shift(
            Complex64::new(2.0, 0.1),
            &physical_coeffs,
            Complex64::new(2.0, 1.0),
        );

        assert!(
            a_without
                .values
                .iter()
                .zip(&a_with.values)
                .any(|(a, b)| (a - b).norm() > 1e-12)
        );
        assert!(
            p_without
                .values
                .iter()
                .zip(&p_with.values)
                .any(|(a, b)| (a - b).norm() > 1e-12),
            "physical Robin terms must be represented in the shifted preconditioner",
        );
    }
}
