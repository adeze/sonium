//! Stokes weak forms
//!
//! Three forms needed for the Stokes saddle-point system:
//!
//! - [`StokesViscousForm`]: `a(u,v) = mu * integral(grad(u) : grad(v) dOmega)` on VectorH1
//! - [`StokesDivergenceForm`]: `b(v,p) = -integral(p * div(v) dOmega)` (VectorH1 test x H1 trial)
//! - [`StokesBodyForceForm`]: `l(v) = integral(f . v dOmega)` on VectorH1

use crate::forms::{
    BilinearForm, DivergenceForm, LinearForm, LocalMatrix, LocalVector, VectorLaplacianForm,
};
use crate::spaces::BasisTabulation;

/// Viscous bilinear form: `a(u,v) = mu * integral(grad(u) : grad(v) dOmega)`
///
/// Delegates to [`VectorLaplacianForm`] with `coefficient = mu`.
#[derive(Debug, Clone)]
pub struct StokesViscousForm(VectorLaplacianForm);

impl StokesViscousForm {
    pub fn new(mu: f64) -> Self {
        Self(VectorLaplacianForm { coefficient: mu })
    }
}

impl BilinearForm<f64> for StokesViscousForm {
    #[inline]
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        self.0.integrate_point(test, trial, weight, local_matrix);
    }
}

/// Divergence coupling form: `b(v,p) = -integral(p * div(v) dOmega)`
///
/// Delegates to [`DivergenceForm`].
#[derive(Debug, Clone, Copy)]
pub struct StokesDivergenceForm;

impl BilinearForm<f64> for StokesDivergenceForm {
    #[inline]
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        DivergenceForm.integrate_point(test, trial, weight, local_matrix);
    }
}

/// Body force linear form: `l(v) = integral(f . v dOmega)` on VectorH1.
///
/// The force function `f` returns a vector value at each physical point.
pub struct StokesBodyForceForm {
    force: Box<dyn Fn([f64; 3]) -> Vec<f64> + Send + Sync>,
}

impl StokesBodyForceForm {
    /// Create a body force form from a function `f(x,y,z) -> [fx, fy, ...]`.
    pub fn new(force: impl Fn([f64; 3]) -> Vec<f64> + Send + Sync + 'static) -> Self {
        Self {
            force: Box::new(force),
        }
    }
}

impl LinearForm<f64> for StokesBodyForceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let w = weight * test.det_j;
        let f = (self.force)(test.physical_coords);

        for i in 0..n {
            let ci = i % dim;
            // v_i has value phi_{node} in component c_i only
            let phi = test.vector_value(i, ci);
            local_vector.add(i, f[ci] * phi * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

    #[test]
    fn test_stokes_viscous_form_symmetry() {
        let mesh = unit_square_triangles(4);
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let form = StokesViscousForm::new(1.0);
        let triplet = assembly::assemble_bilinear(&form, &space, &space, &mesh, 3);
        let csr = triplet.to_csr();

        // Viscous form should be symmetric: A[i,j] == A[j,i]
        let n = csr.num_rows;
        for i in 0..n {
            let start = csr.row_ptrs[i];
            let end = csr.row_ptrs[i + 1];
            for k in start..end {
                let j = csr.col_indices[k];
                let aij = csr.values[k];
                // Find A[j,i]
                let j_start = csr.row_ptrs[j];
                let j_end = csr.row_ptrs[j + 1];
                let aji = csr.col_indices[j_start..j_end]
                    .iter()
                    .zip(&csr.values[j_start..j_end])
                    .find(|&(&col, _)| col == i)
                    .map(|(_, &v)| v)
                    .unwrap_or(0.0);
                assert!(
                    (aij - aji).abs() < 1e-12,
                    "A[{i},{j}]={aij} != A[{j},{i}]={aji}"
                );
            }
        }
    }

    #[test]
    fn test_stokes_divergence_form_dimensions() {
        let mesh = unit_square_triangles(4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);

        let viscous = StokesViscousForm::new(1.0);
        let div_form = StokesDivergenceForm;

        // assemble_mixed gives us access to the coupling block
        let system =
            assembly::mixed::assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 3);

        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);

        // Block (0,1) is the divergence coupling: n_v x n_p
        let b01 = system.block_01.to_csr();
        assert_eq!(b01.num_rows, n_v);
        assert_eq!(b01.num_cols, n_p);
        assert!(b01.nnz() > 0);

        // Block (1,0) is the transpose: n_p x n_v
        let b10 = system.block_10.to_csr();
        assert_eq!(b10.num_rows, n_p);
        assert_eq!(b10.num_cols, n_v);
    }

    #[test]
    fn test_stokes_mixed_assembly() {
        let mesh = unit_square_triangles(4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);

        let viscous = StokesViscousForm::new(1.0);
        let div_form = StokesDivergenceForm;

        let system =
            assembly::mixed::assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 3);

        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);

        assert_eq!(system.n_dofs_0, n_v);
        assert_eq!(system.n_dofs_1, n_p);

        // Monolithic should have correct total dimensions
        let mono = system.to_monolithic();
        let csr = mono.to_csr();
        assert_eq!(csr.num_rows, n_v + n_p);
        assert_eq!(csr.num_cols, n_v + n_p);
    }
}
