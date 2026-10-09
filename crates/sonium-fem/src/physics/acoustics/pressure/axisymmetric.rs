//! Axisymmetric acoustic forms for duct/waveguide problems
//!
//! Solves the Helmholtz equation in cylindrical coordinates (r, z) assuming
//! azimuthal symmetry (m=0 mode). The mesh lives in the (r, z) meridional
//! plane; the x-coordinate of each node is the radial distance r.
//!
//! # Weak form
//!
//! The axisymmetric Helmholtz equation `∇²p + k²p = 0` in cylindrical coords
//! for the m=0 mode has the weak form:
//!
//! ```text
//! ∫∫ r [∇p·∇ψ − k² p ψ] dr dz = 0
//! ```
//!
//! The factor of 2π from azimuthal integration is absorbed into the overall
//! normalization and does not appear in the element matrices.
//!
//! The weak form is regular at r=0 because the test functions vanish there
//! (natural boundary condition on the axis of symmetry).
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::acoustics::pressure::axisymmetric::build_axi_acoustics_assembler;
//! use sonium_fem::basis::PolynomialDegree;
//!
//! let mesh = /* 2D mesh in (r, z) plane */;
//! let assembler = build_axi_acoustics_assembler(&mesh, PolynomialDegree::P1);
//! let system = assembler.assemble_helmholtz(k, &Default::default());
//! ```

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LocalMatrix};
use crate::mesh::Mesh;
use crate::spaces::{BasisTabulation, H1Space};
use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// Axisymmetric mass form: `m(u,v) = ∫ r · u · v dr dz`
///
/// The radial coordinate r is taken from `physical_coords[0]` of the
/// quadrature point (the mesh x-coordinate represents r).
#[derive(Debug, Clone, Copy)]
pub struct AxiMassForm;

impl BilinearForm<f64> for AxiMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let r = test.physical_coords[0]; // radial coordinate
        let w = weight * test.det_j * r;

        for i in 0..n {
            for j in 0..n {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Axisymmetric stiffness (Laplacian) form: `k(u,v) = ∫ r · ∇u · ∇v dr dz`
///
/// This is the weak form of the axisymmetric scalar Laplacian for the m=0
/// azimuthal mode. The gradient ∇ operates in the (r, z) plane.
///
/// Note: for m=0, the 1/r terms from the cylindrical Laplacian cancel in the
/// weak form after integration by parts, leaving just the r-weighted gradient
/// dot product. This is a standard result — see e.g. Zienkiewicz & Taylor,
/// "The Finite Element Method", Vol. 1, Section 9.4.
#[derive(Debug, Clone, Copy)]
pub struct AxiLaplacianForm;

impl BilinearForm<f64> for AxiLaplacianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r;

        for i in 0..n {
            for j in 0..n {
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Axisymmetric Helmholtz form: `a(u,v) = ∫ r [∇u·∇v − k² u v] dr dz`
///
/// One-shot assembly at a fixed wavenumber. For frequency sweeps, prefer
/// [`build_axi_acoustics_assembler`] which pre-bakes the stiffness and mass
/// components separately.
#[derive(Debug, Clone)]
pub struct AxiHelmholtzForm {
    pub wavenumber: Complex64,
}

impl AxiHelmholtzForm {
    pub fn new(wavenumber: Complex64) -> Self {
        Self { wavenumber }
    }
}

impl BilinearForm<Complex64> for AxiHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let k_sq = self.wavenumber * self.wavenumber;
        let r = test.physical_coords[0];
        let w = weight * test.det_j * r;

        for i in 0..n {
            for j in 0..n {
                let mut grad_dot = 0.0;
                for d in 0..dim {
                    grad_dot += test.grad(i, d) * trial.grad(j, d);
                }
                let mass = test.value(i) * trial.value(j);
                let val = Complex64::new(grad_dot * w, 0.0) - k_sq * Complex64::new(mass * w, 0.0);
                local_matrix.add(i, j, val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

/// Quadrature orders for radial-coordinate-weighted volume forms.
///
/// Multiplication by `r` raises the Cartesian polynomial degree by one. The
/// gradient order also integrates axisymmetric scalar-convection and streaming
/// forms exactly for the same trial/test degree.
pub(crate) fn axisymmetric_quadrature_orders(degree: PolynomialDegree) -> (usize, usize) {
    let polynomial_degree = degree.degree();
    ((2 * polynomial_degree).max(1), 2 * polynomial_degree + 2)
}

/// Type alias for the axisymmetric acoustics assembler (stiffness + mass).
pub type AxiAcousticsAssembler = PrebakedAssembler<2>;

/// Build an axisymmetric acoustics assembler from a 2D mesh in the (r, z) plane.
///
/// The mesh x-coordinate is treated as the radial distance r, and the
/// y-coordinate as the axial distance z. Elements at r=0 are handled
/// correctly (the r-weighting naturally enforces the axis BC).
///
/// Returns a [`PrebakedAssembler<2>`] with component 0 = stiffness, component 1 = mass.
/// Assemble the Helmholtz system as `K - k² M`.
pub fn build_axi_acoustics_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
) -> AxiAcousticsAssembler {
    let space = H1Space::new(degree);
    let (quad_stiffness, quad_mass) = axisymmetric_quadrature_orders(degree);

    let stiffness: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&AxiLaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass: TripletMatrix<f64> =
        crate::assembly::assemble_bilinear(&AxiMassForm, &space, &space, mesh, quad_mass);

    PrebakedAssembler::from_triplets([stiffness, mass], &[])
}

/// Build an axisymmetric acoustics assembler with boundary mass matrices.
///
/// Boundary mass matrices are also r-weighted for consistency with the
/// volume forms.
pub fn build_axi_acoustics_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
) -> AxiAcousticsAssembler {
    let space = H1Space::new(degree);
    let (quad_stiffness, quad_mass) = axisymmetric_quadrature_orders(degree);

    let stiffness =
        crate::assembly::assemble_bilinear(&AxiLaplacianForm, &space, &space, mesh, quad_stiffness);
    let mass = crate::assembly::assemble_bilinear(&AxiMassForm, &space, &space, mesh, quad_mass);

    let mut boundary_triplets = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_axi_boundary_mass(mesh, degree, tag);
        if b_mass.nnz() > 0 {
            let triplet = TripletMatrix {
                rows: b_mass.rows,
                cols: b_mass.cols,
                values: b_mass.values,
                num_rows: b_mass.dim,
                num_cols: b_mass.dim,
            };
            boundary_triplets.push((tag as usize, triplet));
        }
    }

    PrebakedAssembler::from_triplets([stiffness, mass], &boundary_triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axi_mass_form_r_weighting() {
        // Verify that the axisymmetric mass form produces r-weighted values.
        // At r=0, contributions should vanish; at r=1, should equal standard mass.
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![1.0, 0.5],
            derivatives: vec![0.0; 4],
            hessians: Vec::new(),
            det_j: 1.0,
            physical_coords: [0.5, 0.0, 0.0], // r = 0.5
        };

        let mut lm = LocalMatrix::zeros(2);
        AxiMassForm.integrate_point(&tab, &tab, 1.0, &mut lm);

        // phi_0 * phi_0 * r * det_j * w = 1.0 * 1.0 * 0.5 * 1.0 * 1.0 = 0.5
        assert!((lm.get(0, 0) - 0.5).abs() < 1e-14);
        // phi_0 * phi_1 * r = 1.0 * 0.5 * 0.5 = 0.25
        assert!((lm.get(0, 1) - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_axi_mass_vanishes_on_axis() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![1.0, 1.0],
            derivatives: vec![0.0; 4],
            hessians: Vec::new(),
            det_j: 1.0,
            physical_coords: [0.0, 1.0, 0.0], // r = 0 (on axis)
        };

        let mut lm = LocalMatrix::zeros(2);
        AxiMassForm.integrate_point(&tab, &tab, 1.0, &mut lm);

        // All entries should be zero at r=0
        for i in 0..2 {
            for j in 0..2 {
                assert!(lm.get(i, j).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_axi_laplacian_form_r_weighting() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![1.0, 0.5],
            derivatives: vec![1.0, 0.0, 0.0, 1.0], // grad_0 = (1,0), grad_1 = (0,1)
            hessians: Vec::new(),
            det_j: 1.0,
            physical_coords: [2.0, 0.0, 0.0], // r = 2
        };

        let mut lm = LocalMatrix::zeros(2);
        AxiLaplacianForm.integrate_point(&tab, &tab, 1.0, &mut lm);

        // grad_0 . grad_0 * r = 1.0 * 2.0 = 2.0
        assert!((lm.get(0, 0) - 2.0).abs() < 1e-14);
        // grad_0 . grad_1 * r = 0.0 * 2.0 = 0.0
        assert!(lm.get(0, 1).abs() < 1e-14);
        // grad_1 . grad_1 * r = 1.0 * 2.0 = 2.0
        assert!((lm.get(1, 1) - 2.0).abs() < 1e-14);
    }
}
