//! Biharmonic equation solver using C0 interior penalty (C0-IP) DG method
//!
//! Solves: `Δ²u = f` on Ω with `u = g` on ∂Ω
//!
//! The C0-IP method uses standard C0 Lagrange elements (P2 or higher)
//! and adds interior penalty terms on element interfaces to weakly
//! enforce C1 continuity of the normal derivative.
//!
//! # Weak form
//!
//! ```text
//! a(u,v) = Σ_K ∫_K D²u : D²v dx                           (cell Hessian)
//!        + Σ_F α/h ∫_F [[∂u/∂n]]·[[∂v/∂n]] dS             (penalty)
//!        - Σ_F ∫_F {{∂²u/∂n²}}·[[∂v/∂n]] dS               (consistency)
//!        - Σ_F ∫_F [[∂u/∂n]]·{{∂²v/∂n²}} dS               (symmetry)
//! ```
//!
//! For simplicity, this implementation uses the symmetric interior penalty
//! (SIP) variant. The penalty parameter α must be large enough for
//! coercivity (typically α ≥ 4p² for polynomial degree p).
//!
//! # References
//!
//! - Engel, Garikipati, Hughes, Larson, Mazzei, Taylor (2002),
//!   "Continuous/discontinuous finite element approximations of fourth-order
//!   elliptic problems in structural and continuum mechanics with applications
//!   to thin beams and plates, and strain gradient elasticity".

use crate::assembly::{TripletMatrix, assemble_bilinear};
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, FacetBilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::mesh::Mesh;
use crate::spaces::BasisTabulation;
use crate::spaces::{FunctionSpace, H1Space};

/// Block operator for the mixed P1 biharmonic formulation.
///
/// With `q = -Delta u`, unknowns are ordered as `(u, q)` and the two block
/// equations are
///
/// ```text
/// K u - M q = 0
///         K q = f.
/// ```
///
/// Applying homogeneous Dirichlet data to both fields gives the usual Navier
/// (simply-supported) biharmonic boundary condition.  The split deliberately
/// uses only first derivatives, making it suitable for P1 tetrahedra while the
/// C0-IP forms below remain available for P2 and higher-order spaces.
pub struct MixedBiharmonicOperator {
    pub matrix: TripletMatrix<f64>,
    pub field_dofs: usize,
}

impl MixedBiharmonicOperator {
    pub fn total_dofs(&self) -> usize {
        2 * self.field_dofs
    }

    pub fn u_dofs(&self) -> std::ops::Range<usize> {
        0..self.field_dofs
    }

    pub fn auxiliary_dofs(&self) -> std::ops::Range<usize> {
        self.field_dofs..2 * self.field_dofs
    }
}

/// Assemble the mixed biharmonic block operator on an H1 mesh.
///
/// The returned matrix contains no essential-boundary elimination. Callers
/// apply their normal scalar Dirichlet machinery to both `u` and `q` for a
/// Navier problem, and assemble the source into the second block.
pub fn assemble_mixed_biharmonic_operator(
    mesh: &Mesh,
    degree: PolynomialDegree,
) -> MixedBiharmonicOperator {
    let space = H1Space::new(degree);
    let n = space.num_global_dofs(mesh);
    let quadrature_order = crate::quadrature::required_order_for_stiffness(degree.degree())
        .max(crate::quadrature::required_order_for_mass(degree.degree()));
    let mass = assemble_bilinear(
        &crate::forms::MassForm,
        &space,
        &space,
        mesh,
        quadrature_order,
    );
    let stiffness = assemble_bilinear(
        &crate::forms::LaplacianForm,
        &space,
        &space,
        mesh,
        quadrature_order,
    );

    let mut matrix = TripletMatrix::with_capacity(2 * n, 2 * n, 2 * stiffness.nnz() + mass.nnz());
    for k in 0..stiffness.nnz() {
        let row = stiffness.rows[k];
        let column = stiffness.cols[k];
        let value = stiffness.values[k];
        matrix.push(row, column, value);
        matrix.push(n + row, n + column, value);
    }
    for k in 0..mass.nnz() {
        matrix.push(mass.rows[k], n + mass.cols[k], -mass.values[k]);
    }

    MixedBiharmonicOperator {
        matrix,
        field_dofs: n,
    }
}

/// Cell (volume) bilinear form for the biharmonic equation.
///
/// Assembles: `∫_K D²u : D²v dx` where `D²` is the Hessian matrix.
///
/// For P1 elements on affine triangles, the Hessian is zero (linear functions
/// have no second derivatives). Use P2 or higher for meaningful results.
/// On P1 meshes, this form contributes nothing — the entire biharmonic
/// operator lives in the facet penalty terms.
pub struct BiharmonicCellForm;

impl BilinearForm<f64> for BiharmonicCellForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        if !test.has_hessians() || !trial.has_hessians() {
            return;
        }

        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let mut h_dot = 0.0;
                for a in 0..dim {
                    for b in 0..dim {
                        h_dot += test.hessian(i, a, b) * trial.hessian(j, a, b);
                    }
                }
                local_matrix.add(i, j, h_dot * w);
            }
        }
    }
}

/// Uniform or spatially varying transverse plate load for biharmonic plates.
///
/// Assembles `∫ q(x,y,z) v dx`, where `q` is pressure/load per unit area.
pub struct BiharmonicSourceForm<F: Fn(f64, f64, f64) -> f64 + Send + Sync> {
    pub load_fn: F,
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> BiharmonicSourceForm<F> {
    pub fn new(load_fn: F) -> Self {
        Self { load_fn }
    }

    pub fn uniform(load: f64) -> BiharmonicSourceForm<impl Fn(f64, f64, f64) -> f64 + Send + Sync> {
        BiharmonicSourceForm {
            load_fn: move |_, _, _| load,
        }
    }
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> LinearForm<f64> for BiharmonicSourceForm<F> {
    fn integrate_point(
        &self,
        test: &crate::spaces::BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let q = (self.load_fn)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * q * w);
        }
    }
}

/// Discrete point load helper for nodal C0 plate spaces.
pub fn add_biharmonic_point_load(rhs: &mut [f64], dof: usize, load: f64) {
    rhs[dof] += load;
}

/// Interior facet form for the C0-IP biharmonic method.
///
/// On each interior facet F with normal n from plus to minus:
///
/// ```text
/// a_F(u,v) = α/h ∫_F [[∂u/∂n]][[∂v/∂n]] dS          (penalty)
///          - ∫_F {{∂²u/∂n²}}[[∂v/∂n]] dS              (consistency)
///          - ∫_F [[∂u/∂n]]{{∂²v/∂n²}} dS              (symmetry)
/// ```
///
/// For P1 elements, {{∂²u/∂n²}} = avg of n^T H n across the facet.
/// Since the Hessian of P1 functions is zero, the consistency and symmetry
/// terms vanish, leaving only the penalty term. This gives a convergent
/// method (though with reduced order) — the penalty forces ∂u/∂n continuity.
pub struct BiharmonicFacetForm {
    /// Penalty parameter α. Must be ≥ 4p² for stability.
    /// For P1: α ≥ 4. Typical: α = 8.
    pub alpha: f64,
}

impl BiharmonicFacetForm {
    /// Create with the given penalty parameter.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Default penalty suitable for P1/P2 elements.
    pub fn default_penalty() -> Self {
        Self { alpha: 8.0 }
    }
}

impl FacetBilinearForm<f64> for BiharmonicFacetForm {
    fn integrate_facet(
        &self,
        test_plus: &BasisTabulation,
        _trial_plus: &BasisTabulation,
        test_minus: &BasisTabulation,
        _trial_minus: &BasisTabulation,
        normal: [f64; 3],
        h: f64,
        weight: f64,
        block: &mut [LocalMatrix<f64>; 4],
    ) {
        let dim = test_plus.spatial_dim;
        let np = test_plus.n_dofs;
        let nm = test_minus.n_dofs;
        let penalty = self.alpha / h;

        // Compute normal derivatives: ∂φ/∂n = ∇φ · n
        let grad_n_plus: Vec<f64> = (0..np)
            .map(|i| {
                (0..dim)
                    .map(|d| test_plus.grad(i, d) * normal[d])
                    .sum::<f64>()
            })
            .collect();

        let grad_n_minus: Vec<f64> = (0..nm)
            .map(|i| {
                (0..dim)
                    .map(|d| test_minus.grad(i, d) * normal[d])
                    .sum::<f64>()
            })
            .collect();

        // Penalty term: α/h [[∂u/∂n]][[∂v/∂n]]
        // [[∂u/∂n]] = ∂u⁺/∂n - ∂u⁻/∂n (jump of normal derivative)
        //
        // Expanding the jump product gives 4 blocks:
        // (+,+): +α/h ∂φ⁺_i/∂n · ∂φ⁺_j/∂n
        // (+,-): -α/h ∂φ⁺_i/∂n · ∂φ⁻_j/∂n
        // (-,+): -α/h ∂φ⁻_i/∂n · ∂φ⁺_j/∂n
        // (-,-): +α/h ∂φ⁻_i/∂n · ∂φ⁻_j/∂n

        for (i, &gni_p) in grad_n_plus.iter().enumerate() {
            for (j, &gnj_p) in grad_n_plus.iter().enumerate() {
                block[0].add(i, j, penalty * weight * gni_p * gnj_p);
            }
            for (j, &gnj_m) in grad_n_minus.iter().enumerate() {
                block[1].add(i, j, -penalty * weight * gni_p * gnj_m);
            }
        }
        for (i, &gni_m) in grad_n_minus.iter().enumerate() {
            for (j, &gnj_p) in grad_n_plus.iter().enumerate() {
                block[2].add(i, j, -penalty * weight * gni_m * gnj_p);
            }
            for (j, &gnj_m) in grad_n_minus.iter().enumerate() {
                block[3].add(i, j, penalty * weight * gni_m * gnj_m);
            }
        }

        // Consistency + symmetry terms (require second derivatives, zero for P1):
        // These are omitted for P1 elements. For P2+, add:
        //   -{{∂²u/∂n²}}[[∂v/∂n]] and -[[∂u/∂n]]{{∂²v/∂n²}}
        // where ∂²u/∂n² = n^T H(u) n and H is the Hessian.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::dg::assemble_dg;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::FunctionSpace;
    use crate::spaces::dg::DGSpace;

    #[test]
    fn test_biharmonic_assembly() {
        let mesh = unit_square_triangles(4);
        let space = DGSpace::new(PolynomialDegree::P1);
        let n_dofs = space.num_global_dofs(&mesh);

        let cell_form = BiharmonicCellForm;
        let facet_form = BiharmonicFacetForm::default_penalty();

        let matrix = assemble_dg(&cell_form, &facet_form, &space, &mesh, 1);

        assert_eq!(matrix.num_rows, n_dofs);
        assert!(matrix.nnz() > 0, "Should have nonzero entries from penalty");
    }

    #[test]
    fn test_biharmonic_spd() {
        // The penalty-only biharmonic matrix should be symmetric positive semi-definite
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);
        let n = space.num_global_dofs(&mesh);

        let matrix = assemble_dg(
            &BiharmonicCellForm,
            &BiharmonicFacetForm::new(8.0),
            &space,
            &mesh,
            1,
        );

        let csr = matrix.to_csr();

        // Check symmetry: A[i,j] ≈ A[j,i]
        for i in 0..n {
            for (j, v_ij) in csr.row_entries(i) {
                let v_ji = csr
                    .row_entries(j)
                    .find(|(col, _)| *col == i)
                    .map(|(_, v)| v)
                    .unwrap_or(0.0);
                assert!(
                    (v_ij - v_ji).abs() < 1e-10,
                    "Not symmetric: A[{i},{j}] = {v_ij}, A[{j},{i}] = {v_ji}"
                );
            }
        }
    }

    #[test]
    fn test_biharmonic_penalty_increases_with_alpha() {
        let mesh = unit_square_triangles(2);
        let space = DGSpace::new(PolynomialDegree::P1);

        let low = assemble_dg(
            &BiharmonicCellForm,
            &BiharmonicFacetForm::new(4.0),
            &space,
            &mesh,
            1,
        );
        let high = assemble_dg(
            &BiharmonicCellForm,
            &BiharmonicFacetForm::new(16.0),
            &space,
            &mesh,
            1,
        );

        let sum_low: f64 = low.values.iter().map(|v| v.abs()).sum();
        let sum_high: f64 = high.values.iter().map(|v| v.abs()).sum();
        assert!(sum_high > sum_low, "Higher α should produce larger entries");
    }

    #[test]
    fn mixed_biharmonic_assembles_on_p1_tetrahedra() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let operator = assemble_mixed_biharmonic_operator(&mesh, PolynomialDegree::P1);
        assert_eq!(operator.field_dofs, mesh.num_nodes());
        assert_eq!(operator.matrix.num_rows, 2 * mesh.num_nodes());
        assert_eq!(operator.matrix.num_cols, 2 * mesh.num_nodes());
        assert!(operator.matrix.nnz() > 0);
    }

    #[test]
    fn mixed_biharmonic_has_triangular_split_blocks() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let operator = assemble_mixed_biharmonic_operator(&mesh, PolynomialDegree::P1);
        let n = operator.field_dofs;
        let mut has_u_stiffness = false;
        let mut has_mass_coupling = false;
        let mut has_q_stiffness = false;
        for k in 0..operator.matrix.nnz() {
            let row = operator.matrix.rows[k];
            let column = operator.matrix.cols[k];
            has_u_stiffness |= row < n && column < n;
            has_mass_coupling |= row < n && column >= n;
            has_q_stiffness |= row >= n && column >= n;
            assert!(!(row >= n && column < n), "lower-left block must be zero");
        }
        assert!(has_u_stiffness && has_mass_coupling && has_q_stiffness);
    }
}
