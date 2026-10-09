//! Hyperelastic nonlinear form for total Lagrangian FEM
//!
//! Implements [`NonlinearForm<f64>`] for hyperelastic materials.
//!
//! Residual: `R_i = integral P(F) : grad(N_i) dOmega_0`
//! Tangent:  `K_{ij} = integral A(F) : grad(N_i) otimes grad(N_j) dOmega_0`
//!
//! where `F = I + grad(u)` is the deformation gradient computed from the current
//! displacement field, `P = dW/dF` is the first Piola-Kirchhoff stress, and
//! `A = d²W/dF²` is the material tangent.

use crate::forms::{LocalMatrix, LocalVector, NonlinearForm};
use crate::spaces::BasisTabulation;

use super::materials::HyperelasticMaterial;

/// Hyperelastic nonlinear form.
///
/// Works with [`VectorH1Space`](crate::spaces::VectorH1Space).
/// The displacement `u` is stored in interleaved DOF ordering: `[u0_x, u0_y, u1_x, ...]`.
pub struct HyperelasticForm<M: HyperelasticMaterial> {
    pub material: M,
}

impl<M: HyperelasticMaterial> HyperelasticForm<M> {
    pub fn new(material: M) -> Self {
        Self { material }
    }
}

impl<M: HyperelasticMaterial> NonlinearForm<f64> for HyperelasticForm<M> {
    fn element_residual(
        &self,
        test: &BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = test.spatial_dim;
        let n_dofs = test.n_dofs;
        let n_nodes = n_dofs / dim;
        let w = weight * test.det_j;

        // Compute deformation gradient F = I + grad(u) at this quadrature point
        let f = deformation_gradient(test, local_solution, dim, n_nodes);

        // First Piola-Kirchhoff stress P(F)
        let p = self.material.stress(&f);

        // R_a = integral P_{iJ} dN_a_i/dX_J dOmega_0
        // For vector DOF a = (node, comp_i): dN_a_i/dX_J = delta_{comp_i, i} * dphi_node/dX_J
        for a in 0..n_dofs {
            let comp_i = a % dim;
            // dphi/dX_J for this DOF's underlying scalar shape function
            let mut r = 0.0;
            for jj in 0..dim {
                // P[comp_i * dim + jj] * dphi/dX_jj
                r += p[comp_i * dim + jj] * test.derivatives[a * dim + jj];
            }
            local_vector.add(a, r * w);
        }
    }

    fn element_tangent(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n_dofs = test.n_dofs;
        let n_nodes = n_dofs / dim;
        let w = weight * test.det_j;

        let f = deformation_gradient(test, local_solution, dim, n_nodes);
        let a_tangent = self.material.tangent(&f);
        let nn = dim * dim; // size of the Voigt-like tangent

        // K_{ab} = integral A_{iJkL} dN_a_i/dX_J dN_b_k/dX_L dOmega_0
        // For DOF a = (node_a, comp_i), DOF b = (node_b, comp_k):
        // K_{ab} = sum_{J,L} A[i*dim+J, k*dim+L] * dphi_a/dX_J * dphi_b/dX_L
        for a in 0..n_dofs {
            let comp_i = a % dim;
            for b in 0..n_dofs {
                let comp_k = b % dim;

                let mut k = 0.0;
                for jj in 0..dim {
                    let ij = comp_i * dim + jj;
                    let dphi_a_j = test.derivatives[a * dim + jj];
                    for l in 0..dim {
                        let kl = comp_k * dim + l;
                        let dphi_b_l = trial.derivatives[b * dim + l];
                        k += a_tangent[ij * nn + kl] * dphi_a_j * dphi_b_l;
                    }
                }
                local_matrix.add(a, b, k * w);
            }
        }
    }
}

/// Compute deformation gradient `F = I + grad(u)` at a quadrature point.
///
/// `local_solution` contains displacement DOFs in interleaved order.
/// `test` provides the physical-space gradients of the scalar basis functions.
fn deformation_gradient(
    test: &BasisTabulation,
    local_solution: &[f64],
    dim: usize,
    n_nodes: usize,
) -> Vec<f64> {
    // F_{iJ} = delta_{iJ} + du_i/dX_J
    // du_i/dX_J = sum_node u_{node,i} * dphi_node/dX_J
    let mut f = vec![0.0; dim * dim];

    // Identity
    for i in 0..dim {
        f[i * dim + i] = 1.0;
    }

    // Add displacement gradient
    for node in 0..n_nodes {
        for i in 0..dim {
            let dof = node * dim + i;
            let u_i = local_solution[dof];
            for jj in 0..dim {
                // dphi_node/dX_J: the derivative is stored at test.derivatives[dof * dim + jj]
                // But all component DOFs for the same node share the same scalar gradient.
                f[i * dim + jj] += u_i * test.derivatives[dof * dim + jj];
            }
        }
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::physics::solid::nonlinear_solid::NeoHookean;
    use crate::spaces::{FunctionSpace, VectorH1Space};

    #[test]
    fn test_residual_zero_at_identity() {
        // Zero displacement → F = I → P = 0 → residual = 0
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let mat = NeoHookean::from_young(1.0, 0.3, 2);
        let form = HyperelasticForm::new(mat);

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;
        let zero_sol = vec![0.0; n];
        let mut local_vec = LocalVector::zeros(n);

        form.element_residual(&tab, &zero_sol, 1.0, &mut local_vec);

        for (i, &v) in local_vec.values.iter().enumerate() {
            assert!(v.abs() < 1e-14, "R[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn test_tangent_symmetry_at_deformed_state() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let mat = NeoHookean::from_young(1.0, 0.3, 2);
        let form = HyperelasticForm::new(mat);

        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;
        // Small displacement
        let mut sol = vec![0.0; n];
        sol[0] = 0.01; // slight x-displacement of first node

        let mut local_mat = LocalMatrix::zeros(n);
        form.element_tangent(&tab, &tab, &sol, 1.0, &mut local_mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (local_mat.get(i, j) - local_mat.get(j, i)).abs();
                assert!(
                    diff < 1e-10,
                    "Tangent not symmetric at ({i},{j}): {} vs {}",
                    local_mat.get(i, j),
                    local_mat.get(j, i)
                );
            }
        }
    }

    #[test]
    fn test_element_residual_tangent_has_quadratic_taylor_remainder() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let mat = NeoHookean::from_young(1.0, 0.3, 2);
        let form = HyperelasticForm::new(mat);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let state = vec![0.01, -0.02, 0.005, 0.01, -0.004, 0.003];
        let direction = vec![0.2, -0.1, 0.05, 0.03, -0.07, 0.11];

        let report = crate::validation::check_tangent_consistency(
            |current| {
                let mut residual = LocalVector::zeros(current.len());
                form.element_residual(&tab, current, 1.0, &mut residual);
                residual.values
            },
            |current, tangent_direction| {
                let mut tangent = LocalMatrix::zeros(current.len());
                form.element_tangent(&tab, &tab, current, 1.0, &mut tangent);
                (0..current.len())
                    .map(|row| {
                        (0..current.len())
                            .map(|column| tangent.get(row, column) * tangent_direction[column])
                            .sum()
                    })
                    .collect()
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("hyperelastic residual and tangent dimensions must agree");

        assert!(report.observed_order > 1.8, "{report:?}");
    }
}
