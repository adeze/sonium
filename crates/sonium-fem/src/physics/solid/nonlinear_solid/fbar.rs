//! F-bar formulation for near-incompressible nonlinear bricks.
//!
//! The deformation gradient at each quadrature point is rescaled to carry
//! the element-averaged Jacobian, `F_bar = (J_bar / J)^{1/3} F`, which
//! relieves volumetric locking while preserving the patch test. The tangent
//! is a central-difference Jacobian of the element residual.

// Rust guideline compliant 2026-02-21

use super::hex::{
    HexMesh, det3, hex_jacobian, hex_quadrature, hex_shape_derivatives, physical_gradients,
};
use super::materials::HyperelasticMaterial;

/// F-bar modified deformation gradient for one quadrature point.
#[must_use]
pub fn modified_deformation_gradient(f: &[f64; 9], j: f64, j_bar: f64) -> [f64; 9] {
    let scale = (j_bar / j).cbrt();
    let mut out = [0.0; 9];
    for i in 0..9 {
        out[i] = scale * f[i];
    }
    out
}

/// F-bar hexahedral form for near-incompressible hyperelasticity.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm)
/// with interleaved displacement DOFs (`[ux, uy, uz]` per node).
#[derive(Debug)]
pub struct FBarHexForm<M: HyperelasticMaterial> {
    /// Base hyperelastic material evaluated at the modified gradient.
    pub material: M,
    /// Hexahedral mesh.
    pub mesh: HexMesh,
    /// Total number of global DOFs.
    pub n_dofs: usize,
    /// Finite-difference step for the tangent stiffness.
    pub fd_step: f64,
}

impl<M: HyperelasticMaterial> FBarHexForm<M> {
    /// Build an F-bar form over a hex mesh.
    #[must_use]
    pub fn new(material: M, mesh: HexMesh) -> Self {
        assert_eq!(material.dim(), 3, "F-bar bricks need a 3D material");
        let n_dofs = 3 * mesh.num_nodes();
        Self {
            material,
            mesh,
            n_dofs,
            fd_step: 1.0e-8,
        }
    }

    /// Global DOF indices for one element.
    fn element_dofs(&self, elem_idx: usize) -> [usize; 24] {
        let mut dofs = [0_usize; 24];
        for a in 0..8 {
            let node = self.mesh.conn[elem_idx][a];
            dofs[a * 3] = node * 3;
            dofs[a * 3 + 1] = node * 3 + 1;
            dofs[a * 3 + 2] = node * 3 + 2;
        }
        dofs
    }

    /// Deformation gradients and reference weights at all Gauss points.
    fn quadrature_gradients(&self, elem_idx: usize, u: &[f64]) -> [([f64; 9], f64); 8] {
        let coords = self.mesh.element_coords(elem_idx);
        let dofs = self.element_dofs(elem_idx);
        let mut out = [([0.0; 9], 0.0); 8];
        for (qp, (natural, weight)) in hex_quadrature().iter().enumerate() {
            let derivatives = hex_shape_derivatives(natural);
            let (_, det, inv) = hex_jacobian(&coords, &derivatives);
            let grad = physical_gradients(&inv, &derivatives);
            let mut f = [0.0; 9];
            for i in 0..3 {
                f[i * 3 + i] = 1.0;
            }
            for a in 0..8 {
                for i in 0..3 {
                    let u_a = u[dofs[a * 3 + i]];
                    for col in 0..3 {
                        f[i * 3 + col] += u_a * grad[a][col];
                    }
                }
            }
            out[qp] = (f, weight * det);
        }
        out
    }

    /// Element-averaged Jacobian `J_bar` over the reference volume.
    fn average_jacobian(&self, gradients: &[([f64; 9], f64); 8]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (f, weight) in gradients {
            num += det3(f) * weight;
            den += weight;
        }
        num / den
    }

    /// Internal force residual for one element.
    fn internal_forces(&self, elem_idx: usize, u: &[f64]) -> [f64; 24] {
        let coords = self.mesh.element_coords(elem_idx);
        let gradients = self.quadrature_gradients(elem_idx, u);
        let j_bar = self.average_jacobian(&gradients);
        let mut residual = [0.0; 24];
        for (qp, (natural, _)) in hex_quadrature().iter().enumerate() {
            let (f, weight) = &gradients[qp];
            let j = det3(f);
            let f_bar = modified_deformation_gradient(f, j, j_bar);
            let p = self.material.stress(&f_bar);
            let derivatives = hex_shape_derivatives(natural);
            let (_, _, inv) = hex_jacobian(&coords, &derivatives);
            let grad = physical_gradients(&inv, &derivatives);
            for a in 0..8 {
                for i in 0..3 {
                    let mut r = 0.0;
                    for col in 0..3 {
                        r += p[i * 3 + col] * grad[a][col];
                    }
                    residual[a * 3 + i] += r * weight;
                }
            }
        }
        residual
    }
}

impl<M: HyperelasticMaterial> crate::forms::NonlinearElementForm<f64> for FBarHexForm<M> {
    fn num_elements(&self) -> usize {
        self.mesh.conn.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        24
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.element_dofs(elem_idx).to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        self.internal_forces(elem_idx, global_solution).to_vec()
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let dofs = self.element_dofs(elem_idx);
        let step = self.fd_step;
        let mut tangent = vec![0.0; 576];
        let mut perturbed = global_solution.to_vec();
        for (col, &dof) in dofs.iter().enumerate() {
            let base = global_solution[dof];
            let h = step * (1.0 + base.abs());
            perturbed[dof] = base + h;
            let plus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base - h;
            let minus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base;
            for row in 0..24 {
                tangent[row * 24 + col] = (plus[row] - minus[row]) / (2.0 * h);
            }
        }
        tangent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::NonlinearElementForm;
    use crate::physics::solid::nonlinear_solid::NeoHookean;

    fn cube_form() -> FBarHexForm<NeoHookean> {
        let material = NeoHookean::from_young(1.0e6, 0.45, 3);
        FBarHexForm::new(material, HexMesh::unit_cube(1.0))
    }

    #[test]
    fn homogeneous_deformation_needs_no_correction() {
        let form = cube_form();
        // Uniform dilatation: every quadrature Jacobian equals the average.
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let coords = form.mesh.coords[node];
            for i in 0..3 {
                u[node * 3 + i] = 0.05 * coords[i];
            }
        }
        let gradients = form.quadrature_gradients(0, &u);
        let j_bar = form.average_jacobian(&gradients);
        for (f, _) in &gradients {
            assert!((det3(f) - j_bar).abs() / j_bar < 1.0e-12);
            let corrected = modified_deformation_gradient(f, det3(f), j_bar);
            for i in 0..9 {
                assert!((corrected[i] - f[i]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn rigid_motion_produces_zero_residual() {
        let form = cube_form();
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            u[node * 3] = 0.3;
            u[node * 3 + 1] = -0.2;
            u[node * 3 + 2] = 0.1;
        }
        let residual = form.element_residual(0, &u);
        for (i, &r) in residual.iter().enumerate() {
            assert!(r.abs() < 1.0e-9, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn tangent_is_consistent_with_residual() {
        let form = cube_form();
        let state: Vec<f64> = (0..24)
            .map(|i| 0.01 * ((i * 37 % 11) as f64 - 5.0))
            .collect();
        let direction: Vec<f64> = (0..24)
            .map(|i| 0.05 * ((i * 53 % 7) as f64 - 3.0))
            .collect();
        let report = crate::validation::check_tangent_consistency(
            |current| form.element_residual(0, current),
            |current, tangent_direction| {
                let tangent = form.element_tangent(0, current);
                (0..current.len())
                    .map(|row| {
                        (0..current.len())
                            .map(|column| tangent[row * 24 + column] * tangent_direction[column])
                            .sum()
                    })
                    .collect()
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("f-bar residual and tangent dimensions must agree");
        assert!(report.observed_order > 1.8, "{report:?}");
    }
}
