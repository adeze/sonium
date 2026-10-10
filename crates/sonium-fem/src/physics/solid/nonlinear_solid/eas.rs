//! Enhanced assumed strain bricks for bending and incompressibility.
//!
//! A nine-mode incompatible enhancement of the deformation gradient relieves
//! shear and volumetric locking on H8 bricks. Enhanced parameters equilibrate
//! locally per residual evaluation, so the element stays stateless; the
//! condensed tangent is a central-difference Jacobian of the residual.

// Rust guideline compliant 2026-02-21

use super::hex::{
    HexMesh, dense_solve, hex_jacobian, hex_quadrature, hex_shape_derivatives, physical_gradients,
};
use super::materials::HyperelasticMaterial;

/// EAS incompatible modes as natural-coordinate 3x3 matrices.
///
/// Modes enhance `F11/F22/F33` diagonally plus symmetric shear pairs, the
/// Q1/E9 pattern with zero mean over the reference cube.
fn enhancement_modes(natural: &[f64; 3]) -> [[f64; 9]; 9] {
    let (xi, eta, zeta) = (natural[0], natural[1], natural[2]);
    let mut modes = [[0.0; 9]; 9];
    modes[0][0] = xi;
    modes[1][4] = eta;
    modes[2][8] = zeta;
    modes[3][1] = xi;
    modes[3][3] = xi;
    modes[4][1] = eta;
    modes[4][3] = eta;
    modes[5][5] = eta;
    modes[5][7] = eta;
    modes[6][5] = zeta;
    modes[6][7] = zeta;
    modes[7][2] = xi;
    modes[7][6] = xi;
    modes[8][2] = zeta;
    modes[8][6] = zeta;
    modes
}

/// Row-major 3x3 product `C = A B`.
fn mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    out
}

/// Double contraction `A : B` of two 3x3 matrices.
fn contract(a: &[f64], b: &[f64; 9]) -> f64 {
    let mut sum = 0.0;
    for i in 0..9 {
        sum += a[i] * b[i];
    }
    sum
}

/// Per-quadrature data shared by the local and global passes.
struct QuadraturePoint {
    /// Reference weight `w detJ0`.
    weight: f64,
    /// Physical shape gradients per node.
    grad: [[f64; 3]; 8],
    /// Compatible deformation gradient.
    f: [f64; 9],
    /// Push-forward enhancement per mode.
    g: [[f64; 9]; 9],
}

/// EAS hexahedral form with nine incompatible modes.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm)
/// with interleaved displacement DOFs (`[ux, uy, uz]` per node).
#[derive(Debug)]
pub struct EasHexForm<M: HyperelasticMaterial> {
    /// Base hyperelastic material evaluated at the enhanced gradient.
    pub material: M,
    /// Hexahedral mesh.
    pub mesh: HexMesh,
    /// Total number of global DOFs.
    pub n_dofs: usize,
    /// Finite-difference step for the tangent stiffness.
    pub fd_step: f64,
    /// Local Newton relative step tolerance.
    pub local_tolerance: f64,
    /// Local Newton iteration cap.
    pub local_max_iterations: usize,
}

impl<M: HyperelasticMaterial> EasHexForm<M> {
    /// Build an EAS form over a hex mesh.
    #[must_use]
    pub fn new(material: M, mesh: HexMesh) -> Self {
        assert_eq!(material.dim(), 3, "EAS bricks need a 3D material");
        let n_dofs = 3 * mesh.num_nodes();
        Self {
            material,
            mesh,
            n_dofs,
            fd_step: 1.0e-8,
            local_tolerance: 1.0e-12,
            local_max_iterations: 25,
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

    /// Compatible deformation gradient from nodal displacements.
    fn compatible_f(&self, dofs: &[usize; 24], u: &[f64], grad: &[[f64; 3]; 8]) -> [f64; 9] {
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
        f
    }

    /// Quadrature data with push-forward enhancement operators.
    fn quadrature_data(&self, elem_idx: usize, u: &[f64]) -> [QuadraturePoint; 8] {
        let coords = self.mesh.element_coords(elem_idx);
        let dofs = self.element_dofs(elem_idx);
        // Centroid map and deformation gradient drive the push-forward.
        let center = [0.0, 0.0, 0.0];
        let center_derivatives = hex_shape_derivatives(&center);
        let (_, j0_det, j0_inv) = hex_jacobian(&coords, &center_derivatives);
        let center_grad = physical_gradients(&j0_inv, &center_derivatives);
        let f0 = self.compatible_f(&dofs, u, &center_grad);

        let quadrature = hex_quadrature();
        std::array::from_fn(|qp| {
            let (natural, weight) = quadrature[qp];
            let derivatives = hex_shape_derivatives(&natural);
            let (_, det, inv) = hex_jacobian(&coords, &derivatives);
            let grad = physical_gradients(&inv, &derivatives);
            let f = self.compatible_f(&dofs, u, &grad);
            let modes = enhancement_modes(&natural);
            let scale = j0_det / det;
            let mut g = [[0.0; 9]; 9];
            for k in 0..9 {
                // G = (j0 / j) F0 M J0^{-1}.
                let pushed = mat3(&mat3(&f0, &modes[k]), &j0_inv);
                for i in 0..9 {
                    g[k][i] = scale * pushed[i];
                }
            }
            QuadraturePoint {
                weight: weight * det,
                grad,
                f,
                g,
            }
        })
    }

    /// Enhanced deformation gradient at one quadrature point.
    fn enhanced_f(&self, point: &QuadraturePoint, alpha: &[f64; 9]) -> [f64; 9] {
        let mut f = point.f;
        for (k, mode) in point.g.iter().enumerate() {
            for (entry, increment) in f.iter_mut().zip(mode.iter()) {
                *entry += alpha[k] * increment;
            }
        }
        f
    }

    /// Local Newton solve for the enhanced parameters.
    ///
    /// Stops on the Newton step size, which tracks the parameter error
    /// without depending on the residual scale (`local_tolerance` is a
    /// relative step tolerance).
    fn equilibrate(&self, points: &[QuadraturePoint; 8]) -> [f64; 9] {
        let mut alpha = [0.0; 9];
        for _ in 0..self.local_max_iterations {
            let mut residual = [0.0; 9];
            let mut stiffness = [0.0; 81];
            for point in points {
                let f_enh = self.enhanced_f(point, &alpha);
                let p = self.material.stress(&f_enh);
                let a = self.material.tangent(&f_enh);
                for (entry, mode) in residual.iter_mut().zip(point.g.iter()) {
                    *entry += point.weight * contract(&p, mode);
                }
                // D_kl = sum G_k : A : G_l over quadrature points.
                for k in 0..9 {
                    for l in 0..9 {
                        let mut d = 0.0;
                        for ij in 0..9 {
                            for kl in 0..9 {
                                d += point.g[k][ij] * a[ij * 9 + kl] * point.g[l][kl];
                            }
                        }
                        stiffness[k * 9 + l] += point.weight * d;
                    }
                }
            }
            let mut step = residual;
            for value in &mut step {
                *value = -*value;
            }
            let mut matrix = stiffness.to_vec();
            dense_solve(&mut matrix, &mut step, 9);
            let mut worst: f64 = 0.0;
            let mut scale: f64 = 1.0;
            for k in 0..9 {
                alpha[k] += step[k];
                worst = worst.max(step[k].abs());
                scale = scale.max(alpha[k].abs());
            }
            if worst < self.local_tolerance * scale {
                break;
            }
        }
        alpha
    }

    /// Internal forces for fixed enhanced parameters.
    fn forces_for_alpha(&self, points: &[QuadraturePoint; 8], alpha: &[f64; 9]) -> [f64; 24] {
        let mut residual = [0.0; 24];
        for point in points {
            let f_enh = self.enhanced_f(point, alpha);
            let p = self.material.stress(&f_enh);
            for a in 0..8 {
                for i in 0..3 {
                    let mut r = 0.0;
                    for col in 0..3 {
                        r += p[i * 3 + col] * point.grad[a][col];
                    }
                    residual[a * 3 + i] += r * point.weight;
                }
            }
        }
        residual
    }

    /// Internal force residual with locally equilibrated enhancement.
    fn internal_forces(&self, elem_idx: usize, u: &[f64]) -> [f64; 24] {
        let points = self.quadrature_data(elem_idx, u);
        let alpha = self.equilibrate(&points);
        self.forces_for_alpha(&points, &alpha)
    }
}

impl<M: HyperelasticMaterial> crate::forms::NonlinearElementForm<f64> for EasHexForm<M> {
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

    fn cube_form() -> EasHexForm<NeoHookean> {
        let material = NeoHookean::from_young(1.0e6, 0.3, 3);
        EasHexForm::new(material, HexMesh::unit_cube(1.0))
    }

    #[test]
    fn homogeneous_deformation_needs_no_enhancement() {
        let form = cube_form();
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let coords = form.mesh.coords[node];
            u[node * 3] = 0.02 * coords[0] + 0.01 * coords[1];
            u[node * 3 + 1] = -0.01 * coords[0] + 0.03 * coords[2];
            u[node * 3 + 2] = 0.01 * coords[1] - 0.02 * coords[2];
        }
        let points = form.quadrature_data(0, &u);
        let alpha = form.equilibrate(&points);
        for (k, &a) in alpha.iter().enumerate() {
            assert!(a.abs() < 1.0e-12, "alpha[{k}] = {a}");
        }
    }

    #[test]
    fn bending_relaxes_parasitic_shear() {
        let form = cube_form();
        // Pure-bending displacement: u_x varies quadratically through height.
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let [x, y, _] = form.mesh.coords[node];
            u[node * 3] = 0.05 * (x * x - y * y);
            u[node * 3 + 1] = -0.1 * x * y;
        }
        let eas = form.element_residual(0, &u);
        let base = crate::physics::solid::nonlinear_solid::q1_element_residual(
            &form.material,
            &form.mesh,
            0,
            &u,
        );
        let eas_norm: f64 = eas.iter().map(|r| r * r).sum::<f64>().sqrt();
        let base_norm: f64 = base.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(
            eas_norm < base_norm,
            "EAS {eas_norm} should relax Q1 {base_norm}"
        );
    }

    #[test]
    fn tangent_is_consistent_with_residual() {
        let form = cube_form();
        let state: Vec<f64> = (0..24)
            .map(|i| 0.008 * ((i * 37 % 11) as f64 - 5.0))
            .collect();
        let direction: Vec<f64> = (0..24)
            .map(|i| 0.04 * ((i * 53 % 7) as f64 - 3.0))
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
        .expect("eas residual and tangent dimensions must agree");
        assert!(report.observed_order > 1.8, "{report:?}");
    }
}
