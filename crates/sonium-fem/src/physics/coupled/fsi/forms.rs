//! Arbitrary-Lagrangian-Eulerian fluid forms.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

/// Vector ALE convection linearized about an advecting fluid velocity and mesh
/// velocity: `rho integral ((u_adv - u_mesh).grad u).v`.
#[derive(Debug, Clone, Copy)]
pub struct AleConvectionForm {
    pub density: f64,
    pub advecting_velocity: [f64; 3],
    pub mesh_velocity: [f64; 3],
}

impl AleConvectionForm {
    pub fn relative_velocity(&self) -> [f64; 3] {
        std::array::from_fn(|index| self.advecting_velocity[index] - self.mesh_velocity[index])
    }
}

impl BilinearForm<f64> for AleConvectionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let relative = self.relative_velocity();
        let w = weight * test.det_j * self.density;
        for i in 0..test.n_dofs {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..trial.n_dofs {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                let directional_gradient = (0..dim)
                    .map(|direction| relative[direction] * trial.grad(j, direction))
                    .sum::<f64>();
                local_matrix.add(i, j, phi_i * directional_gradient * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 4,
            spatial_dim: 2,
            values: vec![0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
            derivatives: vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        }
    }

    #[test]
    fn lagrangian_limit_has_zero_relative_convection() {
        let form = AleConvectionForm {
            density: 1000.0,
            advecting_velocity: [2.0, -1.0, 0.0],
            mesh_velocity: [2.0, -1.0, 0.0],
        };
        let tab = vector_tab();
        let mut matrix = LocalMatrix::zeros(4);
        form.integrate_point(&tab, &tab, 1.0, &mut matrix);
        assert!(matrix.values.iter().all(|value| *value == 0.0));
    }
}
