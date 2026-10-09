//! Electromagnetic blocks for inductionless MHD.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

use super::{MhdMaterial, cross};

/// Potential block `integral sigma grad(phi).grad(q)`.
#[derive(Debug, Clone, Copy)]
pub struct MhdPotentialForm {
    pub electrical_conductivity: f64,
}

impl BilinearForm<f64> for MhdPotentialForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.electrical_conductivity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

/// Velocity-to-potential motional-EMF block
/// `-integral sigma (u cross B).grad(q)` on the left-hand side.
#[derive(Debug, Clone, Copy)]
pub struct MotionalEmfCouplingForm {
    pub electrical_conductivity: f64,
    pub magnetic_flux_density: [f64; 3],
}

impl BilinearForm<f64> for MotionalEmfCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = -weight * test.det_j * self.electrical_conductivity;
        for i in 0..test.n_dofs {
            let grad_q = [
                test.grad(i, 0),
                if dim > 1 { test.grad(i, 1) } else { 0.0 },
                if dim > 2 { test.grad(i, 2) } else { 0.0 },
            ];
            for j in 0..trial.n_dofs {
                let component = j % dim;
                let mut basis_velocity = [0.0; 3];
                basis_velocity[component] = trial.vector_value(j, component);
                let emf = cross(basis_velocity, self.magnetic_flux_density);
                let projection = (0..dim).map(|d| emf[d] * grad_q[d]).sum::<f64>();
                local_matrix.add(i, j, projection * w);
            }
        }
    }
}

/// Positive-semidefinite magnetic damping
/// `integral sigma [B^2 u - (u dot B) B].v`.
#[derive(Debug, Clone, Copy)]
pub struct LorentzDampingForm {
    pub electrical_conductivity: f64,
    pub magnetic_flux_density: [f64; 3],
}

impl BilinearForm<f64> for LorentzDampingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let b2 = self
            .magnetic_flux_density
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let w = weight * test.det_j * self.electrical_conductivity;
        for i in 0..test.n_dofs {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..trial.n_dofs {
                let cj = j % dim;
                let phi_j = trial.vector_value(j, cj);
                let tensor = if ci == cj { b2 } else { 0.0 }
                    - self.magnetic_flux_density[ci] * self.magnetic_flux_density[cj];
                local_matrix.add(i, j, tensor * phi_i * phi_j * w);
            }
        }
    }
}

/// Potential-to-momentum block `integral sigma (grad(phi) cross B).v`.
#[derive(Debug, Clone, Copy)]
pub struct LorentzPotentialCouplingForm {
    pub electrical_conductivity: f64,
    pub magnetic_flux_density: [f64; 3],
}

impl BilinearForm<f64> for LorentzPotentialCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.electrical_conductivity;
        for i in 0..test.n_dofs {
            let component = i % dim;
            let phi = test.vector_value(i, component);
            for j in 0..trial.n_dofs {
                let grad_phi = [
                    trial.grad(j, 0),
                    if dim > 1 { trial.grad(j, 1) } else { 0.0 },
                    if dim > 2 { trial.grad(j, 2) } else { 0.0 },
                ];
                let force = cross(grad_phi, self.magnetic_flux_density);
                local_matrix.add(i, j, force[component] * phi * w);
            }
        }
    }
}

impl From<MhdMaterial> for MhdPotentialForm {
    fn from(material: MhdMaterial) -> Self {
        Self {
            electrical_conductivity: material.electrical_conductivity,
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
            derivatives: vec![0.0; 8],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        }
    }

    #[test]
    fn field_parallel_velocity_has_zero_lorentz_damping() {
        let tab = vector_tab();
        let mut matrix = LocalMatrix::zeros(4);
        LorentzDampingForm {
            electrical_conductivity: 2.0,
            magnetic_flux_density: [1.0, 0.0, 0.0],
        }
        .integrate_point(&tab, &tab, 1.0, &mut matrix);
        assert_eq!(matrix.get(0, 0), 0.0);
        assert!(matrix.get(1, 1) > 0.0);
    }

    #[test]
    fn zero_field_recovers_uncoupled_flow() {
        let tab = vector_tab();
        let mut matrix = LocalMatrix::zeros(4);
        LorentzDampingForm {
            electrical_conductivity: 2.0,
            magnetic_flux_density: [0.0; 3],
        }
        .integrate_point(&tab, &tab, 1.0, &mut matrix);
        assert!(matrix.values.iter().all(|value| *value == 0.0));
    }
}
