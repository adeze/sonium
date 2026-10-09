//! DFN/P2D weak-form components.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

/// Spherical-particle storage `integral 4 pi r^2 c v dr`.
#[derive(Debug, Clone, Copy)]
pub struct SphericalParticleMassForm;

impl BilinearForm<f64> for SphericalParticleMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0].abs();
        let w = 4.0 * std::f64::consts::PI * radius * radius * weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Spherical-particle diffusion `integral 4 pi r^2 D grad(c).grad(v) dr`.
#[derive(Debug, Clone, Copy)]
pub struct SphericalParticleDiffusionForm {
    pub solid_diffusivity: f64,
}

impl BilinearForm<f64> for SphericalParticleDiffusionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let radius = test.physical_coords[0].abs();
        let w = 4.0
            * std::f64::consts::PI
            * radius
            * radius
            * self.solid_diffusivity
            * weight
            * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.grad(i, 0) * trial.grad(j, 0) * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BatteryScalarDiffusionForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for BatteryScalarDiffusionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.coefficient;
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

#[derive(Debug, Clone, Copy)]
pub struct BatteryCapacityForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for BatteryCapacityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Linearized interfacial reaction coupling `integral a_s dj/deta u v`.
#[derive(Debug, Clone, Copy)]
pub struct InterfacialReactionJacobianForm {
    pub specific_surface_area: f64,
    pub current_slope: f64,
}

impl BilinearForm<f64> for InterfacialReactionJacobianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.specific_surface_area * self.current_slope;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

pub struct BatteryHeatSourceForm<Q>
where
    Q: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub heat_density: Q,
}

impl<Q> LinearForm<f64> for BatteryHeatSourceForm<Q>
where
    Q: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let heat = (self.heat_density)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, heat * test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_forms_vanish_at_center_without_singularity() {
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 1,
            values: vec![1.0],
            derivatives: vec![2.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut mass = LocalMatrix::zeros(1);
        SphericalParticleMassForm.integrate_point(&tab, &tab, 1.0, &mut mass);
        assert_eq!(mass.get(0, 0), 0.0);
        let mut diffusion = LocalMatrix::zeros(1);
        SphericalParticleDiffusionForm {
            solid_diffusivity: 1.0,
        }
        .integrate_point(&tab, &tab, 1.0, &mut diffusion);
        assert_eq!(diffusion.get(0, 0), 0.0);
    }
}
