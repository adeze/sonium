//! Drift-diffusion weak forms for carrier density.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::materials::{CarrierMaterial, DopingProfile};

#[derive(Debug, Clone, Copy)]
pub struct CarrierDiffusionForm {
    pub diffusivity: f64,
}

impl CarrierDiffusionForm {
    pub fn new(material: &CarrierMaterial) -> Self {
        Self {
            diffusivity: material.diffusivity(),
        }
    }
}

impl BilinearForm<f64> for CarrierDiffusionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.diffusivity;
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
pub struct CarrierDriftForm {
    pub electric_field: [f64; 3],
    pub drift_coefficient: f64,
}

impl BilinearForm<f64> for CarrierDriftForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.drift_coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let e_dot_grad = (0..test.spatial_dim)
                    .map(|d| self.electric_field[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * e_dot_grad * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CarrierRecombinationForm {
    pub lifetime: f64,
}

impl BilinearForm<f64> for CarrierRecombinationForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j / self.lifetime;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DopingChargeForm {
    pub profile: DopingProfile,
}

impl LinearForm<f64> for DopingChargeForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j * self.profile.net_charge_density();
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::electromagnetics::semiconductor::{CarrierKind, CarrierMaterial};

    #[test]
    fn einstein_relation_and_charge_signs_are_correct() {
        let electron = CarrierMaterial {
            mobility: 0.14,
            lifetime: 1.0e-6,
            temperature: 300.0,
            kind: CarrierKind::Electron,
        };
        assert!((electron.diffusivity() - 0.003_619).abs() < 1.0e-5);
        assert!(electron.drift_coefficient() < 0.0);
    }

    #[test]
    fn drift_form_uses_electric_field_dot_gradient() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 0.0, 2.0, 3.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        CarrierDriftForm {
            electric_field: [5.0, 7.0, 0.0],
            drift_coefficient: -2.0,
        }
        .integrate_point(&tab, &tab, 0.5, &mut local);
        assert!((local.get(0, 1) - 0.25 * (5.0 * 2.0 + 7.0 * 3.0) * -2.0).abs() < 1.0e-12);
    }

    #[test]
    fn doping_profile_reports_net_charge_density() {
        let profile = DopingProfile {
            donors: 3.0,
            acceptors: 1.0,
        };
        assert!(profile.net_charge_density() > 0.0);
    }
}
