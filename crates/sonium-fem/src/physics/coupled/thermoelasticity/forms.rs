//! Weak forms for small-strain thermoelasticity.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::materials::ThermoelasticMaterial;

/// Coupling from temperature increment to mechanical virtual work:
/// `l(v) = integral sigma_th : eps(v)`.
#[derive(Debug, Clone, Copy)]
pub struct ThermalStrainLoadForm {
    pub material: ThermoelasticMaterial,
    pub temperature: f64,
}

impl ThermalStrainLoadForm {
    fn thermal_stress_trace(&self, dim: usize) -> f64 {
        let delta_t = self.temperature - self.material.reference_temperature;
        let lambda = self.material.lame_lambda();
        let mu = self.material.shear_modulus();
        (dim as f64 * lambda + 2.0 * mu) * self.material.thermal_expansion * delta_t
    }
}

impl LinearForm<f64> for ThermalStrainLoadForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.thermal_stress_trace(dim);
        let n_nodes = test.n_dofs / dim;
        for node in 0..n_nodes {
            for c in 0..dim {
                let dof = node * dim + c;
                local_vector.add(dof, test.grad(dof, c) * w);
            }
        }
    }
}

/// Temperature mass-like coupling for staggered thermoelastic damping models.
#[derive(Debug, Clone, Copy)]
pub struct ThermoelasticDampingForm {
    pub coupling_coefficient: f64,
}

impl BilinearForm<f64> for ThermoelasticDampingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.coupling_coefficient;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_thermal_stress_matches_bulk_formula() {
        let material = ThermoelasticMaterial {
            young_modulus: 210.0e9,
            poisson_ratio: 0.3,
            thermal_expansion: 12.0e-6,
            reference_temperature: 293.0,
        };
        let stress = material.constrained_thermal_stress_3d(393.0);
        let bulk = material.lame_lambda() + 2.0 * material.shear_modulus() / 3.0;
        assert!((stress - 3.0 * bulk * 12.0e-4).abs() < 1.0e-3);
    }

    #[test]
    fn thermal_load_uses_divergence_of_virtual_displacement() {
        let material = ThermoelasticMaterial {
            young_modulus: 10.0,
            poisson_ratio: 0.25,
            thermal_expansion: 0.01,
            reference_temperature: 20.0,
        };
        let tab = BasisTabulation {
            n_dofs: 4,
            spatial_dim: 2,
            values: vec![0.0; 8],
            derivatives: vec![1.0, 9.0, 8.0, 2.0, -1.0, 7.0, 6.0, -2.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut rhs = LocalVector::zeros(4);
        ThermalStrainLoadForm {
            material,
            temperature: 30.0,
        }
        .integrate_point(&tab, 0.5, &mut rhs);

        let trace = (2.0 * material.lame_lambda() + 2.0 * material.shear_modulus())
            * material.thermal_expansion
            * 10.0;
        assert!((rhs.values[0] - trace).abs() < 1.0e-12);
        assert!((rhs.values[1] - 2.0 * trace).abs() < 1.0e-12);
    }

    #[test]
    fn damping_form_is_mass_like() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.2, 0.8],
            derivatives: vec![0.0; 4],
            hessians: vec![],
            det_j: 3.0,
            physical_coords: [0.0; 3],
        };
        let mut m = LocalMatrix::zeros(2);
        ThermoelasticDampingForm {
            coupling_coefficient: 5.0,
        }
        .integrate_point(&tab, &tab, 2.0, &mut m);
        assert!((m.get(0, 1) - 0.2 * 0.8 * 30.0).abs() < 1.0e-12);
    }
}
