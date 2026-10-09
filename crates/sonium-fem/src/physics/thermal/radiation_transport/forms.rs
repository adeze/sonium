//! Discrete-ordinates streaming, extinction, scattering, and emission forms.

use std::f64::consts::PI;

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::RadiativeMaterial;

/// One-ordinate transport block
/// `integral v (s.grad I + beta I) dV`.
#[derive(Debug, Clone, Copy)]
pub struct StreamingExtinctionForm {
    pub direction: [f64; 3],
    pub extinction_coefficient: f64,
}

impl BilinearForm<f64> for StreamingExtinctionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let streaming = (0..test.spatial_dim)
                    .map(|d| self.direction[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(
                    i,
                    j,
                    test.value(i) * (streaming + self.extinction_coefficient * trial.value(j)) * w,
                );
            }
        }
    }
}

/// Angular scattering block moved to the left-hand side.
#[derive(Debug, Clone, Copy)]
pub struct ScatteringCouplingForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for ScatteringCouplingForm {
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

pub struct ThermalEmissionForm<T>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub material: RadiativeMaterial,
    pub temperature: T,
}

impl<T> LinearForm<f64> for ThermalEmissionForm<T>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let emission = self.material.absorption_coefficient
            * self
                .material
                .blackbody_radiance((self.temperature)(x, y, z));
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, emission * test.value(i) * w);
        }
    }
}

/// Net material heating `kappa_a (G - 4 pi B)`.
pub fn radiative_material_heating(
    material: RadiativeMaterial,
    incident_radiation: f64,
    temperature: f64,
) -> f64 {
    material.absorption_coefficient
        * (incident_radiation - 4.0 * PI * material.blackbody_radiance(temperature))
}

pub fn is_inflow(direction: [f64; 3], outward_normal: [f64; 3]) -> bool {
    direction
        .iter()
        .zip(outward_normal)
        .map(|(s, n)| s * n)
        .sum::<f64>()
        < 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflow_sign_uses_outward_normal() {
        assert!(is_inflow([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]));
        assert!(!is_inflow([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    }

    #[test]
    fn radiative_equilibrium_has_zero_material_heating() {
        let material = RadiativeMaterial {
            absorption_coefficient: 2.0,
            scattering_coefficient: 1.0,
            asymmetry_parameter: 0.0,
            refractive_index: 1.0,
            density: 1.0,
            specific_heat: 1.0,
        };
        let temperature = 1000.0;
        let equilibrium = 4.0 * PI * material.blackbody_radiance(temperature);
        assert_eq!(
            radiative_material_heating(material, equilibrium, temperature),
            0.0
        );
    }
}
