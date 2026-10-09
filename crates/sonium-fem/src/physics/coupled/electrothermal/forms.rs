//! Electric-current and electromagnetic-loss transfer forms.

use num_complex::Complex64;

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::ElectrothermalMaterial;

/// DC current-continuity form with temperature feedback.
pub struct DcCurrentForm<T>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub material: ElectrothermalMaterial,
    pub temperature: T,
}

impl<T> BilinearForm<f64> for DcCurrentForm<T>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let sigma = self
            .material
            .electrical_conductivity((self.temperature)(x, y, z));
        let w = weight * test.det_j * sigma;
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

/// DC Joule heat from a recovered potential gradient.
pub struct DcJouleHeatForm<T, G>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub material: ElectrothermalMaterial,
    pub temperature: T,
    /// Recovered `grad(V)`.
    pub potential_gradient: G,
}

impl<T, G> LinearForm<f64> for DcJouleHeatForm<T, G>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let gradient = (self.potential_gradient)(x, y, z);
        let sigma = self
            .material
            .electrical_conductivity((self.temperature)(x, y, z));
        let heat = sigma
            * gradient
                .iter()
                .take(test.spatial_dim)
                .map(|value| value * value)
                .sum::<f64>();
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, heat * test.value(i) * w);
        }
    }
}

/// Time-averaged harmonic Joule heat from a complex electric field.
pub struct HarmonicJouleHeatForm<T, E>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    E: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync,
{
    pub material: ElectrothermalMaterial,
    pub temperature: T,
    pub electric_field: E,
}

impl<T, E> LinearForm<f64> for HarmonicJouleHeatForm<T, E>
where
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
    E: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let field = (self.electric_field)(x, y, z);
        let sigma = self
            .material
            .electrical_conductivity((self.temperature)(x, y, z));
        let heat = 0.5
            * sigma
            * field
                .iter()
                .take(test.spatial_dim)
                .map(Complex64::norm_sqr)
                .sum::<f64>();
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, heat * test.value(i) * w);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectrothermalEnergyBalance {
    pub electrical_input_power: f64,
    pub integrated_joule_heat: f64,
    pub relative_imbalance: f64,
}

impl ElectrothermalEnergyBalance {
    pub fn new(electrical_input_power: f64, integrated_joule_heat: f64) -> Self {
        let scale = electrical_input_power
            .abs()
            .max(integrated_joule_heat.abs())
            .max(f64::MIN_POSITIVE);
        Self {
            electrical_input_power,
            integrated_joule_heat,
            relative_imbalance: (electrical_input_power - integrated_joule_heat).abs() / scale,
        }
    }

    pub fn is_conservative(&self, tolerance: f64) -> bool {
        tolerance >= 0.0 && tolerance.is_finite() && self.relative_imbalance <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material() -> ElectrothermalMaterial {
        ElectrothermalMaterial {
            reference_electrical_conductivity: 4.0,
            resistivity_temperature_coefficient: 0.01,
            reference_temperature: 300.0,
            relative_permeability: 1.0,
            thermal_conductivity: 10.0,
            density: 1000.0,
            specific_heat: 500.0,
        }
    }

    fn tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 0.0, -1.0, 0.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        }
    }

    #[test]
    fn harmonic_heat_uses_rms_half_factor_and_partitions_power() {
        let form = HarmonicJouleHeatForm {
            material: material(),
            temperature: |_, _, _| 300.0,
            electric_field: |_, _, _| [Complex64::new(3.0, 4.0), Complex64::ZERO, Complex64::ZERO],
        };
        let mut rhs = LocalVector::zeros(2);
        form.integrate_point(&tab(), 0.5, &mut rhs);
        assert_eq!(rhs.values.iter().sum::<f64>(), 50.0);
        assert!(ElectrothermalEnergyBalance::new(50.0, 50.0).is_conservative(1e-14));
    }

    #[test]
    fn hotter_metal_has_lower_conductivity_for_positive_resistivity_slope() {
        assert!(
            material().electrical_conductivity(400.0) < material().electrical_conductivity(300.0)
        );
    }
}
