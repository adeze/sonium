//! Coupled weak forms for electric potential and temperature.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::ThermoelectricMaterial;

fn gradient_dot(test: &BasisTabulation, trial: &BasisTabulation, i: usize, j: usize) -> f64 {
    (0..test.spatial_dim)
        .map(|d| test.grad(i, d) * trial.grad(j, d))
        .sum()
}

/// Potential-potential block `integral sigma grad(V).grad(w)`.
#[derive(Debug, Clone, Copy)]
pub struct ElectricalConductionForm {
    pub electrical_conductivity: f64,
}

impl BilinearForm<f64> for ElectricalConductionForm {
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
                local_matrix.add(i, j, gradient_dot(test, trial, i, j) * w);
            }
        }
    }
}

/// Temperature-dependent potential block for staggered electrothermal solves.
pub struct TemperatureDependentElectricalForm<S, T>
where
    S: Fn(f64) -> f64 + Send + Sync,
    T: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub conductivity: S,
    pub temperature: T,
}

impl<S, T> BilinearForm<f64> for TemperatureDependentElectricalForm<S, T>
where
    S: Fn(f64) -> f64 + Send + Sync,
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
        let sigma = (self.conductivity)((self.temperature)(x, y, z));
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "conductivity must stay positive"
        );
        let w = weight * test.det_j * sigma;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, gradient_dot(test, trial, i, j) * w);
            }
        }
    }
}

/// Temperature-to-current Seebeck block
/// `integral sigma alpha grad(T).grad(w)`.
#[derive(Debug, Clone, Copy)]
pub struct SeebeckCurrentCouplingForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for SeebeckCurrentCouplingForm {
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
                local_matrix.add(i, j, gradient_dot(test, trial, i, j) * w);
            }
        }
    }
}

/// Potential-to-heat Peltier block
/// `integral Pi sigma grad(V).grad(q)`.
#[derive(Debug, Clone, Copy)]
pub struct PeltierPotentialCouplingForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for PeltierPotentialCouplingForm {
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
                local_matrix.add(i, j, gradient_dot(test, trial, i, j) * w);
            }
        }
    }
}

/// Temperature-temperature heat block including Fourier conduction and the
/// `Pi sigma alpha` contribution induced by eliminating current density.
#[derive(Debug, Clone, Copy)]
pub struct ThermoelectricHeatForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for ThermoelectricHeatForm {
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
                local_matrix.add(i, j, gradient_dot(test, trial, i, j) * w);
            }
        }
    }
}

/// Joule heat from a recovered electric field, `sigma |E|^2`.
pub struct JouleHeatSourceForm<E, S>
where
    E: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
    S: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub electric_field: E,
    pub electrical_conductivity: S,
}

impl<E, S> LinearForm<f64> for JouleHeatSourceForm<E, S>
where
    E: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
    S: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let field = (self.electric_field)(x, y, z);
        let sigma = (self.electrical_conductivity)(x, y, z);
        let heat = sigma
            * field
                .iter()
                .take(test.spatial_dim)
                .map(|component| component * component)
                .sum::<f64>();
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, heat * test.value(i) * w);
        }
    }
}

/// Thomson volumetric heat `-tau J.grad(T)` for a recovered staggered state.
pub struct ThomsonHeatSourceForm<J, G>
where
    J: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub thomson_coefficient: f64,
    pub current_density: J,
    pub temperature_gradient: G,
}

impl<J, G> LinearForm<f64> for ThomsonHeatSourceForm<J, G>
where
    J: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let current = (self.current_density)(x, y, z);
        let gradient = (self.temperature_gradient)(x, y, z);
        let source = -self.thomson_coefficient
            * (0..test.spatial_dim)
                .map(|d| current[d] * gradient[d])
                .sum::<f64>();
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, source * test.value(i) * w);
        }
    }
}

impl From<(ThermoelectricMaterial, f64)> for ElectricalConductionForm {
    fn from((material, _temperature): (ThermoelectricMaterial, f64)) -> Self {
        Self {
            electrical_conductivity: material.electrical_conductivity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.4, 0.6],
            derivatives: vec![1.0, 2.0, -1.0, 0.5],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [0.0; 3],
        }
    }

    #[test]
    fn zero_seebeck_decouples_charge_and_heat() {
        let tab = tab();
        let mut cross = LocalMatrix::zeros(2);
        SeebeckCurrentCouplingForm { coefficient: 0.0 }
            .integrate_point(&tab, &tab, 2.0, &mut cross);
        assert!(cross.values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn joule_source_is_nonnegative_and_conservative() {
        let tab = tab();
        let form = JouleHeatSourceForm {
            electric_field: |_, _, _| [3.0, 4.0, 0.0],
            electrical_conductivity: |_, _, _| 2.0,
        };
        let mut rhs = LocalVector::zeros(2);
        form.integrate_point(&tab, 2.0, &mut rhs);
        assert_eq!(rhs.values, vec![20.0, 30.0]);
        assert_eq!(rhs.values.iter().sum::<f64>(), 50.0);
    }
}
