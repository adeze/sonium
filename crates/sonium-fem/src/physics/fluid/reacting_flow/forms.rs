//! Species transport, nonlinear reaction, and heat-release forms.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

#[derive(Debug, Clone, Copy)]
pub struct SpeciesDiffusionForm {
    pub density_diffusivity: f64,
}

impl BilinearForm<f64> for SpeciesDiffusionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.density_diffusivity;
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

pub struct SpeciesAdvectionForm<U>
where
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub density: f64,
    pub velocity: U,
}

impl<U> BilinearForm<f64> for SpeciesAdvectionForm<U>
where
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let velocity = (self.velocity)(x, y, z);
        let w = weight * test.det_j * self.density;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let directional = (0..test.spatial_dim)
                    .map(|d| velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * directional * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReactionJacobianForm {
    pub source_derivative: f64,
}

impl BilinearForm<f64> for ReactionJacobianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.source_derivative;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

pub struct RecoveredReactionSourceForm<S>
where
    S: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub source: S,
}

impl<S> LinearForm<f64> for RecoveredReactionSourceForm<S>
where
    S: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let source = (self.source)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, source * test.value(i) * w);
        }
    }
}

pub type ReactionHeatSourceForm<Q> = RecoveredReactionSourceForm<Q>;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesConservationReport {
    pub element_residuals: Vec<f64>,
    pub maximum_absolute_residual: f64,
}

impl SpeciesConservationReport {
    pub fn from_element_matrix(element_matrix: &[Vec<f64>], species_sources: &[f64]) -> Self {
        let element_count = element_matrix.first().map_or(0, Vec::len);
        assert!(element_matrix.iter().all(|row| row.len() == element_count));
        assert_eq!(element_matrix.len(), species_sources.len());
        let mut residuals = vec![0.0; element_count];
        for (species, source) in species_sources.iter().enumerate() {
            for (element, coefficient) in element_matrix[species].iter().enumerate() {
                residuals[element] += coefficient * source;
            }
        }
        let maximum = residuals
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        Self {
            element_residuals: residuals,
            maximum_absolute_residual: maximum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stoichiometric_element_balance_is_exact() {
        let report = SpeciesConservationReport::from_element_matrix(
            &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
            &[-1.0, -1.0, 1.0],
        );
        assert_eq!(report.element_residuals, vec![0.0, 0.0]);
    }
}
