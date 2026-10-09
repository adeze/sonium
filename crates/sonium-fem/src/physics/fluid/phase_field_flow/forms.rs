//! Linearized Cahn-Hilliard and capillary-force forms.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

#[derive(Debug, Clone, Copy)]
pub struct PhaseMassForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for PhaseMassForm {
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

#[derive(Debug, Clone, Copy)]
pub struct ChemicalPotentialDiffusionForm {
    pub mobility: f64,
}

impl BilinearForm<f64> for ChemicalPotentialDiffusionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.mobility;
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

/// Newton tangent of the chemical-potential definition with phase frozen at a
/// quadrature-point linearization value.
#[derive(Debug, Clone, Copy)]
pub struct ChemicalPotentialTangentForm {
    pub bulk_second_derivative: f64,
    pub gradient_energy_coefficient: f64,
}

impl BilinearForm<f64> for ChemicalPotentialTangentForm {
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
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                let value = self.bulk_second_derivative * test.value(i) * trial.value(j)
                    + self.gradient_energy_coefficient * grad_dot;
                local_matrix.add(i, j, value * w);
            }
        }
    }
}

pub struct PhaseAdvectionForm<U>
where
    U: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub velocity: U,
}

impl<U> BilinearForm<f64> for PhaseAdvectionForm<U>
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
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let directional_gradient = (0..test.spatial_dim)
                    .map(|d| velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * directional_gradient * w);
            }
        }
    }
}

/// Vector capillary force `-c grad(mu)` from recovered staggered fields.
pub struct CapillaryForceForm<C, G>
where
    C: Fn(f64, f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    pub phase: C,
    pub chemical_potential_gradient: G,
}

impl<C, G> LinearForm<f64> for CapillaryForceForm<C, G>
where
    C: Fn(f64, f64, f64) -> f64 + Send + Sync,
    G: Fn(f64, f64, f64) -> [f64; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let phase = (self.phase)(x, y, z);
        let gradient = (self.chemical_potential_gradient)(x, y, z);
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            let component = i % dim;
            local_vector.add(
                i,
                -phase * gradient[component] * test.vector_value(i, component) * w,
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseConservationReport {
    pub previous_mass: f64,
    pub current_mass: f64,
    pub relative_change: f64,
}

impl PhaseConservationReport {
    pub fn new(previous_mass: f64, current_mass: f64) -> Self {
        Self {
            previous_mass,
            current_mass,
            relative_change: (current_mass - previous_mass).abs()
                / previous_mass.abs().max(f64::MIN_POSITIVE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_phase_has_exact_mass_conservation() {
        assert_eq!(PhaseConservationReport::new(2.0, 2.0).relative_change, 0.0);
    }
}
