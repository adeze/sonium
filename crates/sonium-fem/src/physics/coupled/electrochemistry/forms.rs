//! Poisson-Nernst-Planck weak forms.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

use super::materials::{ButlerVolmerKinetics, FARADAY, IonicSpecies};

/// Nernst-Planck diffusion: `integral D grad c.grad v`.
#[derive(Debug, Clone, Copy)]
pub struct NernstPlanckDiffusionForm {
    pub diffusivity: f64,
}

impl BilinearForm<f64> for NernstPlanckDiffusionForm {
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

/// Linearized electro-migration: `integral z mu c0 grad(phi).grad(v)`.
///
/// [`IonicSpecies::from_diffusivity`] stores the Nernst-Einstein mobility
/// `D F / (R T)`, so the Faraday factor is already included in `mu`.
#[derive(Debug, Clone, Copy)]
pub struct ElectroMigrationForm {
    pub species: IonicSpecies,
    pub reference_concentration: f64,
}

impl BilinearForm<f64> for ElectroMigrationForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let coeff = self.species.valence * self.species.mobility * self.reference_concentration;
        let w = weight * test.det_j * coeff;
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

/// Ionic charge source for Poisson equation: `rho = F sum z_i c_i`.
#[derive(Debug, Clone, Copy)]
pub struct IonicChargeSourceForm {
    pub valence: f64,
    pub concentration: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ButlerVolmerBoundaryFluxForm {
    pub kinetics: ButlerVolmerKinetics,
    pub overpotential: f64,
    pub stoichiometric_coefficient: f64,
}

impl LinearForm<f64> for ButlerVolmerBoundaryFluxForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let molar_flux = self.stoichiometric_coefficient
            * self.kinetics.current_density(self.overpotential)
            / FARADAY;
        let w = weight * test.det_j * molar_flux;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ButlerVolmerTangentForm {
    pub kinetics: ButlerVolmerKinetics,
    pub overpotential: f64,
}

impl BilinearForm<f64> for ButlerVolmerTangentForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let scale = FARADAY / (super::materials::GAS_CONSTANT * self.kinetics.temperature);
        let eta = self.overpotential;
        let tangent = self.kinetics.exchange_current_density
            * scale
            * (self.kinetics.alpha_anodic * (self.kinetics.alpha_anodic * scale * eta).exp()
                + self.kinetics.alpha_cathodic
                    * (-self.kinetics.alpha_cathodic * scale * eta).exp());
        let w = weight * test.det_j * tangent;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

impl LinearForm<f64> for IonicChargeSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let rho = FARADAY * self.valence * self.concentration;
        let w = weight * test.det_j * rho;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::coupled::electrochemistry::{ButlerVolmerKinetics, GAS_CONSTANT};

    #[test]
    fn nernst_einstein_mobility_matches_formula() {
        let species = IonicSpecies::from_diffusivity(1.0e-9, 1.0, 298.15);
        let expected = 1.0e-9 * FARADAY / (GAS_CONSTANT * 298.15);
        assert!((species.mobility - expected).abs() < 1.0e-18);
    }

    #[test]
    fn butler_volmer_is_antisymmetric_for_symmetric_transfer() {
        let bv = ButlerVolmerKinetics {
            exchange_current_density: 2.0,
            alpha_anodic: 0.5,
            alpha_cathodic: 0.5,
            temperature: 300.0,
        };
        assert!((bv.current_density(0.01) + bv.current_density(-0.01)).abs() < 1.0e-12);
        assert!(bv.linearized_charge_transfer_resistance() > 0.0);
    }

    #[test]
    fn ionic_charge_source_scales_with_valence() {
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 2,
            values: vec![0.5],
            derivatives: vec![0.0; 2],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut rhs = LocalVector::zeros(1);
        IonicChargeSourceForm {
            valence: -2.0,
            concentration: 3.0,
        }
        .integrate_point(&tab, 0.25, &mut rhs);
        assert!((rhs.values[0] + 1.5 * FARADAY).abs() < 1.0e-6);
    }

    #[test]
    fn electro_migration_uses_nernst_einstein_mobility_once() {
        let species = IonicSpecies::from_diffusivity(2.0e-9, 2.0, 300.0);
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 2.0, 3.0, 4.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        ElectroMigrationForm {
            species,
            reference_concentration: 4.0,
        }
        .integrate_point(&tab, &tab, 0.5, &mut local);

        let expected = species.valence * species.mobility * 4.0 * (1.0 * 3.0 + 2.0 * 4.0);
        assert!((local.get(0, 1) - expected).abs() < 1.0e-18);
    }

    #[test]
    fn butler_volmer_flux_and_tangent_are_positive_for_anodic_bias() {
        let bv = ButlerVolmerKinetics {
            exchange_current_density: 2.0,
            alpha_anodic: 0.5,
            alpha_cathodic: 0.5,
            temperature: 300.0,
        };
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 1,
            values: vec![1.0],
            derivatives: vec![0.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let mut rhs = LocalVector::zeros(1);
        ButlerVolmerBoundaryFluxForm {
            kinetics: bv,
            overpotential: 0.01,
            stoichiometric_coefficient: 1.0,
        }
        .integrate_point(&tab, 1.0, &mut rhs);
        let mut tangent = LocalMatrix::zeros(1);
        ButlerVolmerTangentForm {
            kinetics: bv,
            overpotential: 0.01,
        }
        .integrate_point(&tab, &tab, 1.0, &mut tangent);
        assert!(rhs.values[0] > 0.0);
        assert!(tangent.get(0, 0) > 0.0);
    }
}
