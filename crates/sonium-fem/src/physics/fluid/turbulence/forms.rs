//! Linearized turbulence-transport weak forms.

use super::{KOmegaSstConstants, SpalartAllmarasConstants};
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;

/// Generic scalar turbulence transport:
/// `(nu + nu_t/sigma) grad phi . grad v + (u . grad phi) v + reaction phi v`.
#[derive(Debug, Clone, Copy)]
pub struct TurbulenceTransportForm {
    pub molecular_viscosity: f64,
    pub eddy_viscosity: f64,
    pub sigma: f64,
    pub velocity: [f64; 3],
    pub reaction: f64,
}

impl BilinearForm<f64> for TurbulenceTransportForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let diffusivity = self.molecular_viscosity + self.eddy_viscosity / self.sigma;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                let adv = (0..test.spatial_dim)
                    .map(|d| self.velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                let mass = test.value(i) * trial.value(j);
                local_matrix.add(
                    i,
                    j,
                    (diffusivity * grad_dot + test.value(i) * adv + self.reaction * mass) * w,
                );
            }
        }
    }
}

/// Turbulence production source `integral P v`.
#[derive(Debug, Clone, Copy)]
pub struct TurbulenceProductionForm {
    pub production: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SpalartAllmarasSourceForm {
    pub constants: SpalartAllmarasConstants,
    pub modified_eddy_viscosity: f64,
    pub vorticity_magnitude: f64,
    pub wall_distance: f64,
}

impl LinearForm<f64> for SpalartAllmarasSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let source = self
            .constants
            .production(self.modified_eddy_viscosity, self.vorticity_magnitude)
            - self
                .constants
                .destruction(self.modified_eddy_viscosity, self.wall_distance);
        let w = weight * test.det_j * source;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KOmegaReactionForm {
    pub constants: KOmegaSstConstants,
    pub omega: f64,
    pub equation: KOmegaEquation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KOmegaEquation {
    K,
    Omega,
}

impl BilinearForm<f64> for KOmegaReactionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let reaction = match self.equation {
            KOmegaEquation::K => self.constants.k_reaction(self.omega),
            KOmegaEquation::Omega => self.constants.omega_reaction(self.omega),
        };
        let w = weight * test.det_j * reaction;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

impl LinearForm<f64> for TurbulenceProductionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j * self.production;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fluid::turbulence::{SpalartAllmarasConstants, eddy_viscosity_k_omega};

    fn tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.5, 0.5],
            derivatives: vec![1.0, 0.0, 0.0, 2.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        }
    }

    #[test]
    fn transport_form_combines_diffusion_advection_and_reaction() {
        let t = tab();
        let form = TurbulenceTransportForm {
            molecular_viscosity: 1.0,
            eddy_viscosity: 4.0,
            sigma: 2.0,
            velocity: [3.0, 5.0, 0.0],
            reaction: 7.0,
        };
        let mut local = LocalMatrix::zeros(2);
        form.integrate_point(&t, &t, 0.5, &mut local);
        let expected = 3.0 * 0.0 + 0.5 * (3.0 * 0.0 + 5.0 * 2.0) + 7.0 * 0.25;
        assert!((local.get(0, 1) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn production_form_scales_by_shape_function() {
        let t = tab();
        let mut rhs = LocalVector::zeros(2);
        TurbulenceProductionForm { production: 8.0 }.integrate_point(&t, 0.25, &mut rhs);
        assert!((rhs.values[0] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn model_constants_and_eddy_viscosity_are_positive() {
        let constants = SpalartAllmarasConstants::default();
        assert!(constants.cb1 > 0.0 && constants.cw1 > 0.0);
        assert!((eddy_viscosity_k_omega(2.0, 4.0, 10.0) - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn sa_and_komega_closures_have_expected_signs() {
        let t = tab();
        let mut rhs = LocalVector::zeros(2);
        SpalartAllmarasSourceForm {
            constants: SpalartAllmarasConstants::default(),
            modified_eddy_viscosity: 0.01,
            vorticity_magnitude: 100.0,
            wall_distance: 1.0,
        }
        .integrate_point(&t, 0.5, &mut rhs);
        assert!(rhs.values[0] > 0.0);

        let mut local = LocalMatrix::zeros(2);
        KOmegaReactionForm {
            constants: KOmegaSstConstants::default(),
            omega: 10.0,
            equation: KOmegaEquation::K,
        }
        .integrate_point(&t, &t, 0.5, &mut local);
        assert!(local.get(0, 0) > 0.0);
    }
}
