//! Weak forms for scalar phase-field fracture damage.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use crate::spaces::BasisTabulation;

use super::materials::PhaseFieldFractureMaterial;

/// Damage regularization:
/// `integral Gc*l grad d.grad eta + Gc/l d eta`.
#[derive(Debug, Clone, Copy)]
pub struct PhaseFieldDamageForm {
    pub material: PhaseFieldFractureMaterial,
}

/// Elasticity block degraded by scalar phase-field damage.
#[derive(Debug, Clone, Copy)]
pub struct DegradedElasticityForm {
    pub elastic_material: IsotropicMaterial,
    pub fracture_material: PhaseFieldFractureMaterial,
    pub damage: f64,
    pub plane: Option<PlaneAssumption>,
}

impl BilinearForm<f64> for DegradedElasticityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let degradation = self.fracture_material.degradation(self.damage);
        let degraded = IsotropicMaterial::new(
            self.elastic_material.youngs_modulus * degradation,
            self.elastic_material.poisson_ratio,
        );
        let form = match self.plane {
            Some(plane) => ElasticityForm::new_2d(degraded, plane),
            None => ElasticityForm::new_3d(degraded),
        };
        form.integrate_point(test, trial, weight, local_matrix);
    }
}

impl BilinearForm<f64> for PhaseFieldDamageForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let gc = self.material.critical_energy_release_rate;
        let ell = self.material.length_scale;
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let grad_dot = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(
                    i,
                    j,
                    (gc * ell * grad_dot + gc / ell * test.value(i) * trial.value(j)) * w,
                );
            }
        }
    }
}

/// History-field driving term `integral 2 H d eta`.
#[derive(Debug, Clone, Copy)]
pub struct DamageHistoryRhsForm {
    pub history_energy_density: f64,
}

/// Local AT2 damage tangent coefficient from the tensile history field.
pub fn damage_history_tangent(
    material: PhaseFieldFractureMaterial,
    history_energy_density: f64,
) -> f64 {
    material.critical_energy_release_rate / material.length_scale + 2.0 * history_energy_density
}

/// History-dependent mass term `integral coefficient * d * eta`.
///
/// The staggered AT2 damage operator is the base regularization plus
/// this form with `coefficient = 2 * H` (see [`damage_history_tangent`]).
/// Dropping it linearizes the update (`d = 2Hl/Gc`, unbounded); keeping
/// it recovers the bounded AT2 law `d = 2Hl/Gc / (1 + 2Hl/Gc)`.
#[derive(Debug, Clone, Copy)]
pub struct DamageHistoryTangentForm {
    pub coefficient: f64,
}

impl BilinearForm<f64> for DamageHistoryTangentForm {
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

impl LinearForm<f64> for DamageHistoryRhsForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j * 2.0 * self.history_energy_density;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degradation_is_bounded_and_monotone() {
        let mat = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 2.0,
            length_scale: 0.1,
            residual_stiffness: 1.0e-6,
        };
        assert!(mat.degradation(0.0) > mat.degradation(0.5));
        assert!(mat.degradation(0.5) > mat.degradation(1.0));
        assert!(mat.degradation(1.0) >= 1.0e-6);
    }

    #[test]
    fn damage_form_matches_gradient_and_mass_terms() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 2.0, -1.0, 3.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let material = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 4.0,
            length_scale: 0.5,
            residual_stiffness: 0.0,
        };
        let mut local = LocalMatrix::zeros(2);
        PhaseFieldDamageForm { material }.integrate_point(&tab, &tab, 0.5, &mut local);
        let expected = 4.0 * 0.5 * (1.0 * -1.0 + 2.0 * 3.0) + 4.0 / 0.5 * 0.25 * 0.75;
        assert!((local.get(0, 1) - expected).abs() < 1.0e-12);
    }

    #[test]
    fn history_tangent_form_scales_p1_mass() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, 2.0, -1.0, 3.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut local = LocalMatrix::zeros(2);
        DamageHistoryTangentForm { coefficient: 3.0 }.integrate_point(&tab, &tab, 0.5, &mut local);
        // weight * det_j * c * N0 * N1 = 0.5 * 2 * 3 * 0.25 * 0.75.
        assert!((local.get(0, 1) - 0.5625).abs() < 1.0e-12);
        assert!((local.get(1, 0) - local.get(0, 1)).abs() < 1.0e-15);
        assert!(local.get(0, 0) > 0.0 && local.get(1, 1) > 0.0);
    }

    #[test]
    fn crack_surface_density_matches_at2_model() {
        let mat = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 2.0,
            length_scale: 0.25,
            residual_stiffness: 0.0,
        };
        assert!((mat.crack_surface_density(0.5, 4.0) - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn degraded_elasticity_scales_linear_stiffness() {
        let fracture = PhaseFieldFractureMaterial {
            critical_energy_release_rate: 1.0,
            length_scale: 0.1,
            residual_stiffness: 0.01,
        };
        let tab = BasisTabulation {
            n_dofs: 4,
            spatial_dim: 2,
            values: vec![0.0; 4],
            derivatives: vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            hessians: vec![],
            det_j: 1.0,
            physical_coords: [0.0; 3],
        };
        let elastic = IsotropicMaterial::new(10.0, 0.25);
        let mut intact = LocalMatrix::zeros(4);
        DegradedElasticityForm {
            elastic_material: elastic,
            fracture_material: fracture,
            damage: 0.0,
            plane: Some(PlaneAssumption::PlaneStress),
        }
        .integrate_point(&tab, &tab, 1.0, &mut intact);
        let mut damaged = LocalMatrix::zeros(4);
        DegradedElasticityForm {
            damage: 0.5,
            ..DegradedElasticityForm {
                elastic_material: elastic,
                fracture_material: fracture,
                damage: 0.0,
                plane: Some(PlaneAssumption::PlaneStress),
            }
        }
        .integrate_point(&tab, &tab, 1.0, &mut damaged);
        assert!(damaged.get(0, 0) < intact.get(0, 0));
        assert!(damage_history_tangent(fracture, 2.0) > 0.0);
    }
}
