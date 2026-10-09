//! Transport weak forms.

use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::physics::fluid::stokes::stabilisation::stabilisation_tau;
use crate::spaces::BasisTabulation;

use super::materials::LinearReactionNetwork;
use super::materials::TransportMaterial;

#[derive(Debug, Clone, Copy)]
pub struct TransportDiffusionForm {
    pub diffusivity: [[f64; 3]; 3],
}

impl TransportDiffusionForm {
    pub fn new(material: &TransportMaterial) -> Self {
        Self {
            diffusivity: material.diffusivity,
        }
    }
}

impl BilinearForm<f64> for TransportDiffusionForm {
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
                let mut value = 0.0;
                for a in 0..test.spatial_dim {
                    for b in 0..test.spatial_dim {
                        value += test.grad(i, a) * self.diffusivity[a][b] * trial.grad(j, b);
                    }
                }
                local_matrix.add(i, j, value * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransportAdvectionForm {
    pub velocity: [f64; 3],
}

impl TransportAdvectionForm {
    pub fn new(material: &TransportMaterial) -> Self {
        Self {
            velocity: material.velocity,
        }
    }
}

impl BilinearForm<f64> for TransportAdvectionForm {
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
                let adv = (0..test.spatial_dim)
                    .map(|d| self.velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                local_matrix.add(i, j, test.value(i) * adv * w);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransportReactionForm {
    pub reaction_rate: f64,
}

impl TransportReactionForm {
    pub fn new(material: &TransportMaterial) -> Self {
        Self {
            reaction_rate: material.reaction_rate,
        }
    }
}

impl BilinearForm<f64> for TransportReactionForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.reaction_rate;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Transient storage/mass form: `m(c, v) = integral capacity c v`.
#[derive(Debug, Clone, Copy)]
pub struct TransportCapacityForm {
    pub capacity: f64,
}

impl TransportCapacityForm {
    pub fn new(material: &TransportMaterial) -> Self {
        Self {
            capacity: material.capacity,
        }
    }
}

impl BilinearForm<f64> for TransportCapacityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let w = weight * test.det_j * self.capacity;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// SUPG stabilization for scalar ADR:
/// `tau (u.grad v) (u.grad c + reaction*c)`.
#[derive(Debug, Clone, Copy)]
pub struct TransportSupgForm {
    pub material: TransportMaterial,
    pub element_length_scale: Option<f64>,
    pub dt: Option<f64>,
}

impl TransportSupgForm {
    pub fn new(material: TransportMaterial) -> Self {
        Self {
            material,
            element_length_scale: None,
            dt: None,
        }
    }
}

impl BilinearForm<f64> for TransportSupgForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let speed = self
            .material
            .velocity
            .iter()
            .take(dim)
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let h = self.element_length_scale.unwrap_or_else(|| {
            if dim == 1 {
                test.det_j.abs()
            } else if dim == 2 {
                (2.0 * test.det_j.abs()).sqrt()
            } else {
                (6.0 * test.det_j.abs()).cbrt()
            }
        });
        let diffusivity = self.material.diffusivity[0][0].abs().max(f64::EPSILON);
        let tau = stabilisation_tau(speed, h.max(f64::EPSILON), diffusivity, self.dt);
        let w = weight * test.det_j * tau;

        for i in 0..test.n_dofs {
            let u_dot_grad_test = (0..dim)
                .map(|d| self.material.velocity[d] * test.grad(i, d))
                .sum::<f64>();
            for j in 0..trial.n_dofs {
                let u_dot_grad_trial = (0..dim)
                    .map(|d| self.material.velocity[d] * trial.grad(j, d))
                    .sum::<f64>();
                let residual_trial =
                    u_dot_grad_trial + self.material.reaction_rate * trial.value(j);
                local_matrix.add(i, j, u_dot_grad_test * residual_trial * w);
            }
        }
    }
}

/// Mass-like Jacobian block for a linear reaction network species pair.
#[derive(Debug)]
pub struct TransportReactionJacobianForm<'a> {
    pub network: &'a LinearReactionNetwork,
    pub target_species: usize,
    pub source_species: usize,
}

impl<'a> BilinearForm<f64> for TransportReactionJacobianForm<'a> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let coeff = self.network.jacobian[self.target_species][self.source_species];
        let w = weight * test.det_j * coeff;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Coupled reaction source from a linear reaction network: `r_i = K_i(c) + s_i`.
#[derive(Debug)]
pub struct TransportReactionNetworkSourceForm<'a> {
    pub network: &'a LinearReactionNetwork,
    pub species_index: usize,
    pub concentrations: &'a [f64],
}

impl<'a> LinearForm<f64> for TransportReactionNetworkSourceForm<'a> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let rates = self.network.evaluate(self.concentrations);
        let source = rates[self.species_index];
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * source * w);
        }
    }
}

pub struct TransportSourceForm<F: Fn(f64, f64, f64) -> f64 + Send + Sync> {
    pub source_fn: F,
}

impl<F: Fn(f64, f64, f64) -> f64 + Send + Sync> LinearForm<f64> for TransportSourceForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let source = (self.source_fn)(x, y, z);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * source * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fluid::transport::{LinearReactionNetwork, TransportMaterial};

    fn tab() -> BasisTabulation {
        BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.4, 0.6],
            derivatives: vec![2.0, -1.0, 0.5, 3.0],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [2.0, 3.0, 0.0],
        }
    }

    #[test]
    fn diffusion_advection_reaction_forms_match_manual_values() {
        let material = TransportMaterial::isotropic(2.0, [3.0, -1.0, 0.0], 5.0);
        let t = tab();

        let mut k = LocalMatrix::zeros(2);
        TransportDiffusionForm::new(&material).integrate_point(&t, &t, 4.0, &mut k);
        let expected_diff = 4.0 * 0.5 * 2.0 * (2.0 * 0.5 + -1.0 * 3.0);
        assert!((k.get(0, 1) - expected_diff).abs() < 1.0e-12);

        let mut a = LocalMatrix::zeros(2);
        TransportAdvectionForm::new(&material).integrate_point(&t, &t, 4.0, &mut a);
        let expected_adv = 4.0 * 0.5 * 0.4 * (3.0 * 0.5 + -1.0 * 3.0);
        assert!((a.get(0, 1) - expected_adv).abs() < 1.0e-12);

        let mut r = LocalMatrix::zeros(2);
        TransportReactionForm::new(&material).integrate_point(&t, &t, 4.0, &mut r);
        assert!((r.get(0, 1) - 4.0 * 0.5 * 5.0 * 0.4 * 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn reaction_network_evaluates_linear_jacobian() {
        let network = LinearReactionNetwork {
            jacobian: vec![vec![-2.0, 1.0], vec![2.0, -1.0]],
            source: vec![0.5, 0.0],
        };
        let r = network.evaluate(&[3.0, 4.0]);
        assert_eq!(r, vec![-1.5, 2.0]);
    }

    #[test]
    fn peclet_number_uses_speed_and_diffusivity() {
        let material = TransportMaterial::isotropic(0.5, [3.0, 4.0, 0.0], 0.0);
        assert!((material.peclet(2.0) - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn capacity_form_is_mass_like() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![0.0; 4],
            hessians: vec![],
            det_j: 3.0,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut local = LocalMatrix::zeros(2);
        TransportCapacityForm { capacity: 1.5 }.integrate_point(&tab, &tab, 2.0, &mut local);
        assert!((local.get(0, 1) - 0.25 * 0.75 * 2.0 * 3.0 * 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn reaction_network_source_uses_current_concentrations() {
        let network = LinearReactionNetwork {
            jacobian: vec![vec![-1.0, 2.0], vec![1.0, -3.0]],
            source: vec![0.5, -0.5],
        };
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 2,
            values: vec![1.0],
            derivatives: vec![0.0; 2],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut rhs = LocalVector::zeros(1);
        TransportReactionNetworkSourceForm {
            network: &network,
            species_index: 0,
            concentrations: &[1.0, 2.0],
        }
        .integrate_point(&tab, 1.0, &mut rhs);
        let expected = 1.0 * 2.0 * 3.5;
        assert!((rhs.values[0] - expected).abs() < 1.0e-12);
    }

    #[test]
    fn supg_form_adds_streamline_residual_weighting() {
        let material = TransportMaterial::isotropic(0.1, [2.0, 0.0, 0.0], 0.0);
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 1,
            values: vec![0.25, 0.75],
            derivatives: vec![1.0, -1.0],
            hessians: vec![],
            det_j: 0.5,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut local = LocalMatrix::zeros(2);
        TransportSupgForm {
            material,
            element_length_scale: Some(1.0),
            dt: None,
        }
        .integrate_point(&tab, &tab, 2.0, &mut local);
        assert!(local.get(0, 0) > 0.0);
        assert!(local.get(0, 1) < 0.0);
    }

    #[test]
    fn reaction_jacobian_form_uses_species_block_coefficient() {
        let network = LinearReactionNetwork {
            jacobian: vec![vec![-2.0, 1.5], vec![2.0, -1.0]],
            source: vec![0.0, 0.0],
        };
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 1,
            values: vec![0.25],
            derivatives: vec![0.0],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0, 0.0, 0.0],
        };
        let mut local = LocalMatrix::zeros(1);
        TransportReactionJacobianForm {
            network: &network,
            target_species: 0,
            source_species: 1,
        }
        .integrate_point(&tab, &tab, 2.0, &mut local);
        assert!((local.get(0, 0) - 1.5 * 0.25 * 0.25 * 4.0).abs() < 1.0e-12);
    }
}
