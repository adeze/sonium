//! Linearized compressible-flow forms.

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

/// Acoustic compressibility mass form for linearized compressible flow:
/// `integral (1 / rho c^2) p v`.
#[derive(Debug, Clone, Copy)]
pub struct CompressibilityMassForm {
    pub density: f64,
    pub sound_speed: f64,
}

impl BilinearForm<f64> for CompressibilityMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let coeff = 1.0 / (self.density * self.sound_speed * self.sound_speed);
        let w = weight * test.det_j * coeff;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * w);
            }
        }
    }
}

/// Viscous/thermal diffusion proxy for linearized compressible Navier-Stokes.
#[derive(Debug, Clone, Copy)]
pub struct CompressibleDiffusionForm {
    pub diffusivity: f64,
}

impl BilinearForm<f64> for CompressibleDiffusionForm {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fluid::compressible_flow::{ConservativeState, IdealGas};

    #[test]
    fn euler_flux_matches_normal_momentum_flux() {
        let gas = IdealGas::AIR;
        let state = ConservativeState {
            density: 2.0,
            momentum: [6.0, 0.0, 0.0],
            total_energy: 30.0,
        };
        let p = state.pressure(gas);
        let flux = state.euler_flux(gas, [1.0, 0.0, 0.0]);
        assert!((p - (0.4 * (30.0 - 9.0))).abs() < 1.0e-12);
        assert!((flux[0] - 6.0).abs() < 1.0e-12);
        assert!((flux[1] - (18.0 + p)).abs() < 1.0e-12);
    }

    #[test]
    fn ideal_gas_sound_speed_and_temperature() {
        let gas = IdealGas::AIR;
        assert!((gas.temperature(101_325.0, 1.225) - 288.15).abs() < 0.2);
        assert!((gas.speed_of_sound(101_325.0, 1.225) - 340.3).abs() < 0.5);
    }

    #[test]
    fn compressibility_mass_uses_inverse_bulk_modulus() {
        let tab = BasisTabulation {
            n_dofs: 2,
            spatial_dim: 2,
            values: vec![0.25, 0.75],
            derivatives: vec![0.0; 4],
            hessians: vec![],
            det_j: 2.0,
            physical_coords: [0.0; 3],
        };
        let mut m = LocalMatrix::zeros(2);
        CompressibilityMassForm {
            density: 2.0,
            sound_speed: 5.0,
        }
        .integrate_point(&tab, &tab, 0.5, &mut m);
        assert!((m.get(0, 1) - 0.25 * 0.75 / 50.0).abs() < 1.0e-12);
    }
}
