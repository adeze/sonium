//! GLS/SUPG stabilisation for aeroacoustics equations
//!
//! The Linearized Euler and Linearized Navier-Stokes equations require
//! stabilisation when using standard Galerkin finite elements, especially
//! for convection-dominated regimes (high Mach number or high frequency).
//!
//! # Stabilisation strategy
//!
//! **GLS (Galerkin Least Squares)**: Adds a consistent residual-based
//! stabilisation term to the weak form. The test function is modified to
//! include a streamline-upwind component:
//!
//! ```text
//! v_stab = v + tau (U_0 . nabla v)
//! ```
//!
//! This is equivalent to SUPG for the convection operator and provides
//! both convection and pressure stabilisation.
//!
//! # Stabilisation parameter
//!
//! For the convected wave equation, the stabilisation parameter tau
//! balances convective and wave-propagation time scales:
//!
//! ```text
//! tau = (4 omega^2 + 4 |U_0|^2 / h^2 + C_inv * nu^2 / h^4)^{-1/2}
//! ```
//!
//! For inviscid flows (LEE), the viscous term is dropped:
//!
//! ```text
//! tau = (4 omega^2 + 4 |U_0|^2 / h^2)^{-1/2}
//! ```

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

/// Compute the GLS stabilisation parameter for convected acoustics.
///
/// Balances the time scale (omega), convection scale (|U|/h), and
/// optionally the diffusion scale (nu/h^2).
///
/// Returns tau such that the stabilisation term is O(h) consistent.
pub fn aeroacoustic_tau(
    omega: f64,
    velocity_magnitude: f64,
    element_diameter: f64,
    kinematic_viscosity: Option<f64>,
) -> f64 {
    let h = element_diameter;
    let u = velocity_magnitude;

    // Time + convection scales
    let mut inv_tau_sq = 4.0 * omega * omega + 4.0 * u * u / (h * h);

    // Add viscous scale if present (LNS)
    if let Some(nu) = kinematic_viscosity {
        let c_inv = 36.0; // inverse estimate constant for P1
        inv_tau_sq += c_inv * nu * nu / (h * h * h * h);
    }

    if inv_tau_sq > 1e-30 {
        1.0 / inv_tau_sq.sqrt()
    } else {
        0.0
    }
}

/// SUPG stabilisation form for scalar convected wave equation (LPF).
///
/// Adds: `tau integral((U_0 . nabla psi)(U_0 . nabla phi)) dOmega`
///
/// This stabilises the convection operator in the convected Helmholtz
/// equation by adding streamline diffusion along the flow direction.
pub struct ScalarSupgForm<F: Fn([f64; 3]) -> f64 + Send + Sync> {
    /// Mean flow velocity (advection direction).
    pub advection: [f64; 3],
    /// Angular frequency omega (for tau computation).
    pub omega: f64,
    /// Optional kinematic viscosity (None for inviscid/LEE).
    pub kinematic_viscosity: Option<f64>,
    /// Velocity magnitude function (for spatially varying flows).
    pub velocity_magnitude_fn: F,
}

impl<F: Fn([f64; 3]) -> f64 + Send + Sync> BilinearForm<f64> for ScalarSupgForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };

        let u_mag = (self.velocity_magnitude_fn)(test.physical_coords);
        let tau = aeroacoustic_tau(self.omega, u_mag, h, self.kinematic_viscosity);

        // tau integral((U . nabla psi)(U . nabla phi)) dOmega
        for i in 0..n {
            let u_dot_grad_test: f64 = (0..dim).map(|d| self.advection[d] * test.grad(i, d)).sum();
            for j in 0..n {
                let u_dot_grad_trial: f64 =
                    (0..dim).map(|d| self.advection[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, tau * u_dot_grad_test * u_dot_grad_trial * w);
            }
        }
    }
}

/// SUPG stabilisation form for vector convection (LEE/LNS momentum).
///
/// Adds: `tau integral((U_0 . nabla v_i)(U_0 . nabla u_j)) dOmega`
///
/// Component-gated: only same-component DOF pairs contribute.
pub struct VectorSupgForm<F: Fn([f64; 3]) -> f64 + Send + Sync> {
    /// Mean flow velocity.
    pub advection: [f64; 3],
    /// Angular frequency omega.
    pub omega: f64,
    /// Optional kinematic viscosity.
    pub kinematic_viscosity: Option<f64>,
    /// Velocity magnitude function.
    pub velocity_magnitude_fn: F,
}

impl<F: Fn([f64; 3]) -> f64 + Send + Sync> BilinearForm<f64> for VectorSupgForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };

        let u_mag = (self.velocity_magnitude_fn)(test.physical_coords);
        let tau = aeroacoustic_tau(self.omega, u_mag, h, self.kinematic_viscosity);

        for i in 0..n {
            let ci = i % dim;
            let u_dot_grad_test: f64 = (0..dim).map(|d| self.advection[d] * test.grad(i, d)).sum();
            for j in 0..n {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                let u_dot_grad_trial: f64 =
                    (0..dim).map(|d| self.advection[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, tau * u_dot_grad_test * u_dot_grad_trial * w);
            }
        }
    }
}

/// Pressure stabilisation form for LEE (analogous to PSPG in Stokes).
///
/// Adds: `tau / rho_0 integral(nabla q . nabla p) dOmega`
///
/// Prevents pressure oscillations in equal-order velocity-pressure
/// elements when solving the linearized Euler equations.
pub struct PressureStabilisationForm<F: Fn([f64; 3]) -> f64 + Send + Sync> {
    /// Mean density rho_0.
    pub rho: f64,
    /// Mean flow velocity magnitude.
    pub advection: [f64; 3],
    /// Angular frequency omega.
    pub omega: f64,
    /// Velocity magnitude function.
    pub velocity_magnitude_fn: F,
}

impl<F: Fn([f64; 3]) -> f64 + Send + Sync> BilinearForm<f64> for PressureStabilisationForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };

        let u_mag = (self.velocity_magnitude_fn)(test.physical_coords);
        let tau = aeroacoustic_tau(self.omega, u_mag, h, None);
        let coeff = tau / self.rho;

        for i in 0..n {
            for j in 0..n {
                let grad_dot: f64 = (0..dim).map(|d| test.grad(i, d) * trial.grad(j, d)).sum();
                local_matrix.add(i, j, coeff * grad_dot * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_inviscid() {
        // For inviscid flow, tau depends on omega and velocity only
        let tau = aeroacoustic_tau(1000.0, 100.0, 0.1, None);
        assert!(tau > 0.0);
        assert!(tau < 1.0);
    }

    #[test]
    fn test_tau_viscous() {
        // Adding viscosity should reduce tau
        let tau_inv = aeroacoustic_tau(1000.0, 100.0, 0.1, None);
        let tau_vis = aeroacoustic_tau(1000.0, 100.0, 0.1, Some(1.5e-5));
        assert!(tau_vis <= tau_inv);
    }

    #[test]
    fn test_tau_high_frequency() {
        // Higher frequency should reduce tau
        let tau_low = aeroacoustic_tau(100.0, 50.0, 0.1, None);
        let tau_high = aeroacoustic_tau(10000.0, 50.0, 0.1, None);
        assert!(tau_high < tau_low);
    }

    #[test]
    fn test_tau_zero_flow() {
        // At zero flow, tau is determined by omega alone
        let tau = aeroacoustic_tau(1000.0, 0.0, 0.1, None);
        let expected = 1.0 / (2.0 * 1000.0); // 1/sqrt(4*omega^2) = 1/(2*omega)
        assert!((tau - expected).abs() / expected < 1e-10);
    }
}
