//! SUPG/PSPG stabilisation for equal-order velocity-pressure elements
//!
//! Standard P1-P1 (or equal-order) elements violate the inf-sup (LBB)
//! condition. PSPG adds pressure stability; SUPG adds convection stability.
//! Together they allow P1-P1 elements at any Reynolds number.
//!
//! # Stabilisation terms
//!
//! Added to the standard Galerkin weak form:
//!
//! **SUPG** (Streamline Upwind Petrov-Galerkin):
//! ```text
//! Σ_K τ_SUPG ∫_K (u·∇v) · R(u,p) dx
//! ```
//! where R = ρ(u·∇u) - μΔu + ∇p - f is the strong-form residual.
//!
//! **PSPG** (Pressure-Stabilising Petrov-Galerkin):
//! ```text
//! Σ_K τ_PSPG ∫_K (1/ρ ∇q) · R(u,p) dx
//! ```
//!
//! For P1 elements, Δu = 0, so the residual simplifies to:
//! ```text
//! R ≈ ρ(u·∇u) + ∇p - f
//! ```
//!
//! # Stabilisation parameter
//!
//! ```text
//! τ = (4/Δt² + 4|u|²/h² + C_inv·ν²/h⁴)^{-1/2}
//! ```
//!
//! For steady problems (Δt → ∞): τ = (4|u|²/h² + C_inv·ν²/h⁴)^{-1/2}
//!
//! `C_inv ≈ 36` is the inverse-estimate constant for P1 triangles. It plays
//! the role of the `9·C` factor in the Tezduyar/Osawa and Codina variants of
//! this formula (there `C ≈ 4` for linear elements, so `9·C = 36`): the `9`
//! is absorbed into `C_inv` rather than written out separately.
//!
//! # References
//!
//! - Tezduyar & Osawa (2000), "Finite element stabilization parameters
//!   computed from element matrices and vectors"
//! - Codina (2001), "A stabilized finite element method for generalized
//!   stationary incompressible flows"

use crate::forms::{
    BilinearForm, LinearForm, LocalMatrix, LocalVector, require_closure_components,
};
use crate::spaces::BasisTabulation;

/// Compute the SUPG/PSPG stabilisation parameter τ for an element.
///
/// τ = (4|u|²/h² + C_inv·ν²/h⁴)^{-1/2} (plus `4/Δt²` when `dt` is given),
/// where `C_inv` = 36 for P1 triangles (inverse estimate constant; equals
/// the `9·C` factor with `C ≈ 4` in the Tezduyar/Osawa and Codina variants).
pub fn stabilisation_tau(
    velocity_magnitude: f64,
    element_diameter: f64,
    kinematic_viscosity: f64,
    dt: Option<f64>,
) -> f64 {
    let h = element_diameter;
    let u = velocity_magnitude;
    let nu = kinematic_viscosity;
    let c_inv = 36.0; // inverse estimate for P1 triangles

    let mut inv_tau_sq = 4.0 * u * u / (h * h) + c_inv * nu * nu / (h * h * h * h);

    if let Some(dt_val) = dt.filter(|&v| v > 0.0) {
        inv_tau_sq += 4.0 / (dt_val * dt_val);
    }

    if inv_tau_sq > 1e-30 {
        1.0 / inv_tau_sq.sqrt()
    } else {
        0.0
    }
}

/// PSPG pressure stabilisation form.
///
/// Adds: `τ_PSPG ∫_K (1/ρ ∇q) · ∇p dx` to the pressure-pressure block.
///
/// This is the dominant stabilisation term for P1-P1 elements. It
/// stabilises the pressure by adding a Laplacian-like operator.
///
/// For equal-order elements: τ_PSPG = τ_SUPG.
pub struct PspgForm<F: Fn([f64; 3]) -> f64 + Send + Sync> {
    /// Density ρ.
    pub rho: f64,
    /// Kinematic viscosity ν = μ/ρ.
    pub nu: f64,
    /// Local velocity magnitude function (for computing τ).
    pub velocity_magnitude_fn: F,
    /// Time step (None for steady).
    pub dt: Option<f64>,
}

impl<F: Fn([f64; 3]) -> f64 + Send + Sync> BilinearForm<f64> for PspgForm<F> {
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

        // Estimate h from det_j: h ≈ det_j^{1/dim} * constant
        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt() // area ≈ det_j, h ≈ sqrt(2A)
        } else {
            (6.0 * test.det_j).cbrt() // vol ≈ det_j, h ≈ (6V)^{1/3}
        };

        let u_mag = (self.velocity_magnitude_fn)(test.physical_coords);
        let tau = stabilisation_tau(u_mag, h, self.nu, self.dt);

        // τ/(ρ) ∫ ∇q · ∇p dx (Laplacian-like pressure stabilisation)
        let coeff = tau / self.rho;
        for i in 0..n {
            for j in 0..n {
                let grad_dot: f64 = (0..dim).map(|d| test.grad(i, d) * trial.grad(j, d)).sum();
                local_matrix.add(i, j, coeff * grad_dot * w);
            }
        }
    }
}

/// SUPG coupling between a vector test function and pressure.
///
/// Adds `tau * integral (w . grad(v)) . grad(p)` from the pressure part of
/// the strong momentum residual.
///
/// `velocity_fn` must return at least `dim` components on a `dim`-dimensional
/// mesh; a short return panics instead of silently truncating the physics.
pub struct SupgPressureForm<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> {
    pub nu: f64,
    pub velocity_fn: F,
    pub dt: Option<f64>,
}

impl<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> BilinearForm<f64> for SupgPressureForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let velocity = (self.velocity_fn)(test.physical_coords);
        require_closure_components("SupgPressureForm velocity", &velocity, dim);
        let velocity_magnitude = velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };
        let tau = stabilisation_tau(velocity_magnitude, h, self.nu, self.dt);

        for i in 0..test.n_dofs {
            let component = i % dim;
            let streamline_test: f64 = (0..dim)
                .map(|direction| velocity[direction] * test.grad(i, direction))
                .sum();
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, tau * streamline_test * trial.grad(j, component) * w);
            }
        }
    }
}

/// PSPG coupling between a scalar pressure test function and velocity.
///
/// Adds `tau * integral grad(q) . ((w . grad)u)`.  The Navier-Stokes
/// monolithic assembler applies the global sign of its `-div(u)` row.
///
/// `velocity_fn` must return at least `dim` components on a `dim`-dimensional
/// mesh; a short return panics instead of silently truncating the physics.
pub struct PspgConvectionForm<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> {
    pub nu: f64,
    pub velocity_fn: F,
    pub dt: Option<f64>,
}

impl<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> BilinearForm<f64> for PspgConvectionForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let velocity = (self.velocity_fn)(test.physical_coords);
        require_closure_components("PspgConvectionForm velocity", &velocity, dim);
        let velocity_magnitude = velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };
        let tau = stabilisation_tau(velocity_magnitude, h, self.nu, self.dt);

        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let component = j % dim;
                let streamline_trial: f64 = (0..dim)
                    .map(|direction| velocity[direction] * trial.grad(j, direction))
                    .sum();
                local_matrix.add(i, j, tau * test.grad(i, component) * streamline_trial * w);
            }
        }
    }
}

/// SUPG body-force contribution to the velocity right-hand side.
///
/// Both closures must return at least `dim` components on a `dim`-dimensional
/// mesh; a short return panics instead of silently truncating the physics.
pub struct SupgBodyForceForm<
    V: Fn([f64; 3]) -> Vec<f64> + Send + Sync,
    F: Fn([f64; 3]) -> Vec<f64> + Send + Sync,
> {
    pub nu: f64,
    pub velocity_fn: V,
    pub force_fn: F,
    pub dt: Option<f64>,
}

impl<V: Fn([f64; 3]) -> Vec<f64> + Send + Sync, F: Fn([f64; 3]) -> Vec<f64> + Send + Sync>
    LinearForm<f64> for SupgBodyForceForm<V, F>
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let velocity = (self.velocity_fn)(test.physical_coords);
        let force = (self.force_fn)(test.physical_coords);
        require_closure_components("SupgBodyForceForm velocity", &velocity, dim);
        require_closure_components("SupgBodyForceForm force", &force, dim);
        let velocity_magnitude = velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };
        let tau = stabilisation_tau(velocity_magnitude, h, self.nu, self.dt);

        for i in 0..test.n_dofs {
            let component = i % dim;
            let streamline_test: f64 = (0..dim)
                .map(|direction| velocity[direction] * test.grad(i, direction))
                .sum();
            local_vector.add(i, tau * streamline_test * force[component] * w);
        }
    }
}

/// PSPG body-force contribution to the pressure right-hand side.
///
/// `force_fn` must return at least `dim` components on a `dim`-dimensional
/// mesh; a short return panics instead of silently truncating the physics.
pub struct PspgBodyForceForm<
    V: Fn([f64; 3]) -> f64 + Send + Sync,
    F: Fn([f64; 3]) -> Vec<f64> + Send + Sync,
> {
    pub rho: f64,
    pub nu: f64,
    pub velocity_magnitude_fn: V,
    pub force_fn: F,
    pub dt: Option<f64>,
}

impl<V: Fn([f64; 3]) -> f64 + Send + Sync, F: Fn([f64; 3]) -> Vec<f64> + Send + Sync>
    LinearForm<f64> for PspgBodyForceForm<V, F>
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let velocity_magnitude = (self.velocity_magnitude_fn)(test.physical_coords);
        let force = (self.force_fn)(test.physical_coords);
        require_closure_components("PspgBodyForceForm force", &force, dim);
        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };
        let tau = stabilisation_tau(velocity_magnitude, h, self.nu, self.dt);

        for i in 0..test.n_dofs {
            let grad_dot_force: f64 = (0..dim)
                .map(|direction| test.grad(i, direction) * force[direction])
                .sum();
            local_vector.add(i, tau / self.rho * grad_dot_force * w);
        }
    }
}

/// SUPG convection stabilisation form for the velocity block.
///
/// Adds: `τ_SUPG ∫_K (u·∇v) · (u·∇u) dx`
///
/// This stabilises the convection operator for convection-dominated flows.
/// The test function is modified: v → v + τ(u·∇v) (Petrov-Galerkin weighting).
///
/// `velocity_fn` must return at least `dim` components on a `dim`-dimensional
/// mesh; a short return panics instead of silently truncating the physics.
pub struct SupgConvectionForm<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> {
    /// Density ρ.
    pub rho: f64,
    /// Kinematic viscosity ν.
    pub nu: f64,
    /// Advection velocity w(x) from previous iteration.
    pub velocity_fn: F,
    /// Time step (None for steady).
    pub dt: Option<f64>,
}

impl<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> BilinearForm<f64> for SupgConvectionForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.rho;

        let vel = (self.velocity_fn)(test.physical_coords);
        require_closure_components("SupgConvectionForm velocity", &vel, dim);
        let u_mag = vel.iter().map(|v| v * v).sum::<f64>().sqrt();

        let h = if dim == 2 {
            (2.0 * test.det_j).sqrt()
        } else {
            (6.0 * test.det_j).cbrt()
        };

        let tau = stabilisation_tau(u_mag, h, self.nu, self.dt);

        // τ ρ ∫ (u·∇v_i)(u·∇u_j) dx
        // For VectorH1, DOF i has component c_i = i % dim.
        // Only same-component pairs contribute.
        for i in 0..n {
            let ci = i % dim;
            // u·∇v_i = Σ_d u_d ∂φ_i/∂x_d (for the c_i component)
            let u_dot_grad_test: f64 = (0..dim).map(|d| vel[d] * test.grad(i, d)).sum();

            for j in 0..n {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                let u_dot_grad_trial: f64 = (0..dim).map(|d| vel[d] * trial.grad(j, d)).sum();

                local_matrix.add(i, j, tau * u_dot_grad_test * u_dot_grad_trial * w);
            }
        }
    }
}

/// LSIC (Least-Squares on Incompressibility Constraint) stabilisation.
///
/// Adds: `τ_LSIC ∫_K div(v) · div(u) dx`
///
/// This improves mass conservation for high Reynolds numbers.
/// τ_LSIC = τ · |u| · h / 2 (or similar scaling).
pub struct LsicForm<F: Fn([f64; 3]) -> f64 + Send + Sync> {
    pub rho: f64,
    pub nu: f64,
    pub velocity_magnitude_fn: F,
    pub dt: Option<f64>,
}

impl<F: Fn([f64; 3]) -> f64 + Send + Sync> BilinearForm<f64> for LsicForm<F> {
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
        let tau = stabilisation_tau(u_mag, h, self.nu, self.dt);
        let tau_lsic = tau * u_mag * h * 0.5;

        // τ_LSIC ∫ div(v) · div(u) dx
        // For VectorH1: div(v_i) = ∂φ_{node}/∂x_{ci}
        for i in 0..n {
            let ci = i % dim;
            let div_vi = test.grad(i, ci);
            for j in 0..n {
                let cj = j % dim;
                let div_uj = trial.grad(j, cj);
                local_matrix.add(i, j, self.rho * tau_lsic * div_vi * div_uj * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_diffusion_dominated() {
        // Low Re (large ν, small u): τ ≈ h²/(sqrt(C_inv)·ν) = h²/(6ν)
        let tau = stabilisation_tau(0.0, 0.1, 1.0, None);
        assert!(tau > 0.0);
        assert!(tau < 1.0, "τ should be small for diffusion-dominated flow");
    }

    #[test]
    fn test_tau_matches_hand_computed_value() {
        // τ = (4u²/h² + C_inv·ν²/h⁴)^{-1/2} with C_inv = 36.
        // u = 2, h = 0.5, ν = 0.1:
        //   4·4/0.25 + 36·0.01/0.0625 = 64 + 5.76 = 69.76 → τ = 1/sqrt(69.76)
        let tau = stabilisation_tau(2.0, 0.5, 0.1, None);
        let expected = 1.0 / 69.76_f64.sqrt();
        assert!(
            (tau - expected).abs() < 1e-15,
            "τ = {tau}, expected {expected}"
        );
        // With Δt: inv_tau_sq gains 4/Δt².
        let tau_dt = stabilisation_tau(2.0, 0.5, 0.1, Some(0.5));
        let expected_dt = 1.0 / (69.76_f64 + 4.0 / 0.25).sqrt();
        assert!(
            (tau_dt - expected_dt).abs() < 1e-15,
            "τ(dt) = {tau_dt}, expected {expected_dt}"
        );
    }

    #[test]
    fn test_tau_convection_dominated() {
        // High Re (small ν, large u): τ ≈ h/(2|u|)
        let tau = stabilisation_tau(10.0, 0.1, 1e-6, None);
        let tau_expected = 0.1 / (2.0 * 10.0); // h/(2u) = 0.005
        assert!(
            (tau - tau_expected).abs() / tau_expected < 0.1,
            "τ = {tau}, expected ≈ {tau_expected}"
        );
    }

    #[test]
    fn test_tau_decreases_with_dt() {
        let tau_steady = stabilisation_tau(1.0, 0.1, 0.01, None);
        let tau_unsteady = stabilisation_tau(1.0, 0.1, 0.01, Some(0.001));
        assert!(tau_unsteady < tau_steady, "Unsteady τ should be smaller");
    }

    #[test]
    #[should_panic(expected = "SupgConvectionForm velocity closure must return at least")]
    fn test_supg_convection_short_velocity_panics() {
        // A closure returning fewer components than the mesh dimension must
        // fail loudly instead of silently truncating the streamline product.
        use crate::basis::PolynomialDegree;
        use crate::mesh::unit_square_triangles;
        use crate::spaces::{FunctionSpace, H1Space};
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = SupgConvectionForm {
            rho: 1.0,
            nu: 0.01,
            velocity_fn: |_| vec![1.0],
            dt: None,
        };
        let mut local = LocalMatrix::zeros(tab.n_dofs);
        form.integrate_point(&tab, &tab, 1.0, &mut local);
    }
}
