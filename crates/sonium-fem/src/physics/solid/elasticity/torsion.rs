//! Saint-Venant torsion formulations for prismatic bars.
//!
//! These helpers provide scalar Poisson formulations for cross-section torsion
//! alongside the displacement-based elasticity forms:
//!
//! - Prandtl stress function `Phi(y,z)`:
//!   `-Δ Phi = 2 G theta`, with `Phi = 0` on the boundary.
//! - Warping function `psi(y,z)`:
//!   `Δ psi = 0`, with problem-specific Neumann data on the boundary.
//!
//! The forms are intentionally scalar `H1` building blocks, so callers can use
//! the generic assembler and boundary APIs already present in this crate.

use crate::forms::{BilinearForm, LaplacianForm, LinearForm, LocalMatrix, LocalVector, SourceForm};
use crate::spaces::BasisTabulation;

/// Scalar Poisson stiffness for a torsion cross-section.
#[derive(Debug, Clone, Copy)]
pub struct TorsionPoissonForm(LaplacianForm);

impl Default for TorsionPoissonForm {
    fn default() -> Self {
        Self(LaplacianForm)
    }
}

impl TorsionPoissonForm {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BilinearForm<f64> for TorsionPoissonForm {
    #[inline]
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        self.0.integrate_point(test, trial, weight, local_matrix);
    }
}

/// RHS for the Prandtl stress-function equation.
///
/// For twist rate `theta` and shear modulus `G`, the weak form is
/// `∫ grad(Phi)·grad(v) dA = ∫ 2G theta v dA`.
pub type PrandtlStressRhsForm = SourceForm<fn(f64, f64, f64) -> f64>;

/// Build the constant RHS form for `-Δ Phi = 2G theta`.
pub fn prandtl_stress_rhs(shear_modulus: f64, twist_rate: f64) -> ConstantSourceForm {
    ConstantSourceForm {
        value: 2.0 * shear_modulus * twist_rate,
    }
}

/// Constant scalar source form.
#[derive(Debug, Clone, Copy)]
pub struct ConstantSourceForm {
    pub value: f64,
}

impl LinearForm<f64> for ConstantSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * self.value * w);
        }
    }
}

/// Recover shear stress components from a Prandtl stress-function gradient.
///
/// For a bar axis `x` and cross-section coordinates `(y, z)`:
///
/// ```text
/// tau_xy =  dPhi/dz
/// tau_xz = -dPhi/dy
/// ```
pub fn prandtl_shear_stress(grad_phi_yz: [f64; 2]) -> [f64; 2] {
    [grad_phi_yz[1], -grad_phi_yz[0]]
}

/// Boundary flux for the warping-function Neumann formulation.
///
/// With cross-section coordinates `(y, z)` and outward normal `(n_y, n_z)`,
/// a common Saint-Venant convention is:
///
/// ```text
/// dpsi/dn = theta * (z n_y - y n_z)
/// ```
pub fn warping_neumann_flux(y: f64, z: f64, normal_y: f64, normal_z: f64, twist_rate: f64) -> f64 {
    twist_rate * (z * normal_y - y * normal_z)
}

/// Torque recovered from a solved Prandtl stress function.
///
/// `T = 2 ∫_A Phi dA`. The caller supplies the already integrated
/// cross-section integral of `Phi`.
pub fn prandtl_torque(phi_integral: f64) -> f64 {
    2.0 * phi_integral
}

/// Torsion constant `J = T / (G theta)`.
pub fn torsion_constant_from_prandtl(
    phi_integral: f64,
    shear_modulus: f64,
    twist_rate: f64,
) -> f64 {
    prandtl_torque(phi_integral) / (shear_modulus * twist_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space};

    #[test]
    fn test_prandtl_stress_rhs_constant() {
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = prandtl_stress_rhs(5.0, 0.2);
        let mut local = LocalVector::zeros(tab.n_dofs);

        form.integrate_point(&tab, 1.0, &mut local);

        assert!(local.values.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_prandtl_shear_stress_rotation() {
        assert_eq!(prandtl_shear_stress([2.0, 3.0]), [3.0, -2.0]);
    }

    #[test]
    fn test_torsion_poisson_form_produces_stiffness() {
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let form = TorsionPoissonForm::new();
        let mut local = LocalMatrix::zeros(tab.n_dofs);

        form.integrate_point(&tab, &tab, 1.0, &mut local);

        let diag_sum: f64 = (0..tab.n_dofs).map(|i| local.get(i, i)).sum();
        assert!(diag_sum > 0.0);
    }
}
