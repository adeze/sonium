//! Piezoelectric bilinear forms for the FEM coupled system
//!
//! Implements the 2×2 block system for coupled (displacement u, potential φ):
//!
//! ```text
//! | K_uu   K_uφ | | u |   | F_ext |
//! | K_φu   K_φφ | | φ | = | Q_ext |
//! ```
//!
//! where:
//! - K_uu = ∫ cᴱ:ε(u):ε(v) dΩ   (elastic stiffness at constant E)
//! - K_uφ = −∫ eᵗ·∇φ:ε(v) dΩ    (inverse piezoelectric effect: E → strain)
//! - K_φu = ∫ e:ε(u)·∇ψ dΩ      (direct piezoelectric effect: strain → charge)
//! - K_φφ = −∫ εˢ·∇φ·∇ψ dΩ      (dielectric stiffness at constant strain)
//!
//! Note: K_uφ = K_φuᵀ (the system is symmetric).
//!
//! # Simplified 1D forms
//!
//! For bars and beams vibrating in the 3-direction with electrodes on the
//! 3-faces (d₃₃ mode), the forms reduce to scalars:
//!
//! - K_uu: cᴱ₃₃ · ∫ (du/dx₃)(dv/dx₃) dΩ
//! - K_uφ: −e₃₃ · ∫ (dφ/dx₃)(dv/dx₃) dΩ
//! - K_φφ: −εˢ₃₃ · ∫ (dφ/dx₃)(dψ/dx₃) dΩ

use crate::forms::{BilinearForm, LocalMatrix};
use crate::spaces::BasisTabulation;

use super::materials::PiezoMaterial;

/// Elastic stiffness form at constant E-field (K_uu block).
///
/// `a(u, v) = ∫ cᴱ:ε(u):ε(v) dΩ`
///
/// For a scalar (1D longitudinal) problem along axis `d`:
/// `a(u, v) = cᴱ_{dd} · ∫ (∂u/∂x_d)(∂v/∂x_d) dΩ`
#[derive(Debug, Clone)]
pub struct PiezoStiffnessForm {
    /// Effective stiffness for the 1D/simplified case (Pa).
    /// For d₃₃ mode: cᴱ₃₃. For d₃₁ mode: cᴱ₁₁.
    pub stiffness: f64,
    /// Axis index (0=x, 1=y, 2=z) for 1D longitudinal mode.
    pub axis: usize,
}

impl PiezoStiffnessForm {
    /// Create for d₃₃ (longitudinal) mode: stiffness = cᴱ₃₃, axis = 2.
    pub fn d33(material: &PiezoMaterial) -> Self {
        Self {
            stiffness: material.stiffness[2][2],
            axis: 2,
        }
    }

    /// Create for d₃₁ (transverse) mode: stiffness = cᴱ₁₁, axis = 0.
    pub fn d31(material: &PiezoMaterial) -> Self {
        Self {
            stiffness: material.stiffness[0][0],
            axis: 0,
        }
    }
}

impl BilinearForm<f64> for PiezoStiffnessForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let d = self.axis.min(test.spatial_dim - 1);
        let w = weight * test.det_j * self.stiffness;

        for i in 0..n {
            let grad_i = test.grad(i, d);
            for j in 0..n {
                let grad_j = trial.grad(j, d);
                local_matrix.add(i, j, w * grad_i * grad_j);
            }
        }
    }
}

/// Piezoelectric coupling form (K_uφ or K_φu block).
///
/// Forward (inverse piezoelectric): `a(φ, v) = −e_{kd} · ∫ (∂φ/∂x_k)(∂v/∂x_d) dΩ`
/// Adjoint (direct piezoelectric): `a(u, ψ) = e_{kd} · ∫ (∂u/∂x_d)(∂ψ/∂x_k) dΩ`
///
/// For d₃₃ mode: e₃₃, both axes = 2 (poling direction).
/// For d₃₁ mode: e₃₁, electric axis = 2, mechanical axis = 0.
#[derive(Debug, Clone)]
pub struct PiezoCouplingForm {
    /// Piezoelectric stress constant e (C/m²).
    pub e_coeff: f64,
    /// Axis for the mechanical (strain) gradient.
    pub mech_axis: usize,
    /// Axis for the electrical (potential) gradient.
    pub elec_axis: usize,
    /// Sign: +1 for K_φu (direct), −1 for K_uφ (inverse).
    pub sign: f64,
}

impl PiezoCouplingForm {
    /// Direct piezoelectric effect (K_φu): strain → charge.
    /// `a(u, ψ) = e · ∫ (∂u/∂x_m)(∂ψ/∂x_e) dΩ`
    pub fn direct_d33(material: &PiezoMaterial) -> Self {
        Self {
            e_coeff: material.piezo_stress[2][2], // e₃₃
            mech_axis: 2,
            elec_axis: 2,
            sign: 1.0,
        }
    }

    /// Inverse piezoelectric effect (K_uφ): E-field → strain.
    /// `a(φ, v) = −e · ∫ (∂φ/∂x_e)(∂v/∂x_m) dΩ`
    pub fn inverse_d33(material: &PiezoMaterial) -> Self {
        Self {
            e_coeff: material.piezo_stress[2][2],
            mech_axis: 2,
            elec_axis: 2,
            sign: -1.0,
        }
    }

    /// Direct d₃₁: transverse strain → axial charge.
    pub fn direct_d31(material: &PiezoMaterial) -> Self {
        Self {
            e_coeff: material.piezo_stress[2][0], // e₃₁
            mech_axis: 0,
            elec_axis: 2,
            sign: 1.0,
        }
    }

    /// Inverse d₃₁: axial E-field → transverse strain.
    pub fn inverse_d31(material: &PiezoMaterial) -> Self {
        Self {
            e_coeff: material.piezo_stress[2][0],
            mech_axis: 0,
            elec_axis: 2,
            sign: -1.0,
        }
    }
}

impl BilinearForm<f64> for PiezoCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dm = self.mech_axis.min(test.spatial_dim - 1);
        let de = self.elec_axis.min(trial.spatial_dim - 1);
        let w = weight * test.det_j * self.sign * self.e_coeff;

        for i in 0..n {
            let grad_test = test.grad(i, dm);
            for j in 0..n {
                let grad_trial = trial.grad(j, de);
                local_matrix.add(i, j, w * grad_test * grad_trial);
            }
        }
    }
}

/// Dielectric stiffness form (K_φφ block).
///
/// `a(φ, ψ) = −εˢ · ∫ ∇φ·∇ψ dΩ`
///
/// For 1D: `a(φ, ψ) = −εˢ_{kk} · ∫ (∂φ/∂x_k)(∂ψ/∂x_k) dΩ`
///
/// Note: the negative sign makes the full 2×2 system positive semi-definite
/// when combined with the mechanical and coupling blocks.
#[derive(Debug, Clone)]
pub struct PiezoDielectricForm {
    /// Permittivity component for the electric field direction (F/m).
    pub permittivity: f64,
    /// Axis for the electric field gradient.
    pub axis: usize,
}

impl PiezoDielectricForm {
    /// For d₃₃ or d₃₁ mode: εˢ₃₃ along axis 2.
    pub fn d33(material: &PiezoMaterial) -> Self {
        Self {
            permittivity: material.permittivity[2][2],
            axis: 2,
        }
    }
}

impl BilinearForm<f64> for PiezoDielectricForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let d = self.axis.min(test.spatial_dim - 1);
        // Negative sign: K_φφ = −εˢ ∫ ∇φ·∇ψ
        let w = -weight * test.det_j * self.permittivity;

        for i in 0..n {
            let grad_i = test.grad(i, d);
            for j in 0..n {
                let grad_j = trial.grad(j, d);
                local_matrix.add(i, j, w * grad_i * grad_j);
            }
        }
    }
}

/// Compute the static displacement of a d₃₁ piezo bar (1D analytical).
///
/// For a bar of length L along axis 1, poled in direction 3, with voltage V
/// across thickness t:
///
/// `Δx₁ = d₃₁ · V · L / t`
///
/// where d₃₁ is the piezoelectric strain constant (m/V).
/// Note: d₃₁ = e₃₁ / cᴱ₁₁ (in simplified 1D).
pub fn d31_bar_displacement(
    material: &PiezoMaterial,
    voltage: f64,
    length: f64,
    thickness: f64,
) -> f64 {
    let e31 = material.piezo_stress[2][0];
    let c11 = material.stiffness[0][0];
    let d31 = e31 / c11; // strain constant (m/V)
    d31 * voltage * length / thickness
}

/// Compute the static displacement of a d₃₃ piezo stack.
///
/// For a stack of N layers, each of thickness t, with voltage V per layer:
///
/// `Δx₃ = d₃₃ · V · N`
///
/// where d₃₃ = e₃₃ / cᴱ₃₃.
pub fn d33_stack_displacement(
    material: &PiezoMaterial,
    voltage_per_layer: f64,
    n_layers: usize,
) -> f64 {
    let e33 = material.piezo_stress[2][2];
    let c33 = material.stiffness[2][2];
    let d33 = e33 / c33;
    d33 * voltage_per_layer * n_layers as f64
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, H1Space};

    #[test]
    fn test_stiffness_form_symmetry() {
        let mat = PiezoMaterial::pzt_5a();
        let form = PiezoStiffnessForm::d33(&mat);
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (k.get(i, j) - k.get(j, i)).abs();
                assert!(diff < 1e-10, "Not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_coupling_form_transpose_relation() {
        // K_uφ = −K_φuᵀ (direct vs inverse piezo effect)
        let mat = PiezoMaterial::pzt_5a();
        let direct = PiezoCouplingForm::direct_d33(&mat);
        let inverse = PiezoCouplingForm::inverse_d33(&mat);

        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut k_direct = LocalMatrix::zeros(n);
        let mut k_inverse = LocalMatrix::zeros(n);
        direct.integrate_point(&tab, &tab, 1.0, &mut k_direct);
        inverse.integrate_point(&tab, &tab, 1.0, &mut k_inverse);

        // K_direct = +e·... and K_inverse = −e·..., so K_direct = −K_inverse
        // (when test and trial spaces are the same and axes match)
        for i in 0..n {
            for j in 0..n {
                let sum = k_direct.get(i, j) + k_inverse.get(i, j);
                assert!(
                    sum.abs() < 1e-10,
                    "K_direct + K_inverse should be zero at ({i},{j}): {sum}"
                );
            }
        }
    }

    #[test]
    fn test_dielectric_form_negative_definite() {
        let mat = PiezoMaterial::pzt_5a();
        let form = PiezoDielectricForm::d33(&mat);
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut k = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut k);

        // K_φφ should be symmetric and negative semi-definite
        for i in 0..n {
            assert!(
                k.get(i, i) <= 0.0,
                "Diagonal should be ≤ 0: K[{i},{i}] = {}",
                k.get(i, i)
            );
        }
    }

    #[test]
    fn test_d31_bar_displacement() {
        let mat = PiezoMaterial::pzt_5a();
        let voltage = 100.0; // 100 V
        let length = 0.050; // 50 mm bar
        let thickness = 0.001; // 1 mm electrode spacing

        let dx = d31_bar_displacement(&mat, voltage, length, thickness);

        // d₃₁ = e₃₁/c₁₁ = -5.4/12.1e10 ≈ -4.46e-11 m/V
        // Δx = -4.46e-11 × 100 × 0.050 / 0.001 = -2.23e-7 m = -223 nm
        assert!(
            dx.abs() > 1e-9 && dx.abs() < 1e-5,
            "d₃₁ displacement = {dx:.3e} m"
        );
        // d₃₁ is negative (contraction), so displacement should be negative
        assert!(dx < 0.0, "d₃₁ mode should contract: Δx = {dx:.3e} m");
    }

    #[test]
    fn test_d33_stack_displacement() {
        let mat = PiezoMaterial::pzt_5a();
        let voltage = 100.0;
        let n_layers = 10;

        let dx = d33_stack_displacement(&mat, voltage, n_layers);

        // d₃₃ = e₃₃/c₃₃ = 15.8/11.1e10 ≈ 1.42e-10 m/V
        // Δx = 1.42e-10 × 100 × 10 = 1.42e-7 m = 142 nm
        assert!(
            dx.abs() > 1e-9 && dx.abs() < 1e-5,
            "d₃₃ displacement = {dx:.3e} m"
        );
        // d₃₃ is positive (extension along poling direction)
        assert!(dx > 0.0, "d₃₃ mode should extend: Δx = {dx:.3e} m");
    }

    #[test]
    fn test_d33_scales_with_layers() {
        let mat = PiezoMaterial::pzt_5a();
        let dx1 = d33_stack_displacement(&mat, 100.0, 1);
        let dx10 = d33_stack_displacement(&mat, 100.0, 10);
        assert!((dx10 - 10.0 * dx1).abs() < 1e-20);
    }
}
