//! Piezoelectric material properties
//!
//! Stores the coupling matrices for the strain-charge constitutive law:
//!
//! ```text
//! S = sᴱ·T + dᵗ·E     (mechanical)
//! D = d·T  + εᵀ·E      (electrical)
//! ```
//!
//! Or in stress-charge (stiffness) form:
//! ```text
//! T = cᴱ·S − eᵗ·E
//! D = e·S  + εˢ·E
//! ```
//!
//! where `e = d·cᴱ` is the piezoelectric stress constant matrix.
//!
//! The FEM formulation uses the stiffness form with fields (u, φ):
//! - Mechanical: `∫ cᴱ:ε(u):ε(v) dΩ − ∫ eᵗ·∇φ:ε(v) dΩ = F_ext`
//! - Electrical: `∫ e:ε(u)·∇ψ dΩ + ∫ εˢ·∇φ·∇ψ dΩ = Q_ext`

/// Piezoelectric material in stress-charge form.
///
/// All tensors are stored in Voigt notation for 3D:
/// - Stress/strain: 6-component vectors (11, 22, 33, 23, 13, 12)
/// - Stiffness: 6×6 matrix
/// - Piezoelectric stress: 3×6 matrix
/// - Dielectric: 3×3 matrix
#[derive(Debug, Clone)]
pub struct PiezoMaterial {
    /// Elastic stiffness at constant E-field, cᴱ [6×6, Pa].
    /// Voigt notation: c[i][j] where i,j ∈ {0..5}.
    pub stiffness: [[f64; 6]; 6],

    /// Piezoelectric stress constant e [3×6, C/m²].
    /// e[k][i] relates electric field component k to stress component i.
    pub piezo_stress: [[f64; 6]; 3],

    /// Dielectric permittivity at constant strain εˢ [3×3, F/m].
    pub permittivity: [[f64; 3]; 3],

    /// Density (kg/m³).
    pub density: f64,
}

impl PiezoMaterial {
    /// PZT-5A (lead zirconate titanate, Navy Type II).
    ///
    /// A general-purpose piezoelectric ceramic with moderate coupling.
    /// Widely used in actuators, sensors, and transducers.
    pub fn pzt_5a() -> Self {
        // Elastic stiffness cᴱ (×10¹⁰ Pa)
        let c11 = 12.1e10;
        let c12 = 7.54e10;
        let c13 = 7.52e10;
        let c33 = 11.1e10;
        let c44 = 2.11e10;
        let c66 = (c11 - c12) / 2.0; // = 2.28e10

        // Piezoelectric stress constants e (C/m²)
        let e31 = -5.4;
        let e33 = 15.8;
        let e15 = 12.3;

        // Dielectric permittivity εˢ (F/m)
        let eps11 = 8.107e-9; // ε₁₁ˢ
        let eps33 = 7.346e-9; // ε₃₃ˢ

        Self::transversely_isotropic(
            c11, c12, c13, c33, c44, c66, e31, e33, e15, eps11, eps33, 7750.0,
        )
    }

    /// PZT-5H (lead zirconate titanate, Navy Type VI).
    ///
    /// Higher coupling coefficient than PZT-5A. Used in high-sensitivity
    /// applications (hydrophones, medical ultrasound).
    pub fn pzt_5h() -> Self {
        let c11 = 12.6e10;
        let c12 = 7.95e10;
        let c13 = 8.41e10;
        let c33 = 11.7e10;
        let c44 = 2.30e10;
        let c66 = (c11 - c12) / 2.0;

        let e31 = -6.5;
        let e33 = 23.3;
        let e15 = 17.0;

        let eps11 = 15.05e-9;
        let eps33 = 13.02e-9;

        Self::transversely_isotropic(
            c11, c12, c13, c33, c44, c66, e31, e33, e15, eps11, eps33, 7500.0,
        )
    }

    /// Build from transversely isotropic constants (poling direction = 3).
    ///
    /// This covers most piezoelectric ceramics (PZT, BaTiO₃, etc).
    #[allow(clippy::too_many_arguments)]
    fn transversely_isotropic(
        c11: f64,
        c12: f64,
        c13: f64,
        c33: f64,
        c44: f64,
        c66: f64,
        e31: f64,
        e33: f64,
        e15: f64,
        eps11: f64,
        eps33: f64,
        density: f64,
    ) -> Self {
        let stiffness = [
            [c11, c12, c13, 0.0, 0.0, 0.0],
            [c12, c11, c13, 0.0, 0.0, 0.0],
            [c13, c13, c33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c66],
        ];

        // e[k][i]: electric direction k, stress component i
        let piezo_stress = [
            [0.0, 0.0, 0.0, 0.0, e15, 0.0], // e₁: shear coupling
            [0.0, 0.0, 0.0, e15, 0.0, 0.0], // e₂: shear coupling
            [e31, e31, e33, 0.0, 0.0, 0.0], // e₃: axial coupling
        ];

        let permittivity = [[eps11, 0.0, 0.0], [0.0, eps11, 0.0], [0.0, 0.0, eps33]];

        Self {
            stiffness,
            piezo_stress,
            permittivity,
            density,
        }
    }

    /// Electromechanical coupling coefficient k₃₃².
    ///
    /// `k₃₃² = e₃₃² / (cᴱ₃₃ · εˢ₃₃)`.
    /// This is the fraction of input electrical energy converted to mechanical.
    pub fn coupling_k33_squared(&self) -> f64 {
        let e33 = self.piezo_stress[2][2]; // e₃₃
        let c33 = self.stiffness[2][2]; // cᴱ₃₃
        let eps33 = self.permittivity[2][2]; // εˢ₃₃
        e33 * e33 / (c33 * eps33)
    }

    /// Electromechanical coupling coefficient k₃₁².
    pub fn coupling_k31_squared(&self) -> f64 {
        let e31 = self.piezo_stress[2][0]; // e₃₁
        let c11 = self.stiffness[0][0]; // cᴱ₁₁
        let eps33 = self.permittivity[2][2];
        e31 * e31 / (c11 * eps33)
    }

    /// Longitudinal wave speed along the poling direction (3-axis).
    ///
    /// `c_L = √(cᴱ₃₃ / ρ)` for open-circuit (uncoupled) propagation.
    pub fn longitudinal_wave_speed(&self) -> f64 {
        (self.stiffness[2][2] / self.density).sqrt()
    }

    /// Resonance frequency of a free bar of length L vibrating in the 3-direction.
    ///
    /// `f_r = c_L / (2L)` for the fundamental (half-wave) mode.
    pub fn bar_resonance_frequency(&self, length: f64) -> f64 {
        self.longitudinal_wave_speed() / (2.0 * length)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pzt5a_stiffness_symmetry() {
        let mat = PiezoMaterial::pzt_5a();
        for i in 0..6 {
            for j in i + 1..6 {
                assert!(
                    (mat.stiffness[i][j] - mat.stiffness[j][i]).abs() < 1e-5,
                    "c[{i}][{j}] ≠ c[{j}][{i}]"
                );
            }
        }
    }

    #[test]
    fn test_pzt5a_coupling_k33() {
        let mat = PiezoMaterial::pzt_5a();
        let k33_sq = mat.coupling_k33_squared();
        // PZT-5A: k₃₃ ≈ 0.70, k₃₃² ≈ 0.49
        assert!(k33_sq > 0.1 && k33_sq < 0.8, "k₃₃² = {k33_sq:.3}");
    }

    #[test]
    fn test_pzt5a_coupling_k31() {
        let mat = PiezoMaterial::pzt_5a();
        let k31_sq = mat.coupling_k31_squared();
        // PZT-5A: k₃₁ ≈ 0.34, k₃₁² ≈ 0.12
        assert!(k31_sq > 0.01 && k31_sq < 0.3, "k₃₁² = {k31_sq:.3}");
    }

    #[test]
    fn test_pzt5h_higher_coupling() {
        let pzt5a = PiezoMaterial::pzt_5a();
        let pzt5h = PiezoMaterial::pzt_5h();
        assert!(
            pzt5h.coupling_k33_squared() > pzt5a.coupling_k33_squared(),
            "PZT-5H should have higher k₃₃ than PZT-5A"
        );
    }

    #[test]
    fn test_wave_speed_reasonable() {
        let mat = PiezoMaterial::pzt_5a();
        let c_l = mat.longitudinal_wave_speed();
        // PZT: ~3500–4500 m/s
        assert!(c_l > 3000.0 && c_l < 5000.0, "c_L = {c_l:.0} m/s");
    }

    #[test]
    fn test_bar_resonance() {
        let mat = PiezoMaterial::pzt_5a();
        let f_r = mat.bar_resonance_frequency(0.01); // 10 mm bar
        // f = c/(2L) ≈ 3800/(0.02) ≈ 190 kHz
        assert!(f_r > 100e3 && f_r < 300e3, "f_r = {:.0} kHz", f_r / 1e3);
    }
}
