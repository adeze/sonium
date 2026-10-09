//! Elastic material models

/// Isotropic linear elastic material defined by Young's modulus and Poisson's ratio.
#[derive(Debug, Clone, Copy)]
pub struct IsotropicMaterial {
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
}

impl IsotropicMaterial {
    pub fn new(youngs_modulus: f64, poisson_ratio: f64) -> Self {
        assert!(youngs_modulus > 0.0, "Young's modulus must be positive");
        assert!(
            (-1.0..0.5).contains(&poisson_ratio),
            "Poisson ratio must be in (-1, 0.5)"
        );
        Self {
            youngs_modulus,
            poisson_ratio,
        }
    }

    /// Steel: E = 210 GPa, nu = 0.3
    pub fn steel() -> Self {
        Self::new(210e9, 0.3)
    }

    /// Aluminum: E = 70 GPa, nu = 0.33
    pub fn aluminum() -> Self {
        Self::new(70e9, 0.33)
    }

    /// Rubber-like: E = 1 MPa, nu = 0.499
    pub fn rubber() -> Self {
        Self::new(1e6, 0.499)
    }

    /// First Lamé parameter: lambda = E*nu / ((1+nu)(1-2*nu))
    pub fn lambda(&self) -> f64 {
        let e = self.youngs_modulus;
        let v = self.poisson_ratio;
        e * v / ((1.0 + v) * (1.0 - 2.0 * v))
    }

    /// Second Lamé parameter (shear modulus): mu = E / (2(1+nu))
    pub fn mu(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus: K = E / (3(1-2*nu))
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Build the 3D elasticity tensor in Voigt notation (6x6 symmetric matrix, row-major).
    ///
    /// Voigt ordering: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]
    ///
    /// C = lambda * (1 ⊗ 1) + 2*mu * I
    pub fn voigt_3d(&self) -> [f64; 36] {
        let l = self.lambda();
        let m = self.mu();
        let m2 = 2.0 * m;
        [
            l + m2,
            l,
            l,
            0.0,
            0.0,
            0.0, // row 0
            l,
            l + m2,
            l,
            0.0,
            0.0,
            0.0, // row 1
            l,
            l,
            l + m2,
            0.0,
            0.0,
            0.0, // row 2
            0.0,
            0.0,
            0.0,
            m,
            0.0,
            0.0, // row 3
            0.0,
            0.0,
            0.0,
            0.0,
            m,
            0.0, // row 4
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            m, // row 5
        ]
    }

    /// Build the 2D plane-stress elasticity tensor in Voigt notation (3x3, row-major).
    ///
    /// Voigt ordering: [sigma_xx, sigma_yy, sigma_xy]
    pub fn voigt_2d_plane_stress(&self) -> [f64; 9] {
        let e = self.youngs_modulus;
        let v = self.poisson_ratio;
        let factor = e / (1.0 - v * v);
        [
            factor,
            factor * v,
            0.0,
            factor * v,
            factor,
            0.0,
            0.0,
            0.0,
            factor * (1.0 - v) / 2.0,
        ]
    }

    /// Build the 2D plane-strain elasticity tensor in Voigt notation (3x3, row-major).
    ///
    /// Voigt ordering: [sigma_xx, sigma_yy, sigma_xy]
    pub fn voigt_2d_plane_strain(&self) -> [f64; 9] {
        let l = self.lambda();
        let m = self.mu();
        let m2 = 2.0 * m;
        [l + m2, l, 0.0, l, l + m2, 0.0, 0.0, 0.0, m]
    }
}

/// Viscoelastic material with frequency-independent structural damping.
///
/// The complex stiffness is `C* = C * (1 + i * eta)` where `eta` is the
/// loss factor (structural damping ratio). Typical values:
/// - Steel: eta ~ 0.001
/// - Aluminum: eta ~ 0.002
/// - Rubber: eta ~ 0.1-1.0
/// - Loudspeaker surround: eta ~ 0.3-0.5
/// - Loudspeaker spider: eta ~ 0.05-0.2
#[derive(Debug, Clone, Copy)]
pub struct ViscoelasticMaterial {
    /// Base elastic material.
    pub elastic: IsotropicMaterial,
    /// Structural loss factor eta.
    pub loss_factor: f64,
}

impl ViscoelasticMaterial {
    pub fn new(elastic: IsotropicMaterial, loss_factor: f64) -> Self {
        assert!(loss_factor >= 0.0, "Loss factor must be non-negative");
        Self {
            elastic,
            loss_factor,
        }
    }

    /// Loudspeaker surround (soft, lossy).
    pub fn surround() -> Self {
        Self::new(IsotropicMaterial::new(5e6, 0.45), 0.4)
    }

    /// Loudspeaker spider (stiffer, moderate damping).
    pub fn spider() -> Self {
        Self::new(IsotropicMaterial::new(50e6, 0.35), 0.1)
    }

    /// Complex multiplier: `1 + i * eta`.
    pub fn complex_factor(&self) -> num_complex::Complex64 {
        num_complex::Complex64::new(1.0, self.loss_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lame_parameters() {
        let mat = IsotropicMaterial::new(1.0, 0.25);
        // lambda = 1*0.25 / ((1.25)(0.5)) = 0.25 / 0.625 = 0.4
        assert!((mat.lambda() - 0.4).abs() < 1e-14);
        // mu = 1 / (2*1.25) = 0.4
        assert!((mat.mu() - 0.4).abs() < 1e-14);
    }

    #[test]
    fn test_voigt_3d_symmetry() {
        let mat = IsotropicMaterial::steel();
        let c = mat.voigt_3d();
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (c[i * 6 + j] - c[j * 6 + i]).abs() < 1e-6,
                    "C[{i},{j}] != C[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_voigt_2d_plane_stress_symmetry() {
        let mat = IsotropicMaterial::new(200e9, 0.3);
        let c = mat.voigt_2d_plane_stress();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (c[i * 3 + j] - c[j * 3 + i]).abs() < 1e-6,
                    "C[{i},{j}] != C[{j},{i}]"
                );
            }
        }
    }
}
