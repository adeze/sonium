//! Two-port transfer matrix networks for acoustic systems
//!
//! Represents acoustic components as 2×2 transfer matrices relating
//! pressure and volume velocity at input/output:
//!
//! ```text
//! | p_out |   | A  B | | p_in  |
//! | U_out | = | C  D | | U_in  |
//! ```
//!
//! Transfer matrices cascade by multiplication: T_total = T₁ × T₂ × ⋯ × Tₙ.
//!
//! # Applications
//!
//! - Pipe/duct networks: cascade tube sections, area changes, side branches
//! - Muffler design: expansion chambers, resonators, perforated pipes
//! - Transmission loss computation from the cascaded transfer matrix

use num_complex::Complex64;

/// 2×2 complex transfer matrix \[A B; C D\].
#[derive(Debug, Clone, Copy)]
pub struct TransferMatrix {
    pub a: Complex64,
    pub b: Complex64,
    pub c: Complex64,
    pub d: Complex64,
}

impl TransferMatrix {
    pub fn new(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Self {
        Self { a, b, c, d }
    }

    /// Identity matrix (pass-through).
    pub fn identity() -> Self {
        Self {
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(0.0, 0.0),
            c: Complex64::new(0.0, 0.0),
            d: Complex64::new(1.0, 0.0),
        }
    }

    /// Cascade (multiply): self × other.
    pub fn cascade(&self, other: &TransferMatrix) -> TransferMatrix {
        TransferMatrix {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }

    /// Determinant: AD − BC. Should be 1 for passive reciprocal elements.
    pub fn determinant(&self) -> Complex64 {
        self.a * self.d - self.b * self.c
    }

    /// Input impedance when the output is terminated with Z_load.
    ///
    /// Z_in = (A Z_load + B) / (C Z_load + D)
    pub fn input_impedance(&self, z_load: Complex64) -> Complex64 {
        (self.a * z_load + self.b) / (self.c * z_load + self.d)
    }

    /// Transmission loss (dB) assuming anechoic termination (Z_load = Z₀).
    ///
    /// TL = 20 log₁₀(|A + B/Z₀ + C Z₀ + D| / 2)
    pub fn transmission_loss(&self, z0: f64) -> f64 {
        let z0c = Complex64::new(z0, 0.0);
        let total = self.a + self.b / z0c + self.c * z0c + self.d;
        20.0 * (total.norm() / 2.0).log10()
    }
}

// ---------------------------------------------------------------------------
// Standard acoustic elements as transfer matrices
// ---------------------------------------------------------------------------

/// Uniform tube of length L and cross-section S (lossless).
///
/// ```text
/// A = D = cos(kL)
/// B = j Z₀ sin(kL)
/// C = j sin(kL) / Z₀
/// ```
/// where Z₀ = ρc/S is the acoustic impedance of the tube.
pub fn tube(k: Complex64, length: f64, z0: f64) -> TransferMatrix {
    let kl = k * length;
    let cos_kl = kl.cos();
    let sin_kl = kl.sin();
    let j = Complex64::new(0.0, 1.0);
    TransferMatrix {
        a: cos_kl,
        b: j * z0 * sin_kl,
        c: j * sin_kl / z0,
        d: cos_kl,
    }
}

/// Area discontinuity (sudden expansion or contraction).
///
/// At a junction from cross-section S₁ to S₂, assuming continuity of
/// pressure and volume velocity: T = I (identity).
///
/// However, when using different Z₀ on each side, the matrix accounts
/// for the impedance change.
///
/// For the simple case (same Z₀ convention): identity.
pub fn area_discontinuity() -> TransferMatrix {
    TransferMatrix::identity()
}

/// Series impedance element: Z in the pressure path.
///
/// ```text
/// T = | 1  Z |
///     | 0  1 |
/// ```
pub fn series_element(z: Complex64) -> TransferMatrix {
    TransferMatrix {
        a: Complex64::new(1.0, 0.0),
        b: z,
        c: Complex64::new(0.0, 0.0),
        d: Complex64::new(1.0, 0.0),
    }
}

/// Shunt (parallel) impedance element: Z across the path.
///
/// ```text
/// T = | 1    0   |
///     | 1/Z  1   |
/// ```
pub fn shunt_element(z: Complex64) -> TransferMatrix {
    TransferMatrix {
        a: Complex64::new(1.0, 0.0),
        b: Complex64::new(0.0, 0.0),
        c: Complex64::new(1.0, 0.0) / z,
        d: Complex64::new(1.0, 0.0),
    }
}

/// Cascade a sequence of transfer matrices.
pub fn cascade_all(matrices: &[TransferMatrix]) -> TransferMatrix {
    matrices
        .iter()
        .fold(TransferMatrix::identity(), |acc, m| acc.cascade(m))
}

/// Simple expansion chamber muffler: inlet tube → expansion → outlet tube.
///
/// Returns the cascaded transfer matrix.
pub fn expansion_chamber(
    k: Complex64,
    inlet_length: f64,
    inlet_z0: f64,
    chamber_length: f64,
    chamber_z0: f64,
    outlet_length: f64,
    outlet_z0: f64,
) -> TransferMatrix {
    let t_in = tube(k, inlet_length, inlet_z0);
    let t_chamber = tube(k, chamber_length, chamber_z0);
    let t_out = tube(k, outlet_length, outlet_z0);
    cascade_all(&[t_in, t_chamber, t_out])
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const RHO: f64 = 1.204;
    const C: f64 = 343.0;

    #[test]
    fn test_identity_cascade() {
        let id = TransferMatrix::identity();
        let t = tube(Complex64::new(18.3, 0.0), 0.5, 1000.0);
        let result = id.cascade(&t);
        assert!((result.a - t.a).norm() < 1e-10);
        assert!((result.b - t.b).norm() < 1e-10);
    }

    #[test]
    fn test_tube_reciprocity() {
        let k = Complex64::new(18.3, 0.0);
        let t = tube(k, 0.5, 1000.0);
        // Reciprocal: det = AD − BC = 1
        let det = t.determinant();
        assert!(
            (det - Complex64::new(1.0, 0.0)).norm() < 1e-10,
            "det = {det}"
        );
    }

    #[test]
    fn test_tube_symmetry() {
        let k = Complex64::new(18.3, 0.0);
        let t = tube(k, 0.5, 1000.0);
        // Symmetric tube: A = D
        assert!((t.a - t.d).norm() < 1e-10);
    }

    #[test]
    fn test_zero_length_tube_is_identity() {
        let k = Complex64::new(18.3, 0.0);
        let t = tube(k, 0.0, 1000.0);
        let id = TransferMatrix::identity();
        assert!((t.a - id.a).norm() < 1e-10);
        assert!((t.b - id.b).norm() < 1e-10);
    }

    #[test]
    fn test_input_impedance_open_end() {
        // Open end: Z_load = 0 (pressure = 0)
        // Z_in = B/D
        let k = Complex64::new(18.3, 0.0);
        let z0 = 1000.0;
        let t = tube(k, 0.5, z0);
        let z_in = t.input_impedance(Complex64::new(0.0, 0.0));
        let expected = t.b / t.d;
        assert!((z_in - expected).norm() < 1e-10);
    }

    #[test]
    fn test_expansion_chamber_tl() {
        // Simple expansion chamber: higher TL at resonance frequencies
        let f = 500.0;
        let omega = 2.0 * PI * f;
        let k = Complex64::new(omega / C, 0.0);

        let s_tube = 1e-4; // 1 cm² tube
        let s_chamber = 1e-3; // 10 cm² chamber (10:1 expansion)
        let z_tube = RHO * C / s_tube;
        let z_chamber = RHO * C / s_chamber;

        let t = expansion_chamber(k, 0.1, z_tube, 0.3, z_chamber, 0.1, z_tube);
        let tl = t.transmission_loss(z_tube);
        // Expansion chambers always have TL > 0
        assert!(tl >= 0.0, "TL = {tl:.1} dB");
    }

    #[test]
    fn test_series_shunt_duality() {
        let z = Complex64::new(100.0, 50.0);
        let ts = series_element(z);
        let tp = shunt_element(z);
        // Series: B = Z, rest identity-like
        assert!((ts.b - z).norm() < 1e-10);
        // Shunt: C = 1/Z, rest identity-like
        assert!((tp.c - Complex64::new(1.0, 0.0) / z).norm() < 1e-10);
    }
}
