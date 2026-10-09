//! Waveguide port boundary conditions with multimode expansion
//!
//! Ports are used to excite or absorb acoustic waves at the inlet and
//! outlet of waveguide structures. Each propagating duct mode contributes
//! independently to the DtN (Dirichlet-to-Neumann) map.
//!
//! # Plane wave port
//!
//! For the fundamental (plane wave) mode, the port BC reduces to the
//! Sommerfeld absorbing condition plus an excitation term.
//!
//! # Multimode port
//!
//! For higher-order modes in rectangular ducts:
//! ```text
//! ∂p/∂n = Σ_n j k_{x,n} ⟨φ_n, p⟩ φ_n / ⟨φ_n, φ_n⟩ + excitation
//! ```
//!
//! where φ_n are the transverse mode shapes and k_{x,n} the axial wavenumber.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Duct modes
// ---------------------------------------------------------------------------

/// A single duct mode with its transverse wavenumber and shape.
#[derive(Debug, Clone)]
pub struct DuctMode {
    /// Mode indices (m, n) for rectangular duct, (m, n) for circular.
    pub indices: (usize, usize),
    /// Transverse wavenumber k_t = √((mπ/a)² + (nπ/b)²) \[1/m\].
    pub transverse_wavenumber: f64,
    /// Cutoff frequency f_c = c k_t / (2π) \[Hz\].
    pub cutoff_frequency: f64,
}

impl DuctMode {
    /// Axial wavenumber: k_x = √(k² − k_t²).
    ///
    /// Real for propagating modes (f > f_c), imaginary for evanescent.
    pub fn axial_wavenumber(&self, k: f64) -> Complex64 {
        let k_sq = k * k;
        let kt_sq = self.transverse_wavenumber * self.transverse_wavenumber;
        if k_sq >= kt_sq {
            Complex64::new((k_sq - kt_sq).sqrt(), 0.0)
        } else {
            Complex64::new(0.0, (kt_sq - k_sq).sqrt())
        }
    }

    /// Whether the mode propagates at wavenumber k.
    pub fn is_propagating(&self, k: f64) -> bool {
        k * k >= self.transverse_wavenumber * self.transverse_wavenumber
    }

    /// Evaluate the mode shape at position (y, z) in the duct cross-section.
    ///
    /// For rectangular ducts: φ_{m,n}(y,z) = cos(mπy/a) cos(nπz/b).
    pub fn evaluate(&self, y: f64, z: f64, width: f64, height: f64) -> f64 {
        let (m, n) = self.indices;
        let phi_y = if m == 0 {
            1.0
        } else {
            (m as f64 * PI * y / width).cos()
        };
        let phi_z = if n == 0 {
            1.0
        } else {
            (n as f64 * PI * z / height).cos()
        };
        phi_y * phi_z
    }
}

/// Rectangular duct geometry.
#[derive(Debug, Clone, Copy)]
pub struct RectangularDuct {
    /// Width a \[m\] (y-dimension).
    pub width: f64,
    /// Height b \[m\] (z-dimension). For 2D ducts, set height = 1.
    pub height: f64,
}

impl RectangularDuct {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    /// 2D duct (slit) with given width.
    pub fn slit(width: f64) -> Self {
        Self { width, height: 1.0 }
    }

    /// Compute all modes up to mode order `max_order` in each dimension.
    pub fn modes(&self, c: f64, max_order: usize) -> Vec<DuctMode> {
        let mut modes = Vec::new();
        for m in 0..=max_order {
            for n in 0..=max_order {
                if m == 0 && n == 0 {
                    modes.push(DuctMode {
                        indices: (0, 0),
                        transverse_wavenumber: 0.0,
                        cutoff_frequency: 0.0,
                    });
                    continue;
                }
                let kt_y = m as f64 * PI / self.width;
                let kt_z = n as f64 * PI / self.height;
                let kt = (kt_y * kt_y + kt_z * kt_z).sqrt();
                let fc = c * kt / (2.0 * PI);
                modes.push(DuctMode {
                    indices: (m, n),
                    transverse_wavenumber: kt,
                    cutoff_frequency: fc,
                });
            }
        }
        modes
    }

    /// Compute only propagating modes at frequency f.
    pub fn propagating_modes(&self, f: f64, c: f64, max_order: usize) -> Vec<DuctMode> {
        self.modes(c, max_order)
            .into_iter()
            .filter(|m| m.cutoff_frequency <= f)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Waveguide port
// ---------------------------------------------------------------------------

/// Waveguide port boundary condition.
///
/// Provides the DtN map coefficient and excitation vector for a set of
/// duct modes at a given frequency.
#[derive(Debug)]
pub struct WaveguidePort {
    /// Boundary tag for this port.
    pub boundary_tag: usize,
    /// Active modes (typically all propagating modes).
    pub modes: Vec<DuctMode>,
}

impl WaveguidePort {
    pub fn new(boundary_tag: usize, modes: Vec<DuctMode>) -> Self {
        Self {
            boundary_tag,
            modes,
        }
    }

    /// Plane wave port (fundamental mode only).
    pub fn plane_wave(boundary_tag: usize) -> Self {
        Self {
            boundary_tag,
            modes: vec![DuctMode {
                indices: (0, 0),
                transverse_wavenumber: 0.0,
                cutoff_frequency: 0.0,
            }],
        }
    }

    /// Robin coefficient for the plane-wave component (mode 0,0).
    ///
    /// For a plane wave port, the DtN map contribution is equivalent to
    /// the Sommerfeld ABC: `boundary_coeffs[tag] = -ik` under `exp(-iωt)`.
    ///
    /// For multimode, each mode contributes its own axial wavenumber to
    /// the DtN map. The plane-wave coefficient `-ik_x,0 = -ik` is the
    /// dominant term.
    pub fn plane_wave_coefficient(&self, k: f64) -> Complex64 {
        if let Some(m0) = self.modes.first() {
            let kx = m0.axial_wavenumber(k);
            Complex64::new(0.0, -1.0) * kx
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Compute the incident pressure excitation for a plane wave of
    /// amplitude `p_inc` arriving from outside the domain.
    ///
    /// For the weak form, the excitation contributes:
    /// ```text
    /// RHS_i += -2 i k_{x,0} p_inc × ∫_Γ φ_i dΓ
    /// ```
    ///
    /// Returns the scalar multiplier for the boundary load vector.
    pub fn plane_wave_excitation(&self, k: f64, p_inc: Complex64) -> Complex64 {
        let kx = self
            .modes
            .first()
            .map_or(Complex64::new(k, 0.0), |m| m.axial_wavenumber(k));
        Complex64::new(0.0, -2.0) * kx * p_inc
    }

    /// Number of propagating modes at wavenumber k.
    pub fn num_propagating(&self, k: f64) -> usize {
        self.modes.iter().filter(|m| m.is_propagating(k)).count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const C: f64 = 343.0;

    #[test]
    fn test_rectangular_duct_plane_wave_mode() {
        let duct = RectangularDuct::new(0.1, 0.1);
        let modes = duct.modes(C, 0);
        assert_eq!(modes.len(), 1);
        assert_eq!(modes[0].indices, (0, 0));
        assert_eq!(modes[0].cutoff_frequency, 0.0);
    }

    #[test]
    fn test_rectangular_duct_cutoff_frequencies() {
        let a = 0.1; // 10 cm
        let duct = RectangularDuct::slit(a);
        let modes = duct.modes(C, 3);

        // Mode (1,0): f_c = c/(2a) = 343/0.2 = 1715 Hz
        let m10 = modes.iter().find(|m| m.indices == (1, 0)).unwrap();
        let expected = C / (2.0 * a);
        assert!(
            (m10.cutoff_frequency - expected).abs() < 1.0,
            "f_c(1,0) = {}, expected {}",
            m10.cutoff_frequency,
            expected
        );

        // Mode (2,0): f_c = c/a = 3430 Hz
        let m20 = modes.iter().find(|m| m.indices == (2, 0)).unwrap();
        let expected2 = C / a;
        assert!(
            (m20.cutoff_frequency - expected2).abs() < 1.0,
            "f_c(2,0) = {}, expected {}",
            m20.cutoff_frequency,
            expected2
        );
    }

    #[test]
    fn test_propagating_vs_evanescent() {
        // Use a small height so z-modes have high cutoffs (> 1 kHz)
        let duct = RectangularDuct::new(0.1, 0.01);
        let modes = duct.modes(C, 3);

        let k = 2.0 * PI * 1000.0 / C; // 1 kHz: below first y-cutoff (1715 Hz)

        // Only mode (0,0) should propagate (first z-cutoff = c/(2×0.01) = 17150 Hz)
        let propagating: Vec<_> = modes.iter().filter(|m| m.is_propagating(k)).collect();
        assert_eq!(propagating.len(), 1);
        assert_eq!(propagating[0].indices, (0, 0));

        // Mode (1,0) should be evanescent
        let m10 = modes.iter().find(|m| m.indices == (1, 0)).unwrap();
        let kx = m10.axial_wavenumber(k);
        assert!(
            kx.re.abs() < 1e-10 && kx.im > 0.0,
            "Evanescent mode should have imaginary k_x: {kx}"
        );
    }

    #[test]
    fn test_plane_wave_port_coefficient_equals_sommerfeld() {
        let port = WaveguidePort::plane_wave(1);
        let k = 2.0 * PI * 1000.0 / C;
        let coeff = port.plane_wave_coefficient(k);
        let expected = Complex64::new(0.0, -k); // -ik for exp(-iωt)
        assert!(
            (coeff - expected).norm() < 1e-10,
            "Plane wave port coeff = {coeff}, expected {expected}"
        );
    }

    #[test]
    fn test_plane_wave_excitation() {
        let port = WaveguidePort::plane_wave(1);
        let k = 2.0 * PI * 1000.0 / C;
        let p_inc = Complex64::new(1.0, 0.0);
        let exc = port.plane_wave_excitation(k, p_inc);
        // At the inlet, ∂n p - ikp = -2ik p_inc.
        let expected = Complex64::new(0.0, -2.0 * k);
        assert!(
            (exc - expected).norm() < 1e-10,
            "Excitation = {exc}, expected {expected}"
        );
    }

    #[test]
    fn test_mode_shape_evaluation() {
        let mode = DuctMode {
            indices: (1, 0),
            transverse_wavenumber: PI / 0.1,
            cutoff_frequency: 1715.0,
        };
        // At y = 0: cos(0) = 1
        assert!((mode.evaluate(0.0, 0.0, 0.1, 1.0) - 1.0).abs() < 1e-10);
        // At y = a/2: cos(π/2) = 0
        assert!(mode.evaluate(0.05, 0.0, 0.1, 1.0).abs() < 1e-10);
        // At y = a: cos(π) = -1
        assert!((mode.evaluate(0.1, 0.0, 0.1, 1.0) + 1.0).abs() < 1e-10);
    }
}
