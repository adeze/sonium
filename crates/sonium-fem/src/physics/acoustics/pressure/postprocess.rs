//! Acoustic post-processing: velocity, intensity, SPL, transmission loss
//!
//! Computes derived quantities from the FEM pressure solution.

use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Sound pressure level
// ---------------------------------------------------------------------------

/// Sound pressure level in dB re 20 μPa.
///
/// SPL = 20 log₁₀(|p_rms| / p_ref) where p_ref = 20 × 10⁻⁶ Pa.
pub fn sound_pressure_level(p_complex: Complex64) -> f64 {
    let p_ref = 20.0e-6;
    let p_rms = p_complex.norm() / 2.0_f64.sqrt();
    20.0 * (p_rms / p_ref).log10()
}

/// SPL from a real-valued RMS pressure.
pub fn spl_from_rms(p_rms: f64) -> f64 {
    let p_ref = 20.0e-6;
    20.0 * (p_rms / p_ref).log10()
}

// ---------------------------------------------------------------------------
// Particle velocity (frequency domain)
// ---------------------------------------------------------------------------

/// Compute particle velocity at a point from the pressure gradient.
///
/// From the linearised Euler equation (e^{−jωt}): `v = ∇p / (jωρ₀)`.
///
/// `grad_p` is the pressure gradient \[∂p/∂x, ∂p/∂y, ∂p/∂z\].
pub fn particle_velocity(grad_p: [Complex64; 3], omega: f64, density: f64) -> [Complex64; 3] {
    let factor = Complex64::new(1.0, 0.0) / (Complex64::new(0.0, omega * density));
    [factor * grad_p[0], factor * grad_p[1], factor * grad_p[2]]
}

// ---------------------------------------------------------------------------
// Acoustic intensity
// ---------------------------------------------------------------------------

/// Time-averaged acoustic intensity: I = ½ Re(p v*).
///
/// Returns the intensity vector \[I_x, I_y, I_z\] in W/m².
pub fn acoustic_intensity(pressure: Complex64, velocity: [Complex64; 3]) -> [f64; 3] {
    [
        0.5 * (pressure * velocity[0].conj()).re,
        0.5 * (pressure * velocity[1].conj()).re,
        0.5 * (pressure * velocity[2].conj()).re,
    ]
}

/// Acoustic power through a surface: W = ∫_Γ I · n dΓ.
///
/// `intensities_dot_n` contains I · n at each quadrature/node point.
/// `weights` contains the integration weights (area × quadrature weight).
pub fn acoustic_power(intensities_dot_n: &[f64], weights: &[f64]) -> f64 {
    intensities_dot_n
        .iter()
        .zip(weights)
        .map(|(&i_n, &w)| i_n * w)
        .sum()
}

// ---------------------------------------------------------------------------
// Transmission loss
// ---------------------------------------------------------------------------

/// Transmission loss in dB.
///
/// TL = 10 log₁₀(W_in / W_out)
///
/// where W_in, W_out are the acoustic powers at inlet and outlet.
pub fn transmission_loss(power_in: f64, power_out: f64) -> f64 {
    10.0 * (power_in / power_out).log10()
}

/// Transmission loss from pressure amplitudes for a plane wave in a duct.
///
/// For plane waves in a uniform duct (same cross-section at inlet and outlet):
/// TL = 20 log₁₀(|p_in| / |p_out|)
pub fn transmission_loss_plane_wave(p_in: Complex64, p_out: Complex64) -> f64 {
    20.0 * (p_in.norm() / p_out.norm()).log10()
}

// ---------------------------------------------------------------------------
// Absorption coefficient
// ---------------------------------------------------------------------------

/// Normal-incidence absorption coefficient from surface impedance.
///
/// α = 1 − |R|² where R = (Z_s − ρ₀c₀)/(Z_s + ρ₀c₀) is the reflection
/// coefficient.
pub fn absorption_coefficient(surface_impedance: Complex64, rho: f64, c: f64) -> f64 {
    let z0 = Complex64::new(rho * c, 0.0);
    let r = (surface_impedance - z0) / (surface_impedance + z0);
    1.0 - r.norm_sqr()
}

/// Normal-incidence reflection coefficient.
pub fn reflection_coefficient(surface_impedance: Complex64, rho: f64, c: f64) -> Complex64 {
    let z0 = Complex64::new(rho * c, 0.0);
    (surface_impedance - z0) / (surface_impedance + z0)
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
    fn test_spl_reference() {
        // 1 Pa RMS → 94 dB SPL
        let spl = spl_from_rms(1.0);
        assert!(
            (spl - 94.0).abs() < 0.1,
            "SPL(1 Pa) = {spl} dB, expected ~94"
        );
    }

    #[test]
    fn test_spl_complex_amplitude() {
        // |p| = √2 → p_rms = 1 → 94 dB
        let p = Complex64::new(1.0, 1.0); // |p| = √2
        let spl = sound_pressure_level(p);
        assert!((spl - 94.0).abs() < 0.1, "SPL = {spl} dB, expected ~94");
    }

    #[test]
    fn test_particle_velocity_plane_wave() {
        // Plane wave: p = P₀ e^{jkx}, grad_p = [jkP₀, 0, 0]
        let omega = 2.0 * PI * 1000.0;
        let k = omega / C;
        let p0 = Complex64::new(1.0, 0.0);
        let grad_p = [
            Complex64::new(0.0, k) * p0,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let v = particle_velocity(grad_p, omega, RHO);

        // For a plane wave: v_x = p/(ρc)
        let expected_vx = p0 / (RHO * C);
        assert!(
            (v[0] - expected_vx).norm() < 1e-10,
            "v_x = {}, expected {}",
            v[0],
            expected_vx
        );
    }

    #[test]
    fn test_intensity_plane_wave() {
        let p = Complex64::new(1.0, 0.0);
        let v = [
            p / (RHO * C),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let i = acoustic_intensity(p, v);
        // I = p²/(2ρc)
        let expected = 1.0 / (2.0 * RHO * C);
        assert!(
            (i[0] - expected).abs() < 1e-10,
            "I_x = {}, expected {}",
            i[0],
            expected
        );
    }

    #[test]
    fn test_transmission_loss() {
        let tl = transmission_loss(1.0, 0.1); // 10:1 power ratio
        assert!((tl - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_absorption_coefficient_rigid_wall() {
        // Rigid wall: Z_s → ∞, R = 1, α = 0
        let z_s = Complex64::new(1e10, 0.0);
        let alpha = absorption_coefficient(z_s, RHO, C);
        assert!(alpha < 1e-6, "α = {alpha}, expected ~0");
    }

    #[test]
    fn test_absorption_coefficient_perfect() {
        // Z_s = ρc → R = 0, α = 1
        let z_s = Complex64::new(RHO * C, 0.0);
        let alpha = absorption_coefficient(z_s, RHO, C);
        assert!((alpha - 1.0).abs() < 1e-10, "α = {alpha}, expected 1.0");
    }
}
