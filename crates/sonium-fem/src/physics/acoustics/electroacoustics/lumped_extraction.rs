//! Lumped parameter extraction from FEM models
//!
//! Extracts equivalent circuit parameters (Thiele-Small, radiation
//! impedance, mechanical impedance) from full FEM solutions.
//!
//! # Thiele-Small parameters
//!
//! The complete set of small-signal driver parameters:
//! - **fs**: resonance frequency
//! - **Qms**, **Qes**, **Qts**: mechanical, electrical, total Q factors
//! - **Vas**: equivalent compliance volume
//! - **Mms**, **Kms**, **Rms**: moving mass, stiffness, resistance
//! - **BL**: force factor
//! - **Sd**: effective radiating area
//! - **Re**, **Le**: DC resistance, inductance
//!
//! These can be computed directly from FEM model properties or extracted
//! by fitting the electrical impedance curve.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Thiele-Small parameters
// ---------------------------------------------------------------------------

/// Complete Thiele-Small parameter set for a loudspeaker driver.
#[derive(Debug, Clone, Copy)]
pub struct ThieleSmallParameters {
    /// Resonance frequency \[Hz\].
    pub fs: f64,
    /// Mechanical Q factor (Qms = 2π fs Mms / Rms).
    pub qms: f64,
    /// Electrical Q factor (Qes = 2π fs Mms Re / BL²).
    pub qes: f64,
    /// Total Q factor (1/Qts = 1/Qms + 1/Qes).
    pub qts: f64,
    /// Equivalent compliance volume \[m³\].
    pub vas: f64,
    /// Moving mass \[kg\].
    pub mms: f64,
    /// Mechanical stiffness \[N/m\].
    pub kms: f64,
    /// Mechanical resistance \[kg/s\].
    pub rms: f64,
    /// Force factor \[T·m\].
    pub bl: f64,
    /// Effective radiating area \[m²\].
    pub sd: f64,
    /// DC resistance \[Ω\].
    pub re: f64,
    /// Voice coil inductance \[H\].
    pub le: f64,
}

/// Compute Thiele-Small parameters from FEM model properties.
///
/// This computes the parameters directly from the mechanical modal
/// properties, force factor, and electrical parameters — no curve
/// fitting required.
#[allow(clippy::too_many_arguments)]
pub fn compute_thiele_small(
    mms: f64,
    kms: f64,
    rms: f64,
    bl: f64,
    sd: f64,
    re: f64,
    le: f64,
    rho: f64,
    c: f64,
) -> ThieleSmallParameters {
    let omega_s = (kms / mms).sqrt();
    let fs = omega_s / (2.0 * PI);
    let qms = omega_s * mms / rms;
    let qes = omega_s * mms * re / (bl * bl);
    let qts = qms * qes / (qms + qes);
    let vas = rho * c * c * sd * sd / kms;

    ThieleSmallParameters {
        fs,
        qms,
        qes,
        qts,
        vas,
        mms,
        kms,
        rms,
        bl,
        sd,
        re,
        le,
    }
}

/// Extract Thiele-Small parameters from an electrical impedance curve.
///
/// Uses the three-point method around the impedance peak:
/// 1. Find the resonance peak (maximum |Z|)
/// 2. Find the half-power frequencies f₁, f₂
/// 3. Compute Q factors from the bandwidth
///
/// Returns `None` if the impedance curve doesn't have a clear resonance.
pub fn extract_from_impedance(
    frequencies: &[f64],
    impedance: &[Complex64],
    re: f64,
    bl: f64,
    sd: f64,
    rho: f64,
    c: f64,
) -> Option<ThieleSmallParameters> {
    if frequencies.len() < 3 {
        return None;
    }

    // Find resonance peak
    let (peak_idx, _) = impedance
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm().total_cmp(&b.norm()))?;

    if peak_idx == 0 || peak_idx >= frequencies.len() - 1 {
        return None;
    }

    let fs = frequencies[peak_idx];
    let r_peak = impedance[peak_idx].re;

    // Peak resistance must exceed DC resistance for a valid resonance
    if r_peak <= re {
        return None;
    }

    // Half-power level: |Z| at which the resistive part equals the geometric mean
    let z_half = (re * r_peak).sqrt();

    // Find f₁ (below resonance) and f₂ (above resonance) where |Z| = z_half
    let f1 = find_crossing_below(frequencies, impedance, peak_idx, z_half)?;
    let f2 = find_crossing_above(frequencies, impedance, peak_idx, z_half)?;

    let qms = fs / (f2 - f1);
    let rms_over_bl2 = (r_peak - re) / (bl * bl);
    let rms = 1.0 / rms_over_bl2; // Rms = BL² / (Rpeak - Re)
    let qes = qms * re / (r_peak - re);
    let qts = qms * qes / (qms + qes);

    let omega_s = 2.0 * PI * fs;
    let mms = qms * rms / omega_s;
    let kms = mms * omega_s * omega_s;
    let vas = rho * c * c * sd * sd / kms;
    let le = (impedance.last()?.im / (2.0 * PI * frequencies.last()?)).max(0.0);

    Some(ThieleSmallParameters {
        fs,
        qms,
        qes,
        qts,
        vas,
        mms,
        kms,
        rms,
        bl,
        sd,
        re,
        le,
    })
}

fn find_crossing_below(freqs: &[f64], z: &[Complex64], peak: usize, level: f64) -> Option<f64> {
    for i in (1..peak).rev() {
        if z[i].norm() <= level && z[i + 1].norm() > level {
            let t = (level - z[i].norm()) / (z[i + 1].norm() - z[i].norm());
            return Some(freqs[i] + t * (freqs[i + 1] - freqs[i]));
        }
    }
    None
}

fn find_crossing_above(freqs: &[f64], z: &[Complex64], peak: usize, level: f64) -> Option<f64> {
    for i in peak..z.len() - 1 {
        if z[i].norm() > level && z[i + 1].norm() <= level {
            let t = (z[i].norm() - level) / (z[i].norm() - z[i + 1].norm());
            return Some(freqs[i] + t * (freqs[i + 1] - freqs[i]));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Radiation impedance
// ---------------------------------------------------------------------------

/// Compute radiation impedance from FEM boundary solution.
///
/// Z_rad = ∫_Γ p dΓ / ∫_Γ v_n dΓ
///
/// where the integrals are over the radiating surface.
/// `pressures` and `velocities` are the FEM solution at boundary nodes,
/// `areas` are the integration weights (boundary mass diagonal).
pub fn radiation_impedance(
    pressures: &[Complex64],
    velocities: &[Complex64],
    areas: &[f64],
) -> Complex64 {
    let p_integral: Complex64 = pressures.iter().zip(areas).map(|(&p, &a)| p * a).sum();
    let v_integral: Complex64 = velocities.iter().zip(areas).map(|(&v, &a)| v * a).sum();
    p_integral / v_integral
}

/// Analytical radiation impedance of a circular piston in an infinite
/// baffle (low-frequency approximation).
///
/// Z_rad ≈ ρc/S × ((ka)²/2 + j 8ka/(3π))
///
/// where a is the piston radius, S = πa², k = ω/c.
pub fn radiation_impedance_piston_lf(radius: f64, omega: f64, rho: f64, c: f64) -> Complex64 {
    let k = omega / c;
    let ka = k * radius;
    let area = PI * radius * radius;
    let z0 = rho * c / area;
    Complex64::new(z0 * ka * ka / 2.0, z0 * 8.0 * ka / (3.0 * PI))
}

// ---------------------------------------------------------------------------
// Effective radiating area
// ---------------------------------------------------------------------------

/// Compute the effective radiating area Sd from the FEM displacement
/// field on the diaphragm.
///
/// Sd = (∫_Γ u_n dΓ) / u_ref
///
/// where u_ref is a reference displacement (e.g., the peak displacement
/// at the dome center). This gives the equivalent piston area.
pub fn effective_radiating_area(
    displacements: &[f64],
    areas: &[f64],
    reference_displacement: f64,
) -> f64 {
    let volume: f64 = displacements.iter().zip(areas).map(|(&u, &a)| u * a).sum();
    volume / reference_displacement
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const RHO: f64 = 1.204;
    const C: f64 = 343.0;

    #[test]
    fn test_compute_thiele_small() {
        let ts = compute_thiele_small(
            15e-3,  // Mms = 15 g
            3000.0, // Kms = 3000 N/m
            0.8,    // Rms = 0.8 kg/s
            5.0,    // BL = 5 T·m
            50e-4,  // Sd = 50 cm²
            6.0,    // Re = 6 Ω
            0.5e-3, // Le = 0.5 mH
            RHO, C,
        );

        // fs should be around 71 Hz: (1/2π)√(3000/0.015)
        assert!((ts.fs - 71.2).abs() < 1.0, "fs = {:.1} Hz", ts.fs);

        // Qts should be < 1 for a typical driver
        assert!(ts.qts > 0.1 && ts.qts < 2.0, "Qts = {:.2}", ts.qts);

        // Qms × Qes / (Qms + Qes) = Qts
        let qts_check = ts.qms * ts.qes / (ts.qms + ts.qes);
        assert!((ts.qts - qts_check).abs() < 1e-10, "Qts consistency");

        // Vas should be positive and reasonable (litres range)
        assert!(
            ts.vas > 1e-4 && ts.vas < 0.1,
            "Vas = {:.1} L",
            ts.vas * 1000.0
        );
    }

    #[test]
    fn test_radiation_impedance_piston() {
        let a = 0.05; // 5 cm radius
        let omega = 2.0 * PI * 100.0; // 100 Hz (low frequency)
        let z_rad = radiation_impedance_piston_lf(a, omega, RHO, C);

        // Real part (radiation resistance) should be positive
        assert!(z_rad.re > 0.0);
        // Imaginary part (radiation reactance) should be positive (mass-like)
        assert!(z_rad.im > 0.0);
        // At low frequency, reactance > resistance
        assert!(z_rad.im > z_rad.re, "Im should dominate at low f");
    }

    #[test]
    fn test_effective_area() {
        // Uniform piston displacement
        let displacements = vec![1.0; 10];
        let areas = vec![0.001; 10]; // Total area = 0.01 m²
        let sd = effective_radiating_area(&displacements, &areas, 1.0);
        assert!((sd - 0.01).abs() < 1e-10, "Sd = {} m²", sd);
    }

    #[test]
    fn test_effective_area_dome_shape() {
        // Dome: displacement decreases from center to edge
        let displacements = vec![1.0, 0.8, 0.5, 0.2, 0.0];
        let areas = vec![0.001; 5];
        let sd = effective_radiating_area(&displacements, &areas, 1.0);
        // Sd < total area (dome is less effective than piston)
        let total_area: f64 = areas.iter().sum();
        assert!(sd < total_area, "Sd = {sd}, total = {total_area}");
        assert!(sd > 0.0);
    }

    #[test]
    fn test_extract_from_impedance() {
        // Synthesize a typical impedance curve
        let re = 6.0;
        let le = 0.5e-3;
        let bl = 5.0;
        let mms = 15e-3;
        let rms = 0.8;
        let kms = 3000.0;

        let mut freqs = Vec::new();
        let mut z = Vec::new();
        for i in 1..=200 {
            let f = i as f64 * 2.0; // 2 Hz to 400 Hz
            let omega = 2.0 * PI * f;
            let z_e = Complex64::new(re, omega * le);
            let z_mech = Complex64::new(rms, omega * mms - kms / omega);
            let z_mot = Complex64::new(bl * bl, 0.0) / z_mech;
            let z_total = z_e + z_mot;
            freqs.push(f);
            z.push(z_total);
        }

        let sd = 50e-4;
        let ts = extract_from_impedance(&freqs, &z, re, bl, sd, RHO, C);
        assert!(ts.is_some(), "Should find resonance");
        let ts = ts.unwrap();

        // fs should be close to the analytical value
        let fs_analytical = (kms / mms).sqrt() / (2.0 * PI);
        assert!(
            (ts.fs - fs_analytical).abs() < 5.0,
            "fs = {:.1}, expected {:.1}",
            ts.fs,
            fs_analytical
        );
    }
}
