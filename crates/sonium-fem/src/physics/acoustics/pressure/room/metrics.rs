//! Room acoustic metrics computed from impulse responses
//!
//! Given an energetic impulse response (EIR) or a pressure impulse
//! response, computes the standard ISO 3382 room acoustic parameters.
//!
//! # Metrics
//!
//! - **RT60** / **RT30** / **RT20**: Reverberation time (extrapolated)
//! - **EDT**: Early Decay Time (0 to −10 dB)
//! - **C80**: Clarity index for music (early/late energy ratio)
//! - **D50**: Definition for speech
//! - **Ts**: Centre time
//! - **G**: Sound strength (relative to free field at 10 m)
//!
//! # Reference
//!
//! ISO 3382-1:2009, "Acoustics — Measurement of room acoustic parameters."

// ---------------------------------------------------------------------------
// Energy Decay Curve
// ---------------------------------------------------------------------------

/// Compute the Energy Decay Curve (EDC) via Schroeder backward integration.
///
/// EDC(t) = ∫_t^∞ h²(τ) dτ ≈ Σ_{i≥bin(t)} h²_i Δt
///
/// The returned vector has the same length as `ir_squared` and contains
/// the cumulative energy from the end.
pub fn energy_decay_curve(ir_squared: &[f64]) -> Vec<f64> {
    let n = ir_squared.len();
    let mut edc = vec![0.0; n];
    if n == 0 {
        return edc;
    }
    edc[n - 1] = ir_squared[n - 1];
    for i in (0..n - 1).rev() {
        edc[i] = edc[i + 1] + ir_squared[i];
    }
    edc
}

/// Convert EDC to dB (normalised to the initial value).
pub fn edc_to_db(edc: &[f64]) -> Vec<f64> {
    if edc.is_empty() || edc[0] <= 0.0 {
        return vec![0.0; edc.len()];
    }
    let ref_val = edc[0];
    edc.iter()
        .map(|&e| 10.0 * (e / ref_val).max(1e-30).log10())
        .collect()
}

// ---------------------------------------------------------------------------
// Reverberation time
// ---------------------------------------------------------------------------

/// Estimate the reverberation time by linear regression on the EDC (dB).
///
/// Fits a line to the EDC between `start_db` and `end_db` (e.g., −5 to −35
/// for RT30) and extrapolates to a 60 dB decay.
///
/// `dt` is the time resolution of the EDC bins.
///
/// Returns `None` if the EDC doesn't reach `end_db`.
pub fn reverberation_time(edc_db: &[f64], dt: f64, start_db: f64, end_db: f64) -> Option<f64> {
    if dt <= 0.0 {
        return None;
    }
    // Find the bin range for the regression
    let i_start = edc_db.iter().position(|&v| v <= start_db)?;
    let i_end = edc_db.iter().position(|&v| v <= end_db)?;
    if i_end <= i_start + 1 {
        return None;
    }

    // Linear regression: EDC_dB(t) = slope × t + intercept
    let n = (i_end - i_start + 1) as f64;
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_ty = 0.0;
    for (i, &y) in edc_db.iter().enumerate().take(i_end + 1).skip(i_start) {
        let t = i as f64 * dt;
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }
    let slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t);
    if slope >= 0.0 {
        return None; // EDC not decaying
    }

    // RT60 = −60 / slope
    Some(-60.0 / slope)
}

/// RT60 estimated from the −5 to −35 dB range (RT30, extrapolated).
pub fn rt60(edc_db: &[f64], dt: f64) -> Option<f64> {
    reverberation_time(edc_db, dt, -5.0, -35.0)
}

/// RT60 estimated from the −5 to −25 dB range (RT20, extrapolated).
pub fn rt20(edc_db: &[f64], dt: f64) -> Option<f64> {
    reverberation_time(edc_db, dt, -5.0, -25.0)
}

/// Early Decay Time: RT from 0 to −10 dB, extrapolated to 60 dB.
pub fn edt(edc_db: &[f64], dt: f64) -> Option<f64> {
    reverberation_time(edc_db, dt, 0.0, -10.0)
}

// ---------------------------------------------------------------------------
// Clarity, definition, centre time
// ---------------------------------------------------------------------------

/// Clarity index C80 \[dB\]: ratio of early (0–80 ms) to late (>80 ms) energy.
///
/// C80 = 10 log₁₀(∫₀⁸⁰ h² / ∫₈₀∞ h²)
pub fn clarity_c80(ir_squared: &[f64], dt: f64) -> f64 {
    if dt <= 0.0 {
        return f64::NAN;
    }
    let boundary_bin = (0.080 / dt).round() as usize;
    let early: f64 = ir_squared
        .iter()
        .take(boundary_bin.min(ir_squared.len()))
        .sum();
    let late: f64 = ir_squared.iter().skip(boundary_bin).sum();
    if late <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (early / late).log10()
}

/// Definition D50 \[−\]: fraction of energy in the first 50 ms.
///
/// D50 = ∫₀⁵⁰ h² / ∫₀∞ h²
pub fn definition_d50(ir_squared: &[f64], dt: f64) -> f64 {
    if dt <= 0.0 {
        return f64::NAN;
    }
    let boundary_bin = (0.050 / dt).round() as usize;
    let early: f64 = ir_squared
        .iter()
        .take(boundary_bin.min(ir_squared.len()))
        .sum();
    let total: f64 = ir_squared.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    early / total
}

/// Centre time Ts \[s\]: energy-weighted average arrival time.
///
/// Ts = ∫₀∞ t h² / ∫₀∞ h²
pub fn centre_time(ir_squared: &[f64], dt: f64) -> f64 {
    if dt <= 0.0 {
        return f64::NAN;
    }
    let total: f64 = ir_squared.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let weighted: f64 = ir_squared
        .iter()
        .enumerate()
        .map(|(i, &e)| (i as f64 + 0.5) * dt * e)
        .sum();
    weighted / total
}

// ---------------------------------------------------------------------------
// Sabine and Eyring reverberation time (analytical)
// ---------------------------------------------------------------------------

/// Sabine reverberation time: RT60 = 0.161 V / A.
///
/// `absorption_area` A = Σ αᵢ Sᵢ \[m²\] (total absorption area).
/// `volume` V \[m³\].
/// `air_absorption_4m` = 4mV where m is the air absorption coefficient \[1/m\].
pub fn rt60_sabine(volume: f64, absorption_area: f64, air_absorption_4mv: f64) -> f64 {
    0.161 * volume / (absorption_area + air_absorption_4mv)
}

/// Eyring reverberation time: RT60 = 0.161 V / (−S ln(1−ᾱ) + 4mV).
///
/// `mean_alpha` ᾱ = A / S (average absorption coefficient), must be in \[0, 1).
/// `total_surface` S \[m²\].
///
/// Returns 0.0 if `mean_alpha >= 1.0` (fully absorbing room has zero RT).
pub fn rt60_eyring(
    volume: f64,
    total_surface: f64,
    mean_alpha: f64,
    air_absorption_4mv: f64,
) -> f64 {
    if mean_alpha >= 1.0 {
        return 0.0;
    }
    let eyring_absorption = -total_surface * (1.0 - mean_alpha).ln();
    let denominator = eyring_absorption + air_absorption_4mv;
    if denominator <= 0.0 {
        return f64::INFINITY;
    }
    0.161 * volume / denominator
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize an exponentially decaying IR: h²(t) = exp(−t/τ).
    fn exponential_ir(dt: f64, duration: f64, tau: f64) -> Vec<f64> {
        let n = (duration / dt) as usize;
        (0..n).map(|i| (-(i as f64 * dt) / tau).exp()).collect()
    }

    #[test]
    fn test_edc_monotone_decreasing() {
        let ir = exponential_ir(0.001, 2.0, 0.5);
        let edc = energy_decay_curve(&ir);
        for i in 1..edc.len() {
            assert!(edc[i] <= edc[i - 1], "EDC must be monotonically decreasing");
        }
    }

    #[test]
    fn test_rt60_from_exponential() {
        // For h²(t) = exp(−t/τ), the EDC also decays exponentially.
        // RT60 = τ × ln(10⁶) = τ × 13.816
        // But RT60 from 60 dB of EDC decay: EDC(t) ∝ exp(−t/τ)
        // 10 log₁₀(exp(−t/τ)) = −(10/ln10)(t/τ) dB
        // 60 dB decay: t = 60 × ln(10)/10 × τ = 13.816 τ
        let tau = 0.1; // 100 ms time constant
        let ir = exponential_ir(0.001, 2.0, tau);
        let edc = energy_decay_curve(&ir);
        let edc_db = edc_to_db(&edc);
        let rt = rt60(&edc_db, 0.001);
        assert!(rt.is_some());
        let expected = tau * 60.0 * (10.0_f64.ln()) / 10.0; // ≈ 1.38 s
        let rt = rt.unwrap();
        assert!(
            (rt - expected).abs() / expected < 0.05,
            "RT60 = {rt:.3}, expected {expected:.3}"
        );
    }

    #[test]
    fn test_edt() {
        let tau = 0.1;
        let ir = exponential_ir(0.001, 2.0, tau);
        let edc = energy_decay_curve(&ir);
        let edc_db = edc_to_db(&edc);
        let e = edt(&edc_db, 0.001);
        assert!(e.is_some());
        // For exponential decay, EDT = RT60
        let rt = rt60(&edc_db, 0.001).unwrap();
        let e = e.unwrap();
        assert!((e - rt).abs() / rt < 0.1, "EDT = {e:.3}, RT60 = {rt:.3}");
    }

    #[test]
    fn test_clarity_c80() {
        // All energy in first 80 ms: C80 → ∞
        let mut ir = vec![0.0; 1000];
        ir[0] = 1.0;
        let c80 = clarity_c80(&ir, 0.001);
        assert!(c80 > 40.0, "C80 = {c80}"); // Very high

        // Equal energy before/after 80 ms: C80 = 0 dB
        let ir2: Vec<f64> = (0..1000).map(|_| 1.0).collect();
        let c80_2 = clarity_c80(&ir2, 0.001);
        // early = 80 bins, late = 920 bins, C80 = 10 log(80/920) ≈ −10.6 dB
        assert!(c80_2 < 0.0);
    }

    #[test]
    fn test_definition_d50() {
        // All energy at t=0: D50 = 1
        let mut ir = vec![0.0; 1000];
        ir[0] = 1.0;
        let d50 = definition_d50(&ir, 0.001);
        assert!((d50 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_centre_time() {
        // Single impulse at t = 0.1 s (bin 100 with dt = 0.001)
        let mut ir = vec![0.0; 1000];
        ir[100] = 1.0;
        let ts = centre_time(&ir, 0.001);
        // Should be approximately 0.1005 (center of bin)
        assert!((ts - 0.1005).abs() < 0.001, "Ts = {ts}");
    }

    #[test]
    fn test_sabine_rt60() {
        // Concert hall: V = 10000 m³, total absorption = 1000 m², no air absorption
        let rt = rt60_sabine(10000.0, 1000.0, 0.0);
        // RT60 = 0.161 × 10000 / 1000 = 1.61 s
        assert!((rt - 1.61).abs() < 0.01, "RT60 = {rt:.2} s");
    }

    #[test]
    fn test_eyring_vs_sabine() {
        // For low absorption, Eyring ≈ Sabine
        let v = 100.0;
        let s = 150.0;
        let alpha = 0.05; // Low absorption
        let a = alpha * s;
        let sab = rt60_sabine(v, a, 0.0);
        let eyr = rt60_eyring(v, s, alpha, 0.0);
        assert!(
            (sab - eyr).abs() / sab < 0.1,
            "Sabine = {sab:.2}, Eyring = {eyr:.2}"
        );
    }

    // -- Regression tests for edge cases --

    #[test]
    fn test_rt60_returns_none_for_zero_dt() {
        let ir = exponential_ir(0.001, 1.0, 0.1);
        let edc = energy_decay_curve(&ir);
        let edc_db = edc_to_db(&edc);
        assert!(rt60(&edc_db, 0.0).is_none(), "dt=0 should return None");
    }

    #[test]
    fn test_eyring_fully_absorbing_returns_zero() {
        let rt = rt60_eyring(100.0, 150.0, 1.0, 0.0);
        assert_eq!(rt, 0.0, "Fully absorbing room should have RT=0");
    }

    #[test]
    fn test_eyring_over_absorbing_returns_zero() {
        // mean_alpha > 1.0 (physically impossible but must not NaN)
        let rt = rt60_eyring(100.0, 150.0, 1.5, 0.0);
        assert_eq!(rt, 0.0);
    }

    #[test]
    fn test_c80_returns_nan_for_zero_dt() {
        let ir = vec![1.0; 100];
        assert!(clarity_c80(&ir, 0.0).is_nan());
    }

    #[test]
    fn test_d50_returns_nan_for_zero_dt() {
        let ir = vec![1.0; 100];
        assert!(definition_d50(&ir, 0.0).is_nan());
    }

    #[test]
    fn test_centre_time_returns_nan_for_zero_dt() {
        let ir = vec![1.0; 100];
        assert!(centre_time(&ir, 0.0).is_nan());
    }
}
