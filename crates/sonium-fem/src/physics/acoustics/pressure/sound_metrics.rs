//! Acoustic sound power and noise metrics
//!
//! Provides functions for computing radiated sound power, A-weighting,
//! spectral flatness, and tonality penalty — the building blocks for
//! noise-reduction objective functions in shape optimization.

use std::f64::consts::PI;

/// A-weighting filter gain at frequency `f` [Hz].
///
/// Returns the weighting factor in linear scale (not dB). Multiply by
/// pressure or pressure² to get the A-weighted value.
///
/// IEC 61672-1:2013, Equation (1).
pub fn a_weighting(f: f64) -> f64 {
    let f2 = f * f;
    let f4 = f2 * f2;

    let num = 12194.0_f64.powi(2) * f4;
    let den = (f2 + 20.6_f64.powi(2))
        * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
        * (f2 + 12194.0_f64.powi(2));

    // Normalize so that A(1000 Hz) = 1.0 (0 dB)
    let ra = num / den;
    let ra_1k = {
        let f2 = 1e6_f64;
        let f4 = 1e12_f64;
        let n = 12194.0_f64.powi(2) * f4;
        let d = (f2 + 20.6_f64.powi(2))
            * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
            * (f2 + 12194.0_f64.powi(2));
        n / d
    };

    ra / ra_1k
}

/// A-weighted sound power level from a spectrum of (frequency, power) pairs.
///
/// `spectrum` is a slice of `(f_hz, W)` where `W` is the sound power [W]
/// in each frequency band.
///
/// Returns the total A-weighted sound power level in dB re 1e-12 W.
pub fn a_weighted_swl(spectrum: &[(f64, f64)]) -> f64 {
    let w_ref = 1e-12; // reference power [W]
    let total: f64 = spectrum
        .iter()
        .map(|&(f, w)| {
            let aw = a_weighting(f);
            w * aw * aw // A-weighting applied to power (square of pressure weighting)
        })
        .sum();

    if total > 0.0 {
        10.0 * (total / w_ref).log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Spectral flatness (Wiener entropy) of a power spectrum.
///
/// Ratio of geometric mean to arithmetic mean of the spectral values.
/// Ranges from 0 (pure tone) to 1 (white noise). Higher values indicate
/// a more broadband, less tonal spectrum — the desirable direction for
/// noise reduction.
///
/// `powers` are the sound power values in each frequency band (must be > 0).
pub fn spectral_flatness(powers: &[f64]) -> f64 {
    if powers.is_empty() {
        return 0.0;
    }
    let n = powers.len() as f64;

    let log_sum: f64 = powers
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { -300.0 })
        .sum();
    let geometric_mean = (log_sum / n).exp();

    let arithmetic_mean: f64 = powers.iter().sum::<f64>() / n;

    if arithmetic_mean > 0.0 {
        (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Tonality penalty: sum of tonal prominences above the broadband floor.
///
/// For each frequency bin, computes the prominence ratio (level above local
/// average of neighbors). Bins that exceed the broadband floor by more than
/// `threshold_db` contribute to the penalty.
///
/// Returns the total penalty in dB — higher means more annoying tonal content.
pub fn tonality_penalty(powers: &[f64], threshold_db: f64) -> f64 {
    if powers.len() < 3 {
        return 0.0;
    }

    let threshold_ratio = 10.0_f64.powf(threshold_db / 10.0);
    let mut penalty = 0.0;

    for i in 1..powers.len() - 1 {
        // Local broadband floor: average of neighbors (excluding the bin itself)
        let floor = 0.5 * (powers[i - 1] + powers[i + 1]);

        if floor > 0.0 {
            let prominence = powers[i] / floor;
            if prominence > threshold_ratio {
                penalty += 10.0 * prominence.log10();
            }
        }
    }

    penalty
}

/// Sound power from pressure and area on a boundary.
///
/// `W = ∫ |p|² / (2 ρ c) dA`
///
/// For axisymmetric problems, the area element includes the 2πr factor.
/// The `p_squared_area` argument should be `∫ r |p|² dΓ` from the
/// axisymmetric boundary mass (the 2π is applied here).
pub fn sound_power_from_boundary(p_squared_area: f64, rho: f64, c: f64) -> f64 {
    2.0 * PI * p_squared_area / (2.0 * rho * c)
}

/// Sound power level in dB re 1e-12 W.
pub fn sound_power_level(power_watts: f64) -> f64 {
    if power_watts > 0.0 {
        10.0 * (power_watts / 1e-12).log10()
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_weighting_1khz() {
        let aw = a_weighting(1000.0);
        assert!(
            (aw - 1.0).abs() < 1e-6,
            "A-weighting at 1 kHz should be 0 dB, got {aw}"
        );
    }

    #[test]
    fn test_a_weighting_low_freq_attenuation() {
        let aw_50 = a_weighting(50.0);
        let aw_1k = a_weighting(1000.0);
        // 50 Hz should be attenuated ~30 dB relative to 1 kHz
        let diff_db = 20.0 * (aw_50 / aw_1k).log10();
        assert!(
            diff_db < -20.0,
            "50 Hz should be attenuated > 20 dB, got {diff_db:.1} dB"
        );
    }

    #[test]
    fn test_spectral_flatness_white_noise() {
        let powers = vec![1.0; 100]; // constant spectrum = white noise
        let sf = spectral_flatness(&powers);
        assert!(
            (sf - 1.0).abs() < 1e-10,
            "White noise flatness should be 1.0, got {sf}"
        );
    }

    #[test]
    fn test_spectral_flatness_pure_tone() {
        let mut powers = vec![1e-10; 100];
        powers[50] = 1.0; // single tone
        let sf = spectral_flatness(&powers);
        assert!(sf < 0.01, "Pure tone flatness should be near 0, got {sf}");
    }

    #[test]
    fn test_tonality_penalty_flat_spectrum() {
        let powers = vec![1.0; 20];
        let penalty = tonality_penalty(&powers, 3.0);
        assert!(
            penalty < 1e-10,
            "Flat spectrum should have zero tonality penalty, got {penalty}"
        );
    }

    #[test]
    fn test_tonality_penalty_with_tone() {
        let mut powers = vec![1.0; 20];
        powers[10] = 100.0; // 20 dB tone
        let penalty = tonality_penalty(&powers, 3.0);
        assert!(
            penalty > 10.0,
            "Spectrum with 20 dB tone should have penalty > 10 dB, got {penalty:.1}"
        );
    }
}
