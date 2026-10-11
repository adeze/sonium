//! Modal vibration and fatigue post-processing.
//!
//! Evaluates frequency-response functions and random-vibration RMS on
//! existing (mass-normalized) eigenmodes, plus stress-life utilities:
//! Basquin S-N curves with Goodman mean correction, ASTM E1049
//! four-point rainflow counting, Palmgren-Miner accumulation, and
//! force-life damage for spotweld nuggets driven by joint resultants.
//!
//! Spectral densities are one-sided throughout (`unit^2/Hz` over
//! `f >= 0`); S-N curves take stress amplitude with an optional mean.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use thiserror::Error;

/// Failure of a vibration or fatigue evaluation.
#[derive(Debug, Error)]
pub enum VibrationFatigueError {
    /// Input slices disagree in length.
    #[error("post dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Observed length.
        actual: usize,
    },
    /// A parameter is out of range.
    #[error("invalid post-processing parameter: {0}")]
    InvalidParameter(String),
    /// An input series is empty.
    #[error("post-processing input {name} must not be empty")]
    EmptyInput {
        /// Offending input name.
        name: String,
    },
}

/// Modal receptance `H(w)` between two DOFs.
///
/// Sums `phi_in * phi_out / (wr^2 - w^2 + 2 i zr wr w)` over modes
/// with mass-normalized shapes, natural frequencies `wr`, and damping
/// ratios `zr` (modal superposition, proportional damping).
///
/// # Errors
///
/// Returns [`VibrationFatigueError`] for mismatched lengths, empty
/// modes, or non-positive natural frequencies.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::modal_receptance;
///
/// let h = modal_receptance(&[0.0], &[10.0], &[(1.0, 1.0)], &[0.02]).unwrap();
/// assert!((h[0].re - 0.01).abs() < 1e-15);
/// ```
pub fn modal_receptance(
    omegas_rad_s: &[f64],
    natural_freqs_rad_s: &[f64],
    mode_pairs: &[(f64, f64)],
    damping_ratios: &[f64],
) -> Result<Vec<Complex64>, VibrationFatigueError> {
    let modes = natural_freqs_rad_s.len();
    if modes == 0 {
        return Err(VibrationFatigueError::EmptyInput {
            name: "modes".to_string(),
        });
    }
    if mode_pairs.len() != modes || damping_ratios.len() != modes {
        return Err(VibrationFatigueError::DimensionMismatch {
            expected: modes,
            actual: mode_pairs.len().max(damping_ratios.len()),
        });
    }
    for (r, wr) in natural_freqs_rad_s.iter().enumerate() {
        if !wr.is_finite() || *wr <= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(format!(
                "natural frequency {r} must be positive"
            )));
        }
        let zeta = damping_ratios[r];
        if !zeta.is_finite() || zeta < 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(format!(
                "damping ratio {r} must be non-negative"
            )));
        }
    }
    let mut out = Vec::with_capacity(omegas_rad_s.len());
    for &omega in omegas_rad_s {
        let mut sum = Complex64::new(0.0, 0.0);
        for r in 0..modes {
            let wr = natural_freqs_rad_s[r];
            let zeta = damping_ratios[r];
            let (phi_in, phi_out) = mode_pairs[r];
            let denom = Complex64::new(wr * wr - omega * omega, 2.0 * zeta * wr * omega);
            sum += phi_in * phi_out / denom;
        }
        out.push(sum);
    }
    Ok(out)
}

/// Stationary variance under one-sided white force noise.
///
/// Closed form `S0 / (8 zeta wn^3)` for an SDOF oscillator of natural
/// frequency `wn` driven by white noise of one-sided density `S0`
/// (force units squared per Hz). The density pairs with integration
/// over Hz, matching [`psd_response_rms`]; the textbook
/// `pi S0 / (4 zeta wn^3)` form pairs the same density with
/// integration over angular frequency instead.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::sdof_white_noise_variance;
///
/// let var = sdof_white_noise_variance(1.0, 0.05, 10.0);
/// assert!((var - 1.0 / (8.0 * 0.05 * 1000.0)).abs() < 1e-18);
/// ```
pub fn sdof_white_noise_variance(one_sided_s0: f64, zeta: f64, omega_n: f64) -> f64 {
    one_sided_s0 / (8.0 * zeta * omega_n.powi(3))
}

/// Miles RMS under one-sided white base acceleration.
///
/// Closed form `sqrt(pi S0 fn / (4 zeta))` for the absolute response of
/// an SDOF oscillator of natural frequency `fn` (Hz) on a white base
/// of one-sided density `S0` (acceleration units squared per Hz).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::miles_rms;
///
/// let rms = miles_rms(1.0, 0.05, 10.0);
/// assert!((rms * rms - std::f64::consts::PI * 10.0 / 0.2).abs() < 1e-12);
/// ```
pub fn miles_rms(one_sided_psd: f64, zeta: f64, nat_freq_hz: f64) -> f64 {
    (std::f64::consts::PI * one_sided_psd * nat_freq_hz / (4.0 * zeta)).sqrt()
}

/// Response RMS from a sampled FRF and input PSD.
///
/// Integrates `|H(f)|^2 S(f)` over `f >= 0` with the trapezoid rule;
/// `psd` is the one-sided input density at `freqs_hz`.
///
/// # Errors
///
/// Returns [`VibrationFatigueError`] for mismatched or short inputs.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex64;
/// use sonium_fem::solver::vibration_fatigue::psd_response_rms;
///
/// let freqs = vec![0.0, 1.0];
/// let frf = vec![Complex64::new(1.0, 0.0); 2];
/// let rms = psd_response_rms(&freqs, &frf, &[1.0, 1.0]).unwrap();
/// assert!((rms - 1.0).abs() < 1e-15);
/// ```
pub fn psd_response_rms(
    freqs_hz: &[f64],
    frf: &[Complex64],
    psd: &[f64],
) -> Result<f64, VibrationFatigueError> {
    if freqs_hz.len() < 2 {
        return Err(VibrationFatigueError::InvalidParameter(
            "PSD integration needs at least two stations".to_string(),
        ));
    }
    if frf.len() != freqs_hz.len() || psd.len() != freqs_hz.len() {
        return Err(VibrationFatigueError::DimensionMismatch {
            expected: freqs_hz.len(),
            actual: frf.len().max(psd.len()),
        });
    }
    let mut variance = 0.0;
    for i in 0..freqs_hz.len() - 1 {
        let df = freqs_hz[i + 1] - freqs_hz[i];
        if !df.is_finite() || df <= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(
                "PSD frequencies must increase".to_string(),
            ));
        }
        let left = frf[i].norm_sqr() * psd[i];
        let right = frf[i + 1].norm_sqr() * psd[i + 1];
        variance += 0.5 * df * (left + right);
    }
    Ok(variance.max(0.0).sqrt())
}

/// Basquin stress-life curve `N = (Sa / Sf')^(1/b)`.
///
/// Amplitudes at or below the endurance limit (when set) accumulate no
/// damage; a nonzero mean is corrected with the Goodman line when the
/// ultimate tensile strength is set.
#[derive(Debug, Clone)]
pub struct BasquinCurve {
    /// Fatigue strength coefficient in stress units.
    pub fatigue_strength: f64,
    /// Fatigue strength exponent (negative, e.g. -0.1).
    pub exponent: f64,
    /// Endurance-limit amplitude for infinite life (optional).
    pub endurance_limit: Option<f64>,
    /// Ultimate tensile strength for Goodman correction (optional).
    pub ultimate_tensile: Option<f64>,
}

impl BasquinCurve {
    /// Check curve parameters.
    ///
    /// # Errors
    ///
    /// Returns [`VibrationFatigueError::InvalidParameter`] for a
    /// non-positive coefficient, a non-negative exponent, or negative
    /// optional limits.
    pub fn validate(&self) -> Result<(), VibrationFatigueError> {
        if !self.fatigue_strength.is_finite() || self.fatigue_strength <= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(
                "fatigue strength must be positive".to_string(),
            ));
        }
        if !self.exponent.is_finite() || self.exponent >= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(
                "Basquin exponent must be negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Cycles to failure for an amplitude/mean pair.
    ///
    /// Applies the Goodman correction `Sa / (1 - Sm / Sut)` when both
    /// a mean and an ultimate strength are present; returns infinity
    /// below the endurance limit.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::solver::vibration_fatigue::BasquinCurve;
    ///
    /// let curve = BasquinCurve {
    ///     fatigue_strength: 1000.0,
    ///     exponent: -0.1,
    ///     endurance_limit: None,
    ///     ultimate_tensile: None,
    /// };
    /// assert!((curve.cycles_to_failure(1000.0, 0.0) - 1.0).abs() < 1e-12);
    /// assert!((curve.cycles_to_failure(500.0, 0.0) - 1024.0).abs() < 1e-9);
    /// ```
    pub fn cycles_to_failure(&self, amplitude: f64, mean: f64) -> f64 {
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return f64::INFINITY;
        }
        if let Some(limit) = self.endurance_limit
            && amplitude <= limit
        {
            return f64::INFINITY;
        }
        let mut corrected = amplitude;
        if mean != 0.0
            && let Some(uts) = self.ultimate_tensile
        {
            let margin = 1.0 - mean / uts;
            if margin <= 0.0 {
                return 0.5;
            }
            corrected = amplitude / margin;
        }
        (corrected / self.fatigue_strength).powf(1.0 / self.exponent)
    }
}

/// One rainflow-counted load cycle.
#[derive(Debug, Clone, Copy)]
pub struct RainflowCycle {
    /// Peak-to-valley range.
    pub range: f64,
    /// Cycle mean value.
    pub mean: f64,
    /// Full (1.0) or residue half (0.5) cycle weight.
    pub count: f64,
}

/// Extract reversal points (peaks and valleys) from a series.
///
/// Keeps the endpoints and every strict turning point; flat plateaus
/// collapse to one point.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::extract_reversals;
///
/// assert_eq!(extract_reversals(&[0.0, 1.0, 0.5, 0.5, 1.5]), vec![0.0, 1.0, 0.5, 1.5]);
/// ```
pub fn extract_reversals(series: &[f64]) -> Vec<f64> {
    let mut points = Vec::new();
    for &value in series {
        if points.last() == Some(&value) {
            continue;
        }
        points.push(value);
    }
    if points.len() < 2 {
        return points;
    }
    let mut reversals = vec![points[0]];
    let mut direction = 0.0;
    for window in points.windows(2) {
        let slope = (window[1] - window[0]).signum();
        if slope == 0.0 {
            continue;
        }
        if direction != 0.0 && slope != direction {
            reversals.push(window[0]);
        }
        direction = slope;
    }
    reversals.push(*points.last().unwrap());
    reversals
}

/// Count rainflow cycles with the ASTM E1049 four-point rule.
///
/// Full cycles close when the middle range of four consecutive
/// reversals is the smallest; the leftover residue counts as half
/// cycles between consecutive points.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::rainflow_counts;
///
/// let cycles = rainflow_counts(&[0.0, 1.0, -1.0, 1.0, -1.0, 0.0]);
/// let full: f64 = cycles.iter().filter(|c| c.count == 1.0).map(|c| c.count).sum();
/// assert!((full - 1.0).abs() < 1e-12);
/// ```
pub fn rainflow_counts(series: &[f64]) -> Vec<RainflowCycle> {
    let mut cycles = Vec::new();
    let mut stack: Vec<f64> = Vec::new();
    for &point in &extract_reversals(series) {
        stack.push(point);
        while stack.len() >= 4 {
            let n = stack.len();
            let (a, b, c, d) = (stack[n - 4], stack[n - 3], stack[n - 2], stack[n - 1]);
            let (x, y, z) = ((a - b).abs(), (b - c).abs(), (c - d).abs());
            if y <= x && y <= z {
                cycles.push(RainflowCycle {
                    range: y,
                    mean: (b + c) / 2.0,
                    count: 1.0,
                });
                stack.remove(n - 3);
                stack.remove(n - 3);
            } else {
                break;
            }
        }
    }
    for window in stack.windows(2) {
        cycles.push(RainflowCycle {
            range: (window[0] - window[1]).abs(),
            mean: (window[0] + window[1]) / 2.0,
            count: 0.5,
        });
    }
    cycles
}

/// Palmgren-Miner damage for counted cycles on a Basquin curve.
///
/// Sums `count / N(range / 2, mean)`; infinite-life cycles add zero.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::{BasquinCurve, RainflowCycle, miner_damage};
///
/// let curve = BasquinCurve {
///     fatigue_strength: 1000.0,
///     exponent: -0.1,
///     endurance_limit: None,
///     ultimate_tensile: None,
/// };
/// let cycles = vec![RainflowCycle { range: 1000.0, mean: 0.0, count: 512.0 }];
/// assert!((miner_damage(&cycles, &curve) - 0.5).abs() < 1e-12);
/// ```
pub fn miner_damage(cycles: &[RainflowCycle], curve: &BasquinCurve) -> f64 {
    cycles
        .iter()
        .map(|cycle| {
            let life = curve.cycles_to_failure(cycle.range / 2.0, cycle.mean);
            if life.is_finite() && life > 0.0 {
                cycle.count / life
            } else if life == 0.5 {
                // Goodman margin exhausted: half a cycle already fails.
                cycle.count / 0.5
            } else {
                0.0
            }
        })
        .sum()
}

/// Force-life curve for spotweld nuggets, `N = (Fa / F0)^(1/b)`.
///
/// Spotweld fatigue follows resultant-force ranges (joint resultants
/// from `Spotweld::check`) rather than continuum stresses.
#[derive(Debug, Clone)]
pub struct ForceLifeCurve {
    /// Force intercept in Newtons.
    pub force_intercept: f64,
    /// Fatigue exponent (negative).
    pub exponent: f64,
}

impl ForceLifeCurve {
    /// Check curve parameters.
    ///
    /// # Errors
    ///
    /// Returns [`VibrationFatigueError::InvalidParameter`] for a
    /// non-positive intercept or a non-negative exponent.
    pub fn validate(&self) -> Result<(), VibrationFatigueError> {
        if !self.force_intercept.is_finite() || self.force_intercept <= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(
                "force intercept must be positive".to_string(),
            ));
        }
        if !self.exponent.is_finite() || self.exponent >= 0.0 {
            return Err(VibrationFatigueError::InvalidParameter(
                "force-life exponent must be negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Cycles to failure for a force amplitude in Newtons.
    pub fn cycles_to_failure(&self, amplitude: f64) -> f64 {
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return f64::INFINITY;
        }
        (amplitude / self.force_intercept).powf(1.0 / self.exponent)
    }
}

/// Miner damage of a nugget force history on a force-life curve.
///
/// Rainflow-counts the resultant history and sums `count / N`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::vibration_fatigue::{ForceLifeCurve, spotweld_force_damage};
///
/// let curve = ForceLifeCurve { force_intercept: 1000.0, exponent: -0.2 };
/// let history: Vec<f64> = (0..9).map(|i| if i % 2 == 0 { 0.0 } else { 500.0 }).collect();
/// let damage = spotweld_force_damage(&history, &curve);
/// assert!(damage > 0.0 && damage.is_finite());
/// ```
pub fn spotweld_force_damage(history: &[f64], curve: &ForceLifeCurve) -> f64 {
    rainflow_counts(history)
        .iter()
        .map(|cycle| {
            let life = curve.cycles_to_failure(cycle.range / 2.0);
            if life.is_finite() && life > 0.0 {
                cycle.count / life
            } else {
                0.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receptance_matches_sdof_closed_form() {
        let omega_n = 10.0;
        let zeta = 0.02;
        let omegas = [0.0, omega_n, 2.0 * omega_n];
        let h = modal_receptance(&omegas, &[omega_n], &[(1.0, 1.0)], &[zeta]).unwrap();
        assert!((h[0].re - 1.0 / omega_n.powi(2)).abs() < 1e-15);
        assert!(h[0].im.abs() < 1e-18);
        // Resonant peak is purely imaginary with magnitude 1/(2 z w^2).
        assert!(h[1].re.abs() < 1e-15);
        assert!((h[1].im + 1.0 / (2.0 * zeta * omega_n.powi(2))).abs() < 1e-15);
        let expected = Complex64::new(1.0, 0.0)
            / Complex64::new(-3.0 * omega_n.powi(2), 4.0 * zeta * omega_n.powi(2));
        assert!((h[2] - expected).norm() < 1e-15);
    }

    #[test]
    fn numeric_psd_integration_matches_miles() {
        // Absolute-acceleration transmissibility under white base noise.
        let (zeta, freq_n) = (0.05, 10.0);
        let n = 40_001;
        let freqs: Vec<f64> = (0..n).map(|i| 200.0 * i as f64 / (n - 1) as f64).collect();
        let frf: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                let r = f / freq_n;
                Complex64::new(1.0, 2.0 * zeta * r) / Complex64::new(1.0 - r * r, 2.0 * zeta * r)
            })
            .collect();
        let psd = vec![1.0; n];
        let numeric = psd_response_rms(&freqs, &frf, &psd).unwrap();
        let closed = miles_rms(1.0, zeta, freq_n);
        assert!(
            (numeric - closed).abs() / closed < 0.01,
            "numeric {numeric} vs miles {closed}"
        );
    }

    #[test]
    fn white_noise_variance_matches_numeric_integration() {
        let (zeta, omega_n) = (0.02, 25.0);
        let n = 60_001;
        let max_f = 40.0;
        let freqs: Vec<f64> = (0..n).map(|i| max_f * i as f64 / (n - 1) as f64).collect();
        let frf: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * std::f64::consts::PI * f;
                Complex64::new(1.0, 0.0)
                    / Complex64::new(omega_n * omega_n - w * w, 2.0 * zeta * omega_n * w)
            })
            .collect();
        // Force PSD in N^2/Hz maps to displacement via receptance.
        let numeric = psd_response_rms(&freqs, &frf, &vec![2.0; n]).unwrap();
        let closed = sdof_white_noise_variance(2.0, zeta, omega_n).sqrt();
        assert!(
            (numeric - closed).abs() / closed < 0.02,
            "numeric {numeric} vs closed {closed}"
        );
    }

    #[test]
    fn rainflow_decomposes_mixed_history() {
        // Reversals [0,1,-1,1,-1,0]: one full range-2 cycle plus halves.
        let cycles = rainflow_counts(&[0.0, 1.0, -1.0, 1.0, -1.0, 0.0]);
        let full: Vec<_> = cycles.iter().filter(|c| c.count == 1.0).collect();
        assert_eq!(full.len(), 1);
        assert!((full[0].range - 2.0).abs() < 1e-15);
        assert!(full[0].mean.abs() < 1e-15);
        let mut halves: Vec<f64> = cycles
            .iter()
            .filter(|c| c.count == 0.5)
            .map(|c| c.range)
            .collect();
        halves.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(halves, vec![1.0, 1.0, 2.0]);
    }

    #[test]
    fn miner_two_block_matches_hand_sum() {
        let curve = BasquinCurve {
            fatigue_strength: 1000.0,
            exponent: -0.1,
            endurance_limit: None,
            ultimate_tensile: None,
        };
        curve.validate().unwrap();
        let cycles = vec![
            RainflowCycle {
                range: 600.0,
                mean: 0.0,
                count: 100.0,
            },
            RainflowCycle {
                range: 300.0,
                mean: 0.0,
                count: 1000.0,
            },
        ];
        let expected = 100.0 / curve.cycles_to_failure(300.0, 0.0)
            + 1000.0 / curve.cycles_to_failure(150.0, 0.0);
        assert!((miner_damage(&cycles, &curve) - expected).abs() < 1e-15);
        // Endurance cutoff zeroes sub-limit damage (both amplitudes
        // 300 and 150 sit below the 400 limit).
        let limited = BasquinCurve {
            endurance_limit: Some(400.0),
            ..curve
        };
        assert_eq!(miner_damage(&cycles, &limited), 0.0);
    }

    #[test]
    fn goodman_raises_tensile_mean_damage() {
        let curve = BasquinCurve {
            fatigue_strength: 1000.0,
            exponent: -0.1,
            endurance_limit: None,
            ultimate_tensile: Some(1200.0),
        };
        let zero_mean = curve.cycles_to_failure(300.0, 0.0);
        let tensile = curve.cycles_to_failure(300.0, 300.0);
        assert!(tensile < zero_mean);
        // Mean past ultimate fails immediately.
        assert!(curve.cycles_to_failure(300.0, 1500.0) < 1.0);
    }

    #[test]
    fn spotweld_constant_amplitude_is_exact() {
        let curve = ForceLifeCurve {
            force_intercept: 1000.0,
            exponent: -0.2,
        };
        curve.validate().unwrap();
        // Ten reversals 0/500: four full range-500 cycles plus halves.
        let history: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 0.0 } else { 500.0 })
            .collect();
        let cycles = rainflow_counts(&history);
        let expected: f64 = cycles
            .iter()
            .map(|c| c.count / curve.cycles_to_failure(c.range / 2.0))
            .sum();
        assert!((spotweld_force_damage(&history, &curve) - expected).abs() < 1e-15);
        // Constant amplitude 500 about mean 250: life per cycle is fixed.
        let life = curve.cycles_to_failure(250.0);
        let total: f64 = cycles.iter().map(|c| c.count).sum();
        assert!((expected - total / life).abs() < 1e-12);
    }
}
