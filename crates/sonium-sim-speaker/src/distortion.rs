//! Distortion analysis: THD, THD+N, and IMD extraction
//!
//! Post-processing on time-domain simulation output. Runs steady-state
//! sinusoidal (or multi-tone) simulations and extracts harmonic distortion
//! via DFT analysis.
//!
//! # THD measurement
//!
//! 1. Drive the transducer with a sinusoidal voltage at frequency f
//! 2. Wait for steady state (discard initial transient)
//! 3. Window the steady-state signal (Hann)
//! 4. DFT to extract amplitudes at f, 2f, 3f, …, Nf
//! 5. THD = √(ΣH_k² for k≥2) / H_1
//!
//! # IMD measurement
//!
//! Two-tone input V(t) = V₁·sin(2πf₁t) + V₂·sin(2πf₂t).
//! Extract intermodulation products at f₂±f₁, 2f₁±f₂, etc.

use std::f64::consts::PI;

use sonium_fem::physics::acoustics::electroacoustics::time_domain::{
    LumpedTransducerConfig, simulate_lumped_transducer,
};
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;

// ---------------------------------------------------------------------------
// THD result types
// ---------------------------------------------------------------------------

/// Mechanical displacement THD at one frequency and drive level.
///
/// This is not receiver-pressure THD. Acoustic THD requires propagating each
/// velocity harmonic through the acoustic model at its harmonic frequency.
#[derive(Debug, Clone)]
pub struct ThdResult {
    /// Test frequency (Hz).
    pub frequency: f64,
    /// Drive voltage amplitude (V peak).
    pub voltage_amplitude: f64,
    /// Fundamental amplitude (displacement, velocity, or current — depends on signal).
    pub fundamental: f64,
    /// Harmonic amplitudes: index 0 = 2nd harmonic, index 1 = 3rd, etc.
    pub harmonics: Vec<f64>,
    /// THD as a fraction: √(ΣH_k²) / H_1.
    pub thd: f64,
    /// THD in dB: 20·log10(thd).
    pub thd_db: f64,
    /// Individual harmonic distortion in dB relative to fundamental.
    pub harmonic_db: Vec<f64>,
}

/// IMD measurement result.
#[derive(Debug, Clone)]
pub struct ImdResult {
    /// Lower test frequency (Hz).
    pub f1: f64,
    /// Upper test frequency (Hz).
    pub f2: f64,
    /// Amplitude at f1.
    pub amplitude_f1: f64,
    /// Amplitude at f2.
    pub amplitude_f2: f64,
    /// Intermodulation products: (frequency_Hz, amplitude).
    pub products: Vec<(f64, f64)>,
    /// Total IMD ratio (sum of product amplitudes / geometric mean of fundamentals).
    pub imd: f64,
    /// IMD in dB.
    pub imd_db: f64,
}

// ---------------------------------------------------------------------------
// DFT utilities
// ---------------------------------------------------------------------------

/// Extract amplitude at a specific frequency from a time-domain signal using DFT.
///
/// Uses a single-bin Goertzel-like computation: correlate with sin/cos at the
/// target frequency, then compute magnitude. The Hann window is applied to
/// reduce spectral leakage.
pub fn dft_amplitude(signal: &[f64], dt: f64, target_freq: f64) -> f64 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }
    let omega = 2.0 * PI * target_freq;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut window_sum = 0.0;

    for (k, &x) in signal.iter().enumerate() {
        let t = k as f64 * dt;
        // Hann window
        let w = 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos());
        window_sum += w;
        let xw = x * w;
        re += xw * (omega * t).cos();
        im += xw * (omega * t).sin();
    }

    // The symmetric Hann uses n-1 in its cosine denominator. Sum the actual
    // weights so short records and the degenerate two-sample window agree.
    if window_sum <= f64::EPSILON {
        return 0.0;
    }
    let scale = 2.0 / window_sum;
    (re * re + im * im).sqrt() * scale
}

/// Extract amplitudes at multiple frequencies simultaneously.
pub fn dft_amplitudes(signal: &[f64], dt: f64, frequencies: &[f64]) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&f| dft_amplitude(signal, dt, f))
        .collect()
}

#[cfg(test)]
mod amplitude_normalization_tests {
    use super::dft_amplitude;

    #[test]
    fn short_coherent_record_recovers_absolute_peak_amplitude() {
        // One sine period including both endpoints: symmetric Hann sum = 2.
        let signal = [0.0, 1.0, 0.0, -1.0, 0.0];
        assert!((dft_amplitude(&signal, 0.25, 1.0) - 1.0).abs() < 1e-12);
        assert_eq!(dft_amplitude(&[0.0, 1.0], 0.25, 1.0), 0.0);
    }
}

// ---------------------------------------------------------------------------
// THD extraction
// ---------------------------------------------------------------------------

/// Configuration for THD measurement.
pub struct ThdConfig {
    /// Transducer parameters.
    pub mms: f64,
    pub rms: f64,
    pub kms: f64,
    pub voice_coil: VoiceCoilParams,
    /// Number of harmonics to extract (default: 5 → up to 6th harmonic).
    pub n_harmonics: usize,
    /// Settling time in periods of the test frequency (default: 15).
    pub settle_periods: f64,
    /// Measurement time in periods (default: 10).
    pub measure_periods: f64,
    /// Samples per period of the test frequency (default: 200).
    pub samples_per_period: usize,
    /// Optional nonlinear compliance force.
    pub compliance_force_fn: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

impl ThdConfig {
    /// Create a THD config for a linear driver.
    pub fn new(mms: f64, rms: f64, kms: f64, voice_coil: VoiceCoilParams) -> Self {
        Self {
            mms,
            rms,
            kms,
            voice_coil,
            n_harmonics: 5,
            settle_periods: 15.0,
            measure_periods: 10.0,
            samples_per_period: 200,
            compliance_force_fn: None,
        }
    }

    /// Validate T/S, timing, and voice-coil inputs; `Err` names the first
    /// offending field instead of propagating NaN into the THD result.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [("mms", self.mms), ("rms", self.rms), ("kms", self.kms)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{field} must be finite and positive"));
            }
        }
        if !self.voice_coil.resistance.is_finite() || self.voice_coil.resistance <= 0.0 {
            return Err("voice_coil.resistance must be finite and positive".to_string());
        }
        if self.n_harmonics == 0 {
            return Err("n_harmonics must be positive".to_string());
        }
        if self.samples_per_period == 0 {
            return Err("samples_per_period must be positive".to_string());
        }
        if !self.settle_periods.is_finite() || self.settle_periods < 0.0 {
            return Err("settle_periods must be finite and non-negative".to_string());
        }
        if !self.measure_periods.is_finite() || self.measure_periods <= 0.0 {
            return Err("measure_periods must be finite and positive".to_string());
        }
        Ok(())
    }
}

/// Measure THD at a single frequency and voltage level.
///
/// Consumes the config to take ownership of the voice coil parameters
/// (which may contain nonlinear closures that cannot be cloned).
pub fn measure_thd(config: ThdConfig, frequency: f64, voltage_amplitude: f64) -> ThdResult {
    if !frequency.is_finite() || frequency <= 0.0 || config.validate().is_err() {
        return ThdResult {
            frequency,
            voltage_amplitude,
            fundamental: f64::NAN,
            harmonics: Vec::new(),
            thd: f64::NAN,
            thd_db: f64::NAN,
            harmonic_db: Vec::new(),
        };
    }
    let omega = 2.0 * PI * frequency;
    let dt = 1.0 / (frequency * config.samples_per_period as f64);
    let n_settle = (config.settle_periods * config.samples_per_period as f64) as usize;
    let n_measure = (config.measure_periods * config.samples_per_period as f64) as usize;
    let total_steps = n_settle + n_measure;
    let n_harmonics = config.n_harmonics;

    let sim_config = LumpedTransducerConfig {
        mms: config.mms,
        rms: config.rms,
        kms: config.kms,
        voice_coil: config.voice_coil,
        compliance_force_fn: config.compliance_force_fn,
        voltage_fn: Box::new(move |t| voltage_amplitude * (omega * t).sin()),
        beta: 0.25,
        gamma: 0.5,
        dt,
        num_steps: total_steps,
    };

    let result = simulate_lumped_transducer(&sim_config);

    // Extract the measurement window (displacement signal)
    let signal = &result.displacement[result.displacement.len() - n_measure..];

    extract_thd_from_signal(signal, dt, frequency, voltage_amplitude, n_harmonics)
}

/// Measure THD with full control over the transducer config.
///
/// Builds a `LumpedTransducerConfig` from the given parameters, runs the
/// time-domain simulation, and extracts THD from the steady-state displacement.
/// Use this when you need nonlinear BL/Le or nonlinear compliance.
pub fn measure_thd_from_config(
    sim: LumpedTransducerConfig,
    frequency: f64,
    voltage_amplitude: f64,
    n_harmonics: usize,
) -> ThdResult {
    let result = simulate_lumped_transducer(&sim);
    // Use the last half of the simulation as measurement window
    let n_measure = result.displacement.len() / 2;
    let signal = &result.displacement[result.displacement.len() - n_measure..];
    extract_thd_from_signal(signal, sim.dt, frequency, voltage_amplitude, n_harmonics)
}

/// Extract THD from a windowed signal segment.
fn extract_thd_from_signal(
    signal: &[f64],
    dt: f64,
    frequency: f64,
    voltage_amplitude: f64,
    n_harmonics: usize,
) -> ThdResult {
    // Extract harmonic amplitudes via DFT
    let frequencies: Vec<f64> = (1..=n_harmonics + 1)
        .map(|k| k as f64 * frequency)
        .collect();

    let amplitudes = dft_amplitudes(signal, dt, &frequencies);

    let fundamental = amplitudes[0];
    let harmonics: Vec<f64> = amplitudes[1..].to_vec();

    // THD = sqrt(sum of squared harmonics) / fundamental
    let harmonic_power: f64 = harmonics.iter().map(|h| h * h).sum();
    let thd = if fundamental > 1e-30 {
        harmonic_power.sqrt() / fundamental
    } else {
        0.0
    };
    let thd_db = if thd > 1e-30 {
        20.0 * thd.log10()
    } else {
        -200.0
    };

    let harmonic_db: Vec<f64> = harmonics
        .iter()
        .map(|&h| {
            if fundamental > 1e-30 && h > 1e-30 {
                20.0 * (h / fundamental).log10()
            } else {
                -200.0
            }
        })
        .collect();

    ThdResult {
        frequency,
        voltage_amplitude,
        fundamental,
        harmonics,
        thd,
        thd_db,
        harmonic_db,
    }
}

// ---------------------------------------------------------------------------
// THD sweeps
// ---------------------------------------------------------------------------

/// Measure THD across a range of frequencies at a fixed voltage.
///
/// `config_factory` creates a fresh `ThdConfig` for each measurement point
/// (necessary because the config is consumed per measurement).
pub fn thd_vs_frequency<F: Fn() -> ThdConfig>(
    config_factory: F,
    frequencies: &[f64],
    voltage_amplitude: f64,
) -> Vec<ThdResult> {
    frequencies
        .iter()
        .map(|&f| measure_thd(config_factory(), f, voltage_amplitude))
        .collect()
}

/// Measure THD at a fixed frequency across a range of voltage levels.
pub fn thd_vs_level<F: Fn() -> ThdConfig>(
    config_factory: F,
    frequency: f64,
    voltage_amplitudes: &[f64],
) -> Vec<ThdResult> {
    voltage_amplitudes
        .iter()
        .map(|&v| measure_thd(config_factory(), frequency, v))
        .collect()
}

// ---------------------------------------------------------------------------
// IMD measurement
// ---------------------------------------------------------------------------

/// Measure intermodulation distortion with a two-tone signal.
///
/// Drives with V(t) = V₁·sin(2πf₁t) + V₂·sin(2πf₂t) and extracts
/// products at f₂-f₁, f₂+f₁, 2f₁-f₂, 2f₂-f₁.
/// Consumes the config to take ownership of voice coil parameters.
pub fn measure_imd(config: ThdConfig, f1: f64, f2: f64, v1: f64, v2: f64) -> ImdResult {
    assert!(f2 > f1, "f2 must be greater than f1");
    let omega1 = 2.0 * PI * f1;
    let omega2 = 2.0 * PI * f2;

    // Use the higher frequency for timing
    let dt = 1.0 / (f2 * config.samples_per_period as f64);
    let n_settle = (config.settle_periods / f1 / dt) as usize;
    let n_measure = (config.measure_periods / f1 / dt) as usize;
    let total_steps = n_settle + n_measure;

    let sim_config = LumpedTransducerConfig {
        mms: config.mms,
        rms: config.rms,
        kms: config.kms,
        voice_coil: config.voice_coil,
        compliance_force_fn: config.compliance_force_fn,
        voltage_fn: Box::new(move |t| v1 * (omega1 * t).sin() + v2 * (omega2 * t).sin()),
        beta: 0.25,
        gamma: 0.5,
        dt,
        num_steps: total_steps,
    };

    let result = simulate_lumped_transducer(&sim_config);
    let signal = &result.displacement[result.displacement.len() - n_measure..];

    // Extract fundamentals
    let amp_f1 = dft_amplitude(signal, dt, f1);
    let amp_f2 = dft_amplitude(signal, dt, f2);

    // Extract intermodulation products
    let imd_freqs = [
        f2 - f1,       // difference
        f2 + f1,       // sum
        2.0 * f1 - f2, // 3rd order lower
        2.0 * f2 - f1, // 3rd order upper
    ];

    let products: Vec<(f64, f64)> = imd_freqs
        .iter()
        .filter(|&&f| f > 0.0 && f < 0.5 / dt) // only positive frequencies below Nyquist
        .map(|&f| (f, dft_amplitude(signal, dt, f)))
        .collect();

    let product_power: f64 = products.iter().map(|(_, a)| a * a).sum();
    let ref_amplitude = (amp_f1 * amp_f2).sqrt(); // geometric mean
    let imd = if ref_amplitude > 1e-30 {
        product_power.sqrt() / ref_amplitude
    } else {
        0.0
    };
    let imd_db = if imd > 1e-30 {
        20.0 * imd.log10()
    } else {
        -200.0
    };

    ImdResult {
        f1,
        f2,
        amplitude_f1: amp_f1,
        amplitude_f2: amp_f2,
        products,
        imd,
        imd_db,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_driver() -> ThdConfig {
        ThdConfig::new(
            0.015,  // Mms
            1.0,    // Rms
            2400.0, // Kms
            VoiceCoilParams::linear(6.0, 0.5e-3, 5.0),
        )
    }

    #[test]
    fn test_thd_config_validate_accepts_linear_driver() {
        assert!(linear_driver().validate().is_ok());
    }

    #[test]
    fn test_thd_config_validate_names_bad_fields() {
        let mut config = linear_driver();
        config.mms = -0.015;
        assert!(config.validate().unwrap_err().contains("mms"));
        let mut config = linear_driver();
        config.samples_per_period = 0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("samples_per_period")
        );
        let mut config = linear_driver();
        config.measure_periods = 0.0;
        assert!(config.validate().unwrap_err().contains("measure_periods"));
    }

    #[test]
    fn test_measure_thd_rejects_invalid_config_with_nan() {
        let mut config = linear_driver();
        config.rms = f64::NAN;
        let result = measure_thd(config, 100.0, 1.0);
        assert!(result.fundamental.is_nan());
        assert!(result.thd.is_nan());
    }

    #[test]
    fn test_dft_short_signals_return_zero_not_nan() {
        // `n == 0` and `n == 1` have no spectral content; the Hann window
        // (`0/0` for `n == 1`) must not leak NaN into the amplitude.
        assert_eq!(dft_amplitude(&[], 1.0 / 10000.0, 100.0), 0.0);
        assert_eq!(dft_amplitude(&[0.5], 1.0 / 10000.0, 100.0), 0.0);
        assert_eq!(dft_amplitudes(&[0.5], 1.0 / 10000.0, &[100.0]), vec![0.0]);
    }

    #[test]
    fn test_dft_pure_sine() {
        // DFT of a pure sine should recover the amplitude
        let f = 100.0;
        let dt = 1.0 / 10000.0;
        let n = 10000;
        let amplitude = 0.5;
        let signal: Vec<f64> = (0..n)
            .map(|k| amplitude * (2.0 * PI * f * k as f64 * dt).sin())
            .collect();

        let measured = dft_amplitude(&signal, dt, f);
        let error = (measured - amplitude).abs() / amplitude;
        assert!(
            error < 0.01,
            "DFT amplitude: measured={measured:.4}, expected={amplitude:.4} (error {:.1}%)",
            error * 100.0
        );
    }

    #[test]
    fn test_dft_two_tones() {
        // Two tones: should extract each correctly
        let f1 = 100.0;
        let f2 = 300.0;
        let dt = 1.0 / 10000.0;
        let n = 10000;
        let a1 = 1.0;
        let a2 = 0.3;
        let signal: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                a1 * (2.0 * PI * f1 * t).sin() + a2 * (2.0 * PI * f2 * t).sin()
            })
            .collect();

        let m1 = dft_amplitude(&signal, dt, f1);
        let m2 = dft_amplitude(&signal, dt, f2);

        assert!(
            ((m1 - a1) / a1).abs() < 0.02,
            "f1: measured={m1:.4}, expected={a1:.4}"
        );
        assert!(
            ((m2 - a2) / a2).abs() < 0.02,
            "f2: measured={m2:.4}, expected={a2:.4}"
        );
    }

    #[test]
    fn test_dft_no_leakage_at_other_freq() {
        // Pure 100 Hz sine: DFT at 200 Hz should be ~zero
        let f = 100.0;
        let dt = 1.0 / 10000.0;
        let n = 10000;
        let signal: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * f * k as f64 * dt).sin())
            .collect();

        let leaked = dft_amplitude(&signal, dt, 200.0);
        assert!(
            leaked < 0.01,
            "Leakage at 200 Hz from 100 Hz sine: {leaked:.4e}"
        );
    }

    #[test]
    fn test_linear_system_low_thd() {
        // A perfectly linear system should have negligible THD
        let config = linear_driver();
        let result = measure_thd(config, 100.0, 0.1);

        assert!(
            result.thd_db < -60.0,
            "Linear system THD = {:.1} dB, expected < -60 dB",
            result.thd_db
        );
        assert!(result.fundamental > 0.0, "Fundamental should be non-zero");
    }

    fn nonlinear_thd_sim(voltage: f64, frequency: f64) -> ThdResult {
        let bl0 = 5.0;
        let x_max = 0.005;
        let omega = 2.0 * PI * frequency;
        let spp = 200_usize;
        let dt = 1.0 / (frequency * spp as f64);
        let total = (25.0 * spp as f64) as usize; // 25 periods

        let sim = LumpedTransducerConfig {
            mms: 0.015,
            rms: 1.0,
            kms: 2400.0,
            voice_coil: VoiceCoilParams::nonlinear(
                6.0,
                move |x| bl0 * (1.0 - (x / x_max).powi(2)).max(0.1),
                |_x| 0.5e-3,
                |_x| 0.0,
            ),
            compliance_force_fn: None,
            voltage_fn: Box::new(move |t| voltage * (omega * t).sin()),
            beta: 0.25,
            gamma: 0.5,
            dt,
            num_steps: total,
        };
        measure_thd_from_config(sim, frequency, voltage, 5)
    }

    #[test]
    fn test_nonlinear_bl_increases_thd() {
        // Nonlinear BL(x) = BL0*(1 - x²/x_max²) should produce harmonics
        let config_lin = linear_driver();
        let thd_lin = measure_thd(config_lin, 80.0, 0.5);
        let thd_nl = nonlinear_thd_sim(0.5, 80.0);

        assert!(
            thd_nl.thd > thd_lin.thd,
            "Nonlinear THD ({:.1} dB) should exceed linear ({:.1} dB)",
            thd_nl.thd_db,
            thd_lin.thd_db
        );
        assert!(
            thd_nl.thd_db > -80.0,
            "Nonlinear THD should be measurable: {:.1} dB",
            thd_nl.thd_db
        );
    }

    #[test]
    fn test_thd_increases_with_level() {
        // For a nonlinear system, THD should increase with drive level
        let thd_low = nonlinear_thd_sim(0.1, 80.0);
        let thd_high = nonlinear_thd_sim(1.0, 80.0);

        assert!(
            thd_high.thd > thd_low.thd,
            "Higher drive should produce more THD: low={:.1}dB, high={:.1}dB",
            thd_low.thd_db,
            thd_high.thd_db
        );
    }

    #[test]
    fn test_imd_linear_system() {
        // Linear system should have minimal IMD products
        let config = linear_driver();
        let result = measure_imd(config, 60.0, 7000.0, 0.1, 0.1);

        assert!(result.amplitude_f1 > 0.0);
        assert!(result.amplitude_f2 > 0.0);
        assert!(
            result.imd_db < -50.0,
            "Linear IMD = {:.1} dB, expected < -50 dB",
            result.imd_db
        );
    }

    #[test]
    fn test_thd_result_harmonics_count() {
        let config = linear_driver();
        let result = measure_thd(config, 100.0, 0.1);

        // Default n_harmonics = 5 → harmonics[0..5] = H2..H6
        assert_eq!(result.harmonics.len(), 5);
        assert_eq!(result.harmonic_db.len(), 5);
    }
}
