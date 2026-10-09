//! Frequency-response to impulse-response transforms.
//!
//! This module owns the browser-facing IR schema, input contract, shared FFT
//! conversion, and time-domain sensitivity propagation. Room orchestration
//! supplies the complex response; no room-model assumptions belong here.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fft::rustfft_compat::FftPlanner;
use std::f64::consts::PI;

pub(crate) const MAX_IMPULSE_RESPONSE_SAMPLES: usize = 4_194_304;

/// Impulse response data structure.
#[derive(Debug, Clone, Serialize)]
pub struct ImpulseResponse {
    /// Time samples in seconds.
    pub time: Vec<f64>,
    /// Amplitude samples (normalized).
    pub amplitude: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Peak amplitude.
    pub peak_amplitude: f64,
    /// Energy decay curve (ETC) in dB.
    pub energy_decay: Vec<f64>,
    /// Optional normalized-amplitude sensitivity propagated from the complex
    /// frequency-response estimate. This is not a confidence interval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Vec<f64>>,
}

/// Configuration for impulse response generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseResponseConfig {
    /// Sample rate for output IR (default 48000 Hz).
    #[serde(default = "default_ir_sample_rate")]
    pub sample_rate: f64,
    /// IR duration in seconds (default: auto from RT60).
    #[serde(default)]
    pub duration: Option<f64>,
    /// Number of FFT points (default: power of 2 based on duration).
    #[serde(default)]
    pub fft_size: Option<usize>,
    /// Minimum frequency for spectrum (default: 20 Hz).
    #[serde(default = "default_ir_min_freq")]
    pub min_freq: f64,
    /// Maximum frequency for spectrum (default: Nyquist/2).
    #[serde(default)]
    pub max_freq: Option<f64>,
}

/// Invalid input to [`calculate_impulse_response`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpulseResponseError(pub String);

impl std::fmt::Display for ImpulseResponseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ImpulseResponseError {}

fn default_ir_sample_rate() -> f64 {
    48000.0
}

fn default_ir_min_freq() -> f64 {
    20.0
}

impl Default for ImpulseResponseConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_ir_sample_rate(),
            duration: None,
            fft_size: None,
            min_freq: default_ir_min_freq(),
            max_freq: None,
        }
    }
}

/// Calculate an impulse response from a complex frequency response.
pub fn calculate_impulse_response(
    frequencies: &[f64],
    complex_response: &[Complex64],
    config: &ImpulseResponseConfig,
) -> Result<ImpulseResponse, ImpulseResponseError> {
    calculate_impulse_response_with_uncertainty(frequencies, complex_response, None, config)
}

pub(crate) fn calculate_impulse_response_with_uncertainty(
    frequencies: &[f64],
    complex_response: &[Complex64],
    relative_uncertainty: Option<&[f64]>,
    config: &ImpulseResponseConfig,
) -> Result<ImpulseResponse, ImpulseResponseError> {
    validate_impulse_response_input(frequencies, complex_response, relative_uncertainty, config)?;
    let sample_rate = config.sample_rate;
    let nyquist = sample_rate / 2.0;
    let duration = config.duration.unwrap_or(0.5);
    let min_fft_size = (duration * sample_rate).ceil() as usize;
    let fft_size = match config.fft_size {
        Some(size) => size,
        None => min_fft_size
            .max(1024)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                ImpulseResponseError("requested impulse response is too large".into())
            })?,
    };
    let freq_resolution = sample_rate / fft_size as f64;
    let num_bins = fft_size / 2 + 1;
    let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut error_spectrum = relative_uncertainty.map(|_| vec![Complex64::new(0.0, 0.0); fft_size]);

    #[allow(clippy::needless_range_loop)]
    for bin in 0..num_bins {
        let freq = bin as f64 * freq_resolution;
        if freq < config.min_freq || freq > nyquist {
            continue;
        }
        let value = interpolate_complex(frequencies, complex_response, freq);
        spectrum[bin] = value;
        if let (Some(relative), Some(errors)) = (relative_uncertainty, &mut error_spectrum) {
            errors[bin] = value * interpolate_real(frequencies, relative, freq);
        }
    }

    for k in 1..fft_size / 2 {
        spectrum[fft_size - k] = spectrum[k].conj();
        if let Some(errors) = &mut error_spectrum {
            errors[fft_size - k] = errors[k].conj();
        }
    }

    let time_domain = ifft_real(&spectrum);
    let peak = time_domain
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let normalized = if peak > 1.0e-10 {
        time_domain.iter().map(|value| value / peak).collect()
    } else {
        time_domain.clone()
    };
    let time = (0..fft_size).map(|i| i as f64 / sample_rate).collect();
    let energy_decay = calculate_energy_decay(&normalized);
    let uncertainty = error_spectrum.map(|errors| {
        let time_error = ifft_real(&errors);
        time_error
            .into_iter()
            .map(|value| {
                if peak > 1.0e-10 {
                    value.abs() / peak
                } else {
                    value.abs()
                }
            })
            .collect()
    });

    Ok(ImpulseResponse {
        time,
        amplitude: normalized,
        sample_rate,
        duration: fft_size as f64 / sample_rate,
        peak_amplitude: peak,
        energy_decay,
        uncertainty,
    })
}

/// Conservative transform planning allowance, shared by standalone and
/// simulation admission. Invalid/overflowing automatic sizes saturate to
/// rejection; validation separately reports malformed configuration.
pub(crate) fn estimated_transform_bytes(config: &ImpulseResponseConfig) -> u128 {
    let size = config.fft_size.unwrap_or_else(|| {
        ((config.duration.unwrap_or(0.5) * config.sample_rate).ceil() as usize)
            .max(1024)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
    });
    // Complex spectra, FFT scratch, uncertainty, time/normalized/decay
    // vectors and retained output. Does not claim to bound JS heap overhead.
    (size as u128).saturating_mul(128)
}

fn validate_impulse_response_input(
    frequencies: &[f64],
    complex_response: &[Complex64],
    relative_uncertainty: Option<&[f64]>,
    config: &ImpulseResponseConfig,
) -> Result<(), ImpulseResponseError> {
    if frequencies.is_empty() {
        return Err(ImpulseResponseError(
            "frequency response must contain at least one sample".into(),
        ));
    }
    if frequencies.len() != complex_response.len() {
        return Err(ImpulseResponseError(format!(
            "frequency and response lengths differ ({} != {})",
            frequencies.len(),
            complex_response.len()
        )));
    }
    if let Some(uncertainty) = relative_uncertainty
        && (uncertainty.len() != frequencies.len()
            || uncertainty
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0))
    {
        return Err(ImpulseResponseError(
            "relative uncertainty must be finite, non-negative, and aligned with the frequency response".into(),
        ));
    }
    if frequencies
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(ImpulseResponseError(
            "frequencies must be finite and strictly positive".into(),
        ));
    }
    if frequencies.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(ImpulseResponseError(
            "frequencies must be strictly increasing".into(),
        ));
    }
    if complex_response
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ImpulseResponseError(
            "complex response must contain only finite values".into(),
        ));
    }
    if !config.sample_rate.is_finite() || config.sample_rate <= 0.0 {
        return Err(ImpulseResponseError(
            "sample rate must be finite and positive".into(),
        ));
    }
    let duration = config.duration.unwrap_or(0.5);
    let requested_samples = duration * config.sample_rate;
    if !duration.is_finite()
        || duration <= 0.0
        || !requested_samples.is_finite()
        || requested_samples > MAX_IMPULSE_RESPONSE_SAMPLES as f64
    {
        return Err(ImpulseResponseError(
            "duration must produce a finite, representable sample count".into(),
        ));
    }
    if !config.min_freq.is_finite() || config.min_freq < 0.0 {
        return Err(ImpulseResponseError(
            "minimum frequency must be finite and non-negative".into(),
        ));
    }
    if let Some(max_frequency) = config.max_freq
        && (!max_frequency.is_finite() || max_frequency <= config.min_freq)
    {
        return Err(ImpulseResponseError(
            "maximum frequency must be finite and greater than minimum frequency".into(),
        ));
    }
    if let Some(size) = config.fft_size
        && (size < 2 || !size.is_power_of_two() || size > MAX_IMPULSE_RESPONSE_SAMPLES)
    {
        return Err(ImpulseResponseError(format!(
            "FFT size must be a power of two, at least 2, and no larger than {MAX_IMPULSE_RESPONSE_SAMPLES}, got {size}"
        )));
    }
    let estimate = estimated_transform_bytes(config)
        .saturating_add((frequencies.len() as u128).saturating_mul(48));
    if estimate > super::validation::MAX_ESTIMATED_WORKING_SET_BYTES {
        return Err(ImpulseResponseError(
            "estimated impulse response working set exceeds the 256 MiB limit".into(),
        ));
    }
    Ok(())
}

fn interpolate_complex(frequencies: &[f64], values: &[Complex64], target_freq: f64) -> Complex64 {
    if frequencies.is_empty() || values.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    if target_freq <= frequencies[0] {
        return values[0];
    }
    if target_freq >= frequencies[frequencies.len() - 1] {
        return values[values.len() - 1];
    }
    for i in 0..frequencies.len() - 1 {
        if target_freq >= frequencies[i] && target_freq < frequencies[i + 1] {
            let t = (target_freq.ln() - frequencies[i].ln())
                / (frequencies[i + 1].ln() - frequencies[i].ln());
            let mag = values[i].norm() * (1.0 - t) + values[i + 1].norm() * t;
            let phase1 = values[i].arg();
            let phase2 = values[i + 1].arg();
            let mut phase_diff = phase2 - phase1;
            while phase_diff > PI {
                phase_diff -= 2.0 * PI;
            }
            while phase_diff < -PI {
                phase_diff += 2.0 * PI;
            }
            return Complex64::from_polar(mag, phase1 + t * phase_diff);
        }
    }
    values[values.len() - 1]
}

fn interpolate_real(frequencies: &[f64], values: &[f64], target_freq: f64) -> f64 {
    if frequencies.is_empty() || values.is_empty() {
        return 0.0;
    }
    if target_freq <= frequencies[0] {
        return values[0];
    }
    if target_freq >= frequencies[frequencies.len() - 1] {
        return values[values.len() - 1];
    }
    for i in 0..frequencies.len() - 1 {
        if target_freq >= frequencies[i] && target_freq < frequencies[i + 1] {
            let t = (target_freq.ln() - frequencies[i].ln())
                / (frequencies[i + 1].ln() - frequencies[i].ln());
            return values[i] * (1.0 - t) + values[i + 1] * t;
        }
    }
    values[values.len() - 1]
}

fn ifft_real(spectrum: &[Complex64]) -> Vec<f64> {
    let n = spectrum.len();
    if n == 0 {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(n);
    let mut data = spectrum.to_vec();
    plan.process(&mut data);
    data.into_iter().map(|value| value.re / n as f64).collect()
}

fn calculate_energy_decay(ir: &[f64]) -> Vec<f64> {
    if ir.is_empty() {
        return vec![];
    }
    let energy: Vec<f64> = ir.iter().map(|value| value * value).collect();
    let mut decay = vec![0.0; ir.len()];
    let mut cumsum = 0.0;
    for i in (0..ir.len()).rev() {
        cumsum += energy[i];
        decay[i] = cumsum;
    }
    let max_energy = decay[0].max(1.0e-10);
    decay
        .iter()
        .map(|value| 10.0 * (value / max_energy).log10())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_transform_is_rejected_before_allocation() {
        let frequencies = [1.0, 24_000.0];
        let response = [Complex64::new(1.0, 0.0); 2];
        for fft_size in [None, Some(MAX_IMPULSE_RESPONSE_SAMPLES)] {
            let config = ImpulseResponseConfig {
                sample_rate: 48_000.0,
                duration: Some(60.0),
                fft_size,
                min_freq: 0.0,
                max_freq: None,
            };
            let error = calculate_impulse_response(&frequencies, &response, &config).unwrap_err();
            assert!(error.to_string().contains("working set"));
            assert_eq!(estimated_transform_bytes(&config), 128 * 4_194_304);
        }
    }

    #[test]
    fn constant_full_band_response_is_a_normalized_impulse() {
        let config = ImpulseResponseConfig {
            sample_rate: 8_000.0,
            duration: Some(8.0 / 8_000.0),
            fft_size: Some(8),
            min_freq: 0.0,
            max_freq: None,
        };
        let frequencies = [1.0, 4_000.0];
        let response = [Complex64::new(1.0, 0.0); 2];

        let impulse = calculate_impulse_response(&frequencies, &response, &config)
            .expect("constant response should be valid");

        assert_eq!(impulse.time.len(), 8);
        assert_eq!(impulse.sample_rate, 8_000.0);
        assert_eq!(impulse.duration, 8.0 / 8_000.0);
        assert!((impulse.peak_amplitude - 1.0).abs() < 1.0e-12);
        assert!((impulse.amplitude[0] - 1.0).abs() < 1.0e-12);
        assert!(
            impulse.amplitude[1..]
                .iter()
                .all(|sample| sample.abs() < 1.0e-12)
        );
        assert!((impulse.time[1] - 1.0 / 8_000.0).abs() < 1.0e-15);
    }
}
