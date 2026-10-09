//! Backend-independent RoomSim metrics.
//!
//! The FEM, fast-hybrid, and future backends all return complex pressure
//! amplitudes.  This module projects those amplitudes into the small set of
//! spatial level metrics used by RoomEQ-facing consumers without depending on
//! an exporter or on any particular solver implementation.

use crate::SimulationOutput;
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

/// Acoustic reference pressure in pascals.
pub const ACOUSTIC_REFERENCE_PRESSURE_PA: f64 = 20.0e-6;

/// Finite floor used when a complex pressure is effectively zero.
const MIN_PEAK_PRESSURE_PA: f64 = 1.0e-20;

/// SPL floor used by RoomSim's finite CSV/WAV metric contract.
pub const MIN_SPL_DB: f64 = -120.0;

/// Convert a complex peak pressure amplitude to dB SPL.
///
/// Sonium's frequency-domain solvers represent the peak amplitude of the
/// harmonic pressure.  SPL is referenced to RMS pressure, so the complex
/// magnitude is divided by `sqrt(2)` before conversion.  Keeping this rule in
/// one RoomSim-owned helper prevents the high-frequency extension and output
/// exporters from drifting apart.
pub fn peak_pressure_to_spl(pressure: Complex64) -> f64 {
    let magnitude = pressure.norm();
    if magnitude > MIN_PEAK_PRESSURE_PA {
        let rms_pressure = magnitude / SQRT_2;
        20.0 * (rms_pressure / ACOUSTIC_REFERENCE_PRESSURE_PA).log10()
    } else {
        MIN_SPL_DB
    }
}

/// Convert an SPL value and phase back to a complex peak pressure amplitude.
///
/// This is the inverse of [`peak_pressure_to_spl`] for values above the finite
/// floor.  The explicit validation keeps synthetic extension code from
/// manufacturing a non-finite pressure when a caller supplies a bad metric.
pub fn spl_to_peak_pressure(spl_db: f64, phase_rad: f64) -> Result<Complex64, String> {
    if !spl_db.is_finite() || !phase_rad.is_finite() {
        return Err("SPL and phase must be finite".to_string());
    }
    let rms_pressure = ACOUSTIC_REFERENCE_PRESSURE_PA * 10.0_f64.powf(spl_db / 20.0);
    let peak_pressure = SQRT_2 * rms_pressure;
    if !peak_pressure.is_finite() {
        return Err("SPL is outside the finite pressure range".to_string());
    }
    Ok(Complex64::from_polar(peak_pressure, phase_rad))
}

/// Per-frequency spatial level metrics for one source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialMetricSample {
    /// RMS pressure across listeners, in pascals.
    pub rms_pressure_pa: f64,
    /// SPL of the spatial RMS pressure, in dB re 20 µPa.
    pub rms_spl_db: f64,
    /// Arithmetic mean of the individual listener SPL values.
    pub mean_listener_spl_db: f64,
    /// Lowest individual listener SPL.
    pub min_listener_spl_db: f64,
    /// Highest individual listener SPL.
    pub max_listener_spl_db: f64,
    /// `max_listener_spl_db - min_listener_spl_db`.
    pub listener_spread_db: f64,
}

/// A deterministic spatial metric series aligned with a response grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialMetricSeries {
    /// Frequency points in Hz.
    pub frequencies: Vec<f64>,
    /// One metric sample per frequency.
    pub samples: Vec<SpatialMetricSample>,
    /// Number of listener rows used for every sample.
    pub listener_count: usize,
}

impl SpatialMetricSeries {
    /// Reproduce spatial metrics for one source of a validated simulation.
    pub fn for_source(output: &SimulationOutput, source_index: usize) -> Result<Self, String> {
        output.validate_layout()?;
        let listener_pressures = output
            .pressures
            .get(source_index)
            .ok_or_else(|| format!("source index {source_index} is outside the output"))?;
        Self::from_listener_pressures(&output.frequencies, listener_pressures)
    }

    /// Compute metrics from listener pressure rows without using a backend or
    /// an exporter.  `listener_pressures[listener][frequency]` contains
    /// complex peak amplitudes.
    pub fn from_listener_pressures(
        frequencies: &[f64],
        listener_pressures: &[Vec<Complex64>],
    ) -> Result<Self, String> {
        validate_frequency_grid(frequencies)?;
        if listener_pressures.is_empty() {
            return Err("spatial metrics require at least one listener".to_string());
        }
        for (listener_index, pressures) in listener_pressures.iter().enumerate() {
            if pressures.len() != frequencies.len() {
                return Err(format!(
                    "pressure/frequency length mismatch for listener {listener_index}: {} samples for {} frequencies",
                    pressures.len(),
                    frequencies.len()
                ));
            }
            if pressures
                .iter()
                .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
            {
                return Err(format!("non-finite pressure for listener {listener_index}"));
            }
        }

        let samples = (0..frequencies.len())
            .map(|frequency_index| {
                let mut magnitudes: Vec<f64> = listener_pressures
                    .iter()
                    .map(|pressures| pressures[frequency_index].norm())
                    .collect();
                // Sorting makes reductions independent of listener traversal
                // order.  This matters when two backends enumerate spatial
                // probes differently but expose the same physical stencil.
                magnitudes.sort_by(f64::total_cmp);

                let mean_square_peak = magnitudes
                    .iter()
                    .map(|magnitude| magnitude.powi(2))
                    .sum::<f64>()
                    / magnitudes.len() as f64;
                let rms_peak = mean_square_peak.sqrt();
                let rms_pressure_pa = rms_peak / SQRT_2;
                let listener_spl: Vec<f64> = magnitudes
                    .iter()
                    .map(|&magnitude| peak_pressure_to_spl(Complex64::new(magnitude, 0.0)))
                    .collect();
                let mean_listener_spl_db =
                    listener_spl.iter().sum::<f64>() / listener_spl.len() as f64;
                let min_listener_spl_db = listener_spl[0];
                let max_listener_spl_db = *listener_spl.last().unwrap();
                let rms_spl_db = peak_pressure_to_spl(Complex64::new(rms_peak, 0.0));
                let listener_spread_db = max_listener_spl_db - min_listener_spl_db;

                let sample = SpatialMetricSample {
                    rms_pressure_pa,
                    rms_spl_db,
                    mean_listener_spl_db,
                    min_listener_spl_db,
                    max_listener_spl_db,
                    listener_spread_db,
                };
                if [
                    sample.rms_pressure_pa,
                    sample.rms_spl_db,
                    sample.mean_listener_spl_db,
                    sample.min_listener_spl_db,
                    sample.max_listener_spl_db,
                    sample.listener_spread_db,
                ]
                .iter()
                .all(|value| value.is_finite())
                {
                    Ok(sample)
                } else {
                    Err(format!(
                        "non-finite spatial metric at frequency index {frequency_index}"
                    ))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            frequencies: frequencies.to_vec(),
            samples,
            listener_count: listener_pressures.len(),
        })
    }
}

fn validate_frequency_grid(frequencies: &[f64]) -> Result<(), String> {
    if frequencies.is_empty() {
        return Err("spatial metrics require at least one frequency".to_string());
    }
    if frequencies
        .iter()
        .any(|&frequency| !frequency.is_finite() || frequency <= 0.0)
    {
        return Err("frequency grid must contain finite positive values".to_string());
    }
    if frequencies.windows(2).any(|window| window[1] <= window[0]) {
        return Err("frequency grid must be strictly increasing".to_string());
    }
    Ok(())
}
