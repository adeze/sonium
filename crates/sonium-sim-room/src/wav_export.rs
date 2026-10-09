//! WAV export for simulation results
//!
//! Converts frequency-domain complex pressure to impulse responses via IFFT
//! and writes them as 32-bit float WAV files.

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

use crate::{SimulationOutput, room_metrics::peak_pressure_to_spl};

/// Sample rate for output WAV files (Hz)
const SAMPLE_RATE: u32 = 48000;

/// Export simulation output to WAV files (one per source × listening position).
///
/// Each WAV contains the impulse response synthesised from the frequency-domain
/// pressure data via `sonium_audio::dsp::compute_impulse_response_from_fr`.
///
/// Files are named `{source_name}_lp{lp_idx}.wav` under `output_dir`.
pub fn export_wavs(output: &SimulationOutput, output_dir: &Path) -> Result<Vec<String>> {
    output
        .validate_layout()
        .map_err(|error| anyhow::anyhow!(error))?;
    std::fs::create_dir_all(output_dir)?;

    let mut files_written = Vec::new();

    for (src_idx, source_name) in output.source_names.iter().enumerate() {
        let n_lps = output.pressures[src_idx].len();
        for lp_idx in 0..n_lps {
            let (_times, impulse) = impulse_response(output, src_idx, lp_idx)?;

            let filename = format!("{}_lp{}.wav", source_name, lp_idx);
            let filepath = output_dir.join(&filename);

            let spec = WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            };
            let mut writer = WavWriter::create(&filepath, spec)?;
            for &sample in &impulse {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;

            files_written.push(filename);
        }
    }

    Ok(files_written)
}

/// Reconstruct the exact impulse response that [`export_wavs`] writes for one
/// source/listener pair. Gallery code uses this to plot the same time-domain
/// data as the downloadable WAV rather than an independently approximated IR.
pub fn impulse_response(
    output: &SimulationOutput,
    source_index: usize,
    listener_index: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    output
        .validate_layout()
        .map_err(|error| anyhow::anyhow!(error))?;
    let pressures = output
        .pressures
        .get(source_index)
        .and_then(|source| source.get(listener_index))
        .ok_or_else(|| anyhow::anyhow!("source/listener index is outside the simulation output"))?;
    let frequencies: Vec<f32> = output
        .frequencies
        .iter()
        .map(|&frequency| frequency as f32)
        .collect();
    let mut magnitude_db = Vec::with_capacity(pressures.len());
    let mut phase_deg = Vec::with_capacity(pressures.len());
    for &pressure in pressures {
        let spl = peak_pressure_to_spl(pressure) as f32;
        magnitude_db.push(if spl.is_finite() {
            spl.clamp(-120.0, 150.0)
        } else {
            -120.0
        });
        let phase = pressure.arg().to_degrees() as f32;
        phase_deg.push(if phase.is_finite() { phase } else { 0.0 });
    }
    Ok(sonium_audio::dsp::compute_impulse_response_from_fr(
        &frequencies,
        &magnitude_db,
        &phase_deg,
        SAMPLE_RATE as f32,
    ))
}
