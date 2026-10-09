//! High-frequency extension for simulation output
//!
//! Below 500 Hz room interaction dominates and is computed by BEM/FEM.
//! Above 500 Hz the response is mostly the speaker's direct sound, so we
//! synthesise a realistic-looking curve: a gentle speaker response shape
//! plus slowly-varying random deviations (seeded per source/LP pair for
//! reproducibility).
//!
//! Subwoofer sources (lowpass crossover when simulation context is
//! supplied, otherwise names starting with "sub"/"lfe",
//! case-insensitive) are rolled off steeply since they already have an
//! 80 Hz lowpass crossover. Callers that know the crossover should prefer
//! the `*_with_room` entry points so test and production paths agree.

use crate::SimulationOutput;
use crate::room_metrics::{peak_pressure_to_spl, spl_to_peak_pressure};
use num_complex::Complex64;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sonium_xem::{CrossoverFilter, RoomSimulation, log_space};
#[cfg(test)]
use sonium_xem::{Point3D, RectangularRoom, RoomGeometry, Source};
use std::f64::consts::PI;

/// Number of log-spaced points in the 500 Hz – 20 kHz extension
const HF_NUM_POINTS: usize = 100;

/// Upper frequency limit of the extension
const HF_MAX_FREQ: f64 = 20_000.0;

/// Number of random control points for cosine-interpolated noise
const NOISE_CONTROL_POINTS: usize = 12;

pub fn extend_to_full_range(output: &SimulationOutput) -> SimulationOutput {
    try_extend_to_full_range(output).unwrap_or_else(|err| {
        log::warn!("HF extension failed ({err}); returning unextended grid");
        output.clone()
    })
}

/// Checked high-frequency extension entry point.
pub fn try_extend_to_full_range(output: &SimulationOutput) -> Result<SimulationOutput, String> {
    output.validate_layout()?;
    Ok(extend_to_full_range_with_simulation(output, None))
}

pub fn extend_to_full_range_with_room(
    output: &SimulationOutput,
    simulation: &RoomSimulation,
) -> SimulationOutput {
    try_extend_to_full_range_with_room(output, simulation).unwrap_or_else(|err| {
        log::warn!("HF extension with room failed ({err}); returning unextended grid");
        output.clone()
    })
}

/// Checked extension entry point that also verifies source/listener geometry
/// matches the pressure tensor before computing propagation-delay anchors.
pub fn try_extend_to_full_range_with_room(
    output: &SimulationOutput,
    simulation: &RoomSimulation,
) -> Result<SimulationOutput, String> {
    output.validate_layout()?;
    if simulation.sources.len() != output.pressures.len() {
        return Err(format!(
            "room/source mismatch: {} geometry sources for {} pressure sources",
            simulation.sources.len(),
            output.pressures.len()
        ));
    }
    if output
        .pressures
        .iter()
        .any(|source| source.len() != simulation.listening_positions.len())
    {
        return Err("room/listener mismatch in pressure tensor".into());
    }
    Ok(extend_to_full_range_with_simulation(
        output,
        Some(simulation),
    ))
}

fn extend_to_full_range_with_simulation(
    output: &SimulationOutput,
    simulation: Option<&RoomSimulation>,
) -> SimulationOutput {
    // Callers reach this function through a checked entry point. Keeping the
    // indexing local avoids repeating fallible plumbing in the hot loop.
    let last_sim_freq = output.frequencies[output.frequencies.len() - 1];

    // The simulated response already covers the requested full-range band.
    // Appending a 20 kHz extension here would create a descending grid when
    // the solver was intentionally run to 20.2 kHz (or any higher endpoint).
    if last_sim_freq >= HF_MAX_FREQ {
        return output.clone();
    }

    // Include the splice anchor while constructing the logarithmic grid, then
    // discard that first point. This produces exactly HF_NUM_POINTS samples,
    // all strictly above the final simulated frequency and ending at 20 kHz.
    let hf_freqs: Vec<f64> = log_space(last_sim_freq, HF_MAX_FREQ, HF_NUM_POINTS + 1)
        .into_iter()
        .skip(1)
        .collect();

    let mut extended_freqs = output.frequencies.clone();
    extended_freqs.extend_from_slice(&hf_freqs);

    let mut extended_pressures = Vec::with_capacity(output.pressures.len());

    for (src_idx, src_pressures) in output.pressures.iter().enumerate() {
        let source_name = &output.source_names[src_idx];
        // A source is treated as a subwoofer (steep HF rolloff) iff it has a
        // lowpass crossover. Falls back to the legacy name-prefix check when
        // no simulation context is supplied (e.g. unit tests).
        let is_sub = match simulation {
            Some(sim) => sim
                .sources
                .get(src_idx)
                .map(|s| matches!(s.crossover, CrossoverFilter::Lowpass { .. }))
                .unwrap_or_else(|| is_subwoofer_name(source_name)),
            None => is_subwoofer_name(source_name),
        };
        let mut src_extended = Vec::with_capacity(src_pressures.len());

        for (lp_idx, lp_pressures) in src_pressures.iter().enumerate() {
            let last_pressure = lp_pressures[lp_pressures.len() - 1];
            let last_spl = pressure_to_spl(last_pressure);
            let last_phase = last_pressure.arg();

            // Group delay anchor for the HF extension. Above the simulated
            // band the response is direct-sound dominated, so the physical
            // acoustic propagation delay (distance / c) is the right group
            // delay to extrapolate phase with. Earlier code used hardcoded
            // 1 ms / 0.3 ms anchors that bore no relation to actual geometry.
            let group_delay = simulation
                .map(|sim| source_to_lp_distance(sim, src_idx, lp_idx) / sim.speed_of_sound)
                .unwrap_or(0.0);

            let mut extended = lp_pressures.clone();

            match is_sub {
                true => {
                    for &freq in &hf_freqs {
                        let octaves_above = (freq / last_sim_freq).log2();
                        let spl = last_spl - 60.0 * octaves_above;
                        let phase = hf_phase(last_phase, last_sim_freq, freq, group_delay);
                        extended.push(spl_phase_to_pressure(spl, phase));
                    }
                }
                false => {
                    let seed = (src_idx as u64) * 10_000 + (lp_idx as u64) + 0xCAFE;
                    let noise = generate_smooth_noise(&hf_freqs, seed);
                    let angle_deg = simulation
                        .map(|sim| off_axis_angle_degrees(sim, src_idx, lp_idx))
                        .unwrap_or(0.0);

                    for (i, &freq) in hf_freqs.iter().enumerate() {
                        let shape_db = speaker_response_shape_for_name(source_name, freq);
                        let off_axis_db = hf_off_axis_tilt(freq, angle_deg);
                        let spl = last_spl + shape_db + noise[i] + off_axis_db;
                        let phase = hf_phase(last_phase, last_sim_freq, freq, group_delay);
                        extended.push(spl_phase_to_pressure(spl, phase));
                    }
                }
            }

            src_extended.push(extended);
        }

        extended_pressures.push(src_extended);
    }

    let mut extended_provenance = output.provenance.clone();
    debug_assert_eq!(
        extended_provenance.len(),
        output.frequencies.len(),
        "provenance mask must align with the frequency grid"
    );
    extended_provenance.resize(output.frequencies.len(), crate::FrequencyProvenance::Solved);
    extended_provenance.resize(extended_freqs.len(), crate::FrequencyProvenance::Synthetic);

    SimulationOutput {
        frequencies: extended_freqs,
        pressures: extended_pressures,
        source_names: output.source_names.clone(),
        provenance: extended_provenance,
    }
}

fn speaker_response_shape(freq: f64) -> f64 {
    // Presence bump: Gaussian centred at 3 kHz, ~1 octave wide
    let presence = 1.5 * (-0.5 * ((freq / 3000.0).ln() / 0.5_f64.ln()).powi(2)).exp();

    // Treble rolloff: starts at ~8 kHz, -6 dB at 20 kHz
    let rolloff = if freq > 8000.0 {
        -6.0 * ((freq / 8000.0).log2() / (20000.0_f64 / 8000.0).log2())
    } else {
        0.0
    };

    presence + rolloff
}

fn surround_response_shape(freq: f64) -> f64 {
    let base = speaker_response_shape(freq);
    if freq > 4000.0 {
        base - 2.0 * ((freq / 4000.0).log2()).min(1.5)
    } else {
        base
    }
}

fn center_response_shape(freq: f64) -> f64 {
    let base = speaker_response_shape(freq);
    if freq < 300.0 { base - 1.0 } else { base }
}

fn speaker_response_shape_for_name(name: &str, freq: f64) -> f64 {
    if name.contains("surround") {
        surround_response_shape(freq)
    } else if name == "center" {
        center_response_shape(freq)
    } else {
        speaker_response_shape(freq)
    }
}

fn source_to_lp_distance(simulation: &RoomSimulation, src_idx: usize, lp_idx: usize) -> f64 {
    let src = &simulation.sources[src_idx];
    let lp = simulation.listening_positions[lp_idx];
    let dx = lp.x - src.position.x;
    let dy = lp.y - src.position.y;
    let dz = lp.z - src.position.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Continue the simulated phase from its final sample using the acoustic
/// propagation delay under Sonium's `exp(-iωt)` convention.  The offset is
/// relative to the final simulated frequency so the first synthetic sample is
/// continuous with the solved phase rather than re-anchored at absolute zero.
fn hf_phase(last_phase: f64, last_frequency: f64, frequency: f64, group_delay: f64) -> f64 {
    last_phase + 2.0 * PI * (frequency - last_frequency) * group_delay
}

fn off_axis_angle_degrees(simulation: &RoomSimulation, src_idx: usize, lp_idx: usize) -> f64 {
    let src = &simulation.sources[src_idx];
    let lp = simulation.listening_positions[lp_idx];
    let dx = lp.x - src.position.x;
    let dy = lp.y - src.position.y;
    let dz = lp.z - src.position.z;
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < 1e-9 {
        return 0.0;
    }

    let n = simulation.listening_positions.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in &simulation.listening_positions {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    cx /= n;
    cy /= n;
    cz /= n;

    let fx = cx - src.position.x;
    let fy = cy - src.position.y;
    let fz = cz - src.position.z;
    let fr = (fx * fx + fy * fy + fz * fz).sqrt();
    if fr < 1e-9 {
        return 0.0;
    }

    let dot = dx * fx + dy * fy + dz * fz;
    let cos_angle = (dot / (r * fr)).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

fn hf_off_axis_tilt(freq: f64, angle_deg: f64) -> f64 {
    if freq < 2000.0 {
        return 0.0;
    }

    let clamped_angle = angle_deg.clamp(0.0, 120.0);
    let base_db = if clamped_angle <= 15.0 {
        0.0
    } else {
        let t = ((clamped_angle - 15.0) / 75.0).min(1.0);
        -6.0 * t
    };

    let hf_ratio = (freq / 2000.0).log2() / (HF_MAX_FREQ / 2000.0).log2();
    let hf_factor = hf_ratio.clamp(0.0, 1.0);
    base_db * hf_factor
}

/// Generate slowly-varying noise via cosine interpolation between random
/// control points.  Values are in the range roughly [-5, +3] dB.
fn generate_smooth_noise(freqs: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let n = freqs.len();
    if n == 0 {
        return Vec::new();
    }

    let log_min = freqs[0].ln();
    let log_max = freqs[n - 1].ln();
    let log_span = log_max - log_min;

    // Generate control-point dB values: biased slightly negative [-5, +3]
    let control_vals: Vec<f64> = (0..NOISE_CONTROL_POINTS)
        .map(|_| rng.random_range(-5.0..3.0))
        .collect();

    // Cosine-interpolate at each frequency
    freqs
        .iter()
        .map(|&f| {
            let t = (f.ln() - log_min) / log_span; // 0..1
            let scaled = t * (NOISE_CONTROL_POINTS - 1) as f64;
            let idx = (scaled as usize).min(NOISE_CONTROL_POINTS - 2);
            let frac = scaled - idx as f64;

            // Cosine interpolation for smooth transitions
            let cos_frac = (1.0 - (frac * PI).cos()) * 0.5;
            control_vals[idx] * (1.0 - cos_frac) + control_vals[idx + 1] * cos_frac
        })
        .collect()
}

fn pressure_to_spl(p: Complex64) -> f64 {
    peak_pressure_to_spl(p)
}

/// Legacy name-based subwoofer check for callers without simulation context.
/// Matches the production crossover rule for the standard "sub*"/"lfe*"
/// (case-insensitive) names; anything else needs the `*_with_room` entry
/// points to be classified correctly.
fn is_subwoofer_name(source_name: &str) -> bool {
    let lower = source_name.to_lowercase();
    lower.starts_with("sub") || lower.starts_with("lfe")
}

fn spl_phase_to_pressure(spl: f64, phase: f64) -> Complex64 {
    match spl_to_peak_pressure(spl, phase) {
        Ok(pressure) => pressure,
        Err(err) => {
            // Clamp to a finite (if physically absurd) pressure instead of
            // panicking: this helper runs inside the infallible extension
            // path, and a saturated anchor is strictly better than a crash.
            log::warn!("HF extension hit non-finite pressure ({err}); saturating");
            let phase = if phase.is_finite() { phase } else { 0.0 };
            let spl = spl.clamp(-1000.0, 1000.0);
            spl_to_peak_pressure(spl, phase).unwrap_or(Complex64::new(0.0, 0.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_response_shape_flat_at_midband() {
        // Should be close to 0 dB at 1 kHz (midrange, away from presence peak)
        let db = speaker_response_shape(1000.0);
        assert!(
            db.abs() < 0.5,
            "Expected near-flat at 1 kHz, got {db:.2} dB"
        );
    }

    #[test]
    fn test_speaker_response_shape_presence_peak() {
        let db = speaker_response_shape(3000.0);
        assert!(db > 0.5, "Expected presence rise at 3 kHz, got {db:.2} dB");
        assert!(db < 3.0, "Presence peak too large at 3 kHz: {db:.2} dB");
    }

    #[test]
    fn test_speaker_response_shape_rolloff_at_20khz() {
        let db = speaker_response_shape(20000.0);
        assert!(db < -4.0, "Expected rolloff at 20 kHz, got {db:.2} dB");
        assert!(db > -8.0, "Rolloff too steep at 20 kHz: {db:.2} dB");
    }

    #[test]
    fn test_smooth_noise_range() {
        let freqs = log_space(500.0, 20000.0, 100);
        let noise = generate_smooth_noise(&freqs, 42);
        assert_eq!(noise.len(), 100);
        for &v in &noise {
            assert!(
                (-6.0..=4.0).contains(&v),
                "Noise value {v:.2} out of expected range"
            );
        }
    }

    #[test]
    fn test_smooth_noise_deterministic() {
        let freqs = log_space(500.0, 20000.0, 50);
        let a = generate_smooth_noise(&freqs, 123);
        let b = generate_smooth_noise(&freqs, 123);
        assert_eq!(a, b, "Same seed should produce identical noise");
    }

    #[test]
    fn test_smooth_noise_varies_slowly() {
        let freqs = log_space(500.0, 20000.0, 200);
        let noise = generate_smooth_noise(&freqs, 99);
        // Adjacent samples should not jump more than ~2 dB
        for i in 1..noise.len() {
            let jump = (noise[i] - noise[i - 1]).abs();
            assert!(
                jump < 2.0,
                "Noise jumps {jump:.3} dB between samples {i}-{}: too fast",
                i - 1
            );
        }
    }

    #[test]
    fn test_spl_pressure_roundtrip() {
        let original_spl = 80.0;
        let phase = 0.5;
        let p = spl_phase_to_pressure(original_spl, phase);
        let recovered = pressure_to_spl(p);
        assert!(
            (recovered - original_spl).abs() < 0.01,
            "SPL roundtrip failed: {original_spl} -> {recovered}"
        );
    }

    #[test]
    fn test_spl_uses_peak_to_rms_acoustic_reference() {
        let one_rms_pascal_peak = std::f64::consts::SQRT_2;
        let expected = 20.0 * (1.0 / 20.0e-6_f64).log10();
        let actual = pressure_to_spl(Complex64::new(one_rms_pascal_peak, 0.0));
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "1 Pa RMS should be about 94 dB SPL: actual={actual}, expected={expected}"
        );

        let recovered = spl_phase_to_pressure(expected, 0.0);
        assert!((recovered.re - one_rms_pascal_peak).abs() < 1.0e-12);
        assert!(recovered.im.abs() < 1.0e-12);
    }

    #[test]
    fn test_extend_dimensions() {
        // Minimal SimulationOutput: 2 sources, 1 LP, 10 frequencies
        let n_sim = 10;
        let freqs: Vec<f64> = log_space(20.0, 500.0, n_sim);
        let pressures = vec![
            // source 0 ("left"): 1 LP
            vec![vec![Complex64::new(0.01, 0.002); n_sim]],
            // source 1 ("sub1"): 1 LP
            vec![vec![Complex64::new(0.005, 0.001); n_sim]],
        ];
        let output = SimulationOutput::solved(
            freqs,
            pressures,
            vec!["left".to_string(), "sub1".to_string()],
        );

        let extended = extend_to_full_range(&output);

        assert_eq!(
            extended.frequencies.len(),
            n_sim + HF_NUM_POINTS,
            "Expected {} frequencies",
            n_sim + HF_NUM_POINTS
        );
        assert_eq!(extended.pressures.len(), 2);
        assert_eq!(extended.pressures[0][0].len(), n_sim + HF_NUM_POINTS);
        assert_eq!(extended.pressures[1][0].len(), n_sim + HF_NUM_POINTS);

        // First N frequencies unchanged
        assert_eq!(extended.frequencies[0], output.frequencies[0]);
        // Last frequency near 20 kHz
        assert!(extended.frequencies.last().unwrap() > &19000.0);
        assert!(
            extended
                .frequencies
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
    }

    #[test]
    fn extension_near_20khz_remains_strictly_increasing() {
        let frequencies = vec![20.0, 1_000.0, 19_900.0];
        let n = frequencies.len();
        let output = SimulationOutput::solved(
            frequencies,
            vec![vec![vec![Complex64::new(0.01, 0.0); n]]],
            vec!["left".to_string()],
        );

        let extended = extend_to_full_range(&output);

        assert_eq!(extended.frequencies.len(), 3 + HF_NUM_POINTS);
        assert!(
            (extended.frequencies.last().unwrap() - HF_MAX_FREQ).abs() < 1e-9,
            "extension must end at 20 kHz"
        );
        assert!(
            extended
                .frequencies
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
    }

    #[test]
    fn simulation_reaching_20200hz_is_not_extended_backwards() {
        let frequencies = vec![20.0, 1_000.0, 20_000.0, 20_200.0];
        let n = frequencies.len();
        let output = SimulationOutput::solved(
            frequencies,
            vec![vec![vec![Complex64::new(0.01, 0.0); n]]],
            vec!["left".to_string()],
        );

        let extended = extend_to_full_range(&output);

        assert_eq!(extended.frequencies, output.frequencies);
        assert_eq!(extended.pressures, output.pressures);
        assert!(
            extended
                .frequencies
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
    }

    #[test]
    fn test_extend_subwoofer_rolls_off() {
        let n_sim = 10;
        let freqs: Vec<f64> = log_space(20.0, 500.0, n_sim);
        let pressures = vec![vec![vec![Complex64::new(0.01, 0.0); n_sim]]];
        let output = SimulationOutput::solved(freqs, pressures, vec!["sub1".to_string()]);

        let extended = extend_to_full_range(&output);

        // SPL at 20 kHz should be far below SPL at 500 Hz
        let spl_500 = pressure_to_spl(extended.pressures[0][0][n_sim - 1]);
        let spl_20k = pressure_to_spl(*extended.pressures[0][0].last().unwrap());
        assert!(
            spl_20k < spl_500 - 100.0,
            "Sub should be >100 dB below at 20 kHz: {spl_500:.1} vs {spl_20k:.1}"
        );
    }

    #[test]
    fn test_extend_main_speaker_continuous() {
        let n_sim = 50;
        let freqs: Vec<f64> = log_space(20.0, 500.0, n_sim);
        let pressures = vec![vec![vec![Complex64::new(0.02, 0.005); n_sim]]];
        let output = SimulationOutput::solved(freqs, pressures, vec!["left".to_string()]);

        let extended = extend_to_full_range(&output);

        // SPL should be continuous at the junction (within ~5 dB due to noise)
        let spl_last_sim = pressure_to_spl(extended.pressures[0][0][n_sim - 1]);
        let spl_first_hf = pressure_to_spl(extended.pressures[0][0][n_sim]);
        let jump = (spl_first_hf - spl_last_sim).abs();
        assert!(
            jump < 6.0,
            "SPL jump at 500 Hz junction: {jump:.1} dB (last_sim={spl_last_sim:.1}, first_hf={spl_first_hf:.1})"
        );
    }

    #[test]
    fn hf_phase_is_continuous_and_uses_exp_minus_iwt_slope() {
        let n_sim = 50;
        let frequencies = log_space(20.0, 500.0, n_sim);
        let source_position = Point3D::new(0.5, 0.5, 0.5);
        let listener = Point3D::new(3.0, 0.5, 0.5);
        let simulation = RoomSimulation::with_frequencies(
            RoomGeometry::Rectangular(RectangularRoom::new(4.0, 1.5, 1.5)),
            vec![Source::omnidirectional(source_position, 1.0)],
            vec![listener],
            20.0,
            500.0,
            n_sim,
        );
        let last_phase = 0.7;
        let output = SimulationOutput::solved(
            frequencies,
            vec![vec![vec![Complex64::from_polar(0.02, last_phase); n_sim]]],
            vec!["left".to_string()],
        );

        let extended = extend_to_full_range_with_room(&output, &simulation);
        let last_sim_freq = output.frequencies[n_sim - 1];
        let first_hf_freq = extended.frequencies[n_sim];
        let observed_delta =
            (extended.pressures[0][0][n_sim] / output.pressures[0][0][n_sim - 1]).arg();
        let distance = source_position.distance_to(&listener);
        let expected_delta =
            2.0 * PI * (first_hf_freq - last_sim_freq) * distance / simulation.speed_of_sound;
        let wrapped_error = (observed_delta - expected_delta + PI).rem_euclid(2.0 * PI) - PI;
        assert!(
            wrapped_error.abs() < 1.0e-10,
            "HF phase must continue from the simulated anchor: observed={observed_delta:.12}, expected={expected_delta:.12}, error={wrapped_error:.3e}"
        );
    }

    #[test]
    fn subwoofer_name_fallback_covers_standard_names() {
        for name in ["sub1", "Sub", "SUB-2", "lfe", "LFE1"] {
            assert!(
                is_subwoofer_name(name),
                "{name} should count as a subwoofer"
            );
        }
        for name in ["left", "center", "", "absolute", "presub"] {
            assert!(
                !is_subwoofer_name(name),
                "{name} should not count as a subwoofer"
            );
        }
    }

    #[test]
    fn crossover_and_name_paths_agree_on_sub_rolloff() {
        // A lowpass-crossover source must get the steep sub rolloff even when
        // its name gives no hint, and the legacy name path must agree for
        // standard sub names.
        let n_sim = 10;
        let freqs = log_space(20.0, 500.0, n_sim);
        let flat = vec![Complex64::new(0.01, 0.002); n_sim];
        let position = Point3D::new(0.5, 0.5, 0.5);
        let listener = Point3D::new(3.0, 0.5, 0.5);
        let mut sub_source = Source::omnidirectional(position, 1.0);
        sub_source.name = "lfe".to_string();
        sub_source.crossover = CrossoverFilter::Lowpass {
            cutoff_freq: 80.0,
            order: 4,
        };
        let main_source = Source::omnidirectional(position, 1.0);
        let simulation = RoomSimulation::with_frequencies(
            RoomGeometry::Rectangular(RectangularRoom::new(4.0, 1.5, 1.5)),
            vec![main_source, sub_source],
            vec![listener],
            20.0,
            500.0,
            n_sim,
        );
        let output = SimulationOutput::solved(
            freqs,
            vec![vec![flat.clone()], vec![flat.clone()]],
            vec!["main".to_string(), "mystery-bass".to_string()],
        );
        let extended = extend_to_full_range_with_room(&output, &simulation);
        let tail = extended.pressures[1][0].len() - 1;
        let sub_tail_spl = pressure_to_spl(extended.pressures[1][0][tail]);
        let main_tail_spl = pressure_to_spl(extended.pressures[0][0][tail]);
        assert!(
            sub_tail_spl < main_tail_spl - 20.0,
            "lowpass-crossover source must roll off steeply: sub={sub_tail_spl:.1} dB, main={main_tail_spl:.1} dB"
        );

        // Name fallback agrees for a standard sub name without simulation.
        let named = SimulationOutput::solved(
            log_space(20.0, 500.0, n_sim),
            vec![vec![flat]],
            vec!["LFE".to_string()],
        );
        let named_ext = extend_to_full_range(&named);
        let named_tail_spl = pressure_to_spl(named_ext.pressures[0][0][tail]);
        assert!(
            (named_tail_spl - sub_tail_spl).abs() < 6.0,
            "name and crossover paths should agree for standard sub names: {named_tail_spl:.1} vs {sub_tail_spl:.1} dB"
        );
    }

    #[test]
    fn checked_extension_rejects_malformed_output_without_panicking() {
        let malformed =
            SimulationOutput::solved(vec![100.0], vec![Vec::new()], vec!["source".into()]);
        assert!(try_extend_to_full_range(&malformed).is_err());
        assert_eq!(extend_to_full_range(&malformed).frequencies, vec![100.0]);
    }
}
