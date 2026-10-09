//! Spectral and far-field analysis for resolved nonlinear port-flow results.
//!
//! The Hann-windowed single-frequency DFT is deliberately ported from
//! `sonium-sim-speaker::distortion`: the modeling crate sits below the
//! application crate in the dependency graph and must not depend on it.

use std::f64::consts::TAU;

use num_complex::Complex64;

use super::PortFlowError;
#[cfg(test)]
use super::coupled::CoupledEnergyStep;
use super::coupled::{CoupledEnergySummary, CoupledPortResult};
use super::linear_reference::ComplexPortTransfer;

/// Hann-windowed Goertzel amplitude of a real signal at `target_freq`.
///
/// Returns zero when there are too few samples to define the Hann window, or
/// when the sampling interval or target frequency is not finite.
pub fn dft_amplitude(signal: &[f64], dt: f64, target_freq: f64) -> f64 {
    dft_phasor(signal, dt, target_freq).norm()
}

/// Hann-windowed complex peak phasor of a real signal at `target_freq`.
///
/// The returned phasor uses the `exp(-i omega t)` convention. Degenerate or
/// non-finite inputs return zero, matching the existing analysis API's
/// fail-closed convention.
pub fn dft_phasor(signal: &[f64], dt: f64, target_freq: f64) -> Complex64 {
    let n = signal.len();
    if n < 2
        || !(dt.is_finite() && dt > 0.0)
        || !(target_freq.is_finite() && target_freq > 0.0)
        || signal.iter().any(|sample| !sample.is_finite())
    {
        return Complex64::new(0.0, 0.0);
    }

    let omega = TAU * target_freq;
    let mut re = 0.0;
    let mut im = 0.0;

    for (k, &sample) in signal.iter().enumerate() {
        let t = k as f64 * dt;
        // Periodic Hann: preserves exact-bin orthogonality for complete
        // periods while retaining the leakage suppression of the source
        // implementation.
        let window = 0.5 * (1.0 - (TAU * k as f64 / n as f64).cos());
        let windowed = sample * window;
        re += windowed * (omega * t).cos();
        im += windowed * (omega * t).sin();
    }

    // Single-sided amplitude normalization. The Hann coherent gain is 1/2.
    let scale = 2.0 / (n as f64 / 2.0);
    scale * Complex64::new(re, im)
}

/// Full-complex low-drive observables extracted from one common settled tail.
#[derive(Debug, Clone)]
pub struct CoupledComplexObservables {
    /// Number of samples in the common complete-cycle tail.
    pub tail_samples: usize,
    /// Box-pressure peak phasor in Pa, using `exp(-i omega t)`.
    pub box_pressure_pa: Complex64,
    /// Exterior port-flow peak phasor in m³/s, positive box-to-exterior.
    pub port_flow_m3s: Complex64,
    /// Diaphragm-flow peak phasor in m³/s, positive compressing the box.
    pub diaphragm_flow_m3s: Complex64,
    /// `(P_box-P_exterior)/Q_port` in Pa·s/m³.
    pub port_impedance_pa_s_per_m3: Complex64,
    /// Direct `Q_port/Q_diaphragm` transfer.
    pub direct_box_to_port_flow_ratio: Complex64,
    /// Transfer reconstructed as `1/(1-i omega C_box Z)`.
    pub box_to_port_flow_ratio_from_impedance: Complex64,
    /// Complex cap-continuity error `|Q_ext+Q_box,out|/|Q_ext|`.
    pub cap_continuity_relative_error: f64,
}

/// Thresholds for a low-drive coupled-result comparison with the independent
/// harmonic linear reference.
///
/// Passing this gate is evidence for the linear low-drive limit only. It is
/// deliberately not a nonlinear-model validation claim: THD and Mach-number
/// limits reject a candidate that is outside the regime where this reference
/// is an appropriate oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearReferenceGateConfig {
    pub max_impedance_relative_error: f64,
    pub max_flow_ratio_relative_error: f64,
    pub max_transfer_identity_relative_error: f64,
    pub max_cap_continuity_relative_error: f64,
    pub max_reference_relative_residual: f64,
    pub max_reference_flux_relative_difference: f64,
    pub max_energy_algebraic_relative_closure: f64,
    pub max_energy_physical_relative_closure: f64,
    pub max_energy_interface_relative_residual: f64,
    pub max_thd_fraction: f64,
    pub max_mach: f64,
}

impl Default for LinearReferenceGateConfig {
    fn default() -> Self {
        Self {
            max_impedance_relative_error: 0.03,
            max_flow_ratio_relative_error: 0.03,
            max_transfer_identity_relative_error: 0.02,
            max_cap_continuity_relative_error: 1.0e-6,
            max_reference_relative_residual: 1.0e-8,
            max_reference_flux_relative_difference: 1.0e-6,
            max_energy_algebraic_relative_closure: 1.0e-8,
            max_energy_physical_relative_closure: 0.05,
            max_energy_interface_relative_residual: 1.0e-8,
            max_thd_fraction: 0.005,
            max_mach: 0.05,
        }
    }
}

/// Metrics emitted by [`validate_coupled_against_linear_reference`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearReferenceGateReport {
    pub frequency_hz: f64,
    pub impedance_relative_error: f64,
    pub flow_ratio_relative_error: f64,
    pub transfer_identity_relative_error: f64,
    pub reference_relative_residual: f64,
    pub reference_flux_relative_difference: f64,
    pub cap_continuity_relative_error: f64,
    pub energy_algebraic_relative_closure: f64,
    pub energy_physical_relative_closure: f64,
    pub energy_interface_relative_residual: f64,
    pub thd_fraction: f64,
    pub mach: f64,
}

/// A coupled tail and its independent-reference inputs for the linear gate.
#[derive(Debug, Clone, Copy)]
pub struct LinearReferenceGateInput<'a> {
    pub frequency_hz: f64,
    pub observables: &'a CoupledComplexObservables,
    pub reference: &'a ComplexPortTransfer,
    pub energy: &'a CoupledEnergySummary,
    pub thd_fraction: f64,
    pub mach: f64,
}

fn complex_relative_error(actual: Complex64, reference: Complex64) -> Result<f64, PortFlowError> {
    if ![actual.re, actual.im, reference.re, reference.im]
        .into_iter()
        .all(f64::is_finite)
        || reference.norm() <= f64::EPSILON
    {
        return Err(PortFlowError::InvalidConfig(
            "linear-reference comparison has a non-finite or zero complex reference".into(),
        ));
    }
    let error = (actual - reference).norm() / reference.norm();
    if !error.is_finite() {
        return Err(PortFlowError::InvalidConfig(
            "linear-reference complex relative error is non-finite".into(),
        ));
    }
    Ok(error)
}

fn validate_linear_reference_gate_config(
    config: LinearReferenceGateConfig,
) -> Result<(), PortFlowError> {
    let thresholds = [
        config.max_impedance_relative_error,
        config.max_flow_ratio_relative_error,
        config.max_transfer_identity_relative_error,
        config.max_cap_continuity_relative_error,
        config.max_reference_relative_residual,
        config.max_reference_flux_relative_difference,
        config.max_energy_algebraic_relative_closure,
        config.max_energy_physical_relative_closure,
        config.max_energy_interface_relative_residual,
        config.max_thd_fraction,
        config.max_mach,
    ];
    if thresholds
        .into_iter()
        .any(|threshold| !threshold.is_finite() || threshold < 0.0)
    {
        return Err(PortFlowError::InvalidConfig(
            "linear-reference gate thresholds must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

/// Gate a coupled port-flow result against the independent harmonic reference.
///
/// The input observables must come from one settled, complete-cycle tail. The
/// energy summary must cover the same tail. The gate checks both complex
/// impedance/transfer agreement and the internal residual/energy audits, and
/// rejects non-finite inputs or a candidate whose THD/Mach number is too large
/// to support a linear-reference claim.
pub fn validate_coupled_against_linear_reference(
    input: &LinearReferenceGateInput<'_>,
    config: LinearReferenceGateConfig,
) -> Result<LinearReferenceGateReport, PortFlowError> {
    let LinearReferenceGateInput {
        frequency_hz,
        observables,
        reference,
        energy,
        thd_fraction,
        mach,
    } = *input;
    validate_linear_reference_gate_config(config)?;
    if !(frequency_hz.is_finite() && frequency_hz > 0.0)
        || !(reference.frequency_hz.is_finite() && reference.frequency_hz > 0.0)
    {
        return Err(PortFlowError::InvalidConfig(
            "linear-reference gate frequency must be finite and positive".into(),
        ));
    }
    let frequency_error =
        (frequency_hz - reference.frequency_hz).abs() / frequency_hz.max(reference.frequency_hz);
    if frequency_error > 1.0e-12 {
        return Err(PortFlowError::InvalidConfig(format!(
            "linear-reference frequency mismatch: candidate={frequency_hz:e}, reference={:e}",
            reference.frequency_hz
        )));
    }
    if reference.polynomial_degree == 0
        || reference.quadrature_order == 0
        || reference.scalar_dofs == 0
        || ![
            reference.box_pressure_rms_pa,
            reference.exterior_pressure_rms_pa,
            reference.port_volume_velocity_rms_m3_per_s,
            reference.strong_exterior_volume_velocity_rms_m3_per_s,
            reference.weak_box_outward_volume_velocity_rms_m3_per_s,
            reference.weak_net_storage_volume_velocity_rms_m3_per_s,
            reference.port_impedance_pa_s_per_m3,
            reference.box_to_port_flow_ratio,
        ]
        .into_iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
        || ![
            reference.strong_weak_exterior_flux_relative_difference,
            reference.box_to_port_flow_ratio_condition_number,
            reference.relative_residual,
        ]
        .into_iter()
        .all(f64::is_finite)
        || reference.box_to_port_flow_ratio_condition_number <= 0.0
    {
        return Err(PortFlowError::InvalidConfig(
            "linear-reference result contains invalid metadata or non-finite diagnostics".into(),
        ));
    }
    if ![
        observables.box_pressure_pa,
        observables.port_flow_m3s,
        observables.diaphragm_flow_m3s,
        observables.port_impedance_pa_s_per_m3,
        observables.direct_box_to_port_flow_ratio,
        observables.box_to_port_flow_ratio_from_impedance,
    ]
    .into_iter()
    .all(|value| value.re.is_finite() && value.im.is_finite())
        || observables.tail_samples == 0
        || observables.port_flow_m3s.norm() <= f64::EPSILON
        || observables.diaphragm_flow_m3s.norm() <= f64::EPSILON
        || !observables.cap_continuity_relative_error.is_finite()
        || observables.cap_continuity_relative_error < 0.0
    {
        return Err(PortFlowError::InvalidConfig(
            "coupled linear-reference candidate contains invalid observables".into(),
        ));
    }
    if !energy.is_finite()
        || energy.physical_dissipation_j < -1.0e-12 * energy.energy_scale_j
        || thd_fraction.is_nan()
        || !thd_fraction.is_finite()
        || thd_fraction < 0.0
        || !mach.is_finite()
        || mach < 0.0
    {
        return Err(PortFlowError::InvalidConfig(
            "coupled linear-reference candidate contains invalid energy or nonlinearity diagnostics"
                .into(),
        ));
    }
    if thd_fraction > config.max_thd_fraction || mach > config.max_mach {
        return Err(PortFlowError::InvalidConfig(format!(
            "linear-reference claim is outside the low-drive regime: thd={thd_fraction:e} (max {max_thd:e}), mach={mach:e} (max {max_mach:e})",
            max_thd = config.max_thd_fraction,
            max_mach = config.max_mach,
        )));
    }

    let impedance_relative_error = complex_relative_error(
        observables.port_impedance_pa_s_per_m3,
        reference.port_impedance_pa_s_per_m3,
    )?;
    let flow_ratio_relative_error = complex_relative_error(
        observables.direct_box_to_port_flow_ratio,
        reference.box_to_port_flow_ratio,
    )?;
    let transfer_identity_relative_error = complex_relative_error(
        observables.direct_box_to_port_flow_ratio,
        observables.box_to_port_flow_ratio_from_impedance,
    )?;
    if reference.relative_residual > config.max_reference_relative_residual
        || reference.strong_weak_exterior_flux_relative_difference
            > config.max_reference_flux_relative_difference
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "linear reference residual/flux audit failed: residual={:e}, strong_weak_flux={:e}",
            reference.relative_residual, reference.strong_weak_exterior_flux_relative_difference,
        )));
    }
    if observables.cap_continuity_relative_error > config.max_cap_continuity_relative_error {
        return Err(PortFlowError::InvalidConfig(format!(
            "coupled cap-continuity audit failed: {:e} exceeds {:e}",
            observables.cap_continuity_relative_error, config.max_cap_continuity_relative_error,
        )));
    }
    if energy.algebraic_relative_closure > config.max_energy_algebraic_relative_closure
        || energy.physical_relative_closure > config.max_energy_physical_relative_closure
        || energy.interface_relative_residual > config.max_energy_interface_relative_residual
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "coupled energy audit failed: algebraic={:e}, physical={:e}, interface={:e}",
            energy.algebraic_relative_closure,
            energy.physical_relative_closure,
            energy.interface_relative_residual,
        )));
    }
    if impedance_relative_error > config.max_impedance_relative_error
        || flow_ratio_relative_error > config.max_flow_ratio_relative_error
        || transfer_identity_relative_error > config.max_transfer_identity_relative_error
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "coupled result disagrees with linear reference: Z={impedance_relative_error:e}, H={flow_ratio_relative_error:e}, H(Z)={transfer_identity_relative_error:e}"
        )));
    }

    Ok(LinearReferenceGateReport {
        frequency_hz,
        impedance_relative_error,
        flow_ratio_relative_error,
        transfer_identity_relative_error,
        reference_relative_residual: reference.relative_residual,
        reference_flux_relative_difference: reference.strong_weak_exterior_flux_relative_difference,
        cap_continuity_relative_error: observables.cap_continuity_relative_error,
        energy_algebraic_relative_closure: energy.algebraic_relative_closure,
        energy_physical_relative_closure: energy.physical_relative_closure,
        energy_interface_relative_residual: energy.interface_relative_residual,
        thd_fraction,
        mach,
    })
}

/// Extract `P`, `Q`, and `Qd` from the same final complete-cycle tail.
///
/// `samples_per_cycle` is explicit so the caller proves its timestep is an
/// integer subdivision of the drive period. At least `discarded_cycles` must
/// precede the selected tail. All phasors use `exp(-i omega t)`.
pub fn coupled_complex_observables(
    result: &CoupledPortResult,
    dt: f64,
    frequency_hz: f64,
    samples_per_cycle: usize,
    tail_cycles: usize,
    discarded_cycles: usize,
    box_volume_m3: f64,
    density_kg_per_m3: f64,
    sound_speed_m_per_s: f64,
) -> Result<CoupledComplexObservables, PortFlowError> {
    if !(dt.is_finite() && dt > 0.0 && frequency_hz.is_finite() && frequency_hz > 0.0)
        || samples_per_cycle < 4
        || tail_cycles == 0
        || !(box_volume_m3.is_finite() && box_volume_m3 > 0.0)
        || !(density_kg_per_m3.is_finite() && density_kg_per_m3 > 0.0)
        || !(sound_speed_m_per_s.is_finite() && sound_speed_m_per_s > 0.0)
    {
        return Err(PortFlowError::InvalidConfig(
            "invalid coupled complex-observable configuration".into(),
        ));
    }
    let cycle_error = (dt * frequency_hz * samples_per_cycle as f64 - 1.0).abs();
    if cycle_error > 1.0e-10 {
        return Err(PortFlowError::InvalidConfig(format!(
            "dt does not divide the drive period: cycle error={cycle_error:e}"
        )));
    }
    let lengths = [
        result.time_s.len(),
        result.port_flow_m3s.len(),
        result.exterior_cap_flow_m3s.len(),
        result.box_cap_outward_flow_m3s.len(),
        result.diaphragm_volume_velocity_m3s.len(),
        result.box_pressure_pa.len(),
    ];
    if lengths.iter().any(|&length| length != lengths[0]) {
        return Err(PortFlowError::InvalidConfig(format!(
            "coupled result series are not aligned: {lengths:?}"
        )));
    }
    let tail_samples = tail_cycles
        .checked_mul(samples_per_cycle)
        .ok_or_else(|| PortFlowError::InvalidConfig("tail sample count overflow".into()))?;
    let required = (tail_cycles + discarded_cycles)
        .checked_mul(samples_per_cycle)
        .ok_or_else(|| PortFlowError::InvalidConfig("settled sample count overflow".into()))?;
    if lengths[0] < required {
        return Err(PortFlowError::InvalidConfig(format!(
            "need {required} samples for {discarded_cycles} discarded + {tail_cycles} tail cycles, got {}",
            lengths[0]
        )));
    }
    let start = lengths[0] - tail_samples;
    let phasor = |signal: &[f64]| dft_phasor(&signal[start..], dt, frequency_hz);
    let pressure = phasor(&result.box_pressure_pa);
    let flow = phasor(&result.port_flow_m3s);
    let box_flow = phasor(&result.box_cap_outward_flow_m3s);
    let diaphragm = phasor(&result.diaphragm_volume_velocity_m3s);
    let finite = |value: Complex64| value.re.is_finite() && value.im.is_finite();
    if ![pressure, flow, box_flow, diaphragm]
        .into_iter()
        .all(finite)
        || flow.norm() <= f64::EPSILON
        || diaphragm.norm() <= f64::EPSILON
    {
        return Err(PortFlowError::InvalidConfig(
            "settled fundamental is zero or non-finite".into(),
        ));
    }
    let impedance = pressure / flow;
    let direct_transfer = flow / diaphragm;
    let compliance = box_volume_m3 / (density_kg_per_m3 * sound_speed_m_per_s.powi(2));
    let from_impedance = Complex64::new(1.0, 0.0)
        / (Complex64::new(1.0, 0.0)
            - Complex64::new(0.0, TAU * frequency_hz * compliance) * impedance);
    let continuity = (flow + box_flow).norm() / flow.norm();
    if ![impedance, direct_transfer, from_impedance]
        .into_iter()
        .all(finite)
        || !continuity.is_finite()
    {
        return Err(PortFlowError::InvalidConfig(
            "derived coupled complex observable is non-finite".into(),
        ));
    }
    Ok(CoupledComplexObservables {
        tail_samples,
        box_pressure_pa: pressure,
        port_flow_m3s: flow,
        diaphragm_flow_m3s: diaphragm,
        port_impedance_pa_s_per_m3: impedance,
        direct_box_to_port_flow_ratio: direct_transfer,
        box_to_port_flow_ratio_from_impedance: from_impedance,
        cap_continuity_relative_error: continuity,
    })
}

/// THD = `sqrt(sum(H_k^2)) / H_1` over `n_harmonics` harmonics.
pub fn thd_fraction(signal: &[f64], dt: f64, fundamental_hz: f64, n_harmonics: usize) -> f64 {
    let fundamental = dft_amplitude(signal, dt, fundamental_hz);
    if fundamental <= f64::EPSILON || n_harmonics < 2 {
        return 0.0;
    }

    let harmonic_power = (2..=n_harmonics)
        .map(|harmonic| dft_amplitude(signal, dt, harmonic as f64 * fundamental_hz).powi(2))
        .sum::<f64>();
    harmonic_power.sqrt() / fundamental
}

/// Two-tone IMD products at `f2 ± k*f1`, for `k = 1, 2`, relative to `f2`.
pub fn imd_fraction(signal: &[f64], dt: f64, f1_hz: f64, f2_hz: f64) -> f64 {
    let reference = dft_amplitude(signal, dt, f2_hz);
    if reference <= f64::EPSILON {
        return 0.0;
    }

    let product_power = (1..=2)
        .flat_map(|order| {
            let offset = order as f64 * f1_hz;
            [f2_hz - offset, f2_hz + offset]
        })
        .map(|frequency| dft_amplitude(signal, dt, frequency).powi(2))
        .sum::<f64>();
    product_power.sqrt() / reference
}

/// One point on a voltage-swept port compression curve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionPoint {
    /// Applied terminal voltage (V RMS).
    pub voltage_rms_v: f64,
    /// Peak port volume-velocity amplitude (m³/s).
    pub port_flow_amplitude_m3s: f64,
    /// Estimated far-field port SPL (dB re 20 µPa).
    pub estimated_port_spl_db: f64,
    /// Port-output compression relative to the small-signal trend (dB).
    pub compression_db: f64,
    /// Total harmonic distortion as a fraction.
    pub thd: f64,
}

/// Estimate far-field port SPL for a baffled harmonic monopole.
///
/// For peak volume-velocity amplitude `Q`, the pressure amplitude is
/// `|p| = rho * f * |Q| * sqrt(2) / (2*r)`. Converting that peak pressure to
/// RMS gives `p_rms = rho * f * |Q| / (2*r)` before applying the 20 µPa SPL
/// reference.
pub fn port_spl_db(q_amplitude_m3s: f64, frequency_hz: f64, rho: f64, distance_m: f64) -> f64 {
    let pressure_rms = rho * frequency_hz * q_amplitude_m3s.abs() / (2.0 * distance_m);
    20.0 * (pressure_rms / 20e-6).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_bin_signal(amplitude: f64, phase: f64, sine: bool) -> (Vec<f64>, f64, f64) {
        let n = 4096;
        let sample_rate = 4096.0;
        let frequency = 64.0;
        let signal = (0..n)
            .map(|sample| {
                let angle = TAU * frequency * sample as f64 / sample_rate + phase;
                amplitude * if sine { angle.sin() } else { angle.cos() }
            })
            .collect();
        (signal, 1.0 / sample_rate, frequency)
    }

    #[test]
    fn dft_phasor_obeys_exp_minus_i_omega_t_contract() {
        let amplitude = 2.75;
        let phase = 0.37;
        let (cosine, dt, frequency) = complete_bin_signal(amplitude, phase, false);
        let phasor = dft_phasor(&cosine, dt, frequency);
        let expected = Complex64::from_polar(amplitude, -phase);
        assert!(
            (phasor - expected).norm() < 1e-9,
            "{phasor:?} != {expected:?}"
        );

        let (sine, dt, frequency) = complete_bin_signal(amplitude, 0.0, true);
        let phasor = dft_phasor(&sine, dt, frequency);
        assert!((phasor - Complex64::new(0.0, amplitude)).norm() < 1e-9);
    }

    #[test]
    fn phasor_ratios_do_not_depend_on_integer_cycle_tail_origin() {
        let sample_rate = 4096.0;
        let frequency = 64.0;
        let samples_per_cycle = (sample_rate / frequency) as usize;
        let tail_samples = 32 * samples_per_cycle;
        let make_tail = |start: usize, amplitude: f64, phase: f64| {
            (start..start + tail_samples)
                .map(|sample| {
                    amplitude * (TAU * frequency * sample as f64 / sample_rate + phase).cos()
                })
                .collect::<Vec<_>>()
        };
        let ratio_at = |start| {
            let numerator = make_tail(start, 2.0, 0.4);
            let denominator = make_tail(start, 0.5, -0.2);
            dft_phasor(&numerator, 1.0 / sample_rate, frequency)
                / dft_phasor(&denominator, 1.0 / sample_rate, frequency)
        };
        // Keep an integer-cycle tail length, but start the second tail at a
        // genuinely different absolute phase rather than an integer number
        // of cycles later.
        assert!((ratio_at(0) - ratio_at(7)).norm() < 1e-12);
    }

    #[test]
    fn amplitude_is_exactly_the_phasor_norm() {
        let (signal, dt, frequency) = complete_bin_signal(1.25, -0.6, false);
        assert_eq!(
            dft_amplitude(&signal, dt, frequency),
            dft_phasor(&signal, dt, frequency).norm()
        );
    }

    #[test]
    fn thd_of_synthetic_signal_is_exact() {
        let dt = 1e-5;
        let f1 = 100.0;
        let n = 10_000;
        let signal: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                (std::f64::consts::TAU * f1 * t).sin()
                    + 0.02 * (2.0 * std::f64::consts::TAU * f1 * t).sin()
                    + 0.01 * (3.0 * std::f64::consts::TAU * f1 * t).sin()
            })
            .collect();
        let thd = thd_fraction(&signal, dt, f1, 5);
        let exact = (0.02f64.powi(2) + 0.01f64.powi(2)).sqrt();
        assert!((thd - exact).abs() < 1e-9, "thd {thd}, exact {exact}");

        let q_spl = port_spl_db(1e-4, 100.0, 1.2, 1.0);
        assert!((q_spl - 49.54).abs() < 0.1);
    }

    #[test]
    fn imd_of_two_tone_signal_is_exact() {
        let dt = 1e-5;
        let (f1, f2) = (100.0, 1000.0);
        let n = 10_000;
        let signal: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                (std::f64::consts::TAU * f1 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * f2 * t).sin()
                    + 0.01 * (std::f64::consts::TAU * (f2 - f1) * t).sin()
            })
            .collect();
        let imd = imd_fraction(&signal, dt, f1, f2);
        assert!((imd - 0.02).abs() < 1e-3, "imd {imd}");
    }

    #[test]
    fn degenerate_spectra_return_zero_without_panicking() {
        assert_eq!(dft_amplitude(&[], 1e-4, 100.0), 0.0);
        assert_eq!(dft_amplitude(&[1.0], 1e-4, 100.0), 0.0);
        assert_eq!(thd_fraction(&[0.0; 32], 1e-4, 100.0, 5), 0.0);
        assert_eq!(thd_fraction(&[1.0; 32], 1e-4, 100.0, 1), 0.0);
        assert_eq!(imd_fraction(&[0.0; 32], 1e-4, 100.0, 1000.0), 0.0);
        for (dt, frequency) in [
            (0.0, 100.0),
            (-1.0, 100.0),
            (f64::NAN, 100.0),
            (1e-4, 0.0),
            (1e-4, -100.0),
            (1e-4, f64::INFINITY),
        ] {
            assert_eq!(
                dft_phasor(&[0.0; 32], dt, frequency),
                Complex64::new(0.0, 0.0)
            );
        }
        assert_eq!(
            dft_phasor(&[0.0, f64::NAN], 1e-4, 100.0),
            Complex64::new(0.0, 0.0)
        );
    }

    #[test]
    fn coupled_observables_share_tail_and_recover_complex_ratios() {
        use crate::port_flow::coupled::CoupledPortResult;

        let frequency = 8.0;
        let samples_per_cycle = 64;
        let dt = 1.0 / (frequency * samples_per_cycle as f64);
        let cycles = 20;
        let n = cycles * samples_per_cycle;
        let pressure_phasor = Complex64::from_polar(2.0, -0.2);
        let flow_phasor = Complex64::from_polar(0.5, 0.35);
        let diaphragm_phasor = Complex64::from_polar(0.8, -0.1);
        let synthesize = |phasor: Complex64| {
            (0..n)
                .map(|sample| {
                    let phase = TAU * frequency * sample as f64 * dt;
                    phasor.re * phase.cos() + phasor.im * phase.sin()
                })
                .collect::<Vec<_>>()
        };
        let result = CoupledPortResult {
            time_s: (1..=n).map(|sample| sample as f64 * dt).collect(),
            port_flow_m3s: synthesize(flow_phasor),
            exterior_cap_flow_m3s: synthesize(flow_phasor),
            box_cap_outward_flow_m3s: synthesize(-flow_phasor),
            diaphragm_volume_velocity_m3s: synthesize(diaphragm_phasor),
            box_pressure_pa: synthesize(pressure_phasor),
            displacement_m: vec![0.0; n],
            current_a: vec![0.0; n],
            box_conservation_residual_pa: vec![0.0; n],
            energy_steps: vec![CoupledEnergyStep::default(); n],
            linear_solves: Vec::new(),
            accepted_linear_relative_residual: Vec::new(),
            field_snapshots: Vec::new(),
            final_flow_state: None,
            steps_completed: n,
        };
        let observables = coupled_complex_observables(
            &result,
            dt,
            frequency,
            samples_per_cycle,
            8,
            12,
            0.04,
            1.2,
            343.0,
        )
        .unwrap();
        assert!((observables.box_pressure_pa - pressure_phasor).norm() < 1e-12);
        assert!((observables.port_flow_m3s - flow_phasor).norm() < 1e-12);
        assert!((observables.diaphragm_flow_m3s - diaphragm_phasor).norm() < 1e-12);
        assert!(
            (observables.port_impedance_pa_s_per_m3 - pressure_phasor / flow_phasor).norm() < 1e-12
        );
        assert!(
            (observables.direct_box_to_port_flow_ratio - flow_phasor / diaphragm_phasor).norm()
                < 1e-12
        );
        assert!(observables.cap_continuity_relative_error < 1e-12);

        let observables_16 = coupled_complex_observables(
            &result,
            dt,
            frequency,
            samples_per_cycle,
            16,
            4,
            0.04,
            1.2,
            343.0,
        )
        .unwrap();
        assert!(
            (observables_16.port_impedance_pa_s_per_m3 - observables.port_impedance_pa_s_per_m3)
                .norm()
                < 1e-12
        );
    }

    fn linear_reference_gate_fixture() -> (
        CoupledComplexObservables,
        ComplexPortTransfer,
        CoupledEnergySummary,
    ) {
        let impedance = Complex64::new(2.0, 0.5);
        let flow_ratio = Complex64::new(0.7, -0.2);
        (
            CoupledComplexObservables {
                tail_samples: 512,
                box_pressure_pa: Complex64::new(2.0, 0.5),
                port_flow_m3s: Complex64::new(1.0, 0.0),
                diaphragm_flow_m3s: Complex64::new(1.0, 0.0),
                port_impedance_pa_s_per_m3: impedance,
                direct_box_to_port_flow_ratio: flow_ratio,
                box_to_port_flow_ratio_from_impedance: flow_ratio,
                cap_continuity_relative_error: 1.0e-9,
            },
            ComplexPortTransfer {
                frequency_hz: 80.0,
                polynomial_degree: 2,
                quadrature_order: 5,
                scalar_dofs: 128,
                box_pressure_rms_pa: Complex64::new(1.0, 0.0),
                exterior_pressure_rms_pa: Complex64::new(0.0, 0.0),
                port_volume_velocity_rms_m3_per_s: Complex64::new(1.0, 0.0),
                strong_exterior_volume_velocity_rms_m3_per_s: Complex64::new(1.0, 0.0),
                weak_box_outward_volume_velocity_rms_m3_per_s: Complex64::new(-1.0, 0.0),
                weak_net_storage_volume_velocity_rms_m3_per_s: Complex64::new(0.0, 0.0),
                strong_weak_exterior_flux_relative_difference: 1.0e-9,
                port_impedance_pa_s_per_m3: impedance,
                box_to_port_flow_ratio: flow_ratio,
                box_to_port_flow_ratio_condition_number: 1.0,
                relative_residual: 1.0e-10,
            },
            CoupledEnergySummary {
                source_j: 1.0,
                stored_delta_j: 0.1,
                copper_loss_j: 0.4,
                mechanical_damping_j: 0.2,
                viscous_dissipation_j: 0.2,
                physical_dissipation_j: 0.8,
                exterior_export_j: 0.1,
                numerical_residual_j: 0.0,
                energy_scale_j: 1.0,
                numerical_relative_residual: 1.0e-10,
                algebraic_relative_closure: 1.0e-10,
                physical_relative_closure: 0.01,
                interface_relative_residual: 1.0e-10,
            },
        )
    }

    #[test]
    fn linear_reference_gate_accepts_consistent_low_drive_candidate() {
        let (observables, reference, energy) = linear_reference_gate_fixture();
        let report = validate_coupled_against_linear_reference(
            &LinearReferenceGateInput {
                frequency_hz: 80.0,
                observables: &observables,
                reference: &reference,
                energy: &energy,
                thd_fraction: 1.0e-3,
                mach: 1.0e-2,
            },
            LinearReferenceGateConfig::default(),
        )
        .expect("consistent low-drive candidate should pass");
        assert_eq!(report.impedance_relative_error, 0.0);
        assert_eq!(report.flow_ratio_relative_error, 0.0);
        assert_eq!(report.transfer_identity_relative_error, 0.0);
    }

    #[test]
    fn linear_reference_gate_rejects_nonlinear_claim() {
        let (observables, reference, energy) = linear_reference_gate_fixture();
        let error = validate_coupled_against_linear_reference(
            &LinearReferenceGateInput {
                frequency_hz: 80.0,
                observables: &observables,
                reference: &reference,
                energy: &energy,
                thd_fraction: 0.01,
                mach: 1.0e-2,
            },
            LinearReferenceGateConfig::default(),
        )
        .expect_err("high-THD candidate must not pass a linear claim");
        assert!(error.to_string().contains("low-drive regime"));
    }

    #[test]
    fn linear_reference_gate_rejects_reference_and_energy_audit_failures() {
        let (observables, mut reference, mut energy) = linear_reference_gate_fixture();
        reference.relative_residual = 1.0e-6;
        let error = validate_coupled_against_linear_reference(
            &LinearReferenceGateInput {
                frequency_hz: 80.0,
                observables: &observables,
                reference: &reference,
                energy: &energy,
                thd_fraction: 1.0e-3,
                mach: 1.0e-2,
            },
            LinearReferenceGateConfig::default(),
        )
        .expect_err("reference residual must be gated");
        assert!(error.to_string().contains("residual/flux audit"));

        reference.relative_residual = 1.0e-10;
        energy.physical_relative_closure = 0.06;
        let error = validate_coupled_against_linear_reference(
            &LinearReferenceGateInput {
                frequency_hz: 80.0,
                observables: &observables,
                reference: &reference,
                energy: &energy,
                thd_fraction: 1.0e-3,
                mach: 1.0e-2,
            },
            LinearReferenceGateConfig::default(),
        )
        .expect_err("energy closure must be gated");
        assert!(error.to_string().contains("energy audit"));
    }
}
