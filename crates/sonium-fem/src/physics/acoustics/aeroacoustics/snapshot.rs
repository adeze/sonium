//! Snapshot-to-phasor coupling for one-way aeroacoustic analogies.
//!
//! Time-domain flow snapshots become harmonic analogy inputs through one
//! documented path: Hann-windowed DFT peak phasors, Lighthill stress
//! tensors, FW-H surface patches, and axisymmetric modal (ring and
//! revolved-strip) sources. All phasors use the peak amplitude with the
//! `exp(+iωt)` convention shared by the analogy evaluators: a real signal
//! `s(t)` with phasor `P` satisfies `s(t) = Re(P exp(iωt))`.
//!
//! Field sampling stays consumer-side (consumers own their function
//! spaces); this module owns the DFT, the phasor convention, and the
//! tensor/patch/modal assembly from per-cell and per-face time series.

// Rust guideline compliant 2026-02-21

use std::f64::consts::TAU;

use num_complex::Complex64;
use sonium_wave::analytical::azimuthal::AzimuthalMode;

use super::distributed::{AreaQuadrupoleCell, VolumeQuadrupoleCell};
use super::error::{AeroacousticsError, require_finite, require_positive};
use super::fwh::FwhSurfacePatch;
use super::fwh_2d_axi::{FwhContourSegment, FwhRevolvedPatch};
use super::lighthill_2d_axi::LighthillRingSource;

/// Hann-windowed DFT peak phasor of a real uniform signal.
///
/// Applies a periodic Hann window (which preserves exact-bin
/// orthogonality for complete periods), projects onto `cos(ωt)` and
/// `sin(ωt)`, and normalizes by the single-sided Hann coherent gain so a
/// bin-centered tone `A cos(ωt + φ)` returns the peak phasor
/// `A exp(iφ)`. Matches `dft_phasor(...).conj()` of the legacy
/// port-flow analysis helper sample for sample.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for fewer than two samples, a
/// non-positive sample interval or frequency, or non-finite data.
pub fn dft_peak_phasor(
    signal: &[f64],
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<Complex64, AeroacousticsError> {
    if signal.len() < 2 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "snapshot sample count",
            value: signal.len() as f64,
        });
    }
    require_positive("sample_interval_s", sample_interval_s)?;
    require_positive("frequency_hz", frequency_hz)?;
    if signal.iter().any(|sample| !sample.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "snapshot signal",
        });
    }
    let count = signal.len() as f64;
    let omega = TAU * frequency_hz;
    let mut real = 0.0;
    let mut imaginary = 0.0;
    for (index, &sample) in signal.iter().enumerate() {
        let time = index as f64 * sample_interval_s;
        let window = 0.5 * (1.0 - (TAU * index as f64 / count).cos());
        let windowed = sample * window;
        real += windowed * (omega * time).cos();
        imaginary += windowed * (omega * time).sin();
    }
    // Single-sided peak normalization: the Hann coherent gain is 1/2 and
    // the negative-frequency image doubles the one-sided spectrum. The
    // conjugation converts the exp(-iwt) projection to the exp(+iwt)
    // phasor convention of the analogy evaluators.
    let scale = 2.0 / (count / 2.0);
    Ok(scale * Complex64::new(real, -imaginary))
}

/// Harmonic Lighthill stress phasor from a velocity history.
///
/// Forms the `ρ u_i u_j` series for each tensor pair and transforms it
/// with [`dft_peak_phasor`]. The result is the volume-averaged stress
/// when the input series is; multiply by the cell measure to form the
/// quadrupole moment (see [`volume_cell_from_series`]).
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid DFT inputs or a
/// non-positive density.
pub fn lighthill_stress_phasor(
    velocities: &[[f64; 3]],
    density: f64,
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<[[Complex64; 3]; 3], AeroacousticsError> {
    require_positive("density", density)?;
    require_positive("sample_interval_s", sample_interval_s)?;
    require_positive("frequency_hz", frequency_hz)?;
    if velocities.len() < 2 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "snapshot sample count",
            value: velocities.len() as f64,
        });
    }
    if velocities
        .iter()
        .flat_map(|velocity| velocity.iter())
        .any(|value| !value.is_finite())
    {
        return Err(AeroacousticsError::NonFiniteField {
            name: "snapshot velocities",
        });
    }
    let mut stress = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let series: Vec<f64> = velocities
                .iter()
                .map(|velocity| density * velocity[i] * velocity[j])
                .collect();
            stress[i][j] = dft_peak_phasor(&series, sample_interval_s, frequency_hz)?;
        }
    }
    Ok(stress)
}

/// Harmonic planar Lighthill stress phasor from an in-plane history.
///
/// Two-dimensional counterpart of [`lighthill_stress_phasor`] for the
/// planar quadrupole evaluators.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid DFT inputs or a
/// non-positive density.
pub fn lighthill_stress_phasor_2d(
    velocities: &[[f64; 2]],
    density: f64,
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<[[Complex64; 2]; 2], AeroacousticsError> {
    require_positive("density", density)?;
    require_positive("sample_interval_s", sample_interval_s)?;
    require_positive("frequency_hz", frequency_hz)?;
    if velocities.len() < 2 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "snapshot sample count",
            value: velocities.len() as f64,
        });
    }
    if velocities
        .iter()
        .flat_map(|velocity| velocity.iter())
        .any(|value| !value.is_finite())
    {
        return Err(AeroacousticsError::NonFiniteField {
            name: "snapshot velocities",
        });
    }
    let mut stress = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let series: Vec<f64> = velocities
                .iter()
                .map(|velocity| density * velocity[i] * velocity[j])
                .collect();
            stress[i][j] = dft_peak_phasor(&series, sample_interval_s, frequency_hz)?;
        }
    }
    Ok(stress)
}

/// Builds one 3-D quadrupole cell from a velocity history.
///
/// Combines [`lighthill_stress_phasor`] with the cell centroid and
/// volume; the distributed evaluator multiplies stress by volume.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid DFT inputs, density,
/// position, or volume.
pub fn volume_cell_from_series(
    position: [f64; 3],
    volume: f64,
    velocities: &[[f64; 3]],
    density: f64,
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<VolumeQuadrupoleCell, AeroacousticsError> {
    if position.iter().any(|value| !value.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "cell position",
        });
    }
    require_positive("cell volume", volume)?;
    let stress = lighthill_stress_phasor(velocities, density, sample_interval_s, frequency_hz)?;
    Ok(VolumeQuadrupoleCell {
        position,
        volume,
        stress,
    })
}

/// Builds one planar quadrupole cell from an in-plane history.
///
/// Two-dimensional counterpart of [`volume_cell_from_series`].
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid DFT inputs, density,
/// position, or area.
pub fn area_cell_from_series(
    position: [f64; 2],
    area: f64,
    velocities: &[[f64; 2]],
    density: f64,
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<AreaQuadrupoleCell, AeroacousticsError> {
    if position.iter().any(|value| !value.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "cell position",
        });
    }
    require_positive("cell area", area)?;
    let stress = lighthill_stress_phasor_2d(velocities, density, sample_interval_s, frequency_hz)?;
    Ok(AreaQuadrupoleCell {
        position,
        area,
        stress,
    })
}

/// Builds one FW-H patch from face-averaged histories.
///
/// Transforms the pressure and outward-normal-velocity series with
/// [`dft_peak_phasor`] and packs the validated patch.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for mismatched series lengths,
/// invalid DFT inputs, or invalid patch geometry.
pub fn fwh_patch_from_series(
    position: [f64; 3],
    outward_normal: [f64; 3],
    area: f64,
    pressure: &[f64],
    normal_velocity: &[f64],
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<FwhSurfacePatch, AeroacousticsError> {
    if pressure.len() != normal_velocity.len() {
        return Err(AeroacousticsError::NonFiniteField {
            name: "matched pressure/velocity snapshot series",
        });
    }
    let pressure_phasor = dft_peak_phasor(pressure, sample_interval_s, frequency_hz)?;
    let velocity_phasor = dft_peak_phasor(normal_velocity, sample_interval_s, frequency_hz)?;
    FwhSurfacePatch::new(
        position,
        outward_normal,
        area,
        pressure_phasor,
        velocity_phasor,
    )
}

/// Builds one planar FW-H contour segment from histories.
///
/// Two-dimensional counterpart of [`fwh_patch_from_series`].
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for mismatched series lengths,
/// invalid DFT inputs, or invalid segment geometry.
pub fn fwh_contour_from_series(
    position: [f64; 2],
    outward_normal: [f64; 2],
    length: f64,
    pressure: &[f64],
    normal_velocity: &[f64],
    sample_interval_s: f64,
    frequency_hz: f64,
) -> Result<FwhContourSegment, AeroacousticsError> {
    if pressure.len() != normal_velocity.len() {
        return Err(AeroacousticsError::NonFiniteField {
            name: "matched pressure/velocity snapshot series",
        });
    }
    let pressure_phasor = dft_peak_phasor(pressure, sample_interval_s, frequency_hz)?;
    let velocity_phasor = dft_peak_phasor(normal_velocity, sample_interval_s, frequency_hz)?;
    FwhContourSegment::new(
        position,
        outward_normal,
        length,
        pressure_phasor,
        velocity_phasor,
    )
}

/// Extracts one azimuthal Fourier coefficient from hoop phasors.
///
/// Returns `c_m = (1/N) Σ_k s_k exp(-imφ_k)` for phasors sampled at
/// uniform azimuths `φ_k = 2πk/N`. This pairs with the ring and
/// revolved evaluators, which synthesize the field with `exp(+imφ)`.
/// Modes alias modulo `N`: callers resolving spinning order `m` need
/// `N > 2|m|` hoop stations for real-valued fields.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an empty hoop or non-finite
/// phasors.
pub fn azimuthal_mode_coefficient(
    hoop_phasors: &[Complex64],
    mode_order: i32,
) -> Result<Complex64, AeroacousticsError> {
    if hoop_phasors.is_empty() {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "hoop sample count",
            value: 0.0,
        });
    }
    if hoop_phasors
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(AeroacousticsError::NonFiniteField {
            name: "hoop phasors",
        });
    }
    let count = hoop_phasors.len() as f64;
    let mut coefficient = Complex64::new(0.0, 0.0);
    for (index, phasor) in hoop_phasors.iter().enumerate() {
        let azimuth = TAU * index as f64 / count;
        let phase = Complex64::from_polar(1.0, -(f64::from(mode_order)) * azimuth);
        coefficient += phasor * phase;
    }
    Ok(coefficient / count)
}

/// Extracts one spinning-mode phasor from hoop time histories.
///
/// Transforms each azimuth station with [`dft_peak_phasor`], then
/// extracts the `exp(+imφ)` coefficient with
/// [`azimuthal_mode_coefficient`]. A signal spinning as `cos(ωt - mφ)`
/// carries mode `-m` in this basis (its time phasor is `A exp(-imφ)`).
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an empty hoop, ragged series, or
/// invalid DFT inputs.
pub fn spinning_phasor(
    hoop_signals: &[&[f64]],
    sample_interval_s: f64,
    frequency_hz: f64,
    mode_order: i32,
) -> Result<Complex64, AeroacousticsError> {
    if hoop_signals.is_empty() {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "hoop sample count",
            value: 0.0,
        });
    }
    let expected = hoop_signals[0].len();
    if hoop_signals.iter().any(|signal| signal.len() != expected) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "uniform hoop snapshot series",
        });
    }
    let mut phasors = Vec::with_capacity(hoop_signals.len());
    for signal in hoop_signals {
        phasors.push(dft_peak_phasor(signal, sample_interval_s, frequency_hz)?);
    }
    azimuthal_mode_coefficient(&phasors, mode_order)
}

/// Revolved volume per radian of one meridional cell.
///
/// A meridional cell of area `A` at radius `r` sweeps volume `A·r`
/// per radian of azimuth; multiply a volume-averaged stress by this
/// to form the ring moment per radian.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for a non-positive area or a
/// negative radius.
pub fn revolved_cell_volume_per_rad(area: f64, radius: f64) -> Result<f64, AeroacousticsError> {
    require_positive("meridional cell area", area)?;
    require_finite("meridional cell radius", radius)?;
    if radius < 0.0 {
        return Err(AeroacousticsError::NonFiniteParameter {
            name: "meridional cell radius",
            value: radius,
        });
    }
    Ok(area * radius)
}

/// Builds one spinning Lighthill ring from a modal cylindrical stress.
///
/// Scales the volume-averaged cylindrical `(r, φ, z)` stress phasor by
/// `volume_per_rad` (see [`revolved_cell_volume_per_rad`]) into the
/// moment per radian the ring evaluator integrates.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid ring parameters or a
/// non-positive volume per radian.
pub fn ring_source_from_modal_stress(
    ring_radius_m: f64,
    axial_position_m: f64,
    azimuthal: AzimuthalMode,
    stress_cylindrical: [[Complex64; 3]; 3],
    volume_per_rad: f64,
) -> Result<LighthillRingSource, AeroacousticsError> {
    require_positive("volume_per_rad", volume_per_rad)?;
    let moment = stress_cylindrical.map(|row| row.map(|entry| entry * volume_per_rad));
    LighthillRingSource::new(ring_radius_m, axial_position_m, azimuthal, moment)
}

/// Builds one revolved FW-H patch from modal meridional data.
///
/// Thin validated constructor pairing [`FwhRevolvedPatch::new`] with
/// [`spinning_phasor`] outputs: pressure and normal-velocity modal
/// amplitudes in the `exp(+imφ)` basis at one meridional strip.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid strip geometry or
/// non-finite modal amplitudes.
pub fn revolved_patch_from_modal_data(
    node_a: [f64; 2],
    node_b: [f64; 2],
    outward_normal: [f64; 2],
    azimuthal: AzimuthalMode,
    pressure: Complex64,
    normal_velocity: Complex64,
) -> Result<FwhRevolvedPatch, AeroacousticsError> {
    FwhRevolvedPatch::new(
        node_a,
        node_b,
        outward_normal,
        azimuthal,
        pressure,
        normal_velocity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const DENSITY: f64 = 1.2;

    fn tone(samples: usize, dt: f64, frequency: f64, amplitude: f64, phase: f64) -> Vec<f64> {
        (0..samples)
            .map(|index| amplitude * (TAU * frequency * index as f64 * dt + phase).cos())
            .collect()
    }

    #[test]
    fn dft_peak_phasor_recovers_bin_centered_tones() {
        // 64 samples at 6.4 kHz: 100 Hz lands exactly on bin 1.
        let signal = tone(64, 1.0 / 6_400.0, 100.0, 2.0, 0.0);
        let phasor = dft_peak_phasor(&signal, 1.0 / 6_400.0, 100.0).unwrap();
        assert!((phasor.re - 2.0).abs() < 1e-9, "real={}", phasor.re);
        assert!(phasor.im.abs() < 1e-9, "imag={}", phasor.im);
        // sin(wt) = Re(-i A exp(iwt)).
        let sine = tone(64, 1.0 / 6_400.0, 100.0, 2.0, -std::f64::consts::FRAC_PI_2);
        let sine_phasor = dft_peak_phasor(&sine, 1.0 / 6_400.0, 100.0).unwrap();
        assert!(sine_phasor.re.abs() < 1e-9);
        assert!((sine_phasor.im + 2.0).abs() < 1e-9);
        // Zero signal gives a zero phasor.
        let zero = dft_peak_phasor(&vec![0.0; 64], 1.0 / 6_400.0, 100.0).unwrap();
        assert_eq!(zero, Complex64::new(0.0, 0.0));
    }

    #[test]
    fn dft_peak_phasor_rejects_bad_inputs() {
        assert!(dft_peak_phasor(&[1.0], 0.001, 100.0).is_err());
        assert!(dft_peak_phasor(&vec![1.0; 8], 0.0, 100.0).is_err());
        assert!(dft_peak_phasor(&vec![1.0; 8], 0.001, 0.0).is_err());
        assert!(dft_peak_phasor(&vec![1.0, f64::NAN], 0.001, 100.0).is_err());
    }

    #[test]
    fn lighthill_stress_phasor_doubles_axial_oscillation_frequency() {
        // u = A cos(wt): rho u^2 = rho A^2/2 (1 + cos 2wt).
        let velocities: Vec<[f64; 3]> = tone(64, 1.0 / 6_400.0, 100.0, 3.0, 0.0)
            .into_iter()
            .map(|u| [u, 0.0, 0.0])
            .collect();
        let stress = lighthill_stress_phasor(&velocities, DENSITY, 1.0 / 6_400.0, 200.0).unwrap();
        let expected = DENSITY * 9.0 / 2.0;
        assert!((stress[0][0].re - expected).abs() < 1e-9);
        assert!(stress[0][0].im.abs() < 1e-9);
        assert!(stress[0][1].norm() < 1e-12);
        assert!(stress[1][1].norm() < 1e-12);
    }

    #[test]
    fn volume_cell_from_series_matches_moment_convention() {
        let velocities = vec![[1.0, 0.5, 0.0]; 32];
        // Steady flow has no finite-frequency content: 62.5 Hz is DFT bin
        // 2 of the 32 ms record, where the periodic Hann sums to zero.
        let cell = volume_cell_from_series([0.1, 0.2, 0.3], 0.5, &velocities, DENSITY, 0.001, 62.5)
            .unwrap();
        assert_eq!(cell.position, [0.1, 0.2, 0.3]);
        assert!((cell.volume - 0.5).abs() < 1e-14);
        assert!(cell.stress[0][0].norm() < 1e-6);
    }

    #[test]
    fn fwh_builders_reject_ragged_series_and_degenerate_geometry() {
        let pressure = vec![1.0; 16];
        let short = vec![1.0; 8];
        assert!(
            fwh_patch_from_series(
                [0.0; 3],
                [1.0, 0.0, 0.0],
                0.5,
                &pressure,
                &short,
                0.001,
                100.0
            )
            .is_err()
        );
        assert!(
            fwh_contour_from_series(
                [0.0; 2],
                [1.0, 0.0],
                0.0,
                &pressure,
                &pressure,
                0.001,
                100.0
            )
            .is_err()
        );
    }

    #[test]
    fn azimuthal_extraction_pairs_with_modal_synthesis() {
        // Pure exp(+i3phi) phasors: extraction at order 3 returns 1.
        let order = 3;
        let phasors: Vec<Complex64> = (0..24)
            .map(|k| Complex64::from_polar(1.0, f64::from(order) * TAU * f64::from(k) / 24.0))
            .collect();
        let matched = azimuthal_mode_coefficient(&phasors, order).unwrap();
        assert!((matched - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        let rejected = azimuthal_mode_coefficient(&phasors, order + 1).unwrap();
        assert!(rejected.norm() < 1e-12);
        assert!(azimuthal_mode_coefficient(&[], order).is_err());
    }

    #[test]
    fn spinning_phasor_identifies_modal_order_of_traveling_wave() {
        // cos(wt - 2phi) spins with mode -2 in the exp(+imphi) basis.
        let stations = 16;
        let hoop: Vec<Vec<f64>> = (0..stations)
            .map(|k| {
                let azimuth = TAU * f64::from(k) / f64::from(stations);
                tone(64, 1.0 / 6_400.0, 100.0, 1.5, -2.0 * azimuth)
            })
            .collect();
        let refs: Vec<&[f64]> = hoop.iter().map(Vec::as_slice).collect();
        let matched = spinning_phasor(&refs, 1.0 / 6_400.0, 100.0, -2).unwrap();
        assert!((matched - Complex64::new(1.5, 0.0)).norm() < 1e-9);
        let rejected = spinning_phasor(&refs, 1.0 / 6_400.0, 100.0, 2).unwrap();
        assert!(rejected.norm() < 1e-9);
    }

    #[test]
    fn revolved_helpers_assemble_moment_per_radian() {
        let volume_per_rad = revolved_cell_volume_per_rad(0.02, 0.25).unwrap();
        assert!((volume_per_rad - 0.005).abs() < 1e-14);
        assert!(revolved_cell_volume_per_rad(0.02, -0.1).is_err());
        let mut stress = [[Complex64::new(0.0, 0.0); 3]; 3];
        stress[0][0] = Complex64::new(4.0, 0.0);
        let ring =
            ring_source_from_modal_stress(0.25, 0.1, AzimuthalMode::new(2), stress, volume_per_rad)
                .unwrap();
        assert!((ring.stress_moment_per_rad[0][0].re - 0.02).abs() < 1e-14);
        let patch = revolved_patch_from_modal_data(
            [0.2, 0.0],
            [0.2, 0.1],
            [1.0, 0.0],
            AzimuthalMode::axisymmetric(),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.5),
        )
        .unwrap();
        assert_eq!(patch.node_a, [0.2, 0.0]);
    }
}
