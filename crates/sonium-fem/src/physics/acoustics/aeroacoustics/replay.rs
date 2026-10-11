//! Golden replay glue for aeroacoustic analogy checks.
//!
//! Lane 5D validates the existing Lighthill, FW-H, and revolved FW-H
//! evaluators against engine time histories without modifying them
//! (Phase-5 contract P5C1: wrap, don't edit). This module is the only
//! new library code the lane needs: it turns raw history series into
//! compact harmonic sources through the existing
//! [`snapshot`](super::snapshot) path and scores the radiation checks
//! the `qa-fem-caa` G-leg reports per probe.
//!
//! # Source classification
//!
//! * Point quadrupoles ([`LighthillQuadrupole`](super::lighthill::LighthillQuadrupole))
//!   and spinning rings are quadrupole-order sources, compact by
//!   construction (`ka -> 0` for points; callers keep `ka_ring << 1`
//!   for rings). FW-H patches superpose a monopole thickness term
//!   (normal velocity) and a dipole loading term (surface pressure).
//! * Near versus far is pinned by `kr`: `kr << 1` is reactive
//!   (`1/r^2`, `1/r^3` terms dominate) while `kr >> 1` is radiating
//!   (`1/r`). [`lighthill_far_field_gap`] measures the reactive
//!   remainder at caller-chosen `kr`, and [`observer_at_kr`] places
//!   observers at prescribed `kr`.
//! * Far-field versus reactive terms are never point-matched alone:
//!   the battery checks limit behavior (far-field agreement,
//!   `1/r` spreading, reciprocity, ring-to-point and revolved-to-3D
//!   convergence) plus the directional factor
//!   [`directivity_factor`] whose sphere normalization the
//!   `caa_lighthill` Wolfram fixture pins independently.
//!
//! # Data honesty
//!
//! Golden `~k` ordinal identities live in deck TH-card order (a
//! Phase-4 reader concern per `goldens/SCHEMA.md`) and are never
//! assumed here: probe series enter as raw unsteady signals, are
//! mean-removed into acoustic fluctuations ([`detrend`]), and feed
//! the kernel checks at their spectral peak bin ([`peak_tone`]).
//! Conservation rows in the G-leg use bare-name channels only.
//! The propagation medium (`rho0`, `c0`) and the replay geometry are
//! documented harness parameters owned by the caller.
//!
//! # References
//!
//! * M. J. Lighthill, "On Sound Generated Aerodynamically. I.
//!   General Theory", *Proc. Roy. Soc. A* 211 (1952), 564--587.
//! * J. E. Ffowcs Williams and D. L. Hawkings, "Sound Generation by
//!   Turbulence and Surfaces in Arbitrary Motion", *Phil. Trans.
//!   Roy. Soc. A* 264 (1969), 321--342.
//! * `sonium/crates/sonium-qa/goldens/SCHEMA.md` (comparison rule).

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use std::f64::consts::TAU;

use sonium_wave::analytical::azimuthal::AzimuthalMode;

use super::error::{AeroacousticsError, require_finite, require_positive};
use super::fwh::{FwhComponents, FwhSurfacePatch, stationary_fwh_pressure};
use super::fwh_2d_axi::{FwhRevolvedPatch, RevolvedQuadrature, stationary_fwh_ring_pressure};
use super::lighthill::{LighthillQuadrupole, lighthill_far_field_pressure, lighthill_pressure};
use super::lighthill_2d_axi::{LighthillRingSource, lighthill_ring_pressure};
use super::snapshot::dft_peak_phasor;

/// Near-zero skip band mirroring vendor `Zero_tolerance`.
///
/// Samples where both sides sit below this, or differ by less than
/// twice it, are engine noise rather than signal; see
/// `sonium/crates/sonium-qa/goldens/SCHEMA.md` and
/// `sonium_qa::rad::oc_golden::ZERO_TOL`, which this intentionally
/// duplicates so `sonium-fem` stays dependency-free of `sonium-qa`.
pub const ZERO_TOLERANCE: f64 = 5e-14;

/// Unit-normal tolerance shared with the FW-H patch validators.
///
/// Patch constructors accept `||n| - 1| <= 1e-8`; replay geometry
/// checks use the same band so harnesses never disagree with the
/// evaluators about what counts as a unit direction.
const UNIT_DIRECTION_TOL: f64 = 1e-8;

/// Tests one sample against the golden comparison rule.
///
/// Passes when `|actual - expected| <= abs_tol + rel_tol * |expected|`,
/// with the vendor near-zero skip: both sides below
/// [`ZERO_TOLERANCE`], or a difference below twice it, pass
/// regardless. Mirrors `compare_history` sample for sample.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::within_band;
///
/// assert!(within_band(1.0005, 1.0, 1e-12, 6.6e-4));
/// assert!(!within_band(1.002, 1.0, 1e-12, 6.6e-4));
/// // Near-zero noise on both sides always passes.
/// assert!(within_band(1e-15, -1e-15, 0.0, 0.0));
/// ```
pub fn within_band(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) -> bool {
    let difference = (actual - expected).abs();
    if (expected.abs() < ZERO_TOLERANCE && actual.abs() < ZERO_TOLERANCE)
        || difference < 2.0 * ZERO_TOLERANCE
    {
        return true;
    }
    difference <= abs_tol + rel_tol * expected.abs()
}

/// Mean step of a strictly increasing time grid.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for fewer than two stamps, a
/// non-finite stamp, or a non-positive step.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::mean_step;
///
/// let step = mean_step(&[0.0, 0.5, 1.0]).unwrap();
/// assert!((step - 0.5).abs() < 1e-15);
/// ```
pub fn mean_step(times: &[f64]) -> Result<f64, AeroacousticsError> {
    if times.len() < 2 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "replay time stamp count",
            value: times.len() as f64,
        });
    }
    for stamp in times {
        require_finite("replay time stamp", *stamp)?;
    }
    for pair in times.windows(2) {
        let step = pair[1] - pair[0];
        if !step.is_finite() || step <= 0.0 {
            return Err(AeroacousticsError::NonPositiveParameter {
                name: "replay time step",
                value: step,
            });
        }
    }
    Ok((times[times.len() - 1] - times[0]) / (times.len() - 1) as f64)
}

/// Relative non-uniformity of a time grid.
///
/// Returns `max|dt_i - mean| / mean`, the figure the G-leg bands with
/// the golden `TIME STEP` tolerance before trusting the uniform-grid
/// DFT. Zero for an exactly uniform grid.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] under the same conditions as
/// [`mean_step`].
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::grid_nonuniformity;
///
/// assert!(grid_nonuniformity(&[0.0, 0.5, 1.0]).unwrap() < 1e-15);
/// ```
pub fn grid_nonuniformity(times: &[f64]) -> Result<f64, AeroacousticsError> {
    let mean = mean_step(times)?;
    let mut worst = 0.0f64;
    for pair in times.windows(2) {
        let deviation = ((pair[1] - pair[0]) - mean).abs() / mean;
        worst = worst.max(deviation);
    }
    Ok(worst)
}

/// Mean-removed acoustic fluctuation of a history series.
#[derive(Debug, Clone)]
pub struct Detrended {
    /// Arithmetic mean of the input series (the DC operating point).
    pub mean: f64,
    /// Input minus [`Detrended::mean`], sample for sample.
    pub fluctuation: Vec<f64>,
}

/// Splits a history into DC mean and acoustic fluctuation.
///
/// The analogy DFT consumes fluctuations: mean flow is the operating
/// point, sound is the unsteady remainder. Fails on empty or
/// non-finite input so silent channels never become silent sources.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for an empty or non-finite series.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::detrend;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let split = detrend(&signal).unwrap();
/// assert!((split.mean - 2.5).abs() < 1e-15);
/// assert!((split.fluctuation[0] + 1.5).abs() < 1e-15);
/// ```
pub fn detrend(signal: &[f64]) -> Result<Detrended, AeroacousticsError> {
    if signal.is_empty() {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "replay signal length",
            value: 0.0,
        });
    }
    if signal.iter().any(|sample| !sample.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "replay signal",
        });
    }
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    Ok(Detrended {
        mean,
        fluctuation: signal.iter().map(|sample| sample - mean).collect(),
    })
}

/// Strongest AC tone of a fluctuation series.
#[derive(Debug, Clone, Copy)]
pub struct PeakTone {
    /// DFT bin in `1..N/2` holding the largest peak phasor.
    pub bin: usize,
    /// Bin-center frequency `bin / (N * dt)` in hertz.
    pub frequency_hz: f64,
    /// Peak phasor at the bin center (see [`dft_peak_phasor`]).
    pub phasor: Complex64,
}

/// Finds the strongest AC tone of a fluctuation series.
///
/// Scans bins `1..N/2` (DC and Nyquist excluded) with
/// [`dft_peak_phasor`] and keeps the largest magnitude; ties keep
/// the lowest bin, so selection is deterministic. Callers check
/// `|phasor|` against a documented floor before building sources.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for fewer than four samples, a
/// non-positive sample interval, or non-finite data.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::peak_tone;
///
/// // Bin-3 tone plus a weaker bin-7 tone; the peak search finds bin 3.
/// let samples: Vec<f64> = (0..64)
///     .map(|n| {
///         let phase = 2.0 * std::f64::consts::PI * f64::from(n) / 64.0;
///         2.0 * (3.0 * phase).cos() + 0.5 * (7.0 * phase).cos()
///     })
///     .collect();
/// let tone = peak_tone(&samples, 0.001).unwrap();
/// assert_eq!(tone.bin, 3);
/// assert!((tone.phasor.norm() - 2.0).abs() < 1e-9);
/// ```
pub fn peak_tone(
    fluctuation: &[f64],
    sample_interval_s: f64,
) -> Result<PeakTone, AeroacousticsError> {
    if fluctuation.len() < 4 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "peak-tone sample count",
            value: fluctuation.len() as f64,
        });
    }
    require_positive("sample_interval_s", sample_interval_s)?;
    if fluctuation.iter().any(|sample| !sample.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "replay fluctuation",
        });
    }
    let count = fluctuation.len();
    let fundamental = 1.0 / (count as f64 * sample_interval_s);
    let mut best_bin = 1;
    let mut best_phasor = Complex64::new(0.0, 0.0);
    for bin in 1..count / 2 {
        let phasor = dft_peak_phasor(fluctuation, sample_interval_s, bin as f64 * fundamental)?;
        if phasor.norm() > best_phasor.norm() {
            best_bin = bin;
            best_phasor = phasor;
        }
    }
    Ok(PeakTone {
        bin: best_bin,
        frequency_hz: best_bin as f64 * fundamental,
        phasor: best_phasor,
    })
}

/// Builds a compact quadrupole from a velocity history.
///
/// Forms the `rho0 * u_i u_j` series per tensor pair, removes the DC
/// operating point ([`detrend`]), transforms the stress fluctuation
/// with [`dft_peak_phasor`], and scales by the cell volume into the
/// [`LighthillQuadrupole`] moment. Detrending before the DFT is load
/// bearing: the periodic Hann window carries a `-N/4` sidelobe at
/// bins `+-1`, so transforming raw stresses would fold the mean-flow
/// DC into the replay tone. This is the glue
/// [`snapshot`](super::snapshot) lacks: it yields distributed cells,
/// not detrended compact point sources.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for fewer than two samples,
/// non-positive density, sample interval, frequency, or volume,
/// non-finite data, or a non-finite position.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::quadrupole_from_history;
///
/// // Axial tone: v_y = cos(bin-3 phase); v_y^2 carries bin 6 at half amplitude.
/// let history: Vec<[f64; 3]> = (0..64)
///     .map(|n| {
///         let phase = 2.0 * std::f64::consts::PI * 3.0 * f64::from(n) / 64.0;
///         [0.0, phase.cos(), 0.0]
///     })
///     .collect();
/// let source =
///     quadrupole_from_history(&history, 1.2, 0.001, 6.0 / 0.064, [0.0; 3], 1.0).unwrap();
/// assert!((source.stress_moment[1][1].norm() - 0.6).abs() < 1e-9);
/// ```
pub fn quadrupole_from_history(
    velocities: &[[f64; 3]],
    density: f64,
    sample_interval_s: f64,
    frequency_hz: f64,
    position: [f64; 3],
    volume: f64,
) -> Result<LighthillQuadrupole, AeroacousticsError> {
    require_positive("cell volume", volume)?;
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
    let mut moment = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let series: Vec<f64> = velocities
                .iter()
                .map(|velocity| density * velocity[i] * velocity[j])
                .collect();
            let fluctuation = detrend(&series)?.fluctuation;
            moment[i][j] = volume * dft_peak_phasor(&fluctuation, sample_interval_s, frequency_hz)?;
        }
    }
    LighthillQuadrupole::new(position, moment)
}

/// Places an observer at prescribed `kr` along a direction.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for a non-unit or non-finite
/// direction, or a non-positive `kr` or wavenumber.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::observer_at_kr;
///
/// let observer = observer_at_kr([0.0, 0.0, 1.0], 100.0, 2.0).unwrap();
/// assert_eq!(observer, [0.0, 0.0, 50.0]);
/// ```
pub fn observer_at_kr(
    direction: [f64; 3],
    kr: f64,
    wavenumber: f64,
) -> Result<[f64; 3], AeroacousticsError> {
    if direction.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "observer direction",
        });
    }
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if (norm - 1.0).abs() > UNIT_DIRECTION_TOL {
        return Err(AeroacousticsError::NonFiniteField {
            name: "unit observer direction",
        });
    }
    require_positive("kr", kr)?;
    require_positive("wavenumber", wavenumber)?;
    let distance = kr / wavenumber;
    Ok(direction.map(|x| x * distance))
}

/// Relative reactive remainder of a quadrupole field.
///
/// Returns `|exact - far| / |far|`: the `1/r^2` and `1/r^3` content
/// as a fraction of the radiating `1/r` term. Decays as `1/kr` away
/// from directivity nulls, which callers avoid by observing near the
/// response maximum.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid evaluator inputs or a
/// vanishing far-field pressure (null direction).
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::lighthill::LighthillQuadrupole;
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::{
///     lighthill_far_field_gap, observer_at_kr,
/// };
///
/// let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
/// tensor[0][0] = Complex64::new(2.0, 0.0);
/// let source = LighthillQuadrupole::new([0.0; 3], tensor).unwrap();
/// let observer = observer_at_kr([1.0, 0.0, 0.0], 1.0e4, 10.0).unwrap();
/// let gap = lighthill_far_field_gap(&source, observer, 20.0, 2.0).unwrap();
/// assert!(gap < 1e-3);
/// ```
pub fn lighthill_far_field_gap(
    source: &LighthillQuadrupole,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<f64, AeroacousticsError> {
    let exact = lighthill_pressure(source, observer, angular_frequency, speed_of_sound)?;
    let far = lighthill_far_field_pressure(source, observer, angular_frequency, speed_of_sound)?;
    if far.norm() == 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "far-field pressure magnitude",
            value: 0.0,
        });
    }
    Ok((exact - far).norm() / far.norm())
}

/// Relative `1/r` spreading error between two field magnitudes.
///
/// Returns `||p_far| r_far - |p_near| r_near| / (|p_near| r_near)`,
/// zero for exact spherical spreading. A vanishing near product
/// yields infinity (fail closed) rather than a vacuous pass.
///
/// # Examples
///
/// ```
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::spreading_error;
///
/// assert!(spreading_error(2.0, 1.0, 1.0, 2.0) < 1e-15);
/// ```
pub fn spreading_error(
    magnitude_near: f64,
    distance_near: f64,
    magnitude_far: f64,
    distance_far: f64,
) -> f64 {
    let near = magnitude_near * distance_near;
    if near == 0.0 {
        return f64::INFINITY;
    }
    ((magnitude_far * distance_far - near) / near).abs()
}

/// Relative reciprocity gap of the quadrupole kernel.
///
/// Evaluates the same tensor from `point_a` at `point_b` and vice
/// versa; the free-space Green Hessian is symmetric, so the two
/// pressures agree. Returns the difference over the larger
/// magnitude. Exact `0/0` agreement returns zero; callers keep
/// sources live (see the peak-tone floor).
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid evaluator inputs.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::lighthill_reciprocity_gap;
///
/// let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
/// tensor[1][2] = Complex64::new(1.0, -1.0);
/// tensor[2][1] = Complex64::new(1.0, -1.0);
/// let gap = lighthill_reciprocity_gap(tensor, [1.0, 0.0, 0.0], [0.0, 2.0, 1.0], 10.0, 2.0).unwrap();
/// assert!(gap < 1e-12);
/// ```
pub fn lighthill_reciprocity_gap(
    moment: [[Complex64; 3]; 3],
    point_a: [f64; 3],
    point_b: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<f64, AeroacousticsError> {
    let forward = lighthill_pressure(
        &LighthillQuadrupole::new(point_a, moment)?,
        point_b,
        angular_frequency,
        speed_of_sound,
    )?;
    let reverse = lighthill_pressure(
        &LighthillQuadrupole::new(point_b, moment)?,
        point_a,
        angular_frequency,
        speed_of_sound,
    )?;
    let scale = forward.norm().max(reverse.norm());
    if scale == 0.0 {
        return Ok(0.0);
    }
    Ok((forward - reverse).norm() / scale)
}

/// Relative thickness/loading summation gap of an FW-H evaluation.
///
/// Returns `|total - (thickness + loading)| / (|thickness| +
/// |loading|)`; exact `0/0` agreement returns zero.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::fwh::FwhComponents;
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::fwh_component_gap;
///
/// let parts = FwhComponents {
///     thickness: Complex64::new(1.0, 0.0),
///     loading: Complex64::new(0.0, 1.0),
///     total: Complex64::new(1.0, 1.0),
/// };
/// assert!(fwh_component_gap(&parts) < 1e-15);
/// ```
pub fn fwh_component_gap(components: &FwhComponents) -> f64 {
    let scale = components.thickness.norm() + components.loading.norm();
    if scale == 0.0 {
        return 0.0;
    }
    (components.total - (components.thickness + components.loading)).norm() / scale
}

/// Quadrupole directional factor along a direction.
///
/// Returns the `n_i n_j T_ij` contraction weighting the radiating
/// term; the caller supplies a unit direction.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::directivity_factor;
///
/// let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
/// tensor[0][0] = Complex64::new(2.0, 0.0);
/// let axial = directivity_factor(tensor, [1.0, 0.0, 0.0]);
/// assert!((axial - Complex64::new(2.0, 0.0)).norm() < 1e-15);
/// ```
pub fn directivity_factor(moment: [[Complex64; 3]; 3], direction: [f64; 3]) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            result += moment[i][j] * direction[i] * direction[j];
        }
    }
    result
}

/// Relative compact limit gap between ring and point quadrupoles.
///
/// Compares an `m = 0` ring of per-radian moment against the point
/// quadrupole of moment `2 pi` times larger at the ring center. The
/// gap vanishes as `(ka)^2 + (a/r)^2`, which callers keep far below
/// the comparison band.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid evaluator inputs or a
/// vanishing point pressure.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::ring_point_gap;
///
/// let mut per_rad = [[Complex64::new(0.0, 0.0); 3]; 3];
/// per_rad[2][2] = Complex64::new(1.0, 0.0);
/// let gap = ring_point_gap(per_rad, 1.0e-4, 0.0, [0.0, 0.0, 5.0], 10.0, 2.0, 1e-12).unwrap();
/// assert!(gap < 1e-6);
/// ```
pub fn ring_point_gap(
    stress_per_rad: [[Complex64; 3]; 3],
    ring_radius: f64,
    axial_position: f64,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
    tolerance: f64,
) -> Result<f64, AeroacousticsError> {
    let ring = LighthillRingSource::new(
        ring_radius,
        axial_position,
        AzimuthalMode::axisymmetric(),
        stress_per_rad,
    )?;
    let mut moment = stress_per_rad;
    for row in &mut moment {
        for entry in row.iter_mut() {
            *entry *= TAU;
        }
    }
    let center = LighthillQuadrupole::new([0.0, 0.0, axial_position], moment)?;
    let ring_pressure = lighthill_ring_pressure(
        &ring,
        observer,
        angular_frequency,
        speed_of_sound,
        tolerance,
    )?;
    let point_pressure = lighthill_pressure(&center, observer, angular_frequency, speed_of_sound)?;
    if point_pressure.norm() == 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "point pressure magnitude",
            value: 0.0,
        });
    }
    Ok((ring_pressure - point_pressure).norm() / point_pressure.norm())
}

/// Relative cross-check gap between revolved and 3-D FW-H paths.
///
/// Evaluates one [`FwhRevolvedPatch`] through
/// [`stationary_fwh_ring_pressure`] and again through
/// [`stationary_fwh_pressure`] over a hoop tiling of midpoint
/// patches (meridional midpoint by `hoop_patches` hoop midpoints,
/// each carrying the modal phase `exp(i m phi)`), then returns the
/// total-pressure difference over the revolved magnitude. The two
/// paths share only the bare patch kernel, so agreement validates
/// the revolved quadrature. Tiling error scales as `(2 pi/N)^2` on
/// the hoop and `(strip/r)^2` meridionally; callers size both.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for fewer than three hoop patches,
/// a strip centered on the axis, invalid evaluator inputs, or a
/// vanishing revolved pressure.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use sonium_fem::physics::acoustics::aeroacoustics::fwh_2d_axi::{
///     FwhRevolvedPatch, RevolvedQuadrature,
/// };
/// use sonium_fem::physics::acoustics::aeroacoustics::replay::revolved_cross_check;
/// use sonium_wave::analytical::azimuthal::AzimuthalMode;
///
/// let patch = FwhRevolvedPatch::new(
///     [0.5, -0.00015],
///     [0.5, 0.00015],
///     [1.0, 0.0],
///     AzimuthalMode::axisymmetric(),
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.0, 0.0),
/// )
/// .unwrap();
/// let gap = revolved_cross_check(
///     &patch,
///     [5.0, 0.0, 0.0],
///     10.0,
///     1.2,
///     2.0,
///     RevolvedQuadrature::new(1e-10, 4).unwrap(),
///     360,
/// )
/// .unwrap();
/// assert!(gap < 1e-3);
/// ```
pub fn revolved_cross_check(
    patch: &FwhRevolvedPatch,
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    quadrature: RevolvedQuadrature,
    hoop_patches: usize,
) -> Result<f64, AeroacousticsError> {
    if hoop_patches < 3 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "hoop patch count",
            value: hoop_patches as f64,
        });
    }
    let revolved = stationary_fwh_ring_pressure(
        std::slice::from_ref(patch),
        observer,
        angular_frequency,
        density,
        speed_of_sound,
        quadrature,
    )?;
    if revolved.total.norm() == 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "revolved pressure magnitude",
            value: 0.0,
        });
    }
    // Meridional midpoint of the strip; the tiling error from the
    // one-point meridional rule is second order in strip size over
    // observer distance, documented on the caller geometry.
    let center_r = 0.5 * (patch.node_a[0] + patch.node_b[0]);
    if !center_r.is_finite() || center_r <= 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "revolved strip radius",
            value: center_r,
        });
    }
    let center_z = 0.5 * (patch.node_a[1] + patch.node_b[1]);
    let strip = ((patch.node_b[0] - patch.node_a[0]).powi(2)
        + (patch.node_b[1] - patch.node_a[1]).powi(2))
    .sqrt();
    let order = f64::from(patch.azimuthal.order());
    let mut flat = Vec::with_capacity(hoop_patches);
    for tile in 0..hoop_patches {
        let azimuth = TAU * (tile as f64 + 0.5) / hoop_patches as f64;
        let (sin, cos) = azimuth.sin_cos();
        let phase = Complex64::from_polar(1.0, order * azimuth);
        flat.push(FwhSurfacePatch::new(
            [center_r * cos, center_r * sin, center_z],
            [
                patch.outward_normal[0] * cos,
                patch.outward_normal[0] * sin,
                patch.outward_normal[1],
            ],
            center_r * TAU / hoop_patches as f64 * strip,
            patch.pressure * phase,
            patch.normal_velocity * phase,
        )?);
    }
    let tiled =
        stationary_fwh_pressure(&flat, observer, angular_frequency, density, speed_of_sound)?;
    Ok((revolved.total - tiled.total).norm() / revolved.total.norm())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axial_tensor() -> [[Complex64; 3]; 3] {
        let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
        tensor[0][0] = Complex64::new(2.0, -0.5);
        tensor
    }

    fn tone_history(bins: usize, count: usize, amplitude: f64) -> Vec<[f64; 3]> {
        (0..count)
            .map(|n| {
                let phase = TAU * bins as f64 * n as f64 / count as f64;
                [0.0, amplitude * phase.cos(), 0.0]
            })
            .collect()
    }

    #[test]
    fn band_rule_matches_schema_samples() {
        assert!(within_band(1.0005, 1.0, 1e-12, 6.6e-4));
        assert!(!within_band(1.002, 1.0, 1e-12, 6.6e-4));
        // Absolute floor rescues small-but-live signals.
        assert!(within_band(1e-9, 0.0, 1e-8, 6.6e-4));
        assert!(!within_band(1e-7, 0.0, 1e-8, 6.6e-4));
        // Vendor near-zero skip in both forms.
        assert!(within_band(1e-15, -1e-15, 0.0, 0.0));
        assert!(within_band(1.0 + 5e-15, 1.0, 0.0, 0.0));
    }

    #[test]
    fn grid_helpers_reject_short_or_folded_time() {
        assert!(mean_step(&[]).is_err());
        assert!(mean_step(&[1.0]).is_err());
        assert!(mean_step(&[0.0, 1.0, 1.0]).is_err());
        assert!(mean_step(&[0.0, 1.0, 0.5]).is_err());
        assert!(grid_nonuniformity(&[0.0, f64::NAN, 1.0]).is_err());
    }

    #[test]
    fn grid_nonuniformity_reports_relative_spread() {
        assert!(grid_nonuniformity(&[0.0, 1.0, 2.0, 3.0]).unwrap() < 1e-15);
        let spread = grid_nonuniformity(&[0.0, 1.0, 2.2]).unwrap();
        assert!((spread - 0.1 / 1.1).abs() < 1e-12);
    }

    #[test]
    fn detrend_splits_mean_and_fluctuation() {
        let split = detrend(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((split.mean - 2.5).abs() < 1e-15);
        assert_eq!(split.fluctuation.len(), 4);
        assert!(split.fluctuation.iter().sum::<f64>().abs() < 1e-15);
        assert!(detrend(&[]).is_err());
        assert!(detrend(&[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn peak_tone_selects_strongest_bin_centered_tone() {
        let signal: Vec<f64> = (0..128)
            .map(|n| {
                let phase = TAU * f64::from(n) / 128.0;
                2.0 * (3.0 * phase).cos() + 0.5 * (9.0 * phase).cos()
            })
            .collect();
        let tone = peak_tone(&signal, 0.001).unwrap();
        assert_eq!(tone.bin, 3);
        assert!((tone.frequency_hz - 3.0 / 0.128).abs() < 1e-9);
        assert!((tone.phasor.norm() - 2.0).abs() < 1e-9);
        assert!(peak_tone(&[1.0, 2.0, 3.0], 0.001).is_err());
        assert!(peak_tone(&signal, 0.0).is_err());
    }

    #[test]
    fn quadrupole_glue_scales_stress_by_volume() {
        let history = tone_history(3, 64, 1.0);
        // v_y^2 carries bin 6 at half the squared amplitude.
        let source =
            quadrupole_from_history(&history, 1.2, 0.001, 6.0 / 0.064, [0.0; 3], 2.0).unwrap();
        assert!((source.stress_moment[1][1].norm() - 1.2).abs() < 1e-9);
        assert!(source.stress_moment[0][0].norm() < 1e-12);
        assert!(quadrupole_from_history(&history, 1.2, 0.001, 6.0 / 0.064, [0.0; 3], 0.0).is_err());
    }

    #[test]
    fn quadrupole_glue_rejects_mean_flow_dc() {
        // Uniform flow carries no tone: detrending must remove the DC
        // operating point instead of leaking it into bin 1 through the
        // periodic-Hann sidelobe (which would read ~2.7 here).
        let history = vec![[0.0, 1.5, 0.0]; 64];
        let source =
            quadrupole_from_history(&history, 1.2, 0.001, 1.0 / 0.064, [0.0; 3], 1.0).unwrap();
        let peak = source
            .stress_moment
            .iter()
            .flatten()
            .map(|entry| entry.norm())
            .fold(0.0, f64::max);
        assert!(peak < 1e-12, "DC leakage: {peak}");
    }

    #[test]
    fn observer_helper_validates_unit_direction() {
        assert_eq!(
            observer_at_kr([1.0, 0.0, 0.0], 50.0, 2.0).unwrap(),
            [25.0, 0.0, 0.0]
        );
        assert!(observer_at_kr([2.0, 0.0, 0.0], 50.0, 2.0).is_err());
        assert!(observer_at_kr([1.0, 0.0, 0.0], 0.0, 2.0).is_err());
    }

    #[test]
    fn far_field_gap_decays_with_kr() {
        let source = LighthillQuadrupole::new([0.0; 3], axial_tensor()).unwrap();
        let near = observer_at_kr([1.0, 0.0, 0.0], 10.0, 10.0).unwrap();
        let far = observer_at_kr([1.0, 0.0, 0.0], 1.0e4, 10.0).unwrap();
        let gap_near = lighthill_far_field_gap(&source, near, 20.0, 2.0).unwrap();
        let gap_far = lighthill_far_field_gap(&source, far, 20.0, 2.0).unwrap();
        // O(1/kr) decay: theory gives gap(10)/gap(1e4) = 1005 for this tensor.
        assert!((gap_near / gap_far - 1005.0).abs() < 20.0);
        assert!(gap_far < 6.6e-4);
    }

    #[test]
    fn far_field_gap_rejects_null_direction() {
        let source = LighthillQuadrupole::new([0.0; 3], axial_tensor()).unwrap();
        let observer = [0.0, 100.0, 0.0];
        assert!(lighthill_far_field_gap(&source, observer, 20.0, 2.0).is_err());
    }

    #[test]
    fn spreading_and_reciprocity_hold_for_live_tensor() {
        let source = LighthillQuadrupole::new([0.0; 3], axial_tensor()).unwrap();
        let near = lighthill_far_field_pressure(&source, [10.0, 0.0, 0.0], 20.0, 2.0).unwrap();
        let far = lighthill_far_field_pressure(&source, [30.0, 0.0, 0.0], 20.0, 2.0).unwrap();
        assert!(spreading_error(near.norm(), 10.0, far.norm(), 30.0) < 1e-12);
        assert!(spreading_error(0.0, 1.0, 1.0, 2.0).is_infinite());
        let gap =
            lighthill_reciprocity_gap(axial_tensor(), [1.0, 0.3, 0.0], [-2.0, 1.0, 0.5], 20.0, 2.0)
                .unwrap();
        assert!(gap < 1e-12);
    }

    #[test]
    fn ring_converges_to_point_as_radius_shrinks() {
        let mut per_rad = [[Complex64::new(0.0, 0.0); 3]; 3];
        per_rad[2][2] = Complex64::new(1.0, 0.25);
        let wide = ring_point_gap(per_rad, 0.05, 0.0, [0.0, 0.0, 5.0], 10.0, 2.0, 1e-12).unwrap();
        let tight =
            ring_point_gap(per_rad, 1.0e-4, 0.0, [0.0, 0.0, 5.0], 10.0, 2.0, 1e-12).unwrap();
        assert!(tight < wide / 1.0e4);
        assert!(tight < 1e-6);
    }

    #[test]
    fn revolved_tiling_matches_ring_pressure() {
        let patch = FwhRevolvedPatch::new(
            [0.5, -0.00015],
            [0.5, 0.00015],
            [1.0, 0.0],
            AzimuthalMode::axisymmetric(),
            Complex64::new(1.0, 0.5),
            Complex64::new(0.1, -0.2),
        )
        .unwrap();
        let quadrature = RevolvedQuadrature::new(1e-10, 4).unwrap();
        // The smooth periodic hoop tiling is spectrally converged at both
        // resolutions; the residual is the tiny meridional midpoint error.
        for patches in [90, 360] {
            let gap =
                revolved_cross_check(&patch, [5.0, 0.0, 0.0], 10.0, 1.2, 2.0, quadrature, patches)
                    .unwrap();
            assert!(gap < 1e-6, "N={patches}: gap={gap}");
        }
        assert!(
            revolved_cross_check(&patch, [5.0, 0.0, 0.0], 10.0, 1.2, 2.0, quadrature, 2).is_err()
        );
    }

    #[test]
    fn revolved_gap_grows_with_strip_length() {
        let quadrature = RevolvedQuadrature::new(1e-10, 4).unwrap();
        let short = FwhRevolvedPatch::new(
            [0.5, -0.00015],
            [0.5, 0.00015],
            [1.0, 0.0],
            AzimuthalMode::axisymmetric(),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
        let long = FwhRevolvedPatch::new(
            [0.5, -0.1],
            [0.5, 0.1],
            [1.0, 0.0],
            AzimuthalMode::axisymmetric(),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
        let gap_short =
            revolved_cross_check(&short, [5.0, 0.0, 0.0], 10.0, 1.2, 2.0, quadrature, 360).unwrap();
        let gap_long =
            revolved_cross_check(&long, [5.0, 0.0, 0.0], 10.0, 1.2, 2.0, quadrature, 360).unwrap();
        // Meridional midpoint error scales as strip^2; the 667x longer
        // strip must show a vastly larger (but still modest) gap.
        assert!(gap_long > gap_short * 1.0e3);
        assert!(gap_long < 0.1);
    }
}
