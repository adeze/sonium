//! Measured packet diagnostics shared by time-domain propagators.
//!
//! Structured DRP and unstructured DG solvers verify their discrete
//! dispersion against two independent oracles: semi-discrete analysis
//! ([`NumericalDispersion`](super::drp_euler::NumericalDispersion)) and
//! direct measurement of a transported packet. This module hosts the
//! measurement side: energy-weighted centroids along a propagation axis
//! and least-squares group-velocity fits over centroid tracks.
//!
//! A typical gate records `(time, centroid)` pairs while a packet crosses
//! the domain, fits [`group_velocity_from_track`], and compares the slope
//! with the analytic characteristic speed (and, for DRP grids, with the
//! semi-discrete prediction from the stencil).

// Rust guideline compliant 2026-02-21

use super::drp_euler::NumericalDispersion;

/// Centroid of a nonnegative weight field along an axis.
///
/// Returns the energy-weighted mean position `Σx·w / Σw`, or `None` when
/// the weights are empty or sum to zero (an empty track carries no
/// position information).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::acoustics::aeroacoustics::wave_diagnostics::centroid_1d;
///
/// let center = centroid_1d(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
/// assert_eq!(center, Some(1.0));
/// ```
///
/// # Panics
///
/// Panics if `coordinates` and `weights` have different lengths, which
/// signals a caller-side sampling bug.
pub fn centroid_1d(coordinates: &[f64], weights: &[f64]) -> Option<f64> {
    assert_eq!(
        coordinates.len(),
        weights.len(),
        "centroid needs one coordinate per weight"
    );
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return None;
    }
    let moment: f64 = coordinates
        .iter()
        .zip(weights.iter())
        .map(|(x, w)| x * w)
        .sum();
    Some(moment / total)
}

/// Least-squares group velocity from a centroid track.
///
/// Fits `centroid = origin + velocity·time` and returns the slope, or
/// `None` when fewer than two samples are present or all times coincide.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::acoustics::aeroacoustics::wave_diagnostics::group_velocity_from_track;
///
/// let speed = group_velocity_from_track(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
/// assert_eq!(speed, Some(2.0));
/// ```
///
/// # Panics
///
/// Panics if `times` and `centroids` have different lengths, which
/// signals a caller-side sampling bug.
pub fn group_velocity_from_track(times: &[f64], centroids: &[f64]) -> Option<f64> {
    assert_eq!(
        times.len(),
        centroids.len(),
        "group-velocity fit needs one centroid per time"
    );
    if times.len() < 2 {
        return None;
    }
    let n = times.len() as f64;
    let mean_time = times.iter().sum::<f64>() / n;
    let mean_centroid = centroids.iter().sum::<f64>() / n;
    let mut slope_numerator = 0.0;
    let mut slope_denominator = 0.0;
    for (time, centroid) in times.iter().zip(centroids.iter()) {
        slope_numerator += (time - mean_time) * (centroid - mean_centroid);
        slope_denominator += (time - mean_time).powi(2);
    }
    if slope_denominator == 0.0 {
        return None;
    }
    Some(slope_numerator / slope_denominator)
}

/// Relative group-velocity error of a semi-discrete prediction.
///
/// Compares [`NumericalDispersion::group_velocity`] against a reference
/// speed (analytic characteristic or measured track slope) as
/// `(predicted − reference) / reference`.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::acoustics::aeroacoustics::drp_euler::{
///     AcousticBranch, DrpEuler1dConfig,
/// };
/// use sonium_fem::physics::acoustics::aeroacoustics::wave_diagnostics::group_velocity_error;
///
/// let config = DrpEuler1dConfig::with_cfl(64, 1.0, 1.2, 343.0, 0.0, 0.5).unwrap();
/// let dispersion = config.dispersion(1.0e-6, AcousticBranch::Downstream);
/// assert!(group_velocity_error(&dispersion, 343.0).abs() < 1e-9);
/// ```
pub fn group_velocity_error(dispersion: &NumericalDispersion, reference: f64) -> f64 {
    (dispersion.group_velocity - reference) / reference
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_recovers_gaussian_center() {
        let center = 2.5;
        let sigma = 0.4;
        let coordinates: Vec<f64> = (0..200).map(|i| i as f64 * 0.025).collect();
        let weights: Vec<f64> = coordinates
            .iter()
            .map(|x| (-0.5 * ((x - center) / sigma).powi(2)).exp())
            .collect();
        let measured = centroid_1d(&coordinates, &weights).unwrap();
        assert!((measured - center).abs() < 1e-9, "center={measured}");
    }

    #[test]
    fn centroid_reports_empty_tracks_as_missing() {
        assert_eq!(centroid_1d(&[], &[]), None);
        assert_eq!(centroid_1d(&[0.0, 1.0], &[0.0, 0.0]), None);
    }

    #[test]
    fn group_velocity_fit_recovers_noisy_track_slope() {
        let times: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
        let centroids: Vec<f64> = times
            .iter()
            .enumerate()
            .map(|(i, t)| 1.7 + 343.0 * t + 1e-6 * (i as f64).sin())
            .collect();
        let speed = group_velocity_from_track(&times, &centroids).unwrap();
        assert!((speed - 343.0).abs() < 1e-6, "speed={speed}");
        assert_eq!(group_velocity_from_track(&[0.0], &[1.0]), None);
        assert_eq!(group_velocity_from_track(&[1.0, 1.0], &[0.0, 1.0]), None);
    }
}
