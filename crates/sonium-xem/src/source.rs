//! Sound source definitions for room acoustics simulations

use crate::conventions::FrequencyHz;
use crate::types::Point3D;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_compute::HostMatrix;

/// Directivity pattern sampled on a grid.
///
/// `magnitude` is a row-major `HostMatrix<f64>` with shape
/// `[n_vertical, n_horizontal]` matching the order of [`Self::vertical_angles`]
/// and [`Self::horizontal_angles`].
#[derive(Debug, Clone)]
pub struct DirectivityGrid {
    /// Horizontal angles (azimuth) in degrees [0, 360) with step 10°
    horizontal_angles: Vec<f64>,
    /// Vertical angles (elevation) in degrees [0, 180] with step 10°
    vertical_angles: Vec<f64>,
    /// Magnitude at each (horizontal, vertical) angle pair.
    /// Shape: `[n_vertical, n_horizontal]`, row-major.
    magnitude: HostMatrix<f64>,
}

/// Invalid directivity-grid axes or matrix dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectivityGridError(pub String);

impl std::fmt::Display for DirectivityGridError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DirectivityGridError {}

impl DirectivityGrid {
    /// Construct a validated directivity grid.
    pub fn try_new(
        horizontal_angles: Vec<f64>,
        vertical_angles: Vec<f64>,
        magnitude: HostMatrix<f64>,
    ) -> Result<Self, DirectivityGridError> {
        validate_angle_axis(&horizontal_angles, 0.0, 360.0, "horizontal")?;
        validate_angle_axis(&vertical_angles, 0.0, 180.0, "vertical")?;
        if magnitude.nrows() != vertical_angles.len()
            || magnitude.ncols() != horizontal_angles.len()
        {
            return Err(DirectivityGridError(format!(
                "magnitude shape {}x{} does not match vertical/horizontal axes {}x{}",
                magnitude.nrows(),
                magnitude.ncols(),
                vertical_angles.len(),
                horizontal_angles.len()
            )));
        }
        if magnitude.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(DirectivityGridError(
                "directivity magnitudes must be finite".into(),
            ));
        }
        Ok(Self {
            horizontal_angles,
            vertical_angles,
            magnitude,
        })
    }

    /// Horizontal sample angles in ascending degrees within `[0, 360)`.
    pub fn horizontal_angles(&self) -> &[f64] {
        &self.horizontal_angles
    }

    /// Vertical sample angles in ascending degrees within `[0, 180]`.
    pub fn vertical_angles(&self) -> &[f64] {
        &self.vertical_angles
    }

    /// Row-major magnitude matrix with vertical rows and horizontal columns.
    pub fn magnitude(&self) -> &HostMatrix<f64> {
        &self.magnitude
    }

    /// Create omnidirectional pattern (uniform radiation)
    pub fn omnidirectional() -> Self {
        let horizontal_angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical_angles: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();

        let magnitude = HostMatrix::from_elem(vertical_angles.len(), horizontal_angles.len(), 1.0);

        Self::try_new(horizontal_angles, vertical_angles, magnitude)
            .expect("built-in omnidirectional grid is valid")
    }

    /// Create a cardioid directivity pattern
    pub fn cardioid() -> Self {
        let horizontal_angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical_angles: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();

        let mut magnitude =
            HostMatrix::<f64>::zeros(vertical_angles.len(), horizontal_angles.len());

        for (v_idx, &v_angle) in vertical_angles.iter().enumerate() {
            for (h_idx, &h_angle) in horizontal_angles.iter().enumerate() {
                // Cardioid pattern: 0.5 * (1 + cos(theta))
                let theta_rad = v_angle.to_radians();
                let phi_rad = h_angle.to_radians();
                // Forward direction is along +Y (phi=90, theta=90)
                let forward_dot = theta_rad.sin() * phi_rad.sin();
                magnitude[[v_idx, h_idx]] = 0.5 * (1.0 + forward_dot).max(0.0);
            }
        }

        Self::try_new(horizontal_angles, vertical_angles, magnitude)
            .expect("built-in cardioid grid is valid")
    }

    /// Left-right mirror image of the pattern in the source's local frame:
    /// the mirrored grid evaluates at `(theta, phi)` what this grid evaluates
    /// at `(theta, pi - phi)`. The mirrored horizontal axis is an exact
    /// permutation of the original samples (no resampling error).
    pub fn mirrored(&self) -> Self {
        let mut columns: Vec<(f64, usize)> = self
            .horizontal_angles
            .iter()
            .enumerate()
            .map(|(idx, &phi)| ((180.0 - phi).rem_euclid(360.0), idx))
            .collect();
        columns.sort_by(|a, b| a.0.total_cmp(&b.0));
        let horizontal_angles: Vec<f64> = columns.iter().map(|column| column.0).collect();
        let magnitude = HostMatrix::from_fn(
            self.vertical_angles.len(),
            self.horizontal_angles.len(),
            |row, column| self.magnitude[[row, columns[column].1]],
        );
        Self::try_new(horizontal_angles, self.vertical_angles.clone(), magnitude)
            .expect("mirrored grid preserves axis validity")
    }

    /// Interpolate directivity at an arbitrary direction.
    ///
    /// At the north and south poles azimuth is physically undefined, so the
    /// value is the azimuthal mean of the nearest sampled elevation row.
    pub fn interpolate(&self, theta: f64, phi: f64) -> f64 {
        // Convert spherical to degrees
        let theta_deg = theta.to_degrees().clamp(0.0, 180.0);
        let phi_deg = phi.to_degrees().rem_euclid(360.0);
        if theta_deg <= 1e-12 {
            return self.pole_mean(0);
        }
        if 180.0 - theta_deg <= 1e-12 {
            return self.pole_mean(self.vertical_angles.len() - 1);
        }
        let (h_idx, h_next, h_frac) = circular_bracket(&self.horizontal_angles, phi_deg);
        let (v_idx, v_next, v_frac) = linear_bracket(&self.vertical_angles, theta_deg);

        let m00 = self.magnitude[[v_idx, h_idx]];
        let m01 = self.magnitude[[v_idx, h_next]];
        let m10 = self.magnitude[[v_next, h_idx]];
        let m11 = self.magnitude[[v_next, h_next]];

        let m0 = m00 * (1.0 - h_frac) + m01 * h_frac;
        let m1 = m10 * (1.0 - h_frac) + m11 * h_frac;

        m0 * (1.0 - v_frac) + m1 * v_frac
    }

    /// Strict interpolation entry point that rejects invalid directions.
    ///
    /// The legacy [`Self::interpolate`] method clamps polar angles for
    /// compatibility. Exchange and solver boundaries should use this method
    /// so malformed coordinates cannot silently become a different direction.
    pub fn try_interpolate(&self, theta: f64, phi: f64) -> Result<f64, DirectivityGridError> {
        validate_interpolation_angles(theta, phi, None)?;
        Ok(self.interpolate(theta, phi))
    }

    fn pole_mean(&self, vertical_index: usize) -> f64 {
        self.magnitude.row(vertical_index).iter().sum::<f64>() / self.horizontal_angles.len() as f64
    }
}

/// Complex, frequency-dependent directivity pattern sampled on a grid.
///
/// This is the speaker-to-room handoff representation: one complex
/// (magnitude *and* phase) far-field pattern per frequency sample. A speaker
/// BEM solve or a measurement can fill it directly. Directivity phase,
/// acoustic-center motion with frequency, and complex multi-driver
/// interference are all captured; the magnitude-only [`DirectivityGrid`]
/// remains as the lightweight L0/L1 fidelity model.
///
/// Frequency samples must be dense enough that the phase rotates by much
/// less than π between neighbours, otherwise the linear frequency
/// interpolation can carve artificial cancellations into the pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexDirectivityGrid {
    /// Horizontal angles (azimuth) in degrees [0, 360), ascending.
    horizontal_angles: Vec<f64>,
    /// Vertical angles (elevation) in degrees [0, 180], ascending.
    vertical_angles: Vec<f64>,
    /// Frequency samples in Hz, positive and strictly increasing.
    frequencies: Vec<f64>,
    /// One complex pattern per frequency sample, each row-major
    /// `[n_vertical, n_horizontal]` matching the angle axes.
    patterns: Vec<HostMatrix<Complex64>>,
}

impl ComplexDirectivityGrid {
    /// Construct a validated complex directivity grid.
    pub fn try_new(
        horizontal_angles: Vec<f64>,
        vertical_angles: Vec<f64>,
        frequencies: Vec<f64>,
        patterns: Vec<HostMatrix<Complex64>>,
    ) -> Result<Self, DirectivityGridError> {
        validate_angle_axis(&horizontal_angles, 0.0, 360.0, "horizontal")?;
        validate_angle_axis(&vertical_angles, 0.0, 180.0, "vertical")?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| FrequencyHz::try_new(*frequency).is_err())
            || frequencies.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(DirectivityGridError(
                "frequency axis must be positive and strictly increasing".into(),
            ));
        }
        if patterns.len() != frequencies.len() {
            return Err(DirectivityGridError(format!(
                "{} patterns for {} frequency samples",
                patterns.len(),
                frequencies.len()
            )));
        }
        for pattern in &patterns {
            if pattern.nrows() != vertical_angles.len()
                || pattern.ncols() != horizontal_angles.len()
            {
                return Err(DirectivityGridError(format!(
                    "pattern shape {}x{} does not match vertical/horizontal axes {}x{}",
                    pattern.nrows(),
                    pattern.ncols(),
                    vertical_angles.len(),
                    horizontal_angles.len()
                )));
            }
            if pattern.as_slice().iter().any(|value| !value.is_finite()) {
                return Err(DirectivityGridError(
                    "complex directivity samples must be finite".into(),
                ));
            }
        }
        Ok(Self {
            horizontal_angles,
            vertical_angles,
            frequencies,
            patterns,
        })
    }

    /// Horizontal sample angles in ascending degrees within `[0, 360)`.
    pub fn horizontal_angles(&self) -> &[f64] {
        &self.horizontal_angles
    }

    /// Vertical sample angles in ascending degrees within `[0, 180]`.
    pub fn vertical_angles(&self) -> &[f64] {
        &self.vertical_angles
    }

    /// Frequency samples in Hz.
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Frequency samples as validated exchange quantities.
    ///
    /// The scalar [`Self::frequencies`] accessor remains available for
    /// compatibility. This typed view is intended for solver/application
    /// boundaries that must carry the positive-Hz contract explicitly.
    pub fn frequency_quantities(&self) -> Vec<FrequencyHz> {
        self.frequencies
            .iter()
            .copied()
            .map(FrequencyHz::new)
            .collect()
    }

    /// Complex patterns, one per frequency sample.
    pub fn patterns(&self) -> &[HostMatrix<Complex64>] {
        &self.patterns
    }

    /// Left-right mirror image of every pattern (see
    /// [`DirectivityGrid::mirrored`]); phase is preserved per sample, which is
    /// what a geometric mirror of a complex pattern requires.
    pub fn mirrored(&self) -> Self {
        let mut columns: Vec<(f64, usize)> = self
            .horizontal_angles
            .iter()
            .enumerate()
            .map(|(idx, &phi)| ((180.0 - phi).rem_euclid(360.0), idx))
            .collect();
        columns.sort_by(|a, b| a.0.total_cmp(&b.0));
        let horizontal_angles: Vec<f64> = columns.iter().map(|column| column.0).collect();
        let patterns = self
            .patterns
            .iter()
            .map(|pattern| {
                HostMatrix::from_fn(
                    self.vertical_angles.len(),
                    self.horizontal_angles.len(),
                    |row, column| pattern[[row, columns[column].1]],
                )
            })
            .collect();
        Self::try_new(
            horizontal_angles,
            self.vertical_angles.clone(),
            self.frequencies.clone(),
            patterns,
        )
        .expect("mirrored grid preserves axis validity")
    }

    /// Interpolate the complex pattern at an arbitrary direction and frequency.
    ///
    /// Bilinear in angle (azimuth wraps, poles use the azimuthal mean) and
    /// piecewise-linear in frequency; outside the sampled frequency range the
    /// nearest edge pattern is held constant. With a single frequency sample
    /// the pattern is frequency-independent.
    pub fn interpolate(&self, theta: f64, phi: f64, frequency: f64) -> Complex64 {
        let theta_deg = theta.to_degrees().clamp(0.0, 180.0);
        let phi_deg = phi.to_degrees().rem_euclid(360.0);

        let pattern_at = |pattern: &HostMatrix<Complex64>| -> Complex64 {
            if theta_deg <= 1e-12 {
                let row = pattern.row(0);
                row.iter().copied().sum::<Complex64>() / self.horizontal_angles.len() as f64
            } else if 180.0 - theta_deg <= 1e-12 {
                let row = pattern.row(self.vertical_angles.len() - 1);
                row.iter().copied().sum::<Complex64>() / self.horizontal_angles.len() as f64
            } else {
                let (h_idx, h_next, h_frac) = circular_bracket(&self.horizontal_angles, phi_deg);
                let (v_idx, v_next, v_frac) = linear_bracket(&self.vertical_angles, theta_deg);
                let m00 = pattern[[v_idx, h_idx]];
                let m01 = pattern[[v_idx, h_next]];
                let m10 = pattern[[v_next, h_idx]];
                let m11 = pattern[[v_next, h_next]];
                let m0 = m00 * (1.0 - h_frac) + m01 * h_frac;
                let m1 = m10 * (1.0 - h_frac) + m11 * h_frac;
                m0 * (1.0 - v_frac) + m1 * v_frac
            }
        };

        if self.patterns.len() == 1 {
            return pattern_at(&self.patterns[0]);
        }
        let (f_idx, f_next, f_frac) = linear_bracket(&self.frequencies, frequency);
        if f_idx == f_next {
            return pattern_at(&self.patterns[f_idx]);
        }
        let low = pattern_at(&self.patterns[f_idx]);
        let high = pattern_at(&self.patterns[f_next]);
        low * (1.0 - f_frac) + high * f_frac
    }

    /// Strict interpolation entry point that rejects invalid directions and
    /// frequencies instead of clamping a malformed exchange payload.
    pub fn try_interpolate(
        &self,
        theta: f64,
        phi: f64,
        frequency: f64,
    ) -> Result<Complex64, DirectivityGridError> {
        validate_interpolation_angles(theta, phi, Some(frequency))?;
        Ok(self.interpolate(theta, phi, frequency))
    }

    /// Typed-frequency interpolation entry point for exchange consumers.
    pub fn try_interpolate_at(
        &self,
        theta: f64,
        phi: f64,
        frequency: FrequencyHz,
    ) -> Result<Complex64, DirectivityGridError> {
        self.try_interpolate(theta, phi, frequency.get())
    }
}

/// Serde mirror of [`ComplexDirectivityGrid`]: patterns travel as nested
/// JSON arrays, one `[n_vertical][n_horizontal]` matrix per frequency sample,
/// with each complex value in num-complex's `[re, im]` pair form.
#[derive(Serialize, Deserialize)]
struct ComplexDirectivityGridSerde {
    horizontal_angles: Vec<f64>,
    vertical_angles: Vec<f64>,
    frequencies: Vec<f64>,
    patterns: Vec<Vec<Vec<Complex64>>>,
}

impl Serialize for ComplexDirectivityGrid {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let patterns = self
            .patterns
            .iter()
            .map(|pattern| {
                (0..pattern.nrows())
                    .map(|row| pattern.row(row).to_vec())
                    .collect::<Vec<_>>()
            })
            .collect();
        ComplexDirectivityGridSerde {
            horizontal_angles: self.horizontal_angles.clone(),
            vertical_angles: self.vertical_angles.clone(),
            frequencies: self.frequencies.clone(),
            patterns,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ComplexDirectivityGrid {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = ComplexDirectivityGridSerde::deserialize(deserializer)?;
        let patterns = raw
            .patterns
            .into_iter()
            .map(|rows| {
                let nrows = rows.len();
                let ncols = rows.first().map_or(0, Vec::len);
                if rows.iter().any(|row| row.len() != ncols) {
                    return Err(serde::de::Error::custom(
                        "complex directivity pattern rows must all have equal length",
                    ));
                }
                Ok(HostMatrix::from_row_major_vec(
                    nrows,
                    ncols,
                    rows.into_iter().flatten().collect(),
                ))
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        ComplexDirectivityGrid::try_new(
            raw.horizontal_angles,
            raw.vertical_angles,
            raw.frequencies,
            patterns,
        )
        .map_err(serde::de::Error::custom)
    }
}

fn validate_angle_axis(
    angles: &[f64],
    minimum: f64,
    maximum: f64,
    name: &str,
) -> Result<(), DirectivityGridError> {
    if angles.len() < 2 {
        return Err(DirectivityGridError(format!(
            "{name} angle axis requires at least two samples"
        )));
    }
    let upper_is_exclusive = name == "horizontal";
    if angles.iter().any(|angle| {
        !angle.is_finite()
            || *angle < minimum
            || *angle > maximum
            || (upper_is_exclusive && *angle == maximum)
    }) || angles.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(DirectivityGridError(format!(
            "{name} angles must be finite, strictly increasing, and inside the supported range"
        )));
    }
    Ok(())
}

fn validate_interpolation_angles(
    theta: f64,
    phi: f64,
    frequency: Option<f64>,
) -> Result<(), DirectivityGridError> {
    if !theta.is_finite() || !(0.0..=std::f64::consts::PI).contains(&theta) {
        return Err(DirectivityGridError(
            "polar angle must be finite and inside [0, pi]".into(),
        ));
    }
    if !phi.is_finite() {
        return Err(DirectivityGridError("azimuth must be finite".into()));
    }
    if let Some(frequency) = frequency
        && (!frequency.is_finite() || frequency <= 0.0)
    {
        return Err(DirectivityGridError(
            "interpolation frequency must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn circular_bracket(angles: &[f64], target: f64) -> (usize, usize, f64) {
    let insertion = angles.partition_point(|angle| *angle <= target);
    if insertion == 0 {
        let lower = angles.len() - 1;
        let lower_angle = angles[lower] - 360.0;
        let span = angles[0] - lower_angle;
        return (lower, 0, (target - lower_angle) / span);
    }
    if insertion == angles.len() {
        let lower = angles.len() - 1;
        let span = angles[0] + 360.0 - angles[lower];
        return (lower, 0, (target - angles[lower]) / span);
    }
    let lower = insertion - 1;
    let span = angles[insertion] - angles[lower];
    (lower, insertion, (target - angles[lower]) / span)
}

fn linear_bracket(angles: &[f64], target: f64) -> (usize, usize, f64) {
    if target <= angles[0] {
        return (0, 0, 0.0);
    }
    let last = angles.len() - 1;
    if target >= angles[last] {
        return (last, last, 0.0);
    }
    let upper = angles.partition_point(|angle| *angle <= target);
    let lower = upper - 1;
    let fraction = (target - angles[lower]) / (angles[upper] - angles[lower]);
    (lower, upper, fraction)
}

/// Directivity model
#[derive(Debug, Clone)]
pub enum Directivity {
    /// Sampled grid (static)
    Grid(DirectivityGrid),
    /// Classical frequency-dependent directional source
    Classical {
        /// Horizontal beamwidth (degrees) at high frequency
        h_angle: f64,
        /// Vertical beamwidth (degrees) at high frequency
        v_angle: f64,
    },
    /// Complex, frequency-dependent sampled pattern (speaker-to-room handoff
    /// fidelity; carries directivity phase and acoustic-center motion).
    ComplexGrid(ComplexDirectivityGrid),
}

impl Directivity {
    /// Amplitude scaling for a unit direction vector in the source's *local* frame
    /// (where `+Y` is the forward / acoustic axis, `+Z` is up, `+X` is to-the-right).
    /// `local_dir` must be a unit vector — the caller is responsible for the world→local
    /// rotation.
    pub fn amplitude_local(&self, local_dir: Point3D, frequency: f64) -> f64 {
        // Convert local Cartesian unit vector to local spherical angles:
        //   theta = polar angle from +Z (in [0, π])
        //   phi   = azimuth from +X (in [-π, π])
        // Forward (+Y) corresponds to theta=π/2, phi=π/2.
        let theta = local_dir.z.clamp(-1.0, 1.0).acos();
        let phi = local_dir.y.atan2(local_dir.x);
        self.amplitude(theta, phi, frequency)
    }

    /// Checked amplitude scaling for a finite, unit direction in the
    /// source-local Cartesian frame.
    pub fn try_amplitude_local(
        &self,
        local_dir: Point3D,
        frequency: f64,
    ) -> Result<f64, DirectivityGridError> {
        let components = [local_dir.x, local_dir.y, local_dir.z];
        if components.iter().any(|value| !value.is_finite()) {
            return Err(DirectivityGridError(
                "local directivity direction must be finite".into(),
            ));
        }
        let length = local_dir.length();
        if !length.is_finite() || (length - 1.0).abs() > 1e-8 {
            return Err(DirectivityGridError(
                "local directivity direction must be a unit vector".into(),
            ));
        }
        let theta = local_dir.z.clamp(-1.0, 1.0).acos();
        let phi = local_dir.y.atan2(local_dir.x);
        self.try_amplitude(theta, phi, frequency)
    }

    /// Complex amplitude scaling in the source's local frame. Magnitude-only
    /// models return their (real) amplitude with zero phase; the complex grid
    /// returns the full complex sample including directivity phase.
    pub fn complex_amplitude_local(&self, local_dir: Point3D, frequency: f64) -> Complex64 {
        let theta = local_dir.z.clamp(-1.0, 1.0).acos();
        let phi = local_dir.y.atan2(local_dir.x);
        match self {
            Directivity::ComplexGrid(grid) => grid.interpolate(theta, phi, frequency),
            _ => Complex64::new(self.amplitude(theta, phi, frequency), 0.0),
        }
    }

    /// Checked amplitude scaling in the source-local spherical frame.
    ///
    /// Unlike [`Self::amplitude`], this rejects non-finite or out-of-domain
    /// angles and frequencies before interpolation. Classical beam widths are
    /// also checked as finite full widths in `(0, 360]` degrees, preventing a
    /// malformed configuration from dividing by zero or silently creating a
    /// non-physical beam. The compatibility method remains for trusted legacy
    /// in-memory configurations.
    pub fn try_amplitude(
        &self,
        theta: f64,
        phi: f64,
        frequency: f64,
    ) -> Result<f64, DirectivityGridError> {
        validate_interpolation_angles(theta, phi, Some(frequency))?;
        match self {
            Self::Grid(grid) => grid.try_interpolate(theta, phi),
            Self::ComplexGrid(grid) => Ok(grid.try_interpolate(theta, phi, frequency)?.norm()),
            Self::Classical { h_angle, v_angle } => {
                for (name, width) in [
                    ("horizontal beam width", h_angle),
                    ("vertical beam width", v_angle),
                ] {
                    if !width.is_finite() || *width <= 0.0 || *width > 360.0 {
                        return Err(DirectivityGridError(format!(
                            "{name} must be finite and in (0, 360] degrees"
                        )));
                    }
                }
                Ok(self.amplitude(theta, phi, frequency))
            }
        }
    }

    /// True if this directivity is rotationally invariant (a true monopole,
    /// for which `orientation` has no effect on the field). The Grid tolerance
    /// is loose enough to survive JSON round-trips and float-renormalisation
    /// while still rejecting any meaningfully non-isotropic pattern.
    pub fn is_isotropic(&self) -> bool {
        match self {
            Directivity::Grid(grid) => grid
                .magnitude
                .as_slice()
                .iter()
                .all(|&m| (m - 1.0).abs() < 1e-6),
            // The Classical pattern's "omni" idiom is full-circle widths in both
            // axes; with h_angle = v_angle = 360 the result is forward-biased
            // (-6 dB at the rear). It is *not* isotropic.
            Directivity::Classical { .. } => false,
            Directivity::ComplexGrid(grid) => grid.patterns().iter().all(|pattern| {
                pattern
                    .as_slice()
                    .iter()
                    .all(|&m| (m.re - 1.0).abs() < 1e-6 && m.im.abs() < 1e-6)
            }),
        }
    }

    /// Mirror image of the pattern under a left-right flip of the source's
    /// local frame (local `+X` → `−X`; spherical `(theta, phi)` →
    /// `(theta, pi - phi)`). Combined with a mirrored [`Source::orientation`],
    /// this is the pattern radiated by the image source across either a world
    /// `x = const` or `y = const` symmetry plane: reflecting the world frame
    /// across such a plane reduces to exactly this flip in the source's local
    /// coordinates, for both planes.
    ///
    /// The Classical pattern depends only on `|phi − 90°|` and `|theta − 90°|`,
    /// so it is already mirror-symmetric and is returned unchanged.
    pub fn mirrored(&self) -> Directivity {
        match self {
            Directivity::Grid(grid) => Directivity::Grid(grid.mirrored()),
            Directivity::Classical { .. } => self.clone(),
            Directivity::ComplexGrid(grid) => Directivity::ComplexGrid(grid.mirrored()),
        }
    }

    /// True if the pattern is invariant under [`Self::mirrored`]. A source
    /// sitting *on* a symmetry plane only contributes even (Neumann) modes, so
    /// the symmetry reconstruction is only valid for on-plane sources whose
    /// pattern passes this check.
    pub fn is_mirror_symmetric(&self) -> bool {
        match self {
            Directivity::Classical { .. } => true,
            Directivity::Grid(grid) => {
                for (v, &theta_deg) in grid.vertical_angles().iter().enumerate() {
                    for (h, &phi_deg) in grid.horizontal_angles().iter().enumerate() {
                        let flipped_phi = (180.0 - phi_deg).rem_euclid(360.0).to_radians();
                        let mirrored = grid.interpolate(theta_deg.to_radians(), flipped_phi);
                        if (grid.magnitude()[[v, h]] - mirrored).abs() > 1e-6 {
                            return false;
                        }
                    }
                }
                true
            }
            Directivity::ComplexGrid(grid) => {
                let mirrored = grid.mirrored();
                grid.patterns()
                    .iter()
                    .zip(mirrored.patterns())
                    .all(|(original, flipped)| {
                        original
                            .as_slice()
                            .iter()
                            .zip(flipped.as_slice())
                            .all(|(a, b)| (a - b).norm() < 1e-6)
                    })
            }
        }
    }

    /// Amplitude scaling factor in the source's local spherical frame.
    /// `theta` is polar angle from `+Z`; `phi` is azimuth from `+X`. Forward
    /// (`+Y` in local frame) corresponds to `theta = π/2`, `phi = π/2`.
    ///
    /// For the complex grid this returns the *magnitude* of the complex
    /// pattern; phase-aware consumers must use
    /// [`Self::complex_amplitude_local`] instead.
    pub fn amplitude(&self, theta: f64, phi: f64, frequency: f64) -> f64 {
        match self {
            Directivity::Grid(grid) => grid.interpolate(theta, phi),
            Directivity::ComplexGrid(grid) => grid.interpolate(theta, phi, frequency).norm(),
            Directivity::Classical { h_angle, v_angle } => {
                // Classical pattern: Gaussian-like beam narrowing with frequency.
                // Transition from omni at 80 Hz to {h_angle, v_angle} at 800 Hz.
                let min_f = 80.0;
                let max_f = 800.0;

                let t = if frequency <= min_f {
                    0.0
                } else if frequency >= max_f {
                    1.0
                } else {
                    (frequency.ln() - min_f.ln()) / (max_f.ln() - min_f.ln())
                };

                let h_width = 360.0 * (1.0 - t) + h_angle * t;
                let v_width = 360.0 * (1.0 - t) + v_angle * t;

                // Forward = (+Y in local frame) = theta=90°, phi=90°. Width is
                // measured as the FULL beam width, so half-width = width/2 is
                // the -6 dB point (r=1 in the Gaussian below).
                let mut phi_deg = phi.to_degrees();
                while phi_deg < 0.0 {
                    phi_deg += 360.0;
                }
                let delta_phi = (phi_deg - 90.0).abs().min((360.0 - (phi_deg - 90.0)).abs());

                let theta_deg = theta.to_degrees();
                let delta_theta = (theta_deg - 90.0).abs();

                let x_h = delta_phi / (h_width / 2.0);
                let x_v = delta_theta / (v_width / 2.0);
                let r_sq = x_h * x_h + x_v * x_v;
                0.5_f64.powf(r_sq)
            }
        }
    }
}

/// Crossover filter for frequency-limited sources
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CrossoverFilter {
    /// Full range (no filter)
    #[default]
    FullRange,
    /// Low-pass filter (Butterworth)
    Lowpass {
        /// Cutoff frequency (Hz)
        cutoff_freq: f64,
        /// Filter order (e.g., 2, 4)
        order: u32,
    },
    /// High-pass filter (Butterworth)
    Highpass {
        /// Cutoff frequency (Hz)
        cutoff_freq: f64,
        /// Filter order (e.g., 2, 4)
        order: u32,
    },
    /// Band-pass filter (combined Lowpass and Highpass)
    Bandpass {
        /// Low cutoff frequency (Hz)
        low_cutoff: f64,
        /// High cutoff frequency (Hz)
        high_cutoff: f64,
        /// Filter order
        order: u32,
    },
}

impl CrossoverFilter {
    /// Checked amplitude multiplier at a given frequency.
    ///
    /// Unlike [`Self::amplitude_at_frequency`], this rejects non-finite
    /// frequencies and non-positive or non-finite cutoffs instead of
    /// silently returning 0 (`cutoff = 0` makes every ratio infinite).
    pub fn try_amplitude_at_frequency(&self, frequency: f64) -> Result<f64, DirectivityGridError> {
        if !frequency.is_finite() || frequency < 0.0 {
            return Err(DirectivityGridError(
                "crossover frequency must be finite and non-negative".into(),
            ));
        }
        let check_cutoff = |name: &str, cutoff: f64| -> Result<(), DirectivityGridError> {
            if cutoff.is_finite() && cutoff > 0.0 {
                Ok(())
            } else {
                Err(DirectivityGridError(format!(
                    "crossover {name} must be finite and positive"
                )))
            }
        };
        match self {
            CrossoverFilter::FullRange => Ok(1.0),
            CrossoverFilter::Lowpass { cutoff_freq, .. } => {
                check_cutoff("cutoff_freq", *cutoff_freq)?;
                Ok(self.amplitude_at_frequency(frequency))
            }
            CrossoverFilter::Highpass { cutoff_freq, .. } => {
                check_cutoff("cutoff_freq", *cutoff_freq)?;
                Ok(self.amplitude_at_frequency(frequency))
            }
            CrossoverFilter::Bandpass {
                low_cutoff,
                high_cutoff,
                ..
            } => {
                check_cutoff("low_cutoff", *low_cutoff)?;
                check_cutoff("high_cutoff", *high_cutoff)?;
                if low_cutoff > high_cutoff {
                    return Err(DirectivityGridError(
                        "crossover low_cutoff must not exceed high_cutoff".into(),
                    ));
                }
                Ok(self.amplitude_at_frequency(frequency))
            }
        }
    }

    /// Get amplitude multiplier at a given frequency
    pub fn amplitude_at_frequency(&self, frequency: f64) -> f64 {
        match self {
            CrossoverFilter::FullRange => 1.0,
            CrossoverFilter::Lowpass { cutoff_freq, order } => {
                let ratio = frequency / cutoff_freq;
                1.0 / (1.0 + ratio.powi(*order as i32 * 2)).sqrt()
            }
            CrossoverFilter::Highpass { cutoff_freq, order } => {
                let ratio = cutoff_freq / frequency;
                1.0 / (1.0 + ratio.powi(*order as i32 * 2)).sqrt()
            }
            CrossoverFilter::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => {
                let high_ratio = low_cutoff / frequency;
                let low_ratio = frequency / high_cutoff;
                let hp_response = 1.0 / (1.0 + high_ratio.powi(*order as i32 * 2)).sqrt();
                let lp_response = 1.0 / (1.0 + low_ratio.powi(*order as i32 * 2)).sqrt();
                hp_response * lp_response
            }
        }
    }

    /// Complex multiplier at a given frequency: the all-pole Butterworth
    /// transfer function, so the phase/group delay is the physical one, not
    /// zero. Its magnitude equals [`Self::amplitude_at_frequency`].
    ///
    /// An n-th order low-pass is `H(ŝ) = Π_k (-p_k)/(ŝ - p_k)` with the
    /// left-half-plane unit-circle poles `p_k`; the high-pass is
    /// `H(ŝ) = Π_k ŝ/(ŝ - p_k)`. Band-pass is the cascade of the two.
    pub fn complex_amplitude_at_frequency(&self, frequency: f64) -> Complex64 {
        // Normalized LHP Butterworth poles p_k = exp(iπ(2k + n + 1) / 2n).
        let poles = |order: u32| -> Vec<Complex64> {
            let n = order.max(1) as usize;
            (0..n)
                .map(|k| {
                    let angle = std::f64::consts::PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
                    Complex64::new(angle.cos(), angle.sin())
                })
                .collect()
        };
        let lowpass = |frequency: f64, cutoff: f64, order: u32| -> Complex64 {
            let s = Complex64::new(0.0, frequency / cutoff);
            poles(order)
                .iter()
                .map(|pole| (-pole) / (s - pole))
                .product()
        };
        let highpass = |frequency: f64, cutoff: f64, order: u32| -> Complex64 {
            let s = Complex64::new(0.0, frequency / cutoff);
            poles(order).iter().map(|pole| s / (s - pole)).product()
        };
        match self {
            CrossoverFilter::FullRange => Complex64::new(1.0, 0.0),
            CrossoverFilter::Lowpass { cutoff_freq, order } => {
                lowpass(frequency, *cutoff_freq, *order)
            }
            CrossoverFilter::Highpass { cutoff_freq, order } => {
                highpass(frequency, *cutoff_freq, *order)
            }
            CrossoverFilter::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => {
                highpass(frequency, *low_cutoff, *order) * lowpass(frequency, *high_cutoff, *order)
            }
        }
    }
}

/// Sound source with position and directivity.
///
/// `orientation` is the loudspeaker's acoustic axis (forward direction) in world
/// coordinates. Directivity patterns are evaluated in the source's local frame,
/// where this vector maps to local `+Y`. Default is `(0, 1, 0)` so existing
/// configs that did not set an orientation remain unchanged.
#[derive(Debug, Clone)]
pub struct Source {
    /// Source position
    pub position: Point3D,
    /// Forward axis (acoustic axis). Need not be unit length; normalized internally.
    pub orientation: Point3D,
    /// Directivity pattern
    pub directivity: Directivity,
    /// Source amplitude (strength)
    pub amplitude: f64,
    /// Crossover filter
    pub crossover: CrossoverFilter,
    /// Source name (e.g., "Left Main", "Subwoofer")
    pub name: String,
}

fn default_orientation() -> Point3D {
    Point3D::new(0.0, 1.0, 0.0)
}

impl Source {
    /// Create a new source
    pub fn new(position: Point3D, directivity: Directivity, amplitude: f64) -> Self {
        Self {
            position,
            orientation: default_orientation(),
            directivity,
            amplitude,
            crossover: CrossoverFilter::FullRange,
            name: String::from("Source"),
        }
    }

    /// Create a simple omnidirectional source at position
    pub fn omnidirectional(position: Point3D, amplitude: f64) -> Self {
        Self::new(
            position,
            Directivity::Grid(DirectivityGrid::omnidirectional()),
            amplitude,
        )
    }

    /// Create a classical directional source with validated beam widths.
    ///
    /// Widths must be finite and in `(0, 360]` degrees — the same contract
    /// [`Directivity::try_amplitude`] enforces at evaluation time. A zero
    /// width would divide by zero in the Gaussian beam below.
    pub fn try_classical(
        position: Point3D,
        h_angle: f64,
        v_angle: f64,
        amplitude: f64,
    ) -> Result<Self, DirectivityGridError> {
        for (name, width) in [
            ("horizontal beam width", h_angle),
            ("vertical beam width", v_angle),
        ] {
            if !width.is_finite() || width <= 0.0 || width > 360.0 {
                return Err(DirectivityGridError(format!(
                    "{name} must be finite and in (0, 360] degrees"
                )));
            }
        }
        Ok(Self::classical(position, h_angle, v_angle, amplitude))
    }

    /// Create a classical directional source
    pub fn classical(position: Point3D, h_angle: f64, v_angle: f64, amplitude: f64) -> Self {
        Self::new(
            position,
            Directivity::Classical { h_angle, v_angle },
            amplitude,
        )
    }

    /// Set crossover filter
    pub fn with_crossover(mut self, crossover: CrossoverFilter) -> Self {
        self.crossover = crossover;
        self
    }

    /// Set source name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Set source forward axis (acoustic axis) in world coordinates.
    /// Need not be unit length; normalized internally.
    pub fn with_orientation(mut self, orientation: Point3D) -> Self {
        self.orientation = orientation;
        self
    }

    /// Point this source toward `target` (e.g., the listening position).
    pub fn aimed_at(mut self, target: Point3D) -> Self {
        let dir = target.sub(&self.position);
        if dir.length() > 1e-10 {
            self.orientation = dir.normalize().expect("non-zero by length check");
        }
        self
    }

    /// Frequency-dependent amplitude excluding directivity (`amplitude × crossover`).
    ///
    /// Propagation solvers should combine this with [`Self::directivity_factor`]
    /// at source-quadrature or source-basis points. Directivity is a source
    /// property and must not be applied later at the receiver.
    pub fn isotropic_amplitude(&self, frequency: f64) -> f64 {
        self.amplitude * self.crossover.amplitude_at_frequency(frequency)
    }

    /// Return whether this source's directivity is a true complex monopole.
    ///
    /// External reference adapters use this checked identity before reducing
    /// a source to its declared-frequency isotropic amplitudes.
    pub fn is_isotropic(&self) -> bool {
        self.directivity.is_isotropic()
    }

    /// Pure directivity factor toward a world point (no amplitude, no crossover).
    /// Returns 1.0 if `point` coincides with the source position.
    pub fn directivity_factor(&self, point: &Point3D, frequency: f64) -> f64 {
        let world_dir = point.sub(&self.position);
        let r = world_dir.length();
        if r < 1e-10 {
            return 1.0;
        }
        let world_unit = Point3D::new(world_dir.x / r, world_dir.y / r, world_dir.z / r);
        let local = self.world_to_local(world_unit);
        self.directivity.amplitude_local(local, frequency)
    }

    /// Checked directivity factor toward a world-space point.
    ///
    /// This validates source/target coordinates, source orientation, frequency,
    /// and the directivity representation before interpolation. The legacy
    /// [`Self::directivity_factor`] remains for trusted hot-path callers.
    pub fn try_directivity_factor(
        &self,
        point: &Point3D,
        frequency: f64,
    ) -> Result<f64, DirectivityGridError> {
        for (name, coordinates) in [
            ("source position", &self.position),
            ("source orientation", &self.orientation),
            ("target point", point),
        ] {
            if [coordinates.x, coordinates.y, coordinates.z]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(DirectivityGridError(format!("{name} must be finite")));
            }
        }
        let orientation_length = self.orientation.length();
        if !orientation_length.is_finite() || orientation_length < 1e-10 {
            return Err(DirectivityGridError(
                "source orientation must have a finite non-zero length".into(),
            ));
        }

        let world_dir = point.sub(&self.position);
        let r = world_dir.length();
        if !r.is_finite() {
            return Err(DirectivityGridError(
                "source-to-target direction must be finite".into(),
            ));
        }
        if r < 1e-10 {
            // Still enter the directivity validation boundary, even though the
            // coincident-point convention returns the isotropic factor.
            self.directivity
                .try_amplitude(std::f64::consts::FRAC_PI_2, 0.0, frequency)?;
            return Ok(1.0);
        }
        let world_unit = Point3D::new(world_dir.x / r, world_dir.y / r, world_dir.z / r);
        let local = self.world_to_local(world_unit);
        self.directivity.try_amplitude_local(local, frequency)
    }

    /// Total directional amplitude (`amplitude × directivity × crossover`) toward a point.
    pub fn amplitude_towards(&self, point: &Point3D, frequency: f64) -> f64 {
        self.isotropic_amplitude(frequency) * self.directivity_factor(point, frequency)
    }

    /// Checked total directional amplitude toward a world-space point.
    pub fn try_amplitude_towards(
        &self,
        point: &Point3D,
        frequency: f64,
    ) -> Result<f64, DirectivityGridError> {
        let factor = self.try_directivity_factor(point, frequency)?;
        let amplitude = self.isotropic_amplitude(frequency) * factor;
        if !amplitude.is_finite() {
            return Err(DirectivityGridError(
                "source amplitude and crossover response must produce a finite value".into(),
            ));
        }
        Ok(amplitude)
    }

    /// Complex frequency-dependent amplitude excluding directivity
    /// (`amplitude × complex crossover`, including crossover phase/group delay).
    pub fn complex_isotropic_amplitude(&self, frequency: f64) -> Complex64 {
        self.amplitude * self.crossover.complex_amplitude_at_frequency(frequency)
    }

    /// Complex directivity factor toward a world point. Magnitude-only
    /// directivity models return their real amplitude at zero phase; the
    /// complex grid returns the full pattern phase.
    pub fn complex_directivity_factor(&self, point: &Point3D, frequency: f64) -> Complex64 {
        let world_dir = point.sub(&self.position);
        let r = world_dir.length();
        if r < 1e-10 {
            return Complex64::new(1.0, 0.0);
        }
        let world_unit = Point3D::new(world_dir.x / r, world_dir.y / r, world_dir.z / r);
        let local = self.world_to_local(world_unit);
        self.directivity.complex_amplitude_local(local, frequency)
    }

    /// Total complex directional amplitude
    /// (`amplitude × complex directivity × complex crossover`) toward a point.
    pub fn complex_amplitude_towards(&self, point: &Point3D, frequency: f64) -> Complex64 {
        self.complex_isotropic_amplitude(frequency)
            * self.complex_directivity_factor(point, frequency)
    }

    /// Rotate a world-frame unit vector into the source's local frame, where
    /// `orientation` becomes local `+Y`. The orthonormal basis is built using
    /// world `+Z` as the up reference (with a fallback for vertically-pointing sources).
    fn world_to_local(&self, world_unit: Point3D) -> Point3D {
        let fwd = self
            .orientation
            .normalize()
            .unwrap_or_else(|| Point3D::new(0.0, 1.0, 0.0));
        let world_up = Point3D::new(0.0, 0.0, 1.0);
        // right = fwd × world_up
        let right = fwd.cross(&world_up).normalize().unwrap_or_else(|| {
            // fwd is parallel to world_up (vertically-firing source) — pick +X.
            Point3D::new(1.0, 0.0, 0.0)
        });
        // up = right × fwd  (completes a right-handed basis with fwd as +Y, up as +Z)
        let up = right.cross(&fwd);
        Point3D::new(
            world_unit.dot(&right),
            world_unit.dot(&fwd),
            world_unit.dot(&up),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directivity_grid_validates_axes_and_shape() {
        let magnitude = HostMatrix::from_elem(2, 2, 1.0);
        assert!(DirectivityGrid::try_new(vec![], vec![0.0, 180.0], magnitude.clone()).is_err());
        assert!(
            DirectivityGrid::try_new(vec![0.0, 10.0], vec![0.0, 90.0, 180.0], magnitude).is_err()
        );

        let valid_magnitude = HostMatrix::from_elem(2, 2, 1.0);
        for horizontal in [
            vec![0.0, 0.0],
            vec![10.0, 5.0],
            vec![0.0, f64::NAN],
            vec![0.0, 360.0],
        ] {
            assert!(
                DirectivityGrid::try_new(horizontal, vec![0.0, 180.0], valid_magnitude.clone(),)
                    .is_err()
            );
        }
        for vertical in [
            vec![0.0, 0.0],
            vec![90.0, 45.0],
            vec![0.0, f64::INFINITY],
            vec![-5.0, 180.0],
        ] {
            assert!(
                DirectivityGrid::try_new(vec![0.0, 180.0], vertical, valid_magnitude.clone(),)
                    .is_err()
            );
        }
        let non_finite = HostMatrix::from_row_major_vec(2, 2, vec![1.0, f64::NAN, 1.0, 1.0]);
        assert!(DirectivityGrid::try_new(vec![0.0, 180.0], vec![0.0, 180.0], non_finite).is_err());
    }

    #[test]
    fn crossover_checked_entry_rejects_degenerate_cutoffs() {
        let zero = CrossoverFilter::Lowpass {
            cutoff_freq: 0.0,
            order: 2,
        };
        assert!(zero.try_amplitude_at_frequency(100.0).is_err());
        let nan = CrossoverFilter::Highpass {
            cutoff_freq: f64::NAN,
            order: 2,
        };
        assert!(nan.try_amplitude_at_frequency(100.0).is_err());
        let inverted = CrossoverFilter::Bandpass {
            low_cutoff: 500.0,
            high_cutoff: 100.0,
            order: 2,
        };
        assert!(inverted.try_amplitude_at_frequency(200.0).is_err());
        let valid = CrossoverFilter::Lowpass {
            cutoff_freq: 100.0,
            order: 2,
        };
        let checked = valid.try_amplitude_at_frequency(100.0).unwrap();
        assert!((checked - valid.amplitude_at_frequency(100.0)).abs() < 1e-15);
        assert!(
            CrossoverFilter::FullRange
                .try_amplitude_at_frequency(100.0)
                .unwrap()
                == 1.0
        );
    }

    #[test]
    fn try_classical_rejects_zero_width() {
        assert!(Source::try_classical(Point3D::new(0.0, 0.0, 0.0), 0.0, 90.0, 1.0).is_err());
        assert!(Source::try_classical(Point3D::new(0.0, 0.0, 0.0), 90.0, 90.0, 1.0).is_ok());
    }

    #[test]
    fn directivity_grid_accepts_five_degree_sampling() {
        let horizontal: Vec<f64> = (0..72).map(|index| index as f64 * 5.0).collect();
        let vertical: Vec<f64> = (0..=36).map(|index| index as f64 * 5.0).collect();
        let magnitude = HostMatrix::from_fn(vertical.len(), horizontal.len(), |row, column| {
            1.0 + vertical[row] / 180.0 + horizontal[column] / 360.0
        });
        let grid = DirectivityGrid::try_new(horizontal, vertical, magnitude).unwrap();
        let value = grid.interpolate(87.5_f64.to_radians(), 42.5_f64.to_radians());
        let expected = 1.0 + 87.5 / 180.0 + 42.5 / 360.0;
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn directivity_grid_interpolates_offset_nonuniform_axes_and_seam() {
        let values = vec![
            1.0, 0.0, 1.0, // elevation 0
            1.0, 0.0, 1.0, // elevation 90
            1.0, 0.0, 1.0, // elevation 180
        ];
        let grid = DirectivityGrid::try_new(
            vec![10.0, 180.0, 350.0],
            vec![0.0, 90.0, 180.0],
            HostMatrix::from_row_major_vec(3, 3, values),
        )
        .unwrap();

        let before_seam = grid.interpolate(90.0_f64.to_radians(), 359.999_f64.to_radians());
        let after_seam = grid.interpolate(90.0_f64.to_radians(), 0.001_f64.to_radians());
        assert!((before_seam - after_seam).abs() < 1e-3);
        assert!(
            (grid.interpolate(90.0_f64.to_radians(), 10.0_f64.to_radians()) - 1.0).abs() < 1e-12
        );
    }

    #[test]
    fn directivity_grid_poles_are_azimuth_invariant() {
        let grid = DirectivityGrid::try_new(
            vec![0.0, 120.0, 240.0],
            vec![0.0, 90.0, 180.0],
            HostMatrix::from_row_major_vec(
                3,
                3,
                vec![
                    1.0, 2.0, 3.0, // north pole: mean 2
                    7.0, 8.0, 9.0, // equator
                    4.0, 5.0, 6.0, // south pole: mean 5
                ],
            ),
        )
        .unwrap();

        for phi in [0.0, 0.7, std::f64::consts::PI, 5.9] {
            assert!((grid.interpolate(0.0, phi) - 2.0).abs() < 1e-12);
            assert!((grid.interpolate(std::f64::consts::PI, phi) - 5.0).abs() < 1e-12);
        }
    }
    use std::f64::consts::PI;

    #[test]
    fn test_omnidirectional_pattern() {
        let grid = DirectivityGrid::omnidirectional();
        // Should be 1.0 in all directions
        assert!((grid.interpolate(0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((grid.interpolate(PI / 2.0, PI) - 1.0).abs() < 1e-6);
        assert!((grid.interpolate(PI, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossover_lowpass() {
        let crossover = CrossoverFilter::Lowpass {
            cutoff_freq: 100.0,
            order: 2,
        };
        // Well below cutoff
        assert!((crossover.amplitude_at_frequency(10.0) - 1.0).abs() < 0.1);
        // At cutoff (-3dB point)
        let at_cutoff = crossover.amplitude_at_frequency(100.0);
        assert!(at_cutoff > 0.6 && at_cutoff < 0.8);
        // Well above cutoff
        assert!(crossover.amplitude_at_frequency(1000.0) < 0.1);
    }

    #[test]
    fn test_source_amplitude() {
        let source = Source::omnidirectional(Point3D::new(0.0, 0.0, 0.0), 1.0);
        let amp = source.amplitude_towards(&Point3D::new(1.0, 0.0, 0.0), 1000.0);
        assert!((amp - 1.0).abs() < 1e-6);
    }

    #[test]
    fn checked_directivity_amplitude_matches_legacy_and_rejects_bad_inputs() {
        let grid = Directivity::Grid(DirectivityGrid::omnidirectional());
        assert_eq!(
            grid.try_amplitude(PI / 3.0, -0.4, 1_000.0).unwrap(),
            grid.amplitude(PI / 3.0, -0.4, 1_000.0)
        );
        assert!(grid.try_amplitude(f64::NAN, 0.0, 1_000.0).is_err());
        assert!(grid.try_amplitude(PI / 2.0, 0.0, 0.0).is_err());

        let malformed = Directivity::Classical {
            h_angle: 0.0,
            v_angle: 90.0,
        };
        assert!(
            malformed
                .try_amplitude(PI / 2.0, PI / 2.0, 1_000.0)
                .is_err()
        );
        let valid = Directivity::Classical {
            h_angle: 90.0,
            v_angle: 60.0,
        };
        assert_eq!(
            valid.try_amplitude(PI / 2.0, PI / 2.0, 1_000.0).unwrap(),
            valid.amplitude(PI / 2.0, PI / 2.0, 1_000.0)
        );
    }

    #[test]
    fn checked_source_directivity_admission_matches_legacy_and_rejects_bad_inputs() {
        let source = Source::classical(Point3D::new(0.0, 0.0, 0.0), 90.0, 60.0, 2.0);
        let target = Point3D::new(1.0, 0.5, 0.25);
        assert_eq!(
            source.try_directivity_factor(&target, 1_000.0).unwrap(),
            source.directivity_factor(&target, 1_000.0)
        );
        assert_eq!(
            source.try_amplitude_towards(&target, 1_000.0).unwrap(),
            source.amplitude_towards(&target, 1_000.0)
        );
        assert!(
            source
                .try_directivity_factor(&Point3D::new(f64::NAN, 0.0, 0.0), 1_000.0)
                .is_err()
        );
        assert!(source.try_directivity_factor(&target, 0.0).is_err());
        let zero_orientation = source.clone().with_orientation(Point3D::new(0.0, 0.0, 0.0));
        assert!(
            zero_orientation
                .try_directivity_factor(&target, 1_000.0)
                .is_err()
        );

        let malformed = Source::classical(Point3D::new(0.0, 0.0, 0.0), 0.0, 60.0, 1.0);
        assert!(malformed.try_directivity_factor(&target, 1_000.0).is_err());
    }

    #[test]
    fn test_classical_source_beaming() {
        let source = Source::classical(Point3D::new(0.0, 0.0, 0.0), 60.0, 40.0, 1.0);

        // At low freq (50Hz), should be omni
        let amp_low = source.amplitude_towards(&Point3D::new(0.0, -1.0, 0.0), 50.0); // Rear (180 deg)
        // Forward is +Y. Rear is -Y.
        // Forward: theta=90, phi=90.
        // Rear: theta=90, phi=270 (-90).
        // Delta phi = 180.
        // If omni, width=360. x = 180/180 = 1. m = 0.5.
        // Wait, omni means width=360 implies +/- 180 deg?
        // Width is full width. +/- 180 covers full circle.
        // x = delta / 180.
        // Rear delta = 180. x=1. m=0.5 (-6dB).
        // So Omni via this formula is -6dB at rear.
        // True Omni should be 1.0 everywhere.
        // My Gaussian model with width=360 is not perfectly Omni.
        // But "Typically omni below 80Hz" usually allows some directivity or just "wide".
        // If I want true Omni, I should handle width >= 360 specially or increase width to infinity.
        // But 360 width (-6dB at +/-180) is acceptable for "Omni-like".
        assert!(amp_low > 0.4);

        // At high freq (2000Hz), should be narrow
        // H=60 -> +/- 30 deg is -6dB.
        // Side (90 deg off axis): x = 90/30 = 3. m = 0.5^9 = 0.0019.
        let amp_high_side = source.amplitude_towards(&Point3D::new(1.0, 0.0, 0.0), 2000.0); // Side (X axis, phi=0, delta=90)
        assert!(amp_high_side < 0.1);

        // On axis should be 1.0
        let amp_high_on = source.amplitude_towards(&Point3D::new(0.0, 1.0, 0.0), 2000.0); // Forward (+Y)
        assert!((amp_high_on - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_source_orientation_rotates_lobe() {
        // A surround speaker at the rear of a room, aimed at a listener in front of it.
        // Forward axis = -Y. The on-axis listener should see full amplitude; a point
        // straight behind (in +Y) should be attenuated.
        let source = Source::classical(Point3D::new(0.0, 0.0, 0.0), 60.0, 40.0, 1.0)
            .with_orientation(Point3D::new(0.0, -1.0, 0.0));

        let on_axis = source.amplitude_towards(&Point3D::new(0.0, -1.0, 0.0), 2000.0);
        assert!(
            (on_axis - 1.0).abs() < 1e-6,
            "on-axis (–Y) should be 0 dB, got {on_axis}"
        );

        // Behind the speaker (+Y world) is now 180° off-axis.
        let behind = source.amplitude_towards(&Point3D::new(0.0, 1.0, 0.0), 2000.0);
        assert!(
            behind < 0.05,
            "rear of rotated speaker should be heavily attenuated at 2 kHz, got {behind}"
        );
    }

    #[test]
    fn test_directivity_grid_mirrored_is_left_right_flip() {
        // Lobe toward local +X (phi = 0): not mirror-symmetric.
        let horizontal: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();
        let magnitude = HostMatrix::from_fn(vertical.len(), horizontal.len(), |v, h| {
            0.5 * (1.0 + vertical[v].to_radians().sin() * horizontal[h].to_radians().cos())
        });
        let grid = DirectivityGrid::try_new(horizontal, vertical, magnitude).unwrap();
        let mirrored = grid.mirrored();
        // Mirrored value at (theta, phi) equals original at (theta, 180° - phi).
        for &theta_deg in &[30.0_f64, 90.0, 150.0] {
            for &phi_deg in &[0.0_f64, 40.0, 130.0, 260.0] {
                let theta = theta_deg.to_radians();
                let phi = phi_deg.to_radians();
                let flipped_phi = (180.0 - phi_deg).rem_euclid(360.0).to_radians();
                assert!(
                    (mirrored.interpolate(theta, phi) - grid.interpolate(theta, flipped_phi)).abs()
                        < 1e-12,
                    "theta={theta_deg} phi={phi_deg}"
                );
            }
        }
        // Mirroring twice restores the original grid exactly.
        let twice = mirrored.mirrored();
        assert_eq!(twice.horizontal_angles(), grid.horizontal_angles());
        assert_eq!(twice.vertical_angles(), grid.vertical_angles());
        assert_eq!(twice.magnitude(), grid.magnitude());
        assert!(!Directivity::Grid(grid).is_mirror_symmetric());
    }

    #[test]
    fn test_directivity_mirror_symmetry_classification() {
        assert!(Directivity::Grid(DirectivityGrid::cardioid()).is_mirror_symmetric());
        assert!(Directivity::Grid(DirectivityGrid::omnidirectional()).is_mirror_symmetric());
        assert!(
            Directivity::Classical {
                h_angle: 60.0,
                v_angle: 40.0
            }
            .is_mirror_symmetric()
        );
        // Mirroring a Classical pattern is a no-op.
        let classical = Directivity::Classical {
            h_angle: 60.0,
            v_angle: 40.0,
        };
        let mirrored = classical.mirrored();
        for &(theta, phi) in &[(1.0, 0.3), (1.2, 2.0)] {
            assert_eq!(
                classical.amplitude(theta, phi, 1000.0),
                mirrored.amplitude(theta, phi, 1000.0)
            );
        }
    }

    #[test]
    fn test_source_aimed_at() {
        let src = Source::classical(Point3D::new(1.0, 1.0, 1.0), 60.0, 40.0, 1.0)
            .aimed_at(Point3D::new(2.0, 1.0, 1.0));
        // Aimed at +X. Listener directly along +X is on-axis, +Y/+Z are off-axis.
        let on_axis = src.amplitude_towards(&Point3D::new(2.0, 1.0, 1.0), 2000.0);
        let off_axis = src.amplitude_towards(&Point3D::new(1.0, 2.0, 1.0), 2000.0);
        assert!(on_axis > 0.99);
        assert!(off_axis < on_axis);
    }

    #[test]
    fn test_isotropic_amplitude_excludes_directivity() {
        let s = Source::classical(Point3D::new(0.0, 0.0, 0.0), 60.0, 40.0, 1.0).with_crossover(
            CrossoverFilter::Highpass {
                cutoff_freq: 80.0,
                order: 4,
            },
        );
        // Crossover only (no directivity factor): well above cutoff → ~1
        let iso = s.isotropic_amplitude(1000.0);
        assert!((iso - 1.0).abs() < 0.01);
        // Below cutoff: heavily attenuated
        let iso_low = s.isotropic_amplitude(20.0);
        assert!(iso_low < 0.05);
    }

    #[test]
    fn complex_crossover_magnitude_matches_scalar_and_carries_butterworth_phase() {
        for filter in [
            CrossoverFilter::Lowpass {
                cutoff_freq: 500.0,
                order: 2,
            },
            CrossoverFilter::Highpass {
                cutoff_freq: 80.0,
                order: 4,
            },
            CrossoverFilter::Bandpass {
                low_cutoff: 80.0,
                high_cutoff: 500.0,
                order: 3,
            },
        ] {
            for frequency in [20.0, 55.5, 100.0, 317.0, 1000.0, 8000.0] {
                let magnitude = filter.amplitude_at_frequency(frequency);
                let complex = filter.complex_amplitude_at_frequency(frequency);
                assert!(
                    (complex.norm() - magnitude).abs() < 1e-12,
                    "{filter:?} at {frequency} Hz: |H|={} vs scalar {magnitude}",
                    complex.norm()
                );
            }
        }

        // Second-order Butterworth: exactly -90 deg (LP) / +90 deg (HP) at cutoff.
        let lp = CrossoverFilter::Lowpass {
            cutoff_freq: 500.0,
            order: 2,
        }
        .complex_amplitude_at_frequency(500.0);
        assert!((lp.arg() + std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert!((lp.norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        let hp = CrossoverFilter::Highpass {
            cutoff_freq: 500.0,
            order: 2,
        }
        .complex_amplitude_at_frequency(500.0);
        assert!((hp.arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);

        // First-order LP: H(0) = 1 exactly (unity DC gain).
        let dc = CrossoverFilter::Lowpass {
            cutoff_freq: 500.0,
            order: 1,
        }
        .complex_amplitude_at_frequency(1e-9);
        assert!((dc - Complex64::new(1.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn complex_directivity_grid_interpolates_angles_and_frequency() {
        let h_angles: Vec<f64> = vec![0.0, 90.0, 180.0, 270.0];
        let v_angles: Vec<f64> = vec![0.0, 90.0, 180.0];
        let pattern = |scale: Complex64| {
            let mut m = HostMatrix::<Complex64>::zeros(3, 4);
            for v in 0..3 {
                for h in 0..4 {
                    m[[v, h]] = scale;
                }
            }
            m
        };
        let grid = ComplexDirectivityGrid::try_new(
            h_angles.clone(),
            v_angles.clone(),
            vec![100.0, 200.0],
            vec![
                pattern(Complex64::new(1.0, 0.0)),
                pattern(Complex64::new(0.0, 1.0)),
            ],
        )
        .expect("valid grid");

        // On a sample frequency the value is the sample itself, anywhere.
        let at_100 = grid.interpolate(1.1, 2.2, 100.0);
        assert!((at_100 - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        let typed_frequencies = grid.frequency_quantities();
        assert_eq!(typed_frequencies[0], FrequencyHz::new(100.0));
        let typed_at_100 = grid
            .try_interpolate_at(
                1.1,
                2.2,
                FrequencyHz::try_new(100.0).expect("positive frequency"),
            )
            .expect("typed frequency interpolation should preserve the grid contract");
        assert!((typed_at_100 - at_100).norm() < 1e-12);
        // Halfway in frequency: linear blend of the two patterns.
        let at_150 = grid.interpolate(0.7, -0.4, 150.0);
        assert!((at_150 - Complex64::new(0.5, 0.5)).norm() < 1e-12);
        // Outside the axis: edge pattern held constant.
        let at_10k = grid.interpolate(1.0, 0.0, 10_000.0);
        assert!((at_10k - Complex64::new(0.0, 1.0)).norm() < 1e-12);

        // Validation failures.
        assert!(
            ComplexDirectivityGrid::try_new(
                h_angles.clone(),
                v_angles.clone(),
                vec![200.0, 100.0],
                vec![pattern(Complex64::new(1.0, 0.0)); 2],
            )
            .is_err(),
            "non-increasing frequency axis must be rejected"
        );
        assert!(
            ComplexDirectivityGrid::try_new(
                h_angles.clone(),
                v_angles.clone(),
                vec![100.0],
                vec![pattern(Complex64::new(1.0, 0.0)); 2],
            )
            .is_err(),
            "pattern/frequency count mismatch must be rejected"
        );

        // Pole handling: azimuthal mean of the pole row.
        let pole = grid.interpolate(0.0, 3.0, 100.0);
        assert!((pole - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn complex_directivity_grid_mirror_and_symmetry_predicates() {
        let h_angles: Vec<f64> = vec![0.0, 90.0, 180.0, 270.0];
        let v_angles: Vec<f64> = vec![0.0, 180.0];
        // Lobe toward +X (phi=0) with a phase tilt: NOT mirror-symmetric.
        let mut m = HostMatrix::<Complex64>::zeros(2, 4);
        m[[0, 0]] = Complex64::new(1.0, 0.5);
        m[[0, 1]] = Complex64::new(0.5, 0.1);
        m[[0, 2]] = Complex64::new(0.1, 0.0);
        m[[0, 3]] = Complex64::new(0.5, -0.1);
        let grid = ComplexDirectivityGrid::try_new(h_angles, v_angles, vec![500.0], vec![m])
            .expect("valid grid");
        let directivity = Directivity::ComplexGrid(grid);
        assert!(!directivity.is_isotropic());
        assert!(!directivity.is_mirror_symmetric());

        // Mirroring twice returns the original samples.
        let twice = directivity.mirrored().mirrored();
        if let Directivity::ComplexGrid(twice_grid) = &twice {
            for (a, b) in twice_grid.patterns()[0].as_slice().iter().zip(
                if let Directivity::ComplexGrid(g) = &directivity {
                    g.patterns()[0].as_slice().iter()
                } else {
                    unreachable!()
                },
            ) {
                assert!((a - b).norm() < 1e-12, "double mirror must be identity");
            }
        } else {
            panic!("mirrored complex grid keeps its variant");
        }

        // Uniform 1+0i at all frequencies is isotropic and mirror-symmetric.
        let omni = Directivity::ComplexGrid(
            ComplexDirectivityGrid::try_new(
                vec![0.0, 90.0, 180.0, 270.0],
                vec![0.0, 180.0],
                vec![100.0, 1000.0],
                vec![
                    HostMatrix::from_elem(2, 4, Complex64::new(1.0, 0.0)),
                    HostMatrix::from_elem(2, 4, Complex64::new(1.0, 0.0)),
                ],
            )
            .expect("valid grid"),
        );
        assert!(omni.is_isotropic());
        assert!(omni.is_mirror_symmetric());
    }

    #[test]
    fn source_complex_factor_combines_crossover_phase_and_pattern_phase() {
        // Complex pattern with a 90 deg phase toward local +Y (forward), and a
        // second-order LP crossover: toward the forward axis the phases add.
        let pattern = HostMatrix::from_elem(2, 4, Complex64::new(0.0, 1.0));
        let source = Source {
            position: Point3D::new(0.0, 0.0, 0.0),
            orientation: Point3D::new(0.0, 1.0, 0.0),
            directivity: Directivity::ComplexGrid(
                ComplexDirectivityGrid::try_new(
                    vec![0.0, 90.0, 180.0, 270.0],
                    vec![0.0, 180.0],
                    vec![500.0],
                    vec![pattern],
                )
                .expect("valid grid"),
            ),
            amplitude: 2.0,
            crossover: CrossoverFilter::Lowpass {
                cutoff_freq: 500.0,
                order: 2,
            },
            name: "test".to_string(),
        };
        let total = source.complex_amplitude_towards(&Point3D::new(0.0, 3.0, 0.0), 500.0);
        // |H| = 2 * 1/sqrt(2); phase = 90 deg (pattern) - 90 deg (LP) = 0 deg.
        assert!((total.norm() - 2.0 * std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
        assert!(total.im.abs() < 1e-9, "phases should cancel: {total}");
        assert!(total.re > 0.0);
    }

    #[test]
    fn complex_grid_serde_round_trip_and_validation() {
        let horizontal = vec![0.0, 90.0, 180.0, 270.0];
        let vertical = vec![0.0, 90.0, 180.0];
        let frequencies = vec![250.0, 500.0, 1000.0];
        let patterns = frequencies
            .iter()
            .enumerate()
            .map(|(fi, _)| {
                HostMatrix::from_fn(vertical.len(), horizontal.len(), |v, h| {
                    Complex64::new(fi as f64 + v as f64 * 0.1, h as f64 * -0.05)
                })
            })
            .collect();
        let grid = ComplexDirectivityGrid::try_new(
            horizontal.clone(),
            vertical.clone(),
            frequencies.clone(),
            patterns,
        )
        .expect("valid grid");

        let json = serde_json::to_string(&grid).expect("serialize");
        let decoded: ComplexDirectivityGrid = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.horizontal_angles(), horizontal.as_slice());
        assert_eq!(decoded.vertical_angles(), vertical.as_slice());
        assert_eq!(decoded.frequencies(), frequencies.as_slice());
        for (original, restored) in grid.patterns().iter().zip(decoded.patterns()) {
            assert_eq!(original.as_slice(), restored.as_slice());
        }

        // Deserialization runs the same validation as try_new.
        let broken = json.replace("500.0", "-500.0");
        assert!(serde_json::from_str::<ComplexDirectivityGrid>(&broken).is_err());
        let ragged = r#"{"horizontal_angles":[0.0,90.0],"vertical_angles":[0.0,180.0],"frequencies":[100.0],"patterns":[[[[1.0,0.0]]]]}"#;
        assert!(serde_json::from_str::<ComplexDirectivityGrid>(ragged).is_err());
    }

    #[test]
    fn strict_interpolation_rejects_nonfinite_and_out_of_range_inputs() {
        let grid = DirectivityGrid::omnidirectional();
        assert!(grid.try_interpolate(-0.01, 0.0).is_err());
        assert!(
            grid.try_interpolate(std::f64::consts::PI, f64::NAN)
                .is_err()
        );

        let complex = ComplexDirectivityGrid::try_new(
            vec![0.0, 180.0],
            vec![0.0, 180.0],
            vec![100.0],
            vec![HostMatrix::from_elem(2, 2, Complex64::new(1.0, 0.0))],
        )
        .expect("valid grid");
        assert!(complex.try_interpolate(0.0, 0.0, 0.0).is_err());
        assert!(complex.try_interpolate(0.0, 0.0, f64::INFINITY).is_err());
    }
}
