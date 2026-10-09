//! 1D Analytical Solutions
//!
//! Exact solutions for 1D wave propagation problems.

use super::{AnalyticalSolution, Point};
use num_complex::Complex64;
use std::{f64::consts::PI, fmt};

/// Default speed of sound (m/s)
pub const SPEED_OF_SOUND: f64 = 343.0;

/// Invalid input to a checked one-dimensional reference solution.
///
/// The unchecked functions in this module are retained for compatibility with
/// trusted fixtures. New numerical-validation callers should use the
/// `try_*_1d` variants so malformed grids and singular modal problems are
/// rejected before a non-finite reference is produced.
#[derive(Debug, Clone, PartialEq)]
pub enum OneDimensionalWaveError {
    /// A scalar is non-finite or outside the model's stated domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// The sampled interval is empty, reversed, or non-finite.
    InvalidInterval {
        /// Start coordinate in metres.
        x_min: f64,
        /// End coordinate in metres.
        x_max: f64,
    },
    /// The spatial phase cannot be represented as finite binary64.
    PhaseOverflow,
    /// The lossy-wave amplitude would overflow at the supplied grid boundary.
    AmplitudeOverflow,
    /// The forced Dirichlet Helmholtz mode is at its singular eigenfrequency.
    Resonance {
        /// Driving wavenumber.
        wave_number: f64,
        /// Dirichlet modal wavenumber `nπ/L`.
        modal_wave_number: f64,
    },
}

impl fmt::Display for OneDimensionalWaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(f, "1D wave parameter {field} must be {requirement}")
            }
            Self::InvalidInterval { x_min, x_max } => write!(
                f,
                "1D wave interval must be finite and strictly increasing, got [{x_min}, {x_max}]"
            ),
            Self::PhaseOverflow => write!(f, "1D wave spatial phase is not finite"),
            Self::AmplitudeOverflow => write!(f, "1D damped-wave amplitude would overflow"),
            Self::Resonance {
                wave_number,
                modal_wave_number,
            } => write!(
                f,
                "1D forced Helmholtz mode is singular or unresolved near k={wave_number}, nπ/L={modal_wave_number}"
            ),
        }
    }
}

impl std::error::Error for OneDimensionalWaveError {}

fn validate_wave_number(wave_number: f64) -> Result<(), OneDimensionalWaveError> {
    if !wave_number.is_finite() || wave_number <= 0.0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "wave_number",
            requirement: "finite and positive",
        });
    }
    if !(wave_number * SPEED_OF_SOUND / (2.0 * PI)).is_finite() {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "wave_number",
            requirement: "small enough to yield a finite frequency",
        });
    }
    Ok(())
}

fn validate_grid(x_min: f64, x_max: f64, num_points: usize) -> Result<(), OneDimensionalWaveError> {
    if num_points < 2 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "num_points",
            requirement: "at least 2",
        });
    }
    if !x_min.is_finite() || !x_max.is_finite() || x_max <= x_min {
        return Err(OneDimensionalWaveError::InvalidInterval { x_min, x_max });
    }
    Ok(())
}

fn validate_phase(wave_number: f64, x_min: f64, x_max: f64) -> Result<(), OneDimensionalWaveError> {
    if !(wave_number * x_min.abs().max(x_max.abs())).is_finite() {
        return Err(OneDimensionalWaveError::PhaseOverflow);
    }
    Ok(())
}

/// 1D plane wave: p(x) = exp(ikx)
///
/// This is the simplest analytical solution. With the crate's
/// `exp(+iωt)` time convention, the spatial phasor `exp(+ikx)` propagates in
/// the negative x-direction. It assumes a homogeneous, lossless, stationary
/// fluid and unit peak pressure amplitude.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c = ω/c
/// * `x_min` - Start of domain
/// * `x_max` - End of domain
/// * `num_points` - Number of evaluation points
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::plane_wave_1d;
/// use std::f64::consts::PI;
///
/// let solution = plane_wave_1d(1.0, 0.0, 2.0 * PI, 100);
/// // At x = 0: p = exp(0) = 1
/// assert!((solution.pressure[0].re - 1.0).abs() < 1e-10);
/// ```
pub fn plane_wave_1d(
    wave_number: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> AnalyticalSolution {
    assert!(num_points >= 2, "Need at least 2 points");

    let dx = (x_max - x_min) / (num_points - 1) as f64;

    let positions: Vec<Point> = (0..num_points)
        .map(|i| Point::new_1d(x_min + i as f64 * dx))
        .collect();

    let pressure: Vec<Complex64> = positions
        .iter()
        .map(|p| {
            let kx = wave_number * p.x;
            Complex64::new(kx.cos(), kx.sin())
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("1D Plane Wave (k={})", wave_number),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "x_min": x_min,
            "x_max": x_max,
            "num_points": num_points,
            "speed_of_sound": SPEED_OF_SOUND,
            "wavelength": 2.0 * PI / wave_number,
        }),
    }
}

/// Checked [`plane_wave_1d`] reference for a finite, non-degenerate grid.
///
/// The legacy function remains suitable for trusted fixtures. This variant
/// rejects non-positive wavenumber, malformed grids, and values whose spatial
/// phase or derived frequency cannot be represented in binary64.
pub fn try_plane_wave_1d(
    wave_number: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> Result<AnalyticalSolution, OneDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    validate_grid(x_min, x_max, num_points)?;
    validate_phase(wave_number, x_min, x_max)?;
    Ok(plane_wave_1d(wave_number, x_min, x_max, num_points))
}

/// 1D standing wave: p(x) = sin(kx)
///
/// Standing wave pattern with nodes at x = nπ/k.
/// This represents the superposition of two counter-propagating waves.
/// It assumes a homogeneous, lossless, stationary fluid and uses the crate's
/// `exp(+iωt)` phasor convention. The returned field has unit peak pressure
/// amplitude at an antinode.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `x_min` - Start of domain
/// * `x_max` - End of domain
/// * `num_points` - Number of evaluation points
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::standing_wave_1d;
/// use std::f64::consts::PI;
///
/// let solution = standing_wave_1d(1.0, 0.0, PI, 100);
/// // At x = 0: sin(0) = 0 (node)
/// assert!(solution.pressure[0].norm() < 1e-10);
/// ```
pub fn standing_wave_1d(
    wave_number: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> AnalyticalSolution {
    assert!(num_points >= 2, "Need at least 2 points");

    let dx = (x_max - x_min) / (num_points - 1) as f64;

    let positions: Vec<Point> = (0..num_points)
        .map(|i| Point::new_1d(x_min + i as f64 * dx))
        .collect();

    let pressure: Vec<Complex64> = positions
        .iter()
        .map(|p| {
            let kx = wave_number * p.x;
            // sin(kx) represented as imaginary for consistency with wave convention
            Complex64::new(0.0, kx.sin())
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("1D Standing Wave (k={})", wave_number),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "x_min": x_min,
            "x_max": x_max,
            "num_points": num_points,
            "wavelength": 2.0 * PI / wave_number,
            "node_spacing": PI / wave_number,
        }),
    }
}

/// Checked [`standing_wave_1d`] reference for a finite, non-degenerate grid.
pub fn try_standing_wave_1d(
    wave_number: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> Result<AnalyticalSolution, OneDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    validate_grid(x_min, x_max, num_points)?;
    validate_phase(wave_number, x_min, x_max)?;
    Ok(standing_wave_1d(wave_number, x_min, x_max, num_points))
}

/// 1D wave with absorption: p(x) = exp(-αx) exp(+ikx)
///
/// Includes damping term for validation of lossy media.
/// The wave decays exponentially with penetration depth 1/α.
/// The phase uses the crate's `exp(+iωt)` convention. This phenomenological
/// attenuation model is not a dispersive constitutive model: use it only when
/// a real non-negative amplitude-decay coefficient is the intended reference.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `absorption` - α (damping coefficient, 1/m)
/// * `x_min` - Start of domain
/// * `x_max` - End of domain
/// * `num_points` - Number of evaluation points
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::damped_wave_1d;
///
/// let solution = damped_wave_1d(1.0, 0.1, 0.0, 10.0, 100);
/// // Magnitude should decay exponentially
/// let ratio = solution.pressure[99].norm() / solution.pressure[0].norm();
/// assert!((ratio - (-0.1 * 10.0_f64).exp()).abs() < 1e-6);
/// ```
pub fn damped_wave_1d(
    wave_number: f64,
    absorption: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> AnalyticalSolution {
    assert!(num_points >= 2, "Need at least 2 points");
    assert!(absorption >= 0.0, "Absorption must be non-negative");

    let dx = (x_max - x_min) / (num_points - 1) as f64;

    let positions: Vec<Point> = (0..num_points)
        .map(|i| Point::new_1d(x_min + i as f64 * dx))
        .collect();

    let pressure: Vec<Complex64> = positions
        .iter()
        .map(|p| {
            // exp(-(α + ik)x) = exp(-αx) * exp(-ikx)
            let damping = (-absorption * p.x).exp();
            let wave = Complex64::new((wave_number * p.x).cos(), (wave_number * p.x).sin());
            damping * wave
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("1D Damped Wave (k={}, α={})", wave_number, absorption),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "x_min": x_min,
            "x_max": x_max,
            "absorption": absorption,
            "penetration_depth": if absorption > 0.0 { 1.0 / absorption } else { f64::INFINITY },
            "quality_factor": wave_number / (2.0 * absorption),
        }),
    }
}

/// Checked [`damped_wave_1d`] reference for a finite, non-degenerate grid.
///
/// In addition to the lossless-plane-wave checks, this rejects negative or
/// non-finite attenuation and exponential growth that would overflow at
/// `x_min` (which is where the supplied increasing grid is largest).
pub fn try_damped_wave_1d(
    wave_number: f64,
    absorption: f64,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> Result<AnalyticalSolution, OneDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    validate_grid(x_min, x_max, num_points)?;
    validate_phase(wave_number, x_min, x_max)?;
    if !absorption.is_finite() || absorption < 0.0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "absorption",
            requirement: "finite and non-negative",
        });
    }
    let largest_exponent = -absorption * x_min;
    if !largest_exponent.is_finite() || largest_exponent > f64::MAX.ln() {
        return Err(OneDimensionalWaveError::AmplitudeOverflow);
    }
    Ok(damped_wave_1d(
        wave_number,
        absorption,
        x_min,
        x_max,
        num_points,
    ))
}

/// 1D Helmholtz solution in a bounded domain [0, L]
///
/// Solves: d²u/dx² + k²u = f(x)
/// with Dirichlet BC: u(0) = u(L) = 0
///
/// For f(x) = sin(nπx/L), the solution is:
/// u(x) = sin(nπx/L) / (k² - (nπ/L)²)
///
/// This manufactured solution assumes a homogeneous lossless 1D medium and
/// homogeneous Dirichlet boundary conditions. It is singular at a modal
/// wavenumber; use [`try_helmholtz_1d_mode`] for externally supplied data.
///
/// # Arguments
///
/// * `wave_number` - k
/// * `length` - L, domain length
/// * `mode_number` - n, mode number (n >= 1)
/// * `num_points` - Number of evaluation points
pub fn helmholtz_1d_mode(
    wave_number: f64,
    length: f64,
    mode_number: usize,
    num_points: usize,
) -> AnalyticalSolution {
    assert!(num_points >= 2, "Need at least 2 points");
    assert!(mode_number >= 1, "Mode number must be >= 1");

    let n = mode_number as f64;
    let kn = n * PI / length;

    // Check for resonance
    let denom = wave_number * wave_number - kn * kn;
    assert!(
        denom.abs() > 1e-10,
        "Resonance condition: k ≈ nπ/L, solution unbounded"
    );

    let dx = length / (num_points - 1) as f64;

    let positions: Vec<Point> = (0..num_points)
        .map(|i| Point::new_1d(i as f64 * dx))
        .collect();

    let pressure: Vec<Complex64> = positions
        .iter()
        .map(|p| {
            let u = (n * PI * p.x / length).sin() / denom;
            Complex64::new(u, 0.0)
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("1D Helmholtz Mode (k={}, n={})", wave_number, mode_number),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "length": length,
            "mode_number": mode_number,
            "resonant_wavenumber": kn,
            "detuning": wave_number - kn,
        }),
    }
}

/// Checked [`helmholtz_1d_mode`] manufactured solution.
///
/// The checked validity region is finite positive `k` and `L`, mode number at
/// least one, and at least two samples. It retains the legacy `1e-10` absolute
/// singularity guard around `k² = (nπ/L)²`, ensuring parity wherever the
/// compatibility function is defined.
pub fn try_helmholtz_1d_mode(
    wave_number: f64,
    length: f64,
    mode_number: usize,
    num_points: usize,
) -> Result<AnalyticalSolution, OneDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    if !length.is_finite() || length <= 0.0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "length",
            requirement: "finite and positive",
        });
    }
    if mode_number == 0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "mode_number",
            requirement: "at least 1",
        });
    }
    if num_points < 2 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "num_points",
            requirement: "at least 2",
        });
    }

    let modal_wave_number = mode_number as f64 * PI / length;
    let denominator = wave_number * wave_number - modal_wave_number * modal_wave_number;
    if !modal_wave_number.is_finite() || !denominator.is_finite() {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "length or mode_number",
            requirement: "small enough to yield finite modal coefficients",
        });
    }
    if denominator.abs() <= 1e-10 {
        return Err(OneDimensionalWaveError::Resonance {
            wave_number,
            modal_wave_number,
        });
    }

    Ok(helmholtz_1d_mode(
        wave_number,
        length,
        mode_number,
        num_points,
    ))
}

/// Rigid-walled one-dimensional duct eigenmode on `[0, L]`.
///
/// The pressure eigenfunction is `p_n(x) = cos(nπx/L)` for `n >= 1`, with
/// wavenumber `k_n = nπ/L` and frequency `f_n = nc/(2L)`. With the crate's
/// `exp(+iωt)` convention, the corresponding particle velocity is
/// proportional to `i ∂p/∂x`; therefore the zero pressure gradient at both
/// ends enforces the rigid-wall (zero normal velocity) boundary condition.
///
/// The reference assumes a uniform, lossless, stationary duct below cutoff,
/// with unit peak pressure amplitude. It does not model end radiation, losses,
/// mean flow, or transverse modes.
pub fn rigid_duct_1d_mode(
    length: f64,
    mode_number: usize,
    num_points: usize,
) -> AnalyticalSolution {
    assert!(length > 0.0, "Duct length must be positive");
    assert!(mode_number >= 1, "Mode number must be at least 1");
    assert!(num_points >= 2, "Need at least 2 points");

    let wave_number = mode_number as f64 * PI / length;
    let dx = length / (num_points - 1) as f64;
    let positions: Vec<Point> = (0..num_points)
        .map(|i| Point::new_1d(i as f64 * dx))
        .collect();
    let pressure = positions
        .iter()
        .map(|point| Complex64::new((wave_number * point.x).cos(), 0.0))
        .collect();
    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("1D Rigid Duct Mode (n={mode_number})"),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "length": length,
            "mode_number": mode_number,
            "boundary_condition": "rigid (zero normal particle velocity)",
            "mode_shape": "cos(n*pi*x/L)",
            "speed_of_sound": SPEED_OF_SOUND,
            "modal_wavelength": 2.0 * length / mode_number as f64,
        }),
    }
}

/// Checked [`rigid_duct_1d_mode`] reference.
///
/// This rejects non-finite/non-positive length, the zero-frequency constant
/// mode, empty sample sets, and modal quantities that cannot be represented
/// as finite binary64 values.
pub fn try_rigid_duct_1d_mode(
    length: f64,
    mode_number: usize,
    num_points: usize,
) -> Result<AnalyticalSolution, OneDimensionalWaveError> {
    if !length.is_finite() || length <= 0.0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "length",
            requirement: "finite and positive",
        });
    }
    if mode_number == 0 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "mode_number",
            requirement: "at least 1 (the DC mode is excluded)",
        });
    }
    if num_points < 2 {
        return Err(OneDimensionalWaveError::InvalidParameter {
            field: "num_points",
            requirement: "at least 2",
        });
    }
    let wave_number = mode_number as f64 * PI / length;
    validate_wave_number(wave_number)?;
    validate_phase(wave_number, 0.0, length)?;
    Ok(rigid_duct_1d_mode(length, mode_number, num_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_plane_wave_1d() {
        let k = 1.0;
        let solution = plane_wave_1d(k, 0.0, 2.0 * PI, 100);

        // Check boundary values
        assert_abs_diff_eq!(solution.pressure[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(solution.pressure[0].im, 0.0, epsilon = 1e-10);

        // At x = 2π, should return to p = 1
        let last_idx = solution.pressure.len() - 1;
        assert_abs_diff_eq!(solution.pressure[last_idx].re, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(solution.pressure[last_idx].im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_plane_wave_unit_magnitude() {
        let k = 2.5;
        let solution = plane_wave_1d(k, 0.0, 10.0, 50);

        // |exp(ikx)| = 1 for all x
        for p in &solution.pressure {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_standing_wave_nodes() {
        let k = 1.0;
        let solution = standing_wave_1d(k, 0.0, PI, 200);

        // At x = 0, sin(0) = 0
        assert_abs_diff_eq!(solution.pressure[0].im, 0.0, epsilon = 1e-10);

        // At x = π/2, sin(π/2) = 1
        let dx = PI / 199.0;
        let target_idx = ((PI / 2.0) / dx).round() as usize;
        assert_abs_diff_eq!(solution.pressure[target_idx].im, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_damped_wave_decay() {
        let k = 1.0;
        let alpha = 0.1;
        let solution = damped_wave_1d(k, alpha, 0.0, 10.0, 100);

        // Magnitude should decay exponentially
        let mag_start = solution.pressure[0].norm();
        let mag_end = solution.pressure[solution.pressure.len() - 1].norm();

        let expected_ratio = (-alpha * 10.0).exp();
        assert_abs_diff_eq!(mag_end / mag_start, expected_ratio, epsilon = 1e-6);
    }

    #[test]
    fn test_wavelength() {
        let k = 2.0;
        let wavelength = 2.0 * PI / k;

        let solution = plane_wave_1d(k, 0.0, wavelength, 100);

        // After one wavelength, phase should return to 0
        let p0 = solution.pressure[0];
        let p_end = solution.pressure[solution.pressure.len() - 1];

        assert_abs_diff_eq!(p0.re, p_end.re, epsilon = 1e-6);
        assert_abs_diff_eq!(p0.im, p_end.im, epsilon = 1e-6);
    }

    #[test]
    fn test_helmholtz_1d_mode() {
        let k = 1.0;
        let l = PI; // Length
        let n = 2; // Mode number

        let solution = helmholtz_1d_mode(k, l, n, 100);

        // Check boundary conditions: u(0) = u(L) = 0
        assert_abs_diff_eq!(solution.pressure[0].re, 0.0, epsilon = 1e-10);
        let last = solution.pressure.len() - 1;
        assert_abs_diff_eq!(solution.pressure[last].re, 0.0, epsilon = 1e-6);

        // Check that solution is non-zero in interior
        let mid = solution.pressure.len() / 4; // Not at a node
        assert!(solution.pressure[mid].norm() > 1e-10);
    }

    #[test]
    #[should_panic(expected = "Resonance")]
    fn test_helmholtz_resonance() {
        let l = PI;
        let k = 1.0; // k = π/L for n=1, resonance
        helmholtz_1d_mode(k, l, 1, 100);
    }

    #[test]
    fn checked_1d_references_match_legacy_values() {
        let plane = try_plane_wave_1d(1.5, -0.5, 1.0, 8).unwrap();
        assert_eq!(plane.pressure, plane_wave_1d(1.5, -0.5, 1.0, 8).pressure);

        let standing = try_standing_wave_1d(1.5, -0.5, 1.0, 8).unwrap();
        assert_eq!(
            standing.pressure,
            standing_wave_1d(1.5, -0.5, 1.0, 8).pressure
        );

        let damped = try_damped_wave_1d(1.5, 0.2, 0.0, 1.0, 8).unwrap();
        assert_eq!(
            damped.pressure,
            damped_wave_1d(1.5, 0.2, 0.0, 1.0, 8).pressure
        );

        let mode = try_helmholtz_1d_mode(1.5, 2.0, 1, 8).unwrap();
        assert_eq!(mode.pressure, helmholtz_1d_mode(1.5, 2.0, 1, 8).pressure);
    }

    #[test]
    fn checked_1d_references_reject_invalid_domains_before_evaluation() {
        assert!(matches!(
            try_plane_wave_1d(0.0, 0.0, 1.0, 8),
            Err(OneDimensionalWaveError::InvalidParameter {
                field: "wave_number",
                ..
            })
        ));
        assert!(matches!(
            try_standing_wave_1d(1.0, 1.0, 0.0, 8),
            Err(OneDimensionalWaveError::InvalidInterval { .. })
        ));
        assert!(matches!(
            try_damped_wave_1d(1.0, f64::NAN, 0.0, 1.0, 8),
            Err(OneDimensionalWaveError::InvalidParameter {
                field: "absorption",
                ..
            })
        ));
        assert!(matches!(
            try_helmholtz_1d_mode(1.0, PI, 1, 8),
            Err(OneDimensionalWaveError::Resonance { .. })
        ));
    }

    #[test]
    fn checked_damped_wave_rejects_exponential_overflow() {
        assert!(matches!(
            try_damped_wave_1d(1.0, 1.0, -1e3, -999.0, 8),
            Err(OneDimensionalWaveError::AmplitudeOverflow)
        ));
    }

    #[test]
    fn rigid_duct_mode_has_closed_end_velocity_nodes_and_expected_frequency() {
        let length = 0.8;
        let mode_number = 3;
        let solution =
            try_rigid_duct_1d_mode(length, mode_number, 801).expect("valid rigid-duct mode");
        let expected_k = mode_number as f64 * PI / length;
        let expected_f = mode_number as f64 * SPEED_OF_SOUND / (2.0 * length);
        assert_abs_diff_eq!(solution.wave_number, expected_k, epsilon = 1e-13);
        assert_abs_diff_eq!(solution.frequency, expected_f, epsilon = 1e-12);
        assert_abs_diff_eq!(solution.pressure[0].re, 1.0, epsilon = 1e-13);
        assert_abs_diff_eq!(
            solution.pressure.last().expect("non-empty grid").re.abs(),
            1.0,
            epsilon = 1e-13
        );

        let dx = solution.positions[1].x - solution.positions[0].x;
        let left_slope = (solution.pressure[1].re - solution.pressure[0].re) / dx;
        let right_index = solution.pressure.len() - 1;
        let right_slope =
            (solution.pressure[right_index].re - solution.pressure[right_index - 1].re) / dx;
        let fine =
            try_rigid_duct_1d_mode(length, mode_number, 1601).expect("refined rigid-duct mode");
        let fine_dx = fine.positions[1].x - fine.positions[0].x;
        let fine_left_slope = (fine.pressure[1].re - fine.pressure[0].re) / fine_dx;
        let fine_right_index = fine.pressure.len() - 1;
        let fine_right_slope =
            (fine.pressure[fine_right_index].re - fine.pressure[fine_right_index - 1].re) / fine_dx;
        // The one-sided derivative approaches the exact zero rigid-wall trace
        // at first order as its first sample moves toward the endpoint.
        assert_abs_diff_eq!(fine_left_slope / left_slope, 0.5, epsilon = 2e-5);
        assert_abs_diff_eq!(fine_right_slope / right_slope, 0.5, epsilon = 2e-5);
    }

    #[test]
    fn checked_rigid_duct_mode_rejects_invalid_modal_domain() {
        assert!(matches!(
            try_rigid_duct_1d_mode(0.0, 1, 8),
            Err(OneDimensionalWaveError::InvalidParameter {
                field: "length",
                ..
            })
        ));
        assert!(matches!(
            try_rigid_duct_1d_mode(1.0, 0, 8),
            Err(OneDimensionalWaveError::InvalidParameter {
                field: "mode_number",
                ..
            })
        ));
    }
}
