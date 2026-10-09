//! 2D Analytical Solutions
//!
//! Exact solutions for 2D acoustic scattering problems, primarily
//! cylinder scattering using Bessel and Hankel function series.

use super::{AnalyticalSolution, Point, SPEED_OF_SOUND};
use num_complex::Complex64;
use spec_math::Bessel;
use std::{f64::consts::PI, fmt};

/// Invalid input to a checked two-dimensional wave reference.
///
/// The legacy constructors remain available for trusted fixtures. New callers
/// should use the `try_*_2d` variants so invalid geometry, sample grids, and
/// non-finite numerical results cannot silently enter a validation comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum TwoDimensionalWaveError {
    /// A scalar parameter is non-finite or outside the model domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// A Cartesian evaluation grid is empty.
    EmptyCartesianGrid {
        /// Grid coordinate name.
        field: &'static str,
    },
    /// A radial or angular evaluation grid is empty.
    EmptyEvaluationGrid {
        /// Grid coordinate name.
        field: &'static str,
    },
    /// A radial evaluation point lies inside a cylinder's exterior domain.
    InteriorEvaluationPoint {
        /// Index in the supplied radial grid.
        index: usize,
        /// Supplied radius in metres.
        radius: f64,
        /// Cylinder radius in metres.
        cylinder_radius: f64,
    },
    /// A legacy numerical evaluation returned a NaN or infinity.
    NonFiniteResult {
        /// Result component that was non-finite.
        field: &'static str,
    },
}

impl fmt::Display for TwoDimensionalWaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(f, "2D wave parameter {field} must be {requirement}")
            }
            Self::EmptyCartesianGrid { field } | Self::EmptyEvaluationGrid { field } => {
                write!(f, "2D wave evaluation grid {field} must be non-empty")
            }
            Self::InteriorEvaluationPoint {
                index,
                radius,
                cylinder_radius,
            } => write!(
                f,
                "cylinder radius[{index}]={radius} lies inside the exterior domain r >= {cylinder_radius}"
            ),
            Self::NonFiniteResult { field } => {
                write!(f, "2D wave reference produced a non-finite {field}")
            }
        }
    }
}

impl std::error::Error for TwoDimensionalWaveError {}

fn validate_wave_number(wave_number: f64) -> Result<(), TwoDimensionalWaveError> {
    if !wave_number.is_finite() || wave_number <= 0.0 {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "wave_number",
            requirement: "finite and positive",
        });
    }
    if !(wave_number * SPEED_OF_SOUND / (2.0 * PI)).is_finite() {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "wave_number",
            requirement: "small enough to yield a finite frequency",
        });
    }
    Ok(())
}

fn validate_cylinder_parameters(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
) -> Result<(), TwoDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "radius",
            requirement: "finite and positive",
        });
    }
    if num_terms == 0 {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "num_terms",
            requirement: "at least 1",
        });
    }
    if !(wave_number * radius).is_finite() {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "wave_number * radius",
            requirement: "finite",
        });
    }
    Ok(())
}

fn validate_cylinder_grid(
    wave_number: f64,
    radius: f64,
    r_points: &[f64],
    theta_points: &[f64],
) -> Result<(), TwoDimensionalWaveError> {
    if r_points.is_empty() {
        return Err(TwoDimensionalWaveError::EmptyEvaluationGrid { field: "r_points" });
    }
    if theta_points.is_empty() {
        return Err(TwoDimensionalWaveError::EmptyEvaluationGrid {
            field: "theta_points",
        });
    }
    for (index, &point_radius) in r_points.iter().enumerate() {
        if !point_radius.is_finite() || point_radius < radius {
            return Err(TwoDimensionalWaveError::InteriorEvaluationPoint {
                index,
                radius: point_radius,
                cylinder_radius: radius,
            });
        }
        if !(wave_number * point_radius).is_finite() {
            return Err(TwoDimensionalWaveError::InvalidParameter {
                field: "wave_number * r_points",
                requirement: "finite",
            });
        }
    }
    if theta_points.iter().any(|theta| !theta.is_finite()) {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "theta_points",
            requirement: "finite",
        });
    }
    Ok(())
}

fn validate_solution(solution: &AnalyticalSolution) -> Result<(), TwoDimensionalWaveError> {
    if solution
        .pressure
        .iter()
        .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
    {
        return Err(TwoDimensionalWaveError::NonFiniteResult { field: "pressure" });
    }
    Ok(())
}

/// Recommended Fourier-series truncation for a rigid cylinder at size parameter
/// `ka`.
///
/// The cylindrical coefficients decay super-algebraically once `n > ka`, so
/// retaining `ceil(2*ka) + 10` terms bounds the worst-case series tail below
/// `1e-9` relative error (measured `<= 1e-13`) on the reference grid
/// (`r/a` in {1.5, 2, 3}, all angles) for `ka <= 20` (see
/// `tests/series_convergence.rs`). This is the documented `2*ka + 10` rule as
/// a single function so callers cannot silently under-resolve a larger `ka`.
///
/// Non-finite or non-positive `ka` saturates to the 10-term floor; physics
/// input validation stays in [`try_cylinder_scattering_2d`], which rejects
/// such inputs before any series evaluation.
pub fn suggested_cylinder_terms(ka: f64) -> usize {
    if ka.is_finite() && ka > 0.0 {
        ((2.0 * ka).ceil() as usize).saturating_add(10)
    } else {
        10
    }
}

/// Cylinder scattering: rigid circular cylinder in a plane wave
///
/// Analytical solution using Bessel/Hankel function series:
///
/// ```text
/// p(r,θ) = exp(ikr cos θ) + Σ εₙ aₙ Hₙ⁽¹⁾(kr) cos(nθ)
/// ```
///
/// where Hₙ⁽¹⁾ are Hankel functions of the first kind.
///
/// For a **rigid cylinder** (Neumann boundary condition: ∂p/∂n = 0):
/// ```text
/// aₙ = -iⁿ Jₙ'(ka) / Hₙ⁽¹⁾'(ka)
/// ```
/// where εₙ = 1 for n=0, εₙ = 2 for n>0 (Neumann factor).
///
/// This is a two-dimensional, homogeneous, stationary, lossless-fluid
/// exterior problem with a unit peak incident-pressure phasor. The crate uses
/// `exp(+iωt)`, so the written `exp(+ikx)` incident phasor travels toward
/// negative x. The truncated series is a reference only when `num_terms` has
/// been independently converged for the requested `ka` and field grid.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Cylinder radius a
/// * `num_terms` - Number of terms in series (typically 2*ka + 10; see `suggested_cylinder_terms`)
/// * `r_points` - Radial evaluation points (must be >= radius)
/// * `theta_points` - Angular evaluation points
///
/// # References
///
/// - Bowman, Senior, Uslenghi, "Electromagnetic and Acoustic Scattering
///   by Simple Shapes", 1987, Section 5.3
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::cylinder_scattering_2d;
/// use std::f64::consts::PI;
///
/// // ka = 1 (low frequency)
/// let solution = cylinder_scattering_2d(
///     1.0, 1.0, 20,
///     vec![1.5, 2.0, 3.0],
///     vec![0.0, PI/4.0, PI/2.0, PI]
/// );
/// ```
pub fn cylinder_scattering_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;

    // Compute scattering coefficients aₙ for rigid cylinder
    let coefficients = compute_rigid_cylinder_coefficients(ka, num_terms);

    // Generate grid
    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &r in &r_points {
        let kr = wave_number * r;

        // Precompute Hankel functions at this radius
        let hankels: Vec<Complex64> = (0..num_terms)
            .map(|n| {
                let n_i64 = n as i64;
                Complex64::new(bessel_j(n_i64, kr), bessel_y(n_i64, kr))
            })
            .collect();

        for &theta in &theta_points {
            positions.push(Point::from_polar(r, theta));

            // Incident wave: exp(ikr cos θ)
            let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());

            // Scattered wave: uses cosine expansion (exploiting symmetry)
            let mut scattered = Complex64::new(0.0, 0.0);

            for n in 0..num_terms {
                let epsilon_n = if n == 0 { 1.0 } else { 2.0 };
                let cos_term = (n as f64 * theta).cos();
                let contribution = coefficients[n] * hankels[n];
                scattered += contribution * (epsilon_n * cos_term);
            }

            pressure.push(incident + scattered);
        }
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("2D Cylinder Scattering (ka={:.2})", ka),
        dimensions: 2,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "radius": radius,
            "ka": ka,
            "num_terms": num_terms,
            "boundary_condition": "rigid",
            "r_points": r_points,
            "theta_range": [theta_points.first(), theta_points.last()],
            "regime": classify_regime_2d(ka),
        }),
    }
}

/// Checked exterior rigid-cylinder scattering reference.
///
/// This validates the positive physical parameters, at least one retained
/// series term, non-empty finite radial/angular grids, and the exterior-domain
/// restriction `r >= radius`. It also rejects non-finite evaluated pressure
/// values, which can indicate that the caller selected an unresolved numerical
/// regime for the underlying special-function implementation.
pub fn try_cylinder_scattering_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> Result<AnalyticalSolution, TwoDimensionalWaveError> {
    validate_cylinder_parameters(wave_number, radius, num_terms)?;
    validate_cylinder_grid(wave_number, radius, &r_points, &theta_points)?;
    let solution = cylinder_scattering_2d(wave_number, radius, num_terms, r_points, theta_points);
    validate_solution(&solution)?;
    Ok(solution)
}

/// Classify scattering regime based on ka
fn classify_regime_2d(ka: f64) -> &'static str {
    if ka < 0.3 {
        "Rayleigh (ka << 1)"
    } else if ka < 3.0 {
        "Resonance (ka ~ 1)"
    } else {
        "Geometric (ka >> 1)"
    }
}

/// Compute scattering coefficients for rigid cylinder
///
/// For Neumann BC (∂p/∂n = 0 on surface):
/// ```text
/// aₙ = -iⁿ Jₙ'(ka) / Hₙ⁽¹⁾'(ka)
/// ```
fn compute_rigid_cylinder_coefficients(ka: f64, num_terms: usize) -> Vec<Complex64> {
    let mut coefficients = Vec::with_capacity(num_terms);

    for n in 0..num_terms {
        let n_i64 = n as i64;
        let n_f64 = n as f64;

        // Derivatives using recurrence: J_n'(x) = J_{n-1}(x) - n/x * J_n(x)
        let jn = bessel_j(n_i64, ka);
        let jn_minus_1 = if n > 0 {
            bessel_j(n_i64 - 1, ka)
        } else {
            -bessel_j(1, ka) // J_{-1} = -J_1
        };
        let jn_prime = jn_minus_1 - n_f64 / ka * jn;

        // Same for Hankel: H_n'(x) = J_n'(x) + i*Y_n'(x)
        let yn = bessel_y(n_i64, ka);
        let yn_minus_1 = if n > 0 {
            bessel_y(n_i64 - 1, ka)
        } else {
            -bessel_y(1, ka)
        };
        let yn_prime = yn_minus_1 - n_f64 / ka * yn;

        let hankel_prime = Complex64::new(jn_prime, yn_prime);

        // i^n = exp(i*n*π/2)
        let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

        let a_n = -jn_prime / hankel_prime * i_power_n;

        coefficients.push(a_n);
    }

    coefficients
}

/// Cylindrical Bessel function of the first kind, order `n`
fn bessel_j(n: i64, x: f64) -> f64 {
    x.bessel_jv(n as f64)
}

/// Cylindrical Bessel function of the second kind (Neumann), order `n`
fn bessel_y(n: i64, x: f64) -> f64 {
    x.bessel_yv(n as f64)
}

/// Pressure directivity pattern (far-field)
///
/// In the far-field (kr >> 1), the scattered pressure becomes:
/// ```text
/// p_scattered ~ exp(ikr) / √(2πkr) * f(θ)
/// ```
///
/// where f(θ) is the scattering amplitude (directivity).
///
/// This is the dimensionless series amplitude used by this module, not a
/// calibrated sound-pressure level or power-normalized directivity. The
/// far-field asymptotic prefactor above must be applied before comparing it to
/// a physical pressure measurement.
pub fn cylinder_directivity_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    theta_points: Vec<f64>,
) -> Vec<Complex64> {
    let ka = wave_number * radius;
    let coefficients = compute_rigid_cylinder_coefficients(ka, num_terms);

    theta_points
        .iter()
        .map(|&theta| {
            let mut directivity = Complex64::new(0.0, 0.0);

            for (n, coeff) in coefficients.iter().enumerate().take(num_terms) {
                let epsilon_n = if n == 0 { 1.0 } else { 2.0 };
                let cos_term = (n as f64 * theta).cos();
                directivity += coeff * (epsilon_n * cos_term);
            }

            directivity
        })
        .collect()
}

/// Checked [`cylinder_directivity_2d`] amplitude series.
pub fn try_cylinder_directivity_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    theta_points: Vec<f64>,
) -> Result<Vec<Complex64>, TwoDimensionalWaveError> {
    validate_cylinder_parameters(wave_number, radius, num_terms)?;
    if theta_points.is_empty() {
        return Err(TwoDimensionalWaveError::EmptyEvaluationGrid {
            field: "theta_points",
        });
    }
    if theta_points.iter().any(|theta| !theta.is_finite()) {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "theta_points",
            requirement: "finite",
        });
    }
    let directivity = cylinder_directivity_2d(wave_number, radius, num_terms, theta_points);
    if directivity
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TwoDimensionalWaveError::NonFiniteResult {
            field: "directivity",
        });
    }
    Ok(directivity)
}

/// Total scattering cross-section (2D)
///
/// The scattering cross-section measures the total scattered power
/// normalized by the incident intensity.
///
/// ```text
/// σ = (4/k) * [|a₀|² + 2 * Σₙ₌₁ |aₙ|²]
/// ```
pub fn cylinder_scattering_cross_section_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
) -> f64 {
    let ka = wave_number * radius;
    let coefficients = compute_rigid_cylinder_coefficients(ka, num_terms);

    let mut sum_sq = 0.0;
    for (n, a_n) in coefficients.iter().enumerate() {
        let epsilon_n = if n == 0 { 1.0 } else { 2.0 };
        sum_sq += epsilon_n * a_n.norm_sqr();
    }

    4.0 / wave_number * sum_sq
}

/// Checked [`cylinder_scattering_cross_section_2d`] exterior reference.
pub fn try_cylinder_scattering_cross_section_2d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
) -> Result<f64, TwoDimensionalWaveError> {
    validate_cylinder_parameters(wave_number, radius, num_terms)?;
    let cross_section = cylinder_scattering_cross_section_2d(wave_number, radius, num_terms);
    if !cross_section.is_finite() || cross_section < 0.0 {
        return Err(TwoDimensionalWaveError::NonFiniteResult {
            field: "cross_section",
        });
    }
    Ok(cross_section)
}

/// 2D plane wave: p(x,y) = exp(ik(x cos θ + y sin θ))
///
/// With the crate's `exp(+iωt)` convention, this `exp(+ik·r)` phasor
/// propagates opposite to the supplied direction vector. It assumes a
/// homogeneous, stationary, lossless fluid and unit peak pressure amplitude.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `direction` - θ, propagation direction (radians)
/// * `x_points` - x-coordinates
/// * `y_points` - y-coordinates
pub fn plane_wave_2d(
    wave_number: f64,
    direction: f64,
    x_points: Vec<f64>,
    y_points: Vec<f64>,
) -> AnalyticalSolution {
    let cos_theta = direction.cos();
    let sin_theta = direction.sin();

    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &x in &x_points {
        for &y in &y_points {
            positions.push(Point::new_2d(x, y));

            let phase = wave_number * (x * cos_theta + y * sin_theta);
            pressure.push(Complex64::new(phase.cos(), phase.sin()));
        }
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("2D Plane Wave (k={}, θ={:.2})", wave_number, direction),
        dimensions: 2,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "direction": direction,
            "direction_vector": [cos_theta, sin_theta],
            "wavelength": 2.0 * PI / wave_number,
        }),
    }
}

/// Checked [`plane_wave_2d`] reference for finite, non-empty Cartesian grids.
pub fn try_plane_wave_2d(
    wave_number: f64,
    direction: f64,
    x_points: Vec<f64>,
    y_points: Vec<f64>,
) -> Result<AnalyticalSolution, TwoDimensionalWaveError> {
    validate_wave_number(wave_number)?;
    if !direction.is_finite() {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "direction",
            requirement: "finite",
        });
    }
    for (field, points) in [("x_points", &x_points), ("y_points", &y_points)] {
        if points.is_empty() {
            return Err(TwoDimensionalWaveError::EmptyCartesianGrid { field });
        }
        if points.iter().any(|point| !point.is_finite()) {
            return Err(TwoDimensionalWaveError::InvalidParameter {
                field,
                requirement: "finite",
            });
        }
    }
    let max_distance = x_points
        .iter()
        .map(|point| point.abs())
        .fold(0.0_f64, f64::max)
        + y_points
            .iter()
            .map(|point| point.abs())
            .fold(0.0_f64, f64::max);
    if !(wave_number * max_distance).is_finite() {
        return Err(TwoDimensionalWaveError::InvalidParameter {
            field: "wave_number * coordinate",
            requirement: "finite",
        });
    }
    let solution = plane_wave_2d(wave_number, direction, x_points, y_points);
    validate_solution(&solution)?;
    Ok(solution)
}

/// Checked analytic plane-wave transmission through a fluid circular
/// inclusion in two dimensions.
///
/// The exterior and interior fields use the cylindrical Fourier expansion
/// with pressure and normal-velocity continuity at `r = radius`. This is an
/// independent reference for reduced-dimensional penetrable BEM paths; it
/// is not a substitute for a boundary-integral solve.
#[derive(Debug, Clone)]
pub struct FluidCylinderTransmission2d {
    radius: f64,
    exterior_wave_number: f64,
    interior_wave_number: f64,
    exterior_density: f64,
    interior_density: f64,
    exterior_coefficients: Vec<Complex64>,
    interior_coefficients: Vec<Complex64>,
}

impl FluidCylinderTransmission2d {
    /// Construct a finite cylindrical transmission reference.
    pub fn try_new(
        radius: f64,
        exterior_wave_number: f64,
        interior_wave_number: f64,
        exterior_density: f64,
        interior_density: f64,
        num_terms: usize,
    ) -> Result<Self, TwoDimensionalWaveError> {
        if !radius.is_finite()
            || radius <= 0.0
            || !exterior_wave_number.is_finite()
            || exterior_wave_number <= 0.0
            || !interior_wave_number.is_finite()
            || interior_wave_number <= 0.0
            || !exterior_density.is_finite()
            || exterior_density <= 0.0
            || !interior_density.is_finite()
            || interior_density <= 0.0
        {
            return Err(TwoDimensionalWaveError::InvalidParameter {
                field: "cylinder transmission parameters",
                requirement: "finite positive radius, wave numbers, and densities",
            });
        }
        if num_terms == 0 {
            return Err(TwoDimensionalWaveError::InvalidParameter {
                field: "num_terms",
                requirement: "at least one Fourier term",
            });
        }

        let ka_ext = exterior_wave_number * radius;
        let ka_int = interior_wave_number * radius;
        let mut exterior_coefficients = Vec::with_capacity(num_terms);
        let mut interior_coefficients = Vec::with_capacity(num_terms);
        for n in 0..num_terms {
            let n_i64 = n as i64;
            let n_f64 = n as f64;
            let i_power = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());
            let je = bessel_j(n_i64, ka_ext);
            let ji = bessel_j(n_i64, ka_int);
            let ye = bessel_y(n_i64, ka_ext);
            let he = Complex64::new(je, ye);
            let d_je = bessel_j_derivative(n_i64, ka_ext);
            let d_ji = bessel_j_derivative(n_i64, ka_int);
            let d_ye = bessel_y_derivative(n_i64, ka_ext);
            let d_he = Complex64::new(d_je, d_ye);

            let ext_pressure_derivative = exterior_wave_number * d_je / exterior_density;
            let ext_hankel_derivative = exterior_wave_number * d_he / exterior_density;
            let int_pressure_derivative = interior_wave_number * d_ji / interior_density;
            let denominator = ext_hankel_derivative * ji - he * int_pressure_derivative;
            if !denominator.re.is_finite()
                || !denominator.im.is_finite()
                || denominator.norm() <= 1.0e-14
            {
                return Err(TwoDimensionalWaveError::InvalidParameter {
                    field: "cylinder transmission coefficients",
                    requirement: "non-singular truncated Fourier system",
                });
            }

            // A_n H_n is the exterior scattered coefficient and B_n J_n is
            // the interior coefficient for the incident i^n J_n expansion.
            let scattered =
                (je * int_pressure_derivative - ext_pressure_derivative * ji) / denominator;
            let interior = (je + scattered * he) / ji;
            let scattered = i_power * scattered;
            let interior = i_power * interior;
            if !scattered.re.is_finite()
                || !scattered.im.is_finite()
                || !interior.re.is_finite()
                || !interior.im.is_finite()
            {
                return Err(TwoDimensionalWaveError::NonFiniteResult {
                    field: "cylinder transmission coefficients",
                });
            }
            exterior_coefficients.push(scattered);
            interior_coefficients.push(interior);
        }

        Ok(Self {
            radius,
            exterior_wave_number,
            interior_wave_number,
            exterior_density,
            interior_density,
            exterior_coefficients,
            interior_coefficients,
        })
    }

    /// Evaluate the total exterior pressure at a point with `r >= radius`.
    pub fn exterior_pressure(&self, radius: f64, theta: f64) -> Complex64 {
        let kr = self.exterior_wave_number * radius;
        let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());
        let mut scattered = Complex64::default();
        for (n, coefficient) in self.exterior_coefficients.iter().enumerate() {
            let hankel = Complex64::new(bessel_j(n as i64, kr), bessel_y(n as i64, kr));
            let epsilon = if n == 0 { 1.0 } else { 2.0 };
            scattered += *coefficient * hankel * (epsilon * (n as f64 * theta).cos());
        }
        incident + scattered
    }

    /// Evaluate the total interior pressure at a point with `r <= radius`.
    pub fn interior_pressure(&self, radius: f64, theta: f64) -> Complex64 {
        let kr = self.interior_wave_number * radius;
        self.interior_coefficients
            .iter()
            .enumerate()
            .map(|(n, coefficient)| {
                let epsilon = if n == 0 { 1.0 } else { 2.0 };
                *coefficient * bessel_j(n as i64, kr) * (epsilon * (n as f64 * theta).cos())
            })
            .sum()
    }

    /// Maximum pressure continuity error sampled around the interface.
    pub fn pressure_continuity_error(&self, samples: usize) -> f64 {
        (0..samples.max(1))
            .map(|index| {
                let theta = 2.0 * PI * index as f64 / samples.max(1) as f64;
                (self.exterior_pressure(self.radius, theta)
                    - self.interior_pressure(self.radius, theta))
                .norm()
            })
            .fold(0.0, f64::max)
    }

    /// Maximum density-weighted normal-velocity continuity error sampled at
    /// the interface, evaluated by a centered finite difference.
    pub fn velocity_continuity_error(&self, samples: usize) -> f64 {
        let h = self.radius * 1.0e-5;
        (0..samples.max(1))
            .map(|index| {
                let theta = 2.0 * PI * index as f64 / samples.max(1) as f64;
                let exterior = (self.exterior_pressure(self.radius + h, theta)
                    - self.exterior_pressure(self.radius - h, theta))
                    / (2.0 * h * self.exterior_density);
                let interior = (self.interior_pressure(self.radius + h, theta)
                    - self.interior_pressure(self.radius - h, theta))
                    / (2.0 * h * self.interior_density);
                (exterior - interior).norm()
            })
            .fold(0.0, f64::max)
    }
}

fn bessel_j_derivative(n: i64, x: f64) -> f64 {
    let previous = if n > 0 {
        bessel_j(n - 1, x)
    } else {
        -bessel_j(1, x)
    };
    previous - n as f64 / x * bessel_j(n, x)
}

fn bessel_y_derivative(n: i64, x: f64) -> f64 {
    let previous = if n > 0 {
        bessel_y(n - 1, x)
    } else {
        -bessel_y(1, x)
    };
    previous - n as f64 / x * bessel_y(n, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cylinder_low_frequency() {
        // Low frequency: ka = 0.1 (Rayleigh scattering)
        let k = 0.1;
        let a = 1.0;

        let solution = cylinder_scattering_2d(k, a, 10, vec![2.0], vec![0.0, PI / 2.0, PI]);

        // At low frequency, scattering should be weak
        // Total field ≈ incident field
        for p in &solution.pressure {
            assert!(p.norm() > 0.5); // Not completely scattered
            assert!(p.norm() < 2.0); // Not strongly scattered
        }
    }

    #[test]
    fn test_cylinder_boundary_condition() {
        // On surface of rigid cylinder, solution should be well-defined
        let k = 2.0;
        let a = 1.0;

        let solution = cylinder_scattering_2d(
            k,
            a,
            30,
            vec![a], // On surface
            (0..36).map(|i| i as f64 * 2.0 * PI / 36.0).collect(),
        );

        // All pressures should be finite
        for p in &solution.pressure {
            assert!(p.norm().is_finite());
        }
    }

    #[test]
    fn test_symmetry() {
        // Plane wave at θ=0 → solution should be symmetric about x-axis
        let k = 1.0;
        let a = 1.0;

        let solution = cylinder_scattering_2d(k, a, 20, vec![2.0], vec![PI / 4.0, -PI / 4.0]);

        // Magnitudes should be equal due to symmetry
        assert_abs_diff_eq!(
            solution.pressure[0].norm(),
            solution.pressure[1].norm(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_bessel_functions() {
        // Verify Bessel function implementation
        let x = 1.0;

        // J_0(1) ≈ 0.7651976865579666
        assert_abs_diff_eq!(bessel_j(0, x), 0.7651976865579666, epsilon = 1e-10);

        // J_1(1) ≈ 0.4400505857449335
        assert_abs_diff_eq!(bessel_j(1, x), 0.4400505857449335, epsilon = 1e-10);
    }

    #[test]
    fn test_scattering_cross_section() {
        // Scattering cross-section should be positive
        let k = 1.0;
        let a = 1.0;

        let sigma = cylinder_scattering_cross_section_2d(k, a, 30);
        assert!(sigma > 0.0);
        assert!(sigma.is_finite());
    }

    #[test]
    fn test_plane_wave_2d() {
        let k = 1.0;
        let direction = 0.0; // Along x-axis

        let solution = plane_wave_2d(k, direction, vec![0.0, 1.0, 2.0], vec![0.0]);

        // At (0,0): exp(0) = 1
        assert_abs_diff_eq!(solution.pressure[0].re, 1.0, epsilon = 1e-10);

        // |exp(ikx)| = 1 for all points
        for p in &solution.pressure {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn checked_2d_references_match_legacy_values() {
        let checked =
            try_cylinder_scattering_2d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0]).unwrap();
        let legacy = cylinder_scattering_2d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0]);
        assert_eq!(checked.pressure, legacy.pressure);

        assert_eq!(
            try_cylinder_directivity_2d(1.0, 0.5, 20, vec![0.0, PI / 3.0]).unwrap(),
            cylinder_directivity_2d(1.0, 0.5, 20, vec![0.0, PI / 3.0])
        );
        assert_eq!(
            try_cylinder_scattering_cross_section_2d(1.0, 0.5, 20).unwrap(),
            cylinder_scattering_cross_section_2d(1.0, 0.5, 20)
        );

        let checked_plane =
            try_plane_wave_2d(1.0, PI / 4.0, vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        assert_eq!(
            checked_plane.pressure,
            plane_wave_2d(1.0, PI / 4.0, vec![0.0, 1.0], vec![0.0, 1.0]).pressure
        );
    }

    #[test]
    fn checked_2d_references_reject_invalid_domains() {
        assert!(matches!(
            try_cylinder_scattering_2d(0.0, 1.0, 20, vec![1.0], vec![0.0]),
            Err(TwoDimensionalWaveError::InvalidParameter {
                field: "wave_number",
                ..
            })
        ));
        assert!(matches!(
            try_cylinder_scattering_2d(1.0, 1.0, 20, vec![0.9], vec![0.0]),
            Err(TwoDimensionalWaveError::InteriorEvaluationPoint { index: 0, .. })
        ));
        assert!(matches!(
            try_cylinder_directivity_2d(1.0, 1.0, 0, vec![0.0]),
            Err(TwoDimensionalWaveError::InvalidParameter {
                field: "num_terms",
                ..
            })
        ));
        assert!(matches!(
            try_plane_wave_2d(1.0, 0.0, vec![], vec![0.0]),
            Err(TwoDimensionalWaveError::EmptyCartesianGrid { field: "x_points" })
        ));
        assert!(matches!(
            try_plane_wave_2d(1.0, f64::NAN, vec![0.0], vec![0.0]),
            Err(TwoDimensionalWaveError::InvalidParameter {
                field: "direction",
                ..
            })
        ));
    }

    #[test]
    fn rigid_cylinder_series_satisfies_surface_neumann_condition() {
        let wave_number = 1.2;
        let radius = 0.7;
        let radial_step = 1e-5;
        let theta = 0.41;
        let solution = try_cylinder_scattering_2d(
            wave_number,
            radius,
            40,
            vec![radius, radius + radial_step, radius + 2.0 * radial_step],
            vec![theta],
        )
        .unwrap();

        // The three-point forward difference is O(h²) and evaluates solely
        // exterior points. The rigid boundary requires ∂p/∂r = 0 at r = a.
        let normal_derivative = (-3.0 * solution.pressure[0] + 4.0 * solution.pressure[1]
            - solution.pressure[2])
            / (2.0 * radial_step);
        assert!(
            normal_derivative.norm() < 1e-7,
            "surface normal derivative magnitude = {:.3e}",
            normal_derivative.norm()
        );
    }
}
