//! 3D Analytical Solutions
//!
//! Exact solutions for 3D acoustic scattering problems, primarily
//! sphere scattering using Mie theory (spherical Bessel/Hankel functions).

use super::{AnalyticalSolution, Point, SPEED_OF_SOUND};
use num_complex::Complex64;
use std::{f64::consts::PI, fmt};

/// Invalid input to the checked pulsating-sphere exterior reference.
#[derive(Debug, Clone, PartialEq)]
pub enum PulsatingSphereError {
    /// A scalar parameter is non-finite or violates the stated physical domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// An evaluation radius is outside the exterior acoustic domain.
    InteriorEvaluationPoint {
        /// Index in the supplied radial evaluation grid.
        index: usize,
        /// Supplied radius in metres.
        radius: f64,
        /// Pulsating-sphere surface radius in metres.
        sphere_radius: f64,
    },
}

impl fmt::Display for PulsatingSphereError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "pulsating-sphere parameter {field} must be {requirement}"
                )
            }
            Self::InteriorEvaluationPoint {
                index,
                radius,
                sphere_radius,
            } => write!(
                f,
                "pulsating-sphere radius[{index}]={radius} lies inside the exterior domain r >= {sphere_radius}"
            ),
        }
    }
}

impl std::error::Error for PulsatingSphereError {}

/// Invalid input to a checked exterior sphere-scattering reference.
///
/// The unchecked Mie-series functions remain available for established
/// fixtures. New validation callers should use the `try_*_scattering_3d`
/// variants so invalid geometry, angular domains, and non-finite series values
/// are rejected before they can be mistaken for an analytical reference.
#[derive(Debug, Clone, PartialEq)]
pub enum SphereScatteringError {
    /// A scalar parameter is non-finite or violates the model domain.
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Required domain or invariant.
        requirement: &'static str,
    },
    /// A supplied radial or angular grid is empty.
    EmptyEvaluationGrid {
        /// Grid coordinate name.
        field: &'static str,
    },
    /// A radial evaluation point lies inside an exterior scattering domain.
    InteriorEvaluationPoint {
        /// Index in the supplied radial grid.
        index: usize,
        /// Supplied radius in metres.
        radius: f64,
        /// Sphere radius in metres.
        sphere_radius: f64,
    },
    /// A legacy numerical evaluation returned a NaN or infinity.
    NonFiniteResult {
        /// Result component that was non-finite.
        field: &'static str,
    },
}

impl fmt::Display for SphereScatteringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { field, requirement } => {
                write!(
                    f,
                    "sphere-scattering parameter {field} must be {requirement}"
                )
            }
            Self::EmptyEvaluationGrid { field } => {
                write!(
                    f,
                    "sphere-scattering evaluation grid {field} must be non-empty"
                )
            }
            Self::InteriorEvaluationPoint {
                index,
                radius,
                sphere_radius,
            } => write!(
                f,
                "sphere radius[{index}]={radius} lies inside the exterior domain r >= {sphere_radius}"
            ),
            Self::NonFiniteResult { field } => {
                write!(
                    f,
                    "sphere-scattering reference produced a non-finite {field}"
                )
            }
        }
    }
}

impl std::error::Error for SphereScatteringError {}

fn validate_sphere_scattering_parameters(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
) -> Result<(), SphereScatteringError> {
    if !wave_number.is_finite() || wave_number <= 0.0 {
        return Err(SphereScatteringError::InvalidParameter {
            field: "wave_number",
            requirement: "finite and positive",
        });
    }
    if !(wave_number * SPEED_OF_SOUND / (2.0 * PI)).is_finite() {
        return Err(SphereScatteringError::InvalidParameter {
            field: "wave_number",
            requirement: "small enough to yield a finite frequency",
        });
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err(SphereScatteringError::InvalidParameter {
            field: "radius",
            requirement: "finite and positive",
        });
    }
    if num_terms == 0 {
        return Err(SphereScatteringError::InvalidParameter {
            field: "num_terms",
            requirement: "at least 1",
        });
    }
    if !(wave_number * radius).is_finite() {
        return Err(SphereScatteringError::InvalidParameter {
            field: "wave_number * radius",
            requirement: "finite",
        });
    }
    Ok(())
}

fn validate_sphere_scattering_grid(
    wave_number: f64,
    radius: f64,
    r_points: &[f64],
    theta_points: &[f64],
) -> Result<(), SphereScatteringError> {
    if r_points.is_empty() {
        return Err(SphereScatteringError::EmptyEvaluationGrid { field: "r_points" });
    }
    if theta_points.is_empty() {
        return Err(SphereScatteringError::EmptyEvaluationGrid {
            field: "theta_points",
        });
    }
    for (index, &point_radius) in r_points.iter().enumerate() {
        if !point_radius.is_finite() || point_radius < radius {
            return Err(SphereScatteringError::InteriorEvaluationPoint {
                index,
                radius: point_radius,
                sphere_radius: radius,
            });
        }
        if !(wave_number * point_radius).is_finite() {
            return Err(SphereScatteringError::InvalidParameter {
                field: "wave_number * r_points",
                requirement: "finite",
            });
        }
    }
    if theta_points
        .iter()
        .any(|theta| !theta.is_finite() || !(0.0..=PI).contains(theta))
    {
        return Err(SphereScatteringError::InvalidParameter {
            field: "theta_points",
            requirement: "finite values in [0, π]",
        });
    }
    Ok(())
}

fn validate_sphere_scattering_solution(
    solution: &AnalyticalSolution,
) -> Result<(), SphereScatteringError> {
    if solution
        .pressure
        .iter()
        .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
    {
        return Err(SphereScatteringError::NonFiniteResult { field: "pressure" });
    }
    Ok(())
}

/// Sphere scattering: rigid sphere in a plane wave (Mie theory)
///
/// Analytical solution using spherical harmonics expansion:
///
/// ```text
/// p(r,θ) = Σₙ (2n+1) iⁿ [jₙ(kr) - aₙ hₙ⁽¹⁾(kr)] Pₙ(cos θ)
/// ```
///
/// Recommended Mie-series truncation for a rigid sphere at size parameter `ka`.
///
/// The forward-scattering lobe (`theta = 0`, where every partial wave adds in
/// phase) converges far more slowly than the documented `ka + 10` heuristic:
/// at `ka = 5` that rule leaves an O(1) relative tail, and at `ka = 20` even
/// 63 terms leave a 0.6 relative error (see `tests/series_convergence.rs`).
/// Retaining `4*ceil(ka) + 16` terms bounds the worst-case tail below `1e-9`
/// relative error on the reference grid (`r/a` in {1.5, 2, 3}, all polar
/// angles including forward scattering) for `ka <= 20`.
///
/// This is an empirical rule with a stated validity domain, not an
/// asymptotic bound: callers with larger `ka`, tighter near-field grids, or
/// stricter tolerances must run their own successive-refinement check.
/// Non-finite or non-positive `ka` saturates to the 16-term floor; physics
/// input validation stays in [`try_sphere_scattering_3d`], which rejects such
/// inputs before any series evaluation.
pub fn suggested_sphere_terms(ka: f64) -> usize {
    if ka.is_finite() && ka > 0.0 {
        (ka.ceil() as usize)
            .saturating_mul(4)
            .saturating_add(16)
            .max(17)
    } else {
        16
    }
}

/// Suggested Mie truncation accounting for observer distance.
///
/// The partial-wave series needs `n_max` beyond `k·r` of the farthest
/// evaluation point (terms with `n > kr` are evanescent there), so
/// [`suggested_sphere_terms`] alone undercounts for far observers:
/// `max(suggested(ka), ceil(kr_max) + 20)`. Callers pushing `n` past
/// ~120 at small `ka` should watch for upward-recurrence overflow in
/// `y_n(ka)` (non-finite sums); `try_` constructors reject such grids.
pub fn suggested_sphere_terms_for_grid(ka: f64, kr_max: f64) -> usize {
    let distance_terms = if kr_max.is_finite() && kr_max > 0.0 {
        (kr_max.ceil() as usize).saturating_add(20)
    } else {
        0
    };
    suggested_sphere_terms(ka).max(distance_terms)
}

/// For a **rigid sphere** (Neumann BC: ∂p/∂r = 0 on surface):
/// ```text
/// aₙ = jₙ'(ka) / hₙ⁽¹⁾'(ka)
/// ```
///
/// where:
/// - `jₙ` = spherical Bessel functions of the first kind
/// - `hₙ⁽¹⁾` = spherical Hankel functions of the first kind
/// - `Pₙ` = Legendre polynomials
///
/// This is an axisymmetric exterior-obstacle reference in a homogeneous,
/// stationary, lossless fluid with a unit peak incident-pressure phasor. The
/// crate uses `exp(+iωt)`, so the displayed `exp(+ik·r)` spatial phasor
/// propagates opposite the specified wave vector. The Mie series must be
/// independently converged in `num_terms` for the requested `ka` and field
/// grid.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Sphere radius a
/// * `num_terms` - Number of terms (use `suggested_sphere_terms(ka)`; the old
///   `ka + 10` heuristic is unconverged for `ka >= 2` at forward angles)
/// * `r_points` - Radial evaluation points (must be >= radius)
/// * `theta_points` - Polar angle points (from z-axis, 0 to π)
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 8.3
/// - Bowman et al., "Electromagnetic and Acoustic Scattering", 1987
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::sphere_scattering_3d;
/// use std::f64::consts::PI;
///
/// // ka = 1 (Mie regime)
/// let solution = sphere_scattering_3d(
///     1.0,  // wave_number
///     1.0,  // radius
///     20,   // num_terms
///     vec![1.5, 2.0, 3.0],  // r
///     vec![0.0, PI/4.0, PI/2.0, PI],  // theta
/// );
/// ```
pub fn sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;

    // Compute scattering coefficients for rigid sphere
    let coefficients = compute_rigid_sphere_coefficients(ka, num_terms);

    // Generate grid (axisymmetric, so φ = 0)
    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &r in &r_points {
        for &theta in &theta_points {
            positions.push(Point::from_spherical(r, theta, 0.0));

            let kr = wave_number * r;
            let cos_theta = theta.cos();

            // Total field: Σₙ (2n+1) iⁿ [jₙ(kr) - aₙ hₙ⁽¹⁾(kr)] Pₙ(cos θ)
            let mut total = Complex64::new(0.0, 0.0);

            for (n, coeff) in coefficients.iter().enumerate().take(num_terms) {
                let n_f64 = n as f64;

                // (2n+1)
                let prefactor = 2.0 * n_f64 + 1.0;

                // iⁿ = exp(i*n*π/2)
                let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

                // Spherical Bessel jₙ(kr)
                let jn = spherical_bessel_j(n, kr);

                // Spherical Hankel hₙ⁽¹⁾(kr) = jₙ(kr) + i*yₙ(kr)
                let yn = spherical_bessel_y(n, kr);
                let hn = Complex64::new(jn, yn);

                // Legendre polynomial Pₙ(cos θ)
                let pn = legendre_p(n, cos_theta);

                // Add term
                total += prefactor * i_power_n * (jn - coeff * hn) * pn;
            }

            pressure.push(total);
        }
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Sphere Scattering (ka={:.2})", ka),
        dimensions: 3,
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
            "regime": classify_regime(ka),
        }),
    }
}

/// Checked exterior rigid-sphere (Neumann) Mie reference.
pub fn try_sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> Result<AnalyticalSolution, SphereScatteringError> {
    validate_sphere_scattering_parameters(wave_number, radius, num_terms)?;
    validate_sphere_scattering_grid(wave_number, radius, &r_points, &theta_points)?;
    let solution = sphere_scattering_3d(wave_number, radius, num_terms, r_points, theta_points);
    validate_sphere_scattering_solution(&solution)?;
    Ok(solution)
}

/// Axisymmetric Mie-series reference for a penetrable fluid sphere.
///
/// The exterior total pressure and interior pressure are expanded as
///
/// ```text
/// p_e = Σ [I_n j_n(k_e r) + A_n h_n^(1)(k_e r)] P_n(cos θ)
/// p_i = Σ B_n j_n(k_i r) P_n(cos θ),
/// I_n = (2n + 1) i^n.
/// ```
///
/// `A_n` and `B_n` are obtained from continuity of pressure and normal
/// particle velocity at `r = radius`. Densities and wave numbers must be
/// real, finite, and positive; the model is therefore the lossless,
/// stationary, linear-fluid reference used by the bounded CPU BEM gate.
#[derive(Debug, Clone)]
pub struct FluidSphereTransmission3d {
    radius: f64,
    exterior_wave_number: f64,
    interior_wave_number: f64,
    exterior_density: f64,
    interior_density: f64,
    exterior_coefficients: Vec<Complex64>,
    interior_coefficients: Vec<Complex64>,
}

impl FluidSphereTransmission3d {
    /// Construct a checked unit-amplitude penetrable-sphere reference.
    pub fn try_new(
        radius: f64,
        exterior_wave_number: f64,
        interior_wave_number: f64,
        exterior_density: f64,
        interior_density: f64,
        num_terms: usize,
    ) -> Result<Self, SphereScatteringError> {
        for (field, value) in [
            ("radius", radius),
            ("exterior_wave_number", exterior_wave_number),
            ("interior_wave_number", interior_wave_number),
            ("exterior_density", exterior_density),
            ("interior_density", interior_density),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SphereScatteringError::InvalidParameter {
                    field,
                    requirement: "finite and positive",
                });
            }
        }
        if num_terms == 0 {
            return Err(SphereScatteringError::InvalidParameter {
                field: "num_terms",
                requirement: "at least one",
            });
        }

        let exterior_argument = exterior_wave_number * radius;
        let interior_argument = interior_wave_number * radius;
        let exterior_velocity_scale = exterior_wave_number / exterior_density;
        let interior_velocity_scale = interior_wave_number / interior_density;
        let mut exterior_coefficients = Vec::with_capacity(num_terms);
        let mut interior_coefficients = Vec::with_capacity(num_terms);

        for order in 0..num_terms {
            let incident = plane_wave_mode_coefficient(order);
            let je = spherical_bessel_j(order, exterior_argument);
            let ji = spherical_bessel_j(order, interior_argument);
            let ye = spherical_bessel_y(order, exterior_argument);
            let je_derivative = spherical_bessel_j_scalar_derivative(order, exterior_argument);
            let ji_derivative = spherical_bessel_j_scalar_derivative(order, interior_argument);
            let ye_derivative = spherical_bessel_y_scalar_derivative(order, exterior_argument);
            let he = Complex64::new(je, ye);
            let he_derivative = Complex64::new(je_derivative, ye_derivative);

            let determinant = exterior_velocity_scale * ji * he_derivative
                - interior_velocity_scale * ji_derivative * he;
            if !determinant.re.is_finite()
                || !determinant.im.is_finite()
                || determinant.norm() <= f64::MIN_POSITIVE
            {
                return Err(SphereScatteringError::NonFiniteResult {
                    field: "fluid-sphere modal determinant",
                });
            }

            let exterior = incident
                * (interior_velocity_scale * je * ji_derivative
                    - exterior_velocity_scale * je_derivative * ji)
                / determinant;
            let interior =
                incident * exterior_velocity_scale * (je * he_derivative - je_derivative * he)
                    / determinant;
            if [exterior, interior]
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(SphereScatteringError::NonFiniteResult {
                    field: "fluid-sphere modal coefficient",
                });
            }
            exterior_coefficients.push(exterior);
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

    /// Sphere radius in metres.
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Number of retained spherical modes.
    pub fn num_terms(&self) -> usize {
        self.exterior_coefficients.len()
    }

    /// Exterior total pressure for a unit plane wave travelling along +z.
    pub fn exterior_pressure(
        &self,
        radius: f64,
        theta: f64,
    ) -> Result<Complex64, SphereScatteringError> {
        validate_fluid_sphere_point(radius, theta, self.radius, true)?;
        let argument = self.exterior_wave_number * radius;
        let cosine = theta.cos();
        Ok(self
            .exterior_coefficients
            .iter()
            .enumerate()
            .map(|(order, &coefficient)| {
                let incident = plane_wave_mode_coefficient(order);
                let regular = spherical_bessel_j(order, argument);
                let outgoing = Complex64::new(regular, spherical_bessel_y(order, argument));
                (incident * regular + coefficient * outgoing) * legendre_p(order, cosine)
            })
            .sum())
    }

    /// Exterior scattered pressure (the outgoing Hankel series only).
    pub fn exterior_scattered_pressure(
        &self,
        radius: f64,
        theta: f64,
    ) -> Result<Complex64, SphereScatteringError> {
        validate_fluid_sphere_point(radius, theta, self.radius, true)?;
        let argument = self.exterior_wave_number * radius;
        let cosine = theta.cos();
        Ok(self
            .exterior_coefficients
            .iter()
            .enumerate()
            .map(|(order, &coefficient)| {
                coefficient
                    * Complex64::new(
                        spherical_bessel_j(order, argument),
                        spherical_bessel_y(order, argument),
                    )
                    * legendre_p(order, cosine)
            })
            .sum())
    }

    /// Interior transmitted pressure.
    pub fn interior_pressure(
        &self,
        radius: f64,
        theta: f64,
    ) -> Result<Complex64, SphereScatteringError> {
        validate_fluid_sphere_point(radius, theta, self.radius, false)?;
        let argument = self.interior_wave_number * radius;
        let cosine = theta.cos();
        Ok(self
            .interior_coefficients
            .iter()
            .enumerate()
            .map(|(order, &coefficient)| {
                coefficient * spherical_bessel_j(order, argument) * legendre_p(order, cosine)
            })
            .sum())
    }

    /// Exact interface pressure trace.
    pub fn interface_pressure(&self, theta: f64) -> Result<Complex64, SphereScatteringError> {
        self.exterior_pressure(self.radius, theta)
    }

    /// Exact outward normal-velocity trace at the interface.
    ///
    /// `angular_frequency` is required because the modal continuity equations
    /// cancel the common `1/(iω)` factor, whereas the BEM Cauchy trace stores
    /// particle velocity in metres per second.
    pub fn interface_normal_velocity(
        &self,
        theta: f64,
        angular_frequency: f64,
    ) -> Result<Complex64, SphereScatteringError> {
        if !angular_frequency.is_finite() || angular_frequency <= 0.0 {
            return Err(SphereScatteringError::InvalidParameter {
                field: "angular_frequency",
                requirement: "finite and positive",
            });
        }
        validate_fluid_sphere_point(self.radius, theta, self.radius, true)?;
        let argument = self.exterior_wave_number * self.radius;
        let cosine = theta.cos();
        let derivative: Complex64 = self
            .exterior_coefficients
            .iter()
            .enumerate()
            .map(|(order, &coefficient)| {
                let regular_derivative = spherical_bessel_j_scalar_derivative(order, argument);
                let outgoing_derivative = Complex64::new(
                    regular_derivative,
                    spherical_bessel_y_scalar_derivative(order, argument),
                );
                (plane_wave_mode_coefficient(order) * regular_derivative
                    + coefficient * outgoing_derivative)
                    * self.exterior_wave_number
                    * legendre_p(order, cosine)
            })
            .sum();
        Ok(derivative / Complex64::new(0.0, angular_frequency * self.exterior_density))
    }

    /// Maximum relative pressure mismatch across the interface.
    pub fn pressure_continuity_error(&self, samples: usize) -> f64 {
        continuity_sample_angles(samples)
            .into_iter()
            .map(|theta| {
                let exterior = self
                    .exterior_pressure(self.radius, theta)
                    .expect("constructor validates exterior surface evaluation");
                let interior = self
                    .interior_pressure(self.radius, theta)
                    .expect("constructor validates interior surface evaluation");
                (exterior - interior).norm()
                    / exterior.norm().max(interior.norm()).max(f64::MIN_POSITIVE)
            })
            .fold(0.0, f64::max)
    }

    /// Maximum relative normal-velocity mismatch across the interface.
    pub fn velocity_continuity_error(&self, samples: usize) -> f64 {
        continuity_sample_angles(samples)
            .into_iter()
            .map(|theta| {
                let cosine = theta.cos();
                let exterior_argument = self.exterior_wave_number * self.radius;
                let interior_argument = self.interior_wave_number * self.radius;
                let mut exterior = Complex64::new(0.0, 0.0);
                let mut interior = Complex64::new(0.0, 0.0);
                for order in 0..self.num_terms() {
                    let incident = plane_wave_mode_coefficient(order);
                    let regular_derivative =
                        spherical_bessel_j_scalar_derivative(order, exterior_argument);
                    let outgoing_derivative = Complex64::new(
                        regular_derivative,
                        spherical_bessel_y_scalar_derivative(order, exterior_argument),
                    );
                    let polynomial = legendre_p(order, cosine);
                    exterior += (incident * regular_derivative
                        + self.exterior_coefficients[order] * outgoing_derivative)
                        * (self.exterior_wave_number / self.exterior_density)
                        * polynomial;
                    interior += self.interior_coefficients[order]
                        * spherical_bessel_j_scalar_derivative(order, interior_argument)
                        * (self.interior_wave_number / self.interior_density)
                        * polynomial;
                }
                (exterior - interior).norm()
                    / exterior.norm().max(interior.norm()).max(f64::MIN_POSITIVE)
            })
            .fold(0.0, f64::max)
    }
}

fn plane_wave_mode_coefficient(order: usize) -> Complex64 {
    let phase = order as f64 * PI / 2.0;
    Complex64::from_polar(2.0 * order as f64 + 1.0, phase)
}

fn spherical_bessel_j_scalar_derivative(order: usize, argument: f64) -> f64 {
    if order == 0 {
        -spherical_bessel_j(1, argument)
    } else {
        spherical_bessel_j(order - 1, argument)
            - (order as f64 + 1.0) / argument * spherical_bessel_j(order, argument)
    }
}

fn spherical_bessel_y_scalar_derivative(order: usize, argument: f64) -> f64 {
    if order == 0 {
        -spherical_bessel_y(1, argument)
    } else {
        spherical_bessel_y(order - 1, argument)
            - (order as f64 + 1.0) / argument * spherical_bessel_y(order, argument)
    }
}

fn validate_fluid_sphere_point(
    radius: f64,
    theta: f64,
    sphere_radius: f64,
    exterior: bool,
) -> Result<(), SphereScatteringError> {
    if !radius.is_finite()
        || radius < 0.0
        || (exterior && radius < sphere_radius)
        || (!exterior && radius > sphere_radius)
    {
        return Err(SphereScatteringError::InvalidParameter {
            field: if exterior {
                "exterior evaluation radius"
            } else {
                "interior evaluation radius"
            },
            requirement: if exterior {
                "finite and at least the sphere radius"
            } else {
                "finite and between zero and the sphere radius"
            },
        });
    }
    if !theta.is_finite() || !(0.0..=PI).contains(&theta) {
        return Err(SphereScatteringError::InvalidParameter {
            field: "theta",
            requirement: "finite and in [0, pi]",
        });
    }
    Ok(())
}

fn continuity_sample_angles(samples: usize) -> Vec<f64> {
    let count = samples.max(2);
    (0..count)
        .map(|index| PI * index as f64 / (count - 1) as f64)
        .collect()
}

/// Classify scattering regime based on ka
pub fn classify_regime(ka: f64) -> &'static str {
    if ka < 0.3 {
        "Rayleigh (ka << 1)"
    } else if ka < 3.0 {
        "Mie (ka ~ 1)"
    } else {
        "Geometric (ka >> 1)"
    }
}

/// Compute scattering coefficients for rigid sphere
///
/// For Neumann BC (∂p/∂r = 0 on surface):
/// ```text
/// aₙ = jₙ'(ka) / hₙ⁽¹⁾'(ka)
/// ```
fn compute_rigid_sphere_coefficients(ka: f64, num_terms: usize) -> Vec<Complex64> {
    let mut coefficients = Vec::with_capacity(num_terms);

    for n in 0..num_terms {
        let n_f64 = n as f64;

        // Spherical Bessel and derivatives
        let jn = spherical_bessel_j(n, ka);
        let yn = spherical_bessel_y(n, ka);

        // Derivatives: jₙ'(x) = jₙ₋₁(x) - (n+1)/x * jₙ(x)
        let jn_minus_1 = if n > 0 {
            spherical_bessel_j(n - 1, ka)
        } else {
            // j₋₁(x) = cos(x)/x
            ka.cos() / ka
        };
        let jn_prime = jn_minus_1 - (n_f64 + 1.0) / ka * jn;

        // Same for yₙ
        let yn_minus_1 = if n > 0 {
            spherical_bessel_y(n - 1, ka)
        } else {
            // y₋₁(x) = sin(x)/x
            ka.sin() / ka
        };
        let yn_prime = yn_minus_1 - (n_f64 + 1.0) / ka * yn;

        // Hankel derivative
        let hn_prime = Complex64::new(jn_prime, yn_prime);

        let a_n = Complex64::new(jn_prime, 0.0) / hn_prime;

        coefficients.push(a_n);
    }

    coefficients
}

/// Spherical Bessel function jₙ(x)
///
/// Relation to cylindrical Bessel: jₙ(x) = √(π/2x) * J_{n+1/2}(x)
///
/// Uses Miller's downward recurrence for numerical stability.
pub fn spherical_bessel_j(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return if n == 0 { 1.0 } else { 0.0 };
    }

    // Taylor series for small x: the direct n = 1 formula below computes
    // sin(x)/x^2 - cos(x)/x, whose terms agree to ~2*log10(1/|x|) digits
    // and cancel catastrophically (at x = 1e-8 the result keeps ~0 digits,
    // corrupting recurrences such as j_n' built on top of it). The truncated
    // series below holds full double precision through the branch cutoff.
    if x.abs() < 1e-4 {
        let x2 = x * x;
        if n == 0 {
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        }
        if n == 1 {
            return x * (1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0);
        }
    }

    // For small x or small n, direct formulas
    match n {
        0 => x.sin() / x,
        1 => x.sin() / (x * x) - x.cos() / x,
        _ => {
            // Miller's downward recurrence for stability when n > x
            let start_n = n + (x.abs() as usize) + 20;

            let mut j_next = 0.0;
            let mut j_curr = 1e-30;

            let mut values = vec![0.0; start_n + 1];
            values[start_n] = j_curr;

            for k in (0..start_n).rev() {
                let j_prev = (2 * k + 3) as f64 / x * j_curr - j_next;
                values[k] = j_prev;
                j_next = j_curr;
                j_curr = j_prev;
            }

            // Normalize with whichever of j₀ or j₁ is farther from zero.
            // Using j₀ alone is singular at every non-zero root of sin(x),
            // even though the downward recurrence and the higher-order
            // spherical Bessel functions remain well-defined there.
            let true_j0 = x.sin() / x;
            let true_j1 = x.sin() / (x * x) - x.cos() / x;
            let scale = if true_j0.abs() >= true_j1.abs() {
                true_j0 / values[0]
            } else {
                true_j1 / values[1]
            };

            values[n] * scale
        }
    }
}

/// Spherical Bessel function yₙ(x) (Neumann function)
///
/// Relation: yₙ(x) = √(π/2x) * Y_{n+1/2}(x)
pub fn spherical_bessel_y(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return f64::NEG_INFINITY;
    }

    match n {
        0 => -x.cos() / x,
        1 => -x.cos() / (x * x) - x.sin() / x,
        _ => {
            // Upward recurrence (stable for y_n)
            let mut y_nm2 = -x.cos() / x;
            let mut y_nm1 = -x.cos() / (x * x) - x.sin() / x;

            for k in 2..=n {
                let y_n = (2 * k - 1) as f64 / x * y_nm1 - y_nm2;
                y_nm2 = y_nm1;
                y_nm1 = y_n;
            }

            y_nm1
        }
    }
}

/// Legendre polynomial Pₙ(x)
///
/// Uses recurrence: (n+1)Pₙ₊₁(x) = (2n+1)x Pₙ(x) - n Pₙ₋₁(x)
pub fn legendre_p(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p_nm2 = 1.0;
            let mut p_nm1 = x;

            for k in 2..=n {
                let p_n = ((2 * k - 1) as f64 * x * p_nm1 - (k - 1) as f64 * p_nm2) / k as f64;
                p_nm2 = p_nm1;
                p_nm1 = p_n;
            }

            p_nm1
        }
    }
}

/// Radar Cross Section (RCS) for sphere
///
/// RCS = σ = 4π/k² * Σₙ (2n+1) |aₙ|²
pub fn sphere_rcs_3d(wave_number: f64, radius: f64, num_terms: usize) -> f64 {
    let ka = wave_number * radius;
    let coefficients = compute_rigid_sphere_coefficients(ka, num_terms);

    let mut rcs = 0.0;
    for (n, a_n) in coefficients.iter().enumerate() {
        rcs += (2 * n + 1) as f64 * a_n.norm_sqr();
    }

    4.0 * PI * rcs / (wave_number * wave_number)
}

/// Scattering efficiency Q_scat = σ / (πa²)
///
/// Ratio of scattering cross-section to geometric cross-section.
pub fn sphere_scattering_efficiency_3d(wave_number: f64, radius: f64, num_terms: usize) -> f64 {
    let rcs = sphere_rcs_3d(wave_number, radius, num_terms);
    rcs / (PI * radius * radius)
}

/// 3D plane wave: p(x,y,z) = exp(ik·r)
///
/// A plane wave propagating in direction (θ, φ).
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `theta` - Polar angle (from z-axis)
/// * `phi` - Azimuthal angle
/// * `points` - Evaluation points
pub fn plane_wave_3d(
    wave_number: f64,
    theta: f64,
    phi: f64,
    points: Vec<Point>,
) -> AnalyticalSolution {
    // Wave vector direction
    let kx = wave_number * theta.sin() * phi.cos();
    let ky = wave_number * theta.sin() * phi.sin();
    let kz = wave_number * theta.cos();

    let pressure: Vec<Complex64> = points
        .iter()
        .map(|p| {
            let phase = kx * p.x + ky * p.y + kz * p.z;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!(
            "3D Plane Wave (k={}, θ={:.2}, φ={:.2})",
            wave_number, theta, phi
        ),
        dimensions: 3,
        positions: points,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "direction_theta": theta,
            "direction_phi": phi,
            "wave_vector": [kx, ky, kz],
            "wavelength": 2.0 * PI / wave_number,
        }),
    }
}

/// Monopole radiation from a pulsating sphere
///
/// A sphere of radius `a` vibrating with uniform radial velocity `v₀` radiates
/// an isotropic (monopole) field. The exterior solution is:
///
/// ```text
/// p(r) = A · h₀⁽¹⁾(kr)
/// ```
///
/// where the amplitude `A` is determined by the Neumann boundary condition
/// `∂p/∂r|_{r=a} = -iωρ₀ v₀`:
///
/// ```text
/// A = -i·ρ₀·c₀·v₀ / h₀⁽¹⁾'(ka)
/// ```
///
/// Explicit forms:
/// ```text
/// h₀⁽¹⁾(x)  = -i·exp(ix)/x
/// h₀⁽¹⁾'(x) = -i·exp(ix)·(ix - 1)/x²
/// ```
///
/// The radiation is isotropic; pressure only depends on `r`.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Sphere radius a (evaluation points must satisfy r ≥ a)
/// * `velocity_amplitude` - Surface radial velocity v₀ (m/s)
/// * `speed_of_sound` - c₀ (m/s)
/// * `density` - Ambient fluid density ρ₀ (kg/m³)
/// * `r_points` - Radial evaluation points (must be ≥ radius)
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 7.2
/// - Kinsler et al., "Fundamentals of Acoustics", 4th ed., Section 7.3
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::pulsating_sphere_3d;
///
/// // ka = 1: Mie regime
/// let solution = pulsating_sphere_3d(
///     1.0,   // wave_number
///     1.0,   // radius
///     1.0,   // velocity_amplitude
///     343.0, // speed_of_sound
///     1.21,  // density
///     vec![1.0, 2.0, 4.0],  // r_points
/// );
/// // Pressure magnitude decays as 1/r beyond the sphere
/// let mag = solution.magnitude();
/// ```
///
/// For externally supplied data, use [`try_pulsating_sphere_3d`] to enforce
/// the lossless homogeneous-medium and exterior-domain assumptions before any
/// singular spherical-Hankel factors are evaluated.
pub fn pulsating_sphere_3d(
    wave_number: f64,
    radius: f64,
    velocity_amplitude: f64,
    speed_of_sound: f64,
    density: f64,
    r_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;

    // h₀⁽¹⁾(x) = -i·exp(ix)/x  →  complex value at ka
    // h₀⁽¹⁾'(x) = -i·exp(ix)·(ix-1)/x²
    let exp_ika = Complex64::new(0.0, ka).exp();
    let mi = Complex64::new(0.0, -1.0); // -i
    let h0_prime_ka = mi * exp_ika * Complex64::new(-1.0, ka) / (ka * ka);
    // Note: (ix - 1) = (-1 + ix)

    // With the exp(+ikr) convention used here, the outward normal velocity
    // satisfies ∂p/∂n = iωρ₀v₀, so A = i·ρ₀·c₀·v₀ / h₀'(ka).
    let amplitude = -mi * density * speed_of_sound * velocity_amplitude / h0_prime_ka;

    let mut positions = Vec::with_capacity(r_points.len());
    let mut pressure = Vec::with_capacity(r_points.len());

    for &r in &r_points {
        positions.push(Point::from_spherical(r, 0.0, 0.0));

        // h₀⁽¹⁾(kr) = -i·exp(ikr)/(kr)
        let kr = wave_number * r;
        let exp_ikr = Complex64::new(0.0, kr).exp();
        let h0_kr = mi * exp_ikr / kr;

        pressure.push(amplitude * h0_kr);
    }

    let frequency = wave_number * speed_of_sound / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Pulsating Sphere (ka={:.2})", ka),
        dimensions: 3,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "radius": radius,
            "ka": ka,
            "velocity_amplitude": velocity_amplitude,
            "speed_of_sound": speed_of_sound,
            "density": density,
            "boundary_condition": "Neumann (pulsating)",
            "radiation_pattern": "isotropic (monopole)",
        }),
    }
}

/// Checked exterior reference for a uniformly pulsating sphere.
///
/// The model assumes a stationary, homogeneous, lossless fluid and a peak
/// surface radial-velocity phasor. It is defined only for finite positive
/// wavenumber, sphere radius, sound speed, density, and exterior points
/// `r >= radius`. The returned pressure follows the crate's `exp(+iωt)`
/// convention; consequently this near-field reference must not be interpreted
/// using the progressive-plane-wave impedance away from the far field.
pub fn try_pulsating_sphere_3d(
    wave_number: f64,
    radius: f64,
    velocity_amplitude: f64,
    speed_of_sound: f64,
    density: f64,
    r_points: Vec<f64>,
) -> Result<AnalyticalSolution, PulsatingSphereError> {
    for (field, value) in [
        ("wave_number", wave_number),
        ("radius", radius),
        ("speed_of_sound", speed_of_sound),
        ("density", density),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PulsatingSphereError::InvalidParameter {
                field,
                requirement: "finite and positive",
            });
        }
    }
    if !velocity_amplitude.is_finite() {
        return Err(PulsatingSphereError::InvalidParameter {
            field: "velocity_amplitude",
            requirement: "finite",
        });
    }
    for (index, &point_radius) in r_points.iter().enumerate() {
        if !point_radius.is_finite() || point_radius < radius {
            return Err(PulsatingSphereError::InteriorEvaluationPoint {
                index,
                radius: point_radius,
                sphere_radius: radius,
            });
        }
    }

    Ok(pulsating_sphere_3d(
        wave_number,
        radius,
        velocity_amplitude,
        speed_of_sound,
        density,
        r_points,
    ))
}

/// Dipole radiation from a sphere oscillating along the z-axis
///
/// A rigid sphere of radius `a` translating along z with surface velocity
/// `v_n(θ) = v₀·cos(θ)`. The exterior pressure field is:
///
/// ```text
/// p(r,θ) = B · h₁⁽¹⁾(kr) · cos(θ)
/// ```
///
/// The coefficient `B` is fixed by the boundary condition
/// `∂p/∂r|_{r=a} = -iωρ₀ v₀·cos(θ)`:
///
/// ```text
/// B = -i·ρ₀·c₀·v₀ / h₁⁽¹⁾'(ka)
/// ```
///
/// The derivative uses the recurrence relation:
/// ```text
/// h₁⁽¹⁾'(x) = h₀⁽¹⁾(x) - (2/x)·h₁⁽¹⁾(x)
/// ```
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Sphere radius a
/// * `velocity_amplitude` - Oscillation velocity amplitude v₀ (m/s)
/// * `speed_of_sound` - c₀ (m/s)
/// * `density` - Ambient fluid density ρ₀ (kg/m³)
/// * `r_points` - Radial evaluation points (must be ≥ radius)
/// * `theta_points` - Polar angle points (0 to π)
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 7.2
/// - Bowman et al., "Electromagnetic and Acoustic Scattering", 1987
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::oscillating_sphere_3d;
/// use std::f64::consts::PI;
///
/// let solution = oscillating_sphere_3d(
///     1.0,   // wave_number
///     1.0,   // radius
///     1.0,   // velocity_amplitude
///     343.0, // speed_of_sound
///     1.21,  // density
///     vec![2.0, 4.0],
///     vec![0.0, PI / 2.0, PI],
/// );
/// // Pressure at θ=π/2 (equatorial plane) should be zero
/// ```
pub fn oscillating_sphere_3d(
    wave_number: f64,
    radius: f64,
    velocity_amplitude: f64,
    speed_of_sound: f64,
    density: f64,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;
    let mi = Complex64::new(0.0, -1.0); // -i

    // h₁⁽¹⁾(x) = jₙ(x) + i·yₙ(x)
    let j1_ka = spherical_bessel_j(1, ka);
    let y1_ka = spherical_bessel_y(1, ka);
    let h1_ka = Complex64::new(j1_ka, y1_ka);

    // h₀⁽¹⁾(ka) needed for recurrence
    let j0_ka = spherical_bessel_j(0, ka);
    let y0_ka = spherical_bessel_y(0, ka);
    let h0_ka = Complex64::new(j0_ka, y0_ka);

    // h₁⁽¹⁾'(x) = h₀⁽¹⁾(x) - (2/x)·h₁⁽¹⁾(x)
    let h1_prime_ka = h0_ka - (2.0 / ka) * h1_ka;

    // B = -i·ρ₀·c₀·v₀ / h₁'(ka)
    let amplitude = mi * density * speed_of_sound * velocity_amplitude / h1_prime_ka;

    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &r in &r_points {
        let kr = wave_number * r;
        let j1_kr = spherical_bessel_j(1, kr);
        let y1_kr = spherical_bessel_y(1, kr);
        let h1_kr = Complex64::new(j1_kr, y1_kr);

        for &theta in &theta_points {
            positions.push(Point::from_spherical(r, theta, 0.0));

            // p = B · h₁⁽¹⁾(kr) · cos(θ)  [P₁(cos θ) = cos θ]
            let p = amplitude * h1_kr * theta.cos();
            pressure.push(p);
        }
    }

    let frequency = wave_number * speed_of_sound / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Oscillating Sphere (ka={:.2})", ka),
        dimensions: 3,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "radius": radius,
            "ka": ka,
            "velocity_amplitude": velocity_amplitude,
            "speed_of_sound": speed_of_sound,
            "density": density,
            "boundary_condition": "Neumann (oscillating)",
            "radiation_pattern": "dipole",
            "r_points": r_points,
            "theta_range": [theta_points.first(), theta_points.last()],
        }),
    }
}

/// Plane wave scattering from a soft (pressure-release) sphere (Dirichlet BC)
///
/// Analytical solution using spherical harmonics expansion.  The total field is:
///
/// ```text
/// p(r,θ) = Σₙ (2n+1) iⁿ [jₙ(kr) - aₙ hₙ⁽¹⁾(kr)] Pₙ(cos θ)
/// ```
///
/// For a **soft sphere** (Dirichlet BC: p = 0 on surface), the coefficients are:
///
/// ```text
/// aₙ = jₙ(ka) / hₙ⁽¹⁾(ka)
/// ```
///
/// This ensures that the total field vanishes at `r = a`.
/// The model has the same exterior homogeneous-lossless-fluid, unit peak
/// incident-phasor, and `exp(+iωt)` convention as [`sphere_scattering_3d`].
/// `num_terms` is a convergence parameter, not an accuracy guarantee.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Sphere radius a
/// * `num_terms` - Number of series terms (typically ka + 10)
/// * `r_points` - Radial evaluation points (must be ≥ radius)
/// * `theta_points` - Polar angle points (0 to π)
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 8.3
/// - Bowman et al., "Electromagnetic and Acoustic Scattering", 1987
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::soft_sphere_scattering_3d;
/// use std::f64::consts::PI;
///
/// let solution = soft_sphere_scattering_3d(
///     1.0,  // wave_number
///     1.0,  // radius
///     20,   // num_terms
///     vec![1.0, 2.0],          // r_points (1.0 is on surface)
///     vec![0.0, PI / 2.0, PI], // theta_points
/// );
/// ```
pub fn soft_sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;

    // Dirichlet BC coefficients: aₙ = jₙ(ka) / hₙ⁽¹⁾(ka)
    let coefficients: Vec<Complex64> = (0..num_terms)
        .map(|n| {
            let jn = spherical_bessel_j(n, ka);
            let yn = spherical_bessel_y(n, ka);
            let hn = Complex64::new(jn, yn);
            Complex64::new(jn, 0.0) / hn
        })
        .collect();

    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &r in &r_points {
        for &theta in &theta_points {
            positions.push(Point::from_spherical(r, theta, 0.0));

            let kr = wave_number * r;
            let cos_theta = theta.cos();

            let mut total = Complex64::new(0.0, 0.0);

            for (n, coeff) in coefficients.iter().enumerate().take(num_terms) {
                let n_f64 = n as f64;
                let prefactor = 2.0 * n_f64 + 1.0;
                let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

                let jn = spherical_bessel_j(n, kr);
                let yn = spherical_bessel_y(n, kr);
                let hn = Complex64::new(jn, yn);
                let pn = legendre_p(n, cos_theta);

                total += prefactor * i_power_n * (jn - coeff * hn) * pn;
            }

            pressure.push(total);
        }
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Soft Sphere Scattering (ka={:.2})", ka),
        dimensions: 3,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "radius": radius,
            "ka": ka,
            "num_terms": num_terms,
            "boundary_condition": "soft (Dirichlet)",
            "r_points": r_points,
            "theta_range": [theta_points.first(), theta_points.last()],
            "regime": classify_regime(ka),
        }),
    }
}

/// Checked exterior soft-sphere (Dirichlet) Mie reference.
pub fn try_soft_sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> Result<AnalyticalSolution, SphereScatteringError> {
    validate_sphere_scattering_parameters(wave_number, radius, num_terms)?;
    validate_sphere_scattering_grid(wave_number, radius, &r_points, &theta_points)?;
    let solution =
        soft_sphere_scattering_3d(wave_number, radius, num_terms, r_points, theta_points);
    validate_sphere_scattering_solution(&solution)?;
    Ok(solution)
}

/// Plane wave scattering from a sphere with impedance (Robin) boundary condition
///
/// Analytical solution with a mixed boundary condition on the surface:
///
/// ```text
/// ∂p/∂r - ikβ·p = 0  on r = a
/// ```
///
/// where `β = Z₀ / Z_surface` is the normalized admittance (Z₀ = ρ₀c₀).
///
/// The scattering coefficients are:
///
/// ```text
/// aₙ = [jₙ'(ka) - iβ·jₙ(ka)] / [hₙ⁽¹⁾'(ka) - iβ·hₙ⁽¹⁾(ka)]
/// ```
///
/// This uses the exterior-obstacle convention `∂p/∂r - i k β p = 0`
/// with `exp(+iωt)` and a radial normal pointing from the obstacle into the
/// fluid, matching Sonium BEM's positive normalized-admittance convention.
/// The primes are derivatives with respect to the dimensionless argument
/// `kr`, so the common factor `k` from the physical normal derivative has
/// already been cancelled; there is no extra `k` multiplying `β` above.
/// The reference models a homogeneous stationary lossless exterior fluid and
/// a unit peak incident-pressure phasor. It accepts finite complex `β`, which
/// permits active or passive boundary studies; passivity must be enforced by a
/// caller when that is required by the physical experiment.
///
/// **Special cases:**
/// - β → 0: reduces to rigid sphere (Neumann BC)
/// - β → ∞: reduces to soft sphere (Dirichlet BC)
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c
/// * `radius` - Sphere radius a
/// * `num_terms` - Number of series terms (typically ka + 10)
/// * `normalized_admittance` - β = Z₀/Z_surface (complex)
/// * `r_points` - Radial evaluation points (must be ≥ radius)
/// * `theta_points` - Polar angle points (0 to π)
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 8.3
/// - Bowman et al., "Electromagnetic and Acoustic Scattering", 1987
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::impedance_sphere_scattering_3d;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // Slightly absorbing surface
/// let beta = Complex64::new(0.1, 0.0);
/// let solution = impedance_sphere_scattering_3d(
///     1.0,   // wave_number
///     1.0,   // radius
///     20,    // num_terms
///     beta,  // normalized_admittance
///     vec![2.0],
///     vec![0.0, PI / 2.0, PI],
/// );
/// ```
pub fn impedance_sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    normalized_admittance: Complex64,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> AnalyticalSolution {
    let ka = wave_number * radius;
    let i_beta = Complex64::new(0.0, 1.0) * normalized_admittance;

    // The derivatives are with respect to kr, so the physical k cancels:
    // aₙ = [jₙ'(ka) - iβ·jₙ(ka)] / [hₙ'(ka) - iβ·hₙ(ka)].
    let coefficients: Vec<Complex64> = (0..num_terms)
        .map(|n| {
            let n_f64 = n as f64;

            let jn = spherical_bessel_j(n, ka);
            let yn = spherical_bessel_y(n, ka);

            // jₙ'(x) = jₙ₋₁(x) - (n+1)/x · jₙ(x)
            let jn_minus_1 = if n > 0 {
                spherical_bessel_j(n - 1, ka)
            } else {
                ka.cos() / ka
            };
            let jn_prime = jn_minus_1 - (n_f64 + 1.0) / ka * jn;

            // yₙ'(x) = yₙ₋₁(x) - (n+1)/x · yₙ(x)
            let yn_minus_1 = if n > 0 {
                spherical_bessel_y(n - 1, ka)
            } else {
                // y₋₁(x) = sin(x)/x
                ka.sin() / ka
            };
            let yn_prime = yn_minus_1 - (n_f64 + 1.0) / ka * yn;

            let hn = Complex64::new(jn, yn);
            let hn_prime = Complex64::new(jn_prime, yn_prime);

            let numerator = Complex64::new(jn_prime, 0.0) - i_beta * jn;
            let denominator = hn_prime - i_beta * hn;

            numerator / denominator
        })
        .collect();

    let mut positions = Vec::new();
    let mut pressure = Vec::new();

    for &r in &r_points {
        for &theta in &theta_points {
            positions.push(Point::from_spherical(r, theta, 0.0));

            let kr = wave_number * r;
            let cos_theta = theta.cos();

            let mut total = Complex64::new(0.0, 0.0);

            for (n, coeff) in coefficients.iter().enumerate().take(num_terms) {
                let n_f64 = n as f64;
                let prefactor = 2.0 * n_f64 + 1.0;
                let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());

                let jn = spherical_bessel_j(n, kr);
                let yn = spherical_bessel_y(n, kr);
                let hn = Complex64::new(jn, yn);
                let pn = legendre_p(n, cos_theta);

                total += prefactor * i_power_n * (jn - coeff * hn) * pn;
            }

            pressure.push(total);
        }
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Impedance Sphere Scattering (ka={:.2})", ka),
        dimensions: 3,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "radius": radius,
            "ka": ka,
            "num_terms": num_terms,
            "boundary_condition": "impedance (Robin)",
            "normalized_admittance_re": normalized_admittance.re,
            "normalized_admittance_im": normalized_admittance.im,
            "r_points": r_points,
            "theta_range": [theta_points.first(), theta_points.last()],
            "regime": classify_regime(ka),
        }),
    }
}

/// Checked exterior impedance-sphere (Robin) Mie reference.
pub fn try_impedance_sphere_scattering_3d(
    wave_number: f64,
    radius: f64,
    num_terms: usize,
    normalized_admittance: Complex64,
    r_points: Vec<f64>,
    theta_points: Vec<f64>,
) -> Result<AnalyticalSolution, SphereScatteringError> {
    validate_sphere_scattering_parameters(wave_number, radius, num_terms)?;
    if !normalized_admittance.re.is_finite() || !normalized_admittance.im.is_finite() {
        return Err(SphereScatteringError::InvalidParameter {
            field: "normalized_admittance",
            requirement: "finite real and imaginary components",
        });
    }
    validate_sphere_scattering_grid(wave_number, radius, &r_points, &theta_points)?;
    let solution = impedance_sphere_scattering_3d(
        wave_number,
        radius,
        num_terms,
        normalized_admittance,
        r_points,
        theta_points,
    );
    validate_sphere_scattering_solution(&solution)?;
    Ok(solution)
}

/// Standing wave modes in a rigid-walled duct of length L
///
/// For a 1D duct with rigid walls at both ends (x = 0 and x = L),
/// the resonant frequencies and mode shapes are:
///
/// ```text
/// f_n = n·c / (2L),  n = 0, 1, 2, ...
/// p_n(x) = cos(nπx/L)
/// ```
///
/// For arbitrary driving at wave number k (not necessarily a resonance):
///
/// ```text
/// p(x) = [cos(k·x) + cos(k·(L-x))] / (2·cos(kL))
/// ```
///
/// This represents the standing wave pattern excited by a unit-pressure
/// source that satisfies `∂p/∂x = 0` at both ends.  The solution diverges
/// when `kL = (n+½)π` (anti-resonances of the rigid-rigid duct are
/// non-existent for this configuration; it blows up at resonance `kL = nπ`
/// only if `n > 0` is treated as the driving frequency hitting a mode).
///
/// The function returns the **mode shape** `cos(nπx/L)` when
/// `mode_number > 0`, scaled to unit amplitude at `x = 0`.
///
/// # Arguments
///
/// * `wave_number` - k = 2πf/c (driving frequency)
/// * `length` - Duct length L (m)
/// * `mode_number` - Mode index n (0 = plane wave, 1 = first mode, …)
/// * `num_points` - Number of evaluation points along [0, L]
///
/// # References
///
/// - Morse & Ingard, "Theoretical Acoustics", 1968, Section 9.1
/// - Kinsler et al., "Fundamentals of Acoustics", 4th ed., Chapter 9
///
/// # Example
///
/// ```rust
/// use sonium_wave::analytical::duct_modes_1d;
/// use std::f64::consts::PI;
///
/// // First duct mode (n=1) driven at its resonant frequency
/// let k = PI / 1.0; // kL = π → first resonance, L=1
/// let solution = duct_modes_1d(k, 1.0, 1, 50);
/// // p(0) = 1, p(L/2) = -1 (half-wavelength node at x = L/2 for n=1)
/// ```
pub fn duct_modes_1d(
    wave_number: f64,
    length: f64,
    mode_number: usize,
    num_points: usize,
) -> AnalyticalSolution {
    let resonant_frequency = mode_number as f64 * SPEED_OF_SOUND / (2.0 * length);

    let n_pi_over_l = mode_number as f64 * PI / length;

    let mut positions = Vec::with_capacity(num_points);
    let mut pressure = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let x = if num_points > 1 {
            length * i as f64 / (num_points - 1) as f64
        } else {
            0.0
        };
        positions.push(Point::new_1d(x));

        // Mode shape: cos(nπx/L) — real-valued, stored as complex
        let p_real = (n_pi_over_l * x).cos();
        pressure.push(Complex64::new(p_real, 0.0));
    }

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!(
            "1D Duct Mode n={} (kL={:.2})",
            mode_number,
            wave_number * length
        ),
        dimensions: 1,
        positions,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "length": length,
            "mode_number": mode_number,
            "resonant_frequency_hz": resonant_frequency,
            "resonant_wave_number": n_pi_over_l,
            "num_points": num_points,
            "boundary_condition": "rigid-rigid (Neumann both ends)",
        }),
    }
}

/// Point source (monopole): G(r) = exp(ikr)/(4πr)
///
/// Green's function for the Helmholtz equation in 3D.
///
/// # Arguments
///
/// * `wave_number` - k
/// * `source` - Source location
/// * `points` - Field points
pub fn point_source_3d(wave_number: f64, source: Point, points: Vec<Point>) -> AnalyticalSolution {
    let pressure: Vec<Complex64> = points
        .iter()
        .map(|p| {
            let r = p.distance_to(&source);
            if r < 1e-15 {
                Complex64::new(f64::INFINITY, 0.0)
            } else {
                let kr = wave_number * r;
                Complex64::new(kr.cos(), kr.sin()) / (4.0 * PI * r)
            }
        })
        .collect();

    let frequency = wave_number * SPEED_OF_SOUND / (2.0 * PI);

    AnalyticalSolution {
        name: format!("3D Point Source (k={})", wave_number),
        dimensions: 3,
        positions: points,
        pressure,
        wave_number,
        frequency,
        metadata: serde_json::json!({
            "source": [source.x, source.y, source.z],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn grid_aware_term_rule_covers_far_observers() {
        // Near observers keep the ka rule; far observers (kr >> ka)
        // need n_max past k*r of the evaluation grid.
        assert_eq!(suggested_sphere_terms_for_grid(5.0, 10.0), 36);
        assert_eq!(suggested_sphere_terms_for_grid(5.0, 40.0), 60);
        assert_eq!(suggested_sphere_terms_for_grid(10.0, 80.0), 100);
        assert_eq!(
            suggested_sphere_terms_for_grid(1.0, f64::NAN),
            suggested_sphere_terms(1.0)
        );
    }

    #[test]
    fn spherical_bessel_j1_keeps_precision_at_small_argument() {
        // The direct n = 1 formula sin(x)/x^2 - cos(x)/x cancels
        // catastrophically below ~1e-4 (zero good digits at 1e-8), which
        // corrupted recurrences such as j_n' built on top of it. The Taylor
        // branch must hold full precision through the cutoff.
        // Note: x = 1e-4 itself takes the direct-formula branch (the Taylor
        // cutoff is strict `< 1e-4`), where cancellation already costs ~2e-12,
        // so the tight-epsilon loop stops at 5e-5.
        for x in [1e-12f64, 1e-8, 1e-6, 5e-5] {
            let expected = x / 3.0 - x.powi(3) / 30.0;
            assert_abs_diff_eq!(spherical_bessel_j(1, x), expected, epsilon = 1e-12);
        }
        // Continuity across the Taylor/direct branch boundary.
        let below = spherical_bessel_j(1, 9.9e-5);
        let above = spherical_bessel_j(1, 1.1e-4);
        let slope = (above - below) / 1.1e-5;
        assert!(
            (slope - 1.0 / 3.0).abs() < 1e-6,
            "branch slope drift: {slope}"
        );
    }

    #[test]
    fn fluid_sphere_transparent_limit_reproduces_incident_plane_wave() {
        let reference = FluidSphereTransmission3d::try_new(0.2, 3.0, 3.0, 1.21, 1.21, 32)
            .expect("equal-medium fluid sphere");
        for (radius, theta) in [(0.2, 0.0), (0.7, 0.8), (1.1, PI)] {
            let actual = reference
                .exterior_pressure(radius, theta)
                .expect("exterior point");
            let expected = Complex64::from_polar(1.0, 3.0 * radius * theta.cos());
            assert!(
                (actual - expected).norm() < 1.0e-11,
                "transparent fluid sphere mismatch at r={radius}, theta={theta}: {actual:?} vs {expected:?}"
            );
        }
        assert!(
            reference
                .exterior_coefficients
                .iter()
                .all(|coefficient| coefficient.norm() < 1.0e-13)
        );
    }

    #[test]
    fn fluid_sphere_enforces_pressure_and_velocity_continuity() {
        let reference = FluidSphereTransmission3d::try_new(0.2, 2.5, 4.0, 1.21, 8.0, 40)
            .expect("contrasting fluid sphere");
        assert!(reference.pressure_continuity_error(33) < 1.0e-11);
        assert!(reference.velocity_continuity_error(33) < 1.0e-11);
    }

    #[test]
    fn fluid_sphere_high_density_limit_matches_rigid_sphere() {
        let wave_number = 2.0;
        let radius = 0.2;
        let terms = 40;
        let fluid = FluidSphereTransmission3d::try_new(
            radius,
            wave_number,
            wave_number,
            1.21,
            1.0e12,
            terms,
        )
        .expect("high-density fluid sphere");
        let radii = vec![0.5, 1.0];
        let angles = vec![0.0, 0.6, 1.4, PI];
        let rigid = sphere_scattering_3d(wave_number, radius, terms, radii.clone(), angles.clone());
        let mut index = 0;
        for evaluation_radius in radii {
            for &theta in &angles {
                let actual = fluid
                    .exterior_pressure(evaluation_radius, theta)
                    .expect("exterior point");
                let expected = rigid.pressure[index];
                assert!(
                    (actual - expected).norm() < 1.0e-9,
                    "high-density limit mismatch at r={evaluation_radius}, theta={theta}: {actual:?} vs {expected:?}"
                );
                index += 1;
            }
        }
    }

    #[test]
    fn fluid_sphere_rejects_invalid_media_and_domains() {
        assert!(FluidSphereTransmission3d::try_new(0.0, 2.0, 3.0, 1.2, 2.0, 16).is_err());
        assert!(FluidSphereTransmission3d::try_new(0.2, 2.0, 3.0, 1.2, 2.0, 0).is_err());
        let reference = FluidSphereTransmission3d::try_new(0.2, 2.0, 3.0, 1.2, 2.0, 16)
            .expect("valid fluid sphere");
        assert!(reference.exterior_pressure(0.1, 0.0).is_err());
        assert!(reference.interior_pressure(0.3, 0.0).is_err());
        assert!(reference.exterior_pressure(0.3, PI + 0.1).is_err());
    }

    #[test]
    fn test_spherical_bessel_j0() {
        // j₀(x) = sin(x)/x
        let x = 1.0;
        assert_abs_diff_eq!(spherical_bessel_j(0, x), x.sin() / x, epsilon = 1e-10);

        let x = PI;
        assert_abs_diff_eq!(spherical_bessel_j(0, x), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_bessel_j1() {
        // j₁(x) = sin(x)/x² - cos(x)/x
        let x: f64 = 2.0;
        let expected = x.sin() / (x * x) - x.cos() / x;
        assert_abs_diff_eq!(spherical_bessel_j(1, x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_bessel_y0() {
        // y₀(x) = -cos(x)/x
        let x = 1.0;
        assert_abs_diff_eq!(spherical_bessel_y(0, x), -x.cos() / x, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_polynomials() {
        // P₀(x) = 1
        assert_abs_diff_eq!(legendre_p(0, 0.5), 1.0, epsilon = 1e-10);

        // P₁(x) = x
        assert_abs_diff_eq!(legendre_p(1, 0.5), 0.5, epsilon = 1e-10);

        // P₂(x) = (3x² - 1)/2
        let x = 0.5;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert_abs_diff_eq!(legendre_p(2, x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_rayleigh_scattering() {
        // Rayleigh regime: ka << 1
        let k = 0.1;
        let a = 1.0;
        let ka = k * a;

        let rcs = sphere_rcs_3d(k, a, 10);

        // RCS should be positive and finite
        assert!(rcs > 0.0);
        assert!(rcs.is_finite());

        // In Rayleigh regime, RCS ~ (ka)⁴
        let rayleigh_scaling = ka.powi(4);
        assert!(rcs < rayleigh_scaling * 1000.0); // Rough check
    }

    #[test]
    fn test_sphere_geometric_limit() {
        // Geometric regime: ka >> 1
        // RCS → 2πa² (twice geometric cross section)
        let k = 20.0;
        let a = 1.0;

        let rcs = sphere_rcs_3d(k, a, 50);
        let geometric = 2.0 * PI * a * a;

        // Should approach geometric limit
        assert!((rcs / geometric - 1.0).abs() < 0.2); // Within 20%
    }

    #[test]
    fn test_sphere_scattering_3d() {
        // Basic sanity test
        let k = 1.0;
        let a = 1.0;

        let solution = sphere_scattering_3d(k, a, 20, vec![2.0], vec![0.0, PI / 2.0, PI]);

        assert_eq!(solution.pressure.len(), 3);

        // All pressures should be finite
        for p in &solution.pressure {
            assert!(p.re.is_finite());
            assert!(p.im.is_finite());
        }
    }

    #[test]
    fn checked_sphere_scattering_references_match_legacy_values() {
        let rigid =
            try_sphere_scattering_3d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0]).unwrap();
        assert_eq!(
            rigid.pressure,
            sphere_scattering_3d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0]).pressure
        );

        let soft = try_soft_sphere_scattering_3d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0])
            .unwrap();
        assert_eq!(
            soft.pressure,
            soft_sphere_scattering_3d(1.0, 0.5, 20, vec![0.5, 1.0], vec![0.0, PI / 3.0]).pressure
        );

        let beta = Complex64::new(0.2, -0.1);
        let impedance = try_impedance_sphere_scattering_3d(
            1.0,
            0.5,
            20,
            beta,
            vec![0.5, 1.0],
            vec![0.0, PI / 3.0],
        )
        .unwrap();
        assert_eq!(
            impedance.pressure,
            impedance_sphere_scattering_3d(1.0, 0.5, 20, beta, vec![0.5, 1.0], vec![0.0, PI / 3.0])
                .pressure
        );
    }

    #[test]
    fn checked_sphere_scattering_references_reject_invalid_domains() {
        assert!(matches!(
            try_sphere_scattering_3d(1.0, 1.0, 20, vec![0.9], vec![0.0]),
            Err(SphereScatteringError::InteriorEvaluationPoint { index: 0, .. })
        ));
        assert!(matches!(
            try_soft_sphere_scattering_3d(1.0, 1.0, 0, vec![1.0], vec![0.0]),
            Err(SphereScatteringError::InvalidParameter {
                field: "num_terms",
                ..
            })
        ));
        assert!(matches!(
            try_impedance_sphere_scattering_3d(
                1.0,
                1.0,
                20,
                Complex64::new(f64::NAN, 0.0),
                vec![1.0],
                vec![0.0]
            ),
            Err(SphereScatteringError::InvalidParameter {
                field: "normalized_admittance",
                ..
            })
        ));
        assert!(matches!(
            try_sphere_scattering_3d(1.0, 1.0, 20, vec![1.0], vec![-0.1]),
            Err(SphereScatteringError::InvalidParameter {
                field: "theta_points",
                ..
            })
        ));
    }

    #[test]
    fn checked_rigid_and_impedance_spheres_satisfy_surface_conditions() {
        let wave_number = 1.2;
        let radius = 0.7;
        let radial_step = 1e-5;
        let theta = 0.41;
        let radii = vec![radius, radius + radial_step, radius + 2.0 * radial_step];
        let rigid =
            try_sphere_scattering_3d(wave_number, radius, 40, radii.clone(), vec![theta]).unwrap();
        let rigid_derivative = (-3.0 * rigid.pressure[0] + 4.0 * rigid.pressure[1]
            - rigid.pressure[2])
            / (2.0 * radial_step);
        assert!(
            rigid_derivative.norm() < 1e-7,
            "rigid-sphere normal derivative magnitude = {:.3e}",
            rigid_derivative.norm()
        );

        let beta = Complex64::new(0.2, -0.1);
        let impedance =
            try_impedance_sphere_scattering_3d(wave_number, radius, 40, beta, radii, vec![theta])
                .unwrap();
        let impedance_derivative = (-3.0 * impedance.pressure[0] + 4.0 * impedance.pressure[1]
            - impedance.pressure[2])
            / (2.0 * radial_step);
        let robin_residual =
            impedance_derivative - Complex64::new(0.0, wave_number) * beta * impedance.pressure[0];
        assert!(
            robin_residual.norm() < 1e-7,
            "impedance-sphere Robin residual magnitude = {:.3e}",
            robin_residual.norm()
        );
    }

    #[test]
    fn test_sphere_scattering_3d_is_finite_at_zero_of_j0() {
        // Miller's downward recurrence must not be normalized exclusively with
        // j_0(x): j_0(PI) is zero while the higher-order terms are finite.
        assert_abs_diff_eq!(spherical_bessel_j(2, PI), 3.0 / (PI * PI), epsilon = 1e-13);

        let solution =
            sphere_scattering_3d(PI, 1.0, 20, vec![2.0], vec![0.0, PI / 4.0, PI / 2.0, PI]);

        assert!(
            solution
                .pressure
                .iter()
                .all(|p| p.re.is_finite() && p.im.is_finite())
        );
    }

    #[test]
    fn test_scattering_efficiency() {
        let k = 1.0;
        let a = 1.0;

        let q_scat = sphere_scattering_efficiency_3d(k, a, 30);

        // Efficiency should be positive and reasonable
        assert!(q_scat > 0.0);
        assert!(q_scat < 10.0);
    }

    #[test]
    fn test_regime_classification() {
        assert_eq!(classify_regime(0.1), "Rayleigh (ka << 1)");
        assert_eq!(classify_regime(1.0), "Mie (ka ~ 1)");
        assert_eq!(classify_regime(10.0), "Geometric (ka >> 1)");
    }

    #[test]
    fn test_point_source_3d() {
        let k = 1.0;
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let points = vec![Point::new_3d(1.0, 0.0, 0.0), Point::new_3d(2.0, 0.0, 0.0)];

        let solution = point_source_3d(k, source, points);

        // Magnitude should decay as 1/r
        let ratio = solution.pressure[1].norm() / solution.pressure[0].norm();
        assert_abs_diff_eq!(ratio, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_plane_wave_3d() {
        let k = 1.0;
        let theta = 0.0; // Along z-axis
        let phi = 0.0;

        let points = vec![Point::new_3d(0.0, 0.0, 0.0), Point::new_3d(0.0, 0.0, 1.0)];

        let solution = plane_wave_3d(k, theta, phi, points);

        // |exp(ikz)| = 1
        for p in &solution.pressure {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }
    }

    // --- pulsating_sphere_3d ---

    #[test]
    fn test_pulsating_sphere_1_over_r_decay() {
        // Pressure magnitude of a monopole source must decay as 1/r.
        // h₀⁽¹⁾(kr) = -i·exp(ikr)/(kr), so |p(r)| ∝ 1/r.
        let k = 1.0;
        let a = 1.0;
        let sol = pulsating_sphere_3d(k, a, 1.0, 343.0, 1.21, vec![2.0, 4.0, 8.0]);

        let mags = sol.magnitude();
        // Ratio between consecutive doubles of r should be 0.5
        assert_abs_diff_eq!(mags[1] / mags[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(mags[2] / mags[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pulsating_sphere_surface_impedance() {
        // At r = a, the specific impedance Z/Z₀ = p / (ρ₀c₀ v₀).
        //
        // Derivation:
        //   p(a) = A · h₀⁽¹⁾(ka),  A = iρ₀c₀v₀ / h₀'(ka)
        //   Z/Z₀ = p(a)/(ρ₀c₀v₀) = i · h₀(ka) / h₀'(ka)
        //
        // With h₀(x) = -i·e^{ix}/x and h₀'(x) = -i·e^{ix}(ix-1)/x²:
        //   Z/Z₀ = ka / (ka + i)
        let k = 1.0;
        let a = 1.0;
        let v0 = 1.0;
        let rho = 1.21;
        let c0 = 343.0;
        let ka = k * a;

        // Surface pressure from the function
        let sol = pulsating_sphere_3d(k, a, v0, c0, rho, vec![a]);
        let p_surface = sol.pressure[0];

        // Correct impedance ratio: Z/Z₀ = ka / (ka + i)
        let z_ratio = Complex64::new(ka, 0.0) / Complex64::new(ka, 1.0);

        // p_surface = ρ₀c₀ v₀ · z_ratio
        let p_expected = rho * c0 * v0 * z_ratio;

        assert_abs_diff_eq!(p_surface.re, p_expected.re, epsilon = 1e-6);
        assert_abs_diff_eq!(p_surface.im, p_expected.im, epsilon = 1e-6);
    }

    #[test]
    fn checked_pulsating_sphere_enforces_exterior_domain_and_matches_legacy() {
        let checked = try_pulsating_sphere_3d(1.0, 0.5, 0.1, 343.0, 1.21, vec![0.5, 1.0])
            .expect("valid exterior reference must be accepted");
        let legacy = pulsating_sphere_3d(1.0, 0.5, 0.1, 343.0, 1.21, vec![0.5, 1.0]);
        assert_eq!(checked.pressure, legacy.pressure);

        assert!(matches!(
            try_pulsating_sphere_3d(0.0, 0.5, 0.1, 343.0, 1.21, vec![0.5]),
            Err(PulsatingSphereError::InvalidParameter {
                field: "wave_number",
                ..
            })
        ));
        assert!(matches!(
            try_pulsating_sphere_3d(1.0, 0.5, f64::NAN, 343.0, 1.21, vec![0.5]),
            Err(PulsatingSphereError::InvalidParameter {
                field: "velocity_amplitude",
                ..
            })
        ));
        assert!(matches!(
            try_pulsating_sphere_3d(1.0, 0.5, 0.1, 343.0, 1.21, vec![0.49]),
            Err(PulsatingSphereError::InteriorEvaluationPoint { index: 0, .. })
        ));
    }

    // --- oscillating_sphere_3d ---

    #[test]
    fn test_oscillating_sphere_cos_theta_pattern() {
        // Pressure must be proportional to cos(θ), so:
        // p(r, θ) / cos(θ) = constant for fixed r, varying θ.
        let k = 1.0;
        let a = 1.0;
        let r_eval = 3.0;
        let thetas = vec![0.1, 0.3, 0.7, 1.2]; // away from 0 and π/2

        let sol = oscillating_sphere_3d(k, a, 1.0, 343.0, 1.21, vec![r_eval], thetas.clone());

        let ratios: Vec<f64> = sol
            .pressure
            .iter()
            .zip(thetas.iter())
            .map(|(p, &th)| p.norm() / th.cos().abs())
            .collect();

        // All ratios should be equal
        for ratio in &ratios[1..] {
            assert_abs_diff_eq!(*ratio, ratios[0], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_oscillating_sphere_equatorial_null() {
        // At θ = π/2, cos(θ) = 0 → pressure must vanish (dipole null).
        let k = 1.0;
        let sol = oscillating_sphere_3d(k, 1.0, 1.0, 343.0, 1.21, vec![3.0], vec![PI / 2.0]);
        assert_abs_diff_eq!(sol.pressure[0].norm(), 0.0, epsilon = 1e-14);
    }

    // --- soft_sphere_scattering_3d ---

    #[test]
    fn test_soft_sphere_zero_pressure_on_surface() {
        // Dirichlet BC: p(r=a, θ) = 0 for all θ.
        let k = 1.0;
        let a = 1.0;

        let thetas: Vec<f64> = (0..=8).map(|i| i as f64 * PI / 8.0).collect();
        let sol = soft_sphere_scattering_3d(k, a, 30, vec![a], thetas);

        for (i, p) in sol.pressure.iter().enumerate() {
            assert!(
                p.norm() < 1e-10,
                "Pressure at θ[{}] = {:.4} should be zero on soft surface, got |p|={:.2e}",
                i,
                sol.positions[i].theta_3d(),
                p.norm()
            );
        }
    }

    #[test]
    fn test_soft_sphere_finite_far_field() {
        // Sanity: far-field pressures must be finite and non-trivially non-zero.
        let sol = soft_sphere_scattering_3d(1.0, 1.0, 25, vec![10.0], vec![0.0, PI / 2.0, PI]);
        for p in &sol.pressure {
            assert!(p.re.is_finite());
            assert!(p.im.is_finite());
            assert!(p.norm() > 0.0);
        }
    }

    // --- impedance_sphere_scattering_3d ---

    #[test]
    fn test_impedance_sphere_reduces_to_rigid_at_zero_admittance() {
        // β = 0 → impedance BC becomes Neumann → same result as rigid sphere.
        let k = 1.0;
        let a = 1.0;
        let n = 25;
        let r_pts = vec![1.5, 3.0];
        let th_pts = vec![0.0, PI / 3.0, PI];

        let rigid = sphere_scattering_3d(k, a, n, r_pts.clone(), th_pts.clone());
        let imp = impedance_sphere_scattering_3d(
            k,
            a,
            n,
            Complex64::new(0.0, 0.0), // β = 0
            r_pts,
            th_pts,
        );

        for (p_r, p_i) in rigid.pressure.iter().zip(imp.pressure.iter()) {
            assert_abs_diff_eq!(p_r.re, p_i.re, epsilon = 1e-10);
            assert_abs_diff_eq!(p_r.im, p_i.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_impedance_sphere_approaches_soft_at_large_admittance() {
        // β → ∞ means hₙ'(ka) - iβ·hₙ(ka) ≈ -iβ·hₙ(ka)  (denominator)
        //        and jₙ'(ka) - iβ·jₙ(ka) ≈ -iβ·jₙ(ka)   (numerator)
        // → aₙ ≈ jₙ(ka)/hₙ(ka), i.e., the Dirichlet coefficients.
        let k = 1.0;
        let a = 1.0;
        let n = 25;
        let r_pts = vec![1.5, 3.0];
        let th_pts = vec![0.0, PI / 3.0, PI];

        let soft = soft_sphere_scattering_3d(k, a, n, r_pts.clone(), th_pts.clone());
        // Very large β (purely real, >> 1)
        let imp = impedance_sphere_scattering_3d(k, a, n, Complex64::new(1e8, 0.0), r_pts, th_pts);

        for (p_s, p_i) in soft.pressure.iter().zip(imp.pressure.iter()) {
            assert_abs_diff_eq!(p_s.re, p_i.re, epsilon = 1e-4);
            assert_abs_diff_eq!(p_s.im, p_i.im, epsilon = 1e-4);
        }
    }

    // --- duct_modes_1d ---

    #[test]
    fn test_duct_mode_zero_is_uniform() {
        // n = 0: p(x) = cos(0) = 1 everywhere
        let sol = duct_modes_1d(0.0, 1.0, 0, 10);
        for p in &sol.pressure {
            assert_abs_diff_eq!(p.re, 1.0, epsilon = 1e-14);
            assert_abs_diff_eq!(p.im, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_duct_mode_neumann_boundary_conditions() {
        // For mode n, p(x) = cos(nπx/L).
        // ∂p/∂x = -nπ/L · sin(nπx/L) = 0 at x=0 and x=L (integer n).
        //
        // We use a second-order one-sided finite difference to avoid the
        // O(dx) truncation error of a simple forward difference:
        //   dp/dx|_{x=0} ≈ (-3p[0] + 4p[1] - p[2]) / (2dx)  (O(dx²))
        let n = 3;
        let l = 2.0;
        let num = 1001;
        let sol = duct_modes_1d(1.0, l, n, num);

        let dx = l / (num - 1) as f64;

        // Second-order forward difference at x=0
        let dp_left = (-3.0 * sol.pressure[0].re + 4.0 * sol.pressure[1].re - sol.pressure[2].re)
            / (2.0 * dx);
        assert!(dp_left.abs() < 1e-6, "dp/dx at x=0 = {:.2e}", dp_left);

        // Second-order backward difference at x=L
        let m = num - 1;
        let dp_right = (3.0 * sol.pressure[m].re - 4.0 * sol.pressure[m - 1].re
            + sol.pressure[m - 2].re)
            / (2.0 * dx);
        assert!(dp_right.abs() < 1e-6, "dp/dx at x=L = {:.2e}", dp_right);
    }

    #[test]
    fn test_duct_mode_node_positions() {
        // Mode n=2 has nodes (p=0) at x = L/4 and x = 3L/4.
        // cos(2πx/L) = 0  →  2πx/L = π/2, 3π/2  →  x = L/4, 3L/4.
        let n = 2;
        let l = 1.0;
        let num = 1001;
        let sol = duct_modes_1d(1.0, l, n, num);

        // Index for x = L/4 (i.e., i = 250 when num=1001)
        let i_quarter = num / 4; // 250
        let i_three_quarter = 3 * num / 4; // 750

        assert_abs_diff_eq!(sol.pressure[i_quarter].re, 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(sol.pressure[i_three_quarter].re, 0.0, epsilon = 1e-2);
    }
}
