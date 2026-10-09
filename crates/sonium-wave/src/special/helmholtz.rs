//! Helmholtz Green's function and derivatives
//!
//! The 3D Helmholtz Green's function is:
//! ```text
//! G(x, y) = exp(ik|x-y|) / (4π|x-y|)
//! ```
//!
//! This module provides computation of G and its normal derivatives,
//! which form the kernels for BEM integration.
//!
//! # Singularity policy
//!
//! Every kernel in this module comes in two variants:
//!
//! - Bare (`greens_function_3d`, `greens_function_gradient_3d`, …): for
//!   `r < 1e-15` these return `Complex64(INFINITY, 0)` (or `f64::INFINITY`
//!   for the Laplace kernels) instead of failing. That value propagates
//!   silently through an assembly, so never feed a bare kernel a
//!   coincident source/field pair.
//! - Checked (`try_greens_function_3d`, `try_all_kernels_3d`, …): return
//!   [`HelmholtzFunctionError::InvalidDistance`] for the same input.
//!
//! Prefer the `try_*` variants in all library and application code; reserve
//! the bare functions for hot loops where the caller has already established
//! `r >= 1e-15` (e.g. validated pair lists, analytic self-term handling).
//!
//! # Time convention
//!
//! All kernels use `exp(+ikr)`-style outgoing waves, i.e. outgoing under an
//! `exp(-iωt)` time dependence. Frequency-domain aeroacoustics in
//! `sonium-fem` uses `exp(+iωt)` with `exp(-ikr)` propagation; for real `k`
//! the two pictures differ by complex conjugation, provided here as
//! [`flip_time_convention`]. The trailing section of this module also holds
//! the 2-D Hankel derivatives and the uniform-flow (Garrick) convected
//! kernels shared by the 2-D, axisymmetric, and 3-D analogy evaluators.

use crate::Point;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Validation failures from the checked Helmholtz-kernel API.
#[derive(Debug, Clone, PartialEq)]
pub enum HelmholtzFunctionError {
    /// A radial argument was not finite or was not outside the singular core.
    InvalidDistance(f64),
    /// The wave number was not finite.
    InvalidWaveNumber(f64),
    /// A point or normal contained a non-finite component.
    NonFiniteInput(&'static str),
    /// A kernel evaluation produced a non-finite value.
    NonFiniteResult(&'static str),
    /// A Mach vector was non-finite or not strictly subsonic.
    InvalidMach(f64),
}

impl std::fmt::Display for HelmholtzFunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDistance(r) => {
                write!(
                    f,
                    "Helmholtz distance {r} must be finite and greater than 1e-15"
                )
            }
            Self::InvalidWaveNumber(k) => write!(f, "Helmholtz wave number {k} must be finite"),
            Self::NonFiniteInput(name) => write!(f, "Helmholtz input {name} must be finite"),
            Self::NonFiniteResult(name) => write!(f, "Helmholtz kernel {name} was non-finite"),
            Self::InvalidMach(mach) => write!(
                f,
                "Helmholtz Mach magnitude {mach} must be finite and below 1"
            ),
        }
    }
}

impl std::error::Error for HelmholtzFunctionError {}

fn validate_distance_and_wave_number(r: f64, k: f64) -> Result<(), HelmholtzFunctionError> {
    if !r.is_finite() || r <= 1e-15 {
        return Err(HelmholtzFunctionError::InvalidDistance(r));
    }
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    Ok(())
}

fn validate_point(point: &Point, name: &'static str) -> Result<(), HelmholtzFunctionError> {
    if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
        return Err(HelmholtzFunctionError::NonFiniteInput(name));
    }
    Ok(())
}

fn validate_normal(normal: &[f64; 3]) -> Result<(), HelmholtzFunctionError> {
    if normal.iter().any(|component| !component.is_finite()) {
        return Err(HelmholtzFunctionError::NonFiniteInput("normal"));
    }
    Ok(())
}

fn validate_complex_result(
    value: Complex64,
    name: &'static str,
) -> Result<Complex64, HelmholtzFunctionError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult(name))
    }
}

/// Checked 3D Helmholtz Green's function.
pub fn try_greens_function_3d(r: f64, k: f64) -> Result<Complex64, HelmholtzFunctionError> {
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(greens_function_3d(r, k), "G₃ᴅ")
}

/// Checked 2D Helmholtz Green's function.
pub fn try_greens_function_2d(r: f64, k: f64) -> Result<Complex64, HelmholtzFunctionError> {
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(greens_function_2d(r, k), "G₂ᴅ")
}

/// Checked gradient of the 3D Green's function with respect to the field point.
pub fn try_greens_function_gradient_3d(
    source: &Point,
    field: &Point,
    k: f64,
) -> Result<[Complex64; 3], HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let r = distance(source, field);
    validate_distance_and_wave_number(r, k)?;
    let value = greens_function_gradient_3d(source, field, k);
    if value
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult("gradient₃ᴅ"))
    }
}

/// Checked normal derivative of the 3D Green's function.
pub fn try_greens_function_normal_derivative_3d(
    source: &Point,
    field: &Point,
    normal: &[f64; 3],
    k: f64,
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    validate_normal(normal)?;
    let r = distance(source, field);
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(
        greens_function_normal_derivative_3d(source, field, normal, k),
        "normal derivative₃ᴅ",
    )
}

/// Checked adjoint normal derivative of the 3D Green's function.
pub fn try_greens_function_adjoint_derivative_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    k: f64,
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    validate_normal(normal_source)?;
    let r = distance(source, field);
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(
        greens_function_adjoint_derivative_3d(source, field, normal_source, k),
        "adjoint derivative₃ᴅ",
    )
}

/// Checked hypersingular kernel of the 3D Green's function.
pub fn try_greens_function_hypersingular_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    normal_field: &[f64; 3],
    k: f64,
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    validate_normal(normal_source)?;
    validate_normal(normal_field)?;
    let r = distance(source, field);
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(
        greens_function_hypersingular_3d(source, field, normal_source, normal_field, k),
        "hypersingular₃ᴅ",
    )
}

/// Checked evaluation of all four 3-D BEM kernels at once.
pub fn try_all_kernels_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    normal_field: &[f64; 3],
    k: f64,
) -> Result<(Complex64, Complex64, Complex64, Complex64), HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    validate_normal(normal_source)?;
    validate_normal(normal_field)?;
    let r = distance(source, field);
    validate_distance_and_wave_number(r, k)?;
    let values = all_kernels_3d(source, field, normal_source, normal_field, k);
    if [values.0, values.1, values.2, values.3]
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        Ok(values)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult("all kernels₃ᴅ"))
    }
}

/// Checked 3-D Laplace Green's function (the `k = 0` limit).
pub fn try_laplace_greens_function_3d(r: f64) -> Result<f64, HelmholtzFunctionError> {
    validate_distance_and_wave_number(r, 0.0)?;
    Ok(laplace_greens_function_3d(r))
}

/// Checked 2-D Laplace Green's function (the `k = 0` limit).
pub fn try_laplace_greens_function_2d(r: f64) -> Result<f64, HelmholtzFunctionError> {
    validate_distance_and_wave_number(r, 0.0)?;
    Ok(laplace_greens_function_2d(r))
}

/// 3D Helmholtz Green's function G = exp(ikr)/(4πr)
///
/// # Arguments
/// * `r` - Distance |x - y|
/// * `k` - Wave number
///
/// # Returns
/// Complex value exp(ikr)/(4πr)
///
/// # Example
/// ```
/// use sonium_wave::special::helmholtz::greens_function_3d;
///
/// let g = greens_function_3d(1.0, 2.0);
/// // |G| = 1/(4.0 * std::f64::consts::PI) for any k
/// assert!((g.norm() - 1.0/(4.0 * std::f64::consts::PI)).abs() < 1e-10);
/// ```
#[inline]
pub fn greens_function_3d(r: f64, k: f64) -> Complex64 {
    if r < 1e-15 {
        // Singular at r = 0
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let kr = k * r;
    let exp_ikr = Complex64::new(kr.cos(), kr.sin());
    exp_ikr / (4.0 * PI * r)
}

/// 2D Helmholtz Green's function G = (i/4) H_0^(1)(kr)
///
/// Uses the Hankel function of the first kind.
///
/// # Arguments
/// * `r` - Distance |x - y|
/// * `k` - Wave number
#[inline]
pub fn greens_function_2d(r: f64, k: f64) -> Complex64 {
    use spec_math::Bessel;

    if r < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let kr = k * r;
    let j0 = kr.bessel_jv(0.0);
    let y0 = kr.bessel_yv(0.0);
    let h0 = Complex64::new(j0, y0);

    Complex64::new(0.0, 0.25) * h0
}

/// Gradient of 3D Green's function ∇_y G
///
/// ```text
/// ∇_y G = (ik - 1/r) * G * (y - x)/r
/// ```
///
/// # Arguments
/// * `source` - Source point x
/// * `field` - Field point y
/// * `k` - Wave number
///
/// # Returns
/// Complex 3-vector [∂G/∂y_x, ∂G/∂y_y, ∂G/∂y_z]
pub fn greens_function_gradient_3d(source: &Point, field: &Point, k: f64) -> [Complex64; 3] {
    let rx = field.x - source.x;
    let ry = field.y - source.y;
    let rz = field.z - source.z;
    let r = (rx * rx + ry * ry + rz * rz).sqrt();

    if r < 1e-15 {
        return [
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
        ];
    }

    let g = greens_function_3d(r, k);
    let factor = Complex64::new(-1.0 / r, k) * g;

    [factor * rx / r, factor * ry / r, factor * rz / r]
}

/// Normal derivative of 3D Green's function ∂G/∂n_y
///
/// ```text
/// ∂G/∂n_y = ∇_y G · n_y = (ik - 1/r) * G * (y-x)·n_y / r
/// ```
///
/// This is the kernel for the single-layer potential.
///
/// # Arguments
/// * `source` - Source point x
/// * `field` - Field point y
/// * `normal` - Unit normal at field point (n_x, n_y, n_z)
/// * `k` - Wave number
#[inline]
pub fn greens_function_normal_derivative_3d(
    source: &Point,
    field: &Point,
    normal: &[f64; 3],
    k: f64,
) -> Complex64 {
    let rx = field.x - source.x;
    let ry = field.y - source.y;
    let rz = field.z - source.z;
    let r = (rx * rx + ry * ry + rz * rz).sqrt();

    if r < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let g = greens_function_3d(r, k);
    let r_dot_n = rx * normal[0] + ry * normal[1] + rz * normal[2];

    // Factor: (ik - 1/r)
    let factor = Complex64::new(-1.0 / r, k);

    factor * g * r_dot_n / r
}

/// Adjoint double layer kernel ∂G/∂n_x
///
/// ```text
/// ∂G/∂n_x = -∇_y G · n_x = -(ik - 1/r) * G * (y-x)·n_x / r
/// ```
///
/// This is the kernel for the adjoint double-layer potential.
#[inline]
pub fn greens_function_adjoint_derivative_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    k: f64,
) -> Complex64 {
    let rx = field.x - source.x;
    let ry = field.y - source.y;
    let rz = field.z - source.z;
    let r = (rx * rx + ry * ry + rz * rz).sqrt();

    if r < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let g = greens_function_3d(r, k);
    let r_dot_n = rx * normal_source[0] + ry * normal_source[1] + rz * normal_source[2];

    // Adjoint has opposite sign
    let factor = Complex64::new(1.0 / r, -k);

    factor * g * r_dot_n / r
}

/// Hypersingular kernel ∂²G/(∂n_x ∂n_y)
///
/// ```text
/// ∂²G/(∂n_x ∂n_y) = [((ik)² - 3ik/r + 3/r²) (r·n_x)(r·n_y)/r²
///                    - (ik - 1/r)(n_x·n_y)/r] * G
/// ```
pub fn greens_function_hypersingular_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    normal_field: &[f64; 3],
    k: f64,
) -> Complex64 {
    let rx = field.x - source.x;
    let ry = field.y - source.y;
    let rz = field.z - source.z;
    let r2 = rx * rx + ry * ry + rz * rz;
    let r = r2.sqrt();

    if r < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let g = greens_function_3d(r, k);
    let ik = Complex64::new(0.0, k);

    let r_dot_nx = rx * normal_source[0] + ry * normal_source[1] + rz * normal_source[2];
    let r_dot_ny = rx * normal_field[0] + ry * normal_field[1] + rz * normal_field[2];
    let nx_dot_ny = normal_source[0] * normal_field[0]
        + normal_source[1] * normal_field[1]
        + normal_source[2] * normal_field[2];

    // Term 1: ((ik)² - 3ik/r + 3/r²) (r·n_x)(r·n_y)/r² * G
    let coef1 = ik * ik - 3.0 * ik / r + Complex64::new(3.0 / r2, 0.0);
    let term1 = coef1 * r_dot_nx * r_dot_ny / r2;

    // Term 2: (ik - 1/r)(n_x·n_y)/r * G
    let coef2 = ik - Complex64::new(1.0 / r, 0.0);
    let term2 = coef2 * nx_dot_ny / r;

    (term1 - term2) * g
}

/// Compute all four BEM kernels at once for efficiency
///
/// Returns (G, ∂G/∂n_y, ∂G/∂n_x, ∂²G/∂n_x∂n_y)
pub fn all_kernels_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    normal_field: &[f64; 3],
    k: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let rx = field.x - source.x;
    let ry = field.y - source.y;
    let rz = field.z - source.z;
    let r2 = rx * rx + ry * ry + rz * rz;
    let r = r2.sqrt();

    if r < 1e-15 {
        let inf = Complex64::new(f64::INFINITY, 0.0);
        return (inf, inf, inf, inf);
    }

    let kr = k * r;
    let exp_ikr = Complex64::new(kr.cos(), kr.sin());
    let g = exp_ikr / (4.0 * PI * r);

    let r_dot_nx = rx * normal_source[0] + ry * normal_source[1] + rz * normal_source[2];
    let r_dot_ny = rx * normal_field[0] + ry * normal_field[1] + rz * normal_field[2];
    let nx_dot_ny = normal_source[0] * normal_field[0]
        + normal_source[1] * normal_field[1]
        + normal_source[2] * normal_field[2];

    let ik = Complex64::new(0.0, k);

    // ∂G/∂n_y
    let factor_dg = ik - Complex64::new(1.0 / r, 0.0);
    let dg_dny = factor_dg * g * r_dot_ny / r;

    // ∂G/∂n_x (adjoint - opposite sign)
    let dg_dnx = -factor_dg * g * r_dot_nx / r;

    // Hypersingular
    let coef1 = ik * ik - 3.0 * ik / r + Complex64::new(3.0 / r2, 0.0);
    let term1 = coef1 * r_dot_nx * r_dot_ny / r2;
    let term2 = factor_dg * nx_dot_ny / r;
    let d2g = (term1 - term2) * g;

    (g, dg_dny, dg_dnx, d2g)
}

/// Distance between two points
#[inline]
pub fn distance(p1: &Point, p2: &Point) -> f64 {
    p1.distance_to(p2)
}

/// Laplace Green's function (k=0 limit): G = 1/(4πr)
#[inline]
pub fn laplace_greens_function_3d(r: f64) -> f64 {
    if r < 1e-15 {
        f64::INFINITY
    } else {
        1.0 / (4.0 * PI * r)
    }
}

/// 2D Laplace Green's function: G = -ln(r)/(2π)
#[inline]
pub fn laplace_greens_function_2d(r: f64) -> f64 {
    if r < 1e-15 {
        f64::INFINITY
    } else {
        -r.ln() / (2.0 * PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_greens_function_magnitude() {
        // |G| = 1/(4πr) for any k
        let r = 2.0;
        let k = 1.5;
        let g = greens_function_3d(r, k);

        let expected_magnitude = 1.0 / (4.0 * PI * r);
        assert!((g.norm() - expected_magnitude).abs() < EPSILON);
    }

    #[test]
    fn test_greens_function_k_zero() {
        // For k = 0, G = 1/(4πr) (Laplace Green's function)
        let r = 1.5;
        let g = greens_function_3d(r, 0.0);

        let expected = 1.0 / (4.0 * PI * r);
        assert!((g.re - expected).abs() < EPSILON);
        assert!(g.im.abs() < EPSILON);
    }

    #[test]
    fn test_normal_derivative_radial() {
        // When y-x is parallel to n_y (radial direction)
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.0, 0.0);
        let normal = [1.0, 0.0, 0.0]; // Outward normal

        let k = 2.0;
        let dg_dn = greens_function_normal_derivative_3d(&source, &field, &normal, k);

        // Should be finite
        assert!(dg_dn.norm().is_finite());
    }

    #[test]
    fn test_normal_derivative_tangential() {
        // When y-x is perpendicular to n_y (tangential)
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.0, 0.0);
        let normal = [0.0, 1.0, 0.0]; // Tangent to radial

        let k = 2.0;
        let dg_dn = greens_function_normal_derivative_3d(&source, &field, &normal, k);

        // Should be zero since (y-x)·n = 0
        assert!(dg_dn.norm() < EPSILON);
    }

    #[test]
    fn test_all_kernels_consistency() {
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.5, 0.3);
        let n_source = [0.0, 0.0, 1.0];
        let n_field = [1.0, 0.0, 0.0];
        let k = 2.5;

        let (g, dg_dny, dg_dnx, _d2g) = all_kernels_3d(&source, &field, &n_source, &n_field, k);

        // Compare with individual functions
        let g_single = greens_function_3d(distance(&source, &field), k);
        let dg_dny_single = greens_function_normal_derivative_3d(&source, &field, &n_field, k);
        let dg_dnx_single = greens_function_adjoint_derivative_3d(&source, &field, &n_source, k);

        assert!((g - g_single).norm() < EPSILON);
        assert!((dg_dny - dg_dny_single).norm() < EPSILON);
        assert!((dg_dnx - dg_dnx_single).norm() < EPSILON);
    }

    #[test]
    fn test_laplace_greens() {
        let r = 2.0;

        // 3D: G = 1/(4πr)
        let g3d = laplace_greens_function_3d(r);
        assert!((g3d - 1.0 / (4.0 * PI * r)).abs() < EPSILON);

        // 2D: G = -ln(r)/(2π)
        let g2d = laplace_greens_function_2d(r);
        assert!((g2d - (-r.ln() / (2.0 * PI))).abs() < EPSILON);
    }

    #[test]
    fn test_gradient() {
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.0, 0.0);
        let k = 1.0;

        let grad = greens_function_gradient_3d(&source, &field, k);

        // Gradient should point along (field - source) direction
        // Only x-component should be non-zero
        assert!(grad[0].norm() > EPSILON);
        assert!(grad[1].norm() < EPSILON);
        assert!(grad[2].norm() < EPSILON);
    }

    #[test]
    fn bare_kernels_return_documented_infinity_at_coincidence() {
        // Locks the singularity policy: bare kernels return INF (never NaN,
        // never a finite hallucination) at r = 0, and the `try_*` twins
        // reject the same input. Callers must not feed bare kernels
        // coincident pairs.
        let inf = Complex64::new(f64::INFINITY, 0.0);
        assert_eq!(greens_function_3d(0.0, 1.0), inf);
        assert_eq!(greens_function_2d(0.0, 1.0), inf);
        let p = Point::new_3d(0.0, 0.0, 0.0);
        let n = [0.0, 0.0, 1.0];
        assert_eq!(greens_function_gradient_3d(&p, &p, 1.0), [inf, inf, inf]);
        assert_eq!(greens_function_normal_derivative_3d(&p, &p, &n, 1.0), inf);
        assert_eq!(all_kernels_3d(&p, &p, &n, &n, 1.0), (inf, inf, inf, inf));
        assert_eq!(laplace_greens_function_3d(0.0), f64::INFINITY);
        assert_eq!(laplace_greens_function_2d(0.0), f64::INFINITY);
    }

    #[test]
    fn all_kernels_adjoint_consistency_for_shared_normals() {
        // For a shared normal, ∂G/∂n_x = -∂G/∂n_y by antisymmetry of the
        // gradient kernel; `all_kernels_3d` must reproduce that relation.
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.5, 0.25);
        let normal = [0.0, 0.0, 1.0];
        let (g, dg_dny, dg_dnx, _) = all_kernels_3d(&source, &field, &normal, &normal, 2.0);
        assert!(g.re.is_finite() && g.im.is_finite());
        assert!((dg_dnx + dg_dny).norm() < 1e-15);
    }

    #[test]
    fn checked_kernels_reject_singular_nonfinite_and_coincident_inputs() {
        assert_eq!(
            try_greens_function_3d(0.0, 1.0),
            Err(HelmholtzFunctionError::InvalidDistance(0.0))
        );
        assert!(matches!(
            try_greens_function_2d(1.0, f64::NAN),
            Err(HelmholtzFunctionError::InvalidWaveNumber(value)) if value.is_nan()
        ));

        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(0.0, 0.0, 0.0);
        let normal = [1.0, 0.0, 0.0];
        assert!(matches!(
            try_greens_function_gradient_3d(&source, &field, 1.0),
            Err(HelmholtzFunctionError::InvalidDistance(_))
        ));
        assert!(matches!(
            try_greens_function_normal_derivative_3d(&source, &field, &normal, 1.0),
            Err(HelmholtzFunctionError::InvalidDistance(_))
        ));
    }

    #[test]
    fn checked_kernels_match_legacy_values_for_valid_inputs() {
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.0, 0.5, 0.25);
        let normal = [0.0, 0.0, 1.0];
        let r = distance(&source, &field);
        let k = 2.0;

        assert_eq!(
            try_greens_function_3d(r, k).unwrap(),
            greens_function_3d(r, k)
        );
        assert_eq!(
            try_greens_function_2d(r, k).unwrap(),
            greens_function_2d(r, k)
        );
        assert_eq!(
            try_greens_function_gradient_3d(&source, &field, k).unwrap(),
            greens_function_gradient_3d(&source, &field, k)
        );
        assert_eq!(
            try_greens_function_normal_derivative_3d(&source, &field, &normal, k).unwrap(),
            greens_function_normal_derivative_3d(&source, &field, &normal, k)
        );

        let normal_source = [1.0, 0.0, 0.0];
        assert_eq!(
            try_greens_function_adjoint_derivative_3d(&source, &field, &normal_source, k).unwrap(),
            greens_function_adjoint_derivative_3d(&source, &field, &normal_source, k)
        );
        assert_eq!(
            try_greens_function_hypersingular_3d(&source, &field, &normal_source, &normal, k)
                .unwrap(),
            greens_function_hypersingular_3d(&source, &field, &normal_source, &normal, k)
        );
        assert_eq!(
            try_all_kernels_3d(&source, &field, &normal_source, &normal, k).unwrap(),
            all_kernels_3d(&source, &field, &normal_source, &normal, k)
        );
        assert_eq!(
            try_laplace_greens_function_3d(r).unwrap(),
            laplace_greens_function_3d(r)
        );
        assert_eq!(
            try_laplace_greens_function_2d(r).unwrap(),
            laplace_greens_function_2d(r)
        );

        let invalid_normal = [f64::INFINITY, 0.0, 0.0];
        assert_eq!(
            try_all_kernels_3d(&source, &field, &invalid_normal, &normal, k),
            Err(HelmholtzFunctionError::NonFiniteInput("normal"))
        );
    }
}

// ============================================================================
// Two-dimensional derivatives, uniform-flow convected kernels, and the
// time-convention adapter shared by the aeroacoustic analogy evaluators.
// ============================================================================

/// Maps kernels between the two time conventions.
///
/// For real `k`, outgoing waves under `exp(-iωt)` (this module) are the
/// complex conjugates of outgoing waves under `exp(+iωt)` (the `sonium-fem`
/// aeroacoustics convention), so the map is an involution.
#[inline]
pub fn flip_time_convention(value: Complex64) -> Complex64 {
    value.conj()
}

fn validate_point_2d(point: &[f64; 2], name: &'static str) -> Result<(), HelmholtzFunctionError> {
    if !point[0].is_finite() || !point[1].is_finite() {
        return Err(HelmholtzFunctionError::NonFiniteInput(name));
    }
    Ok(())
}

fn validate_normal_2d(normal: &[f64; 2]) -> Result<(), HelmholtzFunctionError> {
    if normal.iter().any(|component| !component.is_finite()) {
        return Err(HelmholtzFunctionError::NonFiniteInput("normal"));
    }
    Ok(())
}

fn hankel_first_kind(order: f64, argument: f64) -> Complex64 {
    use spec_math::Bessel as _;
    Complex64::new(argument.bessel_jv(order), argument.bessel_yv(order))
}

/// Gradient of the 2-D Green function w.r.t. the field point.
///
/// Returns `[∂G/∂y₀, ∂G/∂y₁]` with `R = field - source`, using
/// `dG/dr = -(ik/4) H₁⁽¹⁾(kr)` and the chain rule.
pub fn greens_function_gradient_2d(source: &[f64; 2], field: &[f64; 2], k: f64) -> [Complex64; 2] {
    let rx = field[0] - source[0];
    let ry = field[1] - source[1];
    let r = (rx * rx + ry * ry).sqrt();
    if r < 1e-15 {
        let inf = Complex64::new(f64::INFINITY, 0.0);
        return [inf, inf];
    }
    let h1 = hankel_first_kind(1.0, k * r);
    let radial = Complex64::new(0.0, -k * 0.25) * h1;
    [radial * rx / r, radial * ry / r]
}

/// Checked gradient of the 2-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// source/field pair, a non-finite wave number, or a non-finite result.
pub fn try_greens_function_gradient_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
) -> Result<[Complex64; 2], HelmholtzFunctionError> {
    validate_point_2d(source, "source")?;
    validate_point_2d(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let r = distance_2d(source, field);
    validate_distance_and_wave_number(r, k)?;
    let value = greens_function_gradient_2d(source, field, k);
    if value
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult("gradient₂ᴅ"))
    }
}

/// Normal derivative of the 2-D Green function.
///
/// Computes `∇_y G · normal` for a caller-supplied (not necessarily unit)
/// normal at the field point.
pub fn greens_function_normal_derivative_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    normal: &[f64; 2],
    k: f64,
) -> Complex64 {
    let gradient = greens_function_gradient_2d(source, field, k);
    gradient[0] * normal[0] + gradient[1] * normal[1]
}

/// Checked normal derivative of the 2-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// source/field pair, a non-finite wave number, or a non-finite result.
pub fn try_greens_function_normal_derivative_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    normal: &[f64; 2],
    k: f64,
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point_2d(source, "source")?;
    validate_point_2d(field, "field")?;
    validate_normal_2d(normal)?;
    let r = distance_2d(source, field);
    validate_distance_and_wave_number(r, k)?;
    validate_complex_result(
        greens_function_normal_derivative_2d(source, field, normal, k),
        "normal derivative₂ᴅ",
    )
}

/// Hessian of the 2-D Green function w.r.t. the field point.
///
/// Uses `H = G'' n⊗n + (G'/r)(I - n⊗n)` with the Hankel radial derivatives
/// `G' = -(ik/4)H₁(kr)` and `G'' = -k²G + (ik/4r)H₁(kr)`. The trace
/// satisfies `tr(H) = -k²G` away from the source.
pub fn greens_function_hessian_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
) -> [[Complex64; 2]; 2] {
    let rx = field[0] - source[0];
    let ry = field[1] - source[1];
    let r = (rx * rx + ry * ry).sqrt();
    if r < 1e-15 {
        let inf = Complex64::new(f64::INFINITY, 0.0);
        return [[inf, inf], [inf, inf]];
    }
    let g = greens_function_2d(r, k);
    let h1 = hankel_first_kind(1.0, k * r);
    let radial_over_r = Complex64::new(0.0, -k * 0.25 / r) * h1;
    let second = radial_over_r * Complex64::new(-1.0, 0.0) - g * (k * k);
    let nx = rx / r;
    let ny = ry / r;
    [
        [
            second * nx * nx + radial_over_r * (1.0 - nx * nx),
            (second - radial_over_r) * nx * ny,
        ],
        [
            (second - radial_over_r) * nx * ny,
            second * ny * ny + radial_over_r * (1.0 - ny * ny),
        ],
    ]
}

/// Checked Hessian of the 2-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// source/field pair, a non-finite wave number, or a non-finite result.
pub fn try_greens_function_hessian_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
) -> Result<[[Complex64; 2]; 2], HelmholtzFunctionError> {
    validate_point_2d(source, "source")?;
    validate_point_2d(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let r = distance_2d(source, field);
    validate_distance_and_wave_number(r, k)?;
    let value = greens_function_hessian_2d(source, field, k);
    if value
        .iter()
        .flatten()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult("hessian₂ᴅ"))
    }
}

/// Euclidean distance between two planar points.
#[inline]
pub fn distance_2d(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}

fn validate_mach(mach: &[f64]) -> Result<(f64, f64), HelmholtzFunctionError> {
    if mach.iter().any(|component| !component.is_finite()) {
        return Err(HelmholtzFunctionError::NonFiniteInput("mach"));
    }
    let magnitude = mach
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if !(0.0..1.0).contains(&magnitude) {
        return Err(HelmholtzFunctionError::InvalidMach(magnitude));
    }
    Ok((magnitude, 1.0 - magnitude * magnitude))
}

/// Garrick effective distance for uniform flow.
///
/// With `s = M·R` and `β² = 1 - |M|²`, the effective distance is
/// `Σ² = s² + β²|R|²` (axial flow: `Δx² + β²r⊥²`), which reduces to `|R|`
/// for zero flow. The convected phase is `k(Σ − s)/β²`, giving the exact
/// travel-time phases on the flow axis.
fn garrick_sigma(r_vec: &[f64], mach: &[f64], beta_sq: f64) -> (f64, f64) {
    let s: f64 = mach.iter().zip(r_vec.iter()).map(|(m, r)| m * r).sum();
    let r_sq: f64 = r_vec.iter().map(|r| r * r).sum();
    let sigma = (s * s + beta_sq * r_sq).sqrt();
    (sigma, s)
}

/// Gradient of the Garrick distance w.r.t. `R = field - source`.
fn garrick_sigma_gradient(
    r_vec: &[f64],
    mach: &[f64],
    beta_sq: f64,
    sigma: f64,
    s: f64,
    out: &mut [f64],
) {
    for ((gradient, r), m) in out.iter_mut().zip(r_vec.iter()).zip(mach.iter()) {
        *gradient = (s * m + beta_sq * r) / sigma;
    }
}

/// Uniform-flow 3-D convected Green function.
///
/// Evaluates `G = exp(iΦ)/(4πσ)` with Garrick distance `σ` and phase
/// `Φ = k(σ − M·R)/β²`. On the flow axis this gives the travel-time
/// phases `kR/(1+M)` downstream and `kR/(1−M)` upstream. Reduces to
/// [`greens_function_3d`] at zero Mach.
pub fn convected_greens_function_3d(
    source: &Point,
    field: &Point,
    k: f64,
    mach: &[f64; 3],
) -> Complex64 {
    let r_vec = [field.x - source.x, field.y - source.y, field.z - source.z];
    let beta_sq = 1.0 - mach.iter().map(|m| m * m).sum::<f64>();
    let (sigma, s) = garrick_sigma(&r_vec, mach, beta_sq);
    if sigma < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    let phase = k * (sigma - s) / beta_sq;
    Complex64::from_polar(1.0 / (4.0 * PI * sigma), phase)
}

/// Checked uniform-flow 3-D convected Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// pair, a non-finite wave number, a non-subsonic Mach vector, or a
/// non-finite result.
pub fn try_convected_greens_function_3d(
    source: &Point,
    field: &Point,
    k: f64,
    mach: &[f64; 3],
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let (_, beta_sq) = validate_mach(mach)?;
    let r_vec = [field.x - source.x, field.y - source.y, field.z - source.z];
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    validate_distance_and_wave_number(sigma, k)?;
    validate_complex_result(
        convected_greens_function_3d(source, field, k, mach),
        "convected G₃ᴅ",
    )
}

/// Gradient of the convected 3-D Green function.
///
/// Returns `∇_y G` (derivatives w.r.t. the field point) via the chain rule
/// through the Garrick distance and the convected phase.
pub fn convected_greens_function_gradient_3d(
    source: &Point,
    field: &Point,
    k: f64,
    mach: &[f64; 3],
) -> [Complex64; 3] {
    let r_vec = [field.x - source.x, field.y - source.y, field.z - source.z];
    let beta_sq = 1.0 - mach.iter().map(|m| m * m).sum::<f64>();
    let (sigma, s) = garrick_sigma(&r_vec, mach, beta_sq);
    if sigma < 1e-15 {
        let inf = Complex64::new(f64::INFINITY, 0.0);
        return [inf, inf, inf];
    }
    let g = convected_greens_function_3d(source, field, k, mach);
    let mut sigma_grad = [0.0; 3];
    garrick_sigma_gradient(&r_vec, mach, beta_sq, sigma, s, &mut sigma_grad);
    std::array::from_fn(|j| {
        let phase_grad = k * (sigma_grad[j] - mach[j]) / beta_sq;
        g * (Complex64::new(0.0, phase_grad) - Complex64::new(sigma_grad[j] / sigma, 0.0))
    })
}

/// Checked gradient of the convected 3-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// pair, a non-finite wave number, a non-subsonic Mach vector, or a
/// non-finite result.
pub fn try_convected_greens_function_gradient_3d(
    source: &Point,
    field: &Point,
    k: f64,
    mach: &[f64; 3],
) -> Result<[Complex64; 3], HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let (_, beta_sq) = validate_mach(mach)?;
    let r_vec = [field.x - source.x, field.y - source.y, field.z - source.z];
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    validate_distance_and_wave_number(sigma, k)?;
    let value = convected_greens_function_gradient_3d(source, field, k, mach);
    if value
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult(
            "convected gradient₃ᴅ",
        ))
    }
}

/// Source normal derivative of the convected 3-D Green function.
///
/// Computes `∇_x G · normal` (derivatives w.r.t. the source point), the
/// loading kernel for uniform-flow FW-H surfaces. Uniform flow preserves
/// translation invariance, so this equals the negated field derivative;
/// the separate entry point keeps FW-H call sites readable.
pub fn convected_greens_function_source_normal_derivative_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    k: f64,
    mach: &[f64; 3],
) -> Complex64 {
    let gradient = convected_greens_function_gradient_3d(source, field, k, mach);
    -(gradient[0] * normal_source[0]
        + gradient[1] * normal_source[1]
        + gradient[2] * normal_source[2])
}

/// Checked source normal derivative of the convected 3-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// pair, a non-finite wave number, a non-subsonic Mach vector, or a
/// non-finite result.
pub fn try_convected_greens_function_source_normal_derivative_3d(
    source: &Point,
    field: &Point,
    normal_source: &[f64; 3],
    k: f64,
    mach: &[f64; 3],
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point(source, "source")?;
    validate_point(field, "field")?;
    validate_normal(normal_source)?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let (_, beta_sq) = validate_mach(mach)?;
    let r_vec = [field.x - source.x, field.y - source.y, field.z - source.z];
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    validate_distance_and_wave_number(sigma, k)?;
    validate_complex_result(
        convected_greens_function_source_normal_derivative_3d(
            source,
            field,
            normal_source,
            k,
            mach,
        ),
        "convected source normal derivative₃ᴅ",
    )
}

/// Uniform-flow 2-D convected Green function.
///
/// Evaluates `G = (i/4β) exp(−ikM·R/β²) H₀⁽¹⁾(kσ/β²)` with the planar
/// Garrick distance `σ`. The Doppler sign gives the travel-time phases
/// `kR/(1+M)` downstream and `kR/(1−M)` upstream on the flow axis.
/// Reduces to [`greens_function_2d`] at zero Mach.
pub fn convected_greens_function_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
    mach: &[f64; 2],
) -> Complex64 {
    let r_vec = [field[0] - source[0], field[1] - source[1]];
    let beta_sq = 1.0 - mach.iter().map(|m| m * m).sum::<f64>();
    let beta = beta_sq.sqrt();
    let (sigma, s) = garrick_sigma(&r_vec, mach, beta_sq);
    if sigma < 1e-15 {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    let hankel = hankel_first_kind(0.0, k * sigma / beta_sq);
    let doppler_phase = Complex64::new(0.0, -k * s / beta_sq).exp();
    Complex64::new(0.0, 0.25 / beta) * doppler_phase * hankel
}

/// Checked uniform-flow 2-D convected Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// pair, a non-finite wave number, a non-subsonic Mach vector, or a
/// non-finite result.
pub fn try_convected_greens_function_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
    mach: &[f64; 2],
) -> Result<Complex64, HelmholtzFunctionError> {
    validate_point_2d(source, "source")?;
    validate_point_2d(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let (_, beta_sq) = validate_mach(mach)?;
    let r_vec = [field[0] - source[0], field[1] - source[1]];
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    validate_distance_and_wave_number(sigma, k)?;
    validate_complex_result(
        convected_greens_function_2d(source, field, k, mach),
        "convected G₂ᴅ",
    )
}

/// Gradient of the convected 2-D Green function.
///
/// Returns `∇_y G` via the chain rule through the Doppler phase and the
/// Hankel argument, using `H₀'/H₀ = -H₁/H₀`.
pub fn convected_greens_function_gradient_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
    mach: &[f64; 2],
) -> [Complex64; 2] {
    let r_vec = [field[0] - source[0], field[1] - source[1]];
    let beta_sq = 1.0 - mach.iter().map(|m| m * m).sum::<f64>();
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    if sigma < 1e-15 {
        let inf = Complex64::new(f64::INFINITY, 0.0);
        return [inf, inf];
    }
    let g = convected_greens_function_2d(source, field, k, mach);
    let argument = k * sigma / beta_sq;
    let hankel_ratio = hankel_first_kind(1.0, argument) / hankel_first_kind(0.0, argument);
    let mut sigma_grad = [0.0; 2];
    let s: f64 = mach.iter().zip(r_vec.iter()).map(|(m, r)| m * r).sum();
    garrick_sigma_gradient(&r_vec, mach, beta_sq, sigma, s, &mut sigma_grad);
    std::array::from_fn(|j| {
        g * (Complex64::new(0.0, -k * mach[j] / beta_sq)
            - hankel_ratio * (k * sigma_grad[j] / beta_sq))
    })
}

/// Checked gradient of the convected 2-D Green function.
///
/// # Errors
///
/// Returns [`HelmholtzFunctionError`] for non-finite inputs, a coincident
/// pair, a non-finite wave number, a non-subsonic Mach vector, or a
/// non-finite result.
pub fn try_convected_greens_function_gradient_2d(
    source: &[f64; 2],
    field: &[f64; 2],
    k: f64,
    mach: &[f64; 2],
) -> Result<[Complex64; 2], HelmholtzFunctionError> {
    validate_point_2d(source, "source")?;
    validate_point_2d(field, "field")?;
    if !k.is_finite() {
        return Err(HelmholtzFunctionError::InvalidWaveNumber(k));
    }
    let (_, beta_sq) = validate_mach(mach)?;
    let r_vec = [field[0] - source[0], field[1] - source[1]];
    let (sigma, _) = garrick_sigma(&r_vec, mach, beta_sq);
    validate_distance_and_wave_number(sigma, k)?;
    let value = convected_greens_function_gradient_2d(source, field, k, mach);
    if value
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
    {
        Ok(value)
    } else {
        Err(HelmholtzFunctionError::NonFiniteResult(
            "convected gradient₂ᴅ",
        ))
    }
}

#[cfg(test)]
mod aero_kernel_tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn central_diff_2d(
        f: impl Fn(&[f64; 2]) -> Complex64,
        base: &[f64; 2],
        axis: usize,
        h: f64,
    ) -> Complex64 {
        let mut plus = *base;
        let mut minus = *base;
        plus[axis] += h;
        minus[axis] -= h;
        (f(&plus) - f(&minus)) / (2.0 * h)
    }

    fn central_diff_3d(
        f: impl Fn(&Point) -> Complex64,
        base: &Point,
        axis: usize,
        h: f64,
    ) -> Complex64 {
        let mut plus = [base.x, base.y, base.z];
        let mut minus = plus;
        plus[axis] += h;
        minus[axis] -= h;
        let plus_point = Point::new_3d(plus[0], plus[1], plus[2]);
        let minus_point = Point::new_3d(minus[0], minus[1], minus[2]);
        (f(&plus_point) - f(&minus_point)) / (2.0 * h)
    }

    #[test]
    fn time_convention_flip_is_involution_to_outgoing_aero_kernel() {
        let g = greens_function_3d(2.5, 1.2);
        let flipped = flip_time_convention(g);
        let expected = Complex64::from_polar(1.0 / (4.0 * PI * 2.5), -1.2 * 2.5);
        assert!((flipped - expected).norm() < TOL);
        assert!((flip_time_convention(flipped) - g).norm() < TOL);
    }

    #[test]
    fn gradient_2d_matches_finite_difference() {
        let source = [0.2, -0.3];
        let field = [1.1, 0.7];
        let gradient = greens_function_gradient_2d(&source, &field, 2.0);
        for axis in 0..2 {
            let numeric = central_diff_2d(
                |y| greens_function_2d(distance_2d(&source, y), 2.0),
                &field,
                axis,
                1e-6,
            );
            assert!(
                (gradient[axis] - numeric).norm() < 1e-7,
                "axis {axis}: analytic {} vs numeric {numeric}",
                gradient[axis]
            );
        }
    }

    #[test]
    fn hessian_2d_is_symmetric_and_satisfies_helmholtz_trace() {
        let source = [0.0, 0.0];
        let field = [1.3, -0.4];
        let k = 1.7;
        let hessian = greens_function_hessian_2d(&source, &field, k);
        assert!((hessian[0][1] - hessian[1][0]).norm() < TOL);
        let trace = hessian[0][0] + hessian[1][1];
        let expected = -greens_function_2d(distance_2d(&source, &field), k) * (k * k);
        assert!((trace - expected).norm() / expected.norm() < 1e-12);
    }

    #[test]
    fn normal_derivative_2d_vanishes_for_tangential_normal() {
        let source = [0.0, 0.0];
        let field = [1.0, 0.0];
        let value = greens_function_normal_derivative_2d(&source, &field, &[0.0, 1.0], 2.0);
        assert!(value.norm() < TOL);
    }

    #[test]
    fn convected_kernels_recover_free_space_at_zero_mach() {
        let source_3d = Point::new_3d(0.1, -0.2, 0.3);
        let field_3d = Point::new_3d(1.0, 0.5, -0.4);
        let mach_3d = [0.0, 0.0, 0.0];
        let free_3d = greens_function_3d(distance(&source_3d, &field_3d), 2.0);
        let convected_3d = convected_greens_function_3d(&source_3d, &field_3d, 2.0, &mach_3d);
        assert!((convected_3d - free_3d).norm() / free_3d.norm() < 1e-14);
        let free_grad_3d = greens_function_gradient_3d(&source_3d, &field_3d, 2.0);
        let convected_grad_3d =
            convected_greens_function_gradient_3d(&source_3d, &field_3d, 2.0, &mach_3d);
        for axis in 0..3 {
            assert!((convected_grad_3d[axis] - free_grad_3d[axis]).norm() < 1e-14);
        }

        let source_2d = [0.1, -0.2];
        let field_2d = [1.0, 0.5];
        let mach_2d = [0.0, 0.0];
        let free_2d = greens_function_2d(distance_2d(&source_2d, &field_2d), 2.0);
        let convected_2d = convected_greens_function_2d(&source_2d, &field_2d, 2.0, &mach_2d);
        assert!((convected_2d - free_2d).norm() / free_2d.norm() < 1e-14);
        let free_grad_2d = greens_function_gradient_2d(&source_2d, &field_2d, 2.0);
        let convected_grad_2d =
            convected_greens_function_gradient_2d(&source_2d, &field_2d, 2.0, &mach_2d);
        for axis in 0..2 {
            assert!((convected_grad_2d[axis] - free_grad_2d[axis]).norm() < 1e-13);
        }
    }

    #[test]
    fn convected_gradient_3d_matches_finite_difference() {
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.2, -0.6, 0.8);
        let mach = [0.3, -0.1, 0.2];
        let gradient = convected_greens_function_gradient_3d(&source, &field, 1.5, &mach);
        for axis in 0..3 {
            let numeric = central_diff_3d(
                |y| convected_greens_function_3d(&source, y, 1.5, &mach),
                &field,
                axis,
                1e-6,
            );
            assert!(
                (gradient[axis] - numeric).norm() < 1e-7,
                "axis {axis}: analytic {} vs numeric {numeric}",
                gradient[axis]
            );
        }
    }

    #[test]
    fn convected_gradient_2d_matches_finite_difference() {
        let source = [0.0, 0.0];
        let field = [1.2, -0.6];
        let mach = [0.25, 0.15];
        let gradient = convected_greens_function_gradient_2d(&source, &field, 1.5, &mach);
        for axis in 0..2 {
            let numeric = central_diff_2d(
                |y| convected_greens_function_2d(&source, y, 1.5, &mach),
                &field,
                axis,
                1e-6,
            );
            assert!(
                (gradient[axis] - numeric).norm() < 1e-7,
                "axis {axis}: analytic {} vs numeric {numeric}",
                gradient[axis]
            );
        }
    }

    #[test]
    fn convected_source_derivative_recovers_free_space_symmetry_at_zero_mach() {
        let source = Point::new_3d(0.2, 0.1, -0.3);
        let field = Point::new_3d(1.0, 0.4, 0.2);
        let normal = [0.0, 1.0, 0.0];
        let mach = [0.0, 0.0, 0.0];
        let source_derivative = convected_greens_function_source_normal_derivative_3d(
            &source, &field, &normal, 2.0, &mach,
        );
        let field_derivative = greens_function_normal_derivative_3d(&source, &field, &normal, 2.0);
        assert!((source_derivative + field_derivative).norm() < 1e-14);
    }

    fn laplacian_3d(f: impl Fn(&Point) -> Complex64, base: &Point, h: f64) -> Complex64 {
        let coords = [base.x, base.y, base.z];
        let mut sum = Complex64::new(0.0, 0.0);
        for axis in 0..3 {
            let mut plus = coords;
            let mut minus = coords;
            plus[axis] += h;
            minus[axis] -= h;
            sum += (f(&Point::new_3d(plus[0], plus[1], plus[2]))
                + f(&Point::new_3d(minus[0], minus[1], minus[2]))
                - 2.0 * f(base))
                / (h * h);
        }
        sum
    }

    #[test]
    fn convected_kernels_satisfy_convected_equation() {
        // Wave-convention operator L = ∇² − (−ik + M·∇)² away from source.
        let k = 2.0;
        let mach = [0.3, -0.1, 0.2];
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let field = Point::new_3d(1.2, -0.6, 0.8);
        let kernel = |y: &Point| convected_greens_function_3d(&source, y, k, &mach);
        let h = 1e-4;
        let laplacian = laplacian_3d(&kernel, &field, h);
        let gradient = convected_greens_function_gradient_3d(&source, &field, k, &mach);
        let advection: Complex64 = gradient.iter().zip(mach.iter()).map(|(g, m)| g * m).sum();
        let material = -Complex64::new(0.0, k) * kernel(&field) + advection;
        // Second material derivative by finite difference along M.
        let m_hat = {
            let norm = mach.iter().map(|m| m * m).sum::<f64>().sqrt();
            mach.map(|m| m / norm)
        };
        let shift = |sign: f64| {
            Point::new_3d(
                field.x + sign * h * m_hat[0],
                field.y + sign * h * m_hat[1],
                field.z + sign * h * m_hat[2],
            )
        };
        let material_plus = -Complex64::new(0.0, k) * kernel(&shift(1.0))
            + convected_greens_function_gradient_3d(&source, &shift(1.0), k, &mach)
                .iter()
                .zip(mach.iter())
                .map(|(g, m)| g * m)
                .sum::<Complex64>();
        let material_minus = -Complex64::new(0.0, k) * kernel(&shift(-1.0))
            + convected_greens_function_gradient_3d(&source, &shift(-1.0), k, &mach)
                .iter()
                .zip(mach.iter())
                .map(|(g, m)| g * m)
                .sum::<Complex64>();
        let mach_norm = mach.iter().map(|m| m * m).sum::<f64>().sqrt();
        let directional = (material_plus - material_minus) / (2.0 * h) * mach_norm
            + Complex64::new(0.0, -k) * material;
        let residual = (laplacian - directional).norm();
        assert!(residual < 2e-5, "convected 3-D PDE residual {residual}");

        // On-axis travel-time phases: kR/(1+M) downstream, kR/(1−M) upstream.
        let axial = [0.4, 0.0, 0.0];
        let down = convected_greens_function_3d(&source, &Point::new_3d(2.0, 0.0, 0.0), k, &axial);
        assert!((down.arg() - k * 2.0 / 1.4).abs() < 1e-12);
        let up = convected_greens_function_3d(&source, &Point::new_3d(-2.0, 0.0, 0.0), k, &axial);
        let wrapped = (up.arg() - k * 2.0 / 0.6 + PI).rem_euclid(2.0 * PI) - PI;
        assert!(wrapped.abs() < 1e-9);
    }

    #[test]
    fn convected_kernel_has_unit_delta_strength() {
        // Flux form of L₊G = −δ over a tiny sphere: ∫[∂_nG − (U·n)D₊G]dS
        // with D₊ = −ik + M·∇ (wave convention, c = 1 units) must be −1.
        // This pins the absolute normalization, including any β factor.
        let k = 2.0;
        let mach = [0.3, -0.1, 0.2];
        let source = Point::new_3d(0.0, 0.0, 0.0);
        let radius = 1e-3;
        let (n_mu, n_phi) = (24, 48);
        let area = 4.0 * PI * radius * radius / (n_mu * n_phi) as f64;
        let mut flux = Complex64::new(0.0, 0.0);
        for m in 0..n_mu {
            let mu = -1.0 + 2.0 * (m as f64 + 0.5) / n_mu as f64;
            let radial = (1.0 - mu * mu).sqrt();
            for p in 0..n_phi {
                let phi = 2.0 * PI * (p as f64 + 0.5) / n_phi as f64;
                let normal = [radial * phi.cos(), radial * phi.sin(), mu];
                let field =
                    Point::new_3d(radius * normal[0], radius * normal[1], radius * normal[2]);
                let g = convected_greens_function_3d(&source, &field, k, &mach);
                let gradient = convected_greens_function_gradient_3d(&source, &field, k, &mach);
                let normal_gradient: Complex64 = gradient
                    .iter()
                    .zip(normal.iter())
                    .map(|(entry, n)| entry * n)
                    .sum();
                let advection: Complex64 = gradient
                    .iter()
                    .zip(mach.iter())
                    .map(|(entry, m)| entry * m)
                    .sum();
                let material = Complex64::new(0.0, -k) * g + advection;
                let mean_normal = mach[0] * normal[0] + mach[1] * normal[1] + mach[2] * normal[2];
                flux += (normal_gradient - mean_normal * material) * area;
            }
        }
        assert!(
            (flux + 1.0).norm() < 5e-3,
            "convected 3-D kernel strength flux {flux}"
        );
    }

    #[test]
    fn checked_convected_kernels_reject_sonic_and_coincident_inputs() {
        let point = Point::new_3d(0.0, 0.0, 0.0);
        let other = Point::new_3d(1.0, 0.0, 0.0);
        assert!(matches!(
            try_convected_greens_function_3d(&point, &other, 1.0, &[1.0, 0.0, 0.0]),
            Err(HelmholtzFunctionError::InvalidMach(_))
        ));
        assert!(matches!(
            try_convected_greens_function_3d(&point, &point, 1.0, &[0.1, 0.0, 0.0]),
            Err(HelmholtzFunctionError::InvalidDistance(_))
        ));
        let planar = [0.0, 0.0];
        assert!(matches!(
            try_convected_greens_function_2d(&planar, &planar, 1.0, &[0.1, 0.0]),
            Err(HelmholtzFunctionError::InvalidDistance(_))
        ));
        assert!(matches!(
            try_greens_function_hessian_2d(&planar, &planar, 1.0),
            Err(HelmholtzFunctionError::InvalidDistance(_))
        ));
    }
}
