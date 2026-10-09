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
