//! Azimuthal ring-source quadrature for axisymmetric radiation.
//!
//! Ring sources and ring surface patches radiate through an azimuthal
//! integral of the 3-D Helmholtz kernel weighted by a spinning phase
//! `exp(imφ)`. This module evaluates integrals of the form
//!
//! ```text
//! I_m = ∫₀²ᵖⁱ f(φ) exp(imφ) dφ
//! ```
//!
//! with Gauss-Legendre rules, plus an adaptive driver that doubles the rule
//! until the value settles. The caller supplies the smooth geometric factor
//! `f(φ)` (kernel values along the ring); the modal phase lives here so
//! every consumer shares one azimuthal convention.
//!
//! The integrand must be smooth on the ring: keep observers off the source
//! ring (`R(φ) > 0` for all `φ`). Near-singular observers converge, but need
//! high orders; the adaptive driver reports the order it needed.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::legendre::{legendre_p, legendre_p_derivative};

/// Failures from ring quadrature construction or evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum RingQuadratureError {
    /// The requested rule order was zero or above the supported cap.
    InvalidOrder(usize),
    /// Doubling the rule never met the tolerance budget.
    NonConvergence {
        /// Finest rule order attempted.
        finest_order: usize,
        /// Residual between the last two refinements.
        residual: f64,
    },
    /// The integrand or the accumulated sum was non-finite.
    NonFiniteResult(&'static str),
}

impl std::fmt::Display for RingQuadratureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder(order) => {
                write!(f, "ring quadrature order {order} must be within 1..=512")
            }
            Self::NonConvergence {
                finest_order,
                residual,
            } => write!(
                f,
                "ring quadrature did not converge by order {finest_order} (residual {residual})"
            ),
            Self::NonFiniteResult(name) => {
                write!(f, "ring quadrature value {name} was non-finite")
            }
        }
    }
}

impl std::error::Error for RingQuadratureError {}

/// Finest Gauss-Legendre order the adaptive driver will attempt.
///
/// Ring integrands are entire in `φ` for off-ring observers, so 512 points
/// resolve modal orders far beyond any rotor-stator application; refusal
/// past this cap signals a near-singular observer, not a resolution gap.
pub const MAX_RING_ORDER: usize = 512;

/// Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Nodes come from Newton iteration on `P_n` (recycled from the sibling
/// Legendre module) with the standard cosine starting values; weights use
/// `w = 2/((1-x²)P_n'(x)²)`.
///
/// # Errors
///
/// Returns [`RingQuadratureError::InvalidOrder`] for a zero order or an
/// order above [`MAX_RING_ORDER`].
pub fn gauss_legendre_rule(order: usize) -> Result<(Vec<f64>, Vec<f64>), RingQuadratureError> {
    if order == 0 || order > MAX_RING_ORDER {
        return Err(RingQuadratureError::InvalidOrder(order));
    }
    let mut nodes = Vec::with_capacity(order);
    let mut weights = Vec::with_capacity(order);
    for index in 1..=order {
        // Cosine starting value brackets the index-th root of P_n.
        let mut x = (PI * (index as f64 - 0.25) / (order as f64 + 0.5)).cos();
        for _ in 0..20 {
            let step = legendre_p(order, x) / legendre_p_derivative(order, x);
            x -= step;
            if step.abs() <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
                break;
            }
        }
        let derivative = legendre_p_derivative(order, x);
        nodes.push(x);
        weights.push(2.0 / ((1.0 - x * x) * derivative * derivative));
    }
    Ok((nodes, weights))
}

/// Evaluates one azimuthal ring integral at a fixed rule order.
///
/// Computes `∫₀²ᵖⁱ f(φ) exp(imφ) dφ` with an `order`-point Gauss-Legendre
/// rule mapped from `[-1, 1]` onto `[0, 2π]`. Positive `m` spins with
/// increasing `φ` under the `exp(+iωt)` convention.
///
/// # Errors
///
/// Returns [`RingQuadratureError`] for an invalid order or a non-finite
/// integrand value or accumulated sum.
pub fn azimuthal_ring_integral(
    azimuthal_order: i32,
    order: usize,
    mut integrand: impl FnMut(f64) -> Complex64,
) -> Result<Complex64, RingQuadratureError> {
    let (nodes, weights) = gauss_legendre_rule(order)?;
    let mut sum = Complex64::new(0.0, 0.0);
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let phi = PI * (node + 1.0);
        let value = integrand(phi);
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(RingQuadratureError::NonFiniteResult("integrand"));
        }
        sum += value * Complex64::from_polar(1.0, azimuthal_order as f64 * phi) * weight;
    }
    sum *= PI;
    if !sum.re.is_finite() || !sum.im.is_finite() {
        return Err(RingQuadratureError::NonFiniteResult("sum"));
    }
    Ok(sum)
}

/// Adaptively refined ring integral with an error estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingIntegral {
    /// The integral value at the finest accepted rule.
    pub value: Complex64,
    /// Finest Gauss-Legendre order used.
    pub rule_order: usize,
    /// Magnitude of the last refinement step.
    pub estimated_error: f64,
}

/// Evaluates a ring integral to a relative tolerance.
///
/// Doubles the rule from 8 points until successive refinements agree to
/// `tolerance` (relative to `max(1, |value|)`), or [`MAX_RING_ORDER`] is
/// reached. Deterministic: the same integrand always takes the same
/// refinement path.
///
/// # Errors
///
/// Returns [`RingQuadratureError`] for a non-positive or non-finite
/// tolerance, a non-finite integrand, or a missed tolerance budget.
pub fn azimuthal_ring_integral_adaptive(
    azimuthal_order: i32,
    tolerance: f64,
    mut integrand: impl FnMut(f64) -> Complex64,
) -> Result<RingIntegral, RingQuadratureError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(RingQuadratureError::NonFiniteResult("tolerance"));
    }
    let mut order = 8usize;
    let mut previous = azimuthal_ring_integral(azimuthal_order, order, &mut integrand)?;
    loop {
        if order >= MAX_RING_ORDER {
            return Err(RingQuadratureError::NonConvergence {
                finest_order: order,
                residual: f64::INFINITY,
            });
        }
        order *= 2;
        let current = azimuthal_ring_integral(azimuthal_order, order, &mut integrand)?;
        let residual = (current - previous).norm();
        if residual <= tolerance * current.norm().max(1.0) {
            return Ok(RingIntegral {
                value: current,
                rule_order: order,
                estimated_error: residual,
            });
        }
        previous = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::helmholtz::greens_function_3d;

    const TOL: f64 = 1e-12;

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // A 4-point rule integrates degree-7 polynomials exactly.
        let (nodes, weights) = gauss_legendre_rule(4).unwrap();
        let integral: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| w * (x.powi(7) - 3.0 * x.powi(4) + 0.5))
            .sum();
        // ∫₋₁¹ (x⁷ - 3x⁴ + 1/2) dx = 0 - 6/5 + 1 = -1/5.
        assert!((integral + 0.2).abs() < TOL);
    }

    #[test]
    fn ring_phase_is_orthonormal_over_modes() {
        for m in -3..=3 {
            let value = azimuthal_ring_integral(m, 16, |_| Complex64::new(1.0, 0.0)).unwrap();
            let expected = if m == 0 { 2.0 * PI } else { 0.0 };
            assert!(
                (value.re - expected).abs() < 1e-10 && value.im.abs() < 1e-10,
                "m = {m}: {value}"
            );
        }
    }

    #[test]
    fn on_axis_ring_matches_point_kernel_times_circumference() {
        // On the axis every ring point sits at the same distance R.
        let ring_radius = 0.4f64;
        let height = 1.2f64;
        let distance = (ring_radius * ring_radius + height * height).sqrt();
        let kernel = greens_function_3d(distance, 2.0);
        let value = azimuthal_ring_integral(0, 8, |_| kernel).unwrap();
        assert!((value - kernel * (2.0 * PI)).norm() < TOL);
    }

    #[test]
    fn adaptive_ring_matches_fine_fixed_rule_for_spinning_mode() {
        let ring_radius = 0.5;
        let observer = (0.8, 0.3);
        let reference =
            azimuthal_ring_integral(4, 256, |phi| ring_kernel(phi, ring_radius, observer)).unwrap();
        let adaptive = azimuthal_ring_integral_adaptive(4, 1e-10, |phi| {
            ring_kernel(phi, ring_radius, observer)
        })
        .unwrap();
        assert!((adaptive.value - reference).norm() / reference.norm() < 1e-9);
        assert!(adaptive.rule_order <= 256);
    }

    fn ring_kernel(phi: f64, ring_radius: f64, observer: (f64, f64)) -> Complex64 {
        let dx = observer.0 - ring_radius * phi.cos();
        let dy = observer.1 - ring_radius * phi.sin();
        greens_function_3d((dx * dx + dy * dy).sqrt(), 3.0)
    }

    #[test]
    fn invalid_orders_and_tolerances_fail_closed() {
        assert_eq!(
            azimuthal_ring_integral(0, 0, |_| Complex64::new(1.0, 0.0)),
            Err(RingQuadratureError::InvalidOrder(0))
        );
        assert!(matches!(
            azimuthal_ring_integral_adaptive(0, 0.0, |_| Complex64::new(1.0, 0.0)),
            Err(RingQuadratureError::NonFiniteResult(_))
        ));
        assert!(matches!(
            azimuthal_ring_integral(0, 8, |_| Complex64::new(f64::NAN, 0.0)),
            Err(RingQuadratureError::NonFiniteResult(_))
        ));
    }
}
