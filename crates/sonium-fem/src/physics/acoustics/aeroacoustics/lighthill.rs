//! Lighthill acoustic analogy for compact quadrupole sources.
//!
//! This module evaluates the free-space frequency-domain solution of
//!
//! ```text
//! (nabla^2 + k^2) p' = -d_i d_j T_ij,
//! ```
//!
//! for a compact volume integral of the Lighthill stress tensor. Harmonic
//! amplitudes use the `exp(i omega t)` convention and the outgoing Green
//! function `G(r) = exp(-i k r) / (4 pi r)`. The exact Green-function Hessian
//! retains the `1/r`, `1/r^2`, and `1/r^3` terms. The far-field function keeps
//! only the radiating `1/r` term.
//!
//! # Validity
//!
//! The source must be compact compared with wavelength and observer distance.
//! The medium is homogeneous, quiescent, and unbounded. For distributed or
//! non-compact sources, sum sufficiently small [`LighthillQuadrupole`] cells.
//!
//! # Reference
//!
//! M. J. Lighthill, "On Sound Generated Aerodynamically. I. General Theory",
//! *Proceedings of the Royal Society A* 211 (1952), 564--587.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::error::{AeroacousticsError, require_finite, require_positive};

/// Compact volume integral of the Lighthill stress tensor.
#[derive(Debug, Clone, Copy)]
pub struct LighthillQuadrupole {
    /// Source centroid `[m]`.
    pub position: [f64; 3],
    /// Harmonic tensor moment `integral(T_ij dV)` `[Pa m^3]`.
    pub stress_moment: [[Complex64; 3]; 3],
}

impl LighthillQuadrupole {
    /// Construct a compact quadrupole after checking all components.
    pub fn new(
        position: [f64; 3],
        stress_moment: [[Complex64; 3]; 3],
    ) -> Result<Self, AeroacousticsError> {
        if position.iter().any(|x| !x.is_finite()) {
            return Err(AeroacousticsError::NonFiniteField { name: "position" });
        }
        if stress_moment
            .iter()
            .flatten()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField {
                name: "stress_moment",
            });
        }
        Ok(Self {
            position,
            stress_moment,
        })
    }
}

/// Evaluate the compact Lighthill quadrupole, including reactive near-field terms.
pub fn lighthill_pressure(
    source: &LighthillQuadrupole,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<Complex64, AeroacousticsError> {
    let geometry = geometry(source, observer, angular_frequency, speed_of_sound)?;
    lighthill_pressure_bare(
        &source.stress_moment,
        geometry.direction,
        geometry.distance,
        geometry.wavenumber,
    )
    .ok_or(AeroacousticsError::SingularObserver)
}

/// Bare Hessian contraction shared by the point and ring evaluators.
///
/// Contracts a Cartesian stress tensor with the aero-convention Green
/// Hessian at one precomputed geometry. Returns `None` for a coincident
/// pair instead of validating inputs.
pub(crate) fn lighthill_pressure_bare(
    stress_moment: &[[Complex64; 3]; 3],
    direction: [f64; 3],
    distance: f64,
    wavenumber: f64,
) -> Option<Complex64> {
    if distance <= f64::EPSILON {
        return None;
    }
    let green = Complex64::from_polar(1.0 / (4.0 * PI * distance), -wavenumber * distance);

    // Hessian of exp(-ikr)/(4*pi*r):
    // d_i d_j G = A n_i n_j + B delta_ij.
    let radial = green
        * Complex64::new(
            -wavenumber * wavenumber + 3.0 / (distance * distance),
            3.0 * wavenumber / distance,
        );
    let isotropic = green * Complex64::new(-1.0 / (distance * distance), -wavenumber / distance);

    Some(
        radial * directional_contraction_raw(stress_moment, direction)
            + isotropic * trace_raw(stress_moment),
    )
}

/// Evaluate only the radiating `1/r` term of the compact Lighthill quadrupole.
pub fn lighthill_far_field_pressure(
    source: &LighthillQuadrupole,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<Complex64, AeroacousticsError> {
    let geometry = geometry(source, observer, angular_frequency, speed_of_sound)?;
    let k = geometry.wavenumber;
    let green = Complex64::from_polar(1.0 / (4.0 * PI * geometry.distance), -k * geometry.distance);
    Ok(-k * k * green * directional_contraction_raw(&source.stress_moment, geometry.direction))
}

struct SourceGeometry {
    distance: f64,
    direction: [f64; 3],
    wavenumber: f64,
}

fn geometry(
    source: &LighthillQuadrupole,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<SourceGeometry, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let displacement = std::array::from_fn(|i| observer[i] - source.position[i]);
    let distance = norm(displacement);
    if distance <= f64::EPSILON {
        return Err(AeroacousticsError::SingularObserver);
    }
    Ok(SourceGeometry {
        distance,
        direction: displacement.map(|x| x / distance),
        wavenumber: angular_frequency / speed_of_sound,
    })
}

fn directional_contraction_raw(
    stress_moment: &[[Complex64; 3]; 3],
    direction: [f64; 3],
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            result += stress_moment[i][j] * direction[i] * direction[j];
        }
    }
    result
}

fn trace_raw(stress_moment: &[[Complex64; 3]; 3]) -> Complex64 {
    stress_moment[0][0] + stress_moment[1][1] + stress_moment[2][2]
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axial_source() -> LighthillQuadrupole {
        let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
        tensor[0][0] = Complex64::new(2.0, -0.5);
        LighthillQuadrupole::new([0.0; 3], tensor).unwrap()
    }

    #[test]
    fn far_field_has_inverse_distance_spreading() {
        let source = axial_source();
        let p1 = lighthill_far_field_pressure(&source, [10.0, 0.0, 0.0], 20.0, 2.0).unwrap();
        let p2 = lighthill_far_field_pressure(&source, [25.0, 0.0, 0.0], 20.0, 2.0).unwrap();
        assert!((p1.norm() / p2.norm() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn axial_quadrupole_has_far_field_null_transverse_to_axis() {
        let pressure =
            lighthill_far_field_pressure(&axial_source(), [0.0, 10.0, 0.0], 20.0, 2.0).unwrap();
        assert_eq!(pressure, Complex64::new(0.0, 0.0));
    }

    #[test]
    fn exact_solution_converges_to_radiating_term() {
        let source = axial_source();
        let exact = lighthill_pressure(&source, [100.0, 0.0, 0.0], 10.0, 1.0).unwrap();
        let far = lighthill_far_field_pressure(&source, [100.0, 0.0, 0.0], 10.0, 1.0).unwrap();
        assert!((exact - far).norm() / far.norm() < 2.1e-3);
    }
}
