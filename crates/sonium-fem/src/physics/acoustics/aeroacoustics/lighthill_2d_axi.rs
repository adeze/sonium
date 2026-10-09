//! Two-dimensional and axisymmetric Lighthill quadrupole radiation.
//!
//! The planar evaluator contracts a per-unit-depth stress moment with the
//! aero-convention Hankel Hessian (`exp(+iωt)` time dependence). The ring
//! evaluator integrates a cylindrical-component stress density around a
//! hoop of radius `r_s`, weighted by the spinning phase `exp(imφ)`, so
//! axisymmetric (`m = 0`) and spinning (`m ≠ 0`) sources share one code
//! path. Both reduce to the point-quadrupole physics of [`super::lighthill`]
//! in their respective limits.

use num_complex::Complex64;
use sonium_wave::analytical::azimuthal::AzimuthalMode;
use sonium_wave::special::helmholtz::{
    flip_time_convention, greens_function_2d, try_greens_function_hessian_2d,
};
use sonium_wave::special::ring::{azimuthal_ring_integral, azimuthal_ring_integral_adaptive};

use super::error::{AeroacousticsError, require_finite, require_positive};
use super::lighthill::lighthill_pressure_bare;

/// Compact line integral of the Lighthill stress tensor (per unit depth).
#[derive(Debug, Clone, Copy)]
pub struct LighthillQuadrupole2d {
    /// Source centroid `[m]` in the plane.
    pub position: [f64; 2],
    /// Harmonic tensor moment `∫(T_ij dA)` `[Pa m²]`, per unit depth.
    pub stress_moment: [[Complex64; 2]; 2],
}

impl LighthillQuadrupole2d {
    /// Builds a compact planar quadrupole after checking all components.
    pub fn new(
        position: [f64; 2],
        stress_moment: [[Complex64; 2]; 2],
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

/// Evaluates the planar Lighthill quadrupole with full near-field terms.
///
/// Contracts the moment with the aero-convention Hankel Hessian, i.e. the
/// conjugated [`try_greens_function_hessian_2d`] kernel for real `k`.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid acoustics parameters or a
/// coincident observer.
pub fn lighthill_pressure_2d(
    source: &LighthillQuadrupole2d,
    observer: [f64; 2],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<Complex64, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let hessian = try_greens_function_hessian_2d(&source.position, &observer, wavenumber).map_err(
        |error| match error {
            sonium_wave::special::helmholtz::HelmholtzFunctionError::InvalidDistance(_) => {
                AeroacousticsError::SingularObserver
            }
            other => AeroacousticsError::NonFiniteField {
                name: kernel_error_name(other),
            },
        },
    )?;
    let mut pressure = Complex64::new(0.0, 0.0);
    for (row, stress_row) in hessian.iter().zip(source.stress_moment.iter()) {
        for (entry, stress) in row.iter().zip(stress_row.iter()) {
            pressure += flip_time_convention(*entry) * stress;
        }
    }
    Ok(pressure)
}

fn kernel_error_name(
    error: sonium_wave::special::helmholtz::HelmholtzFunctionError,
) -> &'static str {
    match error {
        sonium_wave::special::helmholtz::HelmholtzFunctionError::InvalidDistance(_) => "observer",
        sonium_wave::special::helmholtz::HelmholtzFunctionError::InvalidWaveNumber(_) => {
            "angular_frequency"
        }
        sonium_wave::special::helmholtz::HelmholtzFunctionError::NonFiniteInput(_) => "position",
        sonium_wave::special::helmholtz::HelmholtzFunctionError::NonFiniteResult(_) => "kernel",
        sonium_wave::special::helmholtz::HelmholtzFunctionError::InvalidMach(_) => "mach",
    }
}

/// Evaluates only the radiating Hankel term of a planar quadrupole.
///
/// Keeps the leading `-k²G (n·T·n)` far-field term with cylindrical
/// `1/√r` spreading; the reactive near-field terms are dropped.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid acoustics parameters or a
/// coincident observer.
pub fn lighthill_far_field_pressure_2d(
    source: &LighthillQuadrupole2d,
    observer: [f64; 2],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<Complex64, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let displacement = [
        observer[0] - source.position[0],
        observer[1] - source.position[1],
    ];
    let distance = (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt();
    if distance <= f64::EPSILON {
        return Err(AeroacousticsError::SingularObserver);
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let green = flip_time_convention(greens_function_2d(distance, wavenumber));
    let direction = [displacement[0] / distance, displacement[1] / distance];
    let mut contraction = Complex64::new(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            contraction += source.stress_moment[i][j] * direction[i] * direction[j];
        }
    }
    Ok(-wavenumber * wavenumber * green * contraction)
}

/// Spinning Lighthill ring source of one azimuthal order.
///
/// The stress density is constant around the hoop in cylindrical `(r, φ, z)`
/// components; physical rotation enters through the per-sample Cartesian
/// transform, and the modal phase `exp(imφ)` through ring quadrature.
#[derive(Debug, Clone, Copy)]
pub struct LighthillRingSource {
    /// Hoop radius `[m]`.
    pub ring_radius_m: f64,
    /// Axial station of the hoop `[m]`.
    pub axial_position_m: f64,
    /// Azimuthal spinning order of the source.
    pub azimuthal: AzimuthalMode,
    /// Stress moment per radian `[Pa m³/rad]` in cylindrical components.
    pub stress_moment_per_rad: [[Complex64; 3]; 3],
}

impl LighthillRingSource {
    /// Builds a checked ring source.
    pub fn new(
        ring_radius_m: f64,
        axial_position_m: f64,
        azimuthal: AzimuthalMode,
        stress_moment_per_rad: [[Complex64; 3]; 3],
    ) -> Result<Self, AeroacousticsError> {
        if !ring_radius_m.is_finite() || ring_radius_m < 0.0 {
            return Err(AeroacousticsError::NonFiniteParameter {
                name: "ring_radius_m",
                value: ring_radius_m,
            });
        }
        require_finite("axial_position_m", axial_position_m)?;
        if stress_moment_per_rad
            .iter()
            .flatten()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField {
                name: "stress_moment_per_rad",
            });
        }
        Ok(Self {
            ring_radius_m,
            axial_position_m,
            azimuthal,
            stress_moment_per_rad,
        })
    }
}

/// Rotates a cylindrical `(r, φ, z)` tensor to Cartesian components.
///
/// Applies `T_cart = Q(φ) T_cyl Q(φ)ᵀ` with the hoop-angle rotation about
/// the `z` axis.
fn rotate_cylindrical_tensor_to_cartesian(
    tensor: &[[Complex64; 3]; 3],
    azimuth: f64,
) -> [[Complex64; 3]; 3] {
    let (sin, cos) = azimuth.sin_cos();
    let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
    let mut rotated = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = Complex64::new(0.0, 0.0);
            for a in 0..3 {
                for b in 0..3 {
                    sum += rotation[i][a] * tensor[a][b] * rotation[j][b];
                }
            }
            rotated[i][j] = sum;
        }
    }
    rotated
}

fn ring_sample(
    source: &LighthillRingSource,
    observer: [f64; 3],
    wavenumber: f64,
    azimuth: f64,
) -> Option<Complex64> {
    let point = [
        source.ring_radius_m * azimuth.cos(),
        source.ring_radius_m * azimuth.sin(),
        source.axial_position_m,
    ];
    let displacement = [
        observer[0] - point[0],
        observer[1] - point[1],
        observer[2] - point[2],
    ];
    let distance = (displacement[0] * displacement[0]
        + displacement[1] * displacement[1]
        + displacement[2] * displacement[2])
        .sqrt();
    if distance <= f64::EPSILON {
        return None;
    }
    let direction = [
        displacement[0] / distance,
        displacement[1] / distance,
        displacement[2] / distance,
    ];
    let tensor = rotate_cylindrical_tensor_to_cartesian(&source.stress_moment_per_rad, azimuth);
    lighthill_pressure_bare(&tensor, direction, distance, wavenumber)
}

/// Evaluates a spinning Lighthill ring at one Cartesian observer.
///
/// Integrates the hoop with an adaptive rule to `tolerance`. The Cartesian
/// observer enters the kernel geometry directly, so no separate
/// observer-azimuth phase is needed. Observers on the hoop itself are
/// rejected as singular.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, a singular
/// observer, or a missed quadrature tolerance.
pub fn lighthill_ring_pressure(
    source: &LighthillRingSource,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
    tolerance: f64,
) -> Result<Complex64, AeroacousticsError> {
    let integral = lighthill_ring_integral(source, observer, angular_frequency, speed_of_sound)?;
    let value = azimuthal_ring_integral_adaptive(source.azimuthal.order(), tolerance, |azimuth| {
        integral
            .sample(azimuth)
            .unwrap_or(Complex64::new(f64::NAN, 0.0))
    })?;
    if !value.value.re.is_finite() || !value.value.im.is_finite() {
        return Err(AeroacousticsError::SingularObserver);
    }
    Ok(value.value)
}

/// Evaluates a ring with a fixed quadrature order.
///
/// Shares the adaptive evaluator's geometry; use it for observer sweeps
/// where one validated order serves every point.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, a singular
/// observer, or an invalid quadrature order.
pub fn lighthill_ring_pressure_fixed(
    source: &LighthillRingSource,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
    order: usize,
) -> Result<Complex64, AeroacousticsError> {
    let integral = lighthill_ring_integral(source, observer, angular_frequency, speed_of_sound)?;
    let value = azimuthal_ring_integral(source.azimuthal.order(), order, |azimuth| {
        integral
            .sample(azimuth)
            .unwrap_or(Complex64::new(f64::NAN, 0.0))
    })?;
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(AeroacousticsError::SingularObserver);
    }
    Ok(value)
}

struct RingIntegral {
    sample: Box<dyn Fn(f64) -> Option<Complex64>>,
}

impl RingIntegral {
    fn sample(&self, azimuth: f64) -> Option<Complex64> {
        (self.sample)(azimuth)
    }
}

fn lighthill_ring_integral(
    source: &LighthillRingSource,
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
) -> Result<RingIntegral, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    if !source.ring_radius_m.is_finite() || source.ring_radius_m < 0.0 {
        return Err(AeroacousticsError::NonFiniteParameter {
            name: "ring_radius_m",
            value: source.ring_radius_m,
        });
    }
    require_finite("axial_position_m", source.axial_position_m)?;
    // Closest approach of the observer to the hoop, reached at φ' = φ_o.
    // Quadrature nodes never land exactly on the singular azimuth, so
    // on-hoop observers need this explicit rejection.
    let observer_radius = observer[0].hypot(observer[1]);
    let hoop_gap =
        (observer_radius - source.ring_radius_m).hypot(observer[2] - source.axial_position_m);
    if hoop_gap <= 1e-9 * source.ring_radius_m.max(1.0).max(observer_radius) {
        return Err(AeroacousticsError::SingularObserver);
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let moment = source.stress_moment_per_rad;
    let radius = source.ring_radius_m;
    let axial = source.axial_position_m;
    let azimuthal = source.azimuthal;
    let sample = Box::new(move |azimuth: f64| {
        let probe = LighthillRingSource {
            ring_radius_m: radius,
            axial_position_m: axial,
            azimuthal,
            stress_moment_per_rad: moment,
        };
        ring_sample(&probe, observer, wavenumber, azimuth)
    });
    Ok(RingIntegral {
        sample: sample as Box<dyn Fn(f64) -> Option<Complex64>>,
    })
}

#[cfg(test)]
mod tests {
    use super::super::lighthill::{LighthillQuadrupole, lighthill_pressure};
    use super::*;

    fn axial_planar_source() -> LighthillQuadrupole2d {
        let mut tensor = [[Complex64::new(0.0, 0.0); 2]; 2];
        tensor[0][0] = Complex64::new(2.0, -0.5);
        LighthillQuadrupole2d::new([0.0, 0.0], tensor).unwrap()
    }

    #[test]
    fn planar_far_field_has_cylindrical_spreading() {
        let source = axial_planar_source();
        let near = lighthill_far_field_pressure_2d(&source, [10.0, 0.0], 20.0, 2.0).unwrap();
        let far = lighthill_far_field_pressure_2d(&source, [40.0, 0.0], 20.0, 2.0).unwrap();
        // Hankel far field decays as 1/√r: |p(10)/p(40)| → 2.
        assert!((near.norm() / far.norm() - 2.0).abs() < 0.02);
    }

    #[test]
    fn planar_axial_quadrupole_has_far_field_null_transverse_to_axis() {
        let pressure =
            lighthill_far_field_pressure_2d(&axial_planar_source(), [0.0, 10.0], 20.0, 2.0)
                .unwrap();
        assert!(pressure.norm() < 1e-12);
    }

    #[test]
    fn planar_exact_solution_converges_to_radiating_term() {
        let source = axial_planar_source();
        let exact = lighthill_pressure_2d(&source, [100.0, 0.0], 10.0, 1.0).unwrap();
        let far = lighthill_far_field_pressure_2d(&source, [100.0, 0.0], 10.0, 1.0).unwrap();
        assert!((exact - far).norm() / far.norm() < 5e-3);
    }

    fn isotropic_ring(order: i32) -> LighthillRingSource {
        let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            tensor[i][i] = Complex64::new(1.5, 0.25);
        }
        LighthillRingSource::new(0.4, 0.0, AzimuthalMode::new(order), tensor).unwrap()
    }

    #[test]
    fn axisymmetric_ring_on_axis_matches_point_kernel_times_circumference() {
        let ring = isotropic_ring(0);
        let observer = [0.0, 0.0, 2.0];
        let ring_value = lighthill_ring_pressure(&ring, observer, 10.0, 1.0, 1e-10).unwrap();
        // Isotropic tensors are rotation-invariant, so every hoop sample
        // matches the point kernel at any single hoop point.
        let point = LighthillQuadrupole::new([0.4, 0.0, 0.0], ring.stress_moment_per_rad).unwrap();
        let expected = lighthill_pressure(&point, observer, 10.0, 1.0).unwrap();
        assert!(
            (ring_value - expected * (2.0 * std::f64::consts::PI)).norm() / expected.norm() < 1e-9
        );
    }

    #[test]
    fn spinning_ring_vanishes_on_axis_for_isotropic_density() {
        let ring = isotropic_ring(2);
        let value = lighthill_ring_pressure(&ring, [0.0, 0.0, 2.0], 10.0, 1.0, 1e-10).unwrap();
        assert!(value.norm() < 1e-10);
    }

    #[test]
    fn ring_matches_discrete_point_summation_off_axis() {
        let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
        tensor[0][0] = Complex64::new(1.0, 0.0);
        tensor[1][2] = Complex64::new(0.0, 0.5);
        tensor[2][1] = Complex64::new(0.0, 0.5);
        let ring = LighthillRingSource::new(0.5, 0.2, AzimuthalMode::new(1), tensor).unwrap();
        let observer = [1.3, -0.4, 0.9];
        let ring_value = lighthill_ring_pressure(&ring, observer, 8.0, 1.0, 1e-10).unwrap();
        // Independent check: 720 discrete spinning point quadrupoles.
        let samples = 720usize;
        let mut sum = Complex64::new(0.0, 0.0);
        for sample in 0..samples {
            let phi = 2.0 * std::f64::consts::PI * sample as f64 / samples as f64;
            let rotated = rotate_cylindrical_tensor_to_cartesian(&tensor, phi)
                .map(|row| row.map(|entry| entry * Complex64::from_polar(1.0, phi)));
            let point =
                LighthillQuadrupole::new([0.5 * phi.cos(), 0.5 * phi.sin(), 0.2], rotated).unwrap();
            sum += lighthill_pressure(&point, observer, 8.0, 1.0).unwrap();
        }
        sum *= 2.0 * std::f64::consts::PI / samples as f64;
        assert!((ring_value - sum).norm() / sum.norm() < 1e-6);
    }

    #[test]
    fn spinning_ring_carries_observer_azimuth_phase() {
        let ring = isotropic_ring(1);
        let radius = 1.5f64.hypot(0.8);
        let base = lighthill_ring_pressure(&ring, [radius, 0.0, 1.0], 8.0, 1.0, 1e-10).unwrap();
        let rotated = lighthill_ring_pressure(&ring, [1.5, 0.8, 1.0], 8.0, 1.0, 1e-10).unwrap();
        let expected = base * AzimuthalMode::new(1).modal_phase(0.8f64.atan2(1.5));
        assert!((rotated - expected).norm() / expected.norm() < 1e-9);
    }

    #[test]
    fn ring_rejects_observers_on_the_hoop() {
        let ring = isotropic_ring(0);
        assert_eq!(
            lighthill_ring_pressure(&ring, [0.4, 0.0, 0.0], 10.0, 1.0, 1e-8),
            Err(AeroacousticsError::SingularObserver)
        );
    }
}
