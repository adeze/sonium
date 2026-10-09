//! Analytical FSI verification cases.

use std::f64::consts::PI;

use num_complex::Complex64;

/// Fluid slug added mass `rho A L`.
pub fn fluid_column_added_mass(density: f64, area: f64, length: f64) -> f64 {
    assert!(density > 0.0 && area > 0.0 && length >= 0.0);
    density * area * length
}

/// Spring-piston natural frequency including incompressible fluid added mass.
pub fn added_mass_piston_frequency(
    structural_mass: f64,
    spring_stiffness: f64,
    fluid_density: f64,
    column_area: f64,
    column_length: f64,
) -> f64 {
    assert!(structural_mass > 0.0 && spring_stiffness > 0.0);
    let added = fluid_column_added_mass(fluid_density, column_area, column_length);
    (spring_stiffness / (structural_mass + added)).sqrt()
}

/// Moens-Korteweg pulse-wave speed with plane-strain wall correction.
pub fn moens_korteweg_wave_speed(
    youngs_modulus: f64,
    wall_thickness: f64,
    fluid_density: f64,
    tube_radius: f64,
    poisson_ratio: f64,
) -> f64 {
    assert!(youngs_modulus > 0.0 && wall_thickness > 0.0 && fluid_density > 0.0);
    assert!(tube_radius > 0.0 && poisson_ratio.abs() < 1.0);
    (youngs_modulus * wall_thickness
        / (2.0 * fluid_density * tube_radius * (1.0 - poisson_ratio * poisson_ratio)))
        .sqrt()
}

/// Basset-Boussinesq-Oseen harmonic hydrodynamic impedance of a sphere.
///
/// The returned force opposing velocity is `F = -Z U`. The expression is the
/// unsteady-Stokes result `6 pi mu a (1 + lambda + lambda^2/9)` with
/// `lambda=(1+i)a sqrt(omega rho/(2 mu))` for `exp(+i omega t)`.
pub fn oscillating_sphere_impedance(
    radius: f64,
    fluid_density: f64,
    dynamic_viscosity: f64,
    angular_frequency: f64,
) -> Complex64 {
    assert!(radius > 0.0 && fluid_density > 0.0 && dynamic_viscosity > 0.0);
    assert!(angular_frequency >= 0.0);
    let scale = radius * (angular_frequency * fluid_density / (2.0 * dynamic_viscosity)).sqrt();
    let lambda = Complex64::new(scale, scale);
    6.0 * PI * dynamic_viscosity * radius * (Complex64::ONE + lambda + lambda * lambda / 9.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluid_added_mass_reduces_piston_frequency() {
        let dry = added_mass_piston_frequency(1.0, 100.0, 1000.0, 1.0, 0.0);
        let wet = added_mass_piston_frequency(1.0, 100.0, 1000.0, 0.01, 1.0);
        assert!((dry - 10.0).abs() < 1e-12);
        assert!(wet < dry);
        assert!((fluid_column_added_mass(1000.0, 0.01, 1.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn moens_korteweg_has_expected_wall_and_density_scaling() {
        let base = moens_korteweg_wave_speed(1e6, 1e-3, 1000.0, 0.01, 0.3);
        let stiff = moens_korteweg_wave_speed(4e6, 1e-3, 1000.0, 0.01, 0.3);
        let dense = moens_korteweg_wave_speed(1e6, 1e-3, 4000.0, 0.01, 0.3);
        assert!((stiff / base - 2.0).abs() < 1e-12);
        assert!((dense / base - 0.5).abs() < 1e-12);
    }

    #[test]
    fn oscillating_sphere_recovers_stokes_drag_and_positive_added_mass() {
        let radius = 0.01;
        let viscosity = 1e-3;
        let steady = oscillating_sphere_impedance(radius, 1000.0, viscosity, 0.0);
        assert!((steady.re - 6.0 * PI * viscosity * radius).abs() < 1e-15);
        assert_eq!(steady.im, 0.0);
        let harmonic = oscillating_sphere_impedance(radius, 1000.0, viscosity, 100.0);
        assert!(harmonic.re > steady.re);
        assert!(harmonic.im > 0.0);
    }
}
