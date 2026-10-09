//! Cahn-Hilliard/Navier-Stokes analytical benchmarks.

/// Equilibrium planar phase profile for the quartic free energy.
pub fn planar_interface_phase(coordinate: f64, interface_thickness: f64) -> f64 {
    assert!(interface_thickness > 0.0);
    (coordinate / (std::f64::consts::SQRT_2 * interface_thickness)).tanh()
}

/// Sharp-interface Laplace pressure jump.
pub fn laplace_pressure_jump(surface_tension: f64, radius: f64, spatial_dimension: usize) -> f64 {
    assert!(surface_tension > 0.0 && radius > 0.0);
    match spatial_dimension {
        2 => surface_tension / radius,
        3 => 2.0 * surface_tension / radius,
        _ => panic!("Laplace benchmark is defined for 2D circles or 3D spheres"),
    }
}

/// Deep-fluid inviscid capillary-wave angular frequency.
pub fn capillary_wave_frequency(
    surface_tension: f64,
    wavenumber: f64,
    density_one: f64,
    density_two: f64,
) -> f64 {
    assert!(surface_tension > 0.0 && wavenumber >= 0.0);
    assert!(density_one > 0.0 && density_two > 0.0);
    (surface_tension * wavenumber.powi(3) / (density_one + density_two)).sqrt()
}

/// Integrated surface energy of the exact tanh profile, evaluated by Simpson
/// quadrature over a symmetric finite interval.
pub fn planar_interface_surface_energy(
    surface_tension: f64,
    interface_thickness: f64,
    half_width_in_thicknesses: f64,
    intervals: usize,
) -> f64 {
    assert!(surface_tension > 0.0 && interface_thickness > 0.0);
    let beta = 3.0 * surface_tension / (2.0 * std::f64::consts::SQRT_2 * interface_thickness);
    let kappa = beta * interface_thickness.powi(2);
    let width = half_width_in_thicknesses * interface_thickness;
    let n = intervals.max(100) + intervals.max(100) % 2;
    let dx = 2.0 * width / n as f64;
    (0..=n)
        .map(|index| {
            let x = -width + index as f64 * dx;
            let phase = planar_interface_phase(x, interface_thickness);
            let derivative =
                (1.0 - phase * phase) / (std::f64::consts::SQRT_2 * interface_thickness);
            let density =
                0.25 * beta * (phase * phase - 1.0).powi(2) + 0.5 * kappa * derivative * derivative;
            let weight = if index == 0 || index == n {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            weight * density
        })
        .sum::<f64>()
        * dx
        / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_interface_has_pure_phase_limits_and_prescribed_energy() {
        let epsilon = 0.01;
        assert_eq!(planar_interface_phase(0.0, epsilon), 0.0);
        assert!(planar_interface_phase(10.0 * epsilon, epsilon) > 0.999_998);
        assert!(planar_interface_phase(-10.0 * epsilon, epsilon) < -0.999_998);
        let energy = planar_interface_surface_energy(0.07, epsilon, 12.0, 20_000);
        assert!((energy - 0.07).abs() / 0.07 < 1e-10);
    }

    #[test]
    fn laplace_jump_has_correct_curvature_dimension() {
        assert_eq!(laplace_pressure_jump(0.1, 0.5, 2), 0.2);
        assert_eq!(laplace_pressure_jump(0.1, 0.5, 3), 0.4);
    }

    #[test]
    fn capillary_wave_obeys_wavenumber_three_halves_scaling() {
        let base = capillary_wave_frequency(0.07, 2.0, 1000.0, 1.0);
        let doubled = capillary_wave_frequency(0.07, 4.0, 1000.0, 1.0);
        assert!((doubled / base - 2.0_f64.powf(1.5)).abs() < 1e-12);
    }
}
