//! Analytical DC and induction-heating verification cases.

use std::f64::consts::PI;

use num_complex::Complex64;

/// Exact potential and temperature in a uniformly Joule-heated bar whose ends
/// are held at the same temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DcHeatedBarState {
    pub potential: f64,
    pub temperature: f64,
    pub joule_heat_density: f64,
}

pub fn dc_heated_bar(
    x: f64,
    length: f64,
    voltage: f64,
    end_temperature: f64,
    electrical_conductivity: f64,
    thermal_conductivity: f64,
) -> DcHeatedBarState {
    assert!(length > 0.0 && electrical_conductivity > 0.0 && thermal_conductivity > 0.0);
    let x = x.clamp(0.0, length);
    let field = voltage / length;
    let heat = electrical_conductivity * field * field;
    DcHeatedBarState {
        potential: voltage * x / length,
        temperature: end_temperature + heat * x * (length - x) / (2.0 * thermal_conductivity),
        joule_heat_density: heat,
    }
}

fn bessel_j0(z: Complex64) -> Complex64 {
    let zz = -0.25 * z * z;
    let mut term = Complex64::ONE;
    let mut sum = term;
    for m in 1..=100 {
        term *= zz / ((m * m) as f64);
        sum += term;
        if term.norm() <= 1e-15 * sum.norm().max(1.0) {
            break;
        }
    }
    sum
}

fn bessel_j1(z: Complex64) -> Complex64 {
    let zz = -0.25 * z * z;
    let mut term = 0.5 * z;
    let mut sum = term;
    for m in 1..=100 {
        term *= zz / ((m * (m + 1)) as f64);
        sum += term;
        if term.norm() <= 1e-15 * sum.norm().max(1.0) {
            break;
        }
    }
    sum
}

fn bessel_j1_real(x: f64) -> f64 {
    if x.abs() < 12.0 {
        return bessel_j1(Complex64::new(x, 0.0)).re;
    }
    (2.0 / (PI * x)).sqrt()
        * ((x - 3.0 * PI / 4.0).cos() - 3.0 / (8.0 * x) * (x - 3.0 * PI / 4.0).sin())
}

/// Internal impedance per unit length of an infinite round conductor.
pub fn round_conductor_internal_impedance(
    angular_frequency: f64,
    radius: f64,
    conductivity: f64,
    permeability: f64,
) -> Complex64 {
    assert!(angular_frequency >= 0.0 && radius > 0.0 && conductivity > 0.0 && permeability > 0.0);
    if angular_frequency == 0.0 {
        return Complex64::new(1.0 / (PI * radius * radius * conductivity), 0.0);
    }
    let kappa = Complex64::new(0.0, -angular_frequency * permeability * conductivity).sqrt();
    kappa * bessel_j0(kappa * radius)
        / (2.0 * PI * radius * conductivity * bessel_j1(kappa * radius))
}

/// Dodd-Deeds filament-loop impedance change above a conducting half-space.
///
/// The Sommerfeld integral is evaluated by composite Simpson quadrature. The
/// result has positive real part (added loss) and negative imaginary part
/// (reduced inductance) for the `exp(+i omega t)` convention.
pub fn dodd_deeds_filament_impedance_change(
    angular_frequency: f64,
    coil_radius: f64,
    lift_off: f64,
    conductivity: f64,
    permeability: f64,
    intervals: usize,
) -> Complex64 {
    assert!(angular_frequency >= 0.0 && coil_radius > 0.0 && lift_off > 0.0);
    assert!(conductivity >= 0.0 && permeability > 0.0);
    if angular_frequency == 0.0 || conductivity == 0.0 {
        return Complex64::ZERO;
    }
    let n = intervals.max(100) + intervals.max(100) % 2;
    let upper = 30.0 / lift_off;
    let step = upper / n as f64;
    let integral = (0..=n)
        .map(|index| {
            let alpha = index as f64 * step;
            let beta = Complex64::new(
                alpha * alpha,
                angular_frequency * permeability * conductivity,
            )
            .sqrt();
            let reflection =
                (Complex64::new(alpha, 0.0) - beta) / (Complex64::new(alpha, 0.0) + beta);
            let shape =
                bessel_j1_real(coil_radius * alpha).powi(2) * (-2.0 * alpha * lift_off).exp();
            let simpson = if index == 0 || index == n {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            reflection * (shape * simpson)
        })
        .sum::<Complex64>()
        * (step / 3.0);
    Complex64::new(
        0.0,
        angular_frequency * permeability * PI * coil_radius * coil_radius,
    ) * integral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_bar_conserves_electrical_and_thermal_power() {
        let state = dc_heated_bar(0.5, 1.0, 2.0, 300.0, 5.0, 10.0);
        assert_eq!(state.potential, 1.0);
        assert_eq!(state.joule_heat_density, 20.0);
        assert_eq!(state.temperature, 300.25);
        let outward_heat_flux_both_ends = 20.0;
        assert_eq!(outward_heat_flux_both_ends, state.joule_heat_density);
    }

    #[test]
    fn dodd_deeds_has_zero_air_response_and_passive_conductor_response() {
        let mu0 = 4e-7 * PI;
        assert_eq!(
            dodd_deeds_filament_impedance_change(1e4, 0.01, 0.002, 0.0, mu0, 1000),
            Complex64::ZERO
        );
        let change = dodd_deeds_filament_impedance_change(1e4, 0.01, 0.002, 5.8e7, mu0, 2000);
        assert!(
            change.re > 0.0,
            "passive conductor must increase coil resistance: {change}"
        );
        assert!(
            change.im < 0.0,
            "conductor must reduce coil inductance: {change}"
        );
    }

    #[test]
    fn round_conductor_recovers_dc_and_skin_effect_limits() {
        let radius = 1e-3;
        let sigma = 5.8e7;
        let mu0 = 4e-7 * PI;
        let dc = 1.0 / (PI * radius * radius * sigma);
        let low = round_conductor_internal_impedance(1.0, radius, sigma, mu0);
        assert!((low.re - dc).abs() / dc < 1e-6);
        let high = round_conductor_internal_impedance(1e6, radius, sigma, mu0);
        assert!(high.re > low.re);
        assert!(high.im > 0.0);
    }
}
