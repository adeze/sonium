//! SPM and porous-electrode analytical benchmarks.

use num_complex::Complex64;

fn tan_x_equals_x_root(index: usize) -> f64 {
    assert!(index >= 1);
    let mut lower = index as f64 * std::f64::consts::PI + 1e-12;
    let mut upper = (index as f64 + 0.5) * std::f64::consts::PI - 1e-12;
    for _ in 0..100 {
        let middle = 0.5 * (lower + upper);
        let value = middle.tan() - middle;
        if value > 0.0 {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    0.5 * (lower + upper)
}

/// Surface concentration of a sphere under constant outward molar flux.
pub fn spherical_particle_surface_concentration(
    initial_concentration: f64,
    outward_flux: f64,
    particle_radius: f64,
    diffusivity: f64,
    time: f64,
    series_terms: usize,
) -> f64 {
    assert!(particle_radius > 0.0 && diffusivity > 0.0 && time >= 0.0);
    if time == 0.0 || outward_flux == 0.0 {
        return initial_concentration;
    }
    let tau = diffusivity * time / particle_radius.powi(2);
    let series = (1..=series_terms.max(1))
        .map(|index| {
            let root = tan_x_equals_x_root(index);
            (-root * root * tau).exp() / (root * root)
        })
        .sum::<f64>();
    initial_concentration
        - outward_flux * particle_radius / diffusivity * (3.0 * tau + 0.2 - 2.0 * series)
}

pub fn spherical_particle_average_concentration(
    initial_concentration: f64,
    outward_flux: f64,
    particle_radius: f64,
    time: f64,
) -> f64 {
    initial_concentration - 3.0 * outward_flux * time / particle_radius
}

pub fn sand_depletion_time(diffusivity: f64, initial_concentration: f64, molar_flux: f64) -> f64 {
    assert!(diffusivity > 0.0 && initial_concentration >= 0.0 && molar_flux != 0.0);
    std::f64::consts::PI * diffusivity * initial_concentration.powi(2) / (4.0 * molar_flux.powi(2))
}

/// Finite de-Levie blocking-pore transmission-line impedance.
pub fn de_levie_impedance(
    angular_frequency: f64,
    ionic_resistance_per_length: f64,
    double_layer_capacitance_per_length: f64,
    pore_length: f64,
) -> Complex64 {
    assert!(angular_frequency > 0.0);
    assert!(ionic_resistance_per_length > 0.0);
    assert!(double_layer_capacitance_per_length > 0.0 && pore_length > 0.0);
    let propagation = Complex64::new(
        0.0,
        angular_frequency * ionic_resistance_per_length * double_layer_capacitance_per_length,
    )
    .sqrt();
    let characteristic = (Complex64::new(ionic_resistance_per_length, 0.0)
        / Complex64::new(0.0, angular_frequency * double_layer_capacitance_per_length))
    .sqrt();
    characteristic / (propagation * pore_length).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_particle_obeys_exact_inventory_and_surface_depletion() {
        let c0 = 30_000.0;
        let flux = 1e-5;
        let radius = 1e-3;
        let time = 1.0;
        let average = spherical_particle_average_concentration(c0, flux, radius, time);
        assert_eq!(average, c0 - 0.03);
        let surface = spherical_particle_surface_concentration(c0, flux, radius, 1e-8, time, 100);
        assert!(surface < average);
    }

    #[test]
    fn sand_time_has_inverse_current_squared_scaling() {
        let base = sand_depletion_time(1e-9, 1000.0, 1.0);
        let doubled = sand_depletion_time(1e-9, 1000.0, 2.0);
        assert!((base / doubled - 4.0).abs() < 1e-12);
    }

    #[test]
    fn de_levie_impedance_has_blocking_and_distributed_rc_limits() {
        let low = de_levie_impedance(1e-3, 10.0, 1e-2, 1.0);
        let high = de_levie_impedance(1e6, 10.0, 1e-2, 1.0);
        assert!(low.norm() > high.norm());
        assert!(high.re > 0.0 && high.im < 0.0);
        assert!((high.re.abs() - high.im.abs()).abs() / high.re < 1e-6);
    }
}
