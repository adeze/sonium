//! Analytical thermoelectric verification cases.

/// State in a constant-property one-dimensional thermoelectric leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantLegState {
    pub temperature: f64,
    /// Heat flux in the positive x direction.
    pub heat_flux: f64,
    /// Potential relative to the hot end.
    pub potential: f64,
}

/// Domenicali constant-property leg solution.
///
/// `x=0` is the hot end and `x=length` the cold end. Positive current density
/// points from hot to cold. `potential` follows `J=-sigma(grad V+alpha grad T)`.
#[allow(clippy::too_many_arguments)]
pub fn constant_property_leg(
    x: f64,
    length: f64,
    hot_temperature: f64,
    cold_temperature: f64,
    current_density: f64,
    electrical_resistivity: f64,
    thermal_conductivity: f64,
    seebeck_coefficient: f64,
) -> ConstantLegState {
    assert!(length > 0.0 && electrical_resistivity > 0.0 && thermal_conductivity > 0.0);
    let x = x.clamp(0.0, length);
    let slope = (cold_temperature - hot_temperature) / length;
    let joule_curvature =
        electrical_resistivity * current_density * current_density / thermal_conductivity;
    let temperature = hot_temperature + slope * x + 0.5 * joule_curvature * x * (length - x);
    let temperature_gradient = slope + 0.5 * joule_curvature * (length - 2.0 * x);
    let heat_flux = -thermal_conductivity * temperature_gradient
        + seebeck_coefficient * temperature * current_density;
    let potential = -electrical_resistivity * current_density * x
        - seebeck_coefficient * (temperature - hot_temperature);
    ConstantLegState {
        temperature,
        heat_flux,
        potential,
    }
}

/// Ioffe maximum generator efficiency for constant `z_t_bar`.
pub fn maximum_generator_efficiency(
    hot_temperature: f64,
    cold_temperature: f64,
    z_t_bar: f64,
) -> f64 {
    assert!(hot_temperature > cold_temperature && cold_temperature > 0.0 && z_t_bar >= 0.0);
    let root = (1.0 + z_t_bar).sqrt();
    (hot_temperature - cold_temperature) / hot_temperature * (root - 1.0)
        / (root + cold_temperature / hot_temperature)
}

/// Snyder-Ursell compatibility factor.
pub fn compatibility_factor(seebeck_coefficient: f64, temperature: f64, z_t: f64) -> f64 {
    assert!(temperature > 0.0 && z_t >= 0.0 && seebeck_coefficient != 0.0);
    ((1.0 + z_t).sqrt() - 1.0) / (seebeck_coefficient * temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domenicali_leg_has_parabolic_temperature_and_terminal_energy_balance() {
        let args = (0.02, 500.0, 300.0, 2.0e5, 1.0e-5, 1.5, 200.0e-6);
        let hot =
            constant_property_leg(0.0, args.0, args.1, args.2, args.3, args.4, args.5, args.6);
        let mid = constant_property_leg(
            0.5 * args.0,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5,
            args.6,
        );
        let cold = constant_property_leg(
            args.0, args.0, args.1, args.2, args.3, args.4, args.5, args.6,
        );
        assert_eq!(hot.temperature, args.1);
        assert_eq!(cold.temperature, args.2);
        assert!(mid.temperature > 0.5 * (args.1 + args.2));

        let electrical_power_density = -args.3 * cold.potential;
        assert!((cold.heat_flux - hot.heat_flux - electrical_power_density).abs() < 1e-8);
    }

    #[test]
    fn ioffe_efficiency_has_correct_limits() {
        assert_eq!(maximum_generator_efficiency(500.0, 300.0, 0.0), 0.0);
        let eta = maximum_generator_efficiency(500.0, 300.0, 1.0);
        assert!(eta > 0.0 && eta < (500.0 - 300.0) / 500.0);
        assert!(maximum_generator_efficiency(500.0, 300.0, 10.0) > eta);
    }

    #[test]
    fn segmented_compatibility_matches_snyder_ursell_expression() {
        let alpha = 200e-6;
        let temperature = 400.0;
        let z_t = 1.0;
        let expected = (2.0_f64.sqrt() - 1.0) / (alpha * temperature);
        assert!((compatibility_factor(alpha, temperature, z_t) - expected).abs() < 1e-12);
    }
}
