//! Hartmann, Shercliff-layer, and Hunt-flow analytical references.

/// Exact plane Hartmann velocity for plates at `y = +/- half_gap`.
pub fn hartmann_velocity(
    y: f64,
    half_gap: f64,
    pressure_gradient: f64,
    electrical_conductivity: f64,
    magnetic_flux_density: f64,
    hartmann_number: f64,
) -> f64 {
    assert!(half_gap > 0.0 && electrical_conductivity > 0.0 && magnetic_flux_density != 0.0);
    assert!(hartmann_number >= 0.0);
    let coordinate = y.clamp(-half_gap, half_gap) / half_gap;
    if hartmann_number < 1e-6 {
        // Stable zero-field limit with mu inferred from Ha^2 = a^2 B^2 sigma / mu.
        let viscosity = half_gap.powi(2) * magnetic_flux_density.powi(2) * electrical_conductivity
            / hartmann_number.max(f64::MIN_POSITIVE).powi(2);
        return -pressure_gradient * (half_gap * half_gap - y * y) / (2.0 * viscosity);
    }
    let scale = -pressure_gradient / (electrical_conductivity * magnetic_flux_density.powi(2));
    scale * (1.0 - (hartmann_number * coordinate).cosh() / hartmann_number.cosh())
}

/// Hartmann channel mean-velocity factor relative to
/// `u0 = -(dp/dx)/(sigma B^2)`: `1 - tanh(Ha)/Ha`.
pub fn hartmann_mean_velocity_factor(hartmann_number: f64) -> f64 {
    assert!(hartmann_number >= 0.0);
    if hartmann_number < 1e-4 {
        hartmann_number * hartmann_number / 3.0
    } else {
        1.0 - hartmann_number.tanh() / hartmann_number
    }
}

/// Leading Shercliff side-layer thickness `a/sqrt(Ha)`.
pub fn shercliff_layer_thickness(half_width: f64, hartmann_number: f64) -> f64 {
    assert!(half_width > 0.0 && hartmann_number > 0.0);
    half_width / hartmann_number.sqrt()
}

/// Leading exponential Shercliff side-layer velocity profile.
pub fn shercliff_side_layer_velocity(
    wall_distance: f64,
    half_width: f64,
    hartmann_number: f64,
    core_velocity: f64,
) -> f64 {
    assert!(wall_distance >= 0.0);
    let thickness = shercliff_layer_thickness(half_width, hartmann_number);
    core_velocity * (1.0 - (-wall_distance / thickness).exp())
}

/// Thin-wall conductance ratio used in Hunt-flow boundary conditions.
pub fn wall_conductance_ratio(
    wall_conductivity: f64,
    wall_thickness: f64,
    fluid_conductivity: f64,
    duct_half_width: f64,
) -> f64 {
    assert!(wall_conductivity >= 0.0 && wall_thickness >= 0.0);
    assert!(fluid_conductivity > 0.0 && duct_half_width > 0.0);
    wall_conductivity * wall_thickness / (fluid_conductivity * duct_half_width)
}

/// Fraction of transverse current closing through a thin conducting wall in
/// the one-dimensional wall/fluid resistance limit.
pub fn hunt_wall_current_fraction(conductance_ratio: f64) -> f64 {
    assert!(conductance_ratio >= 0.0);
    conductance_ratio / (1.0 + conductance_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hartmann_profile_satisfies_wall_and_symmetry_conditions() {
        let args = (0.1, -100.0, 1e6, 0.5, 10.0);
        let center = hartmann_velocity(0.0, args.0, args.1, args.2, args.3, args.4);
        assert!(center > 0.0);
        assert!(hartmann_velocity(args.0, args.0, args.1, args.2, args.3, args.4).abs() < 1e-12);
        assert_eq!(
            hartmann_velocity(-0.03, args.0, args.1, args.2, args.3, args.4),
            hartmann_velocity(0.03, args.0, args.1, args.2, args.3, args.4)
        );
    }

    #[test]
    fn shercliff_layer_has_inverse_square_root_scaling() {
        let low = shercliff_layer_thickness(1.0, 100.0);
        let high = shercliff_layer_thickness(1.0, 400.0);
        assert_eq!(low, 0.1);
        assert_eq!(high, 0.05);
        assert_eq!(shercliff_side_layer_velocity(0.0, 1.0, 100.0, 2.0), 0.0);
    }

    #[test]
    fn hunt_conducting_wall_closes_current_monotonically() {
        assert_eq!(hunt_wall_current_fraction(0.0), 0.0);
        assert_eq!(hunt_wall_current_fraction(1.0), 0.5);
        assert!(hunt_wall_current_fraction(100.0) > 0.99);
        assert_eq!(wall_conductance_ratio(2.0, 0.1, 1.0, 0.2), 1.0);
    }
}
