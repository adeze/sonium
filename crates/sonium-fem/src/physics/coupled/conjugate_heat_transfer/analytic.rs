//! Analytical conjugate heat-transfer benchmarks.

/// Exact heat rate and interface temperatures for two films separated by a wall.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneWallSolution {
    pub heat_flux: f64,
    pub hot_interface_temperature: f64,
    pub cold_interface_temperature: f64,
    pub overall_heat_transfer_coefficient: f64,
}

pub fn plane_wall_solution(
    hot_bulk_temperature: f64,
    cold_bulk_temperature: f64,
    hot_film_coefficient: f64,
    cold_film_coefficient: f64,
    wall_thickness: f64,
    wall_conductivity: f64,
) -> PlaneWallSolution {
    assert!(hot_film_coefficient > 0.0 && cold_film_coefficient > 0.0);
    assert!(wall_thickness >= 0.0 && wall_conductivity > 0.0);
    let resistance = 1.0 / hot_film_coefficient
        + wall_thickness / wall_conductivity
        + 1.0 / cold_film_coefficient;
    let overall = 1.0 / resistance;
    let heat_flux = overall * (hot_bulk_temperature - cold_bulk_temperature);
    PlaneWallSolution {
        heat_flux,
        hot_interface_temperature: hot_bulk_temperature - heat_flux / hot_film_coefficient,
        cold_interface_temperature: cold_bulk_temperature + heat_flux / cold_film_coefficient,
        overall_heat_transfer_coefficient: overall,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraetzBoundaryCondition {
    ConstantWallTemperature,
    ConstantWallHeatFlux,
}

/// Fully-developed circular-tube Nusselt number in the Graetz problem.
pub fn fully_developed_graetz_nusselt(boundary: GraetzBoundaryCondition) -> f64 {
    match boundary {
        GraetzBoundaryCondition::ConstantWallTemperature => 3.66,
        GraetzBoundaryCondition::ConstantWallHeatFlux => 48.0 / 11.0,
    }
}

/// Bulk temperature in the fully developed constant-wall-temperature limit.
pub fn graetz_bulk_temperature(
    x: f64,
    inlet_temperature: f64,
    wall_temperature: f64,
    heat_transfer_coefficient: f64,
    wetted_perimeter: f64,
    mass_flow_rate: f64,
    specific_heat: f64,
) -> f64 {
    assert!(x >= 0.0 && heat_transfer_coefficient >= 0.0 && wetted_perimeter > 0.0);
    assert!(mass_flow_rate > 0.0 && specific_heat > 0.0);
    wall_temperature
        + (inlet_temperature - wall_temperature)
            * (-heat_transfer_coefficient * wetted_perimeter * x / (mass_flow_rate * specific_heat))
                .exp()
}

/// Exact adiabatic-tip straight-fin temperature excess.
pub fn adiabatic_fin_temperature_excess(
    x: f64,
    length: f64,
    base_temperature_excess: f64,
    convection_coefficient: f64,
    perimeter: f64,
    conductivity: f64,
    area: f64,
) -> f64 {
    assert!(length > 0.0 && convection_coefficient >= 0.0 && perimeter > 0.0);
    assert!(conductivity > 0.0 && area > 0.0);
    let x = x.clamp(0.0, length);
    let m = (convection_coefficient * perimeter / (conductivity * area)).sqrt();
    if m == 0.0 {
        return base_temperature_excess;
    }
    base_temperature_excess * (m * (length - x)).cosh() / (m * length).cosh()
}

pub fn adiabatic_fin_base_heat_rate(
    length: f64,
    base_temperature_excess: f64,
    convection_coefficient: f64,
    perimeter: f64,
    conductivity: f64,
    area: f64,
) -> f64 {
    let m = (convection_coefficient * perimeter / (conductivity * area)).sqrt();
    conductivity * area * m * base_temperature_excess * (m * length).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_wall_has_identical_flux_through_all_resistances() {
        let solution = plane_wall_solution(400.0, 300.0, 100.0, 50.0, 0.02, 10.0);
        let hot_flux = 100.0 * (400.0 - solution.hot_interface_temperature);
        let wall_flux = 10.0
            * (solution.hot_interface_temperature - solution.cold_interface_temperature)
            / 0.02;
        let cold_flux = 50.0 * (solution.cold_interface_temperature - 300.0);
        assert!((hot_flux - wall_flux).abs() < 1e-10);
        assert!((wall_flux - cold_flux).abs() < 1e-10);
    }

    #[test]
    fn graetz_limits_and_bulk_energy_balance_match() {
        assert_eq!(
            fully_developed_graetz_nusselt(GraetzBoundaryCondition::ConstantWallTemperature),
            3.66
        );
        assert!(
            (fully_developed_graetz_nusselt(GraetzBoundaryCondition::ConstantWallHeatFlux)
                - 4.363_636_363_636_363)
                .abs()
                < 1e-14
        );
        let t1 = graetz_bulk_temperature(1.0, 300.0, 400.0, 10.0, 2.0, 1.0, 1000.0);
        assert!(t1 > 300.0 && t1 < 400.0);
    }

    #[test]
    fn adiabatic_fin_has_exact_base_and_zero_tip_gradient() {
        let args = (0.1, 100.0, 20.0, 0.04, 200.0, 1e-4);
        let base =
            adiabatic_fin_temperature_excess(0.0, args.0, args.1, args.2, args.3, args.4, args.5);
        assert!((base - args.1).abs() < 1e-12);
        let dx = 1e-6;
        let tip = adiabatic_fin_temperature_excess(
            args.0, args.0, args.1, args.2, args.3, args.4, args.5,
        );
        let near_tip = adiabatic_fin_temperature_excess(
            args.0 - dx,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5,
        );
        assert!(((tip - near_tip) / dx).abs() < 1e-2);
        assert!(adiabatic_fin_base_heat_rate(args.0, args.1, args.2, args.3, args.4, args.5) > 0.0);
    }
}
