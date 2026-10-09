use std::sync::Arc;

use sonium_sim_propellers::{
    AxisymmetricMeridionalFlowError, AxisymmetricMeridionalForceFn, BodyFittedDuctStation,
    BodyFittedFlowOptions, BodyFittedTurbulenceModel, solve_axisymmetric_meridional_flow,
    solve_axisymmetric_meridional_flow_with_centrifugal_source,
};

fn profile() -> Vec<BodyFittedDuctStation> {
    vec![
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.05,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
    ]
}

fn options() -> BodyFittedFlowOptions {
    BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: 0.1,
        axial_elements: 4,
        radial_elements: 3,
        max_picard_iterations: 12,
        min_picard_iterations: 1,
        picard_tolerance: 1.0e-8,
        picard_relaxation: 0.8,
        linear_solver_tolerance: 1.0e-10,
        ..BodyFittedFlowOptions::default()
    }
}

#[test]
fn zero_inlet_has_a_finite_weighted_state() {
    let mut zero = options();
    zero.inlet_axial_velocity_m_s = 0.0;

    let result = solve_axisymmetric_meridional_flow(&profile(), zero).expect("zero-flow solve");

    assert!(result.is_finite());
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.maximum_speed_m_s <= 1.0e-12);
    assert!(result.inlet_volume_flux_m3_s.abs() <= 1.0e-12);
    assert!(result.outlet_volume_flux_m3_s.abs() <= 1.0e-12);
}

#[test]
fn inlet_velocity_and_weighted_flux_use_the_annular_measure() {
    let flow = options();
    let result = solve_axisymmetric_meridional_flow(&profile(), flow).expect("weighted solve");
    let radial_step = (0.05 - 0.02) / flow.radial_elements as f64;
    let expected_flux = 2.0
        * std::f64::consts::PI
        * (0..=flow.radial_elements)
            .map(|index| {
                let radius = 0.02 + index as f64 * radial_step;
                let velocity = (index != 0 && index != flow.radial_elements)
                    .then_some(flow.inlet_axial_velocity_m_s)
                    .unwrap_or(0.0);
                let quadrature_weight = if index == 0 || index == flow.radial_elements {
                    0.5
                } else {
                    1.0
                };
                quadrature_weight * radial_step * radius * velocity
            })
            .sum::<f64>();

    let inlet_nodes = result
        .nodes
        .iter()
        .filter(|node| node.axial_position_m.abs() <= 1.0e-12)
        .collect::<Vec<_>>();
    assert_eq!(inlet_nodes.len(), flow.radial_elements + 1);
    assert!(inlet_nodes.iter().any(|node| {
        (node.radial_position_m - 0.03).abs() <= 1.0e-12
            && (node.axial_velocity_m_s - flow.inlet_axial_velocity_m_s).abs() <= 1.0e-12
    }));
    assert!(
        (result.inlet_volume_flux_m3_s - expected_flux).abs() <= 1.0e-12,
        "weighted inlet flux {} expected {}",
        result.inlet_volume_flux_m3_s,
        expected_flux
    );
    assert!(result.relative_volume_flux_mismatch.is_finite());
}

#[test]
fn supplied_centrifugal_source_produces_a_radial_pressure_gradient() {
    let mut flow = options();
    flow.inlet_axial_velocity_m_s = 0.0;
    let angular_velocity_rad_s: f64 = 50.0;
    let source: AxisymmetricMeridionalForceFn =
        Arc::new(move |point| [angular_velocity_rad_s.powi(2) * point[0], 0.0]);

    let result =
        solve_axisymmetric_meridional_flow_with_centrifugal_source(&profile(), flow, Some(source))
            .expect("centrifugal pressure solve");

    let inner = result
        .nodes
        .iter()
        .find(|node| {
            (node.axial_position_m - 0.025).abs() <= 1.0e-12
                && (node.radial_position_m - 0.02).abs() <= 1.0e-12
        })
        .expect("inner mid-station node");
    let outer = result
        .nodes
        .iter()
        .find(|node| {
            (node.axial_position_m - 0.025).abs() <= 1.0e-12
                && (node.radial_position_m - 0.05).abs() <= 1.0e-12
        })
        .expect("outer mid-station node");

    assert!(result.is_finite());
    assert!(outer.pressure_pa > inner.pressure_pa);
    assert!(
        result.maximum_speed_m_s <= 1.0e-2,
        "centrifugal pressure balance induced speed {}",
        result.maximum_speed_m_s
    );
}

#[test]
fn invalid_centerline_and_turbulence_are_rejected() {
    let mut centerline = profile();
    centerline[0].inner_radius_m = 0.0;
    assert!(matches!(
        solve_axisymmetric_meridional_flow(&centerline, options()),
        Err(AxisymmetricMeridionalFlowError::InvalidStation {
            field: "inner_radius_m",
            ..
        })
    ));

    let mut turbulent = options();
    turbulent.turbulence_model = BodyFittedTurbulenceModel::KOmegaSstAlgebraic {
        c1: 0.09,
        characteristic_length_m: 0.03,
        nu_t_max_ratio: 20.0,
    };
    assert!(matches!(
        solve_axisymmetric_meridional_flow(&profile(), turbulent),
        Err(AxisymmetricMeridionalFlowError::InvalidOption {
            field: "turbulence_model",
            ..
        })
    ));
}
