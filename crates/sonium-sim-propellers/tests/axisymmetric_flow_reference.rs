use std::f64::consts::PI;

use sonium_sim_propellers::{
    AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1, AxisymmetricFlowError, AxisymmetricFlowOptions,
    AxisymmetricRotatingWallOptions, BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1, BodyFittedDuctStation,
    BodyFittedFlowNode, BodyFittedFlowResult, evaluate_axisymmetric_rotating_wall,
    evaluate_axisymmetric_weighted_flow,
};

fn meridional_flow(outlet_velocity: f64) -> BodyFittedFlowResult {
    let mut nodes = Vec::new();
    for axial_position_m in [0.0, 1.0] {
        for radial_position_m in [2.0, 1.0, 1.5] {
            nodes.push(BodyFittedFlowNode {
                radial_position_m,
                axial_position_m,
                radial_velocity_m_s: 0.0,
                axial_velocity_m_s: if axial_position_m == 0.0 {
                    1.0
                } else {
                    outlet_velocity
                },
                pressure_pa: 0.0,
            });
        }
    }
    BodyFittedFlowResult {
        schema: BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1.into(),
        nodes,
        mesh_node_count: 6,
        mesh_element_count: 4,
        iterations: 3,
        residual: 1.0e-9,
        converged: true,
        inlet_mean_axial_velocity_m_s: 1.0,
        outlet_mean_axial_velocity_m_s: outlet_velocity,
        relative_mean_velocity_mismatch: (1.0 - outlet_velocity).abs(),
        maximum_speed_m_s: 1.0_f64.max(outlet_velocity),
        minimum_pressure_pa: 0.0,
        maximum_pressure_pa: 0.0,
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} expected {expected}"
    );
}

#[test]
fn weighted_uniform_section_matches_annular_flow_identity() {
    let result = evaluate_axisymmetric_weighted_flow(
        &meridional_flow(1.0),
        AxisymmetricFlowOptions {
            density_kg_m3: 2.0,
            axial_tolerance_m: 1.0e-9,
        },
    )
    .expect("uniform weighted flow");

    assert_eq!(result.schema, AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1);
    assert!(result.is_finite());
    assert_close(result.inlet.area_m2, 3.0 * PI, 1.0e-12);
    assert_close(result.inlet.volume_flow_m3_s, 3.0 * PI, 1.0e-12);
    assert_close(result.inlet.mass_flow_kg_s, 6.0 * PI, 1.0e-12);
    assert_close(result.inlet.weighted_mean_axial_velocity_m_s, 1.0, 1.0e-12);
    assert_close(result.inlet.kinetic_energy_flux_w, 3.0 * PI, 1.0e-12);
    assert_close(result.relative_volume_flow_mismatch, 0.0, 1.0e-12);
    assert_close(result.relative_mass_flow_mismatch, 0.0, 1.0e-12);
}

#[test]
fn weighted_sections_are_radially_sorted_before_integration() {
    let result = evaluate_axisymmetric_weighted_flow(
        &meridional_flow(1.0),
        AxisymmetricFlowOptions::default(),
    )
    .expect("sorted sections");

    assert_close(result.inlet.inner_radius_m, 1.0, 1.0e-12);
    assert_close(result.inlet.outer_radius_m, 2.0, 1.0e-12);
    assert!(result.inlet.volume_flow_m3_s > 0.0);
}

#[test]
fn weighted_flow_reports_mass_and_volume_mismatch() {
    let result = evaluate_axisymmetric_weighted_flow(
        &meridional_flow(0.5),
        AxisymmetricFlowOptions {
            density_kg_m3: 3.0,
            axial_tolerance_m: 1.0e-9,
        },
    )
    .expect("mismatched sections");

    assert_close(result.relative_volume_flow_mismatch, 0.5, 1.0e-12);
    assert_close(result.relative_mass_flow_mismatch, 0.5, 1.0e-12);
}

#[test]
fn weighted_flow_rejects_invalid_options_and_states() {
    let invalid_density = evaluate_axisymmetric_weighted_flow(
        &meridional_flow(1.0),
        AxisymmetricFlowOptions {
            density_kg_m3: 0.0,
            ..AxisymmetricFlowOptions::default()
        },
    );
    assert!(matches!(
        invalid_density,
        Err(AxisymmetricFlowError::InvalidOption {
            field: "density_kg_m3",
            ..
        })
    ));

    let mut invalid = meridional_flow(1.0);
    invalid.schema = "wrong.schema".into();
    assert!(matches!(
        evaluate_axisymmetric_weighted_flow(&invalid, AxisymmetricFlowOptions::default()),
        Err(AxisymmetricFlowError::InvalidFlowResult {
            field: "meridional_flow"
        })
    ));
}

#[test]
fn weighted_flow_rejects_undersampled_sections() {
    let mut invalid = meridional_flow(1.0);
    invalid.nodes.truncate(1);
    invalid.mesh_node_count = 1;
    assert!(matches!(
        evaluate_axisymmetric_weighted_flow(&invalid, AxisymmetricFlowOptions::default()),
        Err(AxisymmetricFlowError::MissingSection { .. })
    ));
}

fn profile() -> Vec<BodyFittedDuctStation> {
    vec![
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 0.04,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.5,
            inner_radius_m: 0.021,
            outer_radius_m: 0.041,
        },
    ]
}

#[test]
fn rotating_wall_equal_angular_velocity_is_rigid_body_rotation() {
    let result = evaluate_axisymmetric_rotating_wall(
        &profile(),
        AxisymmetricRotatingWallOptions {
            inner_angular_velocity_rad_s: 100.0,
            outer_angular_velocity_rad_s: 100.0,
            dynamic_viscosity_pa_s: 0.01,
            radial_samples: 5,
            maximum_work: 100,
        },
    )
    .expect("rigid-body Couette solution");

    assert!(result.is_finite());
    assert_close(result.maximum_absolute_torque_n_m_per_m, 0.0, 1.0e-12);
    assert_close(result.dissipation_sum_w_per_m, 0.0, 1.0e-12);
    assert_close(result.stations[0].inner_wall_speed_m_s, 2.0, 1.0e-12);
    assert_close(result.stations[0].outer_wall_speed_m_s, 4.0, 1.0e-12);
    assert_close(
        result.stations[0].maximum_tangential_speed_m_s,
        4.0,
        1.0e-12,
    );
}

#[test]
fn rotating_wall_counter_speed_matches_torque_dissipation_identity() {
    let result = evaluate_axisymmetric_rotating_wall(
        &profile(),
        AxisymmetricRotatingWallOptions {
            inner_angular_velocity_rad_s: 100.0,
            outer_angular_velocity_rad_s: -25.0,
            dynamic_viscosity_pa_s: 0.01,
            radial_samples: 9,
            maximum_work: 100,
        },
    )
    .expect("counter-rotating Couette solution");
    let station = result.stations[0];

    assert_close(station.inner_wall_speed_m_s, 2.0, 1.0e-12);
    assert_close(station.outer_wall_speed_m_s, -1.0, 1.0e-12);
    assert!(station.fluid_torque_on_inner_wall_n_m_per_m < 0.0);
    assert!(station.viscous_dissipation_w_per_m > 0.0);
    assert_close(
        station.viscous_dissipation_w_per_m,
        -station.fluid_torque_on_inner_wall_n_m_per_m * 125.0,
        1.0e-10,
    );
}

#[test]
fn rotating_wall_rejects_axis_and_nonphysical_annuli() {
    let mut axis_profile = profile();
    axis_profile[0].inner_radius_m = 0.0;
    assert!(matches!(
        evaluate_axisymmetric_rotating_wall(
            &axis_profile,
            AxisymmetricRotatingWallOptions::default()
        ),
        Err(AxisymmetricFlowError::InvalidStation {
            field: "inner_radius_m",
            ..
        })
    ));

    let mut crossed_profile = profile();
    crossed_profile[0].outer_radius_m = crossed_profile[0].inner_radius_m;
    assert!(matches!(
        evaluate_axisymmetric_rotating_wall(
            &crossed_profile,
            AxisymmetricRotatingWallOptions::default()
        ),
        Err(AxisymmetricFlowError::InvalidStation {
            field: "outer_radius_m",
            ..
        })
    ));
}

#[test]
fn rotating_wall_rejects_option_and_work_budget_violations() {
    let options = AxisymmetricRotatingWallOptions {
        radial_samples: 1,
        ..AxisymmetricRotatingWallOptions::default()
    };
    assert!(matches!(
        evaluate_axisymmetric_rotating_wall(&profile(), options),
        Err(AxisymmetricFlowError::InvalidOption {
            field: "radial_samples",
            ..
        })
    ));

    let options = AxisymmetricRotatingWallOptions {
        radial_samples: 10,
        maximum_work: 1,
        ..AxisymmetricRotatingWallOptions::default()
    };
    assert!(matches!(
        evaluate_axisymmetric_rotating_wall(&profile(), options),
        Err(AxisymmetricFlowError::WorkBudgetExceeded { .. })
    ));
}
