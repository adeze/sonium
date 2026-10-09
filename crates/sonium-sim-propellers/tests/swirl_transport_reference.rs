use sonium_sim_propellers::{
    BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1, BodyFittedDuctStation, BodyFittedFlowOptions,
    BodyFittedRotatingWallModel, BodyFittedSwirlError, BodyFittedSwirlOptions,
    solve_body_fitted_swirl_transport,
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
            inner_radius_m: 0.021,
            outer_radius_m: 0.042,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.1,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
    ]
}

fn flow_options() -> BodyFittedFlowOptions {
    BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: 0.0,
        axial_elements: 4,
        radial_elements: 3,
        max_picard_iterations: 20,
        min_picard_iterations: 1,
        picard_tolerance: 1.0e-6,
        picard_relaxation: 0.8,
        max_linear_solver_iterations: 100,
        linear_solver_tolerance: 1.0e-8,
        ..BodyFittedFlowOptions::default()
    }
}

#[test]
fn solid_body_rotation_is_recovered_by_transported_swirl_state() {
    let mut flow = flow_options();
    flow.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 10.0,
        outer_angular_velocity_rad_s: 10.0,
    };
    let result = solve_body_fitted_swirl_transport(
        &profile(),
        flow,
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: 10.0,
            ..BodyFittedSwirlOptions::default()
        },
    )
    .expect("solid-body swirl solve");

    assert_eq!(result.schema, BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1);
    assert!(result.is_finite());
    assert!(result.linear_residual < 1.0e-10);
    for node in &result.nodes {
        assert!(
            (node.angular_velocity_rad_s - 10.0).abs() < 1.0e-6,
            "angular velocity {} at r={} differs from solid-body value",
            node.angular_velocity_rad_s,
            node.radial_position_m
        );
        assert!(
            (node.tangential_velocity_m_s - 10.0 * node.radial_position_m).abs() < 1.0e-7,
            "tangential velocity {} at r={} differs from solid-body value",
            node.tangential_velocity_m_s,
            node.radial_position_m
        );
    }
}

#[test]
fn differential_rotating_walls_produce_a_bounded_transport_field() {
    let mut flow = flow_options();
    flow.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 20.0,
        outer_angular_velocity_rad_s: 0.0,
    };
    let result = solve_body_fitted_swirl_transport(
        &profile(),
        flow,
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: 0.0,
            ..BodyFittedSwirlOptions::default()
        },
    )
    .expect("differential-wall swirl solve");

    assert!(result.is_finite());
    assert!(result.maximum_tangential_speed_m_s > 0.0);
    assert!(result.maximum_tangential_speed_m_s <= 20.0 * 0.05 + 1.0e-8);
    assert!(
        result
            .nodes
            .iter()
            .any(|node| node.angular_velocity_rad_s > 1.0)
    );
}

#[test]
fn swirl_transport_rejects_centerline_geometry_and_nonfinite_options() {
    let mut centerline = profile();
    centerline[0].inner_radius_m = 0.0;
    let error = solve_body_fitted_swirl_transport(
        &centerline,
        flow_options(),
        BodyFittedSwirlOptions::default(),
    )
    .expect_err("centerline swirl geometry must be rejected");
    assert!(matches!(
        error,
        BodyFittedSwirlError::InvalidOption {
            field: "inner_radius_m",
            ..
        }
    ));

    let error = solve_body_fitted_swirl_transport(
        &profile(),
        flow_options(),
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: f64::NAN,
            ..BodyFittedSwirlOptions::default()
        },
    )
    .expect_err("non-finite swirl input must be rejected");
    assert!(matches!(error, BodyFittedSwirlError::InvalidOption { .. }));
}
