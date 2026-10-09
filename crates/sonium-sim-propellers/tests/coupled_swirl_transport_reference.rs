use sonium_sim_propellers::{
    BodyFittedCoupledSwirlOptions, BodyFittedDuctStation, BodyFittedFlowOptions,
    BodyFittedRotatingWallModel, BodyFittedSwirlOptions, solve_body_fitted_coupled_swirl_transport,
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

fn couette_tangential_velocity(
    radius_m: f64,
    inner_radius_m: f64,
    outer_radius_m: f64,
    inner_angular_velocity_rad_s: f64,
    outer_angular_velocity_rad_s: f64,
) -> f64 {
    let inner_speed = inner_angular_velocity_rad_s * inner_radius_m;
    let outer_speed = outer_angular_velocity_rad_s * outer_radius_m;
    let denominator = outer_radius_m * outer_radius_m - inner_radius_m * inner_radius_m;
    let coefficient_a = (outer_speed * outer_radius_m - inner_speed * inner_radius_m) / denominator;
    let coefficient_b = inner_radius_m
        * outer_radius_m
        * (inner_speed * outer_radius_m - outer_speed * inner_radius_m)
        / denominator;
    coefficient_a * radius_m + coefficient_b / radius_m
}

#[test]
fn coupled_solid_body_rotation_is_a_fixed_point() {
    let mut flow = flow_options();
    flow.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 10.0,
        outer_angular_velocity_rad_s: 10.0,
    };
    let result = solve_body_fitted_coupled_swirl_transport(
        &profile(),
        flow,
        BodyFittedSwirlOptions {
            inlet_angular_velocity_rad_s: 10.0,
            ..BodyFittedSwirlOptions::default()
        },
        BodyFittedCoupledSwirlOptions {
            min_outer_iterations: 1,
            max_outer_iterations: 4,
            outer_tolerance: 1.0e-7,
            ..BodyFittedCoupledSwirlOptions::default()
        },
    )
    .expect("solid-body coupled solve");

    assert!(result.is_finite());
    assert!(result.converged);
    assert!(result.outer_residual < 1.0e-7);
    assert!(result.maximum_centrifugal_acceleration_m_s2 > 1.0);
    for node in &result.swirl.nodes {
        assert!((node.angular_velocity_rad_s - 10.0).abs() < 1.0e-6);
        assert!((node.tangential_velocity_m_s - 10.0 * node.radial_position_m).abs() < 1.0e-7);
    }
}

#[test]
fn coupled_differential_couette_matches_wall_data_and_feeds_back_force() {
    let mut flow = flow_options();
    flow.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 20.0,
        outer_angular_velocity_rad_s: 0.0,
    };
    let result = solve_body_fitted_coupled_swirl_transport(
        &profile(),
        flow,
        BodyFittedSwirlOptions::default(),
        BodyFittedCoupledSwirlOptions {
            min_outer_iterations: 1,
            max_outer_iterations: 6,
            outer_tolerance: 1.0e-5,
            outer_relaxation: 0.8,
            ..BodyFittedCoupledSwirlOptions::default()
        },
    )
    .expect("differential Couette coupled solve");

    assert!(result.is_finite());
    assert!(result.converged);
    assert!(result.maximum_centrifugal_acceleration_m_s2 > 1.0);
    for node in &result.swirl.nodes {
        let at_inner_wall = (node.radial_position_m - 0.02).abs() < 1.0e-12;
        let at_outer_wall = (node.radial_position_m - 0.05).abs() < 1.0e-12;
        if at_inner_wall {
            assert!((node.tangential_velocity_m_s - 20.0 * node.radial_position_m).abs() < 1.0e-7);
        }
        if at_outer_wall {
            assert!(node.tangential_velocity_m_s.abs() < 1.0e-12);
        }
        if (node.axial_position_m - 0.05).abs() < 1.0e-12 {
            let analytical =
                couette_tangential_velocity(node.radial_position_m, 0.021, 0.042, 20.0, 0.0);
            assert!(
                (node.tangential_velocity_m_s - analytical).abs() < 0.05,
                "z={} r={} computed={} analytical={} diff={}",
                node.axial_position_m,
                node.radial_position_m,
                node.tangential_velocity_m_s,
                analytical,
                (node.tangential_velocity_m_s - analytical).abs()
            );
        }
    }
}
