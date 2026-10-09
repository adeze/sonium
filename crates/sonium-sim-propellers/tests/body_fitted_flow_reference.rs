use sonium_sim_propellers::{
    BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1, BodyFittedDuctStation, BodyFittedFlowError,
    BodyFittedFlowOptions, BodyFittedRotatingWallModel, BodyFittedTurbulenceModel,
    solve_body_fitted_duct_flow,
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

fn options() -> BodyFittedFlowOptions {
    BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: 0.1,
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
fn body_fitted_flow_solves_low_reynolds_profile() {
    let result = solve_body_fitted_duct_flow(&profile(), options()).expect("low-Re solve");

    assert_eq!(result.schema, BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1);
    assert_eq!(result.mesh_node_count, 20);
    assert_eq!(result.mesh_element_count, 24);
    assert!(result.converged);
    assert!(result.residual <= 1.0e-6);
    assert!(result.is_finite());
    assert!(result.maximum_speed_m_s.is_finite());
    assert!(
        result
            .nodes
            .iter()
            .any(|node| node.radial_position_m > 0.02)
    );
    assert!(
        result
            .nodes
            .iter()
            .all(|node| node.axial_position_m >= 0.0 && node.axial_position_m <= 0.1)
    );
}

#[test]
fn body_fitted_flow_is_deterministic_and_preserves_geometry() {
    let first = solve_body_fitted_duct_flow(&profile(), options()).expect("first solve");
    let second = solve_body_fitted_duct_flow(&profile(), options()).expect("second solve");

    assert_eq!(first, second);
    assert!(
        first
            .nodes
            .iter()
            .any(|node| (node.axial_position_m - 0.05).abs() < 1.0e-12)
    );
    assert!(first.nodes.iter().any(|node| {
        (node.axial_position_m - 0.05).abs() < 1.0e-12
            && (node.radial_position_m - 0.042).abs() < 1.0e-12
    }));
}

#[test]
fn body_fitted_flow_zero_inlet_is_a_finite_trivial_state() {
    let mut zero_options = options();
    zero_options.inlet_axial_velocity_m_s = 0.0;
    let result = solve_body_fitted_duct_flow(&profile(), zero_options).expect("zero flow");

    assert!(result.converged);
    assert!(result.maximum_speed_m_s < 1.0e-12);
    assert!(result.inlet_mean_axial_velocity_m_s.abs() < 1.0e-12);
    assert!(result.outlet_mean_axial_velocity_m_s.abs() < 1.0e-12);
}

#[test]
fn body_fitted_flow_rejects_invalid_profile_before_mesh_generation() {
    let mut malformed = profile();
    malformed[1].axial_position_m = malformed[0].axial_position_m;
    assert!(matches!(
        solve_body_fitted_duct_flow(&malformed, options()),
        Err(BodyFittedFlowError::NonMonotonicProfile { index: 1 })
    ));

    let mut invalid_radius = profile();
    invalid_radius[1].outer_radius_m = invalid_radius[1].inner_radius_m;
    assert!(matches!(
        solve_body_fitted_duct_flow(&invalid_radius, options()),
        Err(BodyFittedFlowError::InvalidStation {
            index: 1,
            field: "outer_radius_m",
            ..
        })
    ));
}

#[test]
fn body_fitted_flow_rejects_invalid_options_and_mesh_budget() {
    let mut invalid = options();
    invalid.picard_relaxation = 0.0;
    assert!(matches!(
        solve_body_fitted_duct_flow(&profile(), invalid),
        Err(BodyFittedFlowError::InvalidOption {
            field: "picard_relaxation",
            ..
        })
    ));

    let mut budget = options();
    budget.maximum_nodes = 4;
    assert!(matches!(
        solve_body_fitted_duct_flow(&profile(), budget),
        Err(BodyFittedFlowError::MeshBudgetExceeded { .. })
    ));
}

#[test]
fn body_fitted_flow_can_publish_unconverged_state_when_explicitly_allowed() {
    let mut one_step = options();
    one_step.max_picard_iterations = 1;
    one_step.min_picard_iterations = 1;
    one_step.require_convergence = false;
    let result = solve_body_fitted_duct_flow(&profile(), one_step).expect("finite one-step state");

    assert!(!result.converged || result.iterations == 1);
    assert!(result.is_finite());
}

#[test]
fn body_fitted_flow_result_finiteness_is_fail_closed() {
    let mut result = solve_body_fitted_duct_flow(&profile(), options()).expect("flow result");
    result.nodes[0].pressure_pa = f64::NAN;
    assert!(!result.is_finite());
}

#[test]
fn body_fitted_flow_threads_algebraic_sst_closure() {
    let model = BodyFittedTurbulenceModel::KOmegaSstAlgebraic {
        c1: 0.09,
        characteristic_length_m: 0.03,
        nu_t_max_ratio: 20.0,
    };
    let molecular_viscosity = 1.0;
    let effective = model.effective_dynamic_viscosity_pa_s(molecular_viscosity, 1.0, 0.1);
    assert!(effective > molecular_viscosity);

    let mut turbulent_options = options();
    turbulent_options.turbulence_model = model;
    let result = solve_body_fitted_duct_flow(&profile(), turbulent_options)
        .expect("algebraic SST body-fitted solve");
    assert!(result.is_finite());
}

#[test]
fn body_fitted_flow_rejects_invalid_turbulence_parameters() {
    let mut invalid = options();
    invalid.turbulence_model = BodyFittedTurbulenceModel::KOmegaSstAlgebraic {
        c1: f64::NAN,
        characteristic_length_m: 0.03,
        nu_t_max_ratio: 20.0,
    };
    assert!(matches!(
        solve_body_fitted_duct_flow(&profile(), invalid),
        Err(BodyFittedFlowError::InvalidOption {
            field: "turbulence.c1",
            ..
        })
    ));
}

#[test]
fn rotating_wall_model_matches_solid_body_centrifugal_acceleration() {
    let model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 100.0,
        outer_angular_velocity_rad_s: 100.0,
    };
    let acceleration = model.centrifugal_acceleration_m_s2(0.035, 0.02, 0.05);
    assert!((acceleration - 100.0_f64.powi(2) * 0.035).abs() < 1.0e-10);
    assert_eq!(
        BodyFittedRotatingWallModel::None.centrifugal_acceleration_m_s2(0.035, 0.02, 0.05),
        0.0
    );
}

#[test]
fn rotating_wall_body_force_is_used_by_body_fitted_flow() {
    let mut rotating = options();
    rotating.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: 25.0,
        outer_angular_velocity_rad_s: 25.0,
    };
    let result =
        solve_body_fitted_duct_flow(&profile(), rotating).expect("rotating-wall body-fitted solve");
    assert!(result.is_finite());
}

#[test]
fn rotating_wall_model_rejects_nonfinite_wall_speed() {
    let mut invalid = options();
    invalid.rotating_wall_model = BodyFittedRotatingWallModel::AnnularCouette {
        inner_angular_velocity_rad_s: f64::INFINITY,
        outer_angular_velocity_rad_s: 0.0,
    };
    assert!(matches!(
        solve_body_fitted_duct_flow(&profile(), invalid),
        Err(BodyFittedFlowError::InvalidOption {
            field: "rotating_wall.inner_angular_velocity_rad_s",
            ..
        })
    ));
}
