use sonium_sim_propellers::{
    BladeElementOperatingPoint, Point3, SteadyFreeWakeFilament, SteadyFreeWakeNode,
    SteadyFreeWakeResult, TransientFreeWakeError, TransientFreeWakeOptions,
    advance_prescribed_free_wake, advance_prescribed_free_wake_with_shedding,
};

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn node(position: Point3) -> SteadyFreeWakeNode {
    SteadyFreeWakeNode {
        position,
        induced_velocity_m_s: point(0.0, 0.0, 0.0),
        age_s: 0.0,
    }
}

fn filament(
    blade_index: usize,
    circulation_m2_s: f64,
    nodes: Vec<SteadyFreeWakeNode>,
) -> SteadyFreeWakeFilament {
    SteadyFreeWakeFilament {
        blade_index,
        base_azimuth_rad: blade_index as f64,
        radius_m: 1.0,
        circulation_m2_s,
        axial_convection_speed_m_s: 2.0,
        nodes,
    }
}

fn wake(filaments: Vec<SteadyFreeWakeFilament>) -> SteadyFreeWakeResult {
    let node_count = filaments.iter().map(|item| item.nodes.len()).sum();
    SteadyFreeWakeResult {
        operating_point: BladeElementOperatingPoint {
            rpm: 12_000.0,
            freestream_m_s: 10.0,
            density_kg_m3: 1.2,
        },
        blade_count: filaments.len().max(1),
        iterations: 1,
        final_residual_m: 0.0,
        converged: true,
        node_count,
        biot_savart_evaluation_count: 1,
        core_radius_m: 0.1,
        filaments,
    }
}

fn two_node_zero_circulation_wake() -> SteadyFreeWakeResult {
    wake(vec![filament(
        0,
        0.0,
        vec![node(point(0.0, 0.0, 0.0)), node(point(1.0, 0.0, 0.0))],
    )])
}

#[test]
fn uniform_background_convection_matches_explicit_euler_and_allows_zero_circulation() {
    let options = TransientFreeWakeOptions {
        time_step_s: 0.25,
        time_steps: 2,
        background_velocity_m_s: point(0.0, 0.0, 2.0),
        max_node_displacement_m: 0.6,
        ..TransientFreeWakeOptions::default()
    };

    let result = advance_prescribed_free_wake(&two_node_zero_circulation_wake(), options)
        .expect("uniform convection should be admitted");

    assert!(result.is_finite());
    assert_eq!(result.total_biot_savart_evaluation_count, 4);
    assert_eq!(result.filaments[0].nodes[0].position, point(0.0, 0.0, 1.0));
    assert_eq!(result.filaments[0].nodes[1].position, point(1.0, 0.0, 1.0));
    assert_eq!(
        result.filaments[0].nodes[0].induced_velocity_m_s,
        point(0.0, 0.0, 0.0)
    );
    assert_eq!(result.filaments[0].nodes[0].age_s, 0.5);
}

#[test]
fn parallel_filament_induction_moves_the_receiving_filament() {
    let initial = wake(vec![
        filament(
            0,
            1.0,
            vec![node(point(0.0, 0.0, -0.5)), node(point(0.0, 0.0, 0.5))],
        ),
        filament(
            1,
            0.0,
            vec![node(point(1.0, 0.0, -0.5)), node(point(1.0, 0.0, 0.5))],
        ),
    ]);
    let options = TransientFreeWakeOptions {
        time_step_s: 1.0e-3,
        time_steps: 1,
        background_velocity_m_s: point(0.0, 0.0, 0.0),
        max_node_displacement_m: 1.0e-2,
        ..TransientFreeWakeOptions::default()
    };

    let result = advance_prescribed_free_wake(&initial, options)
        .expect("parallel finite filament induction should be admitted");

    assert!(result.steps[0].maximum_induced_speed_m_s > 0.0);
    assert!(
        result.filaments[1]
            .nodes
            .iter()
            .all(|item| item.position.y > 0.0)
    );
    assert_eq!(result.filaments[0].nodes[0].position, point(0.0, 0.0, -0.5));
}

#[test]
fn replay_is_deterministic_and_core_growth_tracks_age() {
    let initial = two_node_zero_circulation_wake();
    let options = TransientFreeWakeOptions {
        time_step_s: 0.5,
        time_steps: 2,
        background_velocity_m_s: point(0.0, 0.0, 0.0),
        core_growth_m_s: 0.2,
        max_node_displacement_m: 1.0,
        ..TransientFreeWakeOptions::default()
    };

    let first = advance_prescribed_free_wake(&initial, options).expect("first replay");
    let second = advance_prescribed_free_wake(&initial, options).expect("second replay");

    assert_eq!(first, second);
    assert!((first.final_core_radius_m - 0.3).abs() < 1.0e-15);
    assert_eq!(first.steps[1].maximum_age_s, 1.0);
}

#[test]
fn budgets_and_invalid_states_fail_closed() {
    let initial = two_node_zero_circulation_wake();

    let budget_error = advance_prescribed_free_wake(
        &initial,
        TransientFreeWakeOptions {
            max_biot_savart_evaluations_per_step: 1,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect_err("node/segment work must honor the budget");
    assert!(matches!(
        budget_error,
        TransientFreeWakeError::BiotSavartBudgetExceeded {
            step: 1,
            required: 2,
            maximum: 1
        }
    ));

    let displacement_error = advance_prescribed_free_wake(
        &initial,
        TransientFreeWakeOptions {
            time_step_s: 1.0,
            time_steps: 1,
            background_velocity_m_s: point(0.0, 0.0, 2.0),
            max_node_displacement_m: 0.5,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect_err("explicit motion must honor the displacement screen");
    assert!(matches!(
        displacement_error,
        TransientFreeWakeError::NodeDisplacementExceeded {
            step: 1,
            maximum_m: 0.5,
            ..
        }
    ));

    let invalid_options = advance_prescribed_free_wake(
        &initial,
        TransientFreeWakeOptions {
            time_steps: 0,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect_err("zero transient steps are not a complete result");
    assert!(matches!(
        invalid_options,
        TransientFreeWakeError::InvalidOptions {
            field: "time_steps",
            ..
        }
    ));

    let mut unconverged = initial.clone();
    unconverged.converged = false;
    let invalid_initial =
        advance_prescribed_free_wake(&unconverged, TransientFreeWakeOptions::default())
            .expect_err("the transient reference requires a converged initial wake");
    assert!(matches!(
        invalid_initial,
        TransientFreeWakeError::InvalidInitialWake { field: "state", .. }
    ));

    let count_overflow = advance_prescribed_free_wake(
        &initial,
        TransientFreeWakeOptions {
            time_steps: usize::MAX,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect_err("total transient Biot--Savart work must fit usize");
    assert!(matches!(
        count_overflow,
        TransientFreeWakeError::EvaluationCountOverflow { .. }
    ));
}

#[test]
fn shedding_preserves_anchors_and_grows_downstream_topology() {
    let result = advance_prescribed_free_wake_with_shedding(
        &two_node_zero_circulation_wake(),
        TransientFreeWakeOptions {
            time_step_s: 0.25,
            time_steps: 2,
            background_velocity_m_s: point(0.0, 0.0, 2.0),
            max_node_displacement_m: 0.6,
            max_node_count: 4,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect("bounded shedding should preserve anchors and add nodes");

    assert!(result.is_finite());
    assert_eq!(result.filaments[0].nodes.len(), 4);
    assert_eq!(result.filaments[0].nodes[0].position, point(0.0, 0.0, 0.0));
    assert_eq!(result.filaments[0].nodes[1].position, point(0.0, 0.0, 0.0));
    assert_eq!(result.filaments[0].nodes[2].position, point(0.0, 0.0, 0.5));
    assert_eq!(result.filaments[0].nodes[3].position, point(1.0, 0.0, 1.0));
    assert_eq!(result.filaments[0].nodes[0].age_s, 0.0);
    assert!(result.filaments[0].nodes[1].age_s < result.filaments[0].nodes[2].age_s);
}

#[test]
fn shedding_rejects_growth_past_the_node_budget() {
    let error = advance_prescribed_free_wake_with_shedding(
        &two_node_zero_circulation_wake(),
        TransientFreeWakeOptions {
            time_step_s: 0.25,
            time_steps: 2,
            background_velocity_m_s: point(0.0, 0.0, 0.0),
            max_node_count: 3,
            ..TransientFreeWakeOptions::default()
        },
    )
    .expect_err("shedding must honor the growing node budget");

    assert!(matches!(
        error,
        TransientFreeWakeError::NodeBudgetExceeded {
            count: 4,
            maximum: 3
        }
    ));
}
