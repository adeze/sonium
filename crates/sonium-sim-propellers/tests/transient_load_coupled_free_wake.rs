use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, SteadyFreeWakeOptions,
    TransientLoadCoupledFreeWakeError, TransientLoadCoupledFreeWakeOptions,
    evaluate_transient_load_coupled_free_wake,
};

fn model(lift_scale: f64) -> BladeElementModel {
    let source = PropellerSource::drone_5inch(12_000.0);
    let stations = vec![
        BladeElementStation {
            radius_m: 0.018,
            chord_m: 0.018,
            pitch_deg: 24.0,
        },
        BladeElementStation {
            radius_m: 0.036,
            chord_m: 0.014,
            pitch_deg: 21.0,
        },
        BladeElementStation {
            radius_m: 0.054,
            chord_m: 0.010,
            pitch_deg: 18.0,
        },
    ];
    let polar = AirfoilPolar::try_new(vec![
        AirfoilPolarPoint {
            angle_of_attack_deg: -20.0,
            lift_coefficient: -0.8 * lift_scale,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 0.0,
            lift_coefficient: 0.0,
            drag_coefficient: 0.02,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 20.0,
            lift_coefficient: 0.8 * lift_scale,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 40.0,
            lift_coefficient: 0.9 * lift_scale,
            drag_coefficient: 0.16,
        },
    ])
    .expect("valid transient load-coupled polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid transient load-coupled model")
}

fn operating_point() -> BladeElementOperatingPoint {
    BladeElementOperatingPoint {
        rpm: 12_000.0,
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    }
}

fn wake_options() -> SteadyFreeWakeOptions {
    SteadyFreeWakeOptions {
        wake_turns: 1,
        segments_per_turn: 8,
        core_radius_m: 0.05,
        relaxation: 0.05,
        convergence_tolerance_m: 1.0e-4,
        max_iterations: 100,
        max_node_count: 10_000,
        max_biot_savart_evaluations: 1_000_000,
        ..SteadyFreeWakeOptions::default()
    }
}

fn options() -> TransientLoadCoupledFreeWakeOptions {
    TransientLoadCoupledFreeWakeOptions {
        time_step_s: 1.0e-4,
        time_steps: 3,
        max_bem_evaluations: 8,
        max_total_wake_evaluations: 5_000_000,
        max_node_displacement_m: 5.0e-3,
        max_biot_savart_evaluations_per_step: 1_000_000,
        wake_options: wake_options(),
        ..TransientLoadCoupledFreeWakeOptions::default()
    }
}

#[test]
fn load_coupled_transient_updates_bem_inflow_and_wake_with_bounded_counts() {
    let result = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        options(),
    )
    .expect("bounded transient load coupling should converge each checked BEM step");

    assert!(result.is_finite());
    assert_eq!(result.steps.len(), 3);
    assert_eq!(result.total_bem_evaluation_count, 4);
    assert!(result.total_wake_evaluation_count > 0);
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.bem.converged && step.bem.loads.strips.len() == 3)
    );
    assert!(
        result
            .steps
            .iter()
            .any(|step| step.maximum_inflow_update_m_s > 0.0)
    );
    assert!(
        result
            .steps
            .iter()
            .any(|step| step.maximum_circulation_update_m2_s > 0.0)
    );
    assert!(
        result
            .filaments
            .iter()
            .all(|filament| filament.nodes.first().is_some_and(|node| node.age_s == 0.0))
    );
    assert!(
        result.filaments.iter().all(|filament| filament
            .nodes
            .iter()
            .skip(2)
            .all(|node| node.age_s > 0.0))
    );
    assert!(result.filaments[0].nodes.len() > wake_options().segments_per_turn);
}

#[test]
fn load_coupled_transient_replay_is_deterministic() {
    let first = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        options(),
    )
    .expect("first deterministic run");
    let second = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        options(),
    )
    .expect("second deterministic run");

    assert_eq!(first, second);
}

#[test]
fn zero_lift_load_coupling_remains_finite_with_zero_circulation() {
    let result = evaluate_transient_load_coupled_free_wake(
        &model(0.0),
        operating_point(),
        BladeElementBemOptions::default(),
        options(),
    )
    .expect("zero-lift transient wake should remain a valid kinematic state");

    assert!(result.is_finite());
    assert!(
        result
            .filaments
            .iter()
            .all(|filament| filament.circulation_m2_s == 0.0)
    );
}

#[test]
fn coupling_rejects_bem_and_motion_budgets_before_promotion() {
    let bem_budget = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        TransientLoadCoupledFreeWakeOptions {
            max_bem_evaluations: 3,
            ..options()
        },
    )
    .expect_err("initial plus three step BEM evaluations exceed the budget");
    assert!(matches!(
        bem_budget,
        TransientLoadCoupledFreeWakeError::BudgetExceeded {
            field: "bem_evaluations",
            required: 4,
            maximum: 3
        }
    ));

    let motion_budget = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        TransientLoadCoupledFreeWakeOptions {
            max_node_displacement_m: 1.0e-4,
            ..options()
        },
    )
    .expect_err("explicit wake motion must honor the displacement budget");
    assert!(matches!(
        motion_budget,
        TransientLoadCoupledFreeWakeError::StepWake { .. }
    ));

    let invalid_options = evaluate_transient_load_coupled_free_wake(
        &model(0.1),
        operating_point(),
        BladeElementBemOptions::default(),
        TransientLoadCoupledFreeWakeOptions {
            inflow_relaxation: 0.0,
            ..options()
        },
    )
    .expect_err("zero inflow relaxation is outside the coupling contract");
    assert!(matches!(
        invalid_options,
        TransientLoadCoupledFreeWakeError::InvalidOptions {
            field: "inflow_relaxation",
            ..
        }
    ));
}
