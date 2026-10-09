use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, SteadyFreeWakeConvergenceError,
    SteadyFreeWakeConvergenceOptions, SteadyFreeWakeDiscretization, SteadyFreeWakeOptions,
    evaluate_steady_free_wake_convergence,
};

fn model() -> BladeElementModel {
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
            lift_coefficient: -0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 0.0,
            lift_coefficient: 0.0,
            drag_coefficient: 0.02,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 20.0,
            lift_coefficient: 0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 40.0,
            lift_coefficient: 0.9,
            drag_coefficient: 0.16,
        },
    ])
    .expect("valid public wake-convergence polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public wake-convergence model")
}

fn bem(model: &BladeElementModel) -> sonium_sim_propellers::BladeElementBemResult {
    model
        .evaluate_bem(
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 10.0,
                density_kg_m3: 1.204,
            },
            BladeElementBemOptions::default(),
        )
        .expect("converged wake-convergence BEM")
}

fn options() -> SteadyFreeWakeConvergenceOptions {
    SteadyFreeWakeConvergenceOptions {
        levels: vec![
            SteadyFreeWakeDiscretization {
                wake_turns: 1,
                segments_per_turn: 8,
            },
            SteadyFreeWakeDiscretization {
                wake_turns: 1,
                segments_per_turn: 16,
            },
            SteadyFreeWakeDiscretization {
                wake_turns: 2,
                segments_per_turn: 16,
            },
        ],
        tolerance: 0.25,
        wake_options: SteadyFreeWakeOptions {
            relaxation: 0.05,
            max_iterations: 100,
            convergence_tolerance_m: 1.0e-4,
            max_node_count: 10_000,
            max_biot_savart_evaluations: 1_000_000,
            ..SteadyFreeWakeOptions::default()
        },
    }
}

#[test]
fn public_wake_convergence_retains_finite_level_and_work_evidence() {
    let model = model();
    let result = evaluate_steady_free_wake_convergence(&model, &bem(&model), options())
        .expect("wake convergence");

    assert!(result.converged);
    assert!(result.is_finite());
    assert_eq!(result.levels.len(), 3);
    assert_eq!(result.wakes.len(), 3);
    assert!(
        result
            .levels
            .iter()
            .all(|level| level.node_count > 0 && level.biot_savart_evaluation_count > 0)
    );
    assert!(result.final_relative_induced_velocity_change <= result.tolerance);
    assert!(result.levels[1].node_count > result.levels[0].node_count);
    assert!(result.levels[2].node_count > result.levels[1].node_count);
}

#[test]
fn wake_convergence_is_deterministic_and_can_report_nonconvergence() {
    let model = model();
    let first = evaluate_steady_free_wake_convergence(&model, &bem(&model), options())
        .expect("first wake convergence");
    let second = evaluate_steady_free_wake_convergence(&model, &bem(&model), options())
        .expect("second wake convergence");
    assert_eq!(first, second);

    let not_converged = evaluate_steady_free_wake_convergence(
        &model,
        &bem(&model),
        SteadyFreeWakeConvergenceOptions {
            tolerance: 1.0e-12,
            ..options()
        },
    )
    .expect("finite non-converged ladder");
    assert!(!not_converged.converged);
    assert!(not_converged.is_finite());
}

#[test]
fn wake_convergence_rejects_invalid_ladders_and_kernel_budgets() {
    let model = model();
    let invalid = evaluate_steady_free_wake_convergence(
        &model,
        &bem(&model),
        SteadyFreeWakeConvergenceOptions {
            levels: vec![
                SteadyFreeWakeDiscretization {
                    wake_turns: 2,
                    segments_per_turn: 16,
                },
                SteadyFreeWakeDiscretization {
                    wake_turns: 1,
                    segments_per_turn: 8,
                },
            ],
            ..options()
        },
    );
    assert!(matches!(
        invalid,
        Err(SteadyFreeWakeConvergenceError::InvalidConfiguration {
            field: "levels",
            ..
        })
    ));

    let budget = evaluate_steady_free_wake_convergence(
        &model,
        &bem(&model),
        SteadyFreeWakeConvergenceOptions {
            wake_options: SteadyFreeWakeOptions {
                max_biot_savart_evaluations: 1,
                ..options().wake_options
            },
            ..options()
        },
    );
    assert!(matches!(
        budget,
        Err(SteadyFreeWakeConvergenceError::WakeEvaluation { level_index: 0, .. })
    ));
}
