use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, SteadyFreeWakeError, SteadyFreeWakeOptions,
    evaluate_steady_free_wake,
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
    .expect("valid public free-wake polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public free-wake model")
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
        .expect("converged public free-wake BEM")
}

fn options() -> SteadyFreeWakeOptions {
    SteadyFreeWakeOptions {
        wake_turns: 1,
        segments_per_turn: 8,
        relaxation: 0.05,
        max_iterations: 100,
        convergence_tolerance_m: 1.0e-4,
        max_node_count: 10_000,
        max_biot_savart_evaluations: 1_000_000,
        ..SteadyFreeWakeOptions::default()
    }
}

#[test]
fn public_steady_free_wake_api_retains_bem_point_nodes_and_budget_evidence() {
    let model = model();
    let bem = bem(&model);
    let result = evaluate_steady_free_wake(&model, &bem, options()).expect("public steady wake");

    assert_eq!(result.operating_point, bem.loads.operating_point);
    assert_eq!(result.blade_count, model.n_blades());
    assert_eq!(
        result.filaments.len(),
        model.n_blades() * (model.stations().len() + 1)
    );
    assert_eq!(
        result.node_count,
        result
            .filaments
            .iter()
            .map(|filament| filament.nodes.len())
            .sum::<usize>()
    );
    assert!(result.iterations > 0);
    assert!(result.final_residual_m <= options().convergence_tolerance_m);
    assert!(result.biot_savart_evaluation_count > 0);
    assert!(result.is_finite());
}

#[test]
fn public_steady_free_wake_api_rejects_unconverged_and_over_budget_inputs() {
    let model = model();
    let bem = bem(&model);

    let mut unconverged = bem.clone();
    unconverged.converged = false;
    assert!(matches!(
        evaluate_steady_free_wake(&model, &unconverged, options()),
        Err(SteadyFreeWakeError::InvalidBemResult { field: "converged" })
    ));

    assert!(matches!(
        evaluate_steady_free_wake(
            &model,
            &bem,
            SteadyFreeWakeOptions {
                max_biot_savart_evaluations: 1,
                ..options()
            }
        ),
        Err(SteadyFreeWakeError::EvaluationBudgetExceeded { .. })
    ));
}
