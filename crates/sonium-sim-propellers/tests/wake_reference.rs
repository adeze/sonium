use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, PrescribedHelicalWakeError,
    PrescribedHelicalWakeOptions, evaluate_prescribed_helical_wake,
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
    .expect("valid public-test polar");
    BladeElementModel::try_from_source(&source, stations, polar).expect("valid public-test model")
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
        .expect("converged public-test BEM")
}

#[test]
fn public_prescribed_wake_api_retains_bem_point_and_budget_evidence() {
    let model = model();
    let bem = bem(&model);
    let result = evaluate_prescribed_helical_wake(
        &model,
        &bem,
        PrescribedHelicalWakeOptions {
            wake_turns: 2,
            segments_per_turn: 16,
            ..PrescribedHelicalWakeOptions::default()
        },
    )
    .expect("public prescribed wake");
    assert_eq!(result.operating_point, bem.loads.operating_point);
    assert_eq!(result.stations.len(), model.stations().len());
    assert!(result.biot_savart_evaluation_count > result.segment_count);
    assert!(result.is_finite());
}

#[test]
fn public_prescribed_wake_api_rejects_budget_and_unconverged_inputs() {
    let model = model();
    let bem = bem(&model);
    assert!(matches!(
        evaluate_prescribed_helical_wake(
            &model,
            &bem,
            PrescribedHelicalWakeOptions {
                max_biot_savart_evaluations: 1,
                ..PrescribedHelicalWakeOptions::default()
            },
        ),
        Err(PrescribedHelicalWakeError::InvalidOptions {
            field: "biot_savart_evaluations",
            ..
        })
    ));
    let mut unconverged = bem;
    unconverged.converged = false;
    assert!(matches!(
        evaluate_prescribed_helical_wake(
            &model,
            &unconverged,
            PrescribedHelicalWakeOptions::default(),
        ),
        Err(PrescribedHelicalWakeError::InvalidBemResult { field: "converged" })
    ));
}
