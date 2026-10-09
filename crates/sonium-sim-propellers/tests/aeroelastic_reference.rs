use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeAeroelasticError, BladeAeroelasticOptions,
    BladeElementBemOptions, BladeElementModel, BladeElementOperatingPoint, BladeElementStation,
    evaluate_blade_aeroelastic_reference,
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
    .expect("valid public aeroelastic polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public aeroelastic model")
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
        .expect("converged aeroelastic BEM")
}

#[test]
fn public_aeroelastic_reference_is_finite_and_retains_evidence() {
    let model = model();
    let result = evaluate_blade_aeroelastic_reference(
        &model,
        &bem(&model),
        BladeAeroelasticOptions::default(),
    )
    .expect("aeroelastic reference");

    assert!(result.is_finite());
    assert_eq!(result.stations.len(), model.stations().len());
    assert!(result.first_flapwise_natural_frequency_hz > 0.0);
    assert!(result.blade_passing_frequency_hz > 0.0);
    assert!(result.nearest_harmonic > 0);
    assert!(result.max_static_deflection_m > 0.0);
    assert!(result.max_alternating_stress_pa > 0.0);
    assert!(result.applied_cycles > 0.0);
    assert!(result.minimum_fatigue_life_cycles > 0.0);
}

#[test]
fn modal_separation_threshold_is_explicit_and_thickness_changes_frequency() {
    let model = model();
    let base_options = BladeAeroelasticOptions {
        minimum_modal_separation_ratio: 1.0e-12,
        ..BladeAeroelasticOptions::default()
    };
    let base = evaluate_blade_aeroelastic_reference(&model, &bem(&model), base_options)
        .expect("base aeroelastic reference");
    assert!(base.modal_separation_passed);

    let strict = evaluate_blade_aeroelastic_reference(
        &model,
        &bem(&model),
        BladeAeroelasticOptions {
            minimum_modal_separation_ratio: base.minimum_relative_modal_separation * 1.1,
            ..base_options
        },
    )
    .expect("strict modal reference");
    assert!(!strict.modal_separation_passed);

    let thin = evaluate_blade_aeroelastic_reference(
        &model,
        &bem(&model),
        BladeAeroelasticOptions {
            thickness_fraction: 0.06,
            ..base_options
        },
    )
    .expect("thin aeroelastic reference");
    assert!(thin.first_flapwise_natural_frequency_hz < base.first_flapwise_natural_frequency_hz);
    assert!(thin.max_static_deflection_m > base.max_static_deflection_m);
}

#[test]
fn exposure_and_cyclic_fraction_change_fatigue_damage() {
    let model = model();
    let short = evaluate_blade_aeroelastic_reference(
        &model,
        &bem(&model),
        BladeAeroelasticOptions {
            operating_hours: 0.1,
            aerodynamic_load_amplitude_fraction: 0.05,
            ..BladeAeroelasticOptions::default()
        },
    )
    .expect("short fatigue exposure");
    let long = evaluate_blade_aeroelastic_reference(
        &model,
        &bem(&model),
        BladeAeroelasticOptions {
            operating_hours: 10.0,
            aerodynamic_load_amplitude_fraction: 0.2,
            ..BladeAeroelasticOptions::default()
        },
    )
    .expect("long fatigue exposure");
    assert!(long.applied_cycles > short.applied_cycles);
    assert!(long.maximum_fatigue_damage > short.maximum_fatigue_damage);
    assert!(long.minimum_fatigue_life_cycles < short.minimum_fatigue_life_cycles);
}

#[test]
fn zero_aerodynamic_load_has_zero_deflection_and_damage() {
    let model = model();
    let mut zero_bem = bem(&model);
    for strip in &mut zero_bem.loads.strips {
        strip.axial_force_per_m_n = 0.0;
    }
    let result =
        evaluate_blade_aeroelastic_reference(&model, &zero_bem, BladeAeroelasticOptions::default())
            .expect("zero-load aeroelastic reference");
    assert!(result.is_finite());
    assert_eq!(result.max_static_deflection_m, 0.0);
    assert_eq!(result.max_alternating_stress_pa, 0.0);
    assert_eq!(result.maximum_fatigue_damage, 0.0);
    assert!(result.fatigue_passed);
}

#[test]
fn invalid_options_and_bem_state_fail_closed() {
    let model = model();
    let good_bem = bem(&model);
    assert!(matches!(
        evaluate_blade_aeroelastic_reference(
            &model,
            &good_bem,
            BladeAeroelasticOptions {
                youngs_modulus_pa: 0.0,
                ..BladeAeroelasticOptions::default()
            }
        ),
        Err(BladeAeroelasticError::InvalidOptions {
            field: "youngs_modulus_pa",
            ..
        })
    ));
    assert!(matches!(
        evaluate_blade_aeroelastic_reference(
            &model,
            &good_bem,
            BladeAeroelasticOptions {
                max_harmonic: 0,
                ..BladeAeroelasticOptions::default()
            }
        ),
        Err(BladeAeroelasticError::InvalidOptions {
            field: "max_harmonic",
            ..
        })
    ));

    let mut unconverged = good_bem.clone();
    unconverged.converged = false;
    assert!(matches!(
        evaluate_blade_aeroelastic_reference(
            &model,
            &unconverged,
            BladeAeroelasticOptions::default()
        ),
        Err(BladeAeroelasticError::InvalidBemResult { field: "converged" })
    ));

    let mut malformed = good_bem;
    malformed.loads.strips.pop();
    assert!(matches!(
        evaluate_blade_aeroelastic_reference(
            &model,
            &malformed,
            BladeAeroelasticOptions::default()
        ),
        Err(BladeAeroelasticError::StationCountMismatch { .. })
    ));
}
