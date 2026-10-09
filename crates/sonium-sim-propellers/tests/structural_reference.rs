use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, BladeStructuralError, BladeStructuralOptions,
    evaluate_blade_structural_reference,
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
    .expect("valid public structural polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public structural model")
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
        .expect("converged public structural BEM")
}

#[test]
fn public_structural_reference_reports_stress_mass_and_margin_evidence() {
    let model = model();
    let bem = bem(&model);
    let result = evaluate_blade_structural_reference(
        &model,
        &bem,
        BladeStructuralOptions {
            allowable_stress_pa: 1.0e12,
            ..BladeStructuralOptions::default()
        },
    )
    .expect("public structural reference");

    assert_eq!(result.operating_point, bem.loads.operating_point);
    assert_eq!(result.stations.len(), model.stations().len());
    assert!(result.blade_mass_kg > 0.0);
    assert!(result.root_centrifugal_force_n > 0.0);
    assert!(result.root_bending_moment_nm >= 0.0);
    assert!(result.max_combined_stress_pa > 0.0);
    assert!(result.is_finite());
    assert!(result.is_feasible());
}

#[test]
fn public_structural_reference_rejects_invalid_material_and_unconverged_bem() {
    let model = model();
    let bem = bem(&model);
    assert!(matches!(
        evaluate_blade_structural_reference(
            &model,
            &bem,
            BladeStructuralOptions {
                material_density_kg_m3: 0.0,
                ..BladeStructuralOptions::default()
            }
        ),
        Err(BladeStructuralError::InvalidOptions {
            field: "material_density_kg_m3",
            ..
        })
    ));

    let mut unconverged = bem;
    unconverged.converged = false;
    assert!(matches!(
        evaluate_blade_structural_reference(
            &model,
            &unconverged,
            BladeStructuralOptions::default()
        ),
        Err(BladeStructuralError::InvalidBemResult { field: "converged" })
    ));
}
