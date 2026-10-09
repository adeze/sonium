use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementModel, BladeElementOperatingPoint,
    BladeElementStation, CentrifugalModalError, CentrifugalModalOptions,
    evaluate_centrifugal_stiffened_modal_reference,
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
    .expect("valid modal polar");
    BladeElementModel::try_from_source(&source, stations, polar).expect("valid modal model")
}

fn operating_point(rpm: f64) -> BladeElementOperatingPoint {
    BladeElementOperatingPoint {
        rpm,
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    }
}

fn options() -> CentrifugalModalOptions {
    CentrifugalModalOptions {
        minimum_relative_separation_ratio: 1.0e-12,
        ..CentrifugalModalOptions::default()
    }
}

#[test]
fn centrifugal_stiffening_produces_finite_modal_and_campbell_evidence() {
    let result = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(12_000.0),
        options(),
    )
    .expect("finite centrifugal modal reference");

    assert!(result.is_finite());
    assert!(result.is_feasible());
    assert!(result.stiffened_natural_frequency_hz > result.unstiffened_natural_frequency_hz);
    assert!(result.root_centrifugal_load_n > 0.0);
    assert!(result.centrifugal_modal_stiffness_n_m > 0.0);
    assert!(result.centrifugal_stiffening_ratio > 0.0);
    assert!((1..=8).contains(&result.nearest_harmonic));
}

#[test]
fn centrifugal_load_and_geometric_stiffness_follow_rpm_squared_scaling() {
    let twelve = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(12_000.0),
        options(),
    )
    .expect("12 kRPM modal reference");
    let twenty_four = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(24_000.0),
        options(),
    )
    .expect("24 kRPM modal reference");

    assert!(
        (twenty_four.root_centrifugal_load_n / twelve.root_centrifugal_load_n - 4.0).abs()
            < 1.0e-10
    );
    assert!(
        (twenty_four.centrifugal_modal_stiffness_n_m / twelve.centrifugal_modal_stiffness_n_m
            - 4.0)
            .abs()
            < 1.0e-10
    );
    assert!(
        (twenty_four.blade_passing_frequency_hz / twelve.blade_passing_frequency_hz - 2.0).abs()
            < 1.0e-12
    );
    assert!(twenty_four.stiffened_natural_frequency_hz > twelve.stiffened_natural_frequency_hz);
}

#[test]
fn centrifugal_modal_reference_is_deterministic() {
    let first = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(12_000.0),
        options(),
    )
    .expect("first modal reference");
    let second = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(12_000.0),
        options(),
    )
    .expect("second modal reference");

    assert_eq!(first, second);
}

#[test]
fn centrifugal_modal_reference_rejects_invalid_options_and_operation() {
    let invalid_options = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        operating_point(12_000.0),
        CentrifugalModalOptions {
            integration_points: 3,
            ..options()
        },
    )
    .expect_err("quadrature needs a positive resolution");
    assert!(matches!(
        invalid_options,
        CentrifugalModalError::InvalidOptions {
            field: "integration_points",
            ..
        }
    ));

    let invalid_operation = evaluate_centrifugal_stiffened_modal_reference(
        &model(),
        BladeElementOperatingPoint {
            rpm: 0.0,
            ..operating_point(12_000.0)
        },
        options(),
    )
    .expect_err("zero RPM is not a rotating modal reference");
    assert!(matches!(
        invalid_operation,
        CentrifugalModalError::InvalidOperatingPoint { field: "rpm" }
    ));
}
