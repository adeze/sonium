use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, MultiRotorBemError, MultiRotorBemOptions,
    MultiRotorBemRotor, evaluate_multi_rotor_bem,
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
    .expect("valid polar");
    BladeElementModel::try_from_source(&source, stations, polar).expect("valid BEM model")
}

fn operating_point() -> BladeElementOperatingPoint {
    BladeElementOperatingPoint {
        rpm: 12_000.0,
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    }
}

fn rotor(center_m: [f64; 3], axis: [f64; 3]) -> MultiRotorBemRotor {
    MultiRotorBemRotor {
        center_m,
        axis,
        model: model(),
        operating_point: operating_point(),
    }
}

fn options() -> MultiRotorBemOptions {
    MultiRotorBemOptions {
        bem_options: BladeElementBemOptions {
            max_iterations: 150,
            convergence_tolerance: 1.0e-8,
            relaxation: 0.35,
            ..BladeElementBemOptions::default()
        },
        max_iterations: 30,
        convergence_tolerance: 1.0e-3,
        relaxation: 0.3,
        max_bem_evaluations: 128,
        ..MultiRotorBemOptions::default()
    }
}

#[test]
fn single_rotor_reduces_to_checked_bem_solution() {
    let baseline_model = model();
    let baseline = baseline_model
        .evaluate_bem(operating_point(), options().bem_options)
        .expect("baseline BEM should converge");
    let result = evaluate_multi_rotor_bem(&[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])], options())
        .expect("single rotor should converge");

    assert_eq!(result.iterations, 1);
    assert_eq!(result.pairwise_induced_velocity_m_s, vec![vec![0.0]]);
    assert_eq!(result.rotors[0].external_induced_velocity_m_s, 0.0);
    assert!((result.rotors[0].bem_result.loads.thrust_n - baseline.loads.thrust_n).abs() < 1.0e-12);
    assert!(result.is_finite());
}

#[test]
fn downstream_wake_changes_bem_loads_and_retains_pairwise_evidence() {
    let baseline = model()
        .evaluate_bem(operating_point(), options().bem_options)
        .expect("baseline BEM should converge");
    let result = evaluate_multi_rotor_bem(
        &[
            rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            rotor([0.0, 0.0, 0.12], [0.0, 0.0, 1.0]),
        ],
        options(),
    )
    .expect("aligned downstream rotors should converge");

    assert!(result.pairwise_induced_velocity_m_s[1][0] > 0.0);
    assert_eq!(result.pairwise_induced_velocity_m_s[0][1], 0.0);
    assert!(result.rotors[1].external_induced_velocity_m_s > 0.0);
    assert!((result.rotors[1].bem_result.loads.thrust_n - baseline.loads.thrust_n).abs() > 1.0e-8);
    assert!(result.maximum_relative_load_change <= options().convergence_tolerance);
    assert!(result.is_finite());
}

#[test]
fn transverse_receiver_has_zero_projected_wake_and_same_load() {
    let baseline = model()
        .evaluate_bem(operating_point(), options().bem_options)
        .expect("baseline BEM should converge");
    let result = evaluate_multi_rotor_bem(
        &[
            rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            rotor([0.0, 0.0, 0.12], [1.0, 0.0, 0.0]),
        ],
        options(),
    )
    .expect("transverse receiver should converge");

    assert!(result.pairwise_induced_velocity_m_s[1][0].abs() < 1.0e-12);
    assert!(result.rotors[1].external_induced_velocity_m_s.abs() < 1.0e-12);
    assert!((result.rotors[1].bem_result.loads.thrust_n - baseline.loads.thrust_n).abs() < 1.0e-12);
}

#[test]
fn invalid_geometry_and_work_budgets_fail_closed() {
    let invalid_axis =
        evaluate_multi_rotor_bem(&[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])], options());
    assert!(matches!(
        invalid_axis,
        Err(MultiRotorBemError::InvalidRotor { field: "axis", .. })
    ));

    let budget = evaluate_multi_rotor_bem(
        &[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])],
        MultiRotorBemOptions {
            max_bem_evaluations: 1,
            ..options()
        },
    );
    assert!(matches!(
        budget,
        Err(MultiRotorBemError::BemEvaluationBudgetExceeded {
            required: 2,
            maximum: 1,
        })
    ));
}

#[test]
fn malformed_outer_options_fail_before_nested_bem() {
    let result = evaluate_multi_rotor_bem(
        &[rotor([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])],
        MultiRotorBemOptions {
            relaxation: 0.0,
            ..options()
        },
    );
    assert!(matches!(
        result,
        Err(MultiRotorBemError::InvalidOptions {
            field: "relaxation",
            ..
        })
    ));
}
