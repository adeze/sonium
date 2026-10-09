use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, MotorMatchingError,
    MotorPropellerMatchCondition, MotorPropellerMatchOptions, MotorTorqueSpeedCurve,
    MotorTorqueSpeedPoint, evaluate_motor_propeller_match,
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
    .expect("valid public motor-matching polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public motor-matching model")
}

fn condition() -> MotorPropellerMatchCondition {
    MotorPropellerMatchCondition {
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    }
}

fn options() -> MotorPropellerMatchOptions {
    MotorPropellerMatchOptions {
        rpm_min: 6_000.0,
        rpm_max: 12_000.0,
        torque_tolerance_nm: 1.0e-7,
        rpm_tolerance: 1.0e-3,
        ..MotorPropellerMatchOptions::default()
    }
}

fn crossing_motor(model: &BladeElementModel) -> MotorTorqueSpeedCurve {
    let options = options();
    let lower = model
        .evaluate_bem(
            BladeElementOperatingPoint {
                rpm: options.rpm_min,
                freestream_m_s: condition().freestream_m_s,
                density_kg_m3: condition().density_kg_m3,
            },
            BladeElementBemOptions::default(),
        )
        .expect("lower public BEM endpoint");
    let upper = model
        .evaluate_bem(
            BladeElementOperatingPoint {
                rpm: options.rpm_max,
                freestream_m_s: condition().freestream_m_s,
                density_kg_m3: condition().density_kg_m3,
            },
            BladeElementBemOptions::default(),
        )
        .expect("upper public BEM endpoint");
    assert!(upper.loads.torque_nm > 0.0);
    let scale = lower
        .loads
        .torque_nm
        .abs()
        .max(upper.loads.torque_nm.abs())
        .max(1.0e-3);
    MotorTorqueSpeedCurve::try_new(vec![
        MotorTorqueSpeedPoint {
            rpm: options.rpm_min,
            available_torque_nm: lower.loads.torque_nm + scale,
        },
        MotorTorqueSpeedPoint {
            rpm: options.rpm_max,
            available_torque_nm: 0.0,
        },
    ])
    .expect("valid public crossing motor")
}

#[test]
fn public_motor_matching_api_reports_balanced_operating_point_and_work() {
    let model = model();
    let motor = crossing_motor(&model);
    let result = evaluate_motor_propeller_match(&model, &motor, condition(), options())
        .expect("public motor/propeller match");

    assert_eq!(result.condition, condition());
    assert_eq!(
        result.operating_point.freestream_m_s,
        condition().freestream_m_s
    );
    assert_eq!(
        result.operating_point.density_kg_m3,
        condition().density_kg_m3
    );
    assert!(result.operating_point.rpm >= options().rpm_min);
    assert!(result.operating_point.rpm <= options().rpm_max);
    assert!(result.torque_residual_nm.abs() <= options().torque_tolerance_nm);
    assert!(result.bem_evaluation_count >= 2);
    assert!(result.bem_evaluation_count <= options().max_bem_evaluations);
    assert!(result.bracket_rpm[0] <= result.bracket_rpm[1]);
    assert!(result.is_finite());
}

#[test]
fn public_motor_matching_api_rejects_missing_brackets_and_work_overruns() {
    let model = model();
    let options = options();
    let no_bracket = MotorTorqueSpeedCurve::try_new(vec![
        MotorTorqueSpeedPoint {
            rpm: options.rpm_min,
            available_torque_nm: 1.0e6,
        },
        MotorTorqueSpeedPoint {
            rpm: options.rpm_max,
            available_torque_nm: 1.0e6,
        },
    ])
    .expect("valid no-bracket motor");
    assert!(matches!(
        evaluate_motor_propeller_match(&model, &no_bracket, condition(), options),
        Err(MotorMatchingError::NoTorqueBracket { .. })
    ));

    let motor = crossing_motor(&model);
    assert!(matches!(
        evaluate_motor_propeller_match(
            &model,
            &motor,
            condition(),
            MotorPropellerMatchOptions {
                max_bem_evaluations: 2,
                ..options
            }
        ),
        Err(MotorMatchingError::EvaluationBudgetExceeded { .. })
    ));
}

#[test]
fn public_motor_matching_api_propagates_bem_failure() {
    let model = model();
    let motor = crossing_motor(&model);
    let result = evaluate_motor_propeller_match(
        &model,
        &motor,
        condition(),
        MotorPropellerMatchOptions {
            bem_options: BladeElementBemOptions {
                max_iterations: 1,
                ..options().bem_options
            },
            ..options()
        },
    );
    assert!(matches!(
        result,
        Err(MotorMatchingError::BemEvaluationFailed { .. })
    ));
}
