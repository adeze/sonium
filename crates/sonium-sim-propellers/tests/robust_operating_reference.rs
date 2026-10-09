use std::f64::consts::FRAC_PI_2;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    PropellerOperatingCondition, PropellerUncertaintyScenario, RobustOperatingError,
    RobustOperatingOptions, evaluate_robust_operating_envelope,
};

fn conditions() -> Vec<PropellerOperatingCondition> {
    vec![
        PropellerOperatingCondition {
            freestream_m_s: 5.0,
            rpm: 12_000.0,
            inflow_angle_rad: 0.0,
            density_kg_m3: 1.204,
            temperature_k: 293.15,
        },
        PropellerOperatingCondition {
            freestream_m_s: 10.0,
            rpm: 12_000.0,
            inflow_angle_rad: 0.1,
            density_kg_m3: 1.204,
            temperature_k: 293.15,
        },
    ]
}

fn options() -> RobustOperatingOptions {
    RobustOperatingOptions {
        scenarios: vec![
            PropellerUncertaintyScenario::nominal(),
            PropellerUncertaintyScenario {
                label: "hot-low-density".to_owned(),
                thrust_scale: 0.95,
                torque_scale: 1.05,
                rpm_scale: 1.01,
                freestream_scale: 1.05,
                density_scale: 0.92,
                inflow_angle_offset_rad: 0.05,
                temperature_offset_k: 20.0,
            },
        ],
        conditions: conditions(),
    }
}

#[test]
fn robust_operating_reference_retains_worst_case_evidence() {
    let source = PropellerSource::drone_5inch(12_000.0);
    let result = evaluate_robust_operating_envelope(&source, &options())
        .expect("robust operating reference");
    assert!(result.is_finite());
    assert!(result.is_feasible());
    assert_eq!(result.scenarios.len(), 2);
    assert!(result.scenarios.iter().all(|scenario| scenario.admissible));
    assert!(result.worst_case_scenario_index < result.scenarios.len());
    assert!(result.worst_power_margin_w.is_some());
    assert!(result.worst_propulsive_efficiency.is_some());
}

#[test]
fn robust_operating_reference_is_deterministic_and_responds_to_scenario_scales() {
    let source = PropellerSource::drone_5inch(12_000.0);
    let first = evaluate_robust_operating_envelope(&source, &options()).expect("first result");
    let second = evaluate_robust_operating_envelope(&source, &options()).expect("second result");
    assert_eq!(first, second);

    let mut changed_options = options();
    changed_options.scenarios[1].freestream_scale = 0.5;
    let changed = evaluate_robust_operating_envelope(&source, &changed_options)
        .expect("changed scenario result");
    assert_ne!(
        first.scenarios[1].operating_points,
        changed.scenarios[1].operating_points
    );
}

#[test]
fn robust_operating_reference_reports_adverse_rejection_without_panicking() {
    let source = PropellerSource::drone_5inch(12_000.0);
    let result = evaluate_robust_operating_envelope(
        &source,
        &RobustOperatingOptions {
            scenarios: vec![PropellerUncertaintyScenario {
                label: "overspeed".to_owned(),
                freestream_scale: 10.0,
                ..PropellerUncertaintyScenario::nominal()
            }],
            conditions: conditions(),
        },
    )
    .expect("adverse scenario result");
    assert!(result.is_finite());
    assert!(!result.is_feasible());
    assert!(!result.all_admissible);
    assert_eq!(result.scenarios[0].rejected_condition_index, Some(0));
    assert!(result.scenarios[0].rejection.is_some());
}

#[test]
fn robust_operating_reference_rejects_invalid_matrix_and_conditions() {
    let source = PropellerSource::drone_5inch(12_000.0);
    assert!(matches!(
        evaluate_robust_operating_envelope(
            &source,
            &RobustOperatingOptions {
                scenarios: Vec::new(),
                conditions: conditions(),
            }
        ),
        Err(RobustOperatingError::InvalidConfiguration {
            field: "scenarios",
            ..
        })
    ));
    assert!(matches!(
        evaluate_robust_operating_envelope(
            &source,
            &RobustOperatingOptions {
                scenarios: vec![PropellerUncertaintyScenario {
                    label: "zero-density-scale".to_owned(),
                    density_scale: 0.0,
                    ..PropellerUncertaintyScenario::nominal()
                }],
                conditions: conditions(),
            }
        ),
        Err(RobustOperatingError::InvalidConfiguration {
            field: "scenario",
            ..
        })
    ));

    let mut invalid_conditions = conditions();
    invalid_conditions[0].inflow_angle_rad = FRAC_PI_2 + 0.01;
    assert!(matches!(
        evaluate_robust_operating_envelope(
            &source,
            &RobustOperatingOptions {
                scenarios: vec![PropellerUncertaintyScenario::nominal()],
                conditions: invalid_conditions,
            }
        ),
        Err(RobustOperatingError::ScenarioEvaluation {
            scenario_index: 0,
            condition_index: Some(0),
            ..
        })
    ));
}
