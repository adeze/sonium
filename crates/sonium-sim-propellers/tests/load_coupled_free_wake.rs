use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementError,
    BladeElementExternalInflow, BladeElementModel, BladeElementOperatingPoint, BladeElementStation,
    LoadCoupledFreeWakeError, LoadCoupledFreeWakeOptions, SteadyFreeWakeOptions,
    evaluate_load_coupled_steady_free_wake,
};

fn model() -> BladeElementModel {
    model_with_lift_scale(1.0)
}

fn weak_coupled_model() -> BladeElementModel {
    model_with_lift_scale(0.1)
}

fn model_with_lift_scale(lift_scale: f64) -> BladeElementModel {
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
    .expect("valid public load-coupled polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid public load-coupled model")
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
        relaxation: 0.05,
        max_iterations: 100,
        convergence_tolerance_m: 1.0e-4,
        max_node_count: 10_000,
        max_biot_savart_evaluations: 1_000_000,
        ..SteadyFreeWakeOptions::default()
    }
}

#[test]
fn external_inflow_zero_is_compatible_and_nonzero_changes_loads() {
    let model = model();
    let point = operating_point();
    let options = BladeElementBemOptions::default();
    let baseline = model.evaluate_bem(point, options).expect("baseline BEM");
    let zero = vec![
        BladeElementExternalInflow {
            axial_velocity_m_s: 0.0,
            tangential_velocity_m_s: 0.0,
        };
        model.stations().len()
    ];
    let explicit_zero = model
        .evaluate_bem_with_external_inflow(point, options, &zero)
        .expect("zero external inflow BEM");
    assert_eq!(baseline, explicit_zero);

    let nonzero = vec![
        BladeElementExternalInflow {
            axial_velocity_m_s: 0.25,
            tangential_velocity_m_s: 0.5,
        };
        model.stations().len()
    ];
    let changed = model
        .evaluate_bem_with_external_inflow(point, options, &nonzero)
        .expect("nonzero external inflow BEM");
    assert!((changed.loads.thrust_n - baseline.loads.thrust_n).abs() > 1.0e-10);
    assert!((changed.loads.torque_nm - baseline.loads.torque_nm).abs() > 1.0e-10);
}

#[test]
fn external_inflow_contract_rejects_mismatch_and_nonfinite_values() {
    let model = model();
    let point = operating_point();
    let options = BladeElementBemOptions::default();
    assert!(matches!(
        model.evaluate_bem_with_external_inflow(point, options, &[]),
        Err(BladeElementError::ExternalInflowCountMismatch { .. })
    ));

    let invalid = vec![
        BladeElementExternalInflow {
            axial_velocity_m_s: f64::NAN,
            tangential_velocity_m_s: 0.0,
        };
        model.stations().len()
    ];
    assert!(matches!(
        model.evaluate_bem_with_external_inflow(point, options, &invalid),
        Err(BladeElementError::InvalidExternalInflow {
            field: "axial_velocity_m_s",
            ..
        })
    ));
}

#[test]
fn load_coupled_free_wake_converges_with_bounded_counts_and_changes_inflow() {
    let model = weak_coupled_model();
    let result = evaluate_load_coupled_steady_free_wake(
        &model,
        operating_point(),
        BladeElementBemOptions::default(),
        LoadCoupledFreeWakeOptions {
            max_outer_iterations: 100,
            inflow_relaxation: 0.05,
            convergence_tolerance_m_s: 1.0e-4,
            wake_options: wake_options(),
        },
    )
    .expect("load-coupled free wake");

    assert!(result.converged);
    assert!(result.is_finite());
    assert!(result.outer_iterations <= 100);
    assert_eq!(result.bem_evaluation_count, result.wake_evaluation_count);
    assert!(result.bem_evaluation_count >= 1);
    assert!(
        result
            .station_external_inflow
            .iter()
            .any(|inflow| inflow.axial_velocity_m_s.abs() > 1.0e-10
                || inflow.tangential_velocity_m_s.abs() > 1.0e-10)
    );
    assert_eq!(result.bem.loads.strips.len(), model.stations().len());
    assert_eq!(result.wake.operating_point, operating_point());
}

#[test]
fn load_coupled_free_wake_rejects_outer_budget_and_wake_budget() {
    let model = model();
    let outer_budget = evaluate_load_coupled_steady_free_wake(
        &model,
        operating_point(),
        BladeElementBemOptions::default(),
        LoadCoupledFreeWakeOptions {
            max_outer_iterations: 1,
            wake_options: wake_options(),
            ..LoadCoupledFreeWakeOptions::default()
        },
    );
    assert!(matches!(
        outer_budget,
        Err(LoadCoupledFreeWakeError::NotConverged { iterations: 1, .. })
    ));

    let wake_budget = evaluate_load_coupled_steady_free_wake(
        &model,
        operating_point(),
        BladeElementBemOptions::default(),
        LoadCoupledFreeWakeOptions {
            wake_options: SteadyFreeWakeOptions {
                max_biot_savart_evaluations: 1,
                ..wake_options()
            },
            ..LoadCoupledFreeWakeOptions::default()
        },
    );
    assert!(matches!(
        wake_budget,
        Err(LoadCoupledFreeWakeError::WakeEvaluation { .. })
    ));
}

#[test]
fn load_coupled_free_wake_rejects_invalid_outer_options() {
    let model = model();
    let result = evaluate_load_coupled_steady_free_wake(
        &model,
        operating_point(),
        BladeElementBemOptions::default(),
        LoadCoupledFreeWakeOptions {
            max_outer_iterations: 0,
            ..LoadCoupledFreeWakeOptions::default()
        },
    );
    assert!(matches!(
        result,
        Err(LoadCoupledFreeWakeError::InvalidOptions {
            field: "max_outer_iterations",
            ..
        })
    ));
}
