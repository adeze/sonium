use sonium_sim_propellers::{
    MeanFlowFwhError, MeanFlowFwhOptions, MovingFwhElementHistory, MovingFwhOptions,
    MovingFwhSample, Point3, evaluate_mean_flow_fwh_reference, evaluate_moving_fwh_reference,
};

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn history(position_end: Point3, velocity_end_m_s: f64) -> MovingFwhElementHistory {
    MovingFwhElementHistory {
        samples: vec![
            MovingFwhSample {
                time_s: 0.0,
                position: point(0.0, 0.0, 0.0),
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 0.0,
                normal_velocity_m_s: 0.0,
            },
            MovingFwhSample {
                time_s: 2.0,
                position: position_end,
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 1.0,
                normal_velocity_m_s: velocity_end_m_s,
            },
        ],
    }
}

fn options(observer_time_s: f64, mean_flow_m_s: Point3) -> MeanFlowFwhOptions {
    MeanFlowFwhOptions {
        observer_time_s,
        observer_position: point(10.0, 0.0, 0.0),
        mean_flow_m_s,
        minimum_observer_distance_m: 0.1,
        ..MeanFlowFwhOptions::default()
    }
}

#[test]
fn zero_mean_flow_reduces_to_quiescent_moving_reference() {
    let source_history = history(point(4.0, 0.0, 0.0), 1.0);
    let observer_time_s = 1.0 + 8.0 / 343.0;
    let mean_flow = evaluate_mean_flow_fwh_reference(
        std::slice::from_ref(&source_history),
        options(observer_time_s, point(0.0, 0.0, 0.0)),
    )
    .expect("zero-flow mean-flow reference");
    let quiescent = evaluate_moving_fwh_reference(
        std::slice::from_ref(&source_history),
        MovingFwhOptions {
            observer_time_s,
            observer_position: point(10.0, 0.0, 0.0),
            minimum_observer_distance_m: 0.1,
            ..MovingFwhOptions::default()
        },
    )
    .expect("quiescent moving reference");

    assert!(
        (mean_flow.elements[0].retarded_time_s - quiescent.elements[0].retarded_time_s).abs()
            < 1.0e-10
    );
    assert!((mean_flow.total_pressure_pa - quiescent.total_pressure_pa).abs() < 1.0e-12);
    assert_eq!(mean_flow.mean_flow_mach, 0.0);
    assert!(mean_flow.is_finite());
}

#[test]
fn downstream_mean_flow_shortens_stationary_source_arrival() {
    let sound_speed_m_s = 343.0;
    let mean_flow_speed_m_s = 34.3;
    let propagation_time_s = 10.0 / (sound_speed_m_s + mean_flow_speed_m_s);
    let result = evaluate_mean_flow_fwh_reference(
        &[history(point(0.0, 0.0, 0.0), 0.0)],
        options(
            1.0 + propagation_time_s,
            point(mean_flow_speed_m_s, 0.0, 0.0),
        ),
    )
    .expect("downstream uniform-flow arrival");

    let element = result.elements[0];
    assert!((element.retarded_time_s - 1.0).abs() < 1.0e-10);
    assert!((element.propagation_time_s - propagation_time_s).abs() < 1.0e-12);
    assert!((element.convected_distance_m - sound_speed_m_s * propagation_time_s).abs() < 1.0e-12);
    assert!((element.propagation_direction.x - 1.0).abs() < 1.0e-12);
    assert!((element.mean_flow_projection_m_s - mean_flow_speed_m_s).abs() < 1.0e-12);
    assert!(result.is_finite());
}

#[test]
fn relative_source_motion_controls_mean_flow_doppler_factor() {
    let mean_flow_speed_m_s = 5.0;
    let source_speed_m_s = 10.0;
    let sound_speed_m_s = 343.0;
    let source_time_s = 1.0;
    let propagation_time_s = 30.0 / (sound_speed_m_s + mean_flow_speed_m_s);
    let result = evaluate_mean_flow_fwh_reference(
        &[history(point(20.0, 0.0, 0.0), 1.0)],
        MeanFlowFwhOptions {
            observer_time_s: source_time_s + propagation_time_s,
            observer_position: point(40.0, 0.0, 0.0),
            mean_flow_m_s: point(mean_flow_speed_m_s, 0.0, 0.0),
            minimum_observer_distance_m: 0.1,
            ..MeanFlowFwhOptions::default()
        },
    )
    .expect("relative-source mean-flow Doppler reference");

    let expected_doppler = 1.0 - (source_speed_m_s - mean_flow_speed_m_s) / sound_speed_m_s;
    assert!((result.elements[0].retarded_time_s - source_time_s).abs() < 1.0e-10);
    assert!((result.elements[0].doppler_factor - expected_doppler).abs() < 1.0e-12);
    assert!(
        (result.elements[0].relative_source_mach
            - (source_speed_m_s - mean_flow_speed_m_s) / sound_speed_m_s)
            .abs()
            < 1.0e-12
    );
}

#[test]
fn transverse_mean_flow_deflects_the_convected_ray() {
    let speed_m_s: f64 = 34.3;
    let sound_speed_m_s: f64 = 343.0;
    let observer_time_s =
        1.0 + 10.0 / (sound_speed_m_s * (1.0 - (speed_m_s / sound_speed_m_s).powi(2)).sqrt());
    let result = evaluate_mean_flow_fwh_reference(
        &[history(point(0.0, 0.0, 0.0), 0.0)],
        MeanFlowFwhOptions {
            observer_time_s,
            observer_position: point(0.0, 0.0, 10.0),
            mean_flow_m_s: point(speed_m_s, 0.0, 0.0),
            minimum_observer_distance_m: 0.1,
            ..MeanFlowFwhOptions::default()
        },
    )
    .expect("transverse mean-flow ray");

    let direction = result.elements[0].propagation_direction;
    assert!(direction.x < 0.0);
    assert!(direction.z > 0.0);
    let direction_norm =
        (direction.x * direction.x + direction.y * direction.y + direction.z * direction.z).sqrt();
    assert!((direction_norm - 1.0).abs() < 1.0e-12);
}

#[test]
fn malformed_mean_flow_and_history_fail_closed() {
    let source_history = history(point(0.0, 0.0, 0.0), 0.0);
    let mut sonic = options(1.0 + 10.0 / 343.0, point(343.0, 0.0, 0.0));
    sonic.maximum_mean_flow_mach = 0.99;
    assert!(matches!(
        evaluate_mean_flow_fwh_reference(&[source_history.clone()], sonic),
        Err(MeanFlowFwhError::MeanFlowMachLimit { .. })
    ));

    let mut invalid = source_history;
    invalid.samples[1].normal = point(0.0, 0.0, 0.0);
    assert!(matches!(
        evaluate_mean_flow_fwh_reference(
            &[invalid],
            options(1.0 + 10.0 / 343.0, point(0.0, 0.0, 0.0)),
        ),
        Err(MeanFlowFwhError::InvalidSample {
            field: "normal",
            ..
        })
    ));

    let mut budget = options(1.0 + 10.0 / 343.0, point(0.0, 0.0, 0.0));
    budget.maximum_samples_per_element = 1;
    assert!(matches!(
        evaluate_mean_flow_fwh_reference(&[history(point(0.0, 0.0, 0.0), 0.0)], budget),
        Err(MeanFlowFwhError::InvalidOptions {
            field: "maximum_samples_per_element",
            ..
        })
    ));
}

#[test]
fn observer_window_and_doppler_limits_are_rejected() {
    let source_history = history(point(0.0, 0.0, 0.0), 0.0);
    assert!(matches!(
        evaluate_mean_flow_fwh_reference(
            &[source_history.clone()],
            options(0.0, point(0.0, 0.0, 0.0)),
        ),
        Err(MeanFlowFwhError::RetardedTimeOutsideHistory { .. })
    ));

    let mut doppler = options(1.0 + 10.0 / 343.0, point(0.0, 0.0, 0.0));
    doppler.minimum_doppler_factor = 1.01;
    assert!(matches!(
        evaluate_mean_flow_fwh_reference(&[source_history], doppler),
        Err(MeanFlowFwhError::InvalidOptions {
            field: "minimum_doppler_factor",
            ..
        })
    ));
}
