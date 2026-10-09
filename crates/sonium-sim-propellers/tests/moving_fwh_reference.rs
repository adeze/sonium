use std::f64::consts::PI;

use sonium_sim_propellers::{
    MovingFwhElementHistory, MovingFwhError, MovingFwhOptions, MovingFwhSample, Point3,
    evaluate_moving_fwh_reference,
};

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn sample(
    time_s: f64,
    position: Point3,
    area_m2: f64,
    pressure_pa: f64,
    normal_velocity_m_s: f64,
) -> MovingFwhSample {
    MovingFwhSample {
        time_s,
        position,
        normal: point(0.0, 0.0, 1.0),
        area_m2,
        pressure_pa,
        normal_velocity_m_s,
    }
}

fn stationary_history(
    area_m2: f64,
    pressure_end_pa: f64,
    velocity_end_m_s: f64,
) -> MovingFwhElementHistory {
    MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), area_m2, 0.0, 0.0),
            sample(
                1.0,
                point(0.0, 0.0, 0.0),
                area_m2,
                pressure_end_pa,
                velocity_end_m_s,
            ),
        ],
    }
}

fn options(observer_time_s: f64, observer_position: Point3) -> MovingFwhOptions {
    MovingFwhOptions {
        observer_time_s,
        observer_position,
        minimum_observer_distance_m: 0.1,
        ..MovingFwhOptions::default()
    }
}

#[test]
fn stationary_history_matches_monopole_and_dipole_source_identities() {
    let sound_speed_m_s = 343.0;
    let distance_m = 10.0;
    let source_time_s = 0.5;
    let observer_time_s = source_time_s + distance_m / sound_speed_m_s;
    let result = evaluate_moving_fwh_reference(
        &[stationary_history(0.2, 1.0, 1.0)],
        options(observer_time_s, point(0.0, 0.0, distance_m)),
    )
    .expect("stationary moving FW-H reference");

    let expected_thickness = 1.2 * 0.2 / (4.0 * PI * distance_m);
    let expected_loading = 0.2 / (4.0 * PI * sound_speed_m_s * distance_m);
    assert!((result.elements[0].retarded_time_s - source_time_s).abs() < 1.0e-9);
    assert!((result.thickness_pressure_pa - expected_thickness).abs() < 1.0e-12);
    assert!((result.loading_pressure_pa - expected_loading).abs() < 1.0e-12);
    assert!((result.total_pressure_pa - expected_thickness - expected_loading).abs() < 1.0e-12);
    assert!(result.is_finite());
}

#[test]
fn moving_source_solves_retarded_time_and_applies_doppler_factor() {
    let sound_speed_m_s = 343.0;
    let source_speed_m_s = 10.0;
    let source_time_s = 1.0;
    let distance_m = 30.0;
    let observer_time_s = source_time_s + distance_m / sound_speed_m_s;
    let history = MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), 0.2, 0.0, 0.0),
            sample(2.0, point(20.0, 0.0, 0.0), 0.2, 0.0, 1.0),
        ],
    };
    let result =
        evaluate_moving_fwh_reference(&[history], options(observer_time_s, point(40.0, 0.0, 0.0)))
            .expect("moving-source retarded-time reference");

    let doppler_factor = 1.0 - source_speed_m_s / sound_speed_m_s;
    let q = 1.2 * 0.2 * 0.5;
    let q_dot = 1.2 * 0.2 / 2.0;
    let expected = (q_dot / (distance_m * doppler_factor)
        + q * source_speed_m_s / (distance_m * distance_m * doppler_factor))
        / (4.0 * PI);
    assert!((result.elements[0].retarded_time_s - source_time_s).abs() < 1.0e-9);
    assert!((result.elements[0].doppler_factor - doppler_factor).abs() < 1.0e-12);
    assert!((result.thickness_pressure_pa - expected).abs() < 1.0e-12);
    assert!(result.loading_pressure_pa.abs() < 1.0e-14);
}

#[test]
fn multiple_moving_elements_sum_and_retain_per_element_diagnostics() {
    let distance_m = 10.0;
    let observer_time_s = 0.5 + distance_m / 343.0;
    let result = evaluate_moving_fwh_reference(
        &[
            stationary_history(0.2, 0.0, 1.0),
            stationary_history(0.3, 0.0, 2.0),
        ],
        options(observer_time_s, point(0.0, 0.0, distance_m)),
    )
    .expect("multiple moving FW-H elements");

    assert_eq!(result.elements.len(), 2);
    assert!((result.minimum_source_observer_distance_m - distance_m).abs() < 1.0e-12);
    let expected = 1.2 * (0.2 + 0.6) / (4.0 * PI * distance_m);
    assert!((result.thickness_pressure_pa - expected).abs() < 1.0e-12);
    assert!(
        (result.total_pressure_pa - result.thickness_pressure_pa - result.loading_pressure_pa)
            .abs()
            < 1.0e-14
    );
}

#[test]
fn loading_orientation_changes_the_signed_dipole_contribution() {
    let observer_time_s = 0.5 + 10.0 / 343.0;
    let mut reversed = stationary_history(0.2, 1.0, 0.0);
    for item in &mut reversed.samples {
        item.normal.z = -1.0;
    }
    let positive = evaluate_moving_fwh_reference(
        &[stationary_history(0.2, 1.0, 0.0)],
        options(observer_time_s, point(0.0, 0.0, 10.0)),
    )
    .expect("positive loading");
    let negative =
        evaluate_moving_fwh_reference(&[reversed], options(observer_time_s, point(0.0, 0.0, 10.0)))
            .expect("negative loading");
    assert!(positive.loading_pressure_pa > 0.0);
    assert!(negative.loading_pressure_pa < 0.0);
    assert!((positive.loading_pressure_pa + negative.loading_pressure_pa).abs() < 1.0e-14);
}

#[test]
fn retarded_time_replay_is_deterministic() {
    let history = MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), 0.2, 0.0, 0.0),
            sample(0.5, point(0.2, 0.0, 0.0), 0.21, 0.5, 0.3),
            sample(1.0, point(0.4, 0.0, 0.0), 0.22, 1.0, 0.6),
        ],
    };
    let configuration = options(0.75 + 9.8 / 343.0, point(0.0, 0.0, 10.0));
    let first = evaluate_moving_fwh_reference(std::slice::from_ref(&history), configuration)
        .expect("first replay");
    let second = evaluate_moving_fwh_reference(&[history], configuration).expect("second replay");
    assert_eq!(first, second);
}

#[test]
fn malformed_history_and_budget_paths_fail_closed() {
    assert!(matches!(
        evaluate_moving_fwh_reference(&[], MovingFwhOptions::default()),
        Err(MovingFwhError::EmptySurface)
    ));

    let mut non_increasing = stationary_history(0.2, 1.0, 1.0);
    non_increasing.samples[1].time_s = 0.0;
    assert!(matches!(
        evaluate_moving_fwh_reference(&[non_increasing], options(0.5, point(0.0, 0.0, 10.0))),
        Err(MovingFwhError::NonIncreasingTime { .. })
    ));

    let too_fast = MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), 0.2, 0.0, 0.0),
            sample(1.0, point(200.0, 0.0, 0.0), 0.2, 0.0, 0.0),
        ],
    };
    assert!(matches!(
        evaluate_moving_fwh_reference(&[too_fast], options(1.0, point(0.0, 0.0, 10.0))),
        Err(MovingFwhError::SourceMachLimit { .. })
    ));

    let mut budget = stationary_history(0.2, 1.0, 1.0);
    budget
        .samples
        .push(sample(2.0, point(0.0, 0.0, 0.0), 0.2, 2.0, 2.0));
    let mut budget_options = options(0.5 + 10.0 / 343.0, point(0.0, 0.0, 10.0));
    budget_options.maximum_samples_per_element = 2;
    assert!(matches!(
        evaluate_moving_fwh_reference(&[budget], budget_options),
        Err(MovingFwhError::TooManySamples { .. })
    ));
}

#[test]
fn invalid_observer_window_distance_and_doppler_are_rejected() {
    let history = stationary_history(0.2, 1.0, 1.0);
    let outside = evaluate_moving_fwh_reference(
        std::slice::from_ref(&history),
        options(0.0, point(0.0, 0.0, 10.0)),
    );
    assert!(matches!(
        outside,
        Err(MovingFwhError::RetardedTimeOutsideHistory { .. })
    ));

    let mut close_options = options(0.5 + 0.01 / 343.0, point(0.0, 0.0, 0.01));
    close_options.minimum_observer_distance_m = 0.1;
    assert!(matches!(
        evaluate_moving_fwh_reference(&[history], close_options),
        Err(MovingFwhError::ObserverTooClose { .. })
    ));

    let moving = MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), 0.2, 0.0, 0.0),
            sample(1.0, point(100.0, 0.0, 0.0), 0.2, 0.0, 0.0),
        ],
    };
    let mut doppler_options = options(0.5 + 50.0 / 343.0, point(100.0, 0.0, 0.0));
    doppler_options.maximum_source_mach = 0.5;
    doppler_options.minimum_doppler_factor = 0.99;
    assert!(matches!(
        evaluate_moving_fwh_reference(&[moving], doppler_options),
        Err(MovingFwhError::DopplerLimit { .. })
    ));
}

#[test]
fn invalid_samples_and_options_are_rejected() {
    let mut invalid = stationary_history(0.2, 1.0, 1.0);
    invalid.samples[1].area_m2 = f64::NAN;
    assert!(matches!(
        evaluate_moving_fwh_reference(&[invalid], options(0.5, point(0.0, 0.0, 10.0))),
        Err(MovingFwhError::InvalidSample {
            field: "area_m2",
            ..
        })
    ));

    let invalid_options = MovingFwhOptions {
        sound_speed_m_s: 0.0,
        ..MovingFwhOptions::default()
    };
    assert!(matches!(
        evaluate_moving_fwh_reference(&[stationary_history(0.2, 1.0, 1.0)], invalid_options),
        Err(MovingFwhError::InvalidOptions {
            field: "sound_speed_m_s",
            ..
        })
    ));
}
