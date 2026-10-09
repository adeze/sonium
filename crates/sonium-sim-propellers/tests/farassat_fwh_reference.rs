use std::f64::consts::PI;

use sonium_sim_propellers::{
    FarassatFwhElementResult, FarassatFwhError, FarassatFwhOptions, MovingFwhElementHistory,
    MovingFwhSample, Point3, evaluate_farassat_1a_reference,
};

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn sample(
    time_s: f64,
    position: Point3,
    normal: Point3,
    area_m2: f64,
    pressure_pa: f64,
    normal_velocity_m_s: f64,
) -> MovingFwhSample {
    MovingFwhSample {
        time_s,
        position,
        normal,
        area_m2,
        pressure_pa,
        normal_velocity_m_s,
    }
}

fn stationary_history(
    normal: Point3,
    area_m2: f64,
    pressure_rate_pa_s: f64,
    normal_velocity_rate_m_s2: f64,
) -> MovingFwhElementHistory {
    MovingFwhElementHistory {
        samples: vec![
            sample(0.0, point(0.0, 0.0, 0.0), normal, area_m2, 0.0, 0.0),
            sample(
                1.0,
                point(0.0, 0.0, 0.0),
                normal,
                area_m2,
                pressure_rate_pa_s,
                normal_velocity_rate_m_s2,
            ),
        ],
    }
}

fn options(observer_time_s: f64, observer_position: Point3) -> FarassatFwhOptions {
    FarassatFwhOptions {
        observer_time_s,
        observer_position,
        minimum_observer_distance_m: 0.1,
        maximum_source_mach: 0.9,
        minimum_doppler_factor: 0.1,
        retarded_time_tolerance_s: 1.0e-12,
        maximum_retarded_iterations: 128,
        ..FarassatFwhOptions::default()
    }
}

fn expected_stationary_loading(
    pressure_pa: f64,
    pressure_rate_pa_s: f64,
    area_m2: f64,
    normal_projection: f64,
    distance_m: f64,
    sound_speed_m_s: f64,
) -> f64 {
    let load = pressure_pa * area_m2 * normal_projection;
    let load_rate = pressure_rate_pa_s * area_m2 * normal_projection;
    (load_rate / distance_m + load / (distance_m * distance_m)) / (4.0 * PI * sound_speed_m_s)
}

fn assert_element_finite(element: FarassatFwhElementResult) {
    assert!(element.retarded_time_s.is_finite());
    assert!(element.source_observer_distance_m.is_finite());
    assert!(element.source_observer_distance_m > 0.0);
    assert!(element.source_mach.is_finite());
    assert!(element.doppler_factor.is_finite());
    assert!(element.doppler_factor > 0.0);
    assert!(element.thickness_pressure_pa.is_finite());
    assert!(element.loading_pressure_pa.is_finite());
    assert!(element.total_pressure_pa.is_finite());
}

#[test]
fn stationary_thickness_term_matches_farassat_1a_identity() {
    let density = 1.2;
    let sound_speed = 343.0;
    let area = 0.2;
    let velocity_rate = 1.5;
    let distance = 10.0;
    let observer_time = 0.5 + distance / sound_speed;
    let result = evaluate_farassat_1a_reference(
        &[stationary_history(
            point(0.0, 0.0, 1.0),
            area,
            0.0,
            velocity_rate,
        )],
        options(observer_time, point(0.0, 0.0, distance)),
    )
    .expect("stationary thickness reference");

    let expected = density * area * velocity_rate / (4.0 * PI * sound_speed * distance);
    assert!((result.thickness_pressure_pa - expected).abs() < 1.0e-13);
    assert!(result.loading_pressure_pa.abs() < 1.0e-14);
    assert!((result.total_pressure_pa - expected).abs() < 1.0e-13);
    assert!((result.elements[0].retarded_time_s - 0.5).abs() < 1.0e-12);
    assert_element_finite(result.elements[0]);
    assert!(result.is_finite());
}

#[test]
fn stationary_loading_term_matches_farassat_1a_dipole_identity() {
    let sound_speed = 343.0;
    let area = 0.2;
    let pressure_rate = 3.0;
    let pressure_at_retarded_time = 1.5;
    let distance = 10.0;
    let observer_time = 0.5 + distance / sound_speed;
    let history = stationary_history(point(0.0, 0.0, 1.0), area, pressure_rate, 0.0);
    let result = evaluate_farassat_1a_reference(
        &[history],
        options(observer_time, point(0.0, 0.0, distance)),
    )
    .expect("stationary loading reference");

    let expected = expected_stationary_loading(
        pressure_at_retarded_time,
        pressure_rate,
        area,
        1.0,
        distance,
        sound_speed,
    );
    assert!(result.thickness_pressure_pa.abs() < 1.0e-14);
    assert!((result.loading_pressure_pa - expected).abs() < 1.0e-13);
    assert!((result.total_pressure_pa - expected).abs() < 1.0e-13);
}

#[test]
fn signed_loading_follows_surface_normal_directivity() {
    let distance = 10.0;
    let observer_time = 0.5 + distance / 343.0;
    let positive = evaluate_farassat_1a_reference(
        &[stationary_history(point(0.0, 0.0, 1.0), 0.2, 1.0, 0.0)],
        options(observer_time, point(0.0, 0.0, distance)),
    )
    .expect("positive loading");
    let negative = evaluate_farassat_1a_reference(
        &[stationary_history(point(0.0, 0.0, -1.0), 0.2, 1.0, 0.0)],
        options(observer_time, point(0.0, 0.0, distance)),
    )
    .expect("negative loading");
    assert!(positive.loading_pressure_pa > 0.0);
    assert!(negative.loading_pressure_pa < 0.0);
    assert!((positive.loading_pressure_pa + negative.loading_pressure_pa).abs() < 1.0e-14);
}

#[test]
fn stationary_terms_add_over_elements_and_keep_retarded_diagnostics() {
    let distance = 10.0;
    let observer_time = 0.5 + distance / 343.0;
    let first = stationary_history(point(0.0, 0.0, 1.0), 0.2, 2.0, 0.5);
    let second = stationary_history(point(0.0, 0.0, 1.0), 0.3, 4.0, 1.0);
    let result = evaluate_farassat_1a_reference(
        &[first, second],
        options(observer_time, point(0.0, 0.0, distance)),
    )
    .expect("multiple Farassat elements");
    assert_eq!(result.elements.len(), 2);
    for element in result.elements {
        assert_element_finite(element);
    }
    let expected_thickness = 1.2 * (0.2 * 0.5 + 0.3 * 1.0) / (4.0 * PI * 343.0 * distance);
    assert!((result.thickness_pressure_pa - expected_thickness).abs() < 1.0e-13);
    assert!(
        (result.total_pressure_pa - result.thickness_pressure_pa - result.loading_pressure_pa)
            .abs()
            < 1.0e-14
    );
}

#[test]
fn moving_source_matches_explicit_farassat_motion_terms() {
    let speed = 34.3;
    let sound_speed = 343.0;
    let distance_axis = 10.0;
    let area = 0.2;
    let pressure = 2.0;
    let observer_position = point(0.0, 0.0, distance_axis);
    let source_time = 0.5;
    let source_position = point(speed * source_time, 0.0, 0.0);
    let source_distance =
        (source_position.x * source_position.x + distance_axis * distance_axis).sqrt();
    let observer_time = source_time + source_distance / sound_speed;
    let result = evaluate_farassat_1a_reference(
        &[MovingFwhElementHistory {
            samples: vec![
                sample(
                    0.0,
                    point(0.0, 0.0, 0.0),
                    point(0.0, 0.0, 1.0),
                    area,
                    pressure,
                    0.0,
                ),
                sample(
                    1.0,
                    point(speed, 0.0, 0.0),
                    point(0.0, 0.0, 1.0),
                    area,
                    pressure,
                    0.0,
                ),
            ],
        }],
        options(observer_time, observer_position),
    )
    .expect("moving Farassat element");

    let direction = point(
        -source_position.x / source_distance,
        0.0,
        distance_axis / source_distance,
    );
    let mach = speed / sound_speed;
    let m_r = mach * direction.x;
    let m_squared = mach * mach;
    let doppler = 1.0 - m_r;
    let velocity = point(speed, 0.0, 0.0);
    let direction_rate = point(
        (-velocity.x + direction.x * (velocity.x * direction.x)) / source_distance,
        0.0,
        direction.z * (velocity.x * direction.x) / source_distance,
    );
    let mach_vector = point(mach, 0.0, 0.0);
    let m_r_rate = mach_vector.x * direction_rate.x;
    let load = pressure * area;
    let l_r = load * direction.z;
    let l_r_rate = load * direction_rate.z;
    let common = source_distance * m_r_rate + sound_speed * (m_r - m_squared);
    let expected = (l_r_rate / (source_distance * doppler * doppler)
        + (l_r - 0.0) / (source_distance * source_distance * doppler * doppler)
        + l_r * common / (source_distance * source_distance * doppler.powi(3)))
        / (4.0 * PI * sound_speed);
    assert!((result.loading_pressure_pa - expected).abs() < 1.0e-13);
    assert!((result.elements[0].doppler_factor - doppler).abs() < 1.0e-12);
    assert!((result.elements[0].source_observer_distance_m - source_distance).abs() < 1.0e-12);
}

#[test]
fn retarded_time_selects_later_segment_of_piecewise_history() {
    let distance = 10.0;
    let history = MovingFwhElementHistory {
        samples: vec![
            sample(
                0.0,
                point(0.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                0.2,
                0.0,
                0.0,
            ),
            sample(
                1.0,
                point(0.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                0.2,
                1.0,
                0.0,
            ),
            sample(
                2.0,
                point(0.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                0.2,
                3.0,
                0.0,
            ),
        ],
    };
    let result = evaluate_farassat_1a_reference(
        &[history],
        options(1.5 + distance / 343.0, point(0.0, 0.0, distance)),
    )
    .expect("later retarded segment");
    assert!((result.elements[0].retarded_time_s - 1.5).abs() < 1.0e-12);
    assert!(result.loading_pressure_pa > 0.0);
}

#[test]
fn malformed_histories_and_options_fail_closed() {
    let valid = stationary_history(point(0.0, 0.0, 1.0), 0.2, 1.0, 1.0);
    assert!(matches!(
        evaluate_farassat_1a_reference(&[], FarassatFwhOptions::default()),
        Err(FarassatFwhError::EmptySurface)
    ));
    let mut short = valid.clone();
    short.samples.pop();
    assert!(matches!(
        evaluate_farassat_1a_reference(&[short], FarassatFwhOptions::default()),
        Err(FarassatFwhError::InsufficientSamples { .. })
    ));
    let mut non_finite = valid.clone();
    non_finite.samples[1].pressure_pa = f64::NAN;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[non_finite], FarassatFwhOptions::default()),
        Err(FarassatFwhError::InvalidSample {
            field: "pressure_pa",
            ..
        })
    ));
    let mut non_increasing = valid.clone();
    non_increasing.samples[1].time_s = 0.0;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[non_increasing], FarassatFwhOptions::default()),
        Err(FarassatFwhError::NonIncreasingTime { .. })
    ));
    let invalid_options = FarassatFwhOptions {
        sound_speed_m_s: 0.0,
        ..FarassatFwhOptions::default()
    };
    assert!(matches!(
        evaluate_farassat_1a_reference(&[valid], invalid_options),
        Err(FarassatFwhError::InvalidOptions {
            field: "sound_speed_m_s",
            ..
        })
    ));
}

#[test]
fn budgets_and_observer_admission_fail_closed() {
    let history = stationary_history(point(0.0, 0.0, 1.0), 0.2, 1.0, 1.0);
    let mut too_many_samples = FarassatFwhOptions::default();
    too_many_samples.maximum_samples_per_element = 1;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[history.clone()], too_many_samples),
        Err(FarassatFwhError::InvalidOptions { .. })
    ));
    let mut too_many_total = FarassatFwhOptions::default();
    too_many_total.maximum_total_samples = 2;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[history.clone(), history.clone()], too_many_total),
        Err(FarassatFwhError::TooManyTotalSamples { .. })
    ));
    let outside =
        evaluate_farassat_1a_reference(&[history.clone()], options(0.0, point(0.0, 0.0, 10.0)));
    assert!(matches!(
        outside,
        Err(FarassatFwhError::RetardedTimeOutsideHistory { .. })
    ));
    let close = evaluate_farassat_1a_reference(
        &[history.clone()],
        options(0.5 + 0.01 / 343.0, point(0.0, 0.0, 0.01)),
    );
    assert!(matches!(
        close,
        Err(FarassatFwhError::ObserverTooClose { .. })
    ));
    let mut no_convergence = options(0.37 + 10.0 / 343.0, point(0.0, 0.0, 10.0));
    no_convergence.maximum_retarded_iterations = 1;
    no_convergence.retarded_time_tolerance_s = 1.0e-14;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[history], no_convergence),
        Err(FarassatFwhError::RetardedTimeNoConvergence { .. })
    ));
}

#[test]
fn source_mach_and_doppler_limits_fail_closed() {
    let moving = MovingFwhElementHistory {
        samples: vec![
            sample(
                0.0,
                point(0.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                0.2,
                1.0,
                0.0,
            ),
            sample(
                1.0,
                point(0.0, 0.0, 300.0),
                point(0.0, 0.0, 1.0),
                0.2,
                1.0,
                0.0,
            ),
        ],
    };
    let mut mach_options = options(0.5 + 250.0 / 343.0, point(0.0, 0.0, 400.0));
    mach_options.maximum_source_mach = 0.5;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[moving.clone()], mach_options),
        Err(FarassatFwhError::SourceMachLimit { .. })
    ));

    let near_sonic = MovingFwhElementHistory {
        samples: vec![
            sample(
                0.0,
                point(0.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                0.2,
                1.0,
                0.0,
            ),
            sample(
                1.0,
                point(0.0, 0.0, 300.0),
                point(0.0, 0.0, 1.0),
                0.2,
                1.0,
                0.0,
            ),
        ],
    };
    let mut doppler_options = options(0.5 + 250.0 / 343.0, point(0.0, 0.0, 400.0));
    doppler_options.maximum_source_mach = 0.99;
    doppler_options.minimum_doppler_factor = 0.99;
    assert!(matches!(
        evaluate_farassat_1a_reference(&[near_sonic], doppler_options),
        Err(FarassatFwhError::DopplerLimit { .. })
    ));
}
