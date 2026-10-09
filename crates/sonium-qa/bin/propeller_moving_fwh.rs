//! Compile-only QA wrapper for the retarded-time moving-surface FW-H reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MovingFwhElementHistory, MovingFwhOptions, MovingFwhSample, Point3,
    evaluate_moving_fwh_reference,
};

const CASE_ID: &str = "propeller.moving-fwh-reference.v1";

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn run() -> Result<(f64, f64, f64), String> {
    let sound_speed_m_s = 343.0;
    let source_speed_m_s = 10.0;
    let history = MovingFwhElementHistory {
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
                position: point(20.0, 0.0, 0.0),
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 0.0,
                normal_velocity_m_s: 1.0,
            },
        ],
    };
    let options = MovingFwhOptions {
        observer_time_s: 1.0 + 30.0 / sound_speed_m_s,
        observer_position: point(40.0, 0.0, 0.0),
        ..MovingFwhOptions::default()
    };
    let result =
        evaluate_moving_fwh_reference(&[history], options).map_err(|error| error.to_string())?;
    if !result.is_finite() || result.elements.len() != 1 {
        return Err("moving FW-H result is not finite or complete".into());
    }
    let doppler_factor = 1.0 - source_speed_m_s / sound_speed_m_s;
    let q = 1.2 * 0.2 * 0.5;
    let q_dot = 1.2 * 0.2 / 2.0;
    let expected = (q_dot / (30.0 * doppler_factor)
        + q * source_speed_m_s / (30.0 * 30.0 * doppler_factor))
        / (4.0 * std::f64::consts::PI);
    let element = result.elements[0];
    if (element.retarded_time_s - 1.0).abs() > 1.0e-9
        || (element.doppler_factor - doppler_factor).abs() > 1.0e-12
        || (result.total_pressure_pa - expected).abs() > 1.0e-12
    {
        return Err("moving FW-H retarded-time identity did not close".into());
    }
    Ok((
        element.retarded_time_s,
        element.doppler_factor,
        result.total_pressure_pa,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((retarded_time_s, doppler_factor, pressure_pa)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("retarded_time_s", retarded_time_s)
                .metric("doppler_factor", doppler_factor)
                .metric("total_pressure_pa", pressure_pa)
                .input_checksum("analytic/deterministic-fixture:propeller-moving-fwh-v1")
                .reference("retarded-time reduced far-field thickness/loading source identities")
                .reference_checksum("analytic/invariant:propeller-moving-fwh-v1")
                .dimension("one moving surface element; two time samples; f64 CPU")
                .formulation("subsonic retarded-time moving-surface FW-H reference")
                .tolerance("retarded-time <= 1e-9 s; Doppler/pressure identities close")
                .seed("propeller-moving-fwh-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
