//! Independent CPU QA for the propeller ideal-actuator-disk operating envelope.

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::QaResult;
use sonium_sim_propellers::PropellerOperatingEnvelope;
use std::time::Instant;

const CASE_ID: &str = "propeller.operating-envelope.v1";
const RELATIVE_TOLERANCE: f64 = 2.0e-12;

struct Metrics {
    points: f64,
    max_freestream_m_s: f64,
    max_relative_error: f64,
    minimum_power_margin_w: f64,
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
}

fn run() -> Result<Metrics, String> {
    let source = PropellerSource::drone_5inch(18_000.0);
    let envelope = PropellerOperatingEnvelope::try_from_source(source.clone())
        .map_err(|error| error.to_string())?;
    let disk_area_m2 =
        std::f64::consts::PI * (source.tip_radius.powi(2) - source.hub_radius.powi(2));
    let shaft_power_w = source.torque * source.omega().abs();
    let expected_max_freestream_m_s = shaft_power_w / source.thrust
        - source.thrust.powi(2) / (2.0 * source.density * disk_area_m2 * shaft_power_w);
    let max_error = relative_error(envelope.max_freestream_m_s(), expected_max_freestream_m_s);
    if max_error > RELATIVE_TOLERANCE {
        return Err(format!(
            "maximum freestream reference mismatch: actual={} expected={} relative_error={}",
            envelope.max_freestream_m_s(),
            expected_max_freestream_m_s,
            max_error
        ));
    }

    let speeds = [
        0.0,
        envelope.max_freestream_m_s() * 0.25,
        envelope.max_freestream_m_s() * 0.75,
        envelope.max_freestream_m_s(),
    ];
    let points = envelope
        .evaluate_many(&speeds)
        .map_err(|error| error.to_string())?;
    let mut max_relative_error = max_error;
    let mut minimum_power_margin_w = f64::INFINITY;

    for (speed, point) in speeds.into_iter().zip(&points) {
        let discriminant = speed.powi(2) + 2.0 * source.thrust / (source.density * disk_area_m2);
        let expected_induced =
            source.thrust / (source.density * disk_area_m2 * (discriminant.sqrt() + speed));
        let expected_disk_velocity = speed + expected_induced;
        let expected_ideal_power = source.thrust * expected_disk_velocity;
        let expected_useful_power = source.thrust * speed;
        let expected_advance_ratio = speed / ((source.rpm / 60.0) * (2.0 * source.tip_radius));
        let expected_ideal_efficiency = expected_useful_power / expected_ideal_power;
        let expected_propulsive_efficiency = expected_useful_power / shaft_power_w;

        for (label, actual, expected) in [
            (
                "induced velocity",
                point.induced_velocity_m_s,
                expected_induced,
            ),
            (
                "disk velocity",
                point.disk_velocity_m_s,
                expected_disk_velocity,
            ),
            (
                "ideal actuator power",
                point.ideal_actuator_power_w,
                expected_ideal_power,
            ),
            ("useful power", point.useful_power_w, expected_useful_power),
            ("advance ratio", point.advance_ratio, expected_advance_ratio),
            (
                "ideal actuator efficiency",
                point.ideal_actuator_efficiency,
                expected_ideal_efficiency,
            ),
            (
                "propulsive efficiency",
                point.propulsive_efficiency,
                expected_propulsive_efficiency,
            ),
        ] {
            let error = relative_error(actual, expected);
            max_relative_error = max_relative_error.max(error);
            if error > RELATIVE_TOLERANCE {
                return Err(format!(
                    "{} reference mismatch at {} m/s: actual={} expected={} relative_error={}",
                    label, speed, actual, expected, error
                ));
            }
        }
        if !(0.0..=1.0).contains(&point.ideal_actuator_efficiency)
            || !(0.0..=1.0).contains(&point.propulsive_efficiency)
        {
            return Err(format!(
                "efficiency bounds failed at {speed} m/s: {point:?}"
            ));
        }
        minimum_power_margin_w = minimum_power_margin_w.min(point.power_margin_w);
    }

    for pair in points.windows(2) {
        if pair[1].ideal_actuator_power_w < pair[0].ideal_actuator_power_w
            || pair[1].useful_power_w < pair[0].useful_power_w
            || pair[1].induced_velocity_m_s > pair[0].induced_velocity_m_s
        {
            return Err("operating-envelope monotonicity contract failed".into());
        }
    }

    Ok(Metrics {
        points: points.len() as f64,
        max_freestream_m_s: envelope.max_freestream_m_s(),
        max_relative_error,
        minimum_power_margin_w,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("points", metrics.points)
                .metric("max_freestream_m_s", metrics.max_freestream_m_s)
                .metric("max_relative_error", metrics.max_relative_error)
                .metric("minimum_power_margin_w", metrics.minimum_power_margin_w)
                .minimum("points", 4.0)
                .minimum("max_freestream_m_s", 0.0)
                .minimum("minimum_power_margin_w", 0.0)
                .maximum("max_relative_error", RELATIVE_TOLERANCE)
                .reference_checksum(
                    "analytic/deterministic-fixture:propeller-operating-envelope-v1",
                )
                .input_checksum("propeller-5inch:18000rpm:fixed-load:cpu")
                .formulation("annular ideal-actuator-disk forward-flight momentum reference")
                .seed("propeller-operating-envelope-v1:0,0.25,0.75,1.0:max-power")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PROPELLER OPERATING-ENVELOPE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
