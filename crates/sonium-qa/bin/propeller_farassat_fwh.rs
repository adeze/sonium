//! Compile-only QA wrapper for the bounded Farassat 1A surface reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    FarassatFwhOptions, MovingFwhElementHistory, MovingFwhSample, Point3,
    evaluate_farassat_1a_reference,
};

const CASE_ID: &str = "propeller.farassat-fwh-reference.v1";

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn run() -> Result<(f64, f64, f64), String> {
    let density_kg_m3 = 1.2;
    let sound_speed_m_s = 343.0;
    let distance_m = 10.0;
    let observer_time_s = 0.5 + distance_m / sound_speed_m_s;
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
                time_s: 1.0,
                position: point(0.0, 0.0, 0.0),
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 3.0,
                normal_velocity_m_s: 1.0,
            },
        ],
    };
    let result = evaluate_farassat_1a_reference(
        &[history],
        FarassatFwhOptions {
            observer_time_s,
            observer_position: point(0.0, 0.0, distance_m),
            density_kg_m3,
            sound_speed_m_s,
            ..FarassatFwhOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() {
        return Err("Farassat 1A result is not finite".into());
    }
    let expected_thickness =
        density_kg_m3 * 0.2 / (4.0 * std::f64::consts::PI * sound_speed_m_s * distance_m);
    let expected_loading = (3.0 * 0.2 / distance_m + 1.5 * 0.2 / (distance_m * distance_m))
        / (4.0 * std::f64::consts::PI * sound_speed_m_s);
    let identity_error = (result.total_pressure_pa - expected_thickness - expected_loading).abs();
    Ok((
        result.total_pressure_pa,
        result.elements[0].doppler_factor,
        identity_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((total_pressure_pa, doppler_factor, identity_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("total_pressure_pa", total_pressure_pa)
                .metric("doppler_factor", doppler_factor)
                .metric("farassat_1a_identity_error_pa", identity_error)
                .maximum("farassat_1a_identity_error_pa", 1.0e-12)
                .input_checksum("analytic/deterministic-fixture:propeller-farassat-fwh-v1")
                .reference("Farassat 1A stationary thickness/loading identity")
                .reference_checksum("analytic/invariant:propeller-farassat-fwh-v1")
                .dimension("one stationary surface element; two source-time samples; f64 CPU")
                .formulation("bounded quiescent-medium Farassat 1A surface integral")
                .tolerance("stationary thickness/loading identity error <= 1e-12 Pa")
                .seed("propeller-farassat-fwh-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
