//! Compact stationary-surface FW-H-style observer reference QA.

use std::f64::consts::PI;
use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{FwhOptions, FwhSurfaceElement, Point3, evaluate_fwh_reference};

const CASE_ID: &str = "propeller.fwh-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let options = FwhOptions {
        observer_position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 10.0,
        },
        minimum_observer_distance_m: 0.5,
        ..FwhOptions::default()
    };
    let element = FwhSurfaceElement {
        position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        normal: Point3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        area_m2: 0.2,
        pressure_real_pa: 1.0,
        pressure_imaginary_pa: 0.0,
        normal_velocity_real_m_s: 0.1,
        normal_velocity_imaginary_m_s: 0.0,
    };
    let result = evaluate_fwh_reference(&[element], options).map_err(|error| error.to_string())?;
    if !result.is_finite() {
        return Err("FW-H result is not finite".into());
    }
    let omega = 2.0 * PI * options.frequency_hz;
    let expected_monopole =
        omega * options.density_kg_m3 * element.area_m2 * element.normal_velocity_real_m_s
            / (4.0 * PI * 10.0);
    let expected_dipole = omega * element.area_m2 * element.pressure_real_pa
        / (4.0 * PI * options.sound_speed_m_s * 10.0);
    let monopole_error = (result.monopole_pressure_pa.magnitude_pa() - expected_monopole).abs();
    let dipole_error = (result.dipole_pressure_pa.magnitude_pa() - expected_dipole).abs();
    if monopole_error > 1.0e-12 || dipole_error > 1.0e-12 {
        return Err(format!(
            "FW-H source identity errors: monopole={monopole_error:e}, dipole={dipole_error:e}"
        ));
    }
    Ok((
        result.total_pressure_pa.magnitude_pa(),
        monopole_error,
        dipole_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((total_pressure, monopole_error, dipole_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("total_pressure_peak_pa", total_pressure)
                .metric("monopole_identity_error_pa", monopole_error)
                .metric("dipole_identity_error_pa", dipole_error)
                .maximum("monopole_identity_error_pa", 1.0e-12)
                .maximum("dipole_identity_error_pa", 1.0e-12)
                .input_checksum("analytic/deterministic-fixture:propeller-fwh-v1")
                .reference(
                    "compact free-field monopole/dipole source identities with retarded phase",
                )
                .reference_checksum("analytic/invariant:propeller-fwh-v1")
                .dimension("one stationary surface element; 1 kHz observer at 10 m; f64 CPU")
                .formulation("stationary-surface FW-H-style source-observer integration")
                .tolerance("monopole and dipole identity errors <= 1e-12 Pa")
                .seed("propeller-fwh-v1")
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
