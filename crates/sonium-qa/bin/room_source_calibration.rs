//! Release-backed RoomSim source calibration and free-field scaling gate.

use sonium_qa::QaResult;
use sonium_sim_room::source_calibration::{
    SOURCE_VOLUME_VELOCITY_REFERENCE, calibrated_spl_at_distance,
    free_field_peak_pressure_for_amplitude, source_amplitude_for_spl,
};
use sonium_xem::constants::AIR_DENSITY_20C;

const CASE_ID: &str = "roomsim.source-calibration.v1";
const SOUND_SPEED: f64 = 343.0;

fn run() -> Result<QaResult, String> {
    let frequencies = [200.0, 400.0, 800.0, 1_000.0, 2_000.0];
    let distances = [0.5, 1.0, 2.0];
    let mut maximum_reference_error: f64 = 0.0;
    let mut samples = 0_u64;

    for &frequency in &frequencies {
        for &distance in &distances {
            let actual = free_field_peak_pressure_for_amplitude(
                frequency,
                SOUND_SPEED,
                AIR_DENSITY_20C,
                1.0,
                distance,
            )?;
            // Independent recomputation of the physical reference.  Keeping
            // this expression here prevents the QA gate from merely testing
            // that the production helper calls itself consistently.
            let expected = 2.0
                * std::f64::consts::PI
                * frequency
                * AIR_DENSITY_20C
                * SOURCE_VOLUME_VELOCITY_REFERENCE
                / (4.0 * std::f64::consts::PI * distance);
            let relative_error = (actual - expected).abs() / expected.abs().max(1.0e-30);
            if !relative_error.is_finite() {
                return Err(format!(
                    "non-finite calibration error at {frequency} Hz/{distance} m"
                ));
            }
            maximum_reference_error = maximum_reference_error.max(relative_error);
            samples += 1;
        }
    }

    let amplitude = source_amplitude_for_spl(85.0, 1_000.0, SOUND_SPEED, AIR_DENSITY_20C, 1.0)?;
    let round_trip_spl =
        calibrated_spl_at_distance(1_000.0, SOUND_SPEED, AIR_DENSITY_20C, amplitude, 1.0)?;
    let round_trip_error = (round_trip_spl - 85.0).abs();
    if maximum_reference_error > 1.0e-12 || round_trip_error > 1.0e-10 {
        return Err(format!(
            "source calibration failed: reference error {maximum_reference_error:.3e}, round-trip SPL error {round_trip_error:.3e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("reference_relative_error", maximum_reference_error)
        .metric("round_trip_spl_error_db", round_trip_error)
        .metric("sample_count", samples as f64)
        .minimum("sample_count", 15.0)
        .maximum("reference_relative_error", 1.0e-12)
        .maximum("round_trip_spl_error_db", 1.0e-10)
        .input_checksum("analytic/deterministic-fixture:roomsim-source-calibration-v1")
        .reference("independent monopole |p| = omega*rho*|Q|/(4*pi*r)")
        .reference_checksum("analytic/deterministic-fixture:roomsim-source-calibration-v1")
        .dimension("20 C air; 200/400/800/1000/2000 Hz; 0.5/1/2 m")
        .formulation("validated RoomSim point-source volume-velocity calibration")
        .tolerance("15 finite scaling samples; relative equation error <= 1e-12; SPL round-trip <= 1e-10 dB")
        .seed("roomsim-source-calibration:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim source-calibration result must serialize")
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
                    .expect("RoomSim source-calibration failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
