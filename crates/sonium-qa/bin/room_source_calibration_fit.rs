//! Release-backed multi-frequency measured-source calibration contract.

use sonium_qa::QaResult;
use sonium_sim_room::source_calibration::{
    CalibrationObservation, SOURCE_VOLUME_VELOCITY_REFERENCE, fit_source_amplitude_from_spl,
};
use sonium_xem::constants::{AIR_DENSITY_20C, REFERENCE_PRESSURE};

const CASE_ID: &str = "roomsim.source-calibration-fit.v1";
const SOUND_SPEED: f64 = 343.0;

fn independent_spl(frequency_hz: f64, distance_m: f64, source_amplitude: f64) -> f64 {
    let pressure = 2.0
        * std::f64::consts::PI
        * frequency_hz
        * AIR_DENSITY_20C
        * SOURCE_VOLUME_VELOCITY_REFERENCE
        * source_amplitude.abs()
        / (4.0 * std::f64::consts::PI * distance_m);
    20.0 * (pressure / std::f64::consts::SQRT_2 / REFERENCE_PRESSURE).log10()
}

fn run() -> Result<QaResult, String> {
    let known_amplitude = 1.75;
    let observations = [(250.0, 0.75), (500.0, 1.0), (1_000.0, 1.5), (2_000.0, 2.0)]
        .into_iter()
        .map(|(frequency_hz, distance_m)| CalibrationObservation {
            frequency_hz,
            distance_m,
            measured_spl_db: independent_spl(frequency_hz, distance_m, known_amplitude),
        })
        .collect::<Vec<_>>();

    let fit = fit_source_amplitude_from_spl(&observations, SOUND_SPEED, AIR_DENSITY_20C)?;
    if (fit.source_amplitude - known_amplitude).abs() > 1.0e-12 {
        return Err(format!(
            "calibrated amplitude mismatch: expected {known_amplitude:.16e}, got {:.16e}",
            fit.source_amplitude
        ));
    }
    if fit.rms_error_db > 1.0e-12 || fit.max_abs_error_db > 1.0e-12 {
        return Err(format!(
            "calibration residual too large: rms={:.3e}, max={:.3e}",
            fit.rms_error_db, fit.max_abs_error_db
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("observation_count", fit.observation_count as f64)
        .metric("fitted_source_amplitude", fit.source_amplitude)
        .metric("rms_error_db", fit.rms_error_db)
        .metric("max_abs_error_db", fit.max_abs_error_db)
        .minimum("observation_count", 4.0)
        .maximum("rms_error_db", 1.0e-12)
        .maximum("max_abs_error_db", 1.0e-12)
        .input_checksum("analytic/deterministic-fixture:roomsim-source-calibration-fit-v1")
        .reference("independent monopole SPL sweep with positive least-squares amplitude fit")
        .reference_checksum("analytic/deterministic-fixture:roomsim-source-calibration-fit-v1")
        .dimension("20 C air; 250/500/1000/2000 Hz; 0.75/1/1.5/2 m; four observations")
        .formulation("multi-frequency measured-response source-amplitude calibration")
        .tolerance("amplitude and residual error <= 1e-12")
        .seed("roomsim-source-calibration-fit:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim calibration-fit result must serialize")
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
                    .expect("RoomSim calibration-fit failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
