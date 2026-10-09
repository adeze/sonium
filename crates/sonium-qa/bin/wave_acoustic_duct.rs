//! Reference-only QA for a uniform, lossless, below-cutoff acoustic duct.
//!
//! This checks the analytical transfer, impedance, and power identities only.
//! It deliberately does not claim validation of a body-fitted FEM/BEM duct
//! solve, mode matching at discontinuities, or radiation/loading models.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_wave::analytical::{DuctState, LosslessDuct1d};

const CASE_ID: &str = "wave.acoustic-duct-1d.v1";
const TOLERANCE: f64 = 5.0e-13;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("zero_length_state_error", metrics.zero_length_state_error)
                .metric(
                    "matched_load_impedance_relative_error",
                    metrics.matched_load_impedance_relative_error,
                )
                .metric("power_conservation_error", metrics.power_conservation_error)
                .maximum("zero_length_state_error", TOLERANCE)
                .maximum("matched_load_impedance_relative_error", TOLERANCE)
                .maximum("power_conservation_error", TOLERANCE)
                .metric(
                    "xem_harmonic_adapter_error",
                    metrics.xem_harmonic_adapter_error,
                )
                .maximum("xem_harmonic_adapter_error", TOLERANCE)
                .seed("acoustic-duct-1d:uniform-lossless:below-cutoff:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("acoustic duct QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("acoustic duct QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    zero_length_state_error: f64,
    matched_load_impedance_relative_error: f64,
    power_conservation_error: f64,
    xem_harmonic_adapter_error: f64,
}

fn run() -> Result<Metrics, String> {
    let density_kg_per_m3 = 1.2;
    let sound_speed_m_per_s = 343.0;
    let area_m2 = 0.01;
    let frequency_hz = 250.0;
    let inlet = DuctState {
        pressure_pa: Complex64::new(2.0, -0.7),
        volume_velocity_m3_per_s: Complex64::new(1.1e-4, -0.3e-4),
    };
    let zero_length = LosslessDuct1d::try_new(
        density_kg_per_m3,
        sound_speed_m_per_s,
        area_m2,
        0.0,
        frequency_hz,
    )
    .map_err(|error| error.to_string())?;
    let zero_outlet = zero_length
        .try_transfer(inlet)
        .map_err(|error| error.to_string())?;
    let zero_length_state_error = (zero_outlet.pressure_pa - inlet.pressure_pa)
        .norm()
        .max((zero_outlet.volume_velocity_m3_per_s - inlet.volume_velocity_m3_per_s).norm());

    let duct = LosslessDuct1d::try_new(
        density_kg_per_m3,
        sound_speed_m_per_s,
        area_m2,
        0.37,
        frequency_hz,
    )
    .map_err(|error| error.to_string())?;
    let characteristic_impedance = duct.volume_velocity_impedance_pa_s_per_m3();
    let matched_load_impedance_relative_error = (duct
        .try_input_impedance(Complex64::new(characteristic_impedance, 0.0))
        .map_err(|error| error.to_string())?
        - Complex64::new(characteristic_impedance, 0.0))
    .norm()
        / characteristic_impedance;

    let outlet = duct
        .try_transfer(inlet)
        .map_err(|error| error.to_string())?;
    let power_conservation_error = (LosslessDuct1d::time_average_power_w(inlet)
        - LosslessDuct1d::time_average_power_w(outlet))
    .abs();
    let xem_medium =
        sonium_xem::AcousticMedium::try_new(frequency_hz, sound_speed_m_per_s, density_kg_per_m3)
            .map_err(|error| error.to_string())?;
    let xem_duct = LosslessDuct1d::try_from_xem(xem_medium, area_m2, 0.37)
        .map_err(|error| error.to_string())?;
    let real_inlet = DuctState {
        pressure_pa: Complex64::new(2.0, 0.0),
        volume_velocity_m3_per_s: Complex64::new(1.1e-4, 0.0),
    };
    let legacy_real = duct
        .try_transfer(real_inlet)
        .map_err(|error| error.to_string())?;
    let xem_outlet = xem_duct
        .try_transfer(real_inlet)
        .map_err(|error| error.to_string())?;
    let xem_harmonic_adapter_error = (legacy_real.pressure_pa - xem_outlet.pressure_pa.conj())
        .norm()
        .max(
            (legacy_real.volume_velocity_m3_per_s - xem_outlet.volume_velocity_m3_per_s.conj())
                .norm(),
        );
    let metrics = Metrics {
        zero_length_state_error,
        matched_load_impedance_relative_error,
        power_conservation_error,
        xem_harmonic_adapter_error,
    };
    for (name, value) in [
        ("zero-length transfer", metrics.zero_length_state_error),
        (
            "matched-load input impedance",
            metrics.matched_load_impedance_relative_error,
        ),
        (
            "lossless power conservation",
            metrics.power_conservation_error,
        ),
        ("XEM harmonic adapter", metrics.xem_harmonic_adapter_error),
    ] {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name} failed: {value:.3e} > {TOLERANCE:.3e}"));
        }
    }
    Ok(metrics)
}
