//! Reference-only QA for a lossless normal-incidence fluid interface.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_wave::analytical::AcousticInterface1d;

const CASE_ID: &str = "wave.acoustic-interface-1d.v1";
const TOLERANCE: f64 = 5.0e-13;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("pressure_continuity", metrics.pressure_continuity)
                .metric("velocity_continuity", metrics.velocity_continuity)
                .metric("power_balance", metrics.power_balance)
                .metric(
                    "identical_medium_reflection",
                    metrics.identical_medium_reflection,
                )
                .maximum("pressure_continuity", TOLERANCE)
                .maximum("velocity_continuity", TOLERANCE)
                .maximum("power_balance", TOLERANCE)
                .maximum("identical_medium_reflection", TOLERANCE)
                .seed("acoustic-interface-1d:air-water:normal-incidence:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("interface QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("interface QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    pressure_continuity: f64,
    velocity_continuity: f64,
    power_balance: f64,
    identical_medium_reflection: f64,
}

fn run() -> Result<Metrics, String> {
    let interface =
        AcousticInterface1d::try_new(1.2, 343.0, 1000.0, 1480.0, 1000.0, Complex64::new(1.0, 0.4))
            .map_err(|error| error.to_string())?;
    let left = interface
        .try_evaluate_left(0.0)
        .map_err(|error| error.to_string())?;
    let right = interface
        .try_evaluate_right(0.0)
        .map_err(|error| error.to_string())?;
    let [incident, reflected, transmitted] = interface.intensities_w_per_m2();
    let identical =
        AcousticInterface1d::try_new(1.2, 343.0, 1.2, 343.0, 1000.0, Complex64::new(1.0, 0.0))
            .map_err(|error| error.to_string())?;
    let metrics = Metrics {
        pressure_continuity: (left.pressure_pa - right.pressure_pa).norm(),
        velocity_continuity: (left.particle_velocity_m_per_s - right.particle_velocity_m_per_s)
            .norm(),
        power_balance: (incident - reflected - transmitted).abs(),
        identical_medium_reflection: identical.pressure_reflection_coefficient().abs(),
    };
    for (name, value) in [
        ("pressure continuity", metrics.pressure_continuity),
        ("velocity continuity", metrics.velocity_continuity),
        ("power balance", metrics.power_balance),
        (
            "identical-medium reflection",
            metrics.identical_medium_reflection,
        ),
    ] {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name} failed: {value:.3e} > {TOLERANCE:.3e}"));
        }
    }
    Ok(metrics)
}
