//! Production propeller-surrogate load, shaft-power, and acoustic-power QA.

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    Blade3dConfig, Propeller3dOptConfig, baseline_blade_shape, try_optimize_blade_shape,
};
use std::time::Instant;

const CASE_ID: &str = "propeller.load-power.v1";
const CLOSURE_TOLERANCE: f64 = 1.0e-12;
const MAX_THRUST_LOSS: f64 = 0.05;

struct Metrics {
    baseline_shaft_power_w: f64,
    candidate_shaft_power_w: f64,
    candidate_acoustic_power_w: f64,
    candidate_acoustic_to_shaft_ratio: f64,
    candidate_ideal_hover_efficiency: f64,
    candidate_acoustic_power_closure_error: f64,
    thrust_loss: f64,
}

fn run() -> Result<Metrics, String> {
    let config = Propeller3dOptConfig {
        mesh: Blade3dConfig {
            radial_segments: 12,
            chord_segments: 4,
            loop_segments: 6,
            ..Blade3dConfig::default()
        },
        n_harmonics: 6,
        max_iterations: 1,
        ..Propeller3dOptConfig::default()
    };
    let baseline = baseline_blade_shape(&config);
    let candidate = try_optimize_blade_shape(&config).map_err(|error| error.to_string())?;
    if !baseline.is_feasible() || !candidate.is_feasible() {
        return Err(
            "production propeller surrogate did not produce a feasible load/power state".into(),
        );
    }

    for (label, result) in [("baseline", &baseline), ("candidate", &candidate)] {
        let loads = result.loads;
        if ![
            loads.thrust_n,
            loads.torque_nm,
            loads.angular_speed_rad_s,
            loads.shaft_power_w,
            loads.disk_area_m2,
            loads.ideal_hover_power_w,
            loads.ideal_hover_efficiency,
            loads.acoustic_power_w,
            loads.acoustic_to_shaft_power_ratio,
            loads.acoustic_power_closure_error,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(format!(
                "{label} load/power ledger contains a non-finite value"
            ));
        }
        if loads.shaft_power_w <= 0.0
            || loads.acoustic_power_w < 0.0
            || !(0.0..=1.0).contains(&loads.ideal_hover_efficiency)
            || !(0.0..=1.0).contains(&loads.acoustic_to_shaft_power_ratio)
            || loads.acoustic_power_closure_error > CLOSURE_TOLERANCE
        {
            return Err(format!(
                "{label} load/power ledger violates its physical bounds: {loads:?}"
            ));
        }
    }

    Ok(Metrics {
        baseline_shaft_power_w: baseline.loads.shaft_power_w,
        candidate_shaft_power_w: candidate.loads.shaft_power_w,
        candidate_acoustic_power_w: candidate.loads.acoustic_power_w,
        candidate_acoustic_to_shaft_ratio: candidate.loads.acoustic_to_shaft_power_ratio,
        candidate_ideal_hover_efficiency: candidate.loads.ideal_hover_efficiency,
        candidate_acoustic_power_closure_error: candidate.loads.acoustic_power_closure_error,
        thrust_loss: candidate.thrust_loss,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("baseline_shaft_power_w", metrics.baseline_shaft_power_w)
                .metric("candidate_shaft_power_w", metrics.candidate_shaft_power_w)
                .metric(
                    "candidate_acoustic_power_w",
                    metrics.candidate_acoustic_power_w,
                )
                .metric(
                    "candidate_acoustic_to_shaft_ratio",
                    metrics.candidate_acoustic_to_shaft_ratio,
                )
                .metric(
                    "candidate_ideal_hover_efficiency",
                    metrics.candidate_ideal_hover_efficiency,
                )
                .metric(
                    "candidate_ideal_hover_efficiency_lower_margin",
                    metrics.candidate_ideal_hover_efficiency,
                )
                .metric(
                    "candidate_acoustic_power_closure_error",
                    metrics.candidate_acoustic_power_closure_error,
                )
                .metric(
                    "candidate_acoustic_to_shaft_ratio_lower_margin",
                    metrics.candidate_acoustic_to_shaft_ratio,
                )
                .metric("thrust_loss", metrics.thrust_loss)
                .minimum("baseline_shaft_power_w", f64::MIN_POSITIVE)
                .minimum("candidate_shaft_power_w", f64::MIN_POSITIVE)
                .maximum("candidate_ideal_hover_efficiency", 1.0)
                .minimum("candidate_ideal_hover_efficiency_lower_margin", 0.0)
                .maximum("candidate_acoustic_to_shaft_ratio", 1.0)
                .minimum("candidate_acoustic_to_shaft_ratio_lower_margin", 0.0)
                .minimum("candidate_acoustic_power_w", 0.0)
                .maximum("candidate_acoustic_power_closure_error", CLOSURE_TOLERANCE)
                .maximum("thrust_loss", MAX_THRUST_LOSS)
                .seed("propeller-5inch:18000rpm:mesh=12x4+6:harmonics=6:optimizer=1:cpu")
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
            eprintln!("PROPELLER LOAD/POWER QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
