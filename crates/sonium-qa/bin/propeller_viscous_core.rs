//! Bounded Lamb--Oseen viscous-core evolution reference QA.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{ViscousCoreOptions, evaluate_viscous_core_evolution};

const CASE_ID: &str = "propeller.viscous-core-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let options = ViscousCoreOptions {
        initial_core_radius_m: 0.01,
        kinematic_viscosity_m2_s: 0.5,
        time_step_s: 0.01,
        steps: 4,
        maximum_steps: 16,
        probe_radius_m: 0.02,
        maximum_core_radius_m: 1.0,
    };
    let circulation_m2_s = -2.0;
    let result = evaluate_viscous_core_evolution(circulation_m2_s, options)
        .map_err(|error| error.to_string())?;
    if !result.is_finite() || result.circulation_m2_s != circulation_m2_s {
        return Err("viscous-core result is not finite or lost circulation sign".into());
    }
    let expected_radius = (options.initial_core_radius_m.powi(2)
        + 4.0 * options.kinematic_viscosity_m2_s * options.time_step_s * options.steps as f64)
        .sqrt();
    let radius_error = (result.final_core_radius_m - expected_radius).abs();
    if radius_error > 1.0e-12 {
        return Err(format!("Lamb--Oseen radius error {radius_error:e}"));
    }
    Ok((
        result.final_core_radius_m,
        result.final_tangential_velocity_m_s,
        radius_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((final_radius, final_velocity, radius_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("final_core_radius_m", final_radius)
                .metric("final_tangential_velocity_m_s", final_velocity)
                .metric("diffusion_radius_identity_error_m", radius_error)
                .maximum("diffusion_radius_identity_error_m", 1.0e-12)
                .input_checksum("analytic/deterministic-fixture:propeller-viscous-core-v1")
                .reference(
                    "Lamb--Oseen Gaussian-core diffusion and finite-radius velocity identity",
                )
                .reference_checksum("analytic/invariant:propeller-viscous-core-v1")
                .dimension("four-step prescribed signed vortex-core evolution; f64 CPU")
                .formulation("bounded viscous-core coupling reference")
                .tolerance("diffusion radius identity error <= 1e-12 m; circulation sign preserved")
                .seed("propeller-viscous-core-v1")
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
