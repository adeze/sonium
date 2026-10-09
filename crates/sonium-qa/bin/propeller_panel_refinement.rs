//! Compile-only QA wrapper for caller-supplied propeller response refinement.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    PanelRefinementObservation, PanelRefinementOptions, evaluate_panel_refinement,
};

const CASE_ID: &str = "propeller.panel-response-refinement-reference.v1";

fn run() -> Result<(usize, f64), String> {
    let observations = [
        PanelRefinementObservation {
            refinement: 1,
            panel_count: 96,
            thrust_n: 10.0,
            torque_nm: 0.25,
            acoustic_pressure_pa: 0.8,
        },
        PanelRefinementObservation {
            refinement: 2,
            panel_count: 384,
            thrust_n: 10.01,
            torque_nm: 0.2501,
            acoustic_pressure_pa: 0.8005,
        },
        PanelRefinementObservation {
            refinement: 4,
            panel_count: 1536,
            thrust_n: 10.0101,
            torque_nm: 0.250101,
            acoustic_pressure_pa: 0.800501,
        },
    ];
    let result = evaluate_panel_refinement(&observations, PanelRefinementOptions::default())
        .map_err(|error| error.to_string())?;
    if !result.is_converged() {
        return Err("panel-response refinement fixture did not converge".into());
    }

    let mut malformed = observations;
    malformed[1].panel_count = malformed[0].panel_count;
    if evaluate_panel_refinement(&malformed, PanelRefinementOptions::default()).is_ok() {
        return Err("panel-response refinement accepted non-increasing panel count".into());
    }
    Ok((result.points.len(), result.final_relative_change))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((level_count, final_relative_change)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("refinement_level_count", level_count as f64)
                .metric("final_relative_response_change", final_relative_change)
                .input_checksum("analytic/deterministic-fixture:propeller-panel-refinement-v1")
                .reference(
                    "caller-supplied thrust, torque, and acoustic response refinement ledger",
                )
                .seed("propeller-panel-refinement-v1:1,2,4")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
