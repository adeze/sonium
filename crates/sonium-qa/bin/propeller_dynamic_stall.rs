//! Bounded dynamic-stall state reference QA.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    DynamicStallInput, DynamicStallOptions, evaluate_dynamic_stall_reference,
};

const CASE_ID: &str = "propeller.dynamic-stall-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let history = [
        DynamicStallInput {
            time_s: 0.0,
            angle_of_attack_deg: 0.0,
        },
        DynamicStallInput {
            time_s: 0.01,
            angle_of_attack_deg: 20.0,
        },
        DynamicStallInput {
            time_s: 0.02,
            angle_of_attack_deg: 20.0,
        },
        DynamicStallInput {
            time_s: 0.08,
            angle_of_attack_deg: 0.0,
        },
    ];
    let result = evaluate_dynamic_stall_reference(&history, DynamicStallOptions::default())
        .map_err(|error| error.to_string())?;
    if !result.is_finite() {
        return Err("dynamic-stall result is not finite".into());
    }
    let onset = result.samples[1];
    let peak_stall = result.samples[2];
    let recovery = result.samples[3];
    if onset.separation_state >= onset.separation_target {
        return Err("separation state did not lag onset target".into());
    }
    if recovery.separation_state >= peak_stall.separation_state {
        return Err("separation state did not recover".into());
    }
    Ok((
        result.maximum_abs_dynamic_lift_coefficient,
        onset.separation_state,
        recovery.separation_state,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((maximum_abs_lift, onset_state, recovery_state)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("maximum_abs_dynamic_lift_coefficient", maximum_abs_lift)
                .metric("onset_separation_state", onset_state)
                .metric("recovery_separation_state", recovery_state)
                .minimum("maximum_abs_dynamic_lift_coefficient", 0.0)
                .input_checksum("analytic/deterministic-fixture:propeller-dynamic-stall-v1")
                .reference("bounded first-order delayed-separation state identities")
                .reference_checksum("analytic/invariant:propeller-dynamic-stall-v1")
                .dimension("four-sample prescribed angle-of-attack history; f64 CPU")
                .formulation("phenomenological attached-lift and delayed-separation coupling")
                .tolerance("finite states; onset state below target; recovery decreases state")
                .seed("propeller-dynamic-stall-v1")
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
