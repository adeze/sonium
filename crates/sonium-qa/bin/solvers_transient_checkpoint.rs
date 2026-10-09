//! Independent CPU evidence for portable transient-adjoint Revolve schedules.

use sonium_qa::QaResult;
use sonium_solvers::RevolveSchedule;
use std::time::Instant;

const CASE_ID: &str = "solvers.transient-checkpoint.v1";
const TOLERANCE: f64 = 0.0;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let source = RevolveSchedule::new(8, 2).map_err(|error| error.to_string())?;
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored =
        RevolveSchedule::from_checkpoint_bytes(&bytes).map_err(|error| error.to_string())?;
    if restored != source {
        return Err("portable schedule round-trip changed the action sequence".into());
    }

    let mut corrupted = bytes.clone();
    corrupted[0] ^= 0x01;
    let corruption_rejected = RevolveSchedule::from_checkpoint_bytes(&corrupted).is_err();
    if !corruption_rejected {
        return Err("corrupted schedule magic was accepted".into());
    }

    let mut reversed = Vec::new();
    restored
        .execute(
            0usize,
            |from, to, state| {
                if state != from {
                    return Err(format!(
                        "advance state {state} does not match source {from}"
                    ));
                }
                Ok::<_, String>(to)
            },
            |step, state| {
                if *state != step + 1 {
                    return Err(format!("reverse state {} is not step + 1", *state));
                }
                reversed.push(step);
                Ok::<_, String>(())
            },
        )
        .map_err(|error| error.to_string())?;
    if reversed != (0..8).rev().collect::<Vec<_>>() {
        return Err(format!("unexpected reverse order: {reversed:?}"));
    }

    Ok(vec![
        ("serialized_bytes", bytes.len() as f64),
        (
            "round_trip_error",
            if restored == source { 0.0 } else { 1.0 },
        ),
        (
            "corruption_rejection_error",
            if corruption_rejected { 0.0 } else { 1.0 },
        ),
        ("recomputation_steps", restored.recomputation_steps() as f64),
        ("reverse_steps", restored.reverse_steps() as f64),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent deterministic transient state replay")
                .input_checksum("analytic/deterministic-fixture:transient-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:transient-checkpoint-v1")
                .dimension("8 transitions; two checkpoint slots; CPU state replay")
                .tolerance("round-trip and corruption errors exactly zero")
                .seed("solvers-transient-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name.ends_with("error") {
                    result.maximum(name, TOLERANCE).metric(name, value)
                } else {
                    result.metric(name, value)
                };
            }
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result should serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent deterministic transient state replay")
                .input_checksum("analytic/deterministic-fixture:transient-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:transient-checkpoint-v1")
                .tolerance("round-trip and corruption errors exactly zero")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result should serialize")
            );
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
