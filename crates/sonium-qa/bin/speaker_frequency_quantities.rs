//! Release-facing XEM frequency-grid consumer contract for speaker modeling.

use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::SpeakerDriverStudy;
use std::time::Instant;

const CASE_ID: &str = "xem.speaker-frequency-quantities.v1";

fn run() -> Result<(usize, f64, f64), String> {
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let typed = study
        .frequency_quantities()
        .map_err(|error| format!("valid speaker frequency grid rejected: {error}"))?;
    if typed.len() != study.frequency_hz.len() {
        return Err("typed frequency grid changed length".into());
    }
    if typed
        .iter()
        .map(|frequency| frequency.get())
        .zip(study.frequency_hz.iter().copied())
        .any(|(typed, raw)| typed != raw)
    {
        return Err("typed frequency grid changed a value".into());
    }

    let mut invalid = study.clone();
    invalid.frequency_hz.swap(0, 1);
    if invalid.frequency_quantities().is_ok() {
        return Err("reordered frequency grid was accepted".into());
    }

    Ok((typed.len(), typed[0].get(), typed[typed.len() - 1].get()))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((count, first_hz, last_hz)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("XEM FrequencyHz quantity validation and speaker-grid contract")
                .reference_checksum("analytic/deterministic-fixture:xem-speaker-frequency-v1")
                .dimension("speaker driver deterministic octave frequency grid")
                .formulation("speaker study validated frequency grid converted to XEM FrequencyHz")
                .tolerance("exact value/count preservation and malformed-order rejection")
                .metric("frequency_count", count as f64)
                .metric("first_frequency_hz", first_hz)
                .metric("last_frequency_hz", last_hz)
                .metric("invalid_order_rejections", 1.0)
                .minimum("invalid_order_rejections", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("SPEAKER XEM FREQUENCY CONTRACT FAILED");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
