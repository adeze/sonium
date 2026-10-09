//! Replay independent speaker measurements against supplied predictions.

// Rust guideline compliant 2026-02-21
use sonium_sim_speaker::measurement_reference::{
    SpeakerMeasurementSet, SpeakerPredictionSet, compare_speaker_measurements,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: compare_measurements measured.json predicted.json report.json".into());
    }
    let measured: SpeakerMeasurementSet = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let predicted: SpeakerPredictionSet = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let report = compare_speaker_measurements(&measured, &predicted)?;
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    if !report.complete_speaker_passed {
        return Err("speaker validation failed or is incomplete; see the report".into());
    }
    Ok(())
}
