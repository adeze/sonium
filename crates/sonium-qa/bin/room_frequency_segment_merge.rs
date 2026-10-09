//! Independent QA contract for the dense-BLR/FEM frequency splice boundary.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::{FrequencyProvenance, SimulationOutput};

const CASE_ID: &str = "roomsim.frequency-segment-merge.v1";

fn segment(
    frequencies: Vec<f64>,
    source_name: &str,
    provenance: FrequencyProvenance,
    offset: f64,
) -> SimulationOutput {
    let values = frequencies
        .iter()
        .enumerate()
        .map(|(index, _)| Complex64::new(offset + index as f64, -offset))
        .collect::<Vec<_>>();
    SimulationOutput::with_uniform_provenance(
        frequencies,
        vec![vec![values]],
        vec![source_name.to_owned()],
        provenance,
    )
}

fn multi_listener_segment(frequencies: Vec<f64>, listener_count: usize) -> SimulationOutput {
    let values = frequencies
        .iter()
        .enumerate()
        .map(|(index, _)| Complex64::new(index as f64 + 1.0, 0.0))
        .collect::<Vec<_>>();
    SimulationOutput::solved(
        frequencies,
        vec![vec![values; listener_count]],
        vec!["source".into()],
    )
}

fn run() -> Result<QaResult, String> {
    let merged = SimulationOutput::merge_frequency_segments(
        segment(vec![40.0, 80.0], "source", FrequencyProvenance::Solved, 1.0),
        segment(vec![160.0], "source", FrequencyProvenance::Synthetic, 3.0),
    )?;
    merged.validate_layout()?;

    let ordered_frequency = (merged.frequencies == [40.0, 80.0, 160.0]) as u8 as f64;
    let provenance_alignment = (merged.provenance
        == [
            FrequencyProvenance::Solved,
            FrequencyProvenance::Solved,
            FrequencyProvenance::Synthetic,
        ]) as u8 as f64;
    let pressure_axis = (merged.pressures[0][0].len() == merged.frequencies.len()) as u8 as f64;

    let overlap_rejected = SimulationOutput::merge_frequency_segments(
        segment(vec![100.0], "source", FrequencyProvenance::Solved, 1.0),
        segment(vec![100.0], "source", FrequencyProvenance::Solved, 2.0),
    )
    .is_err();
    let malformed_rejected = SimulationOutput::merge_frequency_segments(
        SimulationOutput {
            frequencies: vec![100.0],
            pressures: vec![Vec::new()],
            source_names: vec!["source".into()],
            provenance: vec![FrequencyProvenance::Solved],
        },
        segment(vec![200.0], "source", FrequencyProvenance::Solved, 1.0),
    )
    .is_err();
    let listener_mismatch_rejected = SimulationOutput::merge_frequency_segments(
        segment(vec![100.0], "source", FrequencyProvenance::Solved, 1.0),
        multi_listener_segment(vec![200.0], 2),
    )
    .is_err();

    let overlap = SimulationOutput::merge_frequency_segments_with_overlap(
        segment(
            vec![100.0, 200.0, 300.0],
            "source",
            FrequencyProvenance::Solved,
            1.0,
        ),
        segment(
            vec![200.0, 300.0, 400.0],
            "source",
            FrequencyProvenance::Solved,
            9.0,
        ),
    )?;
    let complex_crossfade = overlap.frequencies == [100.0, 200.0, 300.0, 400.0]
        && overlap.pressures[0][0]
            == vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(10.0, -9.0),
                Complex64::new(11.0, -9.0),
            ];
    let single_shared = SimulationOutput::merge_frequency_segments_with_overlap(
        segment(vec![200.0], "source", FrequencyProvenance::Solved, 2.0),
        segment(vec![200.0], "source", FrequencyProvenance::Solved, 8.0),
    )?;
    let single_point_average = single_shared.pressures[0][0] == vec![Complex64::new(5.0, -5.0)]
        && single_shared.provenance == vec![FrequencyProvenance::Solved];
    let overlap_provenance_rejected = SimulationOutput::merge_frequency_segments_with_overlap(
        segment(
            vec![100.0, 200.0],
            "source",
            FrequencyProvenance::Solved,
            1.0,
        ),
        segment(
            vec![200.0, 300.0],
            "source",
            FrequencyProvenance::Synthetic,
            2.0,
        ),
    )
    .is_err();
    let no_shared_overlap_rejected = SimulationOutput::merge_frequency_segments_with_overlap(
        segment(vec![100.0], "source", FrequencyProvenance::Solved, 1.0),
        segment(vec![200.0], "source", FrequencyProvenance::Solved, 2.0),
    )
    .is_err();

    let metrics = [
        ordered_frequency,
        provenance_alignment,
        pressure_axis,
        overlap_rejected as u8 as f64,
        malformed_rejected as u8 as f64,
        listener_mismatch_rejected as u8 as f64,
        complex_crossfade as u8 as f64,
        single_point_average as u8 as f64,
        overlap_provenance_rejected as u8 as f64,
        no_shared_overlap_rejected as u8 as f64,
    ];
    if metrics.iter().any(|metric| *metric != 1.0) {
        return Err("frequency-segment merge contract failed".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("ordered_frequency", ordered_frequency)
        .metric("provenance_alignment", provenance_alignment)
        .metric("pressure_axis", pressure_axis)
        .metric("overlap_rejected", overlap_rejected as u8 as f64)
        .metric("malformed_rejected", malformed_rejected as u8 as f64)
        .metric(
            "listener_mismatch_rejected",
            listener_mismatch_rejected as u8 as f64,
        )
        .metric("complex_crossfade", complex_crossfade as u8 as f64)
        .metric("single_point_average", single_point_average as u8 as f64)
        .metric(
            "overlap_provenance_rejected",
            overlap_provenance_rejected as u8 as f64,
        )
        .metric(
            "no_shared_overlap_rejected",
            no_shared_overlap_rejected as u8 as f64,
        )
        .input_checksum("analytic/deterministic-fixture:roomsim-frequency-segment-merge-v1")
        .reference_checksum("ordered-concatenation-and-overlap-validation:v2")
        .dimension("one source, one listener, dense [40,80] Hz + FEM [160] Hz")
        .formulation("checked full-band response splice with provenance preservation")
        .tolerance("ten contract metrics exactly 1.0")
        .seed("roomsim-frequency-segment-merge:cpu:v1"))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
