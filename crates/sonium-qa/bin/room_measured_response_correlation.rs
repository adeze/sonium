//! Release-backed complex measured-response comparison contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::measured_validation::{
    MeasuredResponseSet, compare_solved_output_to_measured,
};
use sonium_sim_room::{FrequencyProvenance, SimulationOutput};
use std::f64::consts::PI;

const CASE_ID: &str = "roomsim.measured-response-correlation.v1";

fn fixture() -> (SimulationOutput, MeasuredResponseSet) {
    let frequencies = vec![125.0, 250.0, 500.0];
    let output_pressures = vec![vec![
        vec![
            Complex64::from_polar(0.80, 0.20),
            Complex64::from_polar(1.10, 0.70),
            Complex64::from_polar(1.35, -1.20),
        ],
        vec![
            Complex64::from_polar(0.55, -0.30),
            Complex64::from_polar(0.90, 1.40),
            Complex64::from_polar(1.05, 2.20),
        ],
    ]];
    let measured_pressures = vec![vec![
        vec![
            Complex64::new(0.7840532622729933, 0.1589354646360490),
            Complex64::new(0.8413264060129374, 0.7086394559614602),
            Complex64::new(0.4891829685435094, -1.2582527660557556),
        ],
        vec![
            Complex64::new(0.5254350690190833, -0.16253611366373677),
            Complex64::new(0.15297042861021692, 0.8869047569896141),
            Complex64::new(-0.6179261731181132, 0.8489212240105697),
        ],
    ]];
    let output = SimulationOutput::with_uniform_provenance(
        frequencies.clone(),
        output_pressures,
        vec!["left".into()],
        FrequencyProvenance::Solved,
    );
    let measured = MeasuredResponseSet::new(
        frequencies,
        vec!["left".into()],
        measured_pressures,
        "analytic-measured-fixture/complex-pressure-v1",
    )
    .expect("independent measured fixture should validate");
    (output, measured)
}

fn run() -> Result<QaResult, String> {
    let (output, measured) = fixture();
    let exact = compare_solved_output_to_measured(&output, &measured, 0.0)?;
    exact.require_within(1.0e-12, 1.0e-12)?;

    let branch_cut_model = SimulationOutput::solved(
        vec![100.0],
        vec![vec![vec![Complex64::from_polar(1.0, -PI + 1.0e-6)]]],
        vec!["source".into()],
    );
    let branch_cut_measured = MeasuredResponseSet::new(
        vec![100.0],
        vec!["source".into()],
        vec![vec![vec![Complex64::from_polar(1.0, PI - 1.0e-6)]]],
        "analytic-branch-cut-fixture",
    )?;
    let branch_cut =
        compare_solved_output_to_measured(&branch_cut_model, &branch_cut_measured, 0.0)?;
    branch_cut.require_within(1.0e-12, 3.0e-6)?;

    let synthetic = SimulationOutput::with_uniform_provenance(
        output.frequencies.clone(),
        output.pressures.clone(),
        output.source_names.clone(),
        FrequencyProvenance::Synthetic,
    );
    let synthetic_rejected = compare_solved_output_to_measured(&synthetic, &measured, 0.0).is_err();

    let mut wrong_frequency = measured.clone();
    wrong_frequency.frequencies[1] += 0.5;
    let frequency_rejected =
        compare_solved_output_to_measured(&output, &wrong_frequency, 0.01).is_err();

    let mut wrong_source = measured.clone();
    wrong_source.source_names[0] = "right".into();
    let source_rejected = compare_solved_output_to_measured(&output, &wrong_source, 0.0).is_err();

    let metrics = [
        exact.sample_count == 6,
        exact.phase_sample_count == 6,
        exact.rms_magnitude_error_db.abs() <= 1.0e-12,
        exact.rms_phase_error_rad.abs() <= 1.0e-12,
        branch_cut.max_abs_phase_error_rad <= 3.0e-6,
        synthetic_rejected,
        frequency_rejected,
        source_rejected,
    ];
    if metrics.iter().any(|passed| !passed) {
        return Err("measured-response comparison contract failed".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("sample_count", exact.sample_count as f64)
        .metric("phase_sample_count", exact.phase_sample_count as f64)
        .metric("rms_magnitude_error_db", exact.rms_magnitude_error_db)
        .metric("rms_phase_error_rad", exact.rms_phase_error_rad)
        .metric(
            "branch_cut_phase_error_rad",
            branch_cut.max_abs_phase_error_rad,
        )
        .metric("synthetic_rejected", synthetic_rejected as u8 as f64)
        .metric("frequency_rejected", frequency_rejected as u8 as f64)
        .metric("source_rejected", source_rejected as u8 as f64)
        .input_checksum("analytic/deterministic-fixture:roomsim-measured-response-v1")
        .reference_checksum("complex-pressure-magnitude-phase-comparison:v1")
        .dimension("one source, two listeners, three frequencies")
        .formulation(
            "solved RoomSim response against independently constructed complex measured tensor",
        )
        .tolerance("eight contract metrics valid; exact fixture errors <= 1e-12")
        .seed("roomsim-measured-response-correlation:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim measured-response result must serialize")
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
                    .expect("RoomSim measured-response failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
