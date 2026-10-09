//! Release-backed reciprocal-transfer response contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::reciprocity_validation::compare_reciprocal_transfer_outputs;
use sonium_sim_room::{FrequencyProvenance, SimulationOutput};

fn output(values: Vec<Vec<Vec<Complex64>>>) -> SimulationOutput {
    SimulationOutput::with_uniform_provenance(
        vec![100.0, 200.0],
        values,
        vec!["a".into(), "b".into()],
        FrequencyProvenance::Solved,
    )
}

fn run() -> Result<QaResult, String> {
    let a = Complex64::new(1.0, 0.2);
    let b = Complex64::new(3.0, -0.5);
    let c = Complex64::new(5.0, 0.7);
    let forward = output(vec![
        vec![vec![a, a * 2.0], vec![b, b * 2.0]],
        vec![vec![b, b * 2.0], vec![c, c * 2.0]],
    ]);
    let reverse = output(vec![
        vec![vec![a, a * 2.0], vec![b, b * 2.0]],
        vec![vec![b, b * 2.0], vec![c, c * 2.0]],
    ]);
    let labels = vec!["a".to_string(), "b".to_string()];
    let report =
        compare_reciprocal_transfer_outputs(&forward, &labels, &reverse, &labels, 0.0, 1.0e-12)?;
    report.require_within(1.0e-12, 1.0e-12)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("sample_count", report.sample_count as f64)
        .metric("rms_relative_error", report.rms_relative_error)
        .metric("max_relative_error", report.max_relative_error)
        .minimum("sample_count", 1.0)
        .maximum("rms_relative_error", 1.0e-12)
        .maximum("max_relative_error", 1.0e-12)
        .input_checksum("analytic/deterministic-fixture:roomsim-reciprocity-v1")
        .reference("symmetric transfer-normalized two-source/two-listener complex-pressure matrix")
        .reference_checksum("analytic/deterministic-fixture:roomsim-reciprocity-v1")
        .dimension("two sources, two listeners, two frequencies; reciprocal transfer matrix")
        .formulation(
            "checked solved-response transpose pairing with symmetric relative complex error",
        )
        .tolerance("RMS and maximum reciprocity relative errors <= 1e-12")
        .seed("roomsim-reciprocity-v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim reciprocity result must serialize")
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
                    .expect("RoomSim reciprocity failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "roomsim.reciprocity-contract.v1";
