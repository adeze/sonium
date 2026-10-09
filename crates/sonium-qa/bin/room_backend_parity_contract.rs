//! Release-backed CPU/accelerator complex-response parity contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::backend_parity::compare_backend_outputs;
use sonium_sim_room::{FrequencyProvenance, SimulationOutput};

const CASE_ID: &str = "roomsim.backend-parity-contract.v1";

fn response(values: &[f64]) -> SimulationOutput {
    SimulationOutput::with_uniform_provenance(
        vec![100.0, 200.0, 400.0],
        vec![vec![
            values
                .iter()
                .copied()
                .map(|value| Complex64::new(value, value * 0.1))
                .collect(),
        ]],
        vec!["source".into()],
        FrequencyProvenance::Solved,
    )
}

fn run() -> Result<QaResult, String> {
    let reference = response(&[1.0, 2.0, 4.0]);
    let candidate = response(&[1.0 + 1.0e-13, 2.0, 4.0]);
    let report =
        compare_backend_outputs("cpu", &reference, "accelerator", &candidate, 0.0, 1.0e-12)?;
    report.require_within(1.0e-10, 1.0e-10)?;
    Ok(QaResult::passed(CASE_ID)
        .input_checksum("independent/deterministic-fixture:roomsim-backend-parity-v1")
        .dimension("one source, one listener, three solved frequencies")
        .reference("complex-pressure CPU/accelerator transfer parity")
        .tolerance("RMS and maximum relative errors <= 1e-10")
        .seed("roomsim-backend-parity-contract-v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim backend-parity result must serialize")
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
                    .expect("RoomSim backend-parity failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
