//! Release-backed three-level RoomSim mesh-convergence contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_room::mesh_convergence::compare_mesh_convergence;
use sonium_sim_room::{FrequencyProvenance, SimulationOutput};

fn response(values: &[f64]) -> SimulationOutput {
    SimulationOutput::with_uniform_provenance(
        vec![100.0, 200.0, 400.0],
        vec![vec![
            values
                .iter()
                .map(|value| Complex64::new(*value, *value * 0.1))
                .collect(),
        ]],
        vec!["source".into()],
        FrequencyProvenance::Solved,
    )
}

fn run() -> Result<QaResult, String> {
    let report = compare_mesh_convergence(
        &response(&[10.0, 20.0, 30.0]),
        &response(&[9.0, 18.0, 27.0]),
        &response(&[8.5, 17.0, 25.5]),
        0.0,
        1.0e-12,
    )?;
    report.require_converged(0.07, 0.07)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric(
            "coarse_to_medium_rms_relative_error",
            report.coarse_to_medium_rms_relative_error,
        )
        .metric(
            "medium_to_fine_rms_relative_error",
            report.medium_to_fine_rms_relative_error,
        )
        .metric(
            "medium_to_fine_max_relative_error",
            report.medium_to_fine_max_relative_error,
        )
        .metric("sample_count", report.sample_count as f64)
        .maximum("medium_to_fine_rms_relative_error", 0.07)
        .maximum("medium_to_fine_max_relative_error", 0.07)
        .input_checksum("analytic/deterministic-fixture:roomsim-mesh-convergence-v1")
        .reference(
            "three solved response tensors with identical source/listener/frequency topology",
        )
        .reference_checksum("analytic/deterministic-fixture:roomsim-mesh-convergence-v1")
        .dimension(
            "one source, one listener, three frequencies; coarse/medium/fine response tensors",
        )
        .formulation("checked relative complex-pressure comparison with monotonic refinement gate")
        .tolerance("fine-grid RMS and maximum relative errors <= 0.07; both decrease monotonically")
        .seed("roomsim-mesh-convergence-v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim mesh-convergence result must serialize")
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
                    .expect("RoomSim mesh-convergence failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "roomsim.mesh-convergence-contract.v1";
