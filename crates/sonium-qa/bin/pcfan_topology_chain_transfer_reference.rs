//! Compile-only topology-changing P2/P1 BDF state-transfer contract.
//!
//! This target checks the checked remesh handoff API on deterministic meshes;
//! it does not execute a CFD solve or reuse topology-dependent operators.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::box_mesh_tetrahedra;
use sonium_fem::physics::fluid::stokes::time_stepping::BdfState;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::QaResult;
use sonium_sim_pcfan::transfer_pc_fan_cfd_p2p1_bdf_state_chain;

const CASE_ID: &str = "pcfan.topology-chain-transfer-reference.v1";

fn run() -> Result<(usize, bool), String> {
    let first = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
    let second = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);
    let third = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 2, 2);
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_count = velocity_space.num_global_dofs(&first);
    let pressure_count = pressure_space.num_global_dofs(&first);
    let source_state = BdfState {
        velocity: (0..velocity_count)
            .map(|index| index as f64 * 1.0e-3)
            .collect(),
        velocity_prev: (0..velocity_count)
            .map(|index| 0.5 + index as f64 * 5.0e-4)
            .collect(),
        pressure: (0..pressure_count)
            .map(|index| 100.0 + index as f64)
            .collect(),
        time: 0.125,
        step: 11,
    };
    let report =
        transfer_pc_fan_cfd_p2p1_bdf_state_chain(&[&first, &second, &third], &source_state)
            .map_err(|error| error.to_string())?;
    if report.states.len() != 2
        || report.report.stage_count != 2
        || !report.report.requires_operator_reassembly
        || report
            .states
            .iter()
            .flat_map(|state| {
                state
                    .velocity
                    .iter()
                    .chain(&state.velocity_prev)
                    .chain(&state.pressure)
            })
            .any(|value| !value.is_finite())
    {
        return Err("topology-changing BDF state chain contract failed".into());
    }
    Ok((
        report.report.stage_count,
        report.report.requires_operator_reassembly,
    ))
}

fn main() {
    match run() {
        Ok((stage_count, requires_operator_reassembly)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("stage_count", stage_count as f64)
                .metric(
                    "requires_operator_reassembly",
                    f64::from(requires_operator_reassembly),
                )
                .reference_checksum("analytic:topology-changing-p2p1-bdf-state-transfer:v1")
                .input_checksum("synthetic:three-mesh-chain-p2p1-state:v1")
                .to_json_line()
                .expect("QA result must serialize");
            println!("QA_RESULT: {result}");
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .to_json_line()
                .expect("QA failure result must serialize");
            println!("QA_RESULT: {result}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
