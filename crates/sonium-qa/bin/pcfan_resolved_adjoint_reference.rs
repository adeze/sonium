//! Compile-only contract for the resolved-flow full-state adjoint engine.
//!
//! The linearization is a deterministic two-state fixture. It exercises sparse
//! transpose solving and gradient contraction without running a CFD solver.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanFlowModel, PcFanResolvedAdjointConfig,
    PcFanResolvedDerivativeProbeConfig, PcFanResolvedFlowLinearization,
    PcFanResolvedFlowPicardLinearization, PcFanResolvedFlowSparseMatrix,
    probe_pc_fan_resolved_flow_derivatives, solve_pc_fan_resolved_flow_adjoint,
};

const CASE_ID: &str = "pcfan.resolved-adjoint-reference.v1";

fn run() -> Result<String, String> {
    let picard_observation = PcFanResolvedFlowPicardLinearization {
        flow_model: PcFanFlowModel::IncompressibleCfd,
        backend: PcFanCfdBackend::Cpu,
        provenance: "fixture:pcfan:picard-oseen:v1".into(),
        state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
            vec![2.0, 0.0],
            vec![0.0, 3.0],
        ])
        .map_err(|error| error.to_string())?,
        state: vec![1.0, 2.0],
        fixed_point_residual: vec![0.0, 0.0],
    };
    picard_observation
        .validate()
        .map_err(|error| error.to_string())?;
    let probe = probe_pc_fan_resolved_flow_derivatives(
        &PcFanResolvedDerivativeProbeConfig {
            state_step: 1.0e-5,
            design_step: 1.0e-5,
            taylor_tolerance: 1.0e-8,
            ..Default::default()
        },
        &picard_observation,
        &[0.0],
        |state| {
            let mut observation = picard_observation.clone();
            observation.state = state.to_vec();
            Ok(observation)
        },
        |_| Ok(picard_observation.clone()),
        |state, design| Ok(state[0] + design[0]),
    )
    .map_err(|error| error.to_string())?;
    if probe.linearization.objective_state_gradient.len() != 2
        || probe.linearization.objective_design_gradient.len() != 1
    {
        return Err("resolved derivative probe shape".into());
    }

    let linearization = PcFanResolvedFlowLinearization {
        flow_model: PcFanFlowModel::IncompressibleCfdFull,
        backend: PcFanCfdBackend::Cpu,
        provenance: "fixture:pcfan:resolved-linearization:v1".into(),
        state_jacobian: PcFanResolvedFlowSparseMatrix::from_dense(&[
            vec![3.0, 1.0],
            vec![0.0, 2.0],
        ])
        .map_err(|error| error.to_string())?,
        state_residual: vec![0.0, 0.0],
        residual_design_jacobian: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        objective_state_gradient: vec![5.0, 6.0],
        objective_design_gradient: vec![7.0, 8.0],
    };
    let report =
        solve_pc_fan_resolved_flow_adjoint(&PcFanResolvedAdjointConfig::default(), &linearization)
            .map_err(|error| error.to_string())?;
    if !report.converged || report.adjoint_residual_norm > 1.0e-10 {
        return Err("resolved-adjoint fixture did not converge".into());
    }
    Ok(format!(
        "resolved full-state adjoint fixture passed with {} design gradients; no CFD solve was run",
        report.design_gradient.len()
    ))
}

fn main() {
    match run() {
        Ok(message) => {
            let mut result = QaResult::passed(CASE_ID);
            result.message = Some(message);
            println!(
                "{}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "{}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
