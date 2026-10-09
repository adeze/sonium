//! QA contract for the typed nonlinear port-flow promotion evidence boundary.
//!
//! This case intentionally uses deterministic accepted traces.  It validates
//! the promotion gate and serialized evidence contract without claiming that a
//! nonlinear 3-D runtime ladder has been executed.

use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::port_flow::{
    PORT_FLOW_PROMOTION_SCHEMA_VERSION, PortFlowConvergenceLevel, PortFlowPhasor,
    PortFlowPromotionEvidence, PortFlowPromotionGate, validate_port_flow_promotion,
};
use std::time::Instant;

const CASE_ID: &str = "speaker.port-flow-promotion-contract.v1";

fn level(mesh_size_m: f64, dt_s: f64, scale: f64) -> PortFlowConvergenceLevel {
    PortFlowConvergenceLevel {
        mesh_size_m,
        dt_s,
        frequency_hz: 120.0,
        port_flow: PortFlowPhasor {
            real: scale,
            imaginary: -0.2 * scale,
        },
        box_pressure: PortFlowPhasor {
            real: 4.0 * scale,
            imaginary: 0.5 * scale,
        },
        diaphragm_flow: PortFlowPhasor {
            real: 1.5 * scale,
            imaginary: 0.1 * scale,
        },
        max_linear_relative_residual: 1.0e-10,
        max_box_conservation_relative_residual: 1.0e-10,
        cap_continuity_relative_error: 1.0e-8,
        energy_algebraic_relative_closure: 1.0e-10,
        energy_physical_relative_closure: 1.0e-3,
        energy_interface_relative_residual: 1.0e-10,
    }
}

fn evidence() -> PortFlowPromotionEvidence {
    PortFlowPromotionEvidence {
        schema_version: PORT_FLOW_PROMOTION_SCHEMA_VERSION,
        provenance: "qa-speaker-port-flow-promotion-contract.v1".into(),
        frequency_hz: 120.0,
        spatial_levels: vec![
            level(0.004, 1.0e-5, 1.008),
            level(0.003, 1.0e-5, 1.004),
            level(0.002, 1.0e-5, 1.0),
        ],
        temporal_levels: vec![level(0.002, 2.0e-5, 1.004), level(0.002, 1.0e-5, 1.0)],
    }
}

fn run() -> Result<(f64, f64, f64, usize, usize), String> {
    let report = validate_port_flow_promotion(&evidence(), PortFlowPromotionGate::default())
        .map_err(|error| error.to_string())?;
    if report.spatial_levels < 3 || report.temporal_levels < 2 {
        return Err("promotion report retained too few refinement levels".into());
    }
    Ok((
        report.max_spatial_observable_relative_change,
        report.max_temporal_observable_relative_change,
        report.cross_ladder_observable_relative_change,
        report.spatial_levels,
        report.temporal_levels,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((spatial_change, temporal_change, cross_change, spatial_levels, temporal_levels)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("deterministic typed spatial/temporal port-flow promotion contract")
                .reference_checksum("analytic/deterministic-fixture:speaker-port-flow-promotion-v1")
                .dimension("3-D nonlinear port-flow evidence schema; accepted-trace contract")
                .formulation("common-fine-state phasor, residual, continuity, and energy gate")
                .tolerance("spatial, temporal, and cross-ladder observable changes <= 1e-2")
                .metric("spatial_observable_relative_change", spatial_change)
                .metric("temporal_observable_relative_change", temporal_change)
                .metric("cross_ladder_observable_relative_change", cross_change)
                .metric("spatial_levels", spatial_levels as f64)
                .metric("temporal_levels", temporal_levels as f64)
                .minimum("spatial_levels", 3.0)
                .minimum("temporal_levels", 2.0)
                .maximum("spatial_observable_relative_change", 1.0e-2)
                .maximum("temporal_observable_relative_change", 1.0e-2)
                .maximum("cross_ladder_observable_relative_change", 1.0e-2)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_contract_passes_without_runtime_solver_work() {
        let (spatial, temporal, cross, spatial_levels, temporal_levels) =
            run().expect("promotion contract should pass");
        assert!(spatial < 1.0e-2);
        assert!(temporal < 1.0e-2);
        assert!(cross < 1.0e-2);
        assert_eq!(spatial_levels, 3);
        assert_eq!(temporal_levels, 2);
    }
}
