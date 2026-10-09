//! Compile-time QA contract for resolved PC-fan CFD independence evidence.
//!
//! This case intentionally uses deterministic solved-observation fixtures;
//! it validates the production admission contract without launching a CFD
//! simulation. A resolved-flow runner can replace the fixture observations
//! while preserving the same report schema and promotion gates.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdIndependenceAxis, PcFanCfdIndependenceConfig, PcFanCfdIndependenceSample,
    PcFanCfdIndependenceStudy, evaluate_pc_fan_cfd_independence,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.cfd-independence-reference.v1";

fn sample(level: f64, scale: f64) -> PcFanCfdIndependenceSample {
    PcFanCfdIndependenceSample {
        level,
        airflow_cfm: 20.0 * scale,
        pressure_drop_pa: 80.0 * scale,
        axial_loading_n: 1.5 * scale,
        residual: 1.0e-5,
        relative_mass_imbalance: 0.0,
        relative_momentum_imbalance: 0.0,
        relative_pressure_closure: 0.0,
        relative_energy_imbalance: 0.0,
        converged: true,
    }
}

fn studies() -> Vec<PcFanCfdIndependenceStudy> {
    vec![
        PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::Mesh,
            samples: vec![sample(2.0, 0.99), sample(3.0, 1.0)],
        },
        PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::TimeStep,
            samples: vec![sample(0.002, 0.99), sample(0.001, 1.0)],
        },
        PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::RotorRevolutions,
            samples: vec![sample(3.0, 0.99), sample(6.0, 1.0)],
        },
        PcFanCfdIndependenceStudy {
            axis: PcFanCfdIndependenceAxis::NonlinearIterations,
            samples: vec![sample(20.0, 0.99), sample(40.0, 1.0)],
        },
    ]
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let report =
        evaluate_pc_fan_cfd_independence(&PcFanCfdIndependenceConfig::default(), &studies())
            .map_err(|error| error.to_string())?;
    if !report.is_valid() {
        return Err("deterministic independence fixture did not pass".into());
    }

    let max_airflow_change = report
        .studies
        .iter()
        .map(|study| study.airflow_relative_change)
        .fold(0.0, f64::max);
    let max_pressure_change = report
        .studies
        .iter()
        .map(|study| study.pressure_relative_change)
        .fold(0.0, f64::max);
    let max_loading_change = report
        .studies
        .iter()
        .map(|study| study.axial_loading_relative_change)
        .fold(0.0, f64::max);
    Ok((
        report.studies.len() as f64,
        if report.converged { 1.0 } else { 0.0 },
        max_airflow_change.max(max_pressure_change),
        max_loading_change,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((axes, converged, max_flow_pressure_change, max_loading_change)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("required_axes", axes)
                .metric("converged_status", converged)
                .metric(
                    "max_flow_pressure_relative_change",
                    max_flow_pressure_change,
                )
                .metric("max_loading_relative_change", max_loading_change)
                .minimum("required_axes", 4.0)
                .minimum("converged_status", 1.0)
                .maximum("max_flow_pressure_relative_change", 0.02)
                .maximum("max_loading_relative_change", 0.02)
                .reference_checksum("analytic/deterministic-fixture:pcfan-cfd-independence-v1")
                .input_checksum("pcfan-cfd-independence:mesh-dt-revolutions-iterations")
                .seed("pcfan-cfd-independence-reference:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            record
                .to_json_line()
                .expect("serialize PC-fan CFD independence QA result");
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error);
            record
                .to_json_line()
                .expect("serialize PC-fan CFD independence QA failure");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
