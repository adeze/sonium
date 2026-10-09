//! Compile-only QA wrapper for the uniform-mean-flow moving FW-H reference.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    MeanFlowFwhOptions, MovingFwhElementHistory, MovingFwhSample, Point3,
    evaluate_mean_flow_fwh_reference,
};

const CASE_ID: &str = "propeller.mean-flow-fwh-reference.v1";

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn run() -> Result<(f64, f64, f64), String> {
    let sound_speed_m_s = 343.0;
    let mean_flow_speed_m_s = 34.3;
    let propagation_time_s = 10.0 / (sound_speed_m_s + mean_flow_speed_m_s);
    let history = MovingFwhElementHistory {
        samples: vec![
            MovingFwhSample {
                time_s: 0.0,
                position: point(0.0, 0.0, 0.0),
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 0.0,
                normal_velocity_m_s: 0.0,
            },
            MovingFwhSample {
                time_s: 2.0,
                position: point(0.0, 0.0, 0.0),
                normal: point(0.0, 0.0, 1.0),
                area_m2: 0.2,
                pressure_pa: 1.0,
                normal_velocity_m_s: 0.0,
            },
        ],
    };
    let result = evaluate_mean_flow_fwh_reference(
        &[history],
        MeanFlowFwhOptions {
            observer_time_s: 1.0 + propagation_time_s,
            observer_position: point(10.0, 0.0, 0.0),
            mean_flow_m_s: point(mean_flow_speed_m_s, 0.0, 0.0),
            minimum_observer_distance_m: 0.1,
            ..MeanFlowFwhOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let element = result.elements.first().ok_or("missing element result")?;
    if !result.is_finite()
        || (element.retarded_time_s - 1.0).abs() > 1.0e-9
        || (element.propagation_time_s - propagation_time_s).abs() > 1.0e-9
    {
        return Err("uniform-flow retarded-time identity failed".into());
    }
    Ok((
        element.retarded_time_s,
        result.mean_flow_mach,
        result.total_pressure_pa,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((retarded_time_s, mean_flow_mach, total_pressure_pa)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("retarded_time_s", retarded_time_s)
                .metric("mean_flow_mach", mean_flow_mach)
                .metric("total_pressure_pa", total_pressure_pa)
                .input_checksum("analytic/deterministic-fixture:propeller-mean-flow-fwh-v1")
                .reference("uniform-mean-flow convected retarded-time source identity")
                .reference_checksum("analytic/invariant:propeller-mean-flow-fwh-v1")
                .dimension("one stationary moving-FW-H element; two source samples; f64 CPU")
                .formulation("bounded uniform-mean-flow moving-surface FW-H reference")
                .tolerance("retarded-time <= 1e-9 s; convected-ray diagnostics finite")
                .seed("propeller-mean-flow-fwh-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
