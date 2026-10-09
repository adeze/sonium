//! Compile-only QA wrapper for 3-D panel-state remesh transfer.

use std::time::Instant;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::QaResult;
use sonium_sim_propellers::{
    Blade3dConfig, PanelStateTransferOptions, PropellerDesign, transfer_panel_scalar_state,
};

const CASE_ID: &str = "propeller.mesh-state-transfer-reference.v1";

fn center(panel: &sonium_sim_propellers::BladePanel) -> [f64; 3] {
    panel.vertices.iter().fold([0.0; 3], |mut sum, point| {
        sum[0] += point.x * 0.25;
        sum[1] += point.y * 0.25;
        sum[2] += point.z * 0.25;
        sum
    })
}

fn run() -> Result<(usize, f64, f64), String> {
    let design = PropellerDesign::conventional(PropellerSource::drone_5inch(18_000.0));
    let source_mesh = Blade3dConfig {
        radial_segments: 6,
        chord_segments: 3,
        ..Blade3dConfig::default()
    }
    .try_build_mesh(&design)
    .map_err(|error| error.to_string())?;
    let target_mesh = Blade3dConfig {
        radial_segments: 13,
        chord_segments: 4,
        ..Blade3dConfig::default()
    }
    .try_build_mesh(&design)
    .map_err(|error| error.to_string())?;
    let source_values = source_mesh
        .panels
        .iter()
        .map(|panel| {
            let point = center(panel);
            1.0 + 0.25 * panel.blade_index as f64 + point[0] - 0.5 * point[1] + point[2]
        })
        .collect::<Vec<_>>();
    let options = PanelStateTransferOptions {
        maximum_source_distance_m: 0.05,
        ..PanelStateTransferOptions::default()
    };
    let result = transfer_panel_scalar_state(&source_mesh, &source_values, &target_mesh, options)
        .map_err(|error| error.to_string())?;
    if !result.is_finite() || result.values.len() != target_mesh.panels.len() {
        return Err("mesh-state transfer result is not finite or complete".into());
    }
    if result.maximum_source_distance_m > options.maximum_source_distance_m {
        return Err("mesh-state transfer exceeded its distance gate".into());
    }
    Ok((
        result.values.len(),
        result.maximum_source_distance_m,
        result.area_weighted_mean_delta,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((target_panels, maximum_source_distance_m, area_weighted_mean_delta)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("target_panels", target_panels as f64)
                .metric("maximum_source_distance_m", maximum_source_distance_m)
                .metric("area_weighted_mean_delta", area_weighted_mean_delta)
                .input_checksum("analytic/deterministic-fixture:propeller-mesh-state-transfer-v1")
                .reference(
                    "same-blade inverse-distance geometric panel-state transfer across remeshed 3-D blade panels",
                )
                .reference_checksum("analytic/invariant:propeller-mesh-state-transfer-v1")
                .tolerance("finite transfer, preserved blade identity, bounded nearest-source distance")
                .seed("propeller-mesh-state-transfer-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
