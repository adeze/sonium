//! Compile-only QA contract for bounded 3-D topology convergence evidence.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    Solid3DTopologyStudy, run_solid3d_topology_mesh_convergence,
};

const CASE_ID: &str = "structural.solid-topology-convergence.v2";
const TOLERANCE: f64 = 0.75;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    coarse_nx: usize,
    coarse_ny: usize,
    coarse_nz: usize,
    refined_nx: usize,
    refined_ny: usize,
    refined_nz: usize,
    coarse_nodes: usize,
    refined_nodes: usize,
    coarse_elements: usize,
    refined_elements: usize,
    objective_relative_change: f64,
    stress_relative_change: f64,
    rank_pair_count: usize,
    rank_discordant_pair_count: usize,
    rank_discordance_fraction: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = 2;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.base.modal_modes = 1;
    study.volume_fraction = 0.90;
    study.iterations = 8;
    study.convergence_tolerance = TOLERANCE;
    study.allowable_stress_pa = 1.0e14;

    let report = run_solid3d_topology_mesh_convergence(&study, TOLERANCE)
        .map_err(|error| error.to_string())?;
    if report.coarse_resolution != [2, 1, 1]
        || report.refined_resolution != [4, 2, 2]
        || !report.converged
        || !report.rank_preserved
        || report.coarse.nodes >= report.refined.nodes
        || report.coarse.elements >= report.refined.elements
        || report.rank_pair_count == 0
        || report.rank_discordant_pair_count > report.rank_pair_count
        || !report.objective_relative_change.is_finite()
        || !report.stress_relative_change.is_finite()
        || !report.rank_discordance_fraction.is_finite()
    {
        return Err(format!("3-D topology convergence gate failed: {report:#?}"));
    }

    Ok(Metrics {
        coarse_nx: report.coarse_resolution[0],
        coarse_ny: report.coarse_resolution[1],
        coarse_nz: report.coarse_resolution[2],
        refined_nx: report.refined_resolution[0],
        refined_ny: report.refined_resolution[1],
        refined_nz: report.refined_resolution[2],
        coarse_nodes: report.coarse.nodes,
        refined_nodes: report.refined.nodes,
        coarse_elements: report.coarse.elements,
        refined_elements: report.refined.elements,
        objective_relative_change: report.objective_relative_change,
        stress_relative_change: report.stress_relative_change,
        rank_pair_count: report.rank_pair_count,
        rank_discordant_pair_count: report.rank_discordant_pair_count,
        rank_discordance_fraction: report.rank_discordance_fraction,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("coarse_nx", metrics.coarse_nx as f64)
                .metric("coarse_ny", metrics.coarse_ny as f64)
                .metric("coarse_nz", metrics.coarse_nz as f64)
                .metric("refined_nx", metrics.refined_nx as f64)
                .metric("refined_ny", metrics.refined_ny as f64)
                .metric("refined_nz", metrics.refined_nz as f64)
                .metric("coarse_nodes", metrics.coarse_nodes as f64)
                .metric("refined_nodes", metrics.refined_nodes as f64)
                .metric("coarse_elements", metrics.coarse_elements as f64)
                .metric("refined_elements", metrics.refined_elements as f64)
                .metric(
                    "objective_relative_change",
                    metrics.objective_relative_change,
                )
                .metric("stress_relative_change", metrics.stress_relative_change)
                .metric("rank_pair_count", metrics.rank_pair_count as f64)
                .metric(
                    "rank_discordant_pair_count",
                    metrics.rank_discordant_pair_count as f64,
                )
                .metric(
                    "rank_discordance_fraction",
                    metrics.rank_discordance_fraction,
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-solid-topology-convergence-v2",
                )
                .dimension("3-D P1 tetrahedral nx/ny/nz refinement")
                .formulation(
                    "SIMP topology objective/stress/volume and projected density rank gates",
                )
                .seed("structural-solid-topology-convergence:2x1x1-to-4x2x2:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-solid-topology-convergence:2x1x1-to-4x2x2:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL SOLID TOPOLOGY CONVERGENCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
