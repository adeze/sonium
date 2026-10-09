//! Compile-only/release QA contract for constrained structural topology.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    ConstrainedTopologyOptimizationStudy, run_constrained_topology_optimization,
};

const CASE_ID: &str = "structural.constrained-topology-reference.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    aggregated_stress_pa: f64,
    first_frequency_hz: f64,
    buckling_load_factor: f64,
    overhang_slope: f64,
    projected_reanalysis_relative_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = ConstrainedTopologyOptimizationStudy::introductory();
    study.base.base.iterations = 4;
    let result =
        run_constrained_topology_optimization(&study).map_err(|error| error.to_string())?;

    if !result.stress_feasible
        || !result.frequency_feasible
        || !result.buckling_feasible
        || !result.connected
        || !result.minimum_feature_feasible
        || !result.overhang_feasible
        || !result.reanalysis_feasible
        || !result.feasible
        || result.projected_reanalysis_relative_error > study.reanalysis_tolerance
    {
        return Err("constrained topology feasibility gates failed".into());
    }

    let mut infeasible = study;
    infeasible.allowable_stress_pa = 1.0;
    let rejected =
        run_constrained_topology_optimization(&infeasible).map_err(|error| error.to_string())?;
    if rejected.stress_feasible || rejected.feasible {
        return Err("infeasible stress limit was accepted".into());
    }

    let metrics = Metrics {
        aggregated_stress_pa: result.aggregated_stress_pa,
        first_frequency_hz: result.first_frequency_hz,
        buckling_load_factor: result.buckling_load_factor,
        overhang_slope: result.maximum_overhang_slope,
        projected_reanalysis_relative_error: result.projected_reanalysis_relative_error,
    };
    if [
        metrics.aggregated_stress_pa,
        metrics.first_frequency_hz,
        metrics.buckling_load_factor,
        metrics.overhang_slope,
        metrics.projected_reanalysis_relative_error,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("constrained topology emitted a non-finite metric".into());
    }
    Ok(metrics)
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("aggregated_stress_pa", metrics.aggregated_stress_pa)
                .metric("first_frequency_hz", metrics.first_frequency_hz)
                .metric("buckling_load_factor", metrics.buckling_load_factor)
                .metric("maximum_overhang_slope", metrics.overhang_slope)
                .metric(
                    "projected_reanalysis_relative_error",
                    metrics.projected_reanalysis_relative_error,
                )
                .maximum(
                    "projected_reanalysis_relative_error",
                    ConstrainedTopologyOptimizationStudy::introductory()
                        .reanalysis_tolerance,
                )
                .reference(
                    "bounded robust SIMP topology with stress, frequency, buckling, manufacturing, and projected-reanalysis gates",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-constrained-topology-reference-v1",
                )
                .dimension("1-D serial-bar topology surrogate; bounded CPU reference")
                .formulation(
                    "SIMP/RAMP continuation, morphology and minimum-feature checks, overhang gate, and physics-constrained reanalysis",
                )
                .tolerance(
                    "all feasibility gates pass; invalid stress limit is rejected; projected reanalysis error is within tolerance",
                )
                .seed("structural-constrained-topology-reference:bounded:cpu")
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
                .seed("structural-constrained-topology-reference:bounded:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL CONSTRAINED TOPOLOGY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
