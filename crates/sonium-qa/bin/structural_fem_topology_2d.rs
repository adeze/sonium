//! Release QA for the assembled 2-D FEM topology workflow.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    FemTopology2DStudy, StructuralModelAssumption, run_fem_topology_2d,
};

const CASE_ID: &str = "structural.fem-topology-2d.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-7;
const ENERGY_TOLERANCE: f64 = 1.0e-8;
const GRADIENT_TOLERANCE: f64 = 2.0e-1;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    nodes: usize,
    elements: usize,
    iterations: usize,
    final_volume_fraction: f64,
    projected_reanalysis_relative_error: f64,
    body_fitted_reanalysis_relative_error: f64,
    solver_residual: f64,
    free_equilibrium_residual_n: f64,
    force_balance_residual_n: f64,
    energy_balance_relative_error: f64,
    gradient_taylor_relative_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = FemTopology2DStudy::introductory();
    study.geometry.nx = 6;
    study.geometry.ny = 2;
    study.volume_fraction = 0.95;
    study.move_limit = 0.01;
    study.continuation_steps = 1;
    study.iterations_per_stage = 8;
    study.filter_radius_m = 0.02;
    study.minimum_feature_cells = 1;
    study.maximum_overhang_slope = 100.0;
    let result = run_fem_topology_2d(&study).map_err(|error| error.to_string())?;

    if result.model_assumption != StructuralModelAssumption::SimpTopology2D {
        return Err(format!(
            "unexpected model assumption {:?}",
            result.model_assumption
        ));
    }
    if result.nodes == 0 || result.elements == 0 || result.iterations == 0 {
        return Err("topology run returned an empty mesh or iteration history".into());
    }
    if !result.connected || !result.reanalysis_feasible {
        return Err(format!(
            "topology feasibility gates failed: connected={} reanalysis_feasible={}",
            result.connected, result.reanalysis_feasible
        ));
    }
    let finite = [
        result.initial_compliance_j,
        result.final_compliance_j,
        result.projected_reanalysis_relative_error,
        result.body_fitted_reanalysis_relative_error,
        result.final_volume_fraction,
        result.solver_residual,
        result.free_equilibrium_residual_n,
        result.force_balance_residual_n,
        result.energy_balance_relative_error,
        result.gradient_taylor_relative_error,
    ];
    if finite.iter().any(|value| !value.is_finite()) {
        return Err("topology result contains a non-finite diagnostic".into());
    }
    if result.final_volume_fraction <= 0.0 || result.final_volume_fraction > 1.0 {
        return Err(format!(
            "invalid final volume fraction {}",
            result.final_volume_fraction
        ));
    }
    if result.solver_residual > RESIDUAL_TOLERANCE
        || result.free_equilibrium_residual_n > RESIDUAL_TOLERANCE
        || result.force_balance_residual_n > RESIDUAL_TOLERANCE
        || result.energy_balance_relative_error > ENERGY_TOLERANCE
        || result.gradient_taylor_relative_error > GRADIENT_TOLERANCE
    {
        return Err(format!(
            "topology diagnostic gate failed: residuals=({}, {}, {}), energy={}, gradient={}",
            result.solver_residual,
            result.free_equilibrium_residual_n,
            result.force_balance_residual_n,
            result.energy_balance_relative_error,
            result.gradient_taylor_relative_error
        ));
    }

    Ok(Metrics {
        nodes: result.nodes,
        elements: result.elements,
        iterations: result.iterations,
        final_volume_fraction: result.final_volume_fraction,
        projected_reanalysis_relative_error: result.projected_reanalysis_relative_error,
        body_fitted_reanalysis_relative_error: result.body_fitted_reanalysis_relative_error,
        solver_residual: result.solver_residual,
        free_equilibrium_residual_n: result.free_equilibrium_residual_n,
        force_balance_residual_n: result.force_balance_residual_n,
        energy_balance_relative_error: result.energy_balance_relative_error,
        gradient_taylor_relative_error: result.gradient_taylor_relative_error,
    })
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("nodes", metrics.nodes as f64)
                .metric("elements", metrics.elements as f64)
                .metric("iterations", metrics.iterations as f64)
                .metric("final_volume_fraction", metrics.final_volume_fraction)
                .metric(
                    "projected_reanalysis_relative_error",
                    metrics.projected_reanalysis_relative_error,
                )
                .metric(
                    "body_fitted_reanalysis_relative_error",
                    metrics.body_fitted_reanalysis_relative_error,
                )
                .metric("solver_residual", metrics.solver_residual)
                .metric(
                    "free_equilibrium_residual_n",
                    metrics.free_equilibrium_residual_n,
                )
                .metric("force_balance_residual_n", metrics.force_balance_residual_n)
                .metric(
                    "energy_balance_relative_error",
                    metrics.energy_balance_relative_error,
                )
                .metric(
                    "gradient_taylor_relative_error",
                    metrics.gradient_taylor_relative_error,
                )
                .maximum("solver_residual", RESIDUAL_TOLERANCE)
                .maximum("free_equilibrium_residual_n", RESIDUAL_TOLERANCE)
                .maximum("force_balance_residual_n", RESIDUAL_TOLERANCE)
                .maximum("energy_balance_relative_error", ENERGY_TOLERANCE)
                .maximum("gradient_taylor_relative_error", GRADIENT_TOLERANCE)
                .reference("assembled P1 plane-stress FEM SIMP with filtered/projected design and body-fitted reanalysis")
                .reference_checksum("analytic/deterministic-fixture:structural-fem-topology-2d-v1")
                .dimension("2-D P1 triangular FEM, static compliance, modal, stress, connectivity, and reanalysis gates")
                .formulation("density-scaled assembled elasticity with continuation, volume projection, and adjoint Taylor check")
                .tolerance("residual <= 1e-7; energy <= 1e-8; gradient Taylor error <= 2e-1")
                .seed("structural-fem-topology-2d:introductory:cpu")
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
                .seed("structural-fem-topology-2d:introductory:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            eprintln!("STRUCTURAL FEM TOPOLOGY 2-D QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
