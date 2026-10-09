//! Release QA for the bounded robust 2-D FEM topology workflow.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{FemTopology2DRobustStudy, run_fem_topology_2d_robust};

const CASE_ID: &str = "structural.fem-topology-2d-robust.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-7;
const ENERGY_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cases: usize,
    nominal_volume_fraction: f64,
    eroded_volume_fraction: f64,
    dilated_volume_fraction: f64,
    worst_case_compliance_j: f64,
    worst_case_aggregated_stress_pa: f64,
    max_solver_residual: f64,
    max_equilibrium_residual_n: f64,
    max_force_balance_residual_n: f64,
    max_energy_balance_relative_error: f64,
    gradient_taylor_relative_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = FemTopology2DRobustStudy::introductory();
    study.envelope.topology.geometry.nx = 6;
    study.envelope.topology.geometry.ny = 2;
    study.envelope.topology.volume_fraction = 0.95;
    study.envelope.topology.move_limit = 0.01;
    study.envelope.topology.continuation_steps = 1;
    study.envelope.topology.iterations_per_stage = 3;
    study.envelope.topology.filter_radius_m = 0.02;
    study.envelope.topology.minimum_feature_cells = 1;
    study.envelope.topology.maximum_overhang_slope = 100.0;
    study.envelope.topology.edge_force_n = study.envelope.load_cases[0].edge_force_n;
    study.allowable_compliance_j = 1.0e6;

    let result = run_fem_topology_2d_robust(&study).map_err(|error| error.to_string())?;
    if result.cases.len() != 12 {
        return Err(format!(
            "expected 12 morphology/uncertainty/load cases, got {}",
            result.cases.len()
        ));
    }
    if !result.factorization_reused_within_scenario
        || !result.morphology_feasible
        || !result.uncertainty_feasible
        || !result.robust_feasible
        || !result.erosion_connected
        || !result.dilation_connected
    {
        return Err(format!(
            "robust feasibility gates failed: factorization={}, morphology={}, uncertainty={}, robust={}, erosion_connected={}, dilation_connected={}",
            result.factorization_reused_within_scenario,
            result.morphology_feasible,
            result.uncertainty_feasible,
            result.robust_feasible,
            result.erosion_connected,
            result.dilation_connected,
        ));
    }
    if result.eroded_volume_fraction > result.nominal_volume_fraction + 1.0e-12
        || result.dilated_volume_fraction + 1.0e-12 < result.nominal_volume_fraction
    {
        return Err("morphology volume ordering failed".into());
    }

    let max_solver_residual = result
        .cases
        .iter()
        .map(|case| case.solver_residual)
        .fold(0.0, f64::max);
    let max_equilibrium_residual_n = result
        .cases
        .iter()
        .map(|case| case.free_equilibrium_residual_n)
        .fold(0.0, f64::max);
    let max_force_balance_residual_n = result
        .cases
        .iter()
        .map(|case| case.force_balance_residual_n)
        .fold(0.0, f64::max);
    let max_energy_balance_relative_error = result
        .cases
        .iter()
        .map(|case| case.energy_balance_relative_error)
        .fold(0.0, f64::max);
    let scalars = [
        result.nominal_volume_fraction,
        result.eroded_volume_fraction,
        result.dilated_volume_fraction,
        result.worst_case_compliance_j,
        result.worst_case_aggregated_stress_pa,
        max_solver_residual,
        max_equilibrium_residual_n,
        max_force_balance_residual_n,
        max_energy_balance_relative_error,
        result.gradient_taylor_relative_error,
    ];
    if scalars.iter().any(|value| !value.is_finite())
        || max_solver_residual > RESIDUAL_TOLERANCE
        || max_equilibrium_residual_n > RESIDUAL_TOLERANCE
        || max_force_balance_residual_n > RESIDUAL_TOLERANCE
        || max_energy_balance_relative_error > ENERGY_TOLERANCE
        || !result.stress_feasible
        || !result
            .cases
            .iter()
            .all(|case| case.converged && case.feasible && case.stress_feasible)
    {
        return Err(format!(
            "robust numerical gates failed: solver={max_solver_residual:.3e}, equilibrium={max_equilibrium_residual_n:.3e}, force={max_force_balance_residual_n:.3e}, energy={max_energy_balance_relative_error:.3e}, cases={}",
            result
                .cases
                .iter()
                .filter(|case| case.converged && case.feasible && case.stress_feasible)
                .count()
        ));
    }

    Ok(Metrics {
        cases: result.cases.len(),
        nominal_volume_fraction: result.nominal_volume_fraction,
        eroded_volume_fraction: result.eroded_volume_fraction,
        dilated_volume_fraction: result.dilated_volume_fraction,
        worst_case_compliance_j: result.worst_case_compliance_j,
        worst_case_aggregated_stress_pa: result.worst_case_aggregated_stress_pa,
        max_solver_residual,
        max_equilibrium_residual_n,
        max_force_balance_residual_n,
        max_energy_balance_relative_error,
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
                .metric("cases", metrics.cases as f64)
                .metric("nominal_volume_fraction", metrics.nominal_volume_fraction)
                .metric("eroded_volume_fraction", metrics.eroded_volume_fraction)
                .metric("dilated_volume_fraction", metrics.dilated_volume_fraction)
        .metric("worst_case_compliance_j", metrics.worst_case_compliance_j)
        .metric(
            "worst_case_aggregated_stress_pa",
            metrics.worst_case_aggregated_stress_pa,
        )
                .metric("max_solver_residual", metrics.max_solver_residual)
                .metric(
                    "max_equilibrium_residual_n",
                    metrics.max_equilibrium_residual_n,
                )
                .metric(
                    "max_force_balance_residual_n",
                    metrics.max_force_balance_residual_n,
                )
                .metric(
                    "max_energy_balance_relative_error",
                    metrics.max_energy_balance_relative_error,
                )
                .metric(
                    "gradient_taylor_relative_error",
                    metrics.gradient_taylor_relative_error,
                )
                .minimum("cases", 12.0)
                .maximum("max_solver_residual", RESIDUAL_TOLERANCE)
                .maximum("max_equilibrium_residual_n", RESIDUAL_TOLERANCE)
                .maximum("max_force_balance_residual_n", RESIDUAL_TOLERANCE)
                .maximum("max_energy_balance_relative_error", ENERGY_TOLERANCE)
                .reference(
                    "assembled P1 plane-stress FEM envelope with erosion/dilation and explicit material/load uncertainty; bounded 2-D robustness only",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-fem-topology-2d-robust-v1",
                )
                .dimension("6x2 triangular mesh; 3 morphologies x 2 uncertainty cases x 2 load cases")
                .formulation(
                    "SIMP density filtering/projection, shared-factorization envelope reanalysis, graph morphology, and finite sensitivity gate",
                )
                .tolerance("solver/equilibrium/force <= 1e-7; energy <= 1e-8")
                .seed("structural-fem-topology-2d-robust:6x2:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL ROBUST TOPOLOGY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
