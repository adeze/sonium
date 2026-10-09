//! Release QA for the assembled 3-D tetrahedral solid topology workflow.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    Solid3DTopologyStudy, StructuralModelAssumption, run_solid3d_topology,
};

const CASE_ID: &str = "structural.solid-topology-3d.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const EQUILIBRIUM_TOLERANCE: f64 = 1.0e-6;
const ENERGY_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    nodes: usize,
    elements: usize,
    cases: usize,
    iterations: usize,
    volume_fraction: f64,
    first_frequency_hz: f64,
    max_von_mises_pa: f64,
    aggregated_stress_pa: f64,
    factorization_memory_bytes: usize,
    max_solver_residual: f64,
    max_equilibrium_residual_n: f64,
    max_force_balance_residual_n: f64,
    max_energy_balance_relative_error: f64,
}

fn run() -> Result<Metrics, String> {
    // Keep this release fixture small enough for ordinary CPU smoke runs while
    // still exercising the real tetrahedral assembly, two load cases, density
    // continuation, morphology checks, and body-fitted reanalysis.
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = 2;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.volume_fraction = 0.90;
    study.iterations = 2;
    study.allowable_stress_pa = 1.0e14;

    let result = run_solid3d_topology(&study).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::SimpTopology3D {
        return Err(format!(
            "unexpected model assumption {:?}",
            result.model_assumption
        ));
    }
    if result.cases.len() != 2
        || !result.factorization_reused_across_cases
        || result.factorization_memory_bytes == 0
    {
        return Err(format!(
            "multi-load/factorization contract failed: cases={} reused={} memory={}",
            result.cases.len(),
            result.factorization_reused_across_cases,
            result.factorization_memory_bytes
        ));
    }
    if !result.connected
        || !result.stress_feasible
        || !result.morphology_feasible
        || !result.body_fitted_reanalysis
        || !result.body_fitted_feasible
    {
        return Err(format!(
            "3-D topology feasibility gates failed: connected={} stress={} morphology={} body_fitted={} body_feasible={}",
            result.connected,
            result.stress_feasible,
            result.morphology_feasible,
            result.body_fitted_reanalysis,
            result.body_fitted_feasible
        ));
    }
    let mut max_solver_residual: f64 = 0.0;
    let mut max_equilibrium_residual_n: f64 = 0.0;
    let mut max_force_balance_residual_n: f64 = 0.0;
    let mut max_energy_balance_relative_error: f64 = 0.0;
    for case in result.cases.iter().chain(result.body_fitted_cases.iter()) {
        if !case.converged || !case.aggregated_stress_pa.is_finite() {
            return Err(format!("case {} did not converge", case.name));
        }
        max_solver_residual = max_solver_residual.max(case.solver_residual);
        max_equilibrium_residual_n =
            max_equilibrium_residual_n.max(case.free_equilibrium_residual_n);
        max_force_balance_residual_n =
            max_force_balance_residual_n.max(case.force_balance_residual_n);
        max_energy_balance_relative_error =
            max_energy_balance_relative_error.max(case.energy_balance_relative_error);
    }
    if !result.volume_fraction.is_finite()
        || !result.first_frequency_hz.is_finite()
        || !result.max_von_mises_pa.is_finite()
        || !result.aggregated_stress_pa.is_finite()
        || !result.stress_feasible
        || result.volume_fraction <= 0.0
        || result.volume_fraction > 1.0
        || result.first_frequency_hz <= 0.0
        || max_solver_residual > RESIDUAL_TOLERANCE
        || max_equilibrium_residual_n > EQUILIBRIUM_TOLERANCE
        || max_force_balance_residual_n > EQUILIBRIUM_TOLERANCE
        || max_energy_balance_relative_error > ENERGY_TOLERANCE
    {
        return Err(format!(
            "3-D diagnostic gate failed: volume={} frequency={} max_stress={} aggregated_stress={} residual={} equilibrium={} force={} energy={}",
            result.volume_fraction,
            result.first_frequency_hz,
            result.max_von_mises_pa,
            result.aggregated_stress_pa,
            max_solver_residual,
            max_equilibrium_residual_n,
            max_force_balance_residual_n,
            max_energy_balance_relative_error
        ));
    }

    Ok(Metrics {
        nodes: result.nodes,
        elements: result.elements,
        cases: result.cases.len(),
        iterations: result.iterations,
        volume_fraction: result.volume_fraction,
        first_frequency_hz: result.first_frequency_hz,
        max_von_mises_pa: result.max_von_mises_pa,
        aggregated_stress_pa: result.aggregated_stress_pa,
        factorization_memory_bytes: result.factorization_memory_bytes,
        max_solver_residual,
        max_equilibrium_residual_n,
        max_force_balance_residual_n,
        max_energy_balance_relative_error,
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
                .metric("cases", metrics.cases as f64)
                .metric("iterations", metrics.iterations as f64)
                .metric("volume_fraction", metrics.volume_fraction)
                .metric("first_frequency_hz", metrics.first_frequency_hz)
                .metric("max_von_mises_pa", metrics.max_von_mises_pa)
                .metric("aggregated_stress_pa", metrics.aggregated_stress_pa)
                .metric(
                    "factorization_memory_bytes",
                    metrics.factorization_memory_bytes as f64,
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
                .minimum("cases", 2.0)
                .maximum("max_solver_residual", RESIDUAL_TOLERANCE)
                .maximum("max_equilibrium_residual_n", EQUILIBRIUM_TOLERANCE)
                .maximum("max_force_balance_residual_n", EQUILIBRIUM_TOLERANCE)
                .maximum("max_energy_balance_relative_error", ENERGY_TOLERANCE)
                .reference("assembled P1 tetrahedral SIMP multi-load envelope with morphology and body-fitted reanalysis")
                .reference_checksum("analytic/deterministic-fixture:structural-solid-topology-3d-v1")
                .dimension("3-D linear tetrahedra, two load cases, density continuation, modal and static feasibility")
                .formulation("density-scaled assembled elasticity with shared constrained factorization")
                .tolerance("solver <= 1e-8; equilibrium/force <= 1e-6; energy <= 1e-8")
                .seed("structural-solid-topology-3d:2x1x1:two-load:cpu")
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
                .seed("structural-solid-topology-3d:2x1x1:two-load:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            eprintln!("STRUCTURAL SOLID TOPOLOGY 3-D QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
