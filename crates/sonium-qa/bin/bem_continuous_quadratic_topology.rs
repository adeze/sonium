//! Bounded CPU evidence for globally continuous quadratic BEM surface DOFs.
//!
//! This case exercises the production `BemSolver::with_continuous_quadratic`
//! path on a small Tri3 sphere mesh.  It is intentionally limited to the
//! topology/assembly contract; it does not claim broad higher-order accuracy,
//! large-problem performance, or external-reference parity.

use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.continuous-quadratic-topology.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    panels: usize,
    global_dofs: usize,
    local_dof_slots: usize,
    shared_dof_slots: usize,
    max_pressure_norm: f64,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::try_rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4)
        .map_err(|error| format!("continuous quadratic fixture construction failed: {error}"))?;
    let map = problem
        .mesh
        .continuous_quadratic_dof_map()
        .map_err(|error| format!("continuous quadratic topology rejected fixture: {error}"))?;

    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_tolerance(1.0e-8)
        .with_max_iterations(200);
    solver
        .validate()
        .map_err(|error| format!("continuous quadratic solver validation failed: {error}"))?;
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("continuous quadratic production solve failed: {error}"))?;

    if solution.num_dofs() != map.num_dofs() {
        return Err(format!(
            "solution exposed {} DOFs, topology map has {}",
            solution.num_dofs(),
            map.num_dofs()
        ));
    }
    if solution
        .surface_pressure
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("continuous quadratic solution contains non-finite pressure".into());
    }

    let local_dof_slots = solution
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.dof_addresses.len())
        .sum::<usize>();
    let shared_dof_slots = local_dof_slots.saturating_sub(map.num_dofs());
    if shared_dof_slots == 0 {
        return Err("fixture did not exercise shared continuous DOFs".into());
    }

    Ok(Metrics {
        panels: solution
            .elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .count(),
        global_dofs: map.num_dofs(),
        local_dof_slots,
        shared_dof_slots,
        max_pressure_norm: solution
            .surface_pressure
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max),
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("panels", metrics.panels as f64)
                .metric("global_dofs", metrics.global_dofs as f64)
                .metric("local_dof_slots", metrics.local_dof_slots as f64)
                .metric("shared_dof_slots", metrics.shared_dof_slots as f64)
                .metric("max_pressure_norm", metrics.max_pressure_norm)
                .minimum("panels", 1.0)
                .minimum("global_dofs", 1.0)
                .minimum("shared_dof_slots", 1.0)
                .reference("independent global vertex/edge DOF deduplication finite production solve")
                .dimension("small 3D rigid-sphere Tri3 surface mesh")
                .formulation("CPU TBEM globally continuous quadratic surface unknowns on small rigid-sphere fixture")
                .tolerance("shared vertex/edge DOFs are deduplicated and the production solve is finite")
                .seed("bem-continuous-quadratic-topology:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent global vertex/edge DOF deduplication finite production solve")
                .dimension("small 3D rigid-sphere Tri3 surface mesh")
                .formulation("CPU TBEM globally continuous quadratic surface unknowns on small rigid-sphere fixture")
                .tolerance("shared vertex/edge DOFs are deduplicated and the production solve is finite")
                .seed("bem-continuous-quadratic-topology:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("BEM CONTINUOUS QUADRATIC TOPOLOGY FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
