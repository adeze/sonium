//! High-level SLFMM forward solve with exact near-field ILU and true residual gate.

use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.slfmm-nearfield-ilu.v1";

fn run() -> Result<(usize, f64), String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Slfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8);
    let solution = solver.solve(&problem).map_err(|error| error.to_string())?;
    if !solution
        .surface_pressure
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        return Err("SLFMM near-field ILU returned a non-finite solution".to_string());
    }
    let centers: Vec<[f64; 3]> = problem
        .mesh
        .elements
        .iter()
        .map(|element| element.center)
        .collect();
    let normals: Vec<[f64; 3]> = problem
        .mesh
        .elements
        .iter()
        .map(|element| element.normal)
        .collect();
    let rhs = problem.incident_field.compute_rhs(
        &centers,
        &normals,
        &problem.physics,
        problem.use_burton_miller,
    );
    let residual = solver
        .solve_rhs_many(&problem, &[rhs])
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "SLFMM block residual solve returned no result".to_string())?
        .relative_residual();
    if !residual.is_finite() || residual > 1.0e-6 {
        return Err(format!(
            "SLFMM near-field ILU residual {residual:e} exceeds 1e-6"
        ));
    }
    Ok((solution.num_dofs(), residual))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((dimension, residual))) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("true_residual", residual)
            .maximum("true_residual", 1.0e-6)
            .input_checksum("analytic/deterministic-fixture:bem-slfmm-nearfield-ilu-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-slfmm-nearfield-ilu-v1")
            .dimension("3-D rigid sphere; high-level SLFMM scalar GMRES")
            .reference("independent full-SLFMM operator residual certification")
            .seed("bem-slfmm-nearfield-ilu:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(_) => QaResult::failed(CASE_ID, "panic while running SLFMM ILU QA".to_string()),
    };
    let passed = matches!(&result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
