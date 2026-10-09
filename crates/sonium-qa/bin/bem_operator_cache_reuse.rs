//! Release-backed persistent BEM operator-cache contract.

use sonium_bem::BemReusePolicy;
use sonium_bem::core::{
    AssemblyMethod, BemOperatorCache, BemProblem, BemSolver, IncidentField, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.operator-cache-reuse.v1";

struct Metrics {
    dimension: usize,
    max_solution_difference: f64,
    rebuilds: u64,
    reuse_hits: u64,
}

fn run() -> Result<Metrics, String> {
    let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let mut cache = BemOperatorCache::new();
    solver
        .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
        .map_err(|error| error.to_string())?;
    let second_problem = base
        .clone()
        .with_incident_field(IncidentField::plane_wave_neg_z());
    let cached = solver
        .solve_with_cache(&second_problem, &mut cache, BemReusePolicy::default())
        .map_err(|error| error.to_string())?;
    let independent = solver
        .solve(&second_problem)
        .map_err(|error| error.to_string())?;
    let max_solution_difference = cached
        .surface_pressure
        .iter()
        .zip(&independent.surface_pressure)
        .map(|(cached, independent)| (*cached - *independent).norm())
        .fold(0.0_f64, f64::max);
    if max_solution_difference > 1.0e-10 {
        return Err(format!(
            "cached solution differs from independent solve: {max_solution_difference:e}"
        ));
    }
    if cache.rebuilds() != 1 || cache.reuse_hits() != 1 {
        return Err(format!(
            "unexpected cache telemetry: rebuilds={} reuse_hits={}",
            cache.rebuilds(),
            cache.reuse_hits()
        ));
    }
    Ok(Metrics {
        dimension: base.mesh.elements.len(),
        max_solution_difference,
        rebuilds: cache.rebuilds(),
        reuse_hits: cache.reuse_hits(),
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("max_solution_difference", metrics.max_solution_difference)
            .metric("rebuilds", metrics.rebuilds as f64)
            .metric("reuse_hits", metrics.reuse_hits as f64)
            .maximum("max_solution_difference", 1.0e-10)
            .input_checksum("analytic/deterministic-fixture:bem-operator-cache-reuse-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-operator-cache-reuse-v1")
            .dimension("3-D rigid sphere; dense direct constant-panel TBEM; two incident fields")
            .seed("bem-operator-cache-reuse:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(_) => QaResult::failed(CASE_ID, "panic while running operator-cache contract"),
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
