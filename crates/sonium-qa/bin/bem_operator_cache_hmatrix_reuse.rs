//! CPU QA for persistent H-matrix operator reuse across incident loads.

use num_complex::Complex64;
use sonium_bem::BemReusePolicy;
use sonium_bem::core::{
    AssemblyMethod, BemOperatorCache, BemProblem, BemSolver, IncidentField, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.operator-cache-hmatrix-reuse.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    maximum_solution_difference: f64,
    rebuilds: u64,
    reuse_hits: u64,
}

fn run() -> Result<Metrics, String> {
    let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0);
    let mut cache = BemOperatorCache::new();
    solver
        .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
        .map_err(|error| format!("initial H-matrix cached solve: {error}"))?;
    let changed_load = base.with_incident_field(IncidentField::plane_wave_neg_z());
    let cached = solver
        .solve_with_cache(&changed_load, &mut cache, BemReusePolicy::default())
        .map_err(|error| format!("reused H-matrix solve: {error}"))?;
    let independent = solver
        .solve(&changed_load)
        .map_err(|error| format!("independent H-matrix solve: {error}"))?;
    let maximum_solution_difference = cached
        .surface_pressure
        .iter()
        .zip(&independent.surface_pressure)
        .map(|(cached, independent)| (*cached - *independent).norm())
        .fold(0.0_f64, f64::max);
    if !cached
        .surface_pressure
        .iter()
        .chain(&independent.surface_pressure)
        .all(|value: &Complex64| value.re.is_finite() && value.im.is_finite())
    {
        return Err("H-matrix cache produced non-finite coefficients".into());
    }
    if maximum_solution_difference > 1.0e-10 {
        return Err(format!(
            "cached H-matrix solution differs by {maximum_solution_difference:e}"
        ));
    }
    if cache.rebuilds() != 1 || cache.reuse_hits() != 1 {
        return Err(format!(
            "unexpected cache telemetry: rebuilds={}, reuse_hits={}",
            cache.rebuilds(),
            cache.reuse_hits()
        ));
    }
    Ok(Metrics {
        dimension: cached.surface_pressure.len(),
        maximum_solution_difference,
        rebuilds: cache.rebuilds(),
        reuse_hits: cache.reuse_hits(),
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric(
                "maximum_solution_difference",
                metrics.maximum_solution_difference,
            )
            .metric("rebuilds", metrics.rebuilds as f64)
            .metric("reuse_hits", metrics.reuse_hits as f64)
            .minimum("dimension", 1.0)
            .maximum("maximum_solution_difference", 1.0e-10)
            .minimum("rebuilds", 1.0)
            .minimum("reuse_hits", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-operator-cache-hmatrix-reuse-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-operator-cache-hmatrix-reuse-v1",
            )
            .dimension("3-D rigid sphere; constant-panel H-matrix; changed incident load")
            .formulation("caller-owned persistent H-matrix operator cache")
            .tolerance("cached coefficients match independent solve <= 1e-10")
            .seed("bem-operator-cache-hmatrix-reuse:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic while running H-matrix cache QA")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
