//! Release-backed persistent cache coverage for indirect soft-Dirichlet BEM.

use num_complex::Complex64;
use sonium_bem::BemReusePolicy;
use sonium_bem::core::{
    AssemblyMethod, BemOperatorCache, BemProblem, BemSolver, BoundaryConditionType, IncidentField,
    SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.operator-cache-soft-dirichlet.v1";

struct Metrics {
    dimension: usize,
    maximum_solution_difference: f64,
    maximum_batch_difference: f64,
    maximum_adjoint_residual: f64,
    rebuilds_after_boundary_change: u64,
    reuse_hits: u64,
}

fn run() -> Result<Metrics, String> {
    let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
        .with_boundary_condition(BoundaryConditionType::Soft);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let mut cache = BemOperatorCache::new();

    solver
        .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
        .map_err(|error| error.to_string())?;
    let changed_load = base
        .clone()
        .with_incident_field(IncidentField::plane_wave_neg_z());
    let cached = solver
        .solve_with_cache(&changed_load, &mut cache, BemReusePolicy::default())
        .map_err(|error| error.to_string())?;
    let independent = solver
        .solve(&changed_load)
        .map_err(|error| error.to_string())?;
    let maximum_solution_difference = cached
        .surface_pressure
        .iter()
        .zip(&independent.surface_pressure)
        .map(|(cached, independent)| (*cached - *independent).norm())
        .fold(0.0_f64, f64::max);
    if maximum_solution_difference > 1.0e-10 {
        return Err(format!(
            "soft cached solution differs from independent solve: {maximum_solution_difference:e}"
        ));
    }
    if cache.rebuilds() != 1 || cache.reuse_hits() != 1 {
        return Err(format!(
            "unexpected soft cache counters: rebuilds={}, reuse_hits={}",
            cache.rebuilds(),
            cache.reuse_hits()
        ));
    }

    let incident_fields = [
        IncidentField::plane_wave_z(),
        IncidentField::plane_wave_neg_z(),
    ];
    let batch = solver
        .solve_many(&base, &incident_fields)
        .map_err(|error| error.to_string())?;
    let mut maximum_batch_difference = 0.0_f64;
    for (incident_field, batched) in incident_fields.iter().zip(batch) {
        let independent_problem = base.clone().with_incident_field(incident_field.clone());
        let independent = solver
            .solve(&independent_problem)
            .map_err(|error| error.to_string())?;
        let difference = batched
            .surface_pressure
            .iter()
            .zip(&independent.surface_pressure)
            .map(|(batched, independent)| (*batched - *independent).norm())
            .fold(0.0_f64, f64::max);
        maximum_batch_difference = maximum_batch_difference.max(difference);
    }
    if maximum_batch_difference > 1.0e-10 {
        return Err(format!(
            "soft batch differs from independent solves: {maximum_batch_difference:e}"
        ));
    }

    let dimension = base.mesh.elements.len();
    let rhs_columns = vec![
        (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 31.0, -0.1))
            .collect::<Vec<_>>(),
        (0..dimension)
            .map(|index| Complex64::new(-0.3, 0.15 + index as f64 / 47.0))
            .collect::<Vec<_>>(),
    ];
    let adjoint = solver
        .solve_adjoint_many(&base, &rhs_columns)
        .map_err(|error| error.to_string())?;
    let maximum_adjoint_residual = adjoint
        .iter()
        .map(|solution| solution.relative_residual())
        .fold(0.0_f64, f64::max);
    if maximum_adjoint_residual > 1.0e-10 {
        return Err(format!(
            "soft adjoint residual too large: {maximum_adjoint_residual:e}"
        ));
    }

    let rigid = changed_load.with_boundary_condition(BoundaryConditionType::Rigid);
    solver
        .solve_with_cache(&rigid, &mut cache, BemReusePolicy::default())
        .map_err(|error| error.to_string())?;
    if cache.rebuilds() != 2 || cache.reuse_hits() != 1 {
        return Err(format!(
            "boundary-type change did not rebuild cache: rebuilds={}, reuse_hits={}",
            cache.rebuilds(),
            cache.reuse_hits()
        ));
    }

    Ok(Metrics {
        dimension,
        maximum_solution_difference,
        maximum_batch_difference,
        maximum_adjoint_residual,
        rebuilds_after_boundary_change: cache.rebuilds(),
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
            .metric("maximum_batch_difference", metrics.maximum_batch_difference)
            .metric("maximum_adjoint_residual", metrics.maximum_adjoint_residual)
            .metric(
                "rebuilds_after_boundary_change",
                metrics.rebuilds_after_boundary_change as f64,
            )
            .metric("reuse_hits", metrics.reuse_hits as f64)
            .maximum("maximum_solution_difference", 1.0e-10)
            .maximum("maximum_batch_difference", 1.0e-10)
            .maximum("maximum_adjoint_residual", 1.0e-10)
            .minimum("rebuilds_after_boundary_change", 2.0)
            .minimum("reuse_hits", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-operator-cache-soft-dirichlet-v1")
            .reference("independent indirect soft-Dirichlet solve and boundary-type invalidation")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-operator-cache-soft-dirichlet-v1",
            )
            .dimension("3-D constant-panel TBEM; indirect soft-Dirichlet cache and BC invalidation")
            .formulation("caller-owned persistent assembled single-layer operator cache")
            .tolerance(
                "cache and batch coefficient differences <= 1e-10; exact rebuild/reuse counters",
            )
            .seed("bem-operator-cache-soft-dirichlet:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "panic during soft-Dirichlet cache validation"),
    };
    let passed = matches!(result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
