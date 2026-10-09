//! Release-backed compressed higher-order global rigid-translation derivative.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BasisOrder, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-higher-order-geometry-translation-objective-gradient.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    maximum_relative_error: f64,
    maximum_absolute_error: f64,
}

fn translated(problem: &BemProblem, translation: [f64; 3]) -> BemProblem {
    let mut candidate = problem.clone();
    for node in &mut candidate.mesh.nodes {
        for axis in 0..3 {
            node[axis] += translation[axis];
        }
    }
    for element in &mut candidate.mesh.elements {
        for axis in 0..3 {
            element.center[axis] += translation[axis];
        }
    }
    candidate
}

fn objective(
    solver: &BemSolver,
    problem: &BemProblem,
    objective_gradient: &[Complex64],
) -> Result<f64, String> {
    let solution = solver.solve(problem).map_err(|error| error.to_string())?;
    if solution.surface_pressure.len() != objective_gradient.len() {
        return Err(format!(
            "solution has {} DOFs; objective has {}",
            solution.surface_pressure.len(),
            objective_gradient.len()
        ));
    }
    Ok(objective_gradient
        .iter()
        .zip(&solution.surface_pressure)
        .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
        .sum())
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_basis_order(BasisOrder::Linear)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0);
    let dimension = problem.mesh.elements.len() * 3;
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let analytic = solver
        .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
        .map_err(|error| error.to_string())?;
    let delta = 1.0e-5;
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f64;
    for axis in 0..3 {
        let mut plus = [0.0; 3];
        let mut minus = [0.0; 3];
        plus[axis] += delta;
        minus[axis] -= delta;
        let finite_difference =
            (objective(&solver, &translated(&problem, plus), &objective_gradient)?
                - objective(&solver, &translated(&problem, minus), &objective_gradient)?)
                / (2.0 * delta);
        let absolute_error = (analytic[axis] - finite_difference).abs();
        let relative_error =
            absolute_error / analytic[axis].abs().max(finite_difference.abs()).max(1.0);
        if !relative_error.is_finite() || relative_error > 1.0e-8 {
            return Err(format!(
                "linear H-matrix translation axis {axis}: analytic={:e}, finite_difference={finite_difference:e}, relative_error={relative_error:e}",
                analytic[axis]
            ));
        }
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        maximum_relative_error = maximum_relative_error.max(relative_error);
    }
    Ok(Metrics {
        dimension,
        maximum_relative_error,
        maximum_absolute_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("maximum_relative_error", metrics.maximum_relative_error)
            .metric("maximum_absolute_error", metrics.maximum_absolute_error)
            .minimum("dimension", 1.0)
            .maximum("maximum_relative_error", 1.0e-8)
            .maximum("maximum_absolute_error", 1.0e-8)
            .input_checksum(
                "analytic/deterministic-fixture:bem-compressed-higher-order-geometry-translation-objective-gradient-v1",
            )
            .reference(
                "linear H-matrix production solves versus centered translated-mesh finite differences",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-compressed-higher-order-geometry-translation-objective-gradient-v1",
            )
            .dimension("3-D rigid sphere; linear discontinuous 144-DOF H-matrix")
            .formulation("production solve_geometry_translation_objective_gradient higher-order compressed dispatch")
            .tolerance("per-axis relative and absolute error <= 1e-8")
            .seed("bem-compressed-higher-order-geometry-translation-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while running compressed higher-order translation gradient",
        )
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
