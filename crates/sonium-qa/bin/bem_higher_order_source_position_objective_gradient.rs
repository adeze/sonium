//! Release-backed dense higher-order BEM point-source position derivative.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BasisOrder, BemProblem, BemSolver, IncidentField, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-source-position-objective-gradient.v1";

struct Metrics {
    dimension: usize,
    max_relative_error: f64,
    max_absolute_error: f64,
}

fn objective(
    solver: &BemSolver,
    problem: &BemProblem,
    objective_gradient: &[Complex64],
) -> Result<f64, String> {
    let solution = solver.solve(problem).map_err(|error| error.to_string())?;
    if solution.surface_pressure.len() != objective_gradient.len() {
        return Err(format!(
            "objective dimension mismatch: gradient={} solution={}",
            objective_gradient.len(),
            solution.surface_pressure.len()
        ));
    }
    Ok(objective_gradient
        .iter()
        .zip(&solution.surface_pressure)
        .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
        .sum())
}

fn run() -> Result<Metrics, String> {
    let position = [0.45, -0.3, 0.4];
    let strength = Complex64::new(0.8, -0.2);
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
        .with_incident_field(IncidentField::PointSource { position, strength });
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_basis_order(BasisOrder::Quadratic);
    let base_solution = solver.solve(&problem).map_err(|error| error.to_string())?;
    let dimension = base_solution.surface_pressure.len();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let analytic = solver
        .solve_source_position_objective_gradient(&problem, &objective_gradient)
        .map_err(|error| error.to_string())?;

    let delta = 1.0e-5;
    let mut max_relative_error = 0.0_f64;
    let mut max_absolute_error = 0.0_f64;
    for axis in 0..3 {
        let mut plus = problem.clone();
        let mut minus = problem.clone();
        if let IncidentField::PointSource { position, .. } = &mut plus.incident_field {
            position[axis] += delta;
        }
        if let IncidentField::PointSource { position, .. } = &mut minus.incident_field {
            position[axis] -= delta;
        }
        let finite_difference = (objective(&solver, &plus, &objective_gradient)?
            - objective(&solver, &minus, &objective_gradient)?)
            / (2.0 * delta);
        let absolute_error = (analytic[axis] - finite_difference).abs();
        let relative_error =
            absolute_error / analytic[axis].abs().max(finite_difference.abs()).max(1.0);
        max_absolute_error = max_absolute_error.max(absolute_error);
        max_relative_error = max_relative_error.max(relative_error);
    }
    if !max_relative_error.is_finite() || !max_absolute_error.is_finite() {
        return Err(
            "higher-order source-position derivative produced non-finite error".to_string(),
        );
    }
    if max_relative_error > 2.0e-4 {
        return Err(format!(
            "higher-order source-position objective gradient mismatch: relative_error={max_relative_error:e}"
        ));
    }
    Ok(Metrics {
        dimension,
        max_relative_error,
        max_absolute_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("max_relative_error", metrics.max_relative_error)
            .metric("max_absolute_error", metrics.max_absolute_error)
            .maximum("max_relative_error", 2.0e-4)
            .input_checksum(
                "analytic/deterministic-fixture:bem-higher-order-source-position-objective-gradient-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-higher-order-source-position-objective-gradient-v1",
            )
            .dimension("3-D rigid sphere; dense direct discontinuous quadratic TBEM point-source d/dposition")
            .seed("bem-higher-order-source-position-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(_) => QaResult::failed(CASE_ID, "panic while running higher-order source-position derivative"),
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
