//! Release-backed fixed-geometry BEM point-source position derivative.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, IncidentField, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.source-position-objective-gradient.v1";

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
        .with_solver_method(SolverMethod::Direct);
    let dimension = problem.mesh.elements.len();
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
        let mut plus = position;
        plus[axis] += delta;
        let mut minus = position;
        minus[axis] -= delta;
        let plus_problem = problem
            .clone()
            .with_incident_field(IncidentField::PointSource {
                position: plus,
                strength,
            });
        let minus_problem = problem
            .clone()
            .with_incident_field(IncidentField::PointSource {
                position: minus,
                strength,
            });
        let finite_difference = (objective(&solver, &plus_problem, &objective_gradient)?
            - objective(&solver, &minus_problem, &objective_gradient)?)
            / (2.0 * delta);
        let absolute_error = (analytic[axis] - finite_difference).abs();
        let relative_error =
            absolute_error / analytic[axis].abs().max(finite_difference.abs()).max(1.0);
        max_absolute_error = max_absolute_error.max(absolute_error);
        max_relative_error = max_relative_error.max(relative_error);
    }
    if !max_relative_error.is_finite() || !max_absolute_error.is_finite() {
        return Err("source-position derivative produced non-finite error".to_string());
    }
    if max_relative_error > 2.0e-5 {
        return Err(format!(
            "source-position objective gradient mismatch: relative_error={max_relative_error:e}"
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
            .maximum("max_relative_error", 2.0e-5)
            .input_checksum(
                "analytic/deterministic-fixture:bem-source-position-objective-gradient-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-source-position-objective-gradient-v1",
            )
            .dimension(
                "3-D rigid sphere; dense direct constant-panel TBEM point-source d/dposition",
            )
            .seed("bem-source-position-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(_) => QaResult::failed(CASE_ID, "panic while running source-position derivative"),
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
