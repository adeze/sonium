//! Release-backed fixed-geometry BEM medium-density objective-gradient contract.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, PhysicsParams, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.density-objective-gradient.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    analytic_gradient: f64,
    finite_difference_gradient: f64,
    relative_error: f64,
}

fn objective(
    solver: &BemSolver,
    problem: &BemProblem,
    forward_rhs: &[Complex64],
    objective_gradient: &[Complex64],
) -> Result<f64, String> {
    let solution = solver
        .solve_rhs_many(problem, &[forward_rhs.to_vec()])
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "objective solve returned no solution".to_string())?;
    Ok(objective_gradient
        .iter()
        .zip(solution.coefficients())
        .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
        .sum())
}

fn at_density(problem: &BemProblem, density: f64) -> Result<BemProblem, String> {
    let mut candidate = problem.clone();
    let mut physics = PhysicsParams::try_new(
        problem.physics.frequency,
        problem.physics.speed_of_sound,
        density,
        problem.physics.tau < 0.0,
    )
    .map_err(|error| error.to_string())?;
    physics.harmonic_factor = problem.physics.harmonic_factor;
    physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
    physics.validate().map_err(|error| error.to_string())?;
    candidate.physics = physics;
    Ok(candidate)
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let dimension = solver
        .solve(&problem)
        .map_err(|error| error.to_string())?
        .num_dofs();
    let forward_rhs = (0..dimension)
        .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
        .collect::<Vec<_>>();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let analytic_gradient = solver
        .solve_density_objective_gradient(&problem, &forward_rhs, &objective_gradient)
        .map_err(|error| error.to_string())?;
    let delta = 1.0e-4;
    let plus = at_density(&problem, problem.physics.density + delta)?;
    let minus = at_density(&problem, problem.physics.density - delta)?;
    let finite_difference_gradient =
        (objective(&solver, &plus, &forward_rhs, &objective_gradient)?
            - objective(&solver, &minus, &forward_rhs, &objective_gradient)?)
            / (2.0 * delta);
    let scale = analytic_gradient
        .abs()
        .max(finite_difference_gradient.abs())
        .max(1.0);
    let relative_error = (analytic_gradient - finite_difference_gradient).abs() / scale;
    if !analytic_gradient.is_finite()
        || !finite_difference_gradient.is_finite()
        || !relative_error.is_finite()
        || relative_error > 2.0e-5
    {
        return Err(format!(
            "density objective gradient mismatch: analytic={analytic_gradient:e}, finite_difference={finite_difference_gradient:e}, relative_error={relative_error:e}"
        ));
    }
    Ok(Metrics {
        dimension,
        analytic_gradient,
        finite_difference_gradient,
        relative_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("analytic_gradient", metrics.analytic_gradient)
            .metric(
                "finite_difference_gradient",
                metrics.finite_difference_gradient,
            )
            .metric("relative_error", metrics.relative_error)
            .maximum("relative_error", 2.0e-5)
            .input_checksum("analytic/deterministic-fixture:bem-density-objective-gradient-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-density-objective-gradient-v1")
            .dimension("3-D constant-panel TBEM fixed-geometry density")
            .reference("centered finite-difference objective evaluation")
            .seed("bem-density-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during density-gradient validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
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
