//! Release-facing compressed impedance material-gradient contract.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryConditionType, PhysicsParams, SolverMethod,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-impedance-material-objective-gradient.v1";
const SPEED_STEP: f64 = 1.0e-3;
const DENSITY_STEP: f64 = 1.0e-5;

fn objective(solver: &BemSolver, problem: &BemProblem, rhs: &[Complex64]) -> Result<f64, String> {
    let solution = solver
        .solve_rhs_many(problem, &[rhs.to_vec()])
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "objective solve returned no solution".to_string())?;
    Ok(rhs
        .iter()
        .zip(solution.coefficients())
        .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
        .sum())
}

fn at_medium(problem: &BemProblem, speed: f64, density: f64) -> Result<BemProblem, String> {
    let mut candidate = problem.clone();
    let mut physics = PhysicsParams::try_new(
        problem.physics.frequency,
        speed,
        density,
        problem.physics.tau < 0.0,
    )
    .map_err(|error| error.to_string())?;
    physics.harmonic_factor = problem.physics.harmonic_factor;
    physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
    candidate.physics = physics;
    Ok(candidate)
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(1.0)
}

fn run() -> Result<(usize, f64, f64), String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(100)
        .with_tolerance(1.0e-7);
    let dimension = problem.mesh.elements.len();
    let rhs = vec![Complex64::new(1.0, 0.0); dimension];
    let speed_gradient = solver
        .solve_speed_of_sound_objective_gradient(&problem, &rhs, &rhs)
        .map_err(|error| error.to_string())?;
    let density_gradient = solver
        .solve_density_objective_gradient(&problem, &rhs, &rhs)
        .map_err(|error| error.to_string())?;

    let speed_fd = (objective(
        &solver,
        &at_medium(
            &problem,
            problem.physics.speed_of_sound + SPEED_STEP,
            problem.physics.density,
        )?,
        &rhs,
    )? - objective(
        &solver,
        &at_medium(
            &problem,
            problem.physics.speed_of_sound - SPEED_STEP,
            problem.physics.density,
        )?,
        &rhs,
    )?) / (2.0 * SPEED_STEP);
    let density_fd = (objective(
        &solver,
        &at_medium(
            &problem,
            problem.physics.speed_of_sound,
            problem.physics.density + DENSITY_STEP,
        )?,
        &rhs,
    )? - objective(
        &solver,
        &at_medium(
            &problem,
            problem.physics.speed_of_sound,
            problem.physics.density - DENSITY_STEP,
        )?,
        &rhs,
    )?) / (2.0 * DENSITY_STEP);
    let speed_error = relative_error(speed_gradient, speed_fd);
    let density_error = relative_error(density_gradient, density_fd);
    if !speed_error.is_finite() || !density_error.is_finite() {
        return Err("non-finite compressed impedance gradient metric".into());
    }
    Ok((dimension, speed_error, density_error))
}

fn main() {
    let started = Instant::now();
    let result = match std::panic::catch_unwind(run) {
        Ok(Ok((dimension, speed_error, density_error))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dimension", dimension as f64)
            .metric("speed_relative_error", speed_error)
            .metric("density_relative_error", density_error)
            .minimum("dimension", 1.0)
            .maximum("speed_relative_error", 3.0e-4)
            .maximum("density_relative_error", 3.0e-4)
            .input_checksum(
                "analytic/deterministic-fixture:bem-compressed-impedance-material-objective-gradient-v1",
            )
            .reference("independent centered H-matrix solves with synchronized Y=1/(rho*c)")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-compressed-impedance-material-objective-gradient-v1",
            )
            .dimension("3-D constant-panel impedance H-matrix objective bracket")
            .formulation("CPU compressed H-matrix GMRES with impedance admittance terms")
            .tolerance("speed and density relative errors <= 3e-4")
            .seed("bem-compressed-impedance-material-objective-gradient:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(_) => QaResult::failed(CASE_ID, "panic during QA execution")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
