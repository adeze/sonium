//! Release-backed quadratic compressed BEM material objective gradients.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BasisContinuity, BasisOrder, BemProblem, BemSolver, PhysicsParams, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-higher-order-material-objective-gradients.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    backend_count: usize,
    max_speed_relative_error: f64,
    max_density_relative_error: f64,
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
    if solution.coefficients().len() != objective_gradient.len() {
        return Err(format!(
            "objective coefficient length {}; expected {}",
            solution.coefficients().len(),
            objective_gradient.len()
        ));
    }
    Ok(objective_gradient
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
    physics.validate().map_err(|error| error.to_string())?;
    candidate.physics = physics;
    Ok(candidate)
}

fn backend_metrics(
    assembly: AssemblyMethod,
    problem: &BemProblem,
    forward_rhs: &[Complex64],
    objective_gradient: &[Complex64],
) -> Result<(f64, f64), String> {
    let mut solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(BasisContinuity::Discontinuous)
        .with_assembly_method(assembly)
        .with_solver_method(SolverMethod::Gmres);
    solver.hmatrix_aca_tolerance = 1.0e-10;
    solver.hmatrix_max_rank = 64;

    let speed_gradient = solver
        .solve_speed_of_sound_objective_gradient(problem, forward_rhs, objective_gradient)
        .map_err(|error| format!("{assembly:?} higher-order speed gradient: {error}"))?;
    let speed_step = problem.physics.speed_of_sound * 1.0e-4;
    let speed_fd = (objective(
        &solver,
        &at_medium(
            problem,
            problem.physics.speed_of_sound + speed_step,
            problem.physics.density,
        )?,
        forward_rhs,
        objective_gradient,
    )? - objective(
        &solver,
        &at_medium(
            problem,
            problem.physics.speed_of_sound - speed_step,
            problem.physics.density,
        )?,
        forward_rhs,
        objective_gradient,
    )?) / (2.0 * speed_step);

    let density_gradient = solver
        .solve_density_objective_gradient(problem, forward_rhs, objective_gradient)
        .map_err(|error| format!("{assembly:?} higher-order density gradient: {error}"))?;
    let density_step = problem.physics.density * 1.0e-4;
    let density_fd = (objective(
        &solver,
        &at_medium(
            problem,
            problem.physics.speed_of_sound,
            problem.physics.density + density_step,
        )?,
        forward_rhs,
        objective_gradient,
    )? - objective(
        &solver,
        &at_medium(
            problem,
            problem.physics.speed_of_sound,
            problem.physics.density - density_step,
        )?,
        forward_rhs,
        objective_gradient,
    )?) / (2.0 * density_step);

    let speed_scale = speed_gradient.abs().max(speed_fd.abs()).max(1.0);
    let density_scale = density_gradient.abs().max(density_fd.abs()).max(1.0);
    let speed_error = (speed_gradient - speed_fd).abs() / speed_scale;
    let density_error = (density_gradient - density_fd).abs() / density_scale;
    if !speed_error.is_finite() || !density_error.is_finite() {
        return Err(format!("{assembly:?} produced non-finite gradient error"));
    }
    Ok((speed_error, density_error))
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4);
    let panel_count = problem
        .mesh
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    let dimension = panel_count * 6;
    let forward_rhs = (0..dimension)
        .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
        .collect::<Vec<_>>();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();

    let mut max_speed_relative_error = 0.0_f64;
    let mut max_density_relative_error = 0.0_f64;
    for assembly in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ] {
        let (speed_error, density_error) =
            backend_metrics(assembly, &problem, &forward_rhs, &objective_gradient)?;
        max_speed_relative_error = max_speed_relative_error.max(speed_error);
        max_density_relative_error = max_density_relative_error.max(density_error);
    }

    Ok(Metrics {
        dimension,
        backend_count: 4,
        max_speed_relative_error,
        max_density_relative_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("backend_count", metrics.backend_count as f64)
            .metric("max_speed_relative_error", metrics.max_speed_relative_error)
            .metric(
                "max_density_relative_error",
                metrics.max_density_relative_error,
            )
            .maximum("max_speed_relative_error", 5.0e-2)
            .maximum("max_density_relative_error", 5.0e-2)
            .input_checksum("analytic/deterministic-fixture:bem-compressed-higher-order-material-objective-gradients-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-compressed-higher-order-material-objective-gradients-v1")
            .formulation("quadratic discontinuous H-matrix/SLFMM/MLFMM/directional-MLFMM GMRES")
            .dimension("3-D rigid sphere; quadratic Tri3 compressed material gradients")
            .seed("bem-compressed-higher-order-material-objective-gradients:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during higher-order material-gradient validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(&result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
