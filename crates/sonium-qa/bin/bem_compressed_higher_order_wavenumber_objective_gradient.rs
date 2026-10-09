//! Release-backed quadratic compressed BEM wavenumber objective gradient.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BasisContinuity, BasisOrder, BemProblem, BemSolver, PhysicsParams, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-higher-order-wavenumber-objective-gradient.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    backend_count: usize,
    max_relative_error: f64,
}

fn objective(
    solver: &BemSolver,
    problem: &BemProblem,
    rhs: &[Complex64],
    objective_gradient: &[Complex64],
) -> Result<f64, String> {
    let solution = solver
        .solve_rhs_many(problem, &[rhs.to_vec()])
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

fn at_wavenumber(problem: &BemProblem, wavenumber: f64) -> Result<BemProblem, String> {
    if !wavenumber.is_finite() || wavenumber <= 0.0 {
        return Err(format!("invalid perturbed wavenumber {wavenumber}"));
    }
    let mut candidate = problem.clone();
    let frequency = wavenumber * problem.physics.speed_of_sound / (2.0 * PI);
    let mut physics = PhysicsParams::try_new(
        frequency,
        problem.physics.speed_of_sound,
        problem.physics.density,
        problem.physics.tau < 0.0,
    )
    .map_err(|error| error.to_string())?;
    physics.harmonic_factor = problem.physics.harmonic_factor;
    physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
    physics.validate().map_err(|error| error.to_string())?;
    candidate.physics = physics;
    Ok(candidate)
}

fn backend_error(
    assembly: AssemblyMethod,
    problem: &BemProblem,
    rhs: &[Complex64],
    objective_gradient: &[Complex64],
) -> Result<f64, String> {
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(BasisContinuity::Discontinuous)
        .with_assembly_method(assembly)
        .with_solver_method(SolverMethod::Gmres);
    let analytic = solver
        .solve_wavenumber_objective_gradient(problem, rhs, objective_gradient)
        .map_err(|error| format!("{assembly:?} higher-order wavenumber gradient: {error}"))?;
    let step = 1.0e-4;
    let finite_difference = (objective(
        &solver,
        &at_wavenumber(problem, problem.physics.wave_number + step)?,
        rhs,
        objective_gradient,
    )? - objective(
        &solver,
        &at_wavenumber(problem, problem.physics.wave_number - step)?,
        rhs,
        objective_gradient,
    )?) / (2.0 * step);
    let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
    let relative_error = (analytic - finite_difference).abs() / scale;
    if !relative_error.is_finite() {
        return Err(format!("{assembly:?} produced non-finite gradient error"));
    }
    Ok(relative_error)
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
    let rhs = (0..dimension)
        .map(|index| Complex64::new(0.7 + index as f64 / 13.0, -0.2 + index as f64 / 17.0))
        .collect::<Vec<_>>();

    let mut max_relative_error = 0.0_f64;
    for assembly in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ] {
        max_relative_error = max_relative_error.max(backend_error(assembly, &problem, &rhs, &rhs)?);
    }
    Ok(Metrics {
        dimension,
        backend_count: 4,
        max_relative_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("backend_count", metrics.backend_count as f64)
            .metric("max_relative_error", metrics.max_relative_error)
            .maximum("max_relative_error", 5.0e-2)
            .input_checksum("analytic/deterministic-fixture:bem-compressed-higher-order-wavenumber-objective-gradient-v1")
            .reference_checksum("analytic/deterministic-fixture:bem-compressed-higher-order-wavenumber-objective-gradient-v1")
            .formulation("quadratic discontinuous H-matrix/SLFMM/MLFMM/directional-MLFMM GMRES")
            .dimension("3-D rigid sphere; quadratic Tri3 compressed wavenumber gradient")
            .seed("bem-compressed-higher-order-wavenumber-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during higher-order wavenumber-gradient validation")
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
