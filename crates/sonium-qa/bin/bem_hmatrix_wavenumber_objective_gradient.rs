//! Release-backed compressed BEM wavenumber objective-gradient contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, PhysicsParams, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.hmatrix-wavenumber-objective-gradient.v1";
const BACKENDS: [AssemblyMethod; 4] = [
    AssemblyMethod::Hmatrix,
    AssemblyMethod::Slfmm,
    AssemblyMethod::Mlfmm,
    AssemblyMethod::DirectionalMlfmm,
];

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    backend_count: usize,
    max_analytic_gradient: f64,
    max_finite_difference_gradient: f64,
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
    Ok(objective_gradient
        .iter()
        .zip(solution.coefficients())
        .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
        .sum())
}

fn at_wavenumber(problem: &BemProblem, wavenumber: f64) -> Result<BemProblem, String> {
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

fn backend_metrics(
    assembly: AssemblyMethod,
    problem: &BemProblem,
    rhs: &[Complex64],
    objective_gradient: &[Complex64],
) -> Result<(f64, f64, f64), String> {
    let mut solver = BemSolver::new()
        .with_assembly_method(assembly)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0);
    solver.hmatrix_aca_tolerance = 1.0e-10;
    solver.hmatrix_max_rank = 64;

    let analytic_gradient = solver
        .solve_wavenumber_objective_gradient(problem, rhs, objective_gradient)
        .map_err(|error| format!("{assembly:?} analytic gradient: {error}"))?;
    let delta_k = 1.0e-4;
    let plus = at_wavenumber(problem, problem.physics.wave_number + delta_k)?;
    let minus = at_wavenumber(problem, problem.physics.wave_number - delta_k)?;
    let finite_difference_gradient = (objective(&solver, &plus, rhs, objective_gradient)?
        - objective(&solver, &minus, rhs, objective_gradient)?)
        / (2.0 * delta_k);
    let scale = analytic_gradient
        .abs()
        .max(finite_difference_gradient.abs())
        .max(1.0);
    let relative_error = (analytic_gradient - finite_difference_gradient).abs() / scale;
    if !analytic_gradient.is_finite()
        || !finite_difference_gradient.is_finite()
        || !relative_error.is_finite()
        || relative_error > 5.0e-2
    {
        return Err(format!(
            "{assembly:?} wavenumber gradient mismatch: analytic={analytic_gradient:e}, finite_difference={finite_difference_gradient:e}, relative_error={relative_error:e}"
        ));
    }
    Ok((
        analytic_gradient,
        finite_difference_gradient,
        relative_error,
    ))
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let dimension = BemSolver::new()
        .solve(&problem)
        .map_err(|error| error.to_string())?
        .surface_pressure
        .len();
    let rhs = (0..dimension)
        .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
        .collect::<Vec<_>>();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();

    let mut max_analytic_gradient = 0.0_f64;
    let mut max_finite_difference_gradient = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    for assembly in BACKENDS {
        let (analytic, finite_difference, relative_error) =
            backend_metrics(assembly, &problem, &rhs, &objective_gradient)?;
        max_analytic_gradient = max_analytic_gradient.max(analytic.abs());
        max_finite_difference_gradient =
            max_finite_difference_gradient.max(finite_difference.abs());
        max_relative_error = max_relative_error.max(relative_error);
    }
    Ok(Metrics {
        dimension,
        backend_count: BACKENDS.len(),
        max_analytic_gradient,
        max_finite_difference_gradient,
        max_relative_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("backend_count", metrics.backend_count as f64)
            .metric("max_analytic_gradient", metrics.max_analytic_gradient)
            .metric(
                "max_finite_difference_gradient",
                metrics.max_finite_difference_gradient,
            )
            .metric("max_relative_error", metrics.max_relative_error)
            .minimum("dimension", 1.0)
            .minimum("backend_count", 4.0)
            .maximum("max_relative_error", 5.0e-2)
            .input_checksum("analytic/deterministic-fixture:bem-hmatrix-wavenumber-objective-gradient-v2")
            .reference("compressed H-matrix/SLFMM/MLFMM/directional-MLFMM objective derivative and centered finite difference")
            .reference_checksum("analytic/deterministic-fixture:bem-hmatrix-wavenumber-objective-gradient-v2")
            .dimension("3-D rigid sphere; constant-panel 48-DOF H-matrix/SLFMM/MLFMM/directional-MLFMM GMRES solves")
            .formulation("production solve_wavenumber_objective_gradient compressed backends")
            .tolerance("backend-maximum relative analytic/finite-difference error <= 5e-2")
            .seed("bem-hmatrix-wavenumber-objective-gradient:cpu:v2")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during compressed wavenumber-gradient validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if matches!(&result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
