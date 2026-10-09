//! Release-backed fixed-geometry higher-order BEM objective-gradient contract.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BasisOrder, BemProblem, BemSolver, PhysicsParams, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-objective-gradients.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    analytic_gradient: f64,
    finite_difference_gradient: f64,
    relative_error: f64,
    analytic_speed_gradient: f64,
    finite_difference_speed_gradient: f64,
    speed_relative_error: f64,
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

fn at_frequency(problem: &BemProblem, frequency: f64) -> Result<BemProblem, String> {
    let mut candidate = problem.clone();
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

fn at_speed(problem: &BemProblem, speed_of_sound: f64) -> Result<BemProblem, String> {
    let mut candidate = problem.clone();
    let mut physics = PhysicsParams::try_new(
        problem.physics.frequency,
        speed_of_sound,
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

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Linear)
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
        .solve_wavenumber_objective_gradient(&problem, &forward_rhs, &objective_gradient)
        .map_err(|error| error.to_string())?;

    let delta_k = 1.0e-4;
    let delta_frequency = delta_k * problem.physics.speed_of_sound / (2.0 * PI);
    let plus = at_frequency(&problem, problem.physics.frequency + delta_frequency)?;
    let minus = at_frequency(&problem, problem.physics.frequency - delta_frequency)?;
    let finite_difference_gradient =
        (objective(&solver, &plus, &forward_rhs, &objective_gradient)?
            - objective(&solver, &minus, &forward_rhs, &objective_gradient)?)
            / (2.0 * delta_k);
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
            "wavenumber objective gradient mismatch: analytic={analytic_gradient:e}, finite_difference={finite_difference_gradient:e}, relative_error={relative_error:e}"
        ));
    }
    let analytic_speed_gradient = solver
        .solve_speed_of_sound_objective_gradient(&problem, &forward_rhs, &objective_gradient)
        .map_err(|error| error.to_string())?;
    let delta_speed = 1.0e-2;
    let plus_speed = at_speed(&problem, problem.physics.speed_of_sound + delta_speed)?;
    let minus_speed = at_speed(&problem, problem.physics.speed_of_sound - delta_speed)?;
    let finite_difference_speed_gradient =
        (objective(&solver, &plus_speed, &forward_rhs, &objective_gradient)?
            - objective(&solver, &minus_speed, &forward_rhs, &objective_gradient)?)
            / (2.0 * delta_speed);
    let speed_scale = analytic_speed_gradient
        .abs()
        .max(finite_difference_speed_gradient.abs())
        .max(1.0);
    let speed_relative_error =
        (analytic_speed_gradient - finite_difference_speed_gradient).abs() / speed_scale;
    if !analytic_speed_gradient.is_finite()
        || !finite_difference_speed_gradient.is_finite()
        || !speed_relative_error.is_finite()
        || speed_relative_error > 2.0e-5
    {
        return Err(format!(
            "sound-speed objective gradient mismatch: analytic={analytic_speed_gradient:e}, finite_difference={finite_difference_speed_gradient:e}, relative_error={speed_relative_error:e}"
        ));
    }
    Ok(Metrics {
        dimension,
        analytic_gradient,
        finite_difference_gradient,
        relative_error,
        analytic_speed_gradient,
        finite_difference_speed_gradient,
        speed_relative_error,
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
            .metric("analytic_speed_gradient", metrics.analytic_speed_gradient)
            .metric(
                "finite_difference_speed_gradient",
                metrics.finite_difference_speed_gradient,
            )
            .metric("speed_relative_error", metrics.speed_relative_error)
            .maximum("relative_error", 2.0e-5)
            .maximum("speed_relative_error", 2.0e-5)
                .input_checksum(
                    "analytic/deterministic-fixture:bem-higher-order-objective-gradients-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-higher-order-objective-gradients-v1",
                )
                .dimension(
                    "3-D rigid sphere; dense direct linear-basis TBEM fixed-geometry d/dk and d/dc objectives",
                )
                .seed("bem-higher-order-objective-gradients:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(payload) => QaResult::failed(CASE_ID, format!("panic: {payload:?}"))
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA_RESULT serialization")
    );
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}
