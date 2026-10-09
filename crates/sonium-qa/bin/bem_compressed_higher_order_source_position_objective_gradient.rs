//! CPU reference for higher-order compressed point-source position gradients.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::{
    AssemblyMethod, BasisContinuity, BasisOrder, BemProblem, BemSolver, BoundaryConditionType,
    IncidentField, PhysicsParams, SolverMethod,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-higher-order-source-position-objective-gradient.v1";

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
    let source_position = [0.45, -0.3, 0.4];
    let source_strength = Complex64::new(0.8, -0.2);
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::PointSource {
            position: source_position,
            strength: source_strength,
        },
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(BasisContinuity::Continuous)
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres);
    let base_solution = solver.solve(&problem).map_err(|error| error.to_string())?;
    let dimension = base_solution.surface_pressure.len();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let analytic = solver
        .solve_source_position_objective_gradient(&problem, &objective_gradient)
        .map_err(|error| error.to_string())?;
    let mut max_relative_error = 0.0_f64;
    let mut max_absolute_error = 0.0_f64;
    for axis in 0..3 {
        let mut plus_position = source_position;
        plus_position[axis] += 1.0e-5;
        let mut minus_position = source_position;
        minus_position[axis] -= 1.0e-5;
        let plus = problem
            .clone()
            .with_incident_field(IncidentField::PointSource {
                position: plus_position,
                strength: source_strength,
            });
        let minus = problem
            .clone()
            .with_incident_field(IncidentField::PointSource {
                position: minus_position,
                strength: source_strength,
            });
        let finite_difference = (objective(&solver, &plus, &objective_gradient)?
            - objective(&solver, &minus, &objective_gradient)?)
            / (2.0e-5);
        let absolute_error = (analytic[axis] - finite_difference).abs();
        let relative_error =
            absolute_error / analytic[axis].abs().max(finite_difference.abs()).max(1.0);
        max_absolute_error = max_absolute_error.max(absolute_error);
        max_relative_error = max_relative_error.max(relative_error);
    }
    if !max_relative_error.is_finite() || !max_absolute_error.is_finite() {
        return Err("higher-order compressed source-position error is non-finite".into());
    }
    if max_relative_error > 5.0e-3 {
        return Err(format!(
            "higher-order compressed source-position relative error {max_relative_error} exceeds 5e-3"
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
            .maximum("max_relative_error", 5.0e-3)
            .maximum("max_absolute_error", 5.0e-3)
            .reference("independent centered finite-difference objective evaluations")
            .reference_checksum("analytic/deterministic-fixture:bem-compressed-higher-order-source-position-objective-gradient-v1")
            .dimension("3-D continuous-quadratic Quad4 H-matrix point-source gradient")
            .formulation("production compressed solve with source-position finite difference")
            .tolerance("maximum relative and absolute error <= 5e-3")
            .seed("bem-compressed-higher-order-source-position-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("panic: {}", panic_message(payload.as_ref())),
        ),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}
