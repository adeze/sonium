//! Release-backed compressed global rigid-translation objective derivative.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-geometry-translation-objective-gradient.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    backend_count: usize,
    dimension: usize,
    maximum_relative_error: f64,
    maximum_absolute_error: f64,
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

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let dimension = problem.mesh.elements.len();
    let objective_gradient = (0..dimension)
        .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let delta = 1.0e-5;
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_absolute_error = 0.0_f64;
    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let analytic = solver
            .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
            .map_err(|error| format!("{assembly_method:?}: {error}"))?;
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
            if !relative_error.is_finite() || relative_error > 5.0e-4 {
                return Err(format!(
                    "{assembly_method:?} translation axis {axis}: analytic={:e}, finite_difference={finite_difference:e}, relative_error={relative_error:e}",
                    analytic[axis]
                ));
            }
            maximum_absolute_error = maximum_absolute_error.max(absolute_error);
            maximum_relative_error = maximum_relative_error.max(relative_error);
        }
    }
    Ok(Metrics {
        backend_count: 3,
        dimension,
        maximum_relative_error,
        maximum_absolute_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("backend_count", metrics.backend_count as f64)
            .metric("dimension", metrics.dimension as f64)
            .metric("maximum_relative_error", metrics.maximum_relative_error)
            .metric("maximum_absolute_error", metrics.maximum_absolute_error)
            .minimum("backend_count", 3.0)
            .minimum("dimension", 1.0)
            .maximum("maximum_relative_error", 5.0e-4)
            .maximum("maximum_absolute_error", 5.0e-4)
            .input_checksum("analytic/deterministic-fixture:bem-compressed-geometry-translation-objective-gradient-v1")
            .reference("H-matrix, SLFMM, and MLFMM translation-invariant objective gradient versus centered translated-mesh differences")
            .reference_checksum("analytic/deterministic-fixture:bem-compressed-geometry-translation-objective-gradient-v1")
            .dimension("3-D rigid sphere; constant-panel 48-DOF; three compressed backends")
            .formulation("production solve_geometry_translation_objective_gradient compressed dispatch")
            .tolerance("per-axis relative and absolute error <= 5e-4")
            .seed("bem-compressed-geometry-translation-objective-gradient:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic while running compressed translation gradient")
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
