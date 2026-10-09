//! CPU BEM analytic pressure-gradient contract.

use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.field-gradient.v1";
const MAX_RELATIVE_ERROR: f64 = 3.0e-3;

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((maximum_relative_error, checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("maximum_relative_error", maximum_relative_error)
            .metric("checks_passed", checks as f64)
            .minimum("checks_passed", 4.0)
            .maximum("maximum_relative_error", MAX_RELATIVE_ERROR)
            .input_checksum("analytic/deterministic-fixture:bem-field-gradient-v1")
            .reference("central difference production total-pressure evaluator")
            .reference_checksum("analytic/deterministic-fixture:bem-field-gradient-v1")
            .dimension("3-D rigid sphere; constant-panel dense TBEM; three spatial axes")
            .formulation("analytic Green-function spatial derivative Kirchhoff-Helmholtz field")
            .tolerance("relative analytic-vs-central-difference error <= 3e-3")
            .seed("bem-field-gradient:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    let result = result.elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

fn run() -> Result<(f64, usize), String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let solution = solver.solve(&problem).map_err(|error| error.to_string())?;

    let point = [0.12, 0.04, 0.22];
    let step = 2.0e-5;
    let analytic = solution
        .evaluate_pressure_gradient_checked(&point)
        .map_err(|error| error.to_string())?;
    let mut maximum_relative_error = 0.0_f64;
    let mut checks = 0;

    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += step;
        minus[axis] -= step;
        let finite_difference =
            (solution.evaluate_pressure(&plus) - solution.evaluate_pressure(&minus)) / (2.0 * step);
        let scale = 2.0
            * analytic[axis]
                .norm()
                .max(finite_difference.norm())
                .max(1.0e-10);
        let relative_error = (analytic[axis] - finite_difference).norm() / scale;
        if !relative_error.is_finite() || relative_error > MAX_RELATIVE_ERROR {
            return Err(format!(
                "axis {axis} analytic/finite-difference relative error {relative_error:.6e} exceeds {MAX_RELATIVE_ERROR:.6e}"
            ));
        }
        maximum_relative_error = maximum_relative_error.max(relative_error);
        checks += 1;
    }

    if solution
        .evaluate_pressure_gradient_field_checked(&[[f64::NAN, 0.0, 0.24]])
        .is_ok()
    {
        return Err("non-finite evaluation point was accepted".to_string());
    }
    checks += 1;

    Ok((maximum_relative_error, checks))
}
