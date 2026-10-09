//! CPU evidence for discontinuous higher-order H-matrix assembly.
//!
//! This is a bounded deterministic fixture. It exercises basis-aware dense
//! near blocks and DOF-sized ACA far blocks; it does not launch any simulator
//! workload or claim continuous-Galerkin/device/distributed coverage.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_bem::core::types::BasisOrder;
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-hmatrix.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dofs: usize,
    finite_coefficients: bool,
    field_norm: f64,
}

fn run() -> Result<Metrics, String> {
    // 8x16 gives multiple octree leaves while remaining a small CPU fixture.
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 120.0, 343.0, 1.21, 8, 16);
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Linear)
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("basis-aware H-matrix solve failed: {error}"))?;
    let finite_coefficients = solution
        .surface_pressure
        .iter()
        .all(|value: &Complex64| value.re.is_finite() && value.im.is_finite());
    if !finite_coefficients {
        return Err("higher-order H-matrix coefficients are non-finite".into());
    }
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    if !field.re.is_finite() || !field.im.is_finite() {
        return Err("higher-order H-matrix field is non-finite".into());
    }
    Ok(Metrics {
        dofs: solution.num_dofs(),
        finite_coefficients,
        field_norm: field.norm(),
    })
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dofs", metrics.dofs as f64)
            .metric(
                "finite_coefficients",
                f64::from(metrics.finite_coefficients),
            )
            .metric("field_norm", metrics.field_norm)
            .dimension("3-D discontinuous P1 H-matrix")
            .formulation("basis-aware near blocks and DOF-sized ACA far blocks")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-hmatrix-v1")
            .seed("bem-higher-order-hmatrix:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D discontinuous P1 H-matrix")
            .seed("bem-higher-order-hmatrix:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "higher-order H-matrix QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D discontinuous P1 H-matrix")
            .seed("bem-higher-order-hmatrix:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
