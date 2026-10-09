//! CPU evidence for discontinuous linear/quadratic Quad4 H-matrix solves.
//!
//! This is deliberately a bounded deterministic fixture.  It exercises the
//! public solver dispatch (including mesh preparation and DOF numbering), not
//! just the private H-matrix entry callbacks.  It must not be used as a
//! simulator workload or as evidence for continuous/Galerkin/device support.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemPreconditioner, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::{BasisOrder, PhysicsParams};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-hmatrix-quad4.v1";

#[derive(Debug, Clone, Copy)]
struct CaseMetrics {
    dofs: usize,
    field_norm: f64,
    finite_coefficients: bool,
}

fn solve_case(problem: &BemProblem, basis: BasisOrder) -> Result<CaseMetrics, String> {
    let solver = BemSolver::new()
        .with_basis_order(basis)
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(problem)
        .map_err(|error| format!("Quad4 {basis:?} H-matrix solve failed: {error}"))?;
    let finite_coefficients = solution
        .surface_pressure
        .iter()
        .all(|value: &Complex64| value.re.is_finite() && value.im.is_finite());
    if !finite_coefficients {
        return Err(format!("Quad4 {basis:?} coefficients are non-finite"));
    }
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    if !field.re.is_finite() || !field.im.is_finite() {
        return Err(format!("Quad4 {basis:?} exterior field is non-finite"));
    }
    Ok(CaseMetrics {
        dofs: solution.num_dofs(),
        field_norm: field.norm(),
        finite_coefficients,
    })
}

fn run() -> Result<(CaseMetrics, CaseMetrics), String> {
    // Face resolution one keeps the public solver fixture small and robust;
    // multi-leaf ACA parity is covered by the focused H-matrix unit tests.
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    problem
        .validate()
        .map_err(|error| format!("Quad4 fixture validation failed: {error}"))?;
    let linear = solve_case(&problem, BasisOrder::Linear)?;
    let quadratic = solve_case(&problem, BasisOrder::Quadratic)?;
    Ok((linear, quadratic))
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(Ok((linear, quadratic))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("linear_dofs", linear.dofs as f64)
            .metric("quadratic_dofs", quadratic.dofs as f64)
            .metric("linear_field_norm", linear.field_norm)
            .metric("quadratic_field_norm", quadratic.field_norm)
            .metric(
                "finite_coefficients",
                f64::from(linear.finite_coefficients && quadratic.finite_coefficients),
            )
            .dimension("3-D discontinuous Quad4 linear/quadratic H-matrix")
            .formulation(
                "basis-aware dense near blocks, DOF-sized ACA far blocks, bounded small-system fallback",
            )
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-hmatrix-quad4-v1")
            .seed("bem-higher-order-hmatrix-quad4:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D discontinuous Quad4 linear/quadratic H-matrix")
            .seed("bem-higher-order-hmatrix-quad4:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "Quad4 higher-order H-matrix QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D discontinuous Quad4 linear/quadratic H-matrix")
            .seed("bem-higher-order-hmatrix-quad4:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
}
