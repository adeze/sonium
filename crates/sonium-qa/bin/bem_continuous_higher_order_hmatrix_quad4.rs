//! Bounded CPU evidence for the continuous quadratic Quad4 H-matrix path.
//!
//! This is a manifest-backed source/type-check packet.  It deliberately uses
//! the small deterministic cubed-sphere fixture and must not be treated as a
//! simulator or large-scale performance workload.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemPreconditioner, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.continuous-higher-order-hmatrix-quad4.v1";

fn run() -> Result<(usize, f64), String> {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("continuous Quad4 H-matrix solve failed: {error}"))?;
    if solution.num_dofs() == 0 {
        return Err("continuous Quad4 H-matrix returned zero DOFs".to_string());
    }
    if !solution
        .surface_pressure
        .iter()
        .all(|value: &Complex64| value.re.is_finite() && value.im.is_finite())
    {
        return Err("continuous Quad4 H-matrix returned non-finite coefficients".to_string());
    }
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    if !field.re.is_finite() || !field.im.is_finite() {
        return Err("continuous Quad4 H-matrix returned non-finite field".to_string());
    }
    Ok((solution.num_dofs(), field.norm()))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((dofs, field_norm))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dofs", dofs as f64)
            .metric("field_norm", field_norm)
            .dimension("3-D continuous quadratic Quad4 H-matrix")
            .formulation("shared-DOF proxy expansion with ACA far blocks")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-continuous-higher-order-hmatrix-quad4-v1",
            )
            .seed("bem-continuous-higher-order-hmatrix-quad4:cpu:v1"),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D continuous quadratic Quad4 H-matrix")
            .seed("bem-continuous-higher-order-hmatrix-quad4:cpu:v1"),
        Err(_) => QaResult::failed(CASE_ID, "continuous Quad4 H-matrix QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D continuous quadratic Quad4 H-matrix")
            .seed("bem-continuous-higher-order-hmatrix-quad4:cpu:v1"),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!("{}", result.to_json_line().expect("serialize QA result"));
}
