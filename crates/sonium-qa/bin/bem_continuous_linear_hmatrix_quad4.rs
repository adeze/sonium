//! Independent CPU QA for continuous linear Quad4 H-matrix parity.

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

const CASE_ID: &str = "bem.continuous-linear-hmatrix-quad4.v1";
const MAX_RELATIVE_FIELD_ERROR: f64 = 1.0e-4;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dofs: usize,
    hmatrix_field_norm: f64,
    dense_field_norm: f64,
    max_relative_field_error: f64,
}

fn problem() -> BemProblem {
    BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    }
}

fn hmatrix_solver() -> BemSolver {
    BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
}

fn dense_solver() -> BemSolver {
    BemSolver::new()
        .with_continuous_linear()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
}

fn run() -> Result<Metrics, String> {
    let problem = problem();
    problem
        .validate()
        .map_err(|error| format!("continuous linear Quad4 fixture invalid: {error}"))?;

    let hmatrix = hmatrix_solver()
        .solve(&problem)
        .map_err(|error| format!("continuous linear Quad4 H-matrix solve failed: {error}"))?;
    let dense = dense_solver()
        .solve(&problem)
        .map_err(|error| format!("continuous linear Quad4 dense solve failed: {error}"))?;

    if hmatrix.num_dofs() != 8 || dense.num_dofs() != 8 {
        return Err(format!(
            "expected eight shared vertex DOFs, got H-matrix={} dense={}",
            hmatrix.num_dofs(),
            dense.num_dofs()
        ));
    }
    let finite = |values: &[Complex64]| {
        values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    };
    if !finite(&hmatrix.surface_pressure) || !finite(&dense.surface_pressure) {
        return Err("continuous linear Quad4 coefficients are non-finite".to_string());
    }

    let points = [
        [0.0, 0.0, 0.2],
        [0.13, -0.07, 0.23],
        [-0.11, 0.09, 0.19],
        [0.04, 0.12, -0.18],
    ];
    let mut max_relative_field_error = 0.0_f64;
    let mut hmatrix_field_norm = 0.0_f64;
    let mut dense_field_norm = 0.0_f64;
    for point in points {
        let h_field = hmatrix.evaluate_pressure(&point);
        let d_field = dense.evaluate_pressure(&point);
        if !h_field.re.is_finite()
            || !h_field.im.is_finite()
            || !d_field.re.is_finite()
            || !d_field.im.is_finite()
        {
            return Err("continuous linear Quad4 field is non-finite".to_string());
        }
        hmatrix_field_norm = hmatrix_field_norm.max(h_field.norm());
        dense_field_norm = dense_field_norm.max(d_field.norm());
        max_relative_field_error =
            max_relative_field_error.max((h_field - d_field).norm() / d_field.norm().max(1.0e-12));
    }
    if max_relative_field_error > MAX_RELATIVE_FIELD_ERROR {
        return Err(format!(
            "continuous linear Quad4 H-matrix/dense field error {max_relative_field_error:e} exceeds {MAX_RELATIVE_FIELD_ERROR:e}"
        ));
    }
    Ok(Metrics {
        dofs: hmatrix.num_dofs(),
        hmatrix_field_norm,
        dense_field_norm,
        max_relative_field_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dofs", metrics.dofs as f64)
            .metric("hmatrix_field_norm", metrics.hmatrix_field_norm)
            .metric("dense_field_norm", metrics.dense_field_norm)
            .metric("max_relative_field_error", metrics.max_relative_field_error)
            .maximum("max_relative_field_error", MAX_RELATIVE_FIELD_ERROR)
            .dimension("3-D continuous linear Quad4 H-matrix")
            .formulation("shared-vertex proxy expansion ACA with dense TBEM parity")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-continuous-linear-hmatrix-quad4-v1",
            )
            .seed("bem-continuous-linear-hmatrix-quad4:cpu:v2"),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D continuous linear Quad4 H-matrix")
            .seed("bem-continuous-linear-hmatrix-quad4:cpu:v2"),
        Err(_) => QaResult::failed(CASE_ID, "continuous linear Quad4 H-matrix QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D continuous linear Quad4 H-matrix")
            .seed("bem-continuous-linear-hmatrix-quad4:cpu:v2"),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!("{}", result.to_json_line().expect("serialize QA_RESULT"));
}
