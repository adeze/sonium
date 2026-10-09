//! CPU QA for non-homogeneous complex Dirichlet elimination.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{BoundaryConditionError, FemSystem, try_apply_dirichlet_values_complex};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;
use std::collections::HashSet;

const CASE_ID: &str = "fem.complex-dirichlet-elimination.v1";

fn dense_system(size: usize) -> FemSystem<Complex64> {
    let mut matrix = TripletMatrix::new(size, size);
    for row in 0..size {
        for column in 0..size {
            matrix.push(row, column, Complex64::new(1.0, 0.0));
        }
    }
    FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); size])
}

fn run() -> Result<(usize, f64, f64, usize), String> {
    let mut mesh = unit_square_triangles(2);
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        boundary.marker = 17;
    }
    let space = H1Space::new(PolynomialDegree::P2);
    let size = space.num_global_dofs(&mesh);
    let boundary_dofs = space.boundary_dofs(&mesh, 17);
    let boundary_set: HashSet<_> = boundary_dofs.iter().copied().collect();
    let interior = (0..size)
        .find(|dof| !boundary_set.contains(dof))
        .ok_or("fixture has no unconstrained P2 DOF")?;
    let prescribed = Complex64::new(3.0, -2.0);
    let mut system = dense_system(size);
    try_apply_dirichlet_values_complex(&mut system, &mesh, &space, 17, &|_, _, _| prescribed, None)
        .map_err(|error| error.to_string())?;

    let boundary_error = boundary_dofs
        .iter()
        .map(|&dof| (system.rhs[dof] - prescribed).norm())
        .fold(0.0, f64::max);
    let interior_error = (system.rhs[interior] + prescribed * boundary_dofs.len() as f64).norm();
    if system.dirichlet_dofs.len() != boundary_set.len() {
        return Err("complex Dirichlet DOF set was not recorded exactly".into());
    }
    if boundary_error > 1.0e-12 || interior_error > 1.0e-12 {
        return Err(format!(
            "complex elimination errors are too large: boundary={boundary_error}, interior={interior_error}"
        ));
    }

    let mut rejected = dense_system(size);
    let original_rhs = rejected.rhs.clone();
    let original_nnz = rejected.matrix.nnz();
    let error = try_apply_dirichlet_values_complex(
        &mut rejected,
        &mesh,
        &space,
        17,
        &|_, _, _| Complex64::new(f64::NAN, 0.0),
        None,
    )
    .expect_err("non-finite complex values must fail closed");
    if !matches!(error, BoundaryConditionError::NonFiniteValue { .. })
        || rejected.rhs != original_rhs
        || rejected.matrix.nnz() != original_nnz
    {
        return Err("non-finite rejection was not transactional".into());
    }

    Ok((boundary_dofs.len(), boundary_error, interior_error, 2))
}

fn main() {
    let result = match run() {
        Ok((boundary_dofs, boundary_error, interior_error, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("boundary_dofs", boundary_dofs as f64)
                .minimum("boundary_dofs", 1.0)
                .metric("boundary_value_error", boundary_error)
                .maximum("boundary_value_error", 1.0e-12)
                .metric("interior_rhs_error", interior_error)
                .maximum("interior_rhs_error", 1.0e-12)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 2.0)
                .reference("dense complex system with independent prescribed-value RHS correction")
                .reference_checksum(
                    "analytic/deterministic-fixture:fem-complex-dirichlet-elimination-v1",
                )
                .dimension("2-D P2 scalar H1 complex system")
                .formulation(
                    "transactional non-homogeneous complex Dirichlet row/column elimination",
                )
                .input_checksum("unit-square-triangles-2:p2-boundary-marker-17:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
