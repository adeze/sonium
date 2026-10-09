//! Checked non-homogeneous complex Dirichlet elimination.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{BoundaryConditionError, FemSystem, try_apply_dirichlet_values_complex};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use std::collections::HashSet;

fn dense_complex_system(size: usize) -> FemSystem<Complex64> {
    let mut matrix = TripletMatrix::new(size, size);
    for row in 0..size {
        for column in 0..size {
            matrix.push(row, column, Complex64::new(1.0, 0.0));
        }
    }
    FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); size])
}

#[test]
fn complex_dirichlet_elimination_corrects_rhs_and_preserves_p2_dofs() {
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
        .expect("refined square should have an interior P2 DOF");
    let prescribed = Complex64::new(3.0, -2.0);
    let mut system = dense_complex_system(size);

    try_apply_dirichlet_values_complex(&mut system, &mesh, &space, 17, &|_, _, _| prescribed, None)
        .expect("complex Dirichlet values should be admitted");

    assert_eq!(system.dirichlet_dofs.len(), boundary_set.len());
    for &dof in &boundary_dofs {
        assert_eq!(system.rhs[dof], prescribed);
    }
    let expected_interior_rhs = -prescribed * boundary_dofs.len() as f64;
    assert_eq!(system.rhs[interior], expected_interior_rhs);
    assert!(
        system
            .matrix
            .rows
            .iter()
            .zip(&system.matrix.cols)
            .all(|(&row, &col)| {
                !boundary_set.contains(&row) && !boundary_set.contains(&col)
                    || (row == col && boundary_set.contains(&row))
            })
    );
    assert!(boundary_dofs.iter().any(|&dof| dof >= mesh.num_nodes()));
}

#[test]
fn complex_dirichlet_rejects_non_finite_values_transactionally() {
    let mut mesh = unit_square_triangles(1);
    mesh.detect_boundaries();
    let space = H1Space::new(PolynomialDegree::P1);
    let size = space.num_global_dofs(&mesh);
    let mut system = dense_complex_system(size);
    let original_rhs = system.rhs.clone();
    let original_rows = system.matrix.rows.clone();
    let original_cols = system.matrix.cols.clone();
    let original_values = system.matrix.values.clone();

    let error = try_apply_dirichlet_values_complex(
        &mut system,
        &mesh,
        &space,
        0,
        &|_, _, _| Complex64::new(f64::NAN, 0.0),
        None,
    )
    .expect_err("non-finite complex boundary values must fail closed");
    assert!(matches!(
        error,
        BoundaryConditionError::NonFiniteValue { .. }
    ));
    assert_eq!(system.rhs, original_rhs);
    assert_eq!(system.matrix.rows, original_rows);
    assert_eq!(system.matrix.cols, original_cols);
    assert_eq!(system.matrix.values, original_values);
    assert!(system.dirichlet_dofs.is_empty());
}
