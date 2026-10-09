//! Regression coverage for topology-owned P3 triangle and quadrilateral DOFs.

use sonium_fem::assembly::{assemble_mass, assemble_stiffness};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{unit_square_quads, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn dense_triplets(rows: &[usize], cols: &[usize], values: &[f64], dim: usize) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; dim]; dim];
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(values) {
        dense[row][col] += value;
    }
    dense
}

fn assert_p3_assembly(mesh: &sonium_fem::mesh::Mesh) {
    let space = H1Space::new(PolynomialDegree::P3);
    space.validate(mesh).expect("vertex-only P3 mesh admission");
    let expected_dim = space.num_global_dofs(mesh);

    let stiffness = assemble_stiffness(mesh, PolynomialDegree::P3);
    let mass = assemble_mass(mesh, PolynomialDegree::P3);
    assert_eq!(stiffness.dim, expected_dim);
    assert_eq!(mass.dim, expected_dim);

    let stiffness_dense = dense_triplets(
        &stiffness.rows,
        &stiffness.cols,
        &stiffness.values,
        expected_dim,
    );
    let mass_dense = dense_triplets(&mass.rows, &mass.cols, &mass.values, expected_dim);

    for row in 0..expected_dim {
        let stiffness_row_sum: f64 = stiffness_dense[row].iter().sum();
        assert!(
            stiffness_row_sum.abs() < 2.0e-11,
            "P3 stiffness row {row} does not preserve constants: {stiffness_row_sum:e}"
        );
        for col in 0..expected_dim {
            assert!(
                (stiffness_dense[row][col] - stiffness_dense[col][row]).abs() < 2.0e-11,
                "P3 stiffness is not symmetric at ({row}, {col})"
            );
            assert!(
                (mass_dense[row][col] - mass_dense[col][row]).abs() < 2.0e-11,
                "P3 mass is not symmetric at ({row}, {col})"
            );
            assert!(
                stiffness_dense[row][col].is_finite() && mass_dense[row][col].is_finite(),
                "P3 assembly produced a non-finite entry at ({row}, {col})"
            );
        }
    }

    let mass_sum: f64 = mass_dense.iter().flatten().sum();
    assert!(
        (mass_sum - 1.0).abs() < 2.0e-11,
        "P3 mass does not integrate to area"
    );
}

#[test]
fn p3_triangle_assembly_uses_topology_owned_dofs() {
    assert_p3_assembly(&unit_square_triangles(2));
}

#[test]
fn p3_quadrilateral_assembly_uses_topology_owned_dofs() {
    assert_p3_assembly(&unit_square_quads(2));
}
