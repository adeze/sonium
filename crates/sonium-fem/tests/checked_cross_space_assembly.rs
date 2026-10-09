//! Checked generic assembly for distinct test and trial spaces.

use sonium_fem::assembly::assemble_bilinear_two_spaces_checked;
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{BasisTabulation, DGSpace, FunctionSpace, H1Space};

#[derive(Debug, Clone, Copy)]
struct CrossMassForm;

impl BilinearForm<f64> for CrossMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * factor);
            }
        }
    }
}

fn triangle_area(mesh: &sonium_fem::mesh::Mesh, element: usize) -> f64 {
    let vertices = mesh.elements[element].vertices();
    let a = &mesh.nodes[vertices[0]];
    let b = &mesh.nodes[vertices[1]];
    let c = &mesh.nodes[vertices[2]];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
}

#[test]
fn checked_cross_space_assembly_preserves_rectangular_block_shape() {
    let mesh = unit_square_triangles(2);
    let test_space = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let trial_space = DGSpace::p0();
    let matrix =
        assemble_bilinear_two_spaces_checked(&CrossMassForm, &test_space, &trial_space, &mesh, 2)
            .expect("valid H1/DG spaces should assemble");

    assert_eq!(matrix.num_rows, test_space.num_global_dofs(&mesh));
    assert_eq!(matrix.num_cols, trial_space.num_global_dofs(&mesh));
    assert_eq!(matrix.nnz(), mesh.num_elements() * 3);

    for element in 0..mesh.num_elements() {
        let area = triangle_area(&mesh, element);
        let column = element;
        let mut column_sum = 0.0;
        for ((&row, &col), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
            if col == column {
                assert!(value > 0.0);
                assert!((value - area / 3.0).abs() < 1.0e-12);
                assert!(row < matrix.num_rows);
                column_sum += value;
            }
        }
        assert!((column_sum - area).abs() < 1.0e-12);
    }
}

#[test]
fn checked_cross_space_assembly_rejects_malformed_topology() {
    let mut mesh = unit_square_triangles(1);
    mesh.elements[0].nodes.pop();
    let test_space = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let trial_space = DGSpace::p0();

    let result =
        assemble_bilinear_two_spaces_checked(&CrossMassForm, &test_space, &trial_space, &mesh, 2);
    assert!(result.is_err());
}
