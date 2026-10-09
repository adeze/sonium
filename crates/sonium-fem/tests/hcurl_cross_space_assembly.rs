//! Checked vector-space cross assembly for distinct H(curl) orders.

use sonium_fem::assembly::assemble_bilinear_two_spaces_checked;
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, HcurlSpace};
use std::collections::{BTreeMap, BTreeSet};

struct VectorMassForm;

impl BilinearForm<f64> for VectorMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = weight * test.det_j;
        let components = test.spatial_dim.min(trial.spatial_dim);
        for row in 0..test.n_dofs {
            for column in 0..trial.n_dofs {
                let dot = (0..components)
                    .map(|component| {
                        test.vector_value(row, component) * trial.vector_value(column, component)
                    })
                    .sum::<f64>();
                local_matrix.add(row, column, factor * dot);
            }
        }
    }
}

fn entries(matrix: &sonium_fem::assembly::TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut result = BTreeMap::new();
    for index in 0..matrix.nnz() {
        *result
            .entry((matrix.rows[index], matrix.cols[index]))
            .or_insert(0.0) += matrix.values[index];
    }
    result
}

#[test]
fn hcurl_order1_to_order2_cross_mass_is_rectangular_and_transpose_consistent() {
    let mesh = unit_square_triangles(2);
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();

    let order1_to_order2 =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order1, &order2, &mesh, 8)
            .expect("valid H(curl) spaces should assemble");
    let order2_to_order1 =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order2, &order1, &mesh, 8)
            .expect("swapped valid H(curl) spaces should assemble");

    assert_eq!(
        (order1_to_order2.num_rows, order1_to_order2.num_cols),
        (order1.num_global_dofs(&mesh), order2.num_global_dofs(&mesh))
    );
    assert_eq!(
        (order2_to_order1.num_rows, order2_to_order1.num_cols),
        (order2.num_global_dofs(&mesh), order1.num_global_dofs(&mesh))
    );

    let forward = entries(&order1_to_order2);
    let reverse = entries(&order2_to_order1);
    let keys: BTreeSet<_> = forward
        .keys()
        .copied()
        .chain(reverse.keys().map(|&(row, column)| (column, row)))
        .collect();
    let mut max_transpose_error: f64 = 0.0;
    let mut nonzero_entries = 0;
    for (row, column) in keys {
        let left = forward.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse.get(&(column, row)).copied().unwrap_or(0.0);
        max_transpose_error = max_transpose_error.max((left - right).abs());
        if left.abs() > 1.0e-14 {
            nonzero_entries += 1;
        }
        assert!(left.is_finite() && right.is_finite());
    }
    assert!(max_transpose_error <= 2.0e-12);
    assert!(nonzero_entries > order1.num_global_dofs(&mesh));
}

#[test]
fn hcurl_cross_space_assembly_rejects_malformed_topology() {
    let mut mesh = unit_square_triangles(1);
    mesh.elements[0].nodes.pop();
    let error = assemble_bilinear_two_spaces_checked(
        &VectorMassForm,
        &HcurlSpace::new(),
        &HcurlSpace::order2(),
        &mesh,
        4,
    )
    .expect_err("malformed H(curl) topology must fail before assembly");
    assert!(error.to_string().contains("nodes") || error.to_string().contains("element"));
}
