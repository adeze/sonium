//! Checked vector cross assembly on explicit curved P2 geometry.

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

fn curved_mesh() -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
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
fn curved_hcurl_cross_mass_is_finite_and_transpose_consistent() {
    let mesh = curved_mesh();
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let forward =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order1, &order2, &mesh, 10)
            .expect("curved H(curl) spaces should assemble");
    let reverse =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order2, &order1, &mesh, 10)
            .expect("swapped curved H(curl) spaces should assemble");

    assert_eq!(forward.num_rows, order1.num_global_dofs(&mesh));
    assert_eq!(forward.num_cols, order2.num_global_dofs(&mesh));
    let forward = entries(&forward);
    let reverse = entries(&reverse);
    let keys: BTreeSet<_> = forward
        .keys()
        .copied()
        .chain(reverse.keys().map(|&(row, column)| (column, row)))
        .collect();
    let mut max_error: f64 = 0.0;
    let mut nonzero = 0;
    for (row, column) in keys {
        let left = forward.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse.get(&(column, row)).copied().unwrap_or(0.0);
        assert!(left.is_finite() && right.is_finite());
        max_error = max_error.max((left - right).abs());
        nonzero += usize::from(left.abs() > 1.0e-14);
    }
    assert!(max_error <= 2.0e-10);
    assert!(nonzero > order1.num_global_dofs(&mesh));
}

#[test]
fn curved_hcurl_cross_mass_rejects_malformed_topology() {
    let mut mesh = curved_mesh();
    mesh.elements[0].nodes.pop();
    let error = assemble_bilinear_two_spaces_checked(
        &VectorMassForm,
        &HcurlSpace::new(),
        &HcurlSpace::order2(),
        &mesh,
        6,
    )
    .expect_err("malformed curved H(curl) topology must be rejected");
    assert!(error.to_string().contains("nodes") || error.to_string().contains("element"));
}
