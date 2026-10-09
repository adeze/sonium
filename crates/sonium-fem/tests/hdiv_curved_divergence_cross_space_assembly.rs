//! Checked physical H(div) divergence assembly on curved P2 triangles.

use sonium_fem::assembly::assemble_bilinear_two_spaces_checked;
use sonium_fem::forms::DivergenceDivergenceForm;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use std::collections::{BTreeMap, BTreeSet};

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
fn curved_hdiv_divergence_cross_space_assembly_is_transpose_consistent() {
    let mesh = curved_mesh();
    let rt0 = HdivSpace::new();
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let forward =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt0, &rt1, &mesh, 10)
            .expect("curved RT0/RT1 divergence assembly must succeed");
    let reverse =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt1, &rt0, &mesh, 10)
            .expect("curved RT1/RT0 divergence assembly must succeed");
    assert_eq!(forward.num_rows, rt0.num_global_dofs(&mesh));
    assert_eq!(forward.num_cols, rt1.num_global_dofs(&mesh));
    assert_eq!(reverse.num_rows, rt1.num_global_dofs(&mesh));
    assert_eq!(reverse.num_cols, rt0.num_global_dofs(&mesh));

    let forward = entries(&forward);
    let reverse = entries(&reverse);
    let mut keys: BTreeSet<(usize, usize)> = forward.keys().copied().collect();
    for &(row, column) in reverse.keys() {
        keys.insert((column, row));
    }
    let mut max_error: f64 = 0.0;
    let mut nonzero = 0;
    for (row, column) in keys {
        let left = forward.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse.get(&(column, row)).copied().unwrap_or(0.0);
        assert!(left.is_finite() && right.is_finite());
        max_error = max_error.max((left - right).abs());
        nonzero += usize::from(left.abs() > 1.0e-14 || right.abs() > 1.0e-14);
    }
    assert!(max_error <= 2.0e-10, "curved divergence error {max_error}");
    assert!(nonzero > 0);
}

#[test]
fn curved_hdiv_divergence_cross_space_rejects_malformed_topology() {
    let mut mesh = curved_mesh();
    mesh.elements[0].nodes.pop();
    let error = assemble_bilinear_two_spaces_checked(
        &DivergenceDivergenceForm,
        &HdivSpace::new(),
        &HdivSpace::with_order(HdivOrder::Rt1),
        &mesh,
        6,
    )
    .expect_err("malformed curved H(div) topology must be rejected");
    assert!(error.to_string().contains("nodes") || error.to_string().contains("element"));
}
