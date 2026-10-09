//! Explicit P2 isoparametric scalar H1 transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{transfer_h1_state_checked, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};

#[test]
fn curved_p2_h1_state_reproduces_complete_global_state() {
    let mut mesh = unit_square_triangles(1).to_p2();
    let curved_midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[curved_midpoint].y += 0.08;
    let space = H1Space::new(PolynomialDegree::P2);
    let source_values: Vec<f64> = (0..space.num_global_dofs(&mesh))
        .map(|dof| 0.37 + 0.19 * dof as f64 - 0.013 * (dof * dof) as f64)
        .collect();

    let transferred = transfer_h1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .expect("explicit P2 H1 geometry should be admitted");
    let max_error = transferred
        .iter()
        .zip(&source_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 2.0e-11, "curved P2 transfer error {max_error}");
}
