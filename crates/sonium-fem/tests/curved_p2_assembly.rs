use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};

#[cfg_attr(test, test)]
pub fn curved_p2_stiffness_preserves_symmetry_conservation_and_energy() {
    let mut mesh = unit_square_triangles(1);
    mesh = mesh.to_p2();

    let element = &mesh.elements[0];
    let midpoint = element.nodes[element.element_type.num_vertices()];
    mesh.nodes[midpoint].x += 0.017;
    mesh.nodes[midpoint].y -= 0.011;

    let space = H1Space::new(PolynomialDegree::P2);
    space.validate(&mesh).unwrap();
    let n = space.num_global_dofs(&mesh);
    let dense = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 6)
        .to_csr()
        .to_dense();

    let mut max_symmetry_error: f64 = 0.0;
    let mut max_constant_row_sum: f64 = 0.0;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| dense[[i, j]]).sum();
        max_constant_row_sum = max_constant_row_sum.max(row_sum.abs());
        for j in 0..n {
            max_symmetry_error = max_symmetry_error.max((dense[[i, j]] - dense[[j, i]]).abs());
        }
    }

    let energy_vector: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let energy = energy_vector
        .iter()
        .enumerate()
        .map(|(i, &x_i)| {
            energy_vector
                .iter()
                .enumerate()
                .map(|(j, &x_j)| x_i * dense[[i, j]] * x_j)
                .sum::<f64>()
        })
        .sum::<f64>();

    assert!(
        max_symmetry_error < 2.0e-12,
        "symmetry error={max_symmetry_error:e}"
    );
    assert!(
        max_constant_row_sum < 2.0e-11,
        "constant-field conservation error={max_constant_row_sum:e}"
    );
    assert!(energy > 1.0e-8, "curved stiffness energy={energy:e}");
}
