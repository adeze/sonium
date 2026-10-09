//! Focused regression coverage for the bounded W1-05 high-order slice.

use sonium_fem::basis::{
    PolynomialDegree, nedelec, p2_tetrahedron, p2_tetrahedron_grad, p2_triangle, p2_triangle_grad,
};
use sonium_fem::mesh::{topology::MeshTopology, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space, HcurlSpace};

fn triangle_quadratic(x: f64, y: f64) -> (f64, [f64; 2]) {
    (
        0.7 + 1.1 * x - 0.9 * y + 2.0 * x * x - 1.3 * x * y + 1.7 * y * y,
        [1.1 + 4.0 * x - 1.3 * y, -0.9 - 1.3 * x + 3.4 * y],
    )
}

fn tetrahedron_quadratic(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    (
        0.7 + 1.1 * x - 0.9 * y + 0.5 * z + 2.0 * x * x - 1.3 * x * y + 0.8 * x * z + 1.7 * y * y
            - 0.6 * y * z
            + 0.9 * z * z,
        [
            1.1 + 4.0 * x - 1.3 * y + 0.8 * z,
            -0.9 - 1.3 * x + 3.4 * y - 0.6 * z,
            0.5 + 0.8 * x - 0.6 * y + 1.8 * z,
        ],
    )
}

#[test]
fn p2_simplex_patch_reproduces_quadratic_values_and_gradients() {
    let triangle_nodes = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ];
    let triangle_nodal = triangle_nodes.map(|[x, y]| triangle_quadratic(x, y).0);
    for [x, y] in [[0.17, 0.23], [0.31, 0.19], [0.08, 0.11]] {
        let basis = p2_triangle(x, y);
        let value = basis
            .iter()
            .zip(triangle_nodal)
            .map(|(&shape, nodal)| shape * nodal)
            .sum::<f64>();
        let (exact, exact_gradient) = triangle_quadratic(x, y);
        assert!((value - exact).abs() < 2.0e-12);
        let gradient = p2_triangle_grad(x, y);
        for direction in 0..2 {
            let reproduced = gradient
                .iter()
                .zip(triangle_nodal)
                .map(|(shape_gradient, nodal)| shape_gradient[direction] * nodal)
                .sum::<f64>();
            assert!((reproduced - exact_gradient[direction]).abs() < 2.0e-12);
        }
    }

    let tetrahedron_nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let tetrahedron_nodal = tetrahedron_nodes.map(|[x, y, z]| tetrahedron_quadratic(x, y, z).0);
    for [x, y, z] in [[0.10, 0.20, 0.30], [0.18, 0.11, 0.22], [0.20, 0.25, 0.10]] {
        let basis = p2_tetrahedron(x, y, z);
        let value = basis
            .iter()
            .zip(tetrahedron_nodal)
            .map(|(&shape, nodal)| shape * nodal)
            .sum::<f64>();
        let (exact, exact_gradient) = tetrahedron_quadratic(x, y, z);
        assert!((value - exact).abs() < 2.0e-12);
        let gradient = p2_tetrahedron_grad(x, y, z);
        for direction in 0..3 {
            let reproduced = gradient
                .iter()
                .zip(tetrahedron_nodal)
                .map(|(shape_gradient, nodal)| shape_gradient[direction] * nodal)
                .sum::<f64>();
            assert!((reproduced - exact_gradient[direction]).abs() < 2.0e-12);
        }
    }
}

#[test]
fn order2_tetrahedron_adjacent_edge_reversals_have_expected_parity() {
    let point = (0.19, 0.23, 0.17);
    let canonical = nedelec::nedelec2_tetrahedron_oriented(point.0, point.1, point.2, [0, 1, 2, 3]);
    for (vertex_ids, edge_idx) in [
        ([1, 0, 2, 3], 0usize),
        ([0, 2, 1, 3], 3usize),
        ([0, 1, 3, 2], 5usize),
    ] {
        let reversed =
            nedelec::nedelec2_tetrahedron_oriented(point.0, point.1, point.2, vertex_ids);
        let first = 2 * edge_idx;
        for component in 0..3 {
            assert!((canonical.0[first][component] + reversed.0[first][component]).abs() < 2.0e-13);
            assert!((canonical.1[first][component] + reversed.1[first][component]).abs() < 2.0e-13);
            assert!(
                (canonical.0[first + 1][component] - reversed.0[first + 1][component]).abs()
                    < 2.0e-13
            );
            assert!(
                (canonical.1[first + 1][component] - reversed.1[first + 1][component]).abs()
                    < 2.0e-13
            );
        }
        for basis in 0..12 {
            if basis == first || basis == first + 1 {
                continue;
            }
            for component in 0..3 {
                assert!(
                    (canonical.0[basis][component] - reversed.0[basis][component]).abs() < 2.0e-13
                );
                assert!(
                    (canonical.1[basis][component] - reversed.1[basis][component]).abs() < 2.0e-13
                );
            }
        }
    }
}

fn solve_8x8(mut matrix: [[f64; 8]; 8], mut rhs: [f64; 8]) -> [f64; 8] {
    for column in 0..8 {
        let (pivot_row, pivot) = (column..8)
            .map(|row| (row, matrix[row][column].abs()))
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .expect("non-empty Gaussian-elimination column");
        assert!(
            pivot > 1.0e-11,
            "singular H(curl) sample matrix at column {column}"
        );
        matrix.swap(column, pivot_row);
        rhs.swap(column, pivot_row);
        let diagonal = matrix[column][column];
        for entry in &mut matrix[column][column..] {
            *entry /= diagonal;
        }
        rhs[column] /= diagonal;

        for row in 0..8 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..8 {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs
}

fn manufactured_quadratic([x, y, _z]: [f64; 3]) -> f64 {
    0.3 + 0.7 * x - 0.4 * y + 1.2 * x * x - 0.8 * x * y + 0.9 * y * y
}

fn h1_gradient_at(
    tabulation: &sonium_fem::spaces::BasisTabulation,
    coefficients: &[f64],
) -> [f64; 2] {
    [
        coefficients
            .iter()
            .enumerate()
            .map(|(local, &coefficient)| coefficient * tabulation.grad(local, 0))
            .sum(),
        coefficients
            .iter()
            .enumerate()
            .map(|(local, &coefficient)| coefficient * tabulation.grad(local, 1))
            .sum(),
    ]
}

fn hcurl_value_at(
    tabulation: &sonium_fem::spaces::BasisTabulation,
    coefficients: &[f64; 8],
) -> [f64; 2] {
    [
        coefficients
            .iter()
            .enumerate()
            .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 0))
            .sum(),
        coefficients
            .iter()
            .enumerate()
            .map(|(local, &coefficient)| coefficient * tabulation.vector_value(local, 1))
            .sum(),
    ]
}

#[test]
fn p2_gradient_is_reproduced_and_oriented_across_shared_edge() {
    let mesh = unit_square_triangles(1);
    let h1 = H1Space::new(PolynomialDegree::P2);
    let hcurl = HcurlSpace::order2();
    assert!(h1.validate(&mesh).is_ok());
    assert!(hcurl.validate(&mesh).is_ok());

    let global_dof_coordinates = h1
        .dof_coordinates(&mesh)
        .expect("P2 H1 space must expose vertex and edge coordinates");
    let scalar_dofs: Vec<f64> = global_dof_coordinates
        .iter()
        .copied()
        .map(manufactured_quadratic)
        .collect();

    // Four points provide eight scalar equations for the eight local order-2
    // Nédélec coefficients. The final two points below are deliberately not
    // part of this reconstruction set.
    let reconstruction_points = [[0.13, 0.19], [0.22, 0.17], [0.16, 0.43], [0.37, 0.11]];
    let check_points = [[0.09, 0.27], [0.28, 0.21], [0.14, 0.58]];
    let mut local_coefficients = Vec::new();
    for elem_idx in 0..mesh.num_elements() {
        let h1_dofs = h1.local_to_global(&mesh, elem_idx);
        let h1_coefficients: Vec<f64> = h1_dofs.iter().map(|&dof| scalar_dofs[dof]).collect();
        assert_eq!(h1_coefficients.len(), 6);

        let mut matrix = [[0.0; 8]; 8];
        let mut rhs = [0.0; 8];
        for (point_idx, &[xi, eta]) in reconstruction_points.iter().enumerate() {
            let h1_tab = h1.tabulate(&mesh, elem_idx, xi, eta, 0.0);
            let hcurl_tab = hcurl.tabulate(&mesh, elem_idx, xi, eta, 0.0);
            let gradient = h1_gradient_at(&h1_tab, &h1_coefficients);
            for local in 0..8 {
                matrix[2 * point_idx][local] = hcurl_tab.vector_value(local, 0);
                matrix[2 * point_idx + 1][local] = hcurl_tab.vector_value(local, 1);
            }
            rhs[2 * point_idx] = gradient[0];
            rhs[2 * point_idx + 1] = gradient[1];
        }
        let coefficients = solve_8x8(matrix, rhs);

        for &[xi, eta] in &check_points {
            let h1_tab = h1.tabulate(&mesh, elem_idx, xi, eta, 0.0);
            let hcurl_tab = hcurl.tabulate(&mesh, elem_idx, xi, eta, 0.0);
            let expected = h1_gradient_at(&h1_tab, &h1_coefficients);
            let reproduced = hcurl_value_at(&hcurl_tab, &coefficients);
            for direction in 0..2 {
                assert!(
                    (expected[direction] - reproduced[direction]).abs() < 2.0e-11,
                    "P2 gradient is not in order-2 H(curl) on element {elem_idx}, direction {direction}: expected {}, reproduced {}",
                    expected[direction],
                    reproduced[direction]
                );
            }
        }
        local_coefficients.push((hcurl.local_to_global(&mesh, elem_idx), coefficients));
    }

    let mut global_coefficients: Vec<Option<f64>> = vec![None; hcurl.num_global_dofs(&mesh)];
    for (dofs, coefficients) in &local_coefficients {
        for (&global, &coefficient) in dofs.iter().zip(coefficients) {
            if let Some(previous) = global_coefficients[global] {
                assert!(
                    (previous - coefficient).abs() < 2.0e-10,
                    "shared order-2 H(curl) DOF {global} is not orientation-consistent"
                );
            } else {
                global_coefficients[global] = Some(coefficient);
            }
        }
    }

    let topology = MeshTopology::build(&mesh);
    let first_edges = topology.element_edge_indices(0);
    let second_edges = topology.element_edge_indices(1);
    let (first_local, &shared_edge) = first_edges
        .iter()
        .enumerate()
        .find(|(_, edge)| second_edges.contains(edge))
        .expect("unit square triangles must share an edge");
    let second_local = second_edges
        .iter()
        .position(|edge| *edge == shared_edge)
        .expect("shared edge must have a local index");
    assert_eq!(
        topology.element_edge_signs(0)[first_local],
        -topology.element_edge_signs(1)[second_local],
        "the patch must exercise an oppositely oriented shared edge"
    );
    for mode in 0..2 {
        let first = local_coefficients[0].1[2 * first_local + mode];
        let second = local_coefficients[1].1[2 * second_local + mode];
        assert!(
            (first - second).abs() < 2.0e-10,
            "shared edge mode {mode} does not have a common global coefficient"
        );
    }
}
