//! Independent reference checks for RT1 on an isoparametric P2 hexahedron.

use sonium_fem::basis::{PolynomialDegree, evaluate_shape};
use sonium_fem::mesh::{ElementType, Mesh, unit_cube_hexahedra};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn curved_hexahedron() -> Mesh {
    let mut mesh = unit_cube_hexahedra(1).to_p2();
    let midpoint = mesh.elements[0].nodes[ElementType::Hexahedron.num_vertices()];
    let point = &mut mesh.nodes[midpoint];
    point.x += 0.013;
    point.y -= 0.017;
    point.z += 0.011;
    mesh
}

fn inverse(matrix: [[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    let inverse = [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ];
    (inverse, determinant)
}

fn reference_physical_point(mesh: &Mesh, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
    let shape = evaluate_shape(ElementType::Hexahedron, PolynomialDegree::P2, xi, eta, zeta);
    let nodes = &mesh.elements[0].nodes;
    let mut point = [0.0; 3];
    for (weight, &node) in shape.values.iter().zip(nodes) {
        point[0] += weight * mesh.nodes[node].x;
        point[1] += weight * mesh.nodes[node].y;
        point[2] += weight * mesh.nodes[node].z;
    }
    point
}

pub fn curved_rt1_hdiv_uses_p2_geometry_and_admits_hex() {
    let mesh = curved_hexahedron();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    assert!(space.validate(&mesh).is_ok());
    assert_eq!(space.num_local_dofs(ElementType::Hexahedron), 36);
    assert_eq!(space.num_global_dofs(&mesh), 36);

    let point = [0.17, -0.21, 0.23];
    let tabulation = space.tabulate(&mesh, 0, point[0], point[1], point[2]);
    let expected = reference_physical_point(&mesh, point[0], point[1], point[2]);
    for (actual, expected) in tabulation.physical_coords.iter().zip(expected) {
        assert!((actual - expected).abs() < 2.0e-12);
    }
    assert!(tabulation.det_j > 0.0);
    assert!(tabulation.values.iter().all(|value| value.is_finite()));
    assert!(tabulation.derivatives.iter().all(|value| value.is_finite()));
}

pub fn curved_rt1_hdiv_piola_divergence_matches_physical_finite_difference() {
    let mesh = curved_hexahedron();
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let point = [0.17, -0.21, 0.23];
    let step = 1.0e-6;
    let center = space.tabulate(&mesh, 0, point[0], point[1], point[2]);

    let mut jacobian = [[0.0; 3]; 3];
    let mut value_derivatives = vec![[[0.0; 3]; 3]; center.n_dofs];
    for direction in 0..3 {
        let mut plus_point = point;
        let mut minus_point = point;
        plus_point[direction] += step;
        minus_point[direction] -= step;
        let plus = space.tabulate(&mesh, 0, plus_point[0], plus_point[1], plus_point[2]);
        let minus = space.tabulate(&mesh, 0, minus_point[0], minus_point[1], minus_point[2]);
        for component in 0..3 {
            jacobian[component][direction] =
                (plus.physical_coords[component] - minus.physical_coords[component]) / (2.0 * step);
        }
        for basis in 0..center.n_dofs {
            for component in 0..3 {
                value_derivatives[basis][component][direction] =
                    (plus.values[basis * 3 + component] - minus.values[basis * 3 + component])
                        / (2.0 * step);
            }
        }
    }

    let (inverse_jacobian, determinant) = inverse(jacobian);
    assert!(determinant > 0.0);
    for basis in 0..center.n_dofs {
        let mut divergence = 0.0;
        for component in 0..3 {
            for direction in 0..3 {
                divergence += value_derivatives[basis][component][direction]
                    * inverse_jacobian[direction][component];
            }
        }
        assert!(
            (divergence - center.derivatives[basis]).abs() < 2.0e-5,
            "basis {basis}: physical divergence {divergence} != Piola divergence {}",
            center.derivatives[basis]
        );
    }
}

#[test]
fn curved_rt1_geometry_test() {
    curved_rt1_hdiv_uses_p2_geometry_and_admits_hex();
}

#[test]
fn curved_rt1_piola_test() {
    curved_rt1_hdiv_piola_divergence_matches_physical_finite_difference();
}
