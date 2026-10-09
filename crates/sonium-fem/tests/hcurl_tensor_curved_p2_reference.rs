//! Independent CPU reference for tensor-product H(curl) on curved Q2 cells.
//!
//! The production basis is checked against explicit Q2 isoparametric maps and
//! independently written lowest-order tensor-product Nédélec functions.  The
//! test deliberately covers both dimensions and both covariant-Piola value
//! and curl transformations.

use sonium_fem::mesh::{Mesh, MeshTopology, Point, unit_cube_hexahedra, unit_square_quads};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn q2_1d(x: f64) -> ([f64; 3], [f64; 3]) {
    (
        [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)],
        [x - 0.5, -2.0 * x, x + 0.5],
    )
}

fn q2_quad_map(mesh: &Mesh, xi: f64, eta: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let (lx, dx) = q2_1d(xi);
    let (ly, dy) = q2_1d(eta);
    let indices = [
        (0, 0),
        (2, 0),
        (2, 2),
        (0, 2),
        (1, 0),
        (2, 1),
        (1, 2),
        (0, 1),
        (1, 1),
    ];
    let element = &mesh.elements[0];
    let mut point = [0.0; 2];
    let mut jacobian = [[0.0; 2]; 2];
    for (local, &(i, j)) in indices.iter().enumerate() {
        let value = lx[i] * ly[j];
        let gradient = [dx[i] * ly[j], lx[i] * dy[j]];
        let node = mesh.nodes[element.nodes[local]];
        point[0] += value * node.x;
        point[1] += value * node.y;
        jacobian[0][0] += gradient[0] * node.x;
        jacobian[0][1] += gradient[1] * node.x;
        jacobian[1][0] += gradient[0] * node.y;
        jacobian[1][1] += gradient[1] * node.y;
    }
    (point, jacobian)
}

fn q2_hex_map(mesh: &Mesh, xi: f64, eta: f64, zeta: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    let (lx, dx) = q2_1d(xi);
    let (ly, dy) = q2_1d(eta);
    let (lz, dz) = q2_1d(zeta);
    let indices = [
        (0, 0, 0),
        (2, 0, 0),
        (2, 2, 0),
        (0, 2, 0),
        (0, 0, 2),
        (2, 0, 2),
        (2, 2, 2),
        (0, 2, 2),
        (1, 0, 0),
        (2, 1, 0),
        (1, 2, 0),
        (0, 1, 0),
        (1, 0, 2),
        (2, 1, 2),
        (1, 2, 2),
        (0, 1, 2),
        (0, 0, 1),
        (2, 0, 1),
        (2, 2, 1),
        (0, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
        (1, 0, 1),
        (1, 2, 1),
        (0, 1, 1),
        (2, 1, 1),
        (1, 1, 1),
    ];
    let element = &mesh.elements[0];
    let mut point = [0.0; 3];
    let mut jacobian = [[0.0; 3]; 3];
    for (local, &(i, j, k)) in indices.iter().enumerate() {
        let value = lx[i] * ly[j] * lz[k];
        let gradient = [
            dx[i] * ly[j] * lz[k],
            lx[i] * dy[j] * lz[k],
            lx[i] * ly[j] * dz[k],
        ];
        let node = mesh.nodes[element.nodes[local]];
        let coordinates = [node.x, node.y, node.z];
        for component in 0..3 {
            point[component] += value * coordinates[component];
            for reference_component in 0..3 {
                jacobian[component][reference_component] +=
                    gradient[reference_component] * coordinates[component];
            }
        }
    }
    (point, jacobian)
}

fn inverse_2x2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], f64) {
    let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    (
        [
            [a[1][1] / determinant, -a[0][1] / determinant],
            [-a[1][0] / determinant, a[0][0] / determinant],
        ],
        determinant,
    )
}

fn inverse_3x3(a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let determinant = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let inverse = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / determinant,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / determinant,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / determinant,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / determinant,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / determinant,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / determinant,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / determinant,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / determinant,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / determinant,
        ],
    ];
    (inverse, determinant)
}

fn reference_quad_basis(xi: f64, eta: f64) -> ([[f64; 2]; 4], [f64; 4]) {
    (
        [
            [0.5 * (1.0 - eta), 0.0],
            [0.0, 0.5 * (1.0 + xi)],
            [-0.5 * (1.0 + eta), 0.0],
            [0.0, -0.5 * (1.0 - xi)],
        ],
        [0.5; 4],
    )
}

fn reference_hex_basis(xi: f64, eta: f64, zeta: f64) -> ([[f64; 3]; 12], [[f64; 3]; 12]) {
    let q = 0.25;
    let values = [
        [q * (1.0 - eta) * (1.0 - zeta), 0.0, 0.0],
        [0.0, q * (1.0 + xi) * (1.0 - zeta), 0.0],
        [-q * (1.0 + eta) * (1.0 - zeta), 0.0, 0.0],
        [0.0, -q * (1.0 - xi) * (1.0 - zeta), 0.0],
        [q * (1.0 - eta) * (1.0 + zeta), 0.0, 0.0],
        [0.0, q * (1.0 + xi) * (1.0 + zeta), 0.0],
        [-q * (1.0 + eta) * (1.0 + zeta), 0.0, 0.0],
        [0.0, -q * (1.0 - xi) * (1.0 + zeta), 0.0],
        [0.0, 0.0, q * (1.0 - xi) * (1.0 - eta)],
        [0.0, 0.0, q * (1.0 + xi) * (1.0 - eta)],
        [0.0, 0.0, q * (1.0 + xi) * (1.0 + eta)],
        [0.0, 0.0, q * (1.0 - xi) * (1.0 + eta)],
    ];
    let curls = [
        [0.0, -q * (1.0 - eta), q * (1.0 - zeta)],
        [q * (1.0 + xi), 0.0, q * (1.0 - zeta)],
        [0.0, q * (1.0 + eta), q * (1.0 - zeta)],
        [-q * (1.0 - xi), 0.0, q * (1.0 - zeta)],
        [0.0, q * (1.0 - eta), q * (1.0 + zeta)],
        [-q * (1.0 + xi), 0.0, q * (1.0 + zeta)],
        [0.0, -q * (1.0 + eta), q * (1.0 + zeta)],
        [q * (1.0 - xi), 0.0, q * (1.0 + zeta)],
        [-q * (1.0 - xi), q * (1.0 - eta), 0.0],
        [-q * (1.0 + xi), -q * (1.0 - eta), 0.0],
        [q * (1.0 + xi), -q * (1.0 + eta), 0.0],
        [q * (1.0 - xi), q * (1.0 + eta), 0.0],
    ];
    (values, curls)
}

pub fn curved_q2_quadrilateral_hcurl_reference() {
    let mut mesh = unit_square_quads(1).to_p2();
    let edge_midpoint = mesh.elements[0].nodes[4];
    mesh.nodes[edge_midpoint] = Point::new_2d(0.5, 0.13);

    let space = HcurlSpace::new();
    space
        .validate(&mesh)
        .expect("curved Q2 quadrilateral must validate");
    let topology = MeshTopology::build(&mesh);
    assert_eq!(space.num_global_dofs(&mesh), topology.num_edges());
    assert_eq!(space.num_local_dofs(mesh.elements[0].element_type), 4);
    assert_eq!(space.local_to_global(&mesh, 0).len(), 4);

    let (xi, eta) = (0.23, -0.31);
    let (expected_point, jacobian) = q2_quad_map(&mesh, xi, eta);
    let (inverse, determinant) = inverse_2x2(jacobian);
    let (reference_values, reference_curls) = reference_quad_basis(xi, eta);
    let tabulation = space.tabulate(&mesh, 0, xi, eta, 0.0);
    assert_eq!(tabulation.n_dofs, 4);
    assert!(determinant > 0.0);
    for component in 0..2 {
        assert!((tabulation.physical_coords[component] - expected_point[component]).abs() < 3e-12);
    }

    let signs = topology.element_edge_signs(0);
    for local in 0..4 {
        let sign = signs[local] as f64;
        for component in 0..2 {
            let expected = sign
                * (inverse[0][component] * reference_values[local][0]
                    + inverse[1][component] * reference_values[local][1]);
            assert!(
                (tabulation.values[2 * local + component] - expected).abs() < 3e-11,
                "Q2 quadrilateral H(curl) value mismatch at basis {local}, component {component}"
            );
        }
        assert!(
            (tabulation.derivatives[local] - sign * reference_curls[local] / determinant).abs()
                < 3e-11
        );
    }
}

pub fn curved_q2_hexahedron_hcurl_reference() {
    let mut mesh = unit_cube_hexahedra(1).to_p2();
    let edge_midpoint = mesh.elements[0].nodes[8];
    mesh.nodes[edge_midpoint] = Point::new_3d(0.5, 0.0, 0.08);

    let space = HcurlSpace::new();
    space
        .validate(&mesh)
        .expect("curved Q2 hexahedron must validate");
    let topology = MeshTopology::build(&mesh);
    assert_eq!(space.num_global_dofs(&mesh), topology.num_edges());
    assert_eq!(space.num_local_dofs(mesh.elements[0].element_type), 12);
    assert_eq!(space.local_to_global(&mesh, 0).len(), 12);

    let (xi, eta, zeta) = (0.17, -0.21, 0.29);
    let (expected_point, jacobian) = q2_hex_map(&mesh, xi, eta, zeta);
    let (inverse, determinant) = inverse_3x3(jacobian);
    let (reference_values, reference_curls) = reference_hex_basis(xi, eta, zeta);
    let tabulation = space.tabulate(&mesh, 0, xi, eta, zeta);
    assert_eq!(tabulation.n_dofs, 12);
    assert!(determinant > 0.0);
    for component in 0..3 {
        assert!((tabulation.physical_coords[component] - expected_point[component]).abs() < 4e-12);
    }

    let signs = topology.element_edge_signs(0);
    for local in 0..12 {
        let sign = signs[local] as f64;
        for component in 0..3 {
            let expected = sign
                * (inverse[0][component] * reference_values[local][0]
                    + inverse[1][component] * reference_values[local][1]
                    + inverse[2][component] * reference_values[local][2]);
            assert!(
                (tabulation.values[3 * local + component] - expected).abs() < 5e-11,
                "Q2 hexahedron H(curl) value mismatch at basis {local}, component {component}"
            );
        }
        for component in 0..3 {
            let expected = sign
                * (jacobian[component][0] * reference_curls[local][0]
                    + jacobian[component][1] * reference_curls[local][1]
                    + jacobian[component][2] * reference_curls[local][2])
                / determinant;
            assert!(
                (tabulation.derivatives[3 * local + component] - expected).abs() < 8e-11,
                "Q2 hexahedron curl mismatch at basis {local}, component {component}"
            );
        }
    }
}

pub fn tensor_product_hcurl_curved_p2_reference() {
    curved_q2_quadrilateral_hcurl_reference();
    curved_q2_hexahedron_hcurl_reference();
}

#[test]
fn hcurl_curved_q2_quadrilateral_matches_independent_reference() {
    curved_q2_quadrilateral_hcurl_reference();
}

#[test]
fn hcurl_curved_q2_hexahedron_matches_independent_reference() {
    curved_q2_hexahedron_hcurl_reference();
}
