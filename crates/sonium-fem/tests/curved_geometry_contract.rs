//! W1-05 production coverage for isoparametric P2 geometry.
//!
//! Promoted meshes carry edge/face/interior nodes explicitly. H1 must use
//! those nodes for the geometry map while retaining the legacy vertex-only
//! P2 numbering path used by the existing convergence tests.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn affine_mesh(element_type: ElementType) -> Mesh {
    match element_type {
        ElementType::Triangle => sonium_fem::mesh::unit_square_triangles(1),
        ElementType::Quadrilateral => sonium_fem::mesh::unit_square_quads(1),
        ElementType::Tetrahedron => sonium_fem::mesh::unit_cube_tetrahedra(1),
        ElementType::Hexahedron => sonium_fem::mesh::unit_cube_hexahedra(1),
    }
}

fn add_curved_p2_connectivity(mesh: &mut Mesh) {
    *mesh = mesh.to_p2();
    let element_type = mesh.elements[0].element_type;
    let midpoint = mesh.elements[0].nodes[element_type.num_vertices()];
    let point = &mut mesh.nodes[midpoint];
    point.x += 0.013;
    point.y -= 0.017;
    if mesh.dimension == 3 {
        point.z += 0.011;
    }
}

pub fn w1_05_h1_accepts_curved_p2_geometry_across_supported_cells() {
    for element_type in [
        ElementType::Triangle,
        ElementType::Quadrilateral,
        ElementType::Tetrahedron,
        ElementType::Hexahedron,
    ] {
        let mut mesh = affine_mesh(element_type);
        add_curved_p2_connectivity(&mut mesh);

        let p1 = H1Space::new(PolynomialDegree::P1);
        p1.validate(&mesh)
            .expect("P1 fields must accept explicit P2 geometry");
        let p2 = H1Space::new(PolynomialDegree::P2);
        p2.validate(&mesh)
            .expect("P2 fields must accept explicit P2 geometry");
        assert_eq!(p2.num_global_dofs(&mesh), mesh.num_nodes());

        let (xi, eta, zeta) = match element_type {
            ElementType::Triangle | ElementType::Tetrahedron => (0.5, 0.0, 0.0),
            ElementType::Quadrilateral => (0.0, -1.0, 0.0),
            ElementType::Hexahedron => (0.0, -1.0, -1.0),
        };
        let tabulation = p2.tabulate(&mesh, 0, xi, eta, zeta);
        let midpoint = mesh.elements[0].nodes[element_type.num_vertices()];
        let expected = mesh.nodes[midpoint];
        assert!((tabulation.physical_coords[0] - expected.x).abs() < 1e-12);
        assert!((tabulation.physical_coords[1] - expected.y).abs() < 1e-12);
        assert!((tabulation.physical_coords[2] - expected.z).abs() < 1e-12);
        assert!(
            tabulation.det_j > 0.0,
            "invalid curved Jacobian for {element_type:?}"
        );
    }
}

pub fn w1_05_curved_p2_hessians_match_reference_coordinate_finite_difference() {
    let mut mesh = affine_mesh(ElementType::Triangle);
    add_curved_p2_connectivity(&mut mesh);
    let space = H1Space::new(PolynomialDegree::P2);
    let (xi, eta, zeta) = (0.23, 0.21, 0.0);
    let step = 1.0e-6;
    let tabulation = space.tabulate(&mesh, 0, xi, eta, zeta);
    assert!(tabulation.has_hessians());

    let xi_minus = space.tabulate(&mesh, 0, xi - step, eta, zeta);
    let xi_plus = space.tabulate(&mesh, 0, xi + step, eta, zeta);
    let eta_minus = space.tabulate(&mesh, 0, xi, eta - step, zeta);
    let eta_plus = space.tabulate(&mesh, 0, xi, eta + step, zeta);
    let jacobian = [
        (xi_plus.physical_coords[0] - xi_minus.physical_coords[0]) / (2.0 * step),
        (eta_plus.physical_coords[0] - eta_minus.physical_coords[0]) / (2.0 * step),
        (xi_plus.physical_coords[1] - xi_minus.physical_coords[1]) / (2.0 * step),
        (eta_plus.physical_coords[1] - eta_minus.physical_coords[1]) / (2.0 * step),
    ];
    let determinant = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2];
    let inverse = [
        jacobian[3] / determinant,
        -jacobian[1] / determinant,
        -jacobian[2] / determinant,
        jacobian[0] / determinant,
    ];

    for basis in 0..tabulation.n_dofs {
        let reference_derivative = [
            (xi_plus.derivatives[basis * 2] - xi_minus.derivatives[basis * 2]) / (2.0 * step),
            (eta_plus.derivatives[basis * 2] - eta_minus.derivatives[basis * 2]) / (2.0 * step),
        ];
        let eta_derivative = [
            (xi_plus.derivatives[basis * 2 + 1] - xi_minus.derivatives[basis * 2 + 1])
                / (2.0 * step),
            (eta_plus.derivatives[basis * 2 + 1] - eta_minus.derivatives[basis * 2 + 1])
                / (2.0 * step),
        ];
        let derivative = [reference_derivative, eta_derivative];
        for row in 0..2 {
            for col in 0..2 {
                let expected =
                    derivative[row][0] * inverse[col] + derivative[row][1] * inverse[2 + col];
                assert!(
                    (tabulation.hessian(basis, row, col) - expected).abs() < 3.0e-5,
                    "basis={basis} row={row} col={col} expected={expected} actual={}",
                    tabulation.hessian(basis, row, col)
                );
            }
        }
    }
}

#[test]
fn curved_p2_geometry_contract() {
    w1_05_h1_accepts_curved_p2_geometry_across_supported_cells();
}

#[test]
fn curved_p2_hessian_contract() {
    w1_05_curved_p2_hessians_match_reference_coordinate_finite_difference();
}
