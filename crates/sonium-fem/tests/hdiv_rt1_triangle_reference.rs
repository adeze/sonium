//! Independent reference checks for the affine triangular RT1 contract.
//!
//! The production basis is not used as its own oracle here.  These tests
//! reconstruct the edge moments with Gauss integration, the cell moments with
//! an independent triangle rule, and the physical normal traces from mesh
//! geometry and topology.

use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::raviart_thomas::rt1_triangle;
use sonium_fem::forms::{DivergenceDivergenceForm, VectorMassForm};
use sonium_fem::mesh::{ElementType, MeshTopology, unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

fn edge_point(edge: usize, t: f64) -> [f64; 2] {
    match edge {
        0 => [(t + 1.0) * 0.5, 0.0],
        1 => [(1.0 - t) * 0.5, (t + 1.0) * 0.5],
        2 => [0.0, (1.0 - t) * 0.5],
        _ => panic!("reference triangle has three edges"),
    }
}

fn edge_normal_and_parameter_jacobian(edge: usize) -> ([f64; 2], f64) {
    match edge {
        0 => ([0.0, -1.0], 0.5),
        1 => {
            let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
            ([inv_sqrt_two, inv_sqrt_two], 2.0_f64.sqrt() * 0.5)
        }
        2 => ([-1.0, 0.0], 0.5),
        _ => panic!("reference triangle has three edges"),
    }
}

pub fn rt1_reference_basis_has_identity_edge_and_cell_moments() {
    let gauss_point = 1.0 / 3.0_f64.sqrt();
    let gauss_points = [-gauss_point, gauss_point];

    for edge in 0..3 {
        let (normal, parameter_jacobian) = edge_normal_and_parameter_jacobian(edge);
        for mode in 0..2 {
            for basis_index in 0..8 {
                let integral = gauss_points
                    .iter()
                    .map(|&t| {
                        let point = edge_point(edge, t);
                        let (basis, _) = rt1_triangle(point[0], point[1]);
                        let flux =
                            basis[basis_index][0] * normal[0] + basis[basis_index][1] * normal[1];
                        flux * t.powi(mode as i32) * parameter_jacobian
                    })
                    .sum::<f64>();
                let expected = if basis_index == 2 * edge + mode {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (integral - expected).abs() < 2.0e-12,
                    "edge {edge} mode {mode} basis {basis_index}: {integral} != {expected}"
                );
            }
        }
    }

    // The three-point degree-two rule has points (1/6,1/6), (2/3,1/6),
    // (1/6,2/3) and weight 1/6 on the unit reference triangle.
    let cell_points = [
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ];
    for component in 0..2 {
        for basis_index in 0..8 {
            let integral = cell_points
                .iter()
                .map(|&[xi, eta]| {
                    let (basis, _) = rt1_triangle(xi, eta);
                    basis[basis_index][component] / 6.0
                })
                .sum::<f64>();
            let expected = if basis_index == 6 + component {
                1.0
            } else {
                0.0
            };
            assert!(
                (integral - expected).abs() < 2.0e-12,
                "cell component {component} basis {basis_index}: {integral} != {expected}"
            );
        }
    }
}

pub fn rt1_reference_divergence_matches_independent_finite_difference() {
    let xi = 0.23;
    let eta = 0.31;
    let epsilon = 1.0e-6;
    let (basis, divs) = rt1_triangle(xi, eta);
    for index in 0..basis.len() {
        let (x_plus, _) = rt1_triangle(xi + epsilon, eta);
        let (x_minus, _) = rt1_triangle(xi - epsilon, eta);
        let (y_plus, _) = rt1_triangle(xi, eta + epsilon);
        let (y_minus, _) = rt1_triangle(xi, eta - epsilon);
        let numerical = (x_plus[index][0] - x_minus[index][0]) / (2.0 * epsilon)
            + (y_plus[index][1] - y_minus[index][1]) / (2.0 * epsilon);
        assert!(
            (numerical - divs[index]).abs() < 2.0e-8,
            "basis {index}: finite-difference divergence {numerical} != {}",
            divs[index]
        );
    }
}

pub fn rt1_triangle_space_has_stable_edge_and_cell_topology() {
    let mesh = unit_square_triangles(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let topology = MeshTopology::build(&mesh);

    assert_eq!(space.order(), HdivOrder::Rt1);
    assert_eq!(space.polynomial_degree(), 2);
    assert!(space.validate(&mesh).is_ok());
    assert_eq!(
        space.rt1_global_dof_layout(&mesh),
        Some((2 * topology.num_edges(), 2 * mesh.elements.len()))
    );
    assert_eq!(
        space.num_global_dofs(&mesh),
        2 * topology.num_edges() + 2 * mesh.elements.len()
    );

    for element in 0..mesh.elements.len() {
        let dofs = space.local_to_global(&mesh, element);
        assert_eq!(dofs.len(), 8);
        assert_eq!(
            dofs[..6],
            [
                2 * topology.element_edge_indices(element)[0],
                2 * topology.element_edge_indices(element)[0] + 1,
                2 * topology.element_edge_indices(element)[1],
                2 * topology.element_edge_indices(element)[1] + 1,
                2 * topology.element_edge_indices(element)[2],
                2 * topology.element_edge_indices(element)[2] + 1,
            ]
        );
        assert!(dofs.iter().all(|&dof| dof < space.num_global_dofs(&mesh)));
        assert_eq!(dofs[6], 2 * topology.num_edges() + 2 * element);
        assert_eq!(dofs[7], 2 * topology.num_edges() + 2 * element + 1);
    }

    let mass = assemble_bilinear_checked(
        &VectorMassForm { coefficient: 1.0 },
        &space,
        &space,
        &mesh,
        5,
    )
    .expect("RT1 triangle mass assembly should pass checked admission");
    let divergence = assemble_bilinear_checked(&DivergenceDivergenceForm, &space, &space, &mesh, 5)
        .expect("RT1 triangle divergence assembly should pass checked admission");
    assert_eq!(mass.num_rows, space.num_global_dofs(&mesh));
    assert_eq!(mass.num_cols, space.num_global_dofs(&mesh));
    assert_eq!(divergence.num_rows, space.num_global_dofs(&mesh));
    assert!(mass.values.iter().all(|value| value.is_finite()));
    assert!(divergence.values.iter().all(|value| value.is_finite()));
}

pub fn rt1_shared_edge_traces_are_invariant_under_local_reversal() {
    let mesh = unit_square_triangles(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
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

    let (a, b) = topology.edges[shared_edge];
    let dx = mesh.nodes[b].x - mesh.nodes[a].x;
    let dy = mesh.nodes[b].y - mesh.nodes[a].y;
    let length = (dx * dx + dy * dy).sqrt();
    let canonical_normal = [dy / length, -dx / length];

    for t_global in [-0.7, 0.0, 0.7] {
        for &(element, local_edge) in &[(0, first_local), (1, second_local)] {
            let sign = topology.element_edge_signs(element)[local_edge] as f64;
            let t_local = sign * t_global;
            let point = edge_point(local_edge, t_local);
            let tab = space.tabulate(&mesh, element, point[0], point[1], 0.0);
            for basis_index in 0..tab.n_dofs {
                let value = tab.vector_value(basis_index, 0) * canonical_normal[0]
                    + tab.vector_value(basis_index, 1) * canonical_normal[1];
                let expected = if basis_index == 2 * local_edge {
                    1.0 / length
                } else if basis_index == 2 * local_edge + 1 {
                    3.0 * t_global / length
                } else {
                    0.0
                };
                assert!(
                    (value - expected).abs() < 2.0e-10,
                    "element {element} edge {local_edge} t={t_global} basis {basis_index}: {value} != {expected}"
                );
            }
        }
    }
}

pub fn rt1_admission_accepts_implemented_cell_families() {
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let mesh = unit_cube_tetrahedra(1);
    assert_eq!(space.num_local_dofs(ElementType::Tetrahedron), 15);
    assert!(space.supports_element(ElementType::Tetrahedron));
    assert!(space.validate(&mesh).is_ok());

    let hexahedra = sonium_fem::mesh::unit_cube_hexahedra(1);
    assert!(space.supports_element(ElementType::Hexahedron));
    space
        .validate(&hexahedra)
        .expect("RT1 hexahedral mesh validates");
    assert_eq!(space.local_to_global(&hexahedra, 0).len(), 36);
}

#[test]
fn rt1_reference_basis_test() {
    rt1_reference_basis_has_identity_edge_and_cell_moments();
}

#[test]
fn rt1_reference_divergence_test() {
    rt1_reference_divergence_matches_independent_finite_difference();
}

#[test]
fn rt1_triangle_topology_test() {
    rt1_triangle_space_has_stable_edge_and_cell_topology();
}

#[test]
fn rt1_shared_edge_orientation_test() {
    rt1_shared_edge_traces_are_invariant_under_local_reversal();
}

#[test]
fn rt1_unsupported_admission_test() {
    rt1_admission_accepts_implemented_cell_families();
}
