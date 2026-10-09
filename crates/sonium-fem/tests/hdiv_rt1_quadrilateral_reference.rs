//! Independent reference checks for the affine quadrilateral RT1 contract.

use sonium_fem::basis::raviart_thomas::rt1_quadrilateral;
use sonium_fem::mesh::{MeshTopology, unit_cube_tetrahedra, unit_square_quads};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

const GAUSS_NODES: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_3,
    0.339_981_043_584_856_3,
    0.861_136_311_594_052_6,
];
const GAUSS_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_85,
    0.652_145_154_862_546_1,
    0.652_145_154_862_546_1,
    0.347_854_845_137_453_85,
];

fn edge_point(edge: usize, t: f64) -> [f64; 2] {
    match edge {
        0 => [t, -1.0],
        1 => [1.0, t],
        2 => [-t, 1.0],
        3 => [-1.0, -t],
        _ => panic!("reference quadrilateral has four edges"),
    }
}

fn edge_normal(edge: usize) -> [f64; 2] {
    match edge {
        0 => [0.0, -1.0],
        1 => [1.0, 0.0],
        2 => [0.0, 1.0],
        3 => [-1.0, 0.0],
        _ => panic!("reference quadrilateral has four edges"),
    }
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub fn rt1_quadrilateral_reference_basis_has_identity_moments() {
    let mut moments = [[0.0; 12]; 12];
    for (&t, &weight) in GAUSS_NODES.iter().zip(GAUSS_WEIGHTS.iter()) {
        for edge in 0..4 {
            let point = edge_point(edge, t);
            let normal = edge_normal(edge);
            let (values, _) = rt1_quadrilateral(point[0], point[1]);
            for basis in 0..12 {
                let normal_value = dot(values[basis], normal);
                moments[basis][2 * edge] += weight * normal_value;
                moments[basis][2 * edge + 1] += weight * t * normal_value;
            }
        }
    }

    for (&xi, &weight_x) in GAUSS_NODES.iter().zip(GAUSS_WEIGHTS.iter()) {
        for (&eta, &weight_y) in GAUSS_NODES.iter().zip(GAUSS_WEIGHTS.iter()) {
            let (values, _) = rt1_quadrilateral(xi, eta);
            let weight = weight_x * weight_y;
            for basis in 0..12 {
                moments[basis][8] += weight * values[basis][0];
                moments[basis][9] += weight * eta * values[basis][0];
                moments[basis][10] += weight * values[basis][1];
                moments[basis][11] += weight * xi * values[basis][1];
            }
        }
    }

    for (basis, row) in moments.iter().enumerate() {
        for (functional, &value) in row.iter().enumerate() {
            let expected = if basis == functional { 1.0 } else { 0.0 };
            assert!(
                (value - expected).abs() < 2e-12,
                "RT1 quadrilateral moment ({basis}, {functional}) = {value}"
            );
        }
    }
}

pub fn rt1_quadrilateral_reference_divergence_matches_finite_difference() {
    let xi = 0.217;
    let eta = -0.311;
    let h = 1e-6;
    let (_, divergences) = rt1_quadrilateral(xi, eta);
    let (xi_plus, _) = rt1_quadrilateral(xi + h, eta);
    let (xi_minus, _) = rt1_quadrilateral(xi - h, eta);
    let (eta_plus, _) = rt1_quadrilateral(xi, eta + h);
    let (eta_minus, _) = rt1_quadrilateral(xi, eta - h);

    for basis in 0..12 {
        let dx = (xi_plus[basis][0] - xi_minus[basis][0]) / (2.0 * h);
        let dy = (eta_plus[basis][1] - eta_minus[basis][1]) / (2.0 * h);
        assert!(
            (dx + dy - divergences[basis]).abs() < 2e-8,
            "RT1 quadrilateral divergence {basis}: finite difference {}, basis {}",
            dx + dy,
            divergences[basis]
        );
    }
}

pub fn rt1_quadrilateral_space_preserves_edge_and_cell_topology() {
    let mesh = unit_square_quads(2);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let topology = MeshTopology::build(&mesh);

    assert!(space.supports_element(sonium_fem::mesh::ElementType::Triangle));
    assert!(space.supports_element(sonium_fem::mesh::ElementType::Quadrilateral));
    assert!(space.supports_element(sonium_fem::mesh::ElementType::Tetrahedron));
    assert!(space.supports_element(sonium_fem::mesh::ElementType::Hexahedron));
    space
        .validate(&mesh)
        .expect("RT1 quadrilateral mesh validates");
    let tetra_mesh = unit_cube_tetrahedra(1);
    space
        .validate(&tetra_mesh)
        .expect("RT1 tetrahedral mesh validates");
    assert_eq!(space.local_to_global(&tetra_mesh, 0).len(), 15);

    let (edge_dofs, cell_dofs) = space
        .rt1_global_dof_layout(&mesh)
        .expect("validated RT1 layout");
    assert_eq!(edge_dofs, 2 * topology.num_edges());
    assert_eq!(cell_dofs, 4 * mesh.elements.len());
    assert_eq!(space.num_global_dofs(&mesh), edge_dofs + cell_dofs);

    for element in 0..mesh.elements.len() {
        let dofs = space.local_to_global(&mesh, element);
        assert_eq!(dofs.len(), 12);
        for pair in dofs[..8].chunks_exact(2) {
            assert_eq!(pair[1], pair[0] + 1);
            assert!(pair[0] < edge_dofs);
        }
        assert!(dofs[8] >= edge_dofs);
        assert!(dofs[11] < edge_dofs + cell_dofs);
        let mut unique = dofs.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), dofs.len());
    }

    for (&marker, edges) in &topology.boundary_edges {
        let mut expected: Vec<usize> = edges
            .iter()
            .flat_map(|&edge| [2 * edge, 2 * edge + 1])
            .collect();
        expected.sort_unstable();
        let mut actual = space.boundary_dofs(&mesh, marker);
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }
}

pub fn rt1_quadrilateral_shared_edge_traces_survive_local_reversal() {
    let mesh = unit_square_quads(2);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let topology = MeshTopology::build(&mesh);
    let mut shared = None;
    'outer: for first in 0..mesh.elements.len() {
        for second in first + 1..mesh.elements.len() {
            if let Some((first_local, &edge)) = topology
                .element_edge_indices(first)
                .iter()
                .enumerate()
                .find(|(_, edge)| topology.element_edge_indices(second).contains(edge))
            {
                let second_local = topology
                    .element_edge_indices(second)
                    .iter()
                    .position(|candidate| *candidate == edge)
                    .expect("shared edge has a local index");
                shared = Some((first, first_local, second, second_local, edge));
                break 'outer;
            }
        }
    }
    let (first, first_local, second, second_local, edge) =
        shared.expect("two-by-two quadrilateral mesh has an interior edge");
    let (a, b) = topology.edges[edge];
    let dx = mesh.nodes[b].x - mesh.nodes[a].x;
    let dy = mesh.nodes[b].y - mesh.nodes[a].y;
    let length = (dx * dx + dy * dy).sqrt();
    let canonical_normal = [dy / length, -dx / length];
    let global_t = 0.37;
    let first_t = topology.element_edge_signs(first)[first_local] as f64 * global_t;
    let second_t = topology.element_edge_signs(second)[second_local] as f64 * global_t;
    let first_point = edge_point(first_local, first_t);
    let second_point = edge_point(second_local, second_t);
    let first_tab = space.tabulate(&mesh, first, first_point[0], first_point[1], 0.0);
    let second_tab = space.tabulate(&mesh, second, second_point[0], second_point[1], 0.0);

    for mode in 0..2 {
        let first_dof = 2 * first_local + mode;
        let second_dof = 2 * second_local + mode;
        let first_trace = dot(
            [
                first_tab.values[2 * first_dof],
                first_tab.values[2 * first_dof + 1],
            ],
            canonical_normal,
        );
        let second_trace = dot(
            [
                second_tab.values[2 * second_dof],
                second_tab.values[2 * second_dof + 1],
            ],
            canonical_normal,
        );
        assert!(
            (first_trace - second_trace).abs() < 2e-12,
            "shared edge mode {mode} differs: {first_trace} vs {second_trace}"
        );
        assert_eq!(
            space.local_to_global(&mesh, first)[first_dof],
            space.local_to_global(&mesh, second)[second_dof]
        );
    }
}

#[test]
fn rt1_quadrilateral_reference_basis_test() {
    rt1_quadrilateral_reference_basis_has_identity_moments();
}

#[test]
fn rt1_quadrilateral_reference_divergence_test() {
    rt1_quadrilateral_reference_divergence_matches_finite_difference();
}

#[test]
fn rt1_quadrilateral_topology_test() {
    rt1_quadrilateral_space_preserves_edge_and_cell_topology();
}

#[test]
fn rt1_quadrilateral_orientation_test() {
    rt1_quadrilateral_shared_edge_traces_survive_local_reversal();
}
