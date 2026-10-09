//! Independent 3-D reference checks for the production RT0/H(div) assembly.

use std::collections::{BTreeMap, BTreeSet};

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use sonium_fem::forms::{DivergenceDivergenceForm, VectorMassForm};
use sonium_fem::mesh::{Face, Mesh, MeshTopology, unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};

const TETRA_MOMENTS: [[f64; 4]; 4] = [
    [1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0, 1.0 / 24.0],
    [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0, 1.0 / 24.0],
    [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0, 1.0 / 24.0],
    [1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 6.0],
];

const HEXA_MOMENTS: [[f64; 4]; 4] = [
    [8.0 / 3.0, 0.0, 0.0, 0.0],
    [0.0, 8.0 / 3.0, 0.0, 0.0],
    [0.0, 0.0, 8.0 / 3.0, 0.0],
    [0.0, 0.0, 0.0, 8.0],
];

const TETRA_BASIS: [[[f64; 4]; 3]; 4] = [
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -2.0],
    ],
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, -2.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
    [
        [2.0, 0.0, 0.0, -2.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
];

const HEXA_BASIS: [[[f64; 4]; 3]; 6] = [
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.125, -0.125],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.125, 0.125],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.125, 0.0, -0.125],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.125, 0.0, 0.125],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.125, 0.0, 0.0, -0.125],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.125, 0.0, 0.0, 0.125],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
];

fn aggregate(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut result = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *result.entry((row, column)).or_insert(0.0) += value;
    }
    result
}

fn inner(lhs: &[f64; 4], rhs: &[f64; 4], moments: &[[f64; 4]; 4]) -> f64 {
    lhs.iter()
        .enumerate()
        .map(|(row, &lhs)| {
            rhs.iter()
                .enumerate()
                .map(|(column, &rhs)| lhs * rhs * moments[row][column])
                .sum::<f64>()
        })
        .sum()
}

fn gram_and_det(mesh: &Mesh, element: usize) -> ([[f64; 3]; 3], f64) {
    let element_ref = &mesh.elements[element];
    let shape = evaluate_shape(
        element_ref.element_type,
        PolynomialDegree::P1,
        0.0,
        0.0,
        0.0,
    );
    let coords: Vec<[f64; 3]> = element_ref
        .vertices()
        .iter()
        .map(|&node| {
            let point = mesh.nodes[node];
            [point.x, point.y, point.z]
        })
        .collect();
    let jacobian = Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords);
    let mut gram = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            gram[row][column] = (0..3)
                .map(|physical| jacobian.matrix[physical][row] * jacobian.matrix[physical][column])
                .sum();
        }
    }
    (gram, jacobian.det.abs())
}

fn reference_mass(
    mesh: &Mesh,
    basis: &[[[f64; 4]; 3]],
    moments: &[[f64; 4]; 4],
) -> BTreeMap<(usize, usize), f64> {
    let topology = MeshTopology::build(mesh);
    let space = HdivSpace::new();
    let mut result = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let (gram, determinant) = gram_and_det(mesh, element);
        let signs = topology.element_face_signs(element);
        let dofs = space.local_to_global(mesh, element);
        for (row, row_basis) in basis.iter().enumerate() {
            for (column, column_basis) in basis.iter().enumerate() {
                let mut value = 0.0;
                for component_row in 0..3 {
                    for component_column in 0..3 {
                        value += gram[component_row][component_column]
                            * inner(
                                &row_basis[component_row],
                                &column_basis[component_column],
                                moments,
                            );
                    }
                }
                let global = (dofs[row], dofs[column]);
                *result.entry(global).or_insert(0.0) +=
                    signs[row] as f64 * signs[column] as f64 * value / determinant;
            }
        }
    }
    result
}

fn reference_divergence(mesh: &Mesh, local_divergence: f64) -> BTreeMap<(usize, usize), f64> {
    let topology = MeshTopology::build(mesh);
    let space = HdivSpace::new();
    let mut result = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let (_, determinant) = gram_and_det(mesh, element);
        let signs = topology.element_face_signs(element);
        let dofs = space.local_to_global(mesh, element);
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                *result.entry((global_row, global_column)).or_insert(0.0) +=
                    signs[row] as f64 * signs[column] as f64 * local_divergence / determinant;
            }
        }
    }
    result
}

fn max_error(lhs: &BTreeMap<(usize, usize), f64>, rhs: &BTreeMap<(usize, usize), f64>) -> f64 {
    lhs.keys()
        .chain(rhs.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| (lhs.get(key).unwrap_or(&0.0) - rhs.get(key).unwrap_or(&0.0)).abs())
        .fold(0.0, f64::max)
}

fn assembled_mass(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let matrix = assemble_bilinear_checked(
        &VectorMassForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        7,
    )
    .expect("3-D RT0 mass assembly should pass checked admission");
    aggregate(&matrix)
}

fn assembled_divergence(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let matrix = assemble_bilinear_checked(&DivergenceDivergenceForm, &space, &space, mesh, 5)
        .expect("3-D RT0 divergence assembly should pass checked admission");
    aggregate(&matrix)
}

fn canonical_faces(
    mesh: &Mesh,
    matrix: &BTreeMap<(usize, usize), f64>,
) -> BTreeMap<(Vec<usize>, Vec<usize>), f64> {
    let topology = MeshTopology::build(mesh);
    let key = |index| match topology.face(index) {
        Face::Triangle(nodes) => {
            let mut nodes = nodes.to_vec();
            nodes.sort_unstable();
            nodes
        }
        Face::Quad(nodes) => {
            let mut nodes = nodes.to_vec();
            nodes.sort_unstable();
            nodes
        }
    };
    let mut result = BTreeMap::new();
    for (&(row, column), &value) in matrix {
        *result.entry((key(row), key(column))).or_insert(0.0) += value;
    }
    result
}

#[test]
fn rt0_3d_mass_and_divergence_match_independent_tetra_reference() {
    let mesh = unit_cube_tetrahedra(1);
    assert!(
        max_error(
            &assembled_mass(&mesh),
            &reference_mass(&mesh, &TETRA_BASIS, &TETRA_MOMENTS),
        ) < 5.0e-11
    );
    assert!(
        max_error(
            &assembled_divergence(&mesh),
            &reference_divergence(&mesh, 6.0),
        ) < 5.0e-10
    );
}

#[test]
fn rt0_3d_mass_and_divergence_match_independent_hexa_reference() {
    let mesh = unit_cube_hexahedra(1);
    assert!(
        max_error(
            &assembled_mass(&mesh),
            &reference_mass(&mesh, &HEXA_BASIS, &HEXA_MOMENTS),
        ) < 5.0e-11
    );
    assert!(
        max_error(
            &assembled_divergence(&mesh),
            &reference_divergence(&mesh, 1.0 / 8.0),
        ) < 5.0e-10
    );
}

#[test]
fn rt0_3d_face_orientation_is_invariant_under_tetra_node_permutation() {
    let mesh = unit_cube_tetrahedra(1);
    let mut permuted = mesh.clone();
    for element in &mut permuted.elements {
        element.nodes.rotate_left(1);
    }
    permuted.boundaries.clear();
    permuted.invalidate_caches();

    let original = canonical_faces(&mesh, &assembled_divergence(&mesh));
    let permuted = canonical_faces(&permuted, &assembled_divergence(&permuted));
    assert!(max_error_canonical(&original, &permuted) < 5.0e-10);
}

fn max_error_canonical(
    lhs: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
    rhs: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
) -> f64 {
    lhs.keys()
        .chain(rhs.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| (lhs.get(key).unwrap_or(&0.0) - rhs.get(key).unwrap_or(&0.0)).abs())
        .fold(0.0, f64::max)
}
