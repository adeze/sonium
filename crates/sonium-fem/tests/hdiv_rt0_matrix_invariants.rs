//! CPU evidence for the production lowest-order H(div)/RT0 path.
//!
//! The reference below does not call the RT0 basis or the FEM quadrature.  It
//! integrates the affine reference-triangle basis from its polynomial moments,
//! then applies the physical affine map and topology signs independently.
//! This makes the checked production assembly answerable against an algebraic
//! reference while also exercising shared-edge orientation.

use std::collections::BTreeMap;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::forms::VectorMassForm;
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};

const AFFINE_MOMENTS: [[f64; 3]; 3] = [
    [1.0 / 12.0, 1.0 / 24.0, 1.0 / 6.0],
    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0],
];

// Coefficients of the independent reference RT0 basis in (xi, eta, 1):
// [xi, eta - 1], [xi, eta], and [xi - 1, eta].
const RT0_TRIANGLE_COEFFICIENTS: [[[f64; 3]; 2]; 3] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
];

fn aggregate_triplets(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut aggregate = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *aggregate.entry((row, column)).or_insert(0.0) += value;
    }
    aggregate
}

fn affine_component_inner(lhs: &[f64; 3], rhs: &[f64; 3]) -> f64 {
    lhs.iter()
        .enumerate()
        .map(|(row, &lhs_value)| {
            rhs.iter()
                .enumerate()
                .map(|(column, &rhs_value)| lhs_value * rhs_value * AFFINE_MOMENTS[row][column])
                .sum::<f64>()
        })
        .sum()
}

fn reference_local_mass(mesh: &Mesh, element: usize, use_topology_signs: bool) -> [[f64; 3]; 3] {
    let vertices = mesh.elements[element].vertices();
    let points: Vec<_> = vertices.iter().map(|&vertex| mesh.nodes[vertex]).collect();
    let jacobian = [
        [points[1].x - points[0].x, points[2].x - points[0].x],
        [points[1].y - points[0].y, points[2].y - points[0].y],
    ];
    let determinant = (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]).abs();
    let metric = [
        [
            jacobian[0][0] * jacobian[0][0] + jacobian[1][0] * jacobian[1][0],
            jacobian[0][0] * jacobian[0][1] + jacobian[1][0] * jacobian[1][1],
        ],
        [
            jacobian[0][1] * jacobian[0][0] + jacobian[1][1] * jacobian[1][0],
            jacobian[0][1] * jacobian[0][1] + jacobian[1][1] * jacobian[1][1],
        ],
    ];
    let topology = MeshTopology::build(mesh);
    let signs = topology.element_edge_signs(element);

    let mut local = [[0.0; 3]; 3];
    for (row, row_coefficients) in RT0_TRIANGLE_COEFFICIENTS.iter().enumerate() {
        for (column, column_coefficients) in RT0_TRIANGLE_COEFFICIENTS.iter().enumerate() {
            let mut value = 0.0;
            for (row_component, &row_coefficients) in row_coefficients.iter().enumerate() {
                for (column_component, &column_coefficients) in
                    column_coefficients.iter().enumerate()
                {
                    value += metric[row_component][column_component]
                        * affine_component_inner(&row_coefficients, &column_coefficients);
                }
            }
            let row_sign = if use_topology_signs {
                signs[row] as f64
            } else {
                1.0
            };
            let column_sign = if use_topology_signs {
                signs[column] as f64
            } else {
                1.0
            };
            local[row][column] = row_sign * column_sign * value / determinant;
        }
    }
    local
}

fn reference_global_mass(mesh: &Mesh, use_topology_signs: bool) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let local = reference_local_mass(mesh, element, use_topology_signs);
        let dofs = space.local_to_global(mesh, element);
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                *reference.entry((global_row, global_column)).or_insert(0.0) += local[row][column];
            }
        }
    }
    reference
}

fn assembled_global_mass(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    space
        .validate(mesh)
        .expect("RT0 H(div) DOF map must pass checked admission");
    let assembled = assemble_bilinear_checked(
        &VectorMassForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        7,
    )
    .expect("RT0 vector mass assembly must pass checked admission");
    aggregate_triplets(&assembled)
}

fn max_matrix_error(
    lhs: &BTreeMap<(usize, usize), f64>,
    rhs: &BTreeMap<(usize, usize), f64>,
) -> f64 {
    lhs.keys()
        .chain(rhs.keys())
        .map(|key| {
            (lhs.get(key).copied().unwrap_or(0.0) - rhs.get(key).copied().unwrap_or(0.0)).abs()
        })
        .fold(0.0, f64::max)
}

fn triangle_mesh(permutation: [usize; 3]) -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point {
        x: 0.15,
        y: -0.2,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: 1.4,
        y: 0.25,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: 0.35,
        y: 1.3,
        z: 0.0,
    });
    mesh.add_element(ElementType::Triangle, permutation.to_vec());
    mesh
}

fn shared_triangle_mesh_with_reversed_winding() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [(0.15, -0.2), (1.4, 0.25), (0.35, 1.3), (1.8, 1.7)] {
        mesh.add_node(Point { x, y, z: 0.0 });
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![1, 3, 2]);
    mesh.elements[1].nodes.reverse();
    mesh.invalidate_caches();
    mesh
}

#[test]
fn rt0_checked_mass_matches_algebraic_reference_for_all_vertex_permutations() {
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mesh = triangle_mesh(permutation);
        let actual = assembled_global_mass(&mesh);
        let expected = reference_global_mass(&mesh, true);
        let error = max_matrix_error(&actual, &expected);
        assert!(
            error < 2.0e-11,
            "RT0 mass reference error for permutation {permutation:?}: {error:.3e}"
        );

        // Negative control: dropping orientation signs is materially wrong,
        // so this threshold protects the orientation part of the contract.
        let orientation_blind = reference_global_mass(&mesh, false);
        let orientation_error = max_matrix_error(&actual, &orientation_blind);
        assert!(
            orientation_error > 1.0e-3,
            "orientation-blind RT0 reference unexpectedly passed: {orientation_error:.3e}"
        );
    }
}

#[test]
fn rt0_checked_mass_preserves_shared_edge_orientation_under_reversed_winding() {
    let mesh = shared_triangle_mesh_with_reversed_winding();

    let actual = assembled_global_mass(&mesh);
    let expected = reference_global_mass(&mesh, true);
    let error = max_matrix_error(&actual, &expected);
    assert!(
        error < 2.0e-11,
        "shared-edge RT0 mass reference error after winding reversal: {error:.3e}"
    );

    let orientation_blind = reference_global_mass(&mesh, false);
    let orientation_error = max_matrix_error(&actual, &orientation_blind);
    assert!(
        orientation_error > 1.0e-3,
        "shared-edge orientation loss was not detected: {orientation_error:.3e}"
    );
}
