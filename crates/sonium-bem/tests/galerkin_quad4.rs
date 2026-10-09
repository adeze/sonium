//! Separated bilinear/biquadratic Quad4 Galerkin single-layer coverage.

use num_complex::Complex64;
use sonium_bem::core::integration::{
    galerkin_single_layer_regular_triangle, sauter_schwab_galerkin_common_edge,
    sauter_schwab_galerkin_common_vertex,
};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, assemble_galerkin_single_layer_quad4_pair,
    assemble_galerkin_single_layer_with_basis, assemble_galerkin_single_layer_with_global_dofs,
};
use sonium_solvers::dense::DenseMatrix;

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn separated_quads() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let elements = vec![
        quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]),
        quad(vec![4, 5, 6, 7], (9..18).collect(), [0.5, 0.5, 1.0]),
    ];
    (elements, nodes)
}

fn physics() -> PhysicsParams {
    PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics")
}

fn triangle_coords(nodes: &[[f64; 3]], indices: [usize; 3]) -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(
        3,
        3,
        indices.into_iter().flat_map(|index| nodes[index]).collect(),
    )
}

fn assert_finite(matrix: &sonium_solvers::dense::DenseMatrix<Complex64>) {
    assert!(
        matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn quad4_linear_and_quadratic_separated_galerkin_blocks_are_finite_symmetric() {
    let (mut elements, nodes) = separated_quads();
    let linear_dofs = vec![(0..4).collect::<Vec<_>>(), (4..8).collect::<Vec<_>>()];
    for (element, dofs) in elements.iter_mut().zip(linear_dofs) {
        element.dof_addresses = dofs;
    }
    let linear = assemble_galerkin_single_layer_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        6,
    )
    .expect("separated Quad4 linear Galerkin assembly");
    assert_eq!((linear.nrows(), linear.ncols()), (4, 4));
    assert_finite(&linear);
    let linear_reverse = assemble_galerkin_single_layer_quad4_pair(
        &elements[1],
        &elements[0],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        6,
    )
    .expect("reverse separated Quad4 linear Galerkin assembly");
    assert_finite(&linear_reverse);
    for row in 0..4 {
        for column in 0..4 {
            assert!((linear[[row, column]] - linear_reverse[[column, row]]).norm() < 2.0e-10);
        }
    }
    let full_linear = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics(),
        BasisOrder::Linear,
        6,
    )
    .expect("self and separated Quad4 linear assembly");
    assert_eq!((full_linear.nrows(), full_linear.ncols()), (8, 8));
    assert_finite(&full_linear);

    let (elements, nodes) = separated_quads();
    let mut quadratic_elements = elements.clone();
    quadratic_elements[0].dof_addresses = (0..9).collect();
    quadratic_elements[1].dof_addresses = (9..18).collect();
    let quadratic = assemble_galerkin_single_layer_quad4_pair(
        &quadratic_elements[0],
        &quadratic_elements[1],
        &nodes,
        &physics(),
        BasisOrder::Quadratic,
        6,
    )
    .expect("separated Quad4 quadratic Galerkin assembly");
    assert_eq!((quadratic.nrows(), quadratic.ncols()), (9, 9));
    assert_finite(&quadratic);
    let full_quadratic = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics(),
        BasisOrder::Quadratic,
        6,
    )
    .expect("self and separated Quad4 quadratic assembly");
    assert_eq!((full_quadratic.nrows(), full_quadratic.ncols()), (18, 18));
    assert_finite(&full_quadratic);
}

#[test]
fn quad4_global_dof_assembly_matches_panel_local_ordering() {
    let (mut elements, nodes) = separated_quads();
    for (element, base) in elements.iter_mut().zip([0, 4]) {
        element.dof_addresses = (base..base + 4).collect();
    }
    let forward = assemble_galerkin_single_layer_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        5,
    )
    .expect("forward Quad4 block assembly");
    let reverse = assemble_galerkin_single_layer_quad4_pair(
        &elements[1],
        &elements[0],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        5,
    )
    .expect("reverse Quad4 block assembly");
    for row in 0..4 {
        for column in 0..4 {
            assert!((forward[[row, column]] - reverse[[column, row]]).norm() < 2.0e-10);
        }
    }
}

#[test]
fn quad4_touching_pairs_use_mapped_singular_charts() {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let elements = vec![
        quad(vec![0, 1, 2, 3], (0..4).collect(), [0.5, 0.5, 0.0]),
        quad(vec![1, 4, 5, 2], (4..8).collect(), [1.5, 0.5, 0.0]),
    ];
    let forward = assemble_galerkin_single_layer_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        8,
    )
    .expect("common-edge Quad4 block assembly");
    assert_finite(&forward);
    let reverse = assemble_galerkin_single_layer_quad4_pair(
        &elements[1],
        &elements[0],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        8,
    )
    .expect("reverse common-edge Quad4 block assembly");
    assert_finite(&reverse);
    let mut constant_elements = elements.clone();
    constant_elements[0].dof_addresses = vec![0];
    constant_elements[1].dof_addresses = vec![1];
    let constant_forward = assemble_galerkin_single_layer_quad4_pair(
        &constant_elements[0],
        &constant_elements[1],
        &nodes,
        &physics(),
        BasisOrder::Constant,
        8,
    )
    .expect("constant common-edge Quad4 block assembly");
    let constant_reverse = assemble_galerkin_single_layer_quad4_pair(
        &constant_elements[1],
        &constant_elements[0],
        &nodes,
        &physics(),
        BasisOrder::Constant,
        8,
    )
    .expect("reverse constant common-edge Quad4 block assembly");
    assert!((constant_forward[[0, 0]] - constant_reverse[[0, 0]]).norm() < 2.0e-12);
    let source_a = triangle_coords(&nodes, [0, 1, 2]);
    let source_b = triangle_coords(&nodes, [0, 2, 3]);
    let field_a = triangle_coords(&nodes, [1, 4, 5]);
    let field_b = triangle_coords(&nodes, [1, 5, 2]);
    // Touching Quad4 charts raise the requested order by seven to resolve
    // the singular map. Compare the triangle partition at that same order.
    let reference_order = 15;
    let expected_vertex_a = sauter_schwab_galerkin_common_vertex(
        &source_a,
        &field_a,
        &physics(),
        1,
        1,
        reference_order,
    )
    .basis_results[0][0]
        .g_integral;
    let expected_edge =
        sauter_schwab_galerkin_common_edge(&source_a, &field_b, &physics(), 1, 1, reference_order)
            .basis_results[0][0]
            .g_integral;
    let expected_regular =
        galerkin_single_layer_regular_triangle(&source_b, &field_a, &physics(), reference_order);
    let expected_vertex_b = sauter_schwab_galerkin_common_vertex(
        &source_b,
        &field_b,
        &physics(),
        1,
        1,
        reference_order,
    )
    .basis_results[0][0]
        .g_integral;
    let expected = expected_vertex_a + expected_edge + expected_regular + expected_vertex_b;
    assert!(
        (constant_forward[[0, 0]] - expected).norm() < 2.0e-12,
        "mapped {:?}, triangle {:?}, error {:e}",
        constant_forward[[0, 0]],
        expected,
        (constant_forward[[0, 0]] - expected).norm()
    );
    let max_error = (0..4)
        .flat_map(|row| (0..4).map(move |column| (row, column)))
        .map(|(row, column)| (forward[[row, column]] - reverse[[column, row]]).norm())
        .fold(0.0, f64::max);
    assert!(max_error < 2.0e-12, "max reciprocity error {max_error:e}");
    let mut touching_quadratic_elements = elements.clone();
    touching_quadratic_elements[0].dof_addresses = (0..9).collect();
    touching_quadratic_elements[1].dof_addresses = (9..18).collect();
    let quadratic = assemble_galerkin_single_layer_quad4_pair(
        &touching_quadratic_elements[0],
        &touching_quadratic_elements[1],
        &nodes,
        &physics(),
        BasisOrder::Quadratic,
        6,
    )
    .expect("quadratic common-edge Quad4 block assembly");
    assert_finite(&quadratic);
}

#[test]
fn quad4_common_vertex_pairs_use_mapped_singular_charts() {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ];
    let elements = vec![
        quad(vec![0, 1, 2, 3], (0..4).collect(), [0.5, 0.5, 0.0]),
        quad(vec![2, 4, 5, 6], (4..8).collect(), [1.5, 1.5, 0.0]),
    ];
    let forward = assemble_galerkin_single_layer_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        6,
    )
    .expect("common-vertex Quad4 block assembly");
    let reverse = assemble_galerkin_single_layer_quad4_pair(
        &elements[1],
        &elements[0],
        &nodes,
        &physics(),
        BasisOrder::Linear,
        6,
    )
    .expect("reverse common-vertex Quad4 block assembly");
    assert_finite(&forward);
    assert_finite(&reverse);
    let max_error = (0..4)
        .flat_map(|row| (0..4).map(move |column| (row, column)))
        .map(|(row, column)| (forward[[row, column]] - reverse[[column, row]]).norm())
        .fold(0.0, f64::max);
    assert!(max_error < 2.0e-12, "max reciprocity error {max_error:e}");
}

#[test]
fn quad4_global_dof_assembly_matches_panel_local_blocks() {
    let (mut elements, nodes) = separated_quads();
    for (element, base) in elements.iter_mut().zip([0, 4]) {
        element.dof_addresses = (base..base + 4).collect();
    }
    let local = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics(),
        BasisOrder::Linear,
        5,
    )
    .expect("panel-local Quad4 assembly");
    let global = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics(),
        BasisOrder::Linear,
        5,
        8,
    )
    .expect("global Quad4 assembly");
    assert_eq!((global.nrows(), global.ncols()), (8, 8));
    for row in 0..8 {
        for column in 0..8 {
            assert!((global[[row, column]] - local[[row, column]]).norm() < 2.0e-10);
        }
    }
}
