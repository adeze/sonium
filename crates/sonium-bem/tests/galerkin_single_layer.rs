use sonium_bem::core::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{AdjacencyType, ElementType, detect_adjacency};
use sonium_bem::core::{
    assembly::common::get_element_coords,
    integration::{
        galerkin_single_layer_regular_triangle_with_basis,
        galerkin_single_layer_self_triangle_with_basis,
    },
};
use sonium_bem::{
    BasisOrder, GalerkinAssemblyError, PhysicsParams, assemble_galerkin_double_layer_with_basis,
    assemble_galerkin_double_layer_with_global_dofs,
    assemble_galerkin_hypersingular_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs, assemble_galerkin_single_layer,
    assemble_galerkin_single_layer_with_basis, assemble_galerkin_single_layer_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs,
};
use sonium_solvers::iterative::GmresConfig;

fn setup_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    (elements, mesh.nodes)
}

fn setup_linear_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>) {
    let (mut elements, nodes) = setup_mesh();
    elements.truncate(1);
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    (elements, nodes)
}

fn setup_two_panel_linear_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>) {
    let (mut elements, nodes) = setup_mesh();
    elements.truncate(2);
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    (elements, nodes)
}

fn setup_two_panel_quadratic_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>) {
    let (mut elements, nodes) = setup_mesh();
    elements.truncate(2);
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = (0..6).map(|local| 6 * index + local).collect();
    }
    (elements, nodes)
}

fn setup_two_panel_common_vertex_linear_mesh()
-> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>) {
    let (elements, nodes) = setup_mesh();
    let first = elements[0].clone();
    let second = elements
        .iter()
        .skip(1)
        .find(|element| {
            matches!(
                detect_adjacency(&first, element),
                AdjacencyType::CommonVertex { .. }
            )
        })
        .expect("icosphere must contain a common-vertex panel pair")
        .clone();
    let mut selected = vec![first, second];
    for (index, element) in selected.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    (selected, nodes)
}

fn setup_two_panel_separated_linear_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>)
{
    let (elements, nodes) = setup_mesh();
    let first = elements[0].clone();
    let second = elements
        .iter()
        .skip(1)
        .find(|element| matches!(detect_adjacency(&first, element), AdjacencyType::Separated))
        .expect("icosphere must contain a separated panel")
        .clone();
    let mut elements = vec![first, second];
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    (elements, nodes)
}

fn setup_conforming_closed_linear_mesh()
-> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>, usize) {
    let (mut elements, nodes) = setup_mesh();
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    for element in &mut elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect();
    }
    (elements, nodes, unique_nodes.len())
}

fn setup_conforming_closed_quadratic_mesh()
-> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>, usize) {
    let (mut elements, nodes) = setup_mesh();
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    let mut unique_edges = elements
        .iter()
        .flat_map(|element| {
            [
                (element.connectivity[0], element.connectivity[1]),
                (element.connectivity[1], element.connectivity[2]),
                (element.connectivity[2], element.connectivity[0]),
            ]
            .into_iter()
            .map(|(first, second)| {
                if first < second {
                    (first, second)
                } else {
                    (second, first)
                }
            })
        })
        .collect::<Vec<_>>();
    unique_edges.sort_unstable();
    unique_edges.dedup();
    let edge_offset = unique_nodes.len();
    for element in &mut elements {
        let vertices = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect::<Vec<_>>();
        let edges = [
            (element.connectivity[0], element.connectivity[1]),
            (element.connectivity[1], element.connectivity[2]),
            (element.connectivity[2], element.connectivity[0]),
        ]
        .into_iter()
        .map(|(first, second)| {
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            edge_offset + unique_edges.binary_search(&key).expect("edge is present")
        });
        element.dof_addresses = vertices.into_iter().chain(edges).collect();
    }
    (elements, nodes, edge_offset + unique_edges.len())
}

fn setup_two_component_conforming_linear_mesh()
-> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>, usize) {
    let first = generate_icosphere_mesh(0.2, 0);
    let second = generate_icosphere_mesh(0.2, 0);
    let node_offset = first.nodes.len();
    let mut nodes = first.nodes;
    nodes.extend(
        second
            .nodes
            .into_iter()
            .map(|point| [point[0] + 1.0, point[1], point[2]]),
    );
    let mut elements = first.elements;
    elements.extend(second.elements.into_iter().map(|mut element| {
        element.connectivity = element
            .connectivity
            .into_iter()
            .map(|node| node + node_offset)
            .collect();
        element
    }));
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    for element in &mut elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect();
    }
    (elements, nodes, unique_nodes.len())
}

fn test_triangle_geometry(
    coords: &sonium_solvers::dense::DenseMatrix<f64>,
) -> ([f64; 3], [[f64; 3]; 3]) {
    let p0 = [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]];
    let p1 = [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]];
    let p2 = [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]];
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let jacobian = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let normal = [
        cross[0] / jacobian,
        cross[1] / jacobian,
        cross[2] / jacobian,
    ];
    let grad = |v: [f64; 3]| {
        [
            (normal[1] * v[2] - normal[2] * v[1]) / jacobian,
            (normal[2] * v[0] - normal[0] * v[2]) / jacobian,
            (normal[0] * v[1] - normal[1] * v[0]) / jacobian,
        ]
    };
    (
        normal,
        [
            grad([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]),
            grad([p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]]),
            grad([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]),
        ],
    )
}

#[test]
fn constant_triangle_galerkin_single_layer_is_finite_and_symmetric() {
    let (elements, nodes) = setup_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let matrix = assemble_galerkin_single_layer(&elements, &nodes, &physics, 4)
        .expect("Galerkin single-layer assembly");
    assert_eq!(matrix.nrows(), elements.len());
    assert_eq!(matrix.ncols(), elements.len());
    assert!(
        matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let maximum_asymmetry = (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| (matrix[[row, column]] - matrix[[column, row]]).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_asymmetry <= 2.0e-8,
        "Galerkin single-layer matrix asymmetry {maximum_asymmetry:.3e}"
    );
}

#[test]
fn galerkin_single_layer_rejects_nonconstant_panel_layout() {
    let (mut elements, nodes) = setup_mesh();
    elements[0].dof_addresses = vec![0, 1];
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
    let error = assemble_galerkin_single_layer(&elements, &nodes, &physics, 4)
        .expect_err("higher-order panel must not be relabelled constant Galerkin");
    assert!(matches!(
        error,
        GalerkinAssemblyError::NonConstantPanel {
            element: 0,
            dofs: 2
        }
    ));
}

#[test]
fn galerkin_single_layer_rejects_zero_quadrature_order() {
    let (elements, nodes) = setup_mesh();
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
    let error = assemble_galerkin_single_layer(&elements, &nodes, &physics, 0)
        .expect_err("zero quadrature order must fail closed");
    assert!(matches!(
        error,
        GalerkinAssemblyError::InvalidQuadratureOrder
    ));
}

#[test]
fn galerkin_single_layer_rejects_quad4_in_tri3_slice() {
    let (mut elements, nodes) = setup_mesh();
    elements[0].element_type = ElementType::Quad4;
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
    let error = assemble_galerkin_single_layer(&elements, &nodes, &physics, 4)
        .expect_err("Quad4 must not enter affine Tri3 Galerkin path");
    assert!(matches!(
        error,
        GalerkinAssemblyError::UnsupportedElementType {
            element: 0,
            element_type: ElementType::Quad4
        }
    ));
}

#[test]
fn linear_triangle_galerkin_single_layer_is_finite_symmetric_and_has_constant_mode() {
    let (elements, nodes) = setup_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let linear = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        8,
    )
    .expect("linear Galerkin single-layer assembly");
    assert_eq!(linear.nrows(), 3 * elements.len());
    assert_eq!(linear.ncols(), 3 * elements.len());
    assert!(
        linear
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let mut maximum_asymmetry = 0.0_f64;
    for row in 0..linear.nrows() {
        for column in 0..linear.ncols() {
            maximum_asymmetry =
                maximum_asymmetry.max((linear[[row, column]] - linear[[column, row]]).norm());
        }
    }
    assert!(maximum_asymmetry < 1.0e-7, "asymmetry={maximum_asymmetry}");

    let (mut constant_elements, _) = setup_linear_mesh();
    for (index, element) in constant_elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let constant = assemble_galerkin_single_layer(&constant_elements, &nodes, &physics, 8)
        .expect("constant Galerkin single-layer assembly");
    for panel_row in 0..constant.nrows() {
        for panel_column in 0..constant.ncols() {
            let mut block_sum = num_complex::Complex64::new(0.0, 0.0);
            for source_basis in 0..3 {
                for field_basis in 0..3 {
                    block_sum +=
                        linear[[3 * panel_row + source_basis, 3 * panel_column + field_basis]];
                }
            }
            assert!((block_sum - constant[[panel_row, panel_column]]).norm() < 2.0e-8);
        }
    }
}

#[test]
fn linear_triangle_galerkin_rejects_wrong_local_dof_count() {
    let (elements, nodes) = setup_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let error = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
    )
    .expect_err("constant panel must not enter linear assembly");
    assert_eq!(
        error,
        GalerkinAssemblyError::InvalidBasisDofs {
            element: 0,
            dofs: 1,
        }
    );
}

#[test]
fn quadratic_triangle_galerkin_rejects_wrong_local_dof_count() {
    let (mut elements, nodes) = setup_mesh();
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let error = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        2,
    )
    .expect_err("quadratic panels must provide six local DOFs");
    assert_eq!(
        error,
        GalerkinAssemblyError::InvalidBasisDofs {
            element: 0,
            dofs: 3,
        }
    );
}

#[test]
fn quadratic_triangle_galerkin_single_layer_is_finite_symmetric_and_has_constant_mode() {
    let (mut elements, nodes) = setup_linear_mesh();
    elements[0].dof_addresses = (0..6).collect();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let quadratic = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        8,
    )
    .expect("quadratic Galerkin single-layer assembly");
    assert_eq!(quadratic.nrows(), 6);
    assert_eq!(quadratic.ncols(), 6);
    assert!(
        quadratic
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let maximum_asymmetry = (0..6)
        .flat_map(|row| (0..6).map(move |column| (row, column)))
        .map(|(row, column)| (quadratic[[row, column]] - quadratic[[column, row]]).norm())
        .fold(0.0_f64, f64::max);
    assert!(maximum_asymmetry < 1.0e-7, "asymmetry={maximum_asymmetry}");

    let (mut constant_elements, _) = setup_linear_mesh();
    constant_elements[0].dof_addresses = vec![0];
    let constant = assemble_galerkin_single_layer(&constant_elements, &nodes, &physics, 8)
        .expect("constant Galerkin single-layer assembly");
    let block_sum = quadratic
        .as_slice()
        .iter()
        .copied()
        .sum::<num_complex::Complex64>();
    assert!((block_sum - constant[[0, 0]]).norm() < 2.0e-8);
}

#[test]
fn quadratic_triangle_galerkin_single_layer_covers_touching_and_separated_charts() {
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    for (mut elements, nodes) in [
        setup_two_panel_linear_mesh(),
        setup_two_panel_common_vertex_linear_mesh(),
        setup_two_panel_separated_linear_mesh(),
    ] {
        for (panel, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = (0..6).map(|local| 6 * panel + local).collect();
        }
        let matrix = assemble_galerkin_single_layer_with_basis(
            &elements,
            &nodes,
            &physics,
            BasisOrder::Quadratic,
            4,
        )
        .expect("quadratic chart assembly");
        assert_eq!(matrix.nrows(), 12);
        assert_eq!(matrix.ncols(), 12);
        assert!(
            matrix
                .as_slice()
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );

        let mut constant_elements = elements.clone();
        for (panel, element) in constant_elements.iter_mut().enumerate() {
            element.dof_addresses = vec![panel];
        }
        let constant = assemble_galerkin_single_layer(&constant_elements, &nodes, &physics, 4)
            .expect("constant chart assembly");
        for panel_row in 0..2 {
            for panel_column in 0..2 {
                let block_sum = (0..6)
                    .flat_map(|source| (0..6).map(move |field| (source, field)))
                    .map(|(source, field)| {
                        matrix[[6 * panel_row + source, 6 * panel_column + field]]
                    })
                    .sum::<num_complex::Complex64>();
                assert!((block_sum - constant[[panel_row, panel_column]]).norm() < 2.0e-7);
            }
        }
    }
}

#[test]
fn conforming_quadratic_galerkin_single_layer_global_layout_matches_panel_local() {
    let (mut elements, nodes) = setup_linear_mesh();
    elements[0].dof_addresses = (0..6).collect();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let local = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .expect("panel-local quadratic assembly");
    let global = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
        6,
    )
    .expect("global quadratic assembly");
    assert_eq!(global.nrows(), local.nrows());
    assert_eq!(global.ncols(), local.ncols());
    for row in 0..global.nrows() {
        for column in 0..global.ncols() {
            assert!((global[[row, column]] - local[[row, column]]).norm() < 1.0e-12);
        }
    }
}

#[test]
fn linear_triangle_galerkin_adjacent_panel_blocks_are_finite() {
    let (elements, nodes) = setup_two_panel_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let matrix = assemble_galerkin_single_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        4,
    )
    .expect("adjacent linear Galerkin assembly");
    assert_eq!(matrix.nrows(), 6);
    assert!(
        matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn conforming_linear_galerkin_accumulates_shared_dofs() {
    let (mut elements, nodes) = setup_two_panel_linear_mesh();
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    for element in &mut elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect();
    }

    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let matrix = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        4,
        unique_nodes.len(),
    )
    .expect("conforming Galerkin assembly");
    assert_eq!(matrix.nrows(), unique_nodes.len());
    assert_eq!(matrix.ncols(), unique_nodes.len());
    assert!(
        matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let mut panel_local = elements.clone();
    for (index, element) in panel_local.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    let local = assemble_galerkin_single_layer_with_basis(
        &panel_local,
        &nodes,
        &physics,
        BasisOrder::Linear,
        4,
    )
    .expect("panel-local Galerkin assembly");
    let zero = matrix[[0, 0]] * 0.0;
    let mut expected = vec![vec![zero; unique_nodes.len()]; unique_nodes.len()];
    for (left_panel, left_element) in elements.iter().enumerate() {
        for (right_panel, right_element) in elements.iter().enumerate() {
            for (left_basis, &left_dof) in left_element.dof_addresses.iter().enumerate() {
                for (right_basis, &right_dof) in right_element.dof_addresses.iter().enumerate() {
                    let value = local[[3 * left_panel + left_basis, 3 * right_panel + right_basis]];
                    expected[left_dof][right_dof] += value;
                }
            }
        }
    }
    for row in 0..unique_nodes.len() {
        for column in 0..unique_nodes.len() {
            let error = (matrix[[row, column]] - expected[row][column]).norm();
            assert!(
                error < 1.0e-12,
                "row={row} column={column} error={error} actual={:?} expected={:?}",
                matrix[[row, column]],
                expected[row][column]
            );
        }
    }
}

#[test]
fn conforming_linear_galerkin_rejects_invalid_global_layout() {
    let (mut elements, nodes) = setup_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    elements[0].dof_addresses[2] = 4;
    let error = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        4,
    )
    .expect_err("out-of-range global DOF must be rejected");
    assert_eq!(
        error,
        GalerkinAssemblyError::GlobalDofOutOfRange {
            element: 0,
            local: 2,
            dof: 4,
            global_dof_count: 4,
        }
    );

    elements[0].dof_addresses = vec![0, 0, 1];
    let error = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        2,
    )
    .expect_err("duplicate local global DOF must be rejected");
    assert_eq!(
        error,
        GalerkinAssemblyError::DuplicateGlobalDof {
            element: 0,
            local: 1,
            dof: 0,
        }
    );
}

#[test]
fn conforming_linear_galerkin_rejects_out_of_range_node() {
    let (mut elements, nodes) = setup_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    elements[0].connectivity[2] = nodes.len();

    let error = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        4,
    )
    .expect_err("out-of-range node must be rejected before coordinate lookup");

    assert!(matches!(
        error,
        GalerkinAssemblyError::NodeOutOfRange {
            element: 0,
            local: 2,
            node,
            node_count,
        } if node == nodes.len() && node_count == nodes.len()
    ));
}

#[test]
fn galerkin_double_layer_flat_self_panel_is_zero() {
    let (elements, nodes) = setup_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        10,
    )
    .expect("flat affine self-panel principal value");

    assert!(
        operators
            .double_layer
            .as_slice()
            .iter()
            .all(|value| value.norm() < 1.0e-14)
    );
    assert!(
        operators
            .adjoint_double_layer
            .as_slice()
            .iter()
            .all(|value| value.norm() < 1.0e-14)
    );
}

#[test]
fn galerkin_double_layer_separated_panels_match_adjoint_transpose() {
    let (elements, nodes) = setup_two_panel_separated_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        8,
    )
    .expect("separated affine principal-value blocks");

    assert!(
        operators
            .double_layer
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let maximum_reciprocity_error = (0..operators.double_layer.nrows())
        .flat_map(|row| (0..operators.double_layer.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.adjoint_double_layer[[row, column]] - operators.double_layer[[column, row]])
                .norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_reciprocity_error < 2.0e-8,
        "K/K' reciprocity error={maximum_reciprocity_error}"
    );

    let global = assemble_galerkin_double_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        8,
        6,
    )
    .expect("global separated affine principal-value blocks");
    for row in 0..6 {
        for column in 0..6 {
            assert!(
                (global.double_layer[[row, column]] - operators.double_layer[[row, column]]).norm()
                    < 1.0e-12
            );
            assert!(
                (global.adjoint_double_layer[[row, column]]
                    - operators.adjoint_double_layer[[row, column]])
                .norm()
                    < 1.0e-12
            );
        }
    }
}

#[test]
fn galerkin_double_layer_common_edge_chart_is_finite_and_reciprocal() {
    let (elements, nodes) = setup_two_panel_linear_mesh();
    assert!(matches!(
        detect_adjacency(&elements[0], &elements[1]),
        AdjacencyType::CommonEdge { .. }
    ));
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        10,
    )
    .expect("common-edge Sauter--Schwab principal-value chart");

    assert!(
        operators
            .double_layer
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        operators
            .adjoint_double_layer
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let coarse = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        4,
    )
    .expect("coarse common-edge chart");
    let convergence_error = (0..operators.double_layer.nrows())
        .flat_map(|row| (0..operators.double_layer.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.double_layer[[row, column]] - coarse.double_layer[[row, column]])
                .norm()
                .max(
                    (operators.adjoint_double_layer[[row, column]]
                        - coarse.adjoint_double_layer[[row, column]])
                    .norm(),
                )
        })
        .fold(0.0_f64, f64::max);
    assert!(
        convergence_error.is_finite() && convergence_error < 5.0e-4,
        "common-edge chart failed to converge: {convergence_error}"
    );
}

#[test]
fn galerkin_double_layer_common_vertex_chart_is_finite_and_reciprocal() {
    let (elements, nodes) = setup_two_panel_common_vertex_linear_mesh();
    assert!(matches!(
        detect_adjacency(&elements[0], &elements[1]),
        AdjacencyType::CommonVertex { .. }
    ));
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        6,
    )
    .expect("common-vertex Sauter--Schwab principal-value chart");

    assert!(
        operators
            .double_layer
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        operators
            .adjoint_double_layer
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let reciprocity_error = (0..operators.double_layer.nrows())
        .flat_map(|row| (0..operators.double_layer.ncols()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.adjoint_double_layer[[row, column]] - operators.double_layer[[column, row]])
                .norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(reciprocity_error < 2.0e-6, "K/K' error={reciprocity_error}");
}

#[test]
fn galerkin_quadratic_double_layer_supports_all_affine_tri3_adjacencies() {
    let (mut self_elements, nodes) = setup_two_panel_quadratic_mesh();
    self_elements.truncate(1);
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let self_operators = assemble_galerkin_double_layer_with_basis(
        &self_elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        8,
    )
    .expect("quadratic flat self-panel principal value");
    assert_eq!(self_operators.double_layer.nrows(), 6);
    assert!(
        self_operators
            .double_layer
            .as_slice()
            .iter()
            .chain(self_operators.adjoint_double_layer.as_slice())
            .all(|value| value.norm() < 1.0e-14)
    );

    let (edge_elements, edge_nodes) = setup_two_panel_quadratic_mesh();
    assert!(matches!(
        detect_adjacency(&edge_elements[0], &edge_elements[1]),
        AdjacencyType::CommonEdge { .. }
    ));
    let edge = assemble_galerkin_double_layer_with_basis(
        &edge_elements,
        &edge_nodes,
        &physics,
        BasisOrder::Quadratic,
        8,
    )
    .expect("quadratic common-edge principal-value chart");
    assert!(
        edge.double_layer
            .as_slice()
            .iter()
            .chain(edge.adjoint_double_layer.as_slice())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let (mut vertex_elements, vertex_nodes) = setup_two_panel_common_vertex_linear_mesh();
    for (index, element) in vertex_elements.iter_mut().enumerate() {
        element.dof_addresses = (0..6).map(|local| 6 * index + local).collect();
    }
    assert!(matches!(
        detect_adjacency(&vertex_elements[0], &vertex_elements[1]),
        AdjacencyType::CommonVertex { .. }
    ));
    let vertex = assemble_galerkin_double_layer_with_basis(
        &vertex_elements,
        &vertex_nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .expect("quadratic common-vertex principal-value chart");
    assert!(
        vertex
            .double_layer
            .as_slice()
            .iter()
            .chain(vertex.adjoint_double_layer.as_slice())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let (separated_elements, separated_nodes) = setup_two_panel_separated_linear_mesh();
    let mut separated_quadratic = separated_elements;
    for (index, element) in separated_quadratic.iter_mut().enumerate() {
        element.dof_addresses = (0..6).map(|local| 6 * index + local).collect();
    }
    let separated = assemble_galerkin_double_layer_with_basis(
        &separated_quadratic,
        &separated_nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .expect("quadratic separated principal-value blocks");
    assert!(
        separated
            .double_layer
            .as_slice()
            .iter()
            .chain(separated.adjoint_double_layer.as_slice())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let reciprocity_error = (0..12)
        .flat_map(|row| (0..12).map(move |column| (row, column)))
        .map(|(row, column)| {
            (separated.adjoint_double_layer[[row, column]] - separated.double_layer[[column, row]])
                .norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        reciprocity_error < 2.0e-7,
        "quadratic K/K' error={reciprocity_error}"
    );
}

#[test]
fn galerkin_quadratic_double_layer_global_dof_layout_matches_panel_local() {
    let (elements, nodes) = setup_two_panel_quadratic_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let local = assemble_galerkin_double_layer_with_basis(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
    )
    .expect("quadratic panel-local assembly");
    let global = assemble_galerkin_double_layer_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        6,
        12,
    )
    .expect("quadratic global-DOF assembly");
    for row in 0..12 {
        for column in 0..12 {
            assert!(
                (global.double_layer[[row, column]] - local.double_layer[[row, column]]).norm()
                    < 1.0e-12
            );
            assert!(
                (global.adjoint_double_layer[[row, column]]
                    - local.adjoint_double_layer[[row, column]])
                .norm()
                    < 1.0e-12
            );
        }
    }
}

#[test]
fn conforming_closed_linear_galerkin_hypersingular_is_finite_and_symmetric() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        6,
        global_dof_count,
    )
    .expect("closed conforming P1 hypersingular assembly");
    assert_eq!(operators.hypersingular.nrows(), global_dof_count);
    assert!(
        operators
            .hypersingular
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let maximum_asymmetry = (0..global_dof_count)
        .flat_map(|row| (0..global_dof_count).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.hypersingular[[row, column]] - operators.hypersingular[[column, row]]).norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(maximum_asymmetry < 5.0e-8, "asymmetry={maximum_asymmetry}");
}

#[test]
fn conforming_closed_quadratic_galerkin_hypersingular_is_finite_symmetric_and_conforming() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_quadratic_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        4,
        global_dof_count,
    )
    .expect("closed conforming P2 hypersingular assembly");
    assert_eq!(operators.hypersingular.nrows(), global_dof_count);
    assert!(
        operators
            .hypersingular
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let maximum_asymmetry = (0..global_dof_count)
        .flat_map(|row| (0..global_dof_count).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.hypersingular[[row, column]] - operators.hypersingular[[column, row]]).norm()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_asymmetry < 5.0e-8,
        "quadratic asymmetry={maximum_asymmetry}"
    );

    let mut malformed = elements;
    malformed[1].dof_addresses[3] = malformed[1].dof_addresses[3].saturating_add(1);
    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &malformed,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        2,
        global_dof_count + 1,
    )
    .expect_err("shared quadratic edge must retain one global DOF");
    assert!(matches!(
        error,
        GalerkinAssemblyError::NonConformingHypersingularEdgeDof { .. }
            | GalerkinAssemblyError::HypersingularGlobalDofEntityConflict { .. }
    ));
}

#[test]
fn galerkin_hypersingular_matches_direct_finite_part_across_closed_components() {
    let (elements, nodes, global_dof_count) = setup_two_component_conforming_linear_mesh();
    let node_offset = nodes.len() / 2;
    let source_node = elements[0].connectivity[0];
    let field_panel_start = elements
        .iter()
        .position(|element| element.connectivity.iter().all(|node| *node >= node_offset))
        .expect("second closed component must be present");
    let field_node = elements[field_panel_start].connectivity[0];
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        6,
        global_dof_count,
    )
    .expect("closed conforming P1 hypersingular assembly");
    let source_dof = elements[0].dof_addresses[0];
    let field_dof = elements[field_panel_start].dof_addresses[0];
    let source_support = elements
        .iter()
        .enumerate()
        .filter_map(|(index, element)| {
            element
                .connectivity
                .iter()
                .position(|node| *node == source_node)
                .map(|local| (index, local))
        })
        .collect::<Vec<_>>();
    let field_support = elements
        .iter()
        .enumerate()
        .filter_map(|(index, element)| {
            element
                .connectivity
                .iter()
                .position(|node| *node == field_node)
                .map(|local| (index, local))
        })
        .collect::<Vec<_>>();
    let mut direct_sum = num_complex::Complex64::new(0.0, 0.0);
    let mut gradient_sum = num_complex::Complex64::new(0.0, 0.0);
    let mut curl_sum = num_complex::Complex64::new(0.0, 0.0);
    let mut mass_sum = num_complex::Complex64::new(0.0, 0.0);
    let mut massless_sum = num_complex::Complex64::new(0.0, 0.0);
    for &(source_index, source_local) in &source_support {
        for &(field_index, field_local) in &field_support {
            let source_coords = get_element_coords(&elements[source_index], &nodes);
            let field_coords = get_element_coords(&elements[field_index], &nodes);
            let result = galerkin_single_layer_regular_triangle_with_basis(
                &source_coords,
                &field_coords,
                &physics,
                3,
                3,
                6,
            );
            direct_sum += result.basis_results[source_local][field_local].d2g_dnxdny_integral;
            let (source_normal, source_gradients) = test_triangle_geometry(&source_coords);
            let (field_normal, field_gradients) = test_triangle_geometry(&field_coords);
            let normal_dot = source_normal[0] * field_normal[0]
                + source_normal[1] * field_normal[1]
                + source_normal[2] * field_normal[2];
            let gradient_dot = source_gradients[source_local][0] * field_gradients[field_local][0]
                + source_gradients[source_local][1] * field_gradients[field_local][1]
                + source_gradients[source_local][2] * field_gradients[field_local][2];
            let constant_single_layer = result
                .basis_results
                .iter()
                .flat_map(|row| row.iter())
                .map(|entry| entry.g_integral)
                .sum::<num_complex::Complex64>();
            gradient_sum += num_complex::Complex64::new(gradient_dot, 0.0) * constant_single_layer;
            let source_curl = [
                source_normal[1] * source_gradients[source_local][2]
                    - source_normal[2] * source_gradients[source_local][1],
                source_normal[2] * source_gradients[source_local][0]
                    - source_normal[0] * source_gradients[source_local][2],
                source_normal[0] * source_gradients[source_local][1]
                    - source_normal[1] * source_gradients[source_local][0],
            ];
            let field_curl = [
                field_normal[1] * field_gradients[field_local][2]
                    - field_normal[2] * field_gradients[field_local][1],
                field_normal[2] * field_gradients[field_local][0]
                    - field_normal[0] * field_gradients[field_local][2],
                field_normal[0] * field_gradients[field_local][1]
                    - field_normal[1] * field_gradients[field_local][0],
            ];
            curl_sum += num_complex::Complex64::new(
                source_curl[0] * field_curl[0]
                    + source_curl[1] * field_curl[1]
                    + source_curl[2] * field_curl[2],
                0.0,
            ) * constant_single_layer;
            mass_sum -= num_complex::Complex64::new(
                physics.wave_number * physics.wave_number * normal_dot,
                0.0,
            ) * result.basis_results[source_local][field_local].g_integral;
            massless_sum +=
                num_complex::Complex64::new(physics.wave_number * physics.wave_number, 0.0)
                    * result.basis_results[source_local][field_local].g_integral;
        }
    }
    let _diagnostics = (gradient_sum, curl_sum, mass_sum, massless_sum);
    let _diagnostics = (gradient_sum, curl_sum, mass_sum, massless_sum);
    let error = (operators.hypersingular[[source_dof, field_dof]] - direct_sum).norm();
    assert!(error < 2.0e-5, "weak/direct W error={error}");
}

#[test]
fn galerkin_hypersingular_accepts_open_and_rejects_nonconforming_surfaces() {
    let (elements, nodes) = setup_linear_mesh();
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let basis_error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Constant,
        2,
        3,
    )
    .expect_err("hypersingular weak form requires continuous P1 basis");
    assert_eq!(
        basis_error,
        GalerkinAssemblyError::UnsupportedHypersingularBasis {
            basis: BasisOrder::Constant,
        }
    );
    let open_operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        3,
    )
    .expect("open panel hypersingular boundary terms");
    let open_coords = get_element_coords(&elements[0], &nodes);
    let direct = galerkin_single_layer_self_triangle_with_basis(&open_coords, &physics, 3, 3, 2);
    for row in 0..3 {
        for column in 0..3 {
            let direct_value = (direct.basis_results[row][column].d2g_dnxdny_integral
                + direct.basis_results[column][row].d2g_dnxdny_integral)
                * 0.5;
            let error = (open_operators.hypersingular[[row, column]] - direct_value).norm();
            assert!(error < 1.0e-12, "open direct finite-part error={error}");
        }
    }

    let mut open_quadratic = elements.clone();
    open_quadratic[0].dof_addresses = (0..6).collect();
    let open_quadratic_operators = assemble_galerkin_hypersingular_with_global_dofs(
        &open_quadratic,
        &nodes,
        &physics,
        BasisOrder::Quadratic,
        2,
        6,
    )
    .expect("open quadratic panel hypersingular boundary terms");
    let direct_quadratic =
        galerkin_single_layer_self_triangle_with_basis(&open_coords, &physics, 6, 6, 2);
    for row in 0..6 {
        for column in 0..6 {
            let direct_value = (direct_quadratic.basis_results[row][column].d2g_dnxdny_integral
                + direct_quadratic.basis_results[column][row].d2g_dnxdny_integral)
                * 0.5;
            let error =
                (open_quadratic_operators.hypersingular[[row, column]] - direct_value).norm();
            assert!(error < 1.0e-12, "open quadratic finite-part error={error}");
        }
    }

    let (elements, nodes) = setup_two_panel_linear_mesh();
    let nonconforming_error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        6,
    )
    .expect_err("shared vertices need one conforming global P1 DOF");
    assert!(matches!(
        nonconforming_error,
        GalerkinAssemblyError::NonConformingHypersingularDof { .. }
    ));
}

#[test]
fn galerkin_open_hypersingular_compressed_matches_dense_for_tri3() {
    let (elements, nodes) = setup_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        3,
    )
    .expect("open Tri3 dense Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = dense
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 2)
        .expect("open Tri3 incident projection");
    assert_eq!(pressure.len(), 3);
    assert_eq!(velocity.len(), 3);
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        3,
        2,
        1.0e-10,
        8,
    )
    .expect("open Tri3 compressed Müller assembly");
    let probe = (0..dense.total_dofs())
        .map(|index| num_complex::Complex64::new(0.2 + index as f64 * 0.01, -0.1))
        .collect::<Vec<_>>();
    let dense_action = dense.matrix().matvec(&probe);
    let compressed_action = compressed
        .apply_checked(&probe)
        .expect("open Tri3 compressed action");
    let error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(expected, actual)| (*expected - *actual).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error < 1.0e-9,
        "open Tri3 compressed action error {error:e}"
    );
}

#[test]
fn galerkin_hypersingular_rejects_inconsistent_closed_winding() {
    let (mut elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    // Reverse one triangle's local edge traversal while preserving its
    // vertex-to-DOF association.  Topology remains conforming, but one shared
    // edge is now traversed in the same direction by both incident panels.
    elements[0].connectivity.swap(0, 1);
    elements[0].dof_addresses.swap(0, 1);
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");
    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &physics,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect_err("inconsistent closed winding must fail before integration");
    assert!(matches!(
        error,
        GalerkinAssemblyError::InconsistentHypersingularOrientation { .. }
    ));
}

#[test]
fn galerkin_muller_transmission_composes_linear_medium_blocks() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("Galerkin Müller assembly");

    assert_eq!(operators.interface_dofs(), global_dof_count);
    assert_eq!(operators.total_dofs(), 2 * global_dof_count);
    assert_eq!(operators.basis_order(), BasisOrder::Linear);
    assert!(
        operators
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let exterior_single_layer = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("exterior V");
    let interior_single_layer = assemble_galerkin_single_layer_with_global_dofs(
        &elements,
        &nodes,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("interior V");
    let exterior_double_layer = assemble_galerkin_double_layer_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("exterior K");
    let interior_double_layer = assemble_galerkin_double_layer_with_global_dofs(
        &elements,
        &nodes,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("interior K");
    let exterior_hypersingular = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("exterior W")
    .hypersingular;
    let interior_hypersingular = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("interior W")
    .hypersingular;

    let i_omega = num_complex::Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let i_omega_rho_exterior = i_omega * exterior.density;
    let i_omega_rho_interior = i_omega * interior.density;
    for &(row, column) in &[(0, 0), (global_dof_count / 2, global_dof_count - 1)] {
        let matrix = operators.matrix();
        let expected_pp = interior_double_layer.double_layer[[row, column]]
            + exterior_double_layer.double_layer[[row, column]];
        let expected_pv = -(i_omega_rho_interior * interior_single_layer[[row, column]]
            + i_omega_rho_exterior * exterior_single_layer[[row, column]]);
        let expected_vp = interior_hypersingular[[row, column]] / interior.density
            + exterior_hypersingular[[row, column]] / exterior.density;
        let expected_vv = -i_omega
            * (interior_double_layer.adjoint_double_layer[[row, column]]
                + exterior_double_layer.adjoint_double_layer[[row, column]]);
        assert!((matrix[[row, column]] - expected_pp).norm() < 1.0e-12);
        assert!((matrix[[row, global_dof_count + column]] - expected_pv).norm() < 1.0e-12);
        assert!((matrix[[global_dof_count + row, column]] - expected_vp).norm() < 1.0e-12);
        assert!(
            (matrix[[global_dof_count + row, global_dof_count + column]] - expected_vv).norm()
                < 1.0e-12
        );
    }

    let pressure_trace = (0..global_dof_count)
        .map(|index| num_complex::Complex64::new(0.1 + index as f64 * 0.01, -0.02))
        .collect::<Vec<_>>();
    let velocity_trace = (0..global_dof_count)
        .map(|index| num_complex::Complex64::new(-0.03, 0.05 + index as f64 * 0.02))
        .collect::<Vec<_>>();
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure_trace, &velocity_trace)
        .expect("finite Galerkin incident RHS");
    for row in 0..global_dof_count {
        let expected_pressure = 2.0
            * (0..global_dof_count)
                .map(|column| {
                    exterior_double_layer.double_layer[[row, column]] * pressure_trace[column]
                        - i_omega_rho_exterior
                            * exterior_single_layer[[row, column]]
                            * velocity_trace[column]
                })
                .sum::<num_complex::Complex64>();
        let expected_velocity = 2.0
            * (0..global_dof_count)
                .map(|column| {
                    exterior_hypersingular[[row, column]] / exterior.density
                        * pressure_trace[column]
                        - i_omega
                            * exterior_double_layer.adjoint_double_layer[[row, column]]
                            * velocity_trace[column]
                })
                .sum::<num_complex::Complex64>();
        assert!((rhs[row] - expected_pressure).norm() < 1.0e-12);
        assert!((rhs[global_dof_count + row] - expected_velocity).norm() < 1.0e-12);
    }
    let solution = operators
        .solve(&rhs, 1.0e-8)
        .expect("certified Galerkin solve");
    assert_eq!(solution.len(), 2 * global_dof_count);
    assert!(
        solution
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(matches!(
        operators.solve(&rhs, 0.0),
        Err(sonium_bem::GalerkinTransmissionSolveError::InvalidTolerance { .. })
    ));
    let short = vec![num_complex::Complex64::new(0.0, 0.0); global_dof_count - 1];
    assert!(matches!(
        operators.incident_rhs_from_trace_coefficients(&short, &velocity_trace),
        Err(sonium_bem::GalerkinTransmissionRhsError::PressureLength { .. })
    ));
    let mut non_finite = pressure_trace.clone();
    non_finite[0] = num_complex::Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        operators.incident_rhs_from_trace_coefficients(&non_finite, &velocity_trace),
        Err(sonium_bem::GalerkinTransmissionRhsError::NonFiniteTrace { .. })
    ));
}

#[test]
fn galerkin_muller_transmission_compresses_checked_actions() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("Galerkin Müller assembly");
    let compressed = operators
        .compress(2, 1.0e-8, 2)
        .expect("checked Galerkin compression");
    assert_eq!(compressed.interface_dofs(), global_dof_count);
    assert_eq!(compressed.total_dofs(), operators.total_dofs());
    assert!(compressed.dense_block_count() > 0);

    let probe = (0..operators.total_dofs())
        .map(|index| {
            num_complex::Complex64::new(0.2 + index as f64 * 0.01, -0.1 + index as f64 * 0.003)
        })
        .collect::<Vec<_>>();
    let dense_action = operators.matrix().matvec(&probe);
    let compressed_action = compressed
        .apply_checked(&probe)
        .expect("finite compressed Galerkin action");
    let action_error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(dense, compressed)| (*dense - *compressed).norm())
        .fold(0.0_f64, f64::max);
    assert!(action_error < 1.0e-7, "action error {action_error:e}");

    let adjoint_action = compressed
        .apply_hermitian_checked(&probe)
        .expect("finite compressed Galerkin Hermitian action");
    assert!(
        adjoint_action
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let forward_batch = operators
        .solve_many(&[probe.clone()], 1.0e-8)
        .expect("checked Galerkin forward batch");
    let adjoint_batch = operators
        .solve_adjoint_many(&[probe.clone()], 1.0e-8)
        .expect("checked Galerkin adjoint batch");
    assert_eq!(forward_batch.len(), 1);
    assert_eq!(adjoint_batch.len(), 1);
    assert_eq!(forward_batch[0].len(), operators.total_dofs());
    assert_eq!(adjoint_batch[0].len(), operators.total_dofs());

    let malformed = sonium_solvers::dense::DenseMatrix::<num_complex::Complex64>::zeros(3, 4);
    assert!(matches!(
        sonium_bem::CompressedTransmissionTbemSystem::from_dense_matrix(
            global_dof_count,
            &malformed,
            2,
            1.0e-8,
            2,
        ),
        Err(sonium_bem::TransmissionCompressionError::InvalidDenseMatrixDimension { .. })
    ));
}

#[test]
fn galerkin_muller_transmission_compresses_from_quadrature_entries() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("dense Galerkin Müller assembly");
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
        2,
        1.0e-8,
        2,
    )
    .expect("entry-native Galerkin compression");
    let probe = (0..dense.total_dofs())
        .map(|index| num_complex::Complex64::new(0.2 + index as f64 * 0.01, -0.1))
        .collect::<Vec<_>>();
    let dense_action = dense.matrix().matvec(&probe);
    let compressed_action = compressed.apply_checked(&probe).expect("compressed action");
    let error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(expected, actual)| (*expected - *actual).norm())
        .fold(0.0_f64, f64::max);
    assert!(error <= 1.0e-7, "entry-native action error {error:e}");
    assert!(compressed.compressed_block_count() > 0);
}

#[test]
fn galerkin_compressed_solution_uses_checked_field_context() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("dense Galerkin Müller assembly");
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
        2,
        1.0e-8,
        2,
    )
    .expect("entry-native Galerkin compression");
    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = operators
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 2)
        .expect("incident projection");
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .expect("incident RHS");
    let mut config = GmresConfig::default();
    config.max_iterations = 80;
    config.restart = 24;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield =
        sonium_solvers::dense::DenseMatrix::zeros(operators.total_dofs(), operators.total_dofs());
    for index in 0..operators.total_dofs() {
        nearfield[[index, index]] = num_complex::Complex64::new(1.0, 0.0);
    }
    let solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("compressed Galerkin solve");
    let points = [[0.0, 0.0, 0.8], [0.31, -0.12, 0.77]];
    let from_compressed = operators
        .evaluate_exterior_field_from_compressed_solution_checked(
            &solution, &elements, &nodes, &points, &incident, &exterior, 2,
        )
        .expect("compressed field evaluation");
    let direct = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            solution.values(),
            &points,
            &incident,
            &exterior,
            2,
        )
        .expect("direct field evaluation");
    let field_error = from_compressed
        .iter()
        .zip(&direct)
        .map(|(actual, expected)| (actual.p_total - expected.p_total).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        field_error <= 1.0e-12,
        "compressed field adapter error {field_error:e}"
    );
    assert!(
        from_compressed
            .iter()
            .all(|value| { value.p_total.re.is_finite() && value.p_total.im.is_finite() })
    );
    assert!(matches!(
        operators.evaluate_exterior_field_from_compressed_solution_checked(
            &solution,
            &elements,
            &nodes,
            &[[f64::NAN, 0.0, 0.8]],
            &incident,
            &exterior,
            2,
        ),
        Err(sonium_bem::GalerkinTransmissionFieldError::FieldEvaluation(
            sonium_bem::core::postprocess::pressure::FieldEvaluationError::NonFiniteEvaluationPoint {
                point: 0
            }
        ))
    ));
}

#[test]
fn galerkin_muller_transmission_rejects_mismatched_media_and_constant_basis() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(430.0, 290.0, 1.55, false).expect("valid interior");
    let error = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect_err("frequency mismatch must fail closed");
    assert!(matches!(
        error,
        sonium_bem::GalerkinTransmissionAssemblyError::FrequencyMismatch { .. }
    ));

    let error = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &exterior,
        BasisOrder::Constant,
        2,
        global_dof_count,
    )
    .expect_err("constant traces are not a weak hypersingular basis");
    assert!(matches!(
        error,
        sonium_bem::GalerkinTransmissionAssemblyError::UnsupportedBasis {
            basis: BasisOrder::Constant
        }
    ));
}

#[test]
fn galerkin_muller_transmission_supports_conforming_quadratic_traces() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_quadratic_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        1,
        global_dof_count,
    )
    .expect("quadratic Galerkin Müller assembly");
    assert_eq!(operators.interface_dofs(), global_dof_count);
    assert_eq!(operators.total_dofs(), 2 * global_dof_count);
    assert_eq!(operators.basis_order(), BasisOrder::Quadratic);
    assert!(
        operators
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn galerkin_muller_projects_incident_plane_wave_traces() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("Galerkin Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let (pressure, normal_velocity) = operators
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 3)
        .expect("incident trace L2 projection");
    assert_eq!(pressure.len(), global_dof_count);
    assert_eq!(normal_velocity.len(), global_dof_count);
    assert!(
        pressure
            .iter()
            .chain(normal_velocity.iter())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(normal_velocity.iter().any(|value| value.norm() > 1.0e-12));

    let projected_rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure, &normal_velocity)
        .expect("projected trace RHS");
    let automatic_rhs = operators
        .incident_rhs_from_field(&elements, &nodes, &incident, &exterior, 3)
        .expect("automatic projected RHS");
    assert_eq!(projected_rhs.len(), operators.total_dofs());
    assert!(
        projected_rhs
            .iter()
            .zip(automatic_rhs.iter())
            .all(|(lhs, rhs)| (*lhs - *rhs).norm() < 1.0e-12)
    );
}

#[test]
fn galerkin_muller_projection_rejects_bad_quadrature_and_source_singularity() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("Galerkin Müller assembly");
    let plane_wave = IncidentField::plane_wave_z();
    assert!(matches!(
        operators
            .project_incident_trace_coefficients(&elements, &nodes, &plane_wave, &exterior, 0,),
        Err(sonium_bem::GalerkinTransmissionProjectionError::InvalidQuadratureOrder { .. })
    ));
    assert!(matches!(
        operators.project_incident_trace_coefficients(
            &elements,
            &nodes,
            &plane_wave,
            &exterior,
            sonium_bem::core::integration::N_GAUORDER + 1,
        ),
        Err(sonium_bem::GalerkinTransmissionProjectionError::InvalidQuadratureOrder { .. })
    ));

    let first = &elements[0];
    let source = first
        .connectivity
        .iter()
        .map(|&node| nodes[node])
        .reduce(|a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .map(|sum| [sum[0] / 3.0, sum[1] / 3.0, sum[2] / 3.0])
        .expect("triangle vertices");
    let point_source = IncidentField::PointSource {
        position: source,
        strength: num_complex::Complex64::new(1.0, 0.0),
    };
    assert!(matches!(
        operators.project_incident_trace_coefficients(
            &elements,
            &nodes,
            &point_source,
            &exterior,
            1,
        ),
        Err(sonium_bem::GalerkinTransmissionProjectionError::SourceSingularity { .. })
    ));

    let non_finite_source = IncidentField::PlaneWave {
        direction: [f64::NAN, 0.0, 1.0],
        amplitude: num_complex::Complex64::new(1.0, 0.0),
    };
    assert!(matches!(
        operators.project_incident_trace_coefficients(
            &elements,
            &nodes,
            &non_finite_source,
            &exterior,
            1,
        ),
        Err(sonium_bem::GalerkinTransmissionProjectionError::NonFiniteSource { .. })
    ));

    let mut non_finite_nodes = nodes.clone();
    non_finite_nodes[0][0] = f64::NAN;
    assert!(matches!(
        operators.project_incident_trace_coefficients(
            &elements,
            &non_finite_nodes,
            &plane_wave,
            &exterior,
            1,
        ),
        Err(sonium_bem::GalerkinTransmissionProjectionError::NonFiniteGeometry { .. })
    ));
}

#[test]
fn galerkin_muller_projects_conforming_quadratic_traces() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_quadratic_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        1,
        global_dof_count,
    )
    .expect("quadratic Galerkin Müller assembly");
    let (pressure, normal_velocity) = operators
        .project_incident_trace_coefficients(
            &elements,
            &nodes,
            &IncidentField::plane_wave_z(),
            &exterior,
            2,
        )
        .expect("quadratic incident trace projection");
    assert_eq!(pressure.len(), global_dof_count);
    assert_eq!(normal_velocity.len(), global_dof_count);
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure, &normal_velocity)
        .expect("quadratic projected RHS");
    assert_eq!(rhs.len(), operators.total_dofs());
    assert!(
        rhs.iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn galerkin_muller_evaluates_checked_exterior_field_from_solution() {
    let (elements, nodes, global_dof_count) = setup_conforming_closed_linear_mesh();
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .expect("Galerkin Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let rhs = operators
        .incident_rhs_from_field(&elements, &nodes, &incident, &exterior, 3)
        .expect("projected incident RHS");
    let solution = operators.solve(&rhs, 1.0e-8).expect("checked solve");
    let field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &solution,
            &[[0.0, 0.0, 0.8], [0.31, -0.12, 0.77]],
            &incident,
            &exterior,
            3,
        )
        .expect("checked Galerkin field evaluation");
    assert_eq!(field.len(), 2);
    assert!(field.iter().all(|value| {
        value.p_incident.re.is_finite()
            && value.p_incident.im.is_finite()
            && value.p_scattered.re.is_finite()
            && value.p_scattered.im.is_finite()
            && value.p_total.re.is_finite()
            && value.p_total.im.is_finite()
    }));

    let short_solution = vec![num_complex::Complex64::new(0.0, 0.0); operators.total_dofs() - 1];
    assert!(matches!(
        operators.evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &short_solution,
            &[[0.0, 0.0, 0.8]],
            &incident,
            &exterior,
            3,
        ),
        Err(sonium_bem::GalerkinTransmissionFieldError::SolutionLength { .. })
    ));

    let mut non_finite_solution = solution.clone();
    non_finite_solution[0] = num_complex::Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        operators.evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &non_finite_solution,
            &[[0.0, 0.0, 0.8]],
            &incident,
            &exterior,
            3,
        ),
        Err(sonium_bem::GalerkinTransmissionFieldError::NonFiniteSolution { .. })
    ));

    assert!(matches!(
        operators.evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &solution,
            &[[f64::NAN, 0.0, 0.8]],
            &incident,
            &exterior,
            3,
        ),
        Err(sonium_bem::GalerkinTransmissionFieldError::FieldEvaluation(
            sonium_bem::core::postprocess::pressure::FieldEvaluationError::
                NonFiniteEvaluationPoint { point: 0 }
        ))
    ));
}
