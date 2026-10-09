//! Multi-element explicit P3 H1 conformity and boundary-trace reference.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::spaces::{FunctionSpace, H1Space};

fn two_triangle_p3_mesh() -> Mesh {
    // The square is split along the diagonal (1, 2).  All edge nodes on the
    // shared diagonal use the same global IDs, while each cell keeps its own
    // interior node.  The boundary declarations deliberately contain only
    // corner nodes, matching Mesh::detect_boundaries.
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (1.0, 1.0 / 3.0),
        (1.0, 2.0 / 3.0),
        (1.0 / 3.0, 1.0),
        (2.0 / 3.0, 1.0),
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ];
    let mut mesh = Mesh::new(2);
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }

    // P3 triangle ordering is vertices, two nodes per local edge, then the
    // cell-interior node.  Reverse the shared edge node order in the second
    // element so both elements describe the same physical cubic trace.
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2, 4, 5, 6, 7, 9, 8, 14]);
    mesh.add_element(
        ElementType::Triangle,
        vec![1, 3, 2, 10, 11, 13, 12, 7, 6, 15],
    );

    let boundary = [
        (vec![0, 1], 0),
        (vec![2, 0], 2),
        (vec![1, 3], 0),
        (vec![3, 2], 1),
    ];
    for (nodes, local_idx) in boundary {
        let element_idx = if nodes == vec![1, 3] || nodes == vec![3, 2] {
            1
        } else {
            0
        };
        mesh.add_boundary(nodes, BoundaryType::Dirichlet, 19, element_idx, local_idx);
    }
    mesh
}

#[test]
fn explicit_p3_numbering_shares_cubic_edge_and_recovers_corner_only_trace() {
    let mesh = two_triangle_p3_mesh();
    let space = H1Space::new(PolynomialDegree::P3);

    assert!(space.validate(&mesh).is_ok());
    assert_eq!(space.num_global_dofs(&mesh), 16);

    let first = space.local_to_global(&mesh, 0);
    let second = space.local_to_global(&mesh, 1);
    let shared: Vec<_> = first
        .iter()
        .filter(|dof| second.contains(dof))
        .copied()
        .collect();
    assert_eq!(shared, vec![1, 2, 6, 7]);

    let boundary = space.boundary_dofs(&mesh, 19);
    assert_eq!(boundary, vec![0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]);
    assert!(!boundary.contains(&6));
    assert!(!boundary.contains(&7));
    assert!(!boundary.contains(&14));
    assert!(!boundary.contains(&15));
}
