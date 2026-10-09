use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_continuous_quadratic_stabilized;
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, Mesh, PhysicsParams,
};

fn quad(connectivity: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: Vec::new(),
    }
}

fn two_quad_mesh() -> Mesh {
    Mesh {
        nodes: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        elements: vec![
            quad(vec![0, 1, 2, 3], [0.5, 0.5, 0.0]),
            // The second panel reverses the shared edge locally: its edge
            // (3,0) is the first panel's edge (1,2).
            quad(vec![1, 4, 5, 2], [1.5, 0.5, 0.0]),
        ],
        external_node_numbers: Vec::new(),
        external_element_numbers: Vec::new(),
        num_boundary_nodes: 6,
        num_evaluation_nodes: 0,
        num_boundary_elements: 2,
        num_evaluation_elements: 0,
        symmetry_planes: [0; 3],
        symmetry_coordinates: [0.0; 3],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    }
}

#[test]
fn quad4_continuous_quadratic_map_shares_vertices_and_edges() {
    let mesh = two_quad_mesh();
    let map = mesh
        .continuous_quadratic_dof_map()
        .expect("valid conforming Quad4 topology");
    let first = map.element_dofs(0).expect("first Quad4 map");
    let second = map.element_dofs(1).expect("second Quad4 map");

    // Six vertices + seven unique edges + two panel-interior centers.
    assert_eq!(map.num_dofs(), 15);
    assert_eq!(first.len(), 9);
    assert_eq!(second.len(), 9);

    assert_eq!(first[1], second[0]);
    assert_eq!(first[2], second[3]);
    assert_eq!(first[5], second[7]);
    assert_eq!(map.edge_dof(1, 2), map.edge_dof(2, 1));
    assert_eq!(map.interior_dof(0), Some(first[8]));
    assert_eq!(map.interior_dof(1), Some(second[8]));
    assert_ne!(first[8], second[8]);
}

#[test]
fn quad4_continuous_quadratic_tbem_assembles_global_system() {
    let mesh = two_quad_mesh();
    let map = mesh
        .continuous_quadratic_dof_map()
        .expect("valid conforming Quad4 topology");
    let mut elements = mesh.elements.clone();
    for (element_index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = map
            .element_dofs(element_index)
            .expect("active element map")
            .to_vec();
    }

    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let system = build_tbem_system_continuous_quadratic_stabilized(
        &elements,
        &mesh.nodes,
        &physics,
        1.0,
        0.2,
    );

    assert_eq!(system.matrix.shape(), (15, 15));
    assert_eq!(system.rhs.len(), 15);
    for row in 0..15 {
        assert!(system.matrix[[row, row]].norm() > 0.0);
        for column in 0..15 {
            assert!(system.matrix[[row, column]].re.is_finite());
            assert!(system.matrix[[row, column]].im.is_finite());
        }
    }
}
