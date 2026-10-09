use num_complex::Complex64;
use sonium_bem::core::mesh::element::basis_values;
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Element, ElementProperty, ElementType, Mesh,
};

fn quad(connectivity: Vec<usize>) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.0, 0.0, 0.0],
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
        elements: vec![quad(vec![0, 1, 2, 3]), quad(vec![1, 4, 5, 2])],
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

fn manufactured_edge_trace(t: f64) -> Complex64 {
    Complex64::new(2.0 - 1.5 * t + 0.75 * t * t, 0.0)
}

#[test]
fn quad4_continuous_quadratic_trace_matches_manufactured_field() {
    let mesh = two_quad_mesh();
    let map = mesh
        .continuous_quadratic_dof_map()
        .expect("valid conforming Quad4 topology");

    // The local Q2 ordering is four vertices, four edge midpoints, then the
    // panel center. Set the three shared-edge DOFs to an exact quadratic
    // trace; all other DOFs stay zero because the edge basis vanishes there.
    let mut coefficients = vec![Complex64::default(); map.num_dofs()];
    coefficients[map.vertex_dof(1).expect("shared edge start vertex")] =
        manufactured_edge_trace(-1.0);
    coefficients[map.vertex_dof(2).expect("shared edge end vertex")] = manufactured_edge_trace(1.0);
    coefficients[map.edge_dof(1, 2).expect("shared edge midpoint")] = manufactured_edge_trace(0.0);

    let first_dofs = map.element_dofs(0).expect("first Quad4 map");
    let second_dofs = map.element_dofs(1).expect("second Quad4 map");

    // In the Q2 reference basis the common edge is first-panel s=1 and
    // second-panel s=-1. The second panel's topological edge is (3, 0), the
    // reverse of the first panel's (1, 2), while its Q2 trace parameter is the
    // same t from the shared start vertex to the shared end vertex.
    for &t in &[-0.85, -0.4, 0.0, 0.35, 0.9] {
        let first_basis = basis_values(ElementType::Quad4, BasisOrder::Quadratic, 1.0, t);
        let second_basis = basis_values(ElementType::Quad4, BasisOrder::Quadratic, -1.0, t);
        let first_trace = first_basis
            .iter()
            .zip(first_dofs)
            .map(|(&basis, &dof)| coefficients[dof] * basis)
            .sum::<Complex64>();
        let second_trace = second_basis
            .iter()
            .zip(second_dofs)
            .map(|(&basis, &dof)| coefficients[dof] * basis)
            .sum::<Complex64>();

        let expected = manufactured_edge_trace(t);

        assert!((first_trace - second_trace).norm() < 1.0e-12);
        assert!((first_trace - expected).norm() < 1.0e-12);
    }
}
