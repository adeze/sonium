use num_complex::Complex64;
use sonium_bem::core::assembly::slfmm::{
    SlfmmAssemblyError, build_slfmm_system_stabilized_checked, validate_slfmm_constant_panel_layout,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};

fn element(dof_addresses: Vec<usize>) -> Element {
    Element {
        connectivity: vec![0, 1, 2],
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn cluster(element_indices: Vec<usize>) -> Cluster {
    let mut cluster = Cluster::new([1.0 / 3.0, 1.0 / 3.0, 0.0]);
    cluster.element_indices = element_indices;
    cluster.num_elements = cluster.element_indices.len();
    cluster
}

#[test]
fn checked_slfmm_rejects_higher_order_incidence_before_assembly() {
    let elements = vec![element(vec![0, 1, 2])];
    let clusters = vec![cluster(vec![0])];
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let result = build_slfmm_system_stabilized_checked(
        &elements,
        &nodes,
        &clusters,
        &PhysicsParams::new(100.0, 343.0, 1.21, false),
        4,
        8,
        5,
        -1.0,
        1.0,
    );
    let error = match result {
        Ok(_) => panic!("higher-order incidence must fail before quadrature"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SlfmmAssemblyError::UnsupportedBasis {
            element: 0,
            dofs: 3
        }
    ));
}

#[test]
fn checked_slfmm_rejects_invalid_cluster_topology_without_assembly() {
    let elements = vec![element(vec![0])];
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let error =
        validate_slfmm_constant_panel_layout(&elements, &nodes, &[cluster(vec![1])], 4, 8, 5)
            .expect_err("out-of-range cluster element must be rejected");
    assert!(matches!(
        error,
        SlfmmAssemblyError::InvalidClusterElement {
            cluster: 0,
            element: 1,
            available: 1
        }
    ));
}

#[test]
fn checked_slfmm_rejects_zero_sphere_rule_dimensions() {
    let elements = vec![element(vec![0])];
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let error =
        validate_slfmm_constant_panel_layout(&elements, &nodes, &[cluster(vec![0])], 4, 0, 5)
            .expect_err("zero azimuthal quadrature must be rejected");
    assert!(matches!(
        error,
        SlfmmAssemblyError::InvalidQuadrature {
            n_theta: 4,
            n_phi: 0,
            n_terms: 5
        }
    ));
}
