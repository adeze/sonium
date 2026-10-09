use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    MlfmmAssemblyError, build_directional_mlfmm_system_stabilized_checked,
    build_mlfmm_system_stabilized_checked, validate_mlfmm_constant_panel_layout,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType, PhysicsParams,
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

fn level_with_elements(element_indices: Vec<usize>) -> ClusterLevel {
    let mut cluster = Cluster::new([1.0 / 3.0, 1.0 / 3.0, 0.0]);
    cluster.element_indices = element_indices;
    cluster.num_elements = cluster.element_indices.len();
    let mut level = ClusterLevel::new(1);
    level.clusters.push(cluster);
    level
}

fn nodes() -> [[f64; 3]; 3] {
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}

#[test]
fn checked_mlfmm_accepts_one_constant_panel() {
    let elements = vec![element(vec![0])];
    let levels = vec![level_with_elements(vec![0])];
    assert_eq!(
        validate_mlfmm_constant_panel_layout(&elements, &nodes(), &levels),
        Ok(1)
    );
}

#[test]
fn checked_mlfmm_rejects_higher_order_incidence() {
    let elements = vec![element(vec![0, 1, 2])];
    let levels = vec![level_with_elements(vec![0])];
    assert!(matches!(
        validate_mlfmm_constant_panel_layout(&elements, &nodes(), &levels),
        Err(MlfmmAssemblyError::UnsupportedBasis {
            element: 0,
            dofs: 3
        })
    ));
}

#[test]
fn checked_mlfmm_accepts_one_constant_quad4_panel() {
    let mut quad = element(vec![0]);
    quad.element_type = ElementType::Quad4;
    quad.connectivity = vec![0, 1, 2, 3];
    quad.node_normals = vec![[0.0, 0.0, 1.0]; 4];
    let nodes = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let levels = vec![level_with_elements(vec![0])];
    assert_eq!(
        validate_mlfmm_constant_panel_layout(&[quad], &nodes, &levels),
        Ok(1)
    );
}

#[test]
fn checked_mlfmm_builders_fail_before_leaf_assembly() {
    let elements = vec![element(vec![0, 1, 2])];
    let levels = vec![level_with_elements(vec![0])];
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    assert!(matches!(
        build_mlfmm_system_stabilized_checked(
            &elements,
            &nodes(),
            levels.clone(),
            &physics,
            -1.0,
            1.0,
        ),
        Err(MlfmmAssemblyError::UnsupportedBasis { .. })
    ));
    assert!(matches!(
        build_directional_mlfmm_system_stabilized_checked(
            &elements,
            &nodes(),
            levels,
            &physics,
            -1.0,
            1.0,
            Default::default(),
        ),
        Err(MlfmmAssemblyError::UnsupportedBasis { .. })
    ));
}

#[test]
fn checked_mlfmm_rejects_invalid_leaf_ownership() {
    let elements = vec![element(vec![0])];
    let levels = vec![level_with_elements(vec![4])];
    assert!(matches!(
        validate_mlfmm_constant_panel_layout(&elements, &nodes(), &levels),
        Err(MlfmmAssemblyError::InvalidClusterElement {
            level: 0,
            cluster: 0,
            element: 4,
            available: 1,
        })
    ));
}

#[test]
fn checked_mlfmm_rejects_duplicate_leaf_ownership() {
    let elements = vec![element(vec![0])];
    let mut level = ClusterLevel::new(2);
    level.clusters.push({
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = vec![0];
        cluster
    });
    level.clusters.push({
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = vec![0];
        cluster
    });
    assert!(matches!(
        validate_mlfmm_constant_panel_layout(&elements, &nodes(), &[level]),
        Err(MlfmmAssemblyError::DuplicateLeafElement {
            element: 0,
            first_cluster: 0,
            second_cluster: 1,
        })
    ));
}

#[test]
fn checked_mlfmm_rejects_invalid_parent_reference() {
    let elements = vec![element(vec![0])];
    let mut root = level_with_elements(vec![0]);
    root.clusters[0].sons = vec![0];
    let mut leaf = level_with_elements(vec![0]);
    leaf.clusters[0].father = Some(1);
    assert!(matches!(
        validate_mlfmm_constant_panel_layout(&elements, &nodes(), &[root, leaf]),
        Err(MlfmmAssemblyError::InvalidHierarchyReference {
            level: 1,
            cluster: 0,
            reference: 1,
            available: 1,
        })
    ));
}

#[test]
fn checked_mlfmm_actions_accept_one_constant_panel() {
    let elements = vec![element(vec![0])];
    let levels = vec![level_with_elements(vec![0])];
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let system =
        build_mlfmm_system_stabilized_checked(&elements, &nodes(), levels, &physics, -1.0, 1.0)
            .expect("one-panel MLFMM fixture should build");
    let input = vec![Complex64::new(1.0, 0.0); system.num_dofs];

    assert!(system.matvec_checked(&input).is_ok());
    assert!(system.matvec_transpose_checked(&input).is_ok());
    assert!(system.apply_hermitian_checked(&input).is_ok());
}
