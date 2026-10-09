//! Checked constant-panel MLFMM admission without launching a simulation.
//!
//! This contract gate stops at hierarchy/topology validation and therefore
//! has negligible CPU cost even when simulator workloads are disabled.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    MlfmmAssemblyError, build_mlfmm_system_stabilized_checked, validate_mlfmm_constant_panel_layout,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.mlfmm-checked-admission.v1";

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

fn run() -> Result<(usize, usize, usize, usize, usize), String> {
    let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let valid = vec![element(vec![0])];
    let valid_levels = vec![level_with_elements(vec![0])];
    let valid_dofs = validate_mlfmm_constant_panel_layout(&valid, &nodes, &valid_levels)
        .map_err(|error| format!("valid MLFMM fixture rejected: {error}"))?;

    let quad4_nodes = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let mut valid_quad4 = element(vec![0]);
    valid_quad4.element_type = ElementType::Quad4;
    valid_quad4.connectivity = vec![0, 1, 2, 3];
    valid_quad4.node_normals = vec![[0.0, 0.0, 1.0]; 4];
    let quad4_dofs =
        validate_mlfmm_constant_panel_layout(&[valid_quad4], &quad4_nodes, &valid_levels)
            .map_err(|error| format!("valid Quad4 MLFMM fixture rejected: {error}"))?;

    let higher_order = vec![element(vec![0, 1, 2])];
    let basis_rejections = usize::from(matches!(
        validate_mlfmm_constant_panel_layout(&higher_order, &nodes, &valid_levels),
        Err(MlfmmAssemblyError::UnsupportedBasis { .. })
    ));

    let invalid_cluster = vec![level_with_elements(vec![1])];
    let topology_rejections = usize::from(matches!(
        validate_mlfmm_constant_panel_layout(&valid, &nodes, &invalid_cluster),
        Err(MlfmmAssemblyError::InvalidClusterElement { .. })
    ));
    if basis_rejections == 0 || topology_rejections == 0 {
        return Err("MLFMM checked admission did not reject invalid fixtures".into());
    }

    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let system =
        build_mlfmm_system_stabilized_checked(&valid, &nodes, valid_levels, &physics, -1.0, 1.0)
            .map_err(|error| format!("valid MLFMM action fixture rejected: {error}"))?;
    let input = vec![Complex64::new(1.0, 0.0); system.num_dofs];
    let checked_actions = usize::from(system.matvec_checked(&input).is_ok())
        + usize::from(system.matvec_transpose_checked(&input).is_ok())
        + usize::from(system.apply_hermitian_checked(&input).is_ok());
    if checked_actions != 3 {
        return Err(format!(
            "MLFMM checked action fixture accepted {checked_actions}/3 actions"
        ));
    }

    Ok((
        valid_dofs,
        quad4_dofs,
        basis_rejections,
        topology_rejections,
        checked_actions,
    ))
}

fn main() {
    match run() {
        Ok((valid_dofs, quad4_dofs, basis_rejections, topology_rejections, checked_actions)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("valid_constant_dofs", valid_dofs as f64)
                .metric("valid_quad4_dofs", quad4_dofs as f64)
                .metric("basis_rejections", basis_rejections as f64)
                .metric("topology_rejections", topology_rejections as f64)
                .metric("checked_actions", checked_actions as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("checked constant-panel MLFMM admission")
                .dimension("single Tri3/Quad4 panel pre-quadrature fixtures")
                .formulation("CPU MLFMM constant-panel basis/cluster/hierarchy admission")
                .seed("bem-mlfmm-checked-admission:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("checked constant-panel MLFMM admission")
                .dimension("single Tri3/Quad4 panel pre-quadrature fixtures")
                .formulation("CPU MLFMM constant-panel admission")
                .seed("bem-mlfmm-checked-admission:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("MLFMM checked admission failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
