//! Checked constant-panel SLFMM admission without launching a simulation.
//!
//! This is a topology/contract gate only. It deliberately stops at the
//! pre-quadrature admission seam so CI can validate fail-closed behavior with
//! negligible CPU cost.

use num_complex::Complex64;
use sonium_bem::core::assembly::slfmm::{
    SlfmmAssemblyError, build_slfmm_system_stabilized_checked, validate_slfmm_constant_panel_layout,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.slfmm-checked-admission.v1";

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

fn run() -> Result<(usize, usize, usize), String> {
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

    let constant = vec![element(vec![0])];
    let clusters = vec![cluster(vec![0])];
    let valid = validate_slfmm_constant_panel_layout(&constant, &nodes, &clusters, 4, 8, 5)
        .map_err(|error| format!("constant-panel admission rejected: {error}"))?;
    if valid != 1 {
        return Err(format!("expected one admitted DOF, got {valid}"));
    }

    let higher_order = vec![element(vec![0, 1, 2])];
    let result = build_slfmm_system_stabilized_checked(
        &higher_order,
        &nodes,
        &clusters,
        &physics,
        4,
        8,
        5,
        -1.0,
        1.0,
    );
    if !matches!(
        result,
        Err(SlfmmAssemblyError::UnsupportedBasis {
            element: 0,
            dofs: 3
        })
    ) {
        return Err("unexpected higher-order admission result".into());
    }

    let invalid_cluster =
        validate_slfmm_constant_panel_layout(&constant, &nodes, &[cluster(vec![1])], 4, 8, 5);
    if !matches!(
        invalid_cluster,
        Err(SlfmmAssemblyError::InvalidClusterElement {
            cluster: 0,
            element: 1,
            available: 1
        })
    ) {
        return Err("unexpected cluster admission result".into());
    }

    let invalid_quadrature =
        validate_slfmm_constant_panel_layout(&constant, &nodes, &clusters, 4, 0, 5);
    if !matches!(
        invalid_quadrature,
        Err(SlfmmAssemblyError::InvalidQuadrature {
            n_theta: 4,
            n_phi: 0,
            n_terms: 5
        })
    ) {
        return Err("unexpected quadrature admission result".into());
    }

    Ok((valid, 3, 1))
}

fn main() {
    match run() {
        Ok((valid_dofs, basis_rejections, topology_rejections)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("valid_constant_dofs", valid_dofs as f64)
                .metric("basis_rejections", basis_rejections as f64)
                .metric("topology_rejections", topology_rejections as f64)
                .minimum("valid_constant_dofs", 1.0)
                .minimum("basis_rejections", 1.0)
                .minimum("topology_rejections", 1.0)
                .reference(
                    "checked constant-panel SLFMM admission and fail-closed higher-order topology",
                )
                .dimension("single Tri3 panel pre-quadrature fixture")
                .formulation("CPU SLFMM constant-panel basis/cluster/quadrature admission")
                .tolerance("all invalid layouts rejected before quadrature or simulation work")
                .seed("bem-slfmm-checked-admission:cpu:v1");
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
                .reference(
                    "checked constant-panel SLFMM admission and fail-closed higher-order topology",
                )
                .dimension("single Tri3 panel pre-quadrature fixture")
                .formulation("CPU SLFMM constant-panel basis/cluster/quadrature admission")
                .tolerance("all invalid layouts rejected before quadrature or simulation work")
                .seed("bem-slfmm-checked-admission:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("SLFMM checked admission failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
