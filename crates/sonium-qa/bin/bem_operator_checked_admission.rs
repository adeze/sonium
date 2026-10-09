//! Checked dense constant-panel operator admission without a simulator run.
//!
//! This contract intentionally exercises only malformed-layout rejection. The
//! valid-layout path is covered by the BEM integration suite; all cases here
//! must return typed errors before legacy constant-panel indexing or quadrature.

use num_complex::Complex64;
use sonium_bem::core::assembly::{
    BoundaryOperatorAssemblyError, assemble_boundary_operators_with_basis,
    assemble_boundary_operators_with_basis_shared_dofs,
};
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.operator-checked-admission.v1";

fn element(connectivity: Vec<usize>, dof_addresses: Vec<usize>) -> Element {
    Element {
        connectivity,
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

fn run() -> Result<(usize, usize), String> {
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

    let wrong_dof = assemble_boundary_operators_with_basis(
        &[element(vec![0, 1, 2], Vec::new())],
        &nodes,
        &physics,
        BasisOrder::Constant,
    );
    if !matches!(
        wrong_dof,
        Err(BoundaryOperatorAssemblyError::BasisDofCount {
            element: 0,
            actual: 0,
            expected: 1,
            basis: BasisOrder::Constant,
        })
    ) {
        return Err(format!("wrong local DOF was not rejected: {wrong_dof:?}"));
    }

    let out_of_range = assemble_boundary_operators_with_basis(
        &[element(vec![0, 1, nodes.len()], vec![0])],
        &nodes,
        &physics,
        BasisOrder::Constant,
    );
    if !matches!(
        out_of_range,
        Err(BoundaryOperatorAssemblyError::NodeOutOfRange {
            element: 0,
            local: 2,
            node,
            node_count,
        }) if node == nodes.len() && node_count == nodes.len()
    ) {
        return Err(format!(
            "out-of-range node was not rejected: {out_of_range:?}"
        ));
    }

    let duplicate = assemble_boundary_operators_with_basis_shared_dofs(
        &[
            element(vec![0, 1, 2], vec![0]),
            element(vec![0, 1, 2], vec![0]),
        ],
        &nodes,
        &physics,
        BasisOrder::Constant,
    );
    if !matches!(
        duplicate,
        Err(BoundaryOperatorAssemblyError::DuplicateDof {
            first_element: 0,
            second_element: 1,
            dof: 0,
        })
    ) {
        return Err(format!(
            "duplicate constant DOF was not rejected: {duplicate:?}"
        ));
    }

    let non_dense = assemble_boundary_operators_with_basis(
        &[element(vec![0, 1, 2], vec![1])],
        &nodes,
        &physics,
        BasisOrder::Constant,
    );
    if !matches!(
        non_dense,
        Err(BoundaryOperatorAssemblyError::DofLayout {
            element: 0,
            actual: 1,
            expected: 0,
        })
    ) {
        return Err(format!(
            "non-dense constant DOF was not rejected: {non_dense:?}"
        ));
    }

    Ok((4, 1))
}

fn main() {
    let result = match run() {
        Ok((rejected_layouts, duplicate_layouts)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rejected_layouts", rejected_layouts as f64)
            .metric("duplicate_layouts", duplicate_layouts as f64)
            .minimum("rejected_layouts", 3.0)
            .minimum("duplicate_layouts", 1.0)
            .reference("typed dense constant-panel operator admission")
            .dimension("single Tri3 malformed-layout contract")
            .formulation("checked basis-aware boundary operators")
            .tolerance("all malformed layouts fail before legacy indexing or quadrature")
            .seed("bem-operator-checked-admission:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("typed dense constant-panel operator admission")
            .dimension("single Tri3 malformed-layout contract")
            .formulation("checked basis-aware boundary operators")
            .seed("bem-operator-checked-admission:cpu:v1"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
