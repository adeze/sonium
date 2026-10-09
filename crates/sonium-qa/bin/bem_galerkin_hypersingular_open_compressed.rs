//! CPU evidence for compressed open-surface Galerkin hypersingular entries.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-hypersingular-open-compressed.v1";

fn panel(
    element_type: ElementType,
    connectivity: Vec<usize>,
    dof_addresses: Vec<usize>,
    center: [f64; 3],
) -> Element {
    let node_count = element_type.num_nodes();
    Element {
        connectivity,
        element_type,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; node_count],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn action_error(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
) -> Result<(f64, usize), String> {
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        2,
        global_dof_count,
    )
    .map_err(|error| format!("dense open assembly failed: {error}"))?;
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        2,
        global_dof_count,
        2,
        1.0e-10,
        64,
    )
    .map_err(|error| format!("compressed open assembly failed: {error}"))?;
    let probe = (0..dense.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.1))
        .collect::<Vec<_>>();
    let dense_action = dense.matrix().matvec(&probe);
    let compressed_action = compressed
        .apply_checked(&probe)
        .map_err(|error| format!("compressed action failed: {error}"))?;
    let error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(expected, actual)| (*expected - *actual).norm())
        .fold(0.0_f64, f64::max);
    if !error.is_finite() {
        return Err("open compressed action error is non-finite".to_owned());
    }
    Ok((error, compressed.compressed_block_count()))
}

fn run() -> Result<([f64; 4], [usize; 4]), String> {
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 290.0, 1.55, false).map_err(|error| error.to_string())?;

    let tri_nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let tri_linear = panel(
        ElementType::Tri3,
        vec![0, 1, 2],
        (0..3).collect(),
        [1.0 / 3.0, 1.0 / 3.0, 0.0],
    );
    let tri_quadratic = panel(
        ElementType::Tri3,
        vec![0, 1, 2],
        (0..6).collect(),
        [1.0 / 3.0, 1.0 / 3.0, 0.0],
    );
    let (tri_linear_error, tri_linear_blocks) = action_error(
        std::slice::from_ref(&tri_linear),
        &tri_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        3,
    )?;
    let (tri_quadratic_error, tri_quadratic_blocks) = action_error(
        std::slice::from_ref(&tri_quadratic),
        &tri_nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        6,
    )?;

    let quad_nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let quad_linear = panel(
        ElementType::Quad4,
        vec![0, 1, 2, 3],
        (0..4).collect(),
        [0.5, 0.5, 0.0],
    );
    let quad_quadratic = panel(
        ElementType::Quad4,
        vec![0, 1, 2, 3],
        (0..9).collect(),
        [0.5, 0.5, 0.0],
    );
    let (quad_linear_error, quad_linear_blocks) = action_error(
        std::slice::from_ref(&quad_linear),
        &quad_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        4,
    )?;
    let (quad_quadratic_error, quad_quadratic_blocks) = action_error(
        std::slice::from_ref(&quad_quadratic),
        &quad_nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        9,
    )?;

    Ok((
        [
            tri_linear_error,
            tri_quadratic_error,
            quad_linear_error,
            quad_quadratic_error,
        ],
        [
            tri_linear_blocks,
            tri_quadratic_blocks,
            quad_linear_blocks,
            quad_quadratic_blocks,
        ],
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((errors, blocks)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:single-open-panel:Tri3-Quad4:P1-P2:compressed")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-hypersingular-open-compressed-v1",
            )
            .dimension("3-D CPU; open Tri3/Quad4 P1/P2 two-trace compressed Müller")
            .metric("tri_linear_action_error", errors[0])
            .metric("tri_quadratic_action_error", errors[1])
            .metric("quad_linear_action_error", errors[2])
            .metric("quad_quadratic_action_error", errors[3])
            .metric("tri_linear_compressed_blocks", blocks[0] as f64)
            .metric("tri_quadratic_compressed_blocks", blocks[1] as f64)
            .metric("quad_linear_compressed_blocks", blocks[2] as f64)
            .metric("quad_quadratic_compressed_blocks", blocks[3] as f64)
            .maximum("tri_linear_action_error", 1.0e-9)
            .maximum("tri_quadratic_action_error", 1.0e-9)
            .maximum("quad_linear_action_error", 1.0e-9)
            .maximum("quad_quadratic_action_error", 1.0e-9)
            .minimum("tri_linear_compressed_blocks", 0.0)
            .minimum("tri_quadratic_compressed_blocks", 0.0)
            .minimum("quad_linear_compressed_blocks", 0.0)
            .minimum("quad_quadratic_compressed_blocks", 0.0)
            .seed("bem-galerkin-hypersingular-open-compressed:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
