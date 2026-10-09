//! CPU-only MeshLang-to-FEM curved Hex27 adaptive H1 reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::refine_elements_high_order_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_h1_p2_state_transfer_with_refinement_history,
    unit_cube_hexahedra,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-curved-p2-hexa-reference.v1";

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn entity_map(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn interpolation_plan(mesh: &sonium_fem::mesh::Mesh) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn linear_field(point: sonium_fem::mesh::Point) -> f64 {
    0.7 + 1.25 * point.x - 0.8 * point.y + 0.55 * point.z
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let mut source_mesh = unit_cube_hexahedra(1).to_p2();
    let curved_node = source_mesh.elements[0].nodes[20];
    source_mesh.nodes[curved_node].z = -0.2;

    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut target_mesh, &[0])
        .map_err(|error| format!("curved Hex27 refinement failed: {error}"))?;
    if target_mesh.num_elements() != 8
        || target_mesh.num_nodes() != 125
        || !target_mesh.validate_connectivity().is_ok()
    {
        return Err("curved Hex27 refinement produced an invalid topology".into());
    }

    let source_entities = entity_map(&source_mesh);
    let target_entities = entity_map(&target_mesh);
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(linear_field)
        .collect::<Vec<_>>();
    let transferred = apply_h1_p2_state_transfer_with_refinement_history::<String, _>(
        &interpolation_plan(&target_mesh),
        &source_mesh,
        &target_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_entity, _action, _source| Ok(0.0),
    )
    .map_err(|error| format!("curved Hex27 H1 transfer failed: {error}"))?;
    let max_error = target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (transferred[&node_entity(node)] - linear_field(*point)).abs())
        .fold(0.0_f64, f64::max);
    if !max_error.is_finite() || max_error > 1.0e-10 {
        return Err(format!("curved Hex27 H1 transfer error {max_error:e}"));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("refined_elements", target_mesh.num_elements() as f64)
        .metric("refined_nodes", target_mesh.num_nodes() as f64)
        .metric("max_error", max_error)
        .maximum("max_error", 1.0e-10)
        .dimension("3-D curved Hex27 adaptive H1")
        .formulation("MeshLang state-transfer plan through parent-isoparametric Q2 refinement")
        .input_checksum("curved-hex27-single-cell:meshlang-h1-v1")
        .reference_checksum("analytic-linear-field:meshlang-curved-p2-hexa-v1")
        .seed("meshlang-fem-h1-curved-p2-hexa:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
