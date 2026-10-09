//! CPU-only MeshLang-to-FEM P2 H1 adaptive history reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_high_order_with_history_checked,
};
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_h1_p2_state_transfer_with_refinement_history,
    unit_square_triangles,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p2-adaptive-history-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn quadratic_value(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn all_entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect()
}

fn active_entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    mesh.elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|node| (entity(node), node))
        .collect()
}

fn interpolation_plan(nodes: impl IntoIterator<Item = usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: nodes
            .into_iter()
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn max_error(
    mesh: &sonium_fem::mesh::Mesh,
    values: &BTreeMap<EntityId, f64>,
    entities: &BTreeMap<EntityId, usize>,
) -> f64 {
    entities
        .iter()
        .map(|(id, &node)| (values[id] - quadratic_value(mesh.nodes[node])).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let coarse_mesh = unit_square_triangles(1).to_p2();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .map_err(|error| format!("P2 refinement with history: {error}"))?;
    let parent_id = history
        .parents
        .first()
        .ok_or_else(|| "refinement history did not retain a parent".to_owned())?
        .element
        .id;
    let coarse_entities = all_entities(&coarse_mesh);
    let fine_entities = all_entities(&fine_mesh);
    let coarse_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(quadratic_value)
        .collect::<Vec<_>>();
    let plan = interpolation_plan(fine_entities.values().copied());
    let fine_values = apply_h1_p2_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &coarse_values,
        &history,
        RefinementTransferDirection::Refine,
        &coarse_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P2 refinement transfer: {error}"))?;
    let fine_error = max_error(&fine_mesh, &fine_values, &fine_entities);

    let mut restored_mesh = fine_mesh.clone();
    let coarsening = coarsen_elements_checked(&mut restored_mesh, &history, &[parent_id])
        .map_err(|error| format!("P2 coarsening: {error}"))?;
    let restored_entities = active_entities(&restored_mesh);
    let restored_plan = interpolation_plan(restored_entities.values().copied());
    let restored_values = apply_h1_p2_state_transfer_with_refinement_history::<&'static str, _>(
        &restored_plan,
        &fine_mesh,
        &restored_mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P2),
        &(0..fine_mesh.num_nodes())
            .map(|node| fine_values[&entity(node)])
            .collect::<Vec<_>>(),
        &history,
        RefinementTransferDirection::Coarsen,
        &fine_entities,
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P2 coarsening transfer: {error}"))?;
    let restored_error = max_error(&restored_mesh, &restored_values, &restored_entities);
    if fine_error > 1.0e-9 || restored_error > 1.0e-9 {
        return Err(format!(
            "quadratic adaptive transfer error: refine={fine_error:e}, coarsen={restored_error:e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("fine_nodes", fine_mesh.num_nodes() as f64)
        .metric("restored_active_nodes", restored_entities.len() as f64)
        .metric("removed_children", coarsening.removed_elements.len() as f64)
        .metric("refinement_error", fine_error)
        .metric("coarsening_error", restored_error)
        .metric("history_parents", history.parents.len() as f64)
        .dimension("2-D P2 triangular H1 adaptive MeshLang lineage")
        .input_checksum("unit-square-triangles-p2-refine-coarsen:meshlang-v1")
        .reference_checksum("analytic-quadratic-field:meshlang-p2-adaptive-history-v1")
        .formulation(
            "checked high-order history, MeshLang plan transfer, and reversible coarsening",
        )
        .tolerance("refinement and coarsening errors <= 1e-9")
        .seed("meshlang-fem-h1-p2-adaptive-history:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
