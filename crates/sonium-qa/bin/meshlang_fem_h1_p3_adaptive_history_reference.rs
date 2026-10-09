//! CPU-only MeshLang-to-FEM P3 H1 adaptive history reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_high_order_with_history_checked,
};
use sonium_fem::mesh::{
    ElementType, Mesh, Point, RefinementTransferDirection,
    apply_h1_p3_state_transfer_with_refinement_history,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p3-adaptive-history-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn p3_triangle() -> Mesh {
    let mut mesh = Mesh::new(2);
    let points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
    ];
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

fn cubic_value(point: Point) -> f64 {
    0.4 + 1.1 * point.x - 0.7 * point.y + 0.8 * point.x * point.x - 0.3 * point.x * point.y
        + 0.5 * point.y * point.y
        + 0.2 * point.x * point.x * point.x
        - 0.15 * point.x * point.y * point.y
}

fn all_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect()
}

fn active_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    mesh.elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|node| (entity(node), node))
        .collect()
}

fn interpolation_plan(entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: entities
            .keys()
            .copied()
            .map(|entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let coarse_mesh = p3_triangle();
    let mut fine_mesh = coarse_mesh.clone();
    let (_, history) = refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
        .map_err(|error| format!("P3 refinement: {error}"))?;
    let source_entities = all_entities(&coarse_mesh);
    let fine_entities = all_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_value)
        .collect::<Vec<_>>();
    let fine = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&fine_entities),
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P3 refinement transfer: {error}"))?;
    let refinement_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (fine[&entity(node)] - cubic_value(*point)).abs())
        .fold(0.0_f64, f64::max);

    let refined_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    let coarsening = coarsen_elements_checked(&mut fine_mesh, &history, &[parent_id])
        .map_err(|error| format!("P3 coarsening: {error}"))?;
    let restored_entities = active_entities(&fine_mesh);
    let fine_values = (0..fine_mesh.num_nodes())
        .map(|node| fine[&entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&restored_entities),
        &refined_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &fine_values,
        &history,
        RefinementTransferDirection::Coarsen,
        &fine_entities,
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P3 coarsening transfer: {error}"))?;
    let coarsening_error = restored_entities
        .values()
        .map(|&node| (restored[&entity(node)] - cubic_value(fine_mesh.nodes[node])).abs())
        .fold(0.0_f64, f64::max);
    if refinement_error > 1.0e-9 || coarsening_error > 1.0e-9 {
        return Err(format!(
            "P3 adaptive errors: refine={refinement_error:e}, coarsen={coarsening_error:e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("fine_nodes", fine_values.len() as f64)
        .metric("restored_active_nodes", restored_entities.len() as f64)
        .metric("removed_children", coarsening.removed_elements.len() as f64)
        .metric("refinement_error", refinement_error)
        .metric("coarsening_error", coarsening_error)
        .metric("history_parents", history.parents.len() as f64)
        .dimension("2-D P3 triangular H1 adaptive MeshLang lineage")
        .input_checksum("p3-triangle-refine-coarsen:meshlang-v1")
        .reference_checksum("analytic-cubic-field:meshlang-p3-adaptive-history-v1")
        .formulation("checked P3 history, MeshLang plan transfer, reversible coarsening")
        .tolerance("refinement and coarsening errors <= 1e-9")
        .seed("meshlang-fem-h1-p3-adaptive-history:cpu:v1")
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
