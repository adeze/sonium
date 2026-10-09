//! CPU-only MeshLang-to-FEM two-stage P2 H1 adaptive-chain reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::refine_elements_high_order_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_h1_p2_state_transfer_with_refinement_chain,
    unit_square_triangles,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p2-adaptive-chain-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn value(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect()
}

fn plan(nodes: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..nodes)
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let level0 = unit_square_triangles(1).to_p2();
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_high_order_with_history_checked(&mut level1, &[0])
        .map_err(|error| format!("first P2 history: {error}"))?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_high_order_with_history_checked(&mut level2, &[1])
        .map_err(|error| format!("second P2 history: {error}"))?;
    let meshes = vec![&level0, &level1, &level2];
    let space0 = H1Space::new(PolynomialDegree::P2);
    let space1 = H1Space::new(PolynomialDegree::P2);
    let space2 = H1Space::new(PolynomialDegree::P2);
    let spaces = vec![&space0, &space1, &space2];
    let entity_maps = vec![entities(&level0), entities(&level1), entities(&level2)];
    let histories = vec![history0, history1];
    let refine_plans =
        IrStateTransferPlanChain::new(vec![plan(level1.num_nodes()), plan(level2.num_nodes())])
            .map_err(|error| error.to_string())?;
    let coarse_values = level0.nodes.iter().copied().map(value).collect::<Vec<_>>();
    let refined = apply_h1_p2_state_transfer_with_refinement_chain::<&'static str, _>(
        &refine_plans,
        &meshes,
        &spaces,
        &coarse_values,
        &histories,
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P2 refinement chain: {error}"))?;
    let refinement_error = level2
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (refined[&entity(node)] - value(*point)).abs())
        .fold(0.0, f64::max);

    let coarsen_plans =
        IrStateTransferPlanChain::new(vec![plan(level1.num_nodes()), plan(level0.num_nodes())])
            .map_err(|error| error.to_string())?;
    let fine_values = (0..level2.num_nodes())
        .map(|node| refined[&entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p2_state_transfer_with_refinement_chain::<&'static str, _>(
        &coarsen_plans,
        &meshes,
        &spaces,
        &fine_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P2 coarsening chain: {error}"))?;
    let coarsening_error = level0
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (restored[&entity(node)] - value(*point)).abs())
        .fold(0.0, f64::max);
    if refinement_error > 1.0e-9 || coarsening_error > 1.0e-9 {
        return Err(format!(
            "P2 chain error: refine={refinement_error:e}, coarsen={coarsening_error:e}"
        ));
    }
    Ok(QaResult::passed(CASE_ID)
        .metric("stages", 2.0)
        .metric("level0_nodes", level0.num_nodes() as f64)
        .metric("level2_nodes", level2.num_nodes() as f64)
        .metric("refinement_error", refinement_error)
        .metric("coarsening_error", coarsening_error)
        .metric("orphan_node_range_retained", 1.0)
        .dimension("2-D two-stage P2 triangular H1 MeshLang chain")
        .input_checksum("unit-square-triangles-p2-two-stage-refine:meshlang-v1")
        .reference_checksum("analytic-quadratic-field:meshlang-p2-adaptive-chain-v1")
        .formulation("checked P2 history chain with reverse coarsening and explicit node maps")
        .tolerance("both chain directions errors <= 1e-9")
        .seed("meshlang-fem-h1-p2-adaptive-chain:cpu:v1")
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
