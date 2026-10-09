//! CPU-only MeshLang-to-FEM two-stage P3 H1 adaptive-chain reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::refine_elements_high_order_with_history_checked;
use sonium_fem::mesh::{
    ElementType, Mesh, Point, RefinementTransferDirection,
    apply_h1_p3_state_transfer_with_refinement_chain,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-p3-adaptive-chain-reference.v1";

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

fn entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect()
}

fn plan(mesh: &Mesh) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..mesh.num_nodes())
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let level0 = p3_triangle();
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_high_order_with_history_checked(&mut level1, &[0])
        .map_err(|error| format!("first P3 refinement: {error}"))?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_high_order_with_history_checked(&mut level2, &[0])
        .map_err(|error| format!("second P3 refinement: {error}"))?;
    let meshes = [&level0, &level1, &level2];
    let space0 = H1Space::new(PolynomialDegree::P3);
    let space1 = H1Space::new(PolynomialDegree::P3);
    let space2 = H1Space::new(PolynomialDegree::P3);
    let spaces = [&space0, &space1, &space2];
    let entity_maps = meshes.iter().map(|mesh| entities(mesh)).collect::<Vec<_>>();
    let refinement_plans = IrStateTransferPlanChain::new(vec![plan(&level1), plan(&level2)])
        .map_err(|error| format!("P3 refinement plan chain: {error}"))?;
    let coarse_values = level0
        .nodes
        .iter()
        .copied()
        .map(cubic_value)
        .collect::<Vec<_>>();
    let refined = apply_h1_p3_state_transfer_with_refinement_chain::<&'static str, _>(
        &refinement_plans,
        &meshes,
        &spaces,
        &coarse_values,
        &[history0.clone(), history1.clone()],
        RefinementTransferDirection::Refine,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P3 refinement chain: {error}"))?;
    let refinement_error = level2
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (refined[&entity(node)] - cubic_value(*point)).abs())
        .fold(0.0_f64, f64::max);

    let coarsening_plans = IrStateTransferPlanChain::new(vec![plan(&level1), plan(&level0)])
        .map_err(|error| format!("P3 coarsening plan chain: {error}"))?;
    let fine_values = (0..level2.num_nodes())
        .map(|node| refined[&entity(node)])
        .collect::<Vec<_>>();
    let restored = apply_h1_p3_state_transfer_with_refinement_chain::<&'static str, _>(
        &coarsening_plans,
        &meshes,
        &spaces,
        &fine_values,
        &[history0, history1],
        RefinementTransferDirection::Coarsen,
        &entity_maps,
        |_, _, _, _| Ok(0.0),
    )
    .map_err(|error| format!("P3 coarsening chain: {error}"))?;
    let coarsening_error = level0
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (restored[&entity(node)] - cubic_value(*point)).abs())
        .fold(0.0_f64, f64::max);
    if refinement_error > 1.0e-9 || coarsening_error > 1.0e-9 {
        return Err(format!(
            "P3 chain errors: refine={refinement_error:e}, coarsen={coarsening_error:e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("stages", 2.0)
        .metric("level0_nodes", level0.num_nodes() as f64)
        .metric("level2_nodes", level2.num_nodes() as f64)
        .metric("refinement_error", refinement_error)
        .metric("coarsening_error", coarsening_error)
        .dimension("2-D two-stage P3 triangular H1 MeshLang chain")
        .input_checksum("p3-triangle-two-stage-refine:meshlang-v1")
        .reference_checksum("analytic-cubic-field:meshlang-p3-adaptive-chain-v1")
        .formulation("checked P3 history chain with reverse coarsening")
        .tolerance("both chain directions errors <= 1e-9")
        .seed("meshlang-fem-h1-p3-adaptive-chain:cpu:v1")
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
