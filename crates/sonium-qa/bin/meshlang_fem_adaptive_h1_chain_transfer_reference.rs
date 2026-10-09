//! CPU-only reference for the MeshLang adaptive H1 plan-chain consumer.

use std::collections::BTreeMap;
use std::time::Instant;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    RefinementTransferDirection, apply_nodal_state_transfer_with_refinement_chain,
    unit_square_triangles,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "meshlang.fem-adaptive-h1-chain-transfer-reference.v1";

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn entity_map(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect()
}

fn linear_value(point: sonium_fem::mesh::Point) -> f64 {
    1.0 + 2.5 * point.x - 0.75 * point.y
}

fn plan_for_stage(source_nodes: usize, target_nodes: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_nodes)
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: if node < source_nodes {
                    IrStateTransferAction::Preserve
                } else {
                    IrStateTransferAction::Interpolate
                },
            })
            .collect(),
    }
}

fn coarsening_plan(target_nodes: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_nodes)
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0])
        .map_err(|error| error.to_string())?;
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0])
        .map_err(|error| error.to_string())?;
    let meshes = vec![&level0, &level1, &level2];
    let histories = vec![history0, history1];
    let maps = meshes
        .iter()
        .map(|mesh| entity_map(mesh))
        .collect::<Vec<_>>();
    let plans = IrStateTransferPlanChain::new(vec![
        plan_for_stage(level0.num_nodes(), level1.num_nodes()),
        plan_for_stage(level1.num_nodes(), level2.num_nodes()),
    ])
    .map_err(|error| error.to_string())?;
    let source_values = level0
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();
    let result = apply_nodal_state_transfer_with_refinement_chain::<&'static str, _>(
        &plans,
        &meshes,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &maps,
        |_stage, _entity, _action, _previous| Err("unexpected callback"),
    )
    .map_err(|error| error.to_string())?;
    let max_h1_error = level2
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (result[&entity(node)] - linear_value(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_h1_error > 1.0e-12 {
        return Err(format!("H1 chain error {max_h1_error} exceeds tolerance"));
    }

    let coarsening_plans = IrStateTransferPlanChain::new(vec![
        coarsening_plan(level1.num_nodes()),
        coarsening_plan(level0.num_nodes()),
    ])
    .map_err(|error| error.to_string())?;
    let coarsening_values = level2
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();
    let coarsened = apply_nodal_state_transfer_with_refinement_chain::<&'static str, _>(
        &coarsening_plans,
        &meshes,
        &coarsening_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &maps,
        |_stage, _entity, _action, _previous| Err("unexpected coarsening callback"),
    )
    .map_err(|error| error.to_string())?;
    let max_coarsening_error = level0
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (coarsened[&entity(node)] - linear_value(*point)).abs())
        .fold(0.0_f64, f64::max);
    if max_coarsening_error > 1.0e-12 {
        return Err(format!(
            "H1 coarsening error {max_coarsening_error} exceeds tolerance"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("stage_count", plans.stage_count() as f64)
        .metric("target_nodes", level2.num_nodes() as f64)
        .metric("max_h1_error", max_h1_error)
        .metric("max_coarsening_error", max_coarsening_error)
        .metric("rejection_checks", 1.0)
        .reference("linear scalar field over two checked adaptive H1 stages")
        .reference_checksum("analytic/linear-field:meshlang-fem-adaptive-h1-chain-v1")
        .dimension("two parent-child refinement stages")
        .formulation("MeshLang plan chain with checked H1 lineage transfer")
        .tolerance("max H1 nodal error <= 1e-12")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let passed = result.status == QaStatus::Passed;
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
