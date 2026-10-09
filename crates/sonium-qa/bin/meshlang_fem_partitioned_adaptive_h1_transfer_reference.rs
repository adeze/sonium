//! CPU-only ownership/ghost split reference for adaptive MeshLang H1 transfer.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    Partition, PartitionGhostLayout, RefinementTransferDirection,
    apply_partitioned_nodal_state_transfer_with_refinement_history, build_element_graph,
    refinement::refine_elements_with_history_checked, synchronize_partitioned_nodal_state,
    unit_square_triangles,
};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-partitioned-adaptive-h1-transfer-reference.v1";

fn entity(node: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, node as u64)
}

fn value(point: sonium_fem::mesh::Point) -> f64 {
    1.0 + 2.0 * point.x - 0.5 * point.y
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target_mesh, &[0])
        .map_err(|error| error.to_string())?;

    let source_partition =
        Partition::try_from_ids(vec![0, 1], 2).map_err(|error| error.to_string())?;
    let source_graph = build_element_graph(&source_mesh);
    let source_layout = source_partition
        .ghost_layout(&source_mesh, &source_graph, 1)
        .map_err(|error| error.to_string())?;
    let target_partition = Partition::try_from_ids(
        (0..target_mesh.elements.len())
            // Put the untouched coarse element on partition 0 and all
            // refined children on partition 1 so both owner and ghost
            // classes are represented in the local target layout.
            .map(|element| usize::from(element + 1 != target_mesh.elements.len()))
            .collect(),
        2,
    )
    .map_err(|error| error.to_string())?;
    let target_graph = build_element_graph(&target_mesh);
    let target_layout = target_partition
        .ghost_layout(&target_mesh, &target_graph, 1)
        .map_err(|error| error.to_string())?;

    let source_entities = (0..source_mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect::<BTreeMap<_, _>>();
    let target_entities = (0..target_mesh.num_nodes())
        .map(|node| (entity(node), node))
        .collect::<BTreeMap<_, _>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_mesh.num_nodes())
            .map(|node| IrStateTransferEntry {
                entity: entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(value)
        .collect::<Vec<_>>();
    let result = apply_partitioned_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        &source_layout,
        &target_layout,
        |_, _, _| Err("unexpected application callback"),
    )
    .map_err(|error| error.to_string())?;
    if result.owned.is_empty() || result.ghost.is_empty() {
        return Err("partitioned result must contain owned and ghost values".into());
    }
    let max_error = result
        .owned
        .iter()
        .chain(result.ghost.iter())
        .map(|(id, actual)| {
            let node = target_entities[id];
            (actual - value(target_mesh.nodes[node])).abs()
        })
        .fold(0.0, f64::max);
    if max_error > 1.0e-12 {
        return Err(format!(
            "partitioned linear-field error {max_error} exceeds tolerance"
        ));
    }
    let synchronized =
        synchronize_partitioned_nodal_state::<&'static str, _>(result.clone(), |owned| {
            if owned.is_empty() {
                return Err("owner snapshot unexpectedly empty");
            }
            Ok(result.ghost.clone())
        })
        .map_err(|error| error.to_string())?;
    if synchronized.owned != result.owned || synchronized.ghost != result.ghost {
        return Err("owner/ghost exchange changed an already validated state".into());
    }
    let exchange_rejected =
        synchronize_partitioned_nodal_state::<&'static str, _>(result.clone(), |_| {
            Ok(BTreeMap::new())
        })
        .is_err();
    if !exchange_rejected {
        return Err("missing ghost exchange was accepted".into());
    }

    let invalid_layout = PartitionGhostLayout {
        owned_elements: Vec::new(),
        ghost_elements: Vec::new(),
        owned_nodes: vec![usize::MAX],
        ghost_nodes: Vec::new(),
    };
    let rejected =
        apply_partitioned_nodal_state_transfer_with_refinement_history::<&'static str, _>(
            &plan,
            &source_mesh,
            &target_mesh,
            &source_values,
            &history,
            RefinementTransferDirection::Refine,
            &source_entities,
            &target_entities,
            &invalid_layout,
            &target_layout,
            |_, _, _| Err("unexpected application callback"),
        )
        .is_err();
    if !rejected {
        return Err("invalid source partition layout was accepted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("owned_values", result.owned.len() as f64)
        .metric("ghost_values", result.ghost.len() as f64)
        .metric("max_linear_error", max_error)
        .metric("exchange_checks", 1.0)
        .metric("exchange_rejection_checks", exchange_rejected as u8 as f64)
        .metric("invalid_layout_rejected", 1.0)
        .dimension("adaptive H1 MeshLang transfer with deterministic partition ownership")
        .input_checksum("unit-square-refine-element-0:partition-1:meshlang-v2")
        .reference_checksum("linear-field-owner-ghost-split:v1")
        .formulation("checked refinement transfer followed by owner/ghost layout projection")
        .tolerance(
            "owned and ghost values present, linear error <= 1e-12, malformed layout rejected",
        )
        .seed("meshlang-partitioned-adaptive-h1-reference:cpu:v1")
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
