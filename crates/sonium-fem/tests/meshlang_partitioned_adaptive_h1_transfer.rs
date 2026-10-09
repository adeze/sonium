#![cfg(feature = "meshlang")]

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
use std::collections::BTreeMap;

fn point_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn linear_value(point: sonium_fem::mesh::Point) -> f64 {
    1.0 + 2.0 * point.x - 0.5 * point.y
}

fn all_point_entities(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|node| (point_entity(node), node))
        .collect()
}

#[test]
fn partitioned_adaptive_transfer_splits_owned_and_ghost_values() {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) =
        refine_elements_with_history_checked(&mut target_mesh, &[0]).expect("refinement history");

    let source_graph = build_element_graph(&source_mesh);
    let source_partition = Partition::try_from_ids(vec![0, 1], 2).expect("source partition");
    let source_layout = source_partition
        .ghost_layout(&source_mesh, &source_graph, 1)
        .expect("source ghost layout");

    let target_graph = build_element_graph(&target_mesh);
    // Keep the untouched coarse element on partition 0 and the refined
    // children on partition 1.  This gives partition 1 both midpoint nodes
    // it owns exclusively and coarse vertices it needs as ghosts.
    let target_ids = (0..target_mesh.elements.len())
        .map(|element| usize::from(element + 1 != target_mesh.elements.len()))
        .collect::<Vec<_>>();
    let target_partition = Partition::try_from_ids(target_ids, 2).expect("target partition");
    let target_layout = target_partition
        .ghost_layout(&target_mesh, &target_graph, 1)
        .expect("target ghost layout");

    let source_entities = all_point_entities(&source_mesh);
    let target_entities = all_point_entities(&target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..target_mesh.num_nodes())
            .map(|node| IrStateTransferEntry {
                entity: point_entity(node),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();

    let result = apply_partitioned_adaptive(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_values,
        &history,
        &source_entities,
        &target_entities,
        &source_layout,
        &target_layout,
    );
    assert!(!result.owned.is_empty(), "partition must own target values");
    assert!(
        !result.ghost.is_empty(),
        "partition must retain ghost values"
    );
    for (entity, value) in result.owned.iter().chain(result.ghost.iter()) {
        let node = target_entities[entity];
        assert!((value - linear_value(target_mesh.nodes[node])).abs() < 1.0e-12);
    }
}

#[test]
fn partitioned_adaptive_transfer_rejects_invalid_layout_before_callback() {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) =
        refine_elements_with_history_checked(&mut target_mesh, &[0]).expect("refinement history");
    let entities = all_point_entities(&source_mesh);
    let target_entities = all_point_entities(&target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: point_entity(0),
            action: IrStateTransferAction::Interpolate,
        }],
    };
    let invalid = PartitionGhostLayout {
        owned_elements: Vec::new(),
        ghost_elements: Vec::new(),
        owned_nodes: vec![usize::MAX],
        ghost_nodes: Vec::new(),
    };
    let valid = PartitionGhostLayout {
        owned_elements: Vec::new(),
        ghost_elements: Vec::new(),
        owned_nodes: vec![0],
        ghost_nodes: Vec::new(),
    };
    let mut called = false;
    let result = apply_partitioned_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &vec![1.0; source_mesh.num_nodes()],
        &history,
        RefinementTransferDirection::Refine,
        &entities,
        &target_entities,
        &invalid,
        &valid,
        |_, _, _| {
            called = true;
            Err("callback must not run")
        },
    );
    assert!(matches!(
        result,
        Err(sonium_fem::mesh::MeshlangTransferError::InvalidPartitionNode { .. })
    ));
    assert!(!called);
}

#[test]
fn partitioned_state_exchange_is_exact_and_transactional() {
    let entity_a = point_entity(0);
    let entity_b = point_entity(1);
    let mut owned = BTreeMap::new();
    owned.insert(entity_a, 1.5);
    let mut ghost = BTreeMap::new();
    ghost.insert(entity_b, 2.5);
    let state = sonium_fem::mesh::PartitionedNodalState { owned, ghost };
    let mut callback_calls = 0;
    let synchronized = synchronize_partitioned_nodal_state::<&'static str, _>(state, |owned| {
        callback_calls += 1;
        assert_eq!(owned.get(&entity_a), Some(&1.5));
        let mut exchanged = BTreeMap::new();
        exchanged.insert(entity_b, 3.5);
        Ok(exchanged)
    })
    .expect("exact owner/ghost exchange");
    assert_eq!(callback_calls, 1);
    assert_eq!(synchronized.owned.get(&entity_a), Some(&1.5));
    assert_eq!(synchronized.ghost.get(&entity_b), Some(&3.5));

    let mut missing = BTreeMap::new();
    missing.insert(entity_a, 3.5);
    let state = sonium_fem::mesh::PartitionedNodalState {
        owned: BTreeMap::from([(entity_a, 1.5)]),
        ghost: BTreeMap::from([(entity_b, 2.5)]),
    };
    let error = synchronize_partitioned_nodal_state::<&'static str, _>(state, |_| Ok(missing))
        .expect_err("missing/unknown ghost keys must fail closed");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::UnknownPartitionGhostEntity(_)
    ));

    let state = sonium_fem::mesh::PartitionedNodalState {
        owned: BTreeMap::from([(entity_a, 1.5)]),
        ghost: BTreeMap::from([(entity_b, 2.5)]),
    };
    let error =
        synchronize_partitioned_nodal_state::<&'static str, _>(state, |_| Ok(BTreeMap::new()))
            .expect_err("missing ghost key must fail closed");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::MissingPartitionGhostEntity(_)
    ));

    let state = sonium_fem::mesh::PartitionedNodalState {
        owned: BTreeMap::from([(entity_a, 1.5)]),
        ghost: BTreeMap::from([(entity_b, 2.5)]),
    };
    let error = synchronize_partitioned_nodal_state::<&'static str, _>(state, |_| {
        Err("transport unavailable")
    })
    .expect_err("transport failure must be typed");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::PartitionExchange(_)
    ));
}

#[test]
fn partitioned_state_exchange_rejects_non_finite_ghost_values() {
    let entity = point_entity(2);
    let state = sonium_fem::mesh::PartitionedNodalState {
        owned: BTreeMap::new(),
        ghost: BTreeMap::from([(entity, 0.0)]),
    };
    let error = synchronize_partitioned_nodal_state::<&'static str, _>(state, |_| {
        Ok(BTreeMap::from([(entity, f64::NAN)]))
    })
    .expect_err("non-finite exchanged value");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::NonFinitePartitionGhost { .. }
    ));
}

fn apply_partitioned_adaptive(
    plan: &IrStateTransferPlan,
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    source_values: &[f64],
    history: &sonium_fem::mesh::refinement::RefinementHistory,
    source_entities: &BTreeMap<EntityId, usize>,
    target_entities: &BTreeMap<EntityId, usize>,
    source_layout: &PartitionGhostLayout,
    target_layout: &PartitionGhostLayout,
) -> sonium_fem::mesh::PartitionedNodalState {
    apply_partitioned_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        plan,
        source_mesh,
        target_mesh,
        source_values,
        history,
        RefinementTransferDirection::Refine,
        source_entities,
        target_entities,
        source_layout,
        target_layout,
        |_, _, _| Err("unexpected callback"),
    )
    .expect("partitioned adaptive transfer")
}
