use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferResponsibility,
};
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    MeshlangTransferError, RefinementTransferDirection,
    apply_nodal_state_transfer_with_refinement_chain, unit_square_triangles,
};

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

#[test]
fn chain_bridge_transfers_linear_field_across_two_refinement_stages() {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0]).unwrap();

    let chain = IrStateTransferPlanChain::new(vec![
        plan_for_stage(level0.num_nodes(), level1.num_nodes()),
        plan_for_stage(level1.num_nodes(), level2.num_nodes()),
    ])
    .unwrap();
    let meshes = vec![&level0, &level1, &level2];
    let histories = vec![history0, history1];
    let maps = meshes
        .iter()
        .map(|mesh| entity_map(mesh))
        .collect::<Vec<_>>();
    let source_values = level0
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();

    let result = apply_nodal_state_transfer_with_refinement_chain::<&'static str, _>(
        &chain,
        &meshes,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &maps,
        |_stage, _entity, _action, _previous| Err("unexpected callback"),
    )
    .unwrap();

    assert_eq!(result.len(), level2.num_nodes());
    for (node, point) in level2.nodes.iter().enumerate() {
        assert!((result[&entity(node)] - linear_value(*point)).abs() < 1.0e-12);
    }
}

#[test]
fn chain_bridge_transfers_linear_field_across_two_coarsening_stages() {
    let level0 = unit_square_triangles(1);
    let mut level1 = level0.clone();
    let (_, history0) = refine_elements_with_history_checked(&mut level1, &[0]).unwrap();
    let mut level2 = level1.clone();
    let (_, history1) = refine_elements_with_history_checked(&mut level2, &[0]).unwrap();

    // Coarsening executes the adjacent stages in reverse lineage order, so
    // each plan describes the coarse target and asks the FEM consumer to
    // perform checked numerical restriction rather than relying on shared IDs.
    let chain = IrStateTransferPlanChain::new(vec![
        coarsening_plan(level1.num_nodes()),
        coarsening_plan(level0.num_nodes()),
    ])
    .unwrap();
    let meshes = vec![&level0, &level1, &level2];
    let histories = vec![history0, history1];
    let maps = meshes
        .iter()
        .map(|mesh| entity_map(mesh))
        .collect::<Vec<_>>();
    let source_values = level2
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();

    let result = apply_nodal_state_transfer_with_refinement_chain::<&'static str, _>(
        &chain,
        &meshes,
        &source_values,
        &histories,
        RefinementTransferDirection::Coarsen,
        &maps,
        |_stage, _entity, _action, _previous| Err("coarsening callback must not run"),
    )
    .unwrap();

    assert_eq!(result.len(), level0.num_nodes());
    for (node, point) in level0.nodes.iter().enumerate() {
        assert!((result[&entity(node)] - linear_value(*point)).abs() <= 1.0e-12);
    }
}

#[test]
fn chain_bridge_rejects_scalar_conservative_action_before_callbacks() {
    let source = unit_square_triangles(1);
    let mut target = source.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target, &[0]).unwrap();
    let mut entries = (0..target.num_nodes())
        .map(|node| IrStateTransferEntry {
            entity: entity(node),
            action: if node < source.num_nodes() {
                IrStateTransferAction::Preserve
            } else {
                IrStateTransferAction::Interpolate
            },
        })
        .collect::<Vec<_>>();
    entries[0].action = IrStateTransferAction::Conservative;
    let chain = IrStateTransferPlanChain::new(vec![IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries,
    }])
    .unwrap();
    let meshes = vec![&source, &target];
    let histories = vec![history];
    let maps = meshes
        .iter()
        .map(|mesh| entity_map(mesh))
        .collect::<Vec<_>>();
    let source_values = source
        .nodes
        .iter()
        .copied()
        .map(linear_value)
        .collect::<Vec<_>>();
    let mut callbacks = 0;

    let error = apply_nodal_state_transfer_with_refinement_chain::<&'static str, _>(
        &chain,
        &meshes,
        &source_values,
        &histories,
        RefinementTransferDirection::Refine,
        &maps,
        |_stage, _entity, _action, _previous| {
            callbacks += 1;
            Err("must not run")
        },
    )
    .unwrap_err();

    assert_eq!(callbacks, 0);
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        }
    ));
}
