#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::refinement::refine_elements_with_history_checked;
use sonium_fem::mesh::{
    MeshlangTransferError, RefinementTransferDirection,
    apply_nodal_state_transfer_with_refinement_history, unit_square_triangles,
};
use std::collections::BTreeMap;

fn point_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn field_value(point: sonium_fem::mesh::Point) -> f64 {
    1.0 + 2.0 * point.x - 0.5 * point.y
}

#[test]
fn adaptive_h1_bridge_applies_lineage_plan_and_preserves_linear_field() {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target_mesh, &[0]).unwrap();
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(field_value)
        .collect::<Vec<_>>();

    let source_entities = (0..source_mesh.num_nodes())
        .map(|index| (point_entity(index), index))
        .collect::<BTreeMap<_, _>>();
    let target_entities = (0..target_mesh.num_nodes())
        .map(|index| (point_entity(index), index))
        .collect::<BTreeMap<_, _>>();

    let mut entries = vec![IrStateTransferEntry {
        entity: point_entity(0),
        action: IrStateTransferAction::Preserve,
    }];
    entries.extend(
        (source_mesh.num_nodes()..target_mesh.num_nodes()).map(|index| IrStateTransferEntry {
            entity: point_entity(index),
            action: IrStateTransferAction::Interpolate,
        }),
    );
    entries.push(IrStateTransferEntry {
        entity: point_entity(target_mesh.num_nodes() + 1),
        action: IrStateTransferAction::Initialize,
    });
    entries.push(IrStateTransferEntry {
        entity: point_entity(target_mesh.num_nodes() + 2),
        action: IrStateTransferAction::Drop,
    });

    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries,
    };
    let transferred = apply_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, action, _| match action {
            IrStateTransferAction::Initialize => Ok(42.0),
            _ => Err("unexpected adaptive callback action"),
        },
    )
    .unwrap();

    assert_eq!(transferred[&point_entity(0)], source_values[0]);
    for index in source_mesh.num_nodes()..target_mesh.num_nodes() {
        let expected = field_value(target_mesh.nodes[index]);
        assert!((transferred[&point_entity(index)] - expected).abs() < 1.0e-12);
    }
    assert_eq!(
        transferred[&point_entity(target_mesh.num_nodes() + 1)],
        42.0
    );
    assert!(!transferred.contains_key(&point_entity(target_mesh.num_nodes() + 2)));
}

#[test]
fn adaptive_h1_bridge_rejects_scalar_conservative_transfer_before_lineage_work() {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone();
    let (_, history) = refine_elements_with_history_checked(&mut target_mesh, &[0]).unwrap();
    let entity = point_entity(0);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Conservative,
        }],
    };
    let source_entities = BTreeMap::from([(entity, 0)]);
    let target_entities = BTreeMap::from([(entity, 0)]);
    let error = apply_nodal_state_transfer_with_refinement_history::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &[1.0; 4],
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        }
    ));
}
