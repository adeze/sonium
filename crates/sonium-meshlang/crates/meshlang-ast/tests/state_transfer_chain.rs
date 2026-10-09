use std::collections::BTreeMap;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    IrStateTransferPlanChain, StateTransferChainApplyError, StateTransferResponsibility,
    StateTransferReverseError,
};

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

fn plan(
    entries: impl IntoIterator<Item = (EntityId, IrStateTransferAction)>,
) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: entries
            .into_iter()
            .map(|(entity, action)| IrStateTransferEntry { entity, action })
            .collect(),
    }
}

#[test]
fn multi_stage_chain_applies_in_order_and_publishes_only_final_map() {
    let first = plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::Initialize),
        (id(3), IrStateTransferAction::Drop),
    ]);
    let second = plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::ApplicationDefined),
        (id(4), IrStateTransferAction::Initialize),
    ]);
    let chain = IrStateTransferPlanChain::new(vec![first, second]).unwrap();
    let source = BTreeMap::from([(id(1), 10_i32), (id(2), 2_i32), (id(3), 30_i32)]);
    let mut callbacks = Vec::new();

    let result = chain
        .apply(&source, |stage, entity, action, previous| {
            callbacks.push((stage, entity, action));
            match (stage, entity, action, previous.copied()) {
                (0, entity, IrStateTransferAction::Initialize, Some(2)) if entity == id(2) => {
                    Ok(20)
                }
                (1, entity, IrStateTransferAction::ApplicationDefined, Some(20))
                    if entity == id(2) =>
                {
                    Ok(30)
                }
                (1, entity, IrStateTransferAction::Initialize, None) if entity == id(4) => Ok(40),
                _ => Err("unexpected callback"),
            }
        })
        .unwrap();

    assert_eq!(chain.stage_count(), 2);
    assert_eq!(
        result,
        BTreeMap::from([(id(1), 10), (id(2), 30), (id(4), 40)])
    );
    assert_eq!(callbacks.len(), 3);
    assert!(
        !callbacks
            .iter()
            .any(|(_, entity, action)| *entity == id(1)
                && *action == IrStateTransferAction::Preserve)
    );
    assert!(!result.contains_key(&id(3)));
    assert_eq!(
        source,
        BTreeMap::from([(id(1), 10), (id(2), 2), (id(3), 30)])
    );
}

#[test]
fn chain_rejects_preserve_after_drop_before_callbacks() {
    let error = IrStateTransferPlanChain::new(vec![
        plan([(id(1), IrStateTransferAction::Drop)]),
        plan([(id(1), IrStateTransferAction::Preserve)]),
    ])
    .unwrap_err();

    assert!(matches!(
        error,
        meshlang_ast::StateTransferChainError::PreserveWithoutPreviousOutput { stage: 1, entity }
            if entity == id(1)
    ));
}

#[test]
fn chain_wraps_stage_failure_and_rejects_empty_or_unsorted_input() {
    assert!(matches!(
        IrStateTransferPlanChain::new(Vec::new()),
        Err(meshlang_ast::StateTransferChainError::Empty)
    ));
    let unsorted = plan([
        (id(2), IrStateTransferAction::Preserve),
        (id(1), IrStateTransferAction::Preserve),
    ]);
    assert!(matches!(
        IrStateTransferPlanChain::new(vec![unsorted]),
        Err(meshlang_ast::StateTransferChainError::InvalidStage { stage: 0, .. })
    ));

    let chain = IrStateTransferPlanChain::new(vec![
        plan([(id(1), IrStateTransferAction::Preserve)]),
        plan([(id(1), IrStateTransferAction::ApplicationDefined)]),
    ])
    .unwrap();
    let error = chain
        .apply(
            &BTreeMap::from([(id(1), 1_i32)]),
            |stage, _entity, _action, _previous| {
                if stage == 1 {
                    Err("injected stage failure")
                } else {
                    Ok(1)
                }
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        StateTransferChainApplyError::Stage { stage: 1, .. }
    ));
}

#[test]
fn chain_round_trips_through_json() {
    let chain = IrStateTransferPlanChain::new(vec![plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::Initialize),
    ])])
    .unwrap();
    let json = serde_json::to_string(&chain).unwrap();
    let decoded: IrStateTransferPlanChain = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, chain);
}

#[test]
fn reverse_plan_swaps_topology_actions_for_coarsening() {
    let forward = plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::Initialize),
        (id(3), IrStateTransferAction::Drop),
        (id(4), IrStateTransferAction::ApplicationDefined),
    ]);
    let reverse = forward.reversed_for_coarsening().unwrap();
    assert_eq!(
        reverse.entries,
        vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Initialize,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::ApplicationDefined,
            },
        ]
    );
}

#[test]
fn reverse_plan_rejects_actions_without_entity_only_inverse() {
    let forward = plan([(id(1), IrStateTransferAction::Interpolate)]);
    assert!(matches!(
        forward.reversed_for_coarsening(),
        Err(StateTransferReverseError::NonInvertibleAction {
            entity,
            action: IrStateTransferAction::Interpolate,
        }) if entity == id(1)
    ));
}

#[test]
fn reverse_chain_reverses_stage_order_and_topology_actions() {
    let first = plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::Initialize),
        (id(3), IrStateTransferAction::Drop),
    ]);
    let second = plan([
        (id(1), IrStateTransferAction::Preserve),
        (id(2), IrStateTransferAction::ApplicationDefined),
        (id(4), IrStateTransferAction::Initialize),
    ]);
    let forward = IrStateTransferPlanChain::new(vec![first, second]).unwrap();

    let reverse = forward.reversed_for_coarsening().unwrap();
    assert_eq!(reverse.stage_count(), 2);
    assert_eq!(
        reverse.stages[0].entries,
        vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::ApplicationDefined,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Drop,
            },
        ]
    );
    assert_eq!(
        reverse.stages[1].entries,
        vec![
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Initialize,
            },
        ]
    );
}

#[test]
fn reverse_chain_rejects_numeric_actions_before_reordering() {
    let forward =
        IrStateTransferPlanChain::new(vec![plan([(id(1), IrStateTransferAction::Interpolate)])])
            .unwrap();

    assert!(matches!(
        forward.reversed_for_coarsening(),
        Err(StateTransferReverseError::NonInvertibleAction {
            entity,
            action: IrStateTransferAction::Interpolate,
        }) if entity == id(1)
    ));
}
