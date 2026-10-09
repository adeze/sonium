use std::collections::BTreeMap;

use meshlang_ast::{
    CanonicalGeometryIr, EntityDimension, EntityId, GeometryNode, GeometryOperation,
    IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan, PrimitiveKind,
    RemeshResponsibility, SensitivityEntry, SourceMetadata, StateTransferApplyError,
    StateTransferResponsibility, SymbolId,
};

fn node(id: EntityId, kind: PrimitiveKind) -> GeometryNode {
    GeometryNode {
        id,
        name: format!("fixture::entity_{}", id.index),
        operation: GeometryOperation::Primitive { kind },
        inputs: Vec::new(),
        arguments: Vec::new(),
        provenance: Vec::new(),
    }
}

fn fixture(nodes: &[(u64, PrimitiveKind)]) -> CanonicalGeometryIr {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.entities.extend(
        nodes
            .iter()
            .map(|(index, kind)| node(EntityId::new(EntityDimension::Volume, *index), *kind)),
    );
    ir
}

#[test]
fn geometry_aware_interpolation_weights_are_deterministic_and_normalized() {
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 1),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 3),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 4),
                action: IrStateTransferAction::Interpolate,
            },
        ],
    };
    let source_points = BTreeMap::from([
        (EntityId::new(EntityDimension::Volume, 1), [0.0, 0.0, 0.0]),
        (EntityId::new(EntityDimension::Volume, 2), [2.0, 0.0, 0.0]),
        (EntityId::new(EntityDimension::Volume, 3), [0.0, 2.0, 0.0]),
    ]);
    let target_points = BTreeMap::from([
        (EntityId::new(EntityDimension::Volume, 1), [0.0, 0.0, 0.0]),
        (EntityId::new(EntityDimension::Volume, 2), [2.0, 0.0, 0.0]),
        (EntityId::new(EntityDimension::Volume, 4), [1.0, 0.0, 0.0]),
    ]);
    let weights = plan
        .derive_interpolation_weights(&source_points, &target_points, 2)
        .expect("geometry-aware weights");
    assert_eq!(weights[&EntityId::new(EntityDimension::Volume, 4)].len(), 2);
    assert!(
        (weights[&EntityId::new(EntityDimension::Volume, 4)]
            .iter()
            .map(|(_, weight)| weight)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1.0e-12
    );
    assert_eq!(
        weights[&EntityId::new(EntityDimension::Volume, 1)],
        vec![(EntityId::new(EntityDimension::Volume, 1), 1.0)]
    );
    assert_eq!(
        weights,
        plan.derive_interpolation_weights(&source_points, &target_points, 2)
            .unwrap()
    );

    let conservative_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 1),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 2),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 3),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 4),
                action: IrStateTransferAction::Conservative,
            },
        ],
    };
    let conservative = conservative_plan
        .derive_conservative_weights(
            &source_points,
            &target_points,
            &BTreeMap::from([
                (EntityId::new(EntityDimension::Volume, 1), 1.0),
                (EntityId::new(EntityDimension::Volume, 2), 1.0),
                (EntityId::new(EntityDimension::Volume, 3), 1.0),
            ]),
            &BTreeMap::from([
                (EntityId::new(EntityDimension::Volume, 1), 1.0),
                (EntityId::new(EntityDimension::Volume, 4), 2.0),
            ]),
            2,
        )
        .expect("conservative geometry-aware weights");
    assert_eq!(
        conservative[&EntityId::new(EntityDimension::Volume, 4)],
        vec![
            (EntityId::new(EntityDimension::Volume, 1), 1.0),
            (EntityId::new(EntityDimension::Volume, 2), 1.0),
        ]
    );
}

#[test]
fn geometry_aware_interpolation_weights_reject_invalid_geometry_contracts() {
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 1),
                action: IrStateTransferAction::Drop,
            },
            IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Volume, 2),
                action: IrStateTransferAction::Interpolate,
            },
        ],
    };
    let source_points =
        BTreeMap::from([(EntityId::new(EntityDimension::Volume, 1), [0.0, 0.0, 0.0])]);
    let missing_target = BTreeMap::new();
    assert_eq!(
        plan.derive_interpolation_weights(&source_points, &missing_target, 1),
        Err(meshlang_ast::SpatialInterpolationError::MissingTargetPoint(
            EntityId::new(EntityDimension::Volume, 2)
        ))
    );
    assert_eq!(
        plan.derive_interpolation_weights(
            &source_points,
            &BTreeMap::from([(
                EntityId::new(EntityDimension::Volume, 2),
                [f64::NAN, 0.0, 0.0],
            )]),
            1
        ),
        Err(meshlang_ast::SpatialInterpolationError::NonFinitePoint(
            EntityId::new(EntityDimension::Volume, 2)
        ))
    );
    assert_eq!(
        plan.derive_interpolation_weights(&source_points, &BTreeMap::new(), 0),
        Err(meshlang_ast::SpatialInterpolationError::InvalidNeighborCount)
    );
}

#[test]
fn state_transfer_plan_preserves_drops_and_initializes_stable_ids() {
    let old = fixture(&[(1, PrimitiveKind::Box), (2, PrimitiveKind::Sphere)]);
    let next = fixture(&[(1, PrimitiveKind::Box), (3, PrimitiveKind::Cylinder)]);
    let plan = old.state_transfer_plan(&next);

    assert_eq!(
        plan.responsibility,
        StateTransferResponsibility::ApplicationDefined
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 1)),
        Some(IrStateTransferAction::Preserve)
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 2)),
        Some(IrStateTransferAction::Drop)
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 3)),
        Some(IrStateTransferAction::ApplicationDefined)
    );
    assert!(plan.validate().is_ok());
}

#[test]
fn numeric_value_change_without_topology_uses_initialize_when_transfer_is_none() {
    let mut old = fixture(&[(1, PrimitiveKind::Box)]);
    let mut next = fixture(&[(1, PrimitiveKind::Box)]);
    old.entities[0].arguments.push(meshlang_ast::IrArgument {
        name: meshlang_ast::IrArgumentName::Size,
        value: meshlang_ast::IrValue::Integer(1),
    });
    next.entities[0].arguments.push(meshlang_ast::IrArgument {
        name: meshlang_ast::IrArgumentName::Size,
        value: meshlang_ast::IrValue::Integer(2),
    });

    let plan = old.state_transfer_plan(&next);
    assert_eq!(plan.responsibility, StateTransferResponsibility::None);
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 1)),
        Some(IrStateTransferAction::Initialize)
    );
}

#[test]
fn sensitivity_contract_selects_interpolation_for_modified_entities() {
    let mut old = fixture(&[(1, PrimitiveKind::Box)]);
    let next = fixture(&[(1, PrimitiveKind::Sphere)]);
    old.sensitivity.entries.push(SensitivityEntry {
        variable: SymbolId(7),
        parameters: Vec::new(),
        entities: vec![EntityId::new(EntityDimension::Volume, 1)],
        physical_groups: Vec::new(),
        remesh: RemeshResponsibility::Global,
        state_transfer: StateTransferResponsibility::Interpolate,
    });

    let plan = old.state_transfer_plan(&next);
    assert_eq!(
        plan.responsibility,
        StateTransferResponsibility::Interpolate
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 1)),
        Some(IrStateTransferAction::Interpolate)
    );
}

#[test]
fn topology_without_declared_sensitivity_requires_application_defined_transfer() {
    let old = fixture(&[(1, PrimitiveKind::Box)]);
    let next = fixture(&[(1, PrimitiveKind::Sphere), (2, PrimitiveKind::Cylinder)]);
    let plan = old.state_transfer_plan(&next);

    assert_eq!(
        plan.responsibility,
        StateTransferResponsibility::ApplicationDefined
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 1)),
        Some(IrStateTransferAction::ApplicationDefined)
    );
    assert_eq!(
        plan.action_for(EntityId::new(EntityDimension::Volume, 2)),
        Some(IrStateTransferAction::ApplicationDefined)
    );
}

#[test]
fn state_transfer_plan_round_trips_and_rejects_unsorted_entries() {
    let old = fixture(&[(1, PrimitiveKind::Box), (2, PrimitiveKind::Sphere)]);
    let next = fixture(&[(1, PrimitiveKind::Box), (2, PrimitiveKind::Sphere)]);
    let plan = old.state_transfer_plan(&next);
    let decoded: meshlang_ast::IrStateTransferPlan =
        serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
    assert_eq!(decoded, plan);
    assert!(decoded.validate().is_ok());

    let mut invalid = decoded.clone();
    invalid.entries.reverse();
    assert_eq!(
        invalid.validate(),
        Err("state-transfer entries must be strictly entity-sorted")
    );
}

#[test]
fn state_transfer_apply_copies_drops_and_dispatches_callbacks() {
    let id = |index| EntityId::new(EntityDimension::Volume, index);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
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
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Initialize,
            },
        ],
    };
    let source = BTreeMap::from([(id(1), 2.0_f64), (id(2), 8.0), (id(3), 4.0)]);
    let mut callbacks = Vec::new();
    let target = plan
        .apply::<f64, &'static str, _>(&source, |entity, action, previous| {
            callbacks.push((entity, action));
            match action {
                IrStateTransferAction::Interpolate => {
                    Ok(previous.expect("source").mul_add(0.5, 0.0))
                }
                IrStateTransferAction::Initialize => Ok(entity.index as f64),
                _ => unreachable!("only value-producing actions call the adapter"),
            }
        })
        .expect("valid transfer plan");

    assert_eq!(
        target,
        BTreeMap::from([(id(1), 2.0), (id(3), 2.0), (id(4), 4.0)])
    );
    assert_eq!(
        callbacks,
        vec![
            (id(3), IrStateTransferAction::Interpolate),
            (id(4), IrStateTransferAction::Initialize),
        ]
    );
}

#[test]
fn state_transfer_apply_rejects_stale_and_missing_source_state() {
    let id = |index| EntityId::new(EntityDimension::Volume, index);
    let preserve = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![IrStateTransferEntry {
            entity: id(1),
            action: IrStateTransferAction::Preserve,
        }],
    };
    let missing = preserve
        .apply::<f64, &'static str, _>(&BTreeMap::new(), |_, _, _| Ok(0.0))
        .expect_err("preserve must require source state");
    assert_eq!(missing, StateTransferApplyError::MissingSource(id(1)));

    let stale = preserve
        .apply::<f64, &'static str, _>(&BTreeMap::from([(id(2), 1.0)]), |_, _, _| Ok(0.0))
        .expect_err("unplanned source state must fail closed");
    assert_eq!(stale, StateTransferApplyError::UnexpectedSource(id(2)));
}

#[test]
fn state_transfer_apply_preserves_callback_error_without_partial_result() {
    let id = |index| EntityId::new(EntityDimension::Volume, index);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: id(1),
            action: IrStateTransferAction::Interpolate,
        }],
    };
    let error = plan
        .apply(&BTreeMap::from([(id(1), 3.0_f64)]), |_, action, _| {
            Err::<f64, _>(format!("reject {action:?}"))
        })
        .expect_err("callback failure must be propagated");
    assert_eq!(
        error,
        StateTransferApplyError::Callback {
            entity: id(1),
            action: IrStateTransferAction::Interpolate,
            error: "reject Interpolate".to_owned(),
        }
    );
}

#[test]
fn weighted_state_transfer_applies_interpolation_and_conservative_mass() {
    let id = |index| EntityId::new(EntityDimension::Volume, index);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
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
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: id(5),
                action: IrStateTransferAction::Initialize,
            },
        ],
    };
    let source = BTreeMap::from([(id(1), 2.0), (id(2), 8.0), (id(3), 4.0), (id(4), 3.0)]);
    let weights = BTreeMap::from([
        (id(3), vec![(id(1), 0.25), (id(3), 0.75)]),
        (id(4), vec![(id(1), 1.0), (id(4), 2.0)]),
    ]);
    let mut callbacks = 0;
    let target = plan
        .apply_weighted_f64(&source, &weights, |entity, action, _| {
            callbacks += 1;
            assert_eq!(action, IrStateTransferAction::Initialize);
            Ok::<_, &'static str>(entity.index as f64)
        })
        .expect("weighted state transfer must apply");
    assert_eq!(callbacks, 1);
    assert_eq!(
        target,
        BTreeMap::from([(id(1), 2.0), (id(3), 3.5), (id(4), 8.0), (id(5), 5.0)])
    );
}

#[test]
fn weighted_state_transfer_rejects_invalid_and_missing_weights() {
    let id = |index| EntityId::new(EntityDimension::Volume, index);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity: id(1),
            action: IrStateTransferAction::Interpolate,
        }],
    };
    let source = BTreeMap::from([(id(1), 1.0)]);
    let bad_sum = BTreeMap::from([(id(1), vec![(id(1), 0.5)])]);
    assert!(matches!(
        plan.apply_weighted_f64::<&'static str, _>(&source, &bad_sum, |_, _, _| Ok(0.0)),
        Err(StateTransferApplyError::InvalidWeights { entity, .. }) if entity == id(1)
    ));
    let missing = BTreeMap::from([(id(1), vec![(id(2), 1.0)])]);
    assert!(matches!(
        plan.apply_weighted_f64::<&'static str, _>(&source, &missing, |_, _, _| Ok(0.0)),
        Err(StateTransferApplyError::MissingWeightedSource { entity, source })
            if entity == id(1) && source == id(2)
    ));
    let extra = BTreeMap::from([(id(2), vec![(id(1), 1.0)])]);
    assert!(matches!(
        plan.apply_weighted_f64::<&'static str, _>(&source, &extra, |_, _, _| Ok(0.0)),
        Err(StateTransferApplyError::UnexpectedWeights(entity)) if entity == id(2)
    ));
}
#[test]
fn state_transfer_plan_rejects_actions_above_declared_responsibility() {
    let entity = EntityId::new(EntityDimension::Volume, 77);
    let conservative = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: vec![IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::Conservative,
        }],
    };
    assert_eq!(
        conservative.validate(),
        Err("state-transfer action exceeds declared responsibility")
    );

    let application_defined = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::None,
        entries: vec![IrStateTransferEntry {
            entity,
            action: IrStateTransferAction::ApplicationDefined,
        }],
    };
    assert_eq!(
        application_defined.validate(),
        Err("state-transfer action exceeds declared responsibility")
    );
}
