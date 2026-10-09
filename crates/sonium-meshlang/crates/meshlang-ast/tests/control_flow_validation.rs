use meshlang_ast::{
    CanonicalGeometryIr, EntityDimension, EntityId, IrConditionalBranch, IrConditionalControl,
    IrLoopClause, IrLoopControl, IrValidationError, IrValue, SourceMetadata, SymbolId,
};

fn control(parent: Option<usize>) -> IrConditionalControl {
    IrConditionalControl {
        parent,
        parent_loop: None,
        branches: Vec::new(),
        provenance: Vec::new(),
    }
}

#[test]
fn conditional_parent_must_point_to_an_earlier_control() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(control(Some(0)));

    let errors = ir.validate().expect_err("self-parented control must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalParent {
            control: 0,
            parent: 0,
        }
    )));
}

#[test]
fn conditional_parent_must_be_in_range() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(control(Some(7)));

    let errors = ir.validate().expect_err("unknown parent must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalParent {
            control: 0,
            parent: 7,
        }
    )));
}

#[test]
fn nested_conditional_parent_is_valid_when_ordered() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(control(None));
    ir.controls.push(control(Some(0)));

    ir.validate()
        .expect("ordered nested controls must validate");
}

#[test]
fn unconditional_branch_must_be_last() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(IrConditionalControl {
        parent: None,
        parent_loop: None,
        branches: vec![
            IrConditionalBranch {
                condition: None,
                entities: Vec::new(),
                provenance: Vec::new(),
            },
            IrConditionalBranch {
                condition: Some(IrValue::Boolean(true)),
                entities: Vec::new(),
                provenance: Vec::new(),
            },
        ],
        provenance: Vec::new(),
    });

    let errors = ir.validate().expect_err("else-before-branch must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalBranchOrder {
            control: 0,
            branch: 0,
        }
    )));
}

#[test]
fn conditional_branches_cannot_share_entity_membership() {
    let entity = EntityId::new(EntityDimension::Volume, 1);
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(IrConditionalControl {
        parent: None,
        parent_loop: None,
        branches: vec![
            IrConditionalBranch {
                condition: Some(IrValue::Boolean(true)),
                entities: vec![entity],
                provenance: Vec::new(),
            },
            IrConditionalBranch {
                condition: None,
                entities: vec![entity],
                provenance: Vec::new(),
            },
        ],
        provenance: Vec::new(),
    });

    let errors = ir
        .validate()
        .expect_err("duplicate branch entity must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::DuplicateConditionalEntity {
            control: 0,
            entity: duplicate,
        } if *duplicate == entity
    )));
}
#[test]
fn dynamic_loop_validation_rejects_bad_parent_and_unknown_entity() {
    let entity = EntityId::new(EntityDimension::Volume, 1);
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.entities.push(meshlang_ast::GeometryNode {
        id: entity,
        name: "fixture.box".into(),
        operation: meshlang_ast::GeometryOperation::Primitive {
            kind: meshlang_ast::PrimitiveKind::Box,
        },
        inputs: Vec::new(),
        arguments: Vec::new(),
        provenance: Vec::new(),
    });
    ir.loops.push(IrLoopControl {
        parent: Some(1),
        parent_control: None,
        clauses: vec![IrLoopClause {
            variable: SymbolId(7),
            name: "i".into(),
            start: IrValue::Integer(0),
            end: IrValue::Parameter(SymbolId(8)),
            inclusive: false,
            provenance: Vec::new(),
        }],
        entities: vec![entity, EntityId::new(EntityDimension::Volume, 2)],
        provenance: Vec::new(),
    });
    let errors = ir.validate().expect_err("malformed dynamic loop must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidLoopParent {
            loop_index: 0,
            parent: 1
        }
    )));
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::UnknownLoopEntity { loop_index: 0, .. }
    )));
}

#[test]
fn mixed_control_parents_are_checked_across_ir_collections() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    let mut conditional = control(None);
    conditional.parent_loop = Some(0);
    ir.controls.push(conditional);
    ir.loops.push(IrLoopControl {
        parent: None,
        parent_control: Some(0),
        clauses: vec![IrLoopClause {
            variable: SymbolId(1),
            name: "i".into(),
            start: IrValue::Integer(0),
            end: IrValue::Integer(1),
            inclusive: false,
            provenance: Vec::new(),
        }],
        entities: Vec::new(),
        provenance: Vec::new(),
    });
    assert!(
        ir.validate().is_ok(),
        "existing mixed parents must validate"
    );

    ir.controls[0].parent_loop = Some(2);
    ir.loops[0].parent_control = Some(3);
    let errors = ir.validate().expect_err("unknown mixed parents must fail");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalLoopParent {
            control: 0,
            parent_loop: 2,
        }
    )));
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidLoopConditionalParent {
            loop_index: 0,
            parent_control: 3,
        }
    )));
}

#[test]
fn legacy_control_documents_default_cross_parent_fields() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(control(None));
    ir.loops.push(IrLoopControl {
        parent: None,
        parent_control: None,
        clauses: vec![IrLoopClause {
            variable: SymbolId(1),
            name: "i".into(),
            start: IrValue::Integer(0),
            end: IrValue::Integer(1),
            inclusive: false,
            provenance: Vec::new(),
        }],
        entities: Vec::new(),
        provenance: Vec::new(),
    });
    let mut json = serde_json::to_value(&ir).expect("serialize canonical IR");
    json["controls"][0]
        .as_object_mut()
        .expect("control object")
        .remove("parent_loop");
    json["loops"][0]
        .as_object_mut()
        .expect("loop object")
        .remove("parent_control");
    let decoded: CanonicalGeometryIr =
        serde_json::from_value(json).expect("legacy IR must deserialize");
    assert_eq!(decoded.controls[0].parent_loop, None);
    assert_eq!(decoded.loops[0].parent_control, None);
}
