use meshlang_ast::span::Span;
use meshlang_ast::{
    Comment, CommentKind, NodeId, PROVENANCE_CONTRACT_VERSION, ProvenanceMap, ProvenanceOperation,
    ProvenanceRecord, ProvenanceValidationError, SourceId, SourceIdentity, SourceMap,
    SourceMetadata, SourceReference,
};

#[test]
fn canonical_entity_provenance_admission_is_typed() {
    let mut ir = meshlang_ast::CanonicalGeometryIr::new(meshlang_ast::SourceMetadata::for_source(
        "fixture.mlg",
        "geometry model {}",
    ));
    let entity = meshlang_ast::EntityId::new(meshlang_ast::EntityDimension::Volume, 1);
    ir.entities.push(meshlang_ast::GeometryNode {
        id: entity,
        name: "model.body".into(),
        operation: meshlang_ast::GeometryOperation::Primitive {
            kind: meshlang_ast::PrimitiveKind::Box,
        },
        inputs: Vec::new(),
        arguments: Vec::new(),
        provenance: Vec::new(),
    });

    assert!(matches!(
        ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                meshlang_ast::IrValidationError::MissingEntityProvenance { entity: found }
                    if *found == entity
            ))
    ));

    ir.entities[0]
        .provenance
        .push(meshlang_ast::SourceReference {
            source: meshlang_ast::SourceId::named("fixture.mlg"),
            node: meshlang_ast::NodeId(1),
            span: meshlang_ast::span::Span::new(8, 4),
        });
    assert!(matches!(
        ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                meshlang_ast::IrValidationError::InvalidEntityProvenanceSpan { entity: found }
                    if *found == entity
            ))
    ));
}

#[test]
fn canonical_non_entity_provenance_admission_is_typed() {
    let mut ir = meshlang_ast::CanonicalGeometryIr::new(meshlang_ast::SourceMetadata::for_source(
        "fixture.mlg",
        "geometry Root {}",
    ));
    let entity = meshlang_ast::EntityId::new(meshlang_ast::EntityDimension::Volume, 1);
    ir.entities.push(meshlang_ast::GeometryNode {
        id: entity,
        name: "Root.body".into(),
        operation: meshlang_ast::GeometryOperation::Primitive {
            kind: meshlang_ast::PrimitiveKind::Box,
        },
        inputs: Vec::new(),
        arguments: Vec::new(),
        provenance: vec![reference(1, 1, 0, 8)],
    });
    ir.physical_groups.push(meshlang_ast::PhysicalGroup {
        id: meshlang_ast::PhysicalGroupId(1),
        name: "body".into(),
        dimension: meshlang_ast::EntityDimension::Volume,
        entities: vec![entity],
        material: None,
        provenance: Vec::new(),
    });

    assert!(matches!(
        ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                meshlang_ast::IrValidationError::MissingProvenance { context }
                    if context == "physical group 1"
            ))
    ));

    ir.physical_groups[0].provenance = vec![reference(1, 2, 8, 4)];
    assert!(matches!(
        ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                meshlang_ast::IrValidationError::InvalidProvenanceSpan { context }
                    if context == "physical group 1"
            ))
    ));
}

fn reference(source: u64, node: u64, start: usize, end: usize) -> SourceReference {
    SourceReference {
        source: SourceId(source),
        node: NodeId(node),
        span: Span::with_file(start, end, 3),
    }
}

fn record(
    node: u64,
    sources: Vec<SourceReference>,
    operation: ProvenanceOperation,
) -> ProvenanceRecord {
    ProvenanceRecord {
        node: NodeId(node),
        sources,
        operation,
    }
}

#[test]
fn provenance_contract_is_versioned_and_canonicalization_is_deterministic() {
    assert_eq!(PROVENANCE_CONTRACT_VERSION, 1);
    let mut map = ProvenanceMap {
        records: vec![
            record(
                20,
                vec![reference(2, 8, 20, 24), reference(1, 7, 4, 8)],
                ProvenanceOperation::Transformed,
            ),
            record(10, vec![reference(1, 4, 0, 2)], ProvenanceOperation::Parsed),
        ],
    };

    map.canonicalize()
        .expect("valid ancestry can be canonicalized");
    assert_eq!(map.records[0].node, NodeId(10));
    assert_eq!(map.records[1].sources[0].source, SourceId(1));
    assert_eq!(map.records[1].sources[1].source, SourceId(2));
    assert!(map.validate().is_ok());
}

#[test]
fn duplicate_references_are_rejected_then_deduplicated_by_canonicalization() {
    let duplicate = reference(1, 2, 4, 8);
    let map = ProvenanceMap {
        records: vec![record(
            3,
            vec![duplicate.clone(), duplicate],
            ProvenanceOperation::Expanded,
        )],
    };
    assert!(matches!(
        map.validate(),
        Err(ProvenanceValidationError::DuplicateSourceReference { .. })
    ));

    let canonical = map.canonicalized().expect("duplicates are recoverable");
    assert_eq!(canonical.records[0].sources.len(), 1);
    assert!(canonical.validate().is_ok());
}

#[test]
fn invalid_ancestry_spans_and_empty_sources_fail_closed() {
    let invalid_span = ProvenanceMap {
        records: vec![record(
            3,
            vec![reference(1, 2, 8, 4)],
            ProvenanceOperation::Transformed,
        )],
    };
    assert!(matches!(
        invalid_span.validate(),
        Err(ProvenanceValidationError::InvalidSpan { .. })
    ));

    let empty = ProvenanceMap {
        records: vec![record(3, Vec::new(), ProvenanceOperation::Projected)],
    };
    assert!(matches!(
        empty.validate(),
        Err(ProvenanceValidationError::EmptySourceAncestry { .. })
    ));
}

#[test]
fn source_identity_and_serialized_provenance_round_trip_are_stable() {
    let identity = SourceIdentity::named("fixture.mesh").with_content("point(1, 2, 3);");
    assert!(identity.validate().is_ok());

    let map = ProvenanceMap {
        records: vec![record(
            3,
            vec![reference(identity.id.0, 2, 0, 12)],
            ProvenanceOperation::Migrated { from_schema: 0 },
        )],
    };
    let json = serde_json::to_string(&map).expect("provenance must serialize");
    let decoded: ProvenanceMap = serde_json::from_str(&json).expect("provenance must deserialize");
    assert_eq!(decoded, map);
    assert!(decoded.validate().is_ok());

    let mut mismatched = identity;
    mismatched.id = SourceId(0);
    assert!(mismatched.validate().is_err());
}

#[test]
fn source_map_canonicalization_orders_comments_and_rejects_inverted_spans() {
    let identity = SourceIdentity::named("comments.mlg");
    let metadata = SourceMetadata {
        identity: identity.clone(),
        ..SourceMetadata::default()
    };
    let mut map = SourceMap {
        metadata,
        source_identities: vec![identity],
        comments: vec![
            Comment::new(CommentKind::DocLine, "later", Span::new(8, 12)),
            Comment::new(CommentKind::Line, "first", Span::new(0, 5)),
        ],
        provenance: ProvenanceMap::default(),
        span: Span::new(0, 12),
    };

    map.canonicalize()
        .expect("valid comments can be canonicalized");
    assert_eq!(map.comments[0].text, "first");
    assert_eq!(map.comments[1].text, "later");

    map.comments[0].span = Span::new(9, 2);
    assert!(matches!(
        map.validate(),
        Err(ProvenanceValidationError::InvalidCommentSpan { span })
            if span == Span::new(9, 2)
    ));
}

#[test]
fn source_maps_reject_duplicate_identities_and_unknown_ancestry_sources() {
    let identity = SourceIdentity::named("root.mlg").with_content("geometry Root {}");
    let metadata = SourceMetadata {
        identity: identity.clone(),
        ..SourceMetadata::default()
    };
    let mut map = SourceMap {
        metadata,
        source_identities: vec![identity.clone()],
        comments: Vec::new(),
        provenance: ProvenanceMap {
            records: vec![record(
                3,
                vec![reference(identity.id.0, 2, 0, 8)],
                ProvenanceOperation::Parsed,
            )],
        },
        span: Span::new(0, 16),
    };
    assert!(map.validate().is_ok());

    map.source_identities.push(identity.clone());
    assert_eq!(
        map.validate(),
        Err(ProvenanceValidationError::DuplicateSourceIdentity {
            source: identity.id,
        })
    );

    map.source_identities.pop();
    map.provenance.records[0].sources[0].source = SourceId::named("missing.mlg");
    assert_eq!(
        map.validate(),
        Err(ProvenanceValidationError::UnknownSourceReference {
            node: NodeId(3),
            source: SourceId::named("missing.mlg"),
        })
    );
}
