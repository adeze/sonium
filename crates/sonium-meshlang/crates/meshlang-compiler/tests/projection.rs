use meshlang_ast::units::Dimension;
use meshlang_ast::{
    BooleanOperation, CanonicalGeometryIr, EntityDimension, EntityId, GeometryNode,
    GeometryOperation, IrArgument, IrArgumentName, IrParameter, IrValue, PrimitiveKind,
    ProjectionMethod, SourceMetadata, SymbolId, TransformOperation,
};
use meshlang_compiler::{
    compile_projection_mesh_artifact, entity_node_index, evaluate_projection,
    evaluate_projection_with_index, mesh_projection_iso_surface,
    mesh_projection_iso_surface_with_index, ProjectionArtifactError, ProjectionError,
    ProjectionMeshArtifact, MAX_PROJECTION_MESH_ARTIFACT_BYTES,
    PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
};
use std::collections::BTreeMap;

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

#[test]
fn source_projection_artifact_uses_canonical_ir_authority() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let surface = body |> project_iso_surface(tolerance: 0.01)
        }
    "#;
    let first = compile_projection_mesh_artifact(
        source,
        "projection-source.mlg",
        "model.surface",
        [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        [4, 4, 4],
        0.01,
    )
    .expect("source projection should compile through canonical IR");
    let second = compile_projection_mesh_artifact(
        source,
        "projection-source.mlg",
        "model.surface",
        [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        [4, 4, 4],
        0.01,
    )
    .expect("repeat source projection should be deterministic");
    assert_eq!(first, second);
    assert_eq!(
        first.schema_version,
        PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION
    );
    assert!(!first.mesh.vertices.is_empty());
    assert!(!first.mesh.triangles.is_empty());

    let error = compile_projection_mesh_artifact(
        source,
        "projection-source.mlg",
        "model.missing",
        [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        [4, 4, 4],
        0.01,
    )
    .expect_err("unknown projection must fail before sampling");
    assert!(matches!(
        error,
        meshlang_compiler::CompileError::Unsupported(_)
    ));
}

fn scalar(value: f64) -> IrValue {
    IrValue::Scalar {
        value,
        dimension: Dimension::LENGTH,
    }
}

fn vector(values: [f64; 3]) -> IrValue {
    IrValue::Vector(values.into_iter().map(scalar).collect())
}

fn argument(name: IrArgumentName, value: IrValue) -> IrArgument {
    IrArgument { name, value }
}

fn node(
    entity: EntityId,
    operation: GeometryOperation,
    inputs: &[EntityId],
    arguments: Vec<IrArgument>,
) -> GeometryNode {
    GeometryNode {
        id: entity,
        name: format!("fixture::entity_{}", entity.index),
        operation,
        inputs: inputs.to_vec(),
        arguments,
        provenance: Vec::new(),
    }
}

fn ir(nodes: Vec<GeometryNode>) -> CanonicalGeometryIr {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.entities = nodes;
    ir
}

fn project(input: EntityId, method: ProjectionMethod) -> GeometryNode {
    node(
        id(100),
        GeometryOperation::Project { method },
        &[input],
        Vec::new(),
    )
}

#[test]
fn sphere_signed_distance_and_iso_surface_are_bounded_and_deterministic() {
    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([0.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(2.0)),
        ],
    );
    let projection = project(id(1), ProjectionMethod::IsoSurface);
    let result = evaluate_projection(
        &ir(vec![sphere, projection]),
        id(100),
        [2.0, 0.0, 0.0],
        1e-6,
    )
    .expect("sphere projection should evaluate");
    assert_eq!(result.method, ProjectionMethod::IsoSurface);
    assert_eq!(result.signed_distance, 0.0);
    assert!(result.on_iso_surface);
    assert_eq!(
        evaluate_projection(
            &ir(vec![
                node(
                    id(1),
                    GeometryOperation::Primitive {
                        kind: PrimitiveKind::Sphere,
                    },
                    &[],
                    vec![argument(IrArgumentName::Radius, scalar(2.0))],
                ),
                project(id(1), ProjectionMethod::SignedDistance),
            ]),
            id(100),
            [0.0, 0.0, 0.0],
            1e-6
        )
        .unwrap()
        .signed_distance,
        -2.0
    );
}

#[test]
fn box_vector_size_and_closest_point_work() {
    let boxed = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Box,
        },
        &[],
        vec![
            argument(IrArgumentName::Origin, vector([0.0, 0.0, 0.0])),
            argument(IrArgumentName::Size, vector([2.0, 4.0, 6.0])),
        ],
    );
    let projection = project(id(1), ProjectionMethod::ClosestPoint);
    let result = evaluate_projection(&ir(vec![boxed, projection]), id(100), [3.0, 1.0, 2.0], 1e-5)
        .expect("box closest point should converge");
    assert_eq!(result.signed_distance, 1.0);
    let closest = result
        .closest_point
        .expect("closest point method returns point");
    assert!((closest[0] - 2.0).abs() < 1e-4);
    assert!((closest[1] - 1.0).abs() < 1e-4);
    assert!((closest[2] - 2.0).abs() < 1e-4);
}

#[test]
fn cylinder_and_uniform_scale_have_expected_signed_distance() {
    let cylinder = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Cylinder,
        },
        &[],
        vec![
            argument(IrArgumentName::Origin, vector([0.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(1.0)),
            argument(IrArgumentName::Height, scalar(4.0)),
        ],
    );
    let scaled = node(
        id(2),
        GeometryOperation::Transform {
            kind: TransformOperation::Scale,
        },
        &[id(1)],
        vec![argument(IrArgumentName::Scale, scalar(2.0))],
    );
    let projection = project(id(2), ProjectionMethod::SignedDistance);
    let result = evaluate_projection(
        &ir(vec![cylinder, scaled, projection]),
        id(100),
        [4.0, 0.0, 0.0],
        1e-6,
    )
    .expect("scaled cylinder should evaluate");
    assert!((result.signed_distance - 2.0).abs() < 1e-9);
}

#[test]
fn boolean_and_array_composition_use_sdf_rules() {
    let left = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([-1.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(1.0)),
        ],
    );
    let right = node(
        id(2),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([1.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(1.0)),
        ],
    );
    let union = node(
        id(3),
        GeometryOperation::Boolean {
            kind: BooleanOperation::Union,
        },
        &[id(1), id(2)],
        Vec::new(),
    );
    let difference = node(
        id(4),
        GeometryOperation::Boolean {
            kind: BooleanOperation::Difference,
        },
        &[id(1), id(2)],
        Vec::new(),
    );
    let array = node(id(5), GeometryOperation::Array, &[id(1), id(2)], Vec::new());
    let mut nodes = vec![left, right, union, difference, array];
    nodes.push(project(id(3), ProjectionMethod::SignedDistance));
    let result = evaluate_projection(&ir(nodes), id(100), [0.0, 0.0, 0.0], 1e-6).unwrap();
    assert!((result.signed_distance - 0.0).abs() < 1e-9);

    let mut nodes = vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![argument(IrArgumentName::Radius, scalar(2.0))],
        ),
        node(
            id(2),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![
                argument(IrArgumentName::Center, vector([0.5, 0.0, 0.0])),
                argument(IrArgumentName::Radius, scalar(0.5)),
            ],
        ),
        node(
            id(4),
            GeometryOperation::Boolean {
                kind: BooleanOperation::Difference,
            },
            &[id(1), id(2)],
            Vec::new(),
        ),
    ];
    nodes.push(project(id(4), ProjectionMethod::SignedDistance));
    let result = evaluate_projection(&ir(nodes), id(100), [0.5, 0.0, 0.0], 1e-6).unwrap();
    assert!(result.signed_distance > 0.49);

    let mut nodes = vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![argument(IrArgumentName::Radius, scalar(1.0))],
        ),
        node(
            id(2),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![
                argument(IrArgumentName::Center, vector([3.0, 0.0, 0.0])),
                argument(IrArgumentName::Radius, scalar(1.0)),
            ],
        ),
        node(id(5), GeometryOperation::Array, &[id(1), id(2)], Vec::new()),
    ];
    nodes.push(project(id(5), ProjectionMethod::SignedDistance));
    let result = evaluate_projection(&ir(nodes), id(100), [3.0, 0.0, 0.0], 1e-6).unwrap();
    assert_eq!(result.signed_distance, -1.0);
}

#[test]
fn translate_and_parameter_arguments_are_resolved() {
    let mut geometry = CanonicalGeometryIr::new(SourceMetadata::default());
    geometry.parameters.push(IrParameter {
        id: SymbolId(7),
        name: "radius".into(),
        value: scalar(1.5),
        dimension: Dimension::LENGTH,
        provenance: Vec::new(),
    });
    geometry.entities = vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![
                argument(IrArgumentName::Center, vector([1.0, 0.0, 0.0])),
                argument(IrArgumentName::Radius, IrValue::Parameter(SymbolId(7))),
            ],
        ),
        node(
            id(2),
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            &[id(1)],
            vec![argument(IrArgumentName::Direction, vector([2.0, 0.0, 0.0]))],
        ),
        project(id(2), ProjectionMethod::SignedDistance),
    ];
    let result = evaluate_projection(&geometry, id(100), [3.0, 0.0, 0.0], 1e-6).unwrap();
    assert!((result.signed_distance + 1.5).abs() < 1e-9);
}

#[test]
fn invalid_dynamic_cyclic_and_unsupported_inputs_fail_explicitly() {
    let dynamic = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(
            IrArgumentName::Radius,
            IrValue::Call {
                function: SymbolId(99),
                arguments: Vec::new(),
            },
        )],
    );
    let mut nodes = vec![dynamic, project(id(1), ProjectionMethod::SignedDistance)];
    let error = evaluate_projection(&ir(nodes), id(100), [0.0, 0.0, 0.0], 1e-6).unwrap_err();
    assert!(matches!(error, ProjectionError::DynamicArgument { .. }));

    nodes = vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Cone,
            },
            &[],
            Vec::new(),
        ),
        project(id(1), ProjectionMethod::SignedDistance),
    ];
    let error = evaluate_projection(&ir(nodes), id(100), [0.0, 0.0, 0.0], 1e-6).unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::UnsupportedOperation { .. }
    ));

    nodes = vec![
        node(
            id(1),
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            &[id(2)],
            vec![argument(IrArgumentName::Direction, vector([0.0, 0.0, 0.0]))],
        ),
        node(
            id(2),
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            &[id(1)],
            vec![argument(IrArgumentName::Direction, vector([0.0, 0.0, 0.0]))],
        ),
        project(id(1), ProjectionMethod::SignedDistance),
    ];
    let error = evaluate_projection(&ir(nodes), id(100), [0.0, 0.0, 0.0], 1e-6).unwrap_err();
    assert!(matches!(error, ProjectionError::Cycle(_)));
}

#[test]
fn invalid_probe_inputs_are_rejected_before_geometry_walk() {
    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(IrArgumentName::Radius, scalar(1.0))],
    );
    let projection = project(id(1), ProjectionMethod::SignedDistance);
    let geometry = ir(vec![sphere, projection]);
    assert!(matches!(
        evaluate_projection(&geometry, id(100), [f64::NAN, 0.0, 0.0], 1e-6),
        Err(ProjectionError::NonFinitePoint)
    ));
    assert!(matches!(
        evaluate_projection(&geometry, id(100), [0.0, 0.0, 0.0], -1.0),
        Err(ProjectionError::InvalidTolerance)
    ));
}

#[test]
fn iso_surface_sampling_is_deterministic_bounded_and_watertight() {
    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([0.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(1.0)),
        ],
    );
    let projection = project(id(1), ProjectionMethod::IsoSurface);
    let geometry = ir(vec![sphere, projection]);
    let bounds = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]];
    let first = mesh_projection_iso_surface(&geometry, id(100), bounds, [8, 8, 8], 1.0e-8)
        .expect("bounded sphere iso-surface must sample");
    let repeated = mesh_projection_iso_surface(&geometry, id(100), bounds, [8, 8, 8], 1.0e-8)
        .expect("repeated sphere iso-surface must sample");
    assert_eq!(first, repeated);
    assert!(!first.vertices.is_empty());
    assert!(!first.triangles.is_empty());
    assert!(first.vertices.iter().all(|point| {
        point.iter().all(|coordinate| coordinate.is_finite())
            && point.iter().enumerate().all(|(axis, coordinate)| {
                *coordinate >= bounds[0][axis] && *coordinate <= bounds[1][axis]
            })
            && ((point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt() - 1.0)
                .abs()
                < 0.06
    }));
    assert!(first.triangles.iter().all(|triangle| {
        let [a, b, c] = *triangle;
        (a as usize) < first.vertices.len()
            && (b as usize) < first.vertices.len()
            && (c as usize) < first.vertices.len()
            && a != b
            && b != c
            && a != c
    }));

    let mut edge_uses = BTreeMap::<(u32, u32), Vec<bool>>::new();
    for triangle in &first.triangles {
        for edge_index in 0..3 {
            let from = triangle[edge_index];
            let to = triangle[(edge_index + 1) % 3];
            edge_uses
                .entry((from.min(to), from.max(to)))
                .or_default()
                .push(from < to);
        }
    }
    let manifold_edges = edge_uses
        .values()
        .filter(|uses| uses.len() == 2)
        .collect::<Vec<_>>();
    assert!(!manifold_edges.is_empty());
    assert!(
        manifold_edges.iter().all(|uses| uses[0] != uses[1]),
        "iso-surface winding must oppose shared edges"
    );
}

#[test]
fn iso_surface_artifact_round_trip_is_versioned_and_deterministic() {
    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(IrArgumentName::Radius, scalar(1.0))],
    );
    let mesh = mesh_projection_iso_surface(
        &ir(vec![sphere, project(id(1), ProjectionMethod::IsoSurface)]),
        id(100),
        [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
        [8, 8, 8],
        1.0e-8,
    )
    .expect("bounded sphere iso-surface must sample");

    let artifact = ProjectionMeshArtifact::from_mesh(mesh).expect("mesh contract must validate");
    assert_eq!(
        artifact.schema_version,
        PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION
    );
    let encoded = artifact.to_json().expect("artifact must serialize");
    assert_eq!(
        encoded,
        artifact
            .to_json()
            .expect("serialization must be deterministic")
    );
    let decoded = ProjectionMeshArtifact::from_json(&encoded).expect("artifact must deserialize");
    assert_eq!(decoded.schema_version, artifact.schema_version);
    assert_eq!(decoded.mesh.cells, artifact.mesh.cells);
    assert_eq!(decoded.mesh.triangles, artifact.mesh.triangles);
    assert_eq!(decoded.mesh.vertices.len(), artifact.mesh.vertices.len());
    for (decoded_vertex, original_vertex) in
        decoded.mesh.vertices.iter().zip(&artifact.mesh.vertices)
    {
        for (decoded_coordinate, original_coordinate) in decoded_vertex.iter().zip(original_vertex)
        {
            assert!((decoded_coordinate - original_coordinate).abs() <= 1.0e-15);
        }
    }
}

#[test]
fn iso_surface_artifact_rejects_tampered_schema_and_indices() {
    let artifact = ProjectionMeshArtifact {
        schema_version: PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
        mesh: meshlang_compiler::ProjectionMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            bounds: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cells: [1, 1, 1],
        },
    };
    let encoded = artifact.to_json().expect("valid artifact must serialize");
    let tampered_schema = encoded.replace(
        &format!("\"schema_version\":{PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION}"),
        "\"schema_version\":999",
    );
    assert!(matches!(
        ProjectionMeshArtifact::from_json(&tampered_schema),
        Err(ProjectionArtifactError::UnsupportedSchema { .. })
    ));

    let unknown_field = encoded.replace(
        &format!("\"schema_version\":{PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION}"),
        &format!("\"schema_version\":{PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION},\"extra\":true"),
    );
    assert!(matches!(
        ProjectionMeshArtifact::from_json(&unknown_field),
        Err(ProjectionArtifactError::Json(_))
    ));

    let tampered_indices = encoded.replace("\"triangles\":[[0,1,2]]", "\"triangles\":[[0,1,3]]");
    assert!(matches!(
        ProjectionMeshArtifact::from_json(&tampered_indices),
        Err(ProjectionArtifactError::InvalidMesh(_))
    ));

    let oversized = " ".repeat(MAX_PROJECTION_MESH_ARTIFACT_BYTES + 1);
    assert!(matches!(
        ProjectionMeshArtifact::from_json(&oversized),
        Err(ProjectionArtifactError::ArtifactTooLarge { .. })
    ));
}

#[test]
fn iso_surface_sampling_rejects_invalid_bounds_grid_and_method() {
    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(IrArgumentName::Radius, scalar(1.0))],
    );
    let geometry = ir(vec![sphere, project(id(1), ProjectionMethod::IsoSurface)]);
    assert!(matches!(
        mesh_projection_iso_surface(
            &geometry,
            id(100),
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [0, 2, 2],
            0.0,
        ),
        Err(ProjectionError::InvalidGrid)
    ));
    assert!(matches!(
        mesh_projection_iso_surface(
            &geometry,
            id(100),
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [65, 1, 1],
            0.0,
        ),
        Err(ProjectionError::GridTooLarge { .. })
    ));
    assert!(matches!(
        mesh_projection_iso_surface(
            &geometry,
            id(100),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            [2, 2, 2],
            0.0,
        ),
        Err(ProjectionError::InvalidBounds)
    ));
    let closest_geometry = ir(vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![argument(IrArgumentName::Radius, scalar(1.0))],
        ),
        project(id(1), ProjectionMethod::ClosestPoint),
    ]);
    assert!(matches!(
        mesh_projection_iso_surface(
            &closest_geometry,
            id(100),
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            [2, 2, 2],
            0.0,
        ),
        Err(ProjectionError::NotIsoSurface { .. })
    ));
}

#[test]
fn parallel_grid_sampling_and_hoisted_index_match_serial_paths() {
    // Hermetic env override: this is the only test in this binary that
    // writes the gate keys, and both paths compute identical values, so a
    // concurrent reader can at most take the other (correct) path.
    let prev_threshold = std::env::var_os("SONIUM_MESHLANG_PAR_THRESHOLD");
    let prev_serial = std::env::var_os("SONIUM_SERIAL");
    let set_env = |threshold: Option<&str>, serial: Option<&str>| {
        // SAFETY: single writer in this test binary; restored below.
        unsafe {
            match threshold {
                Some(value) => std::env::set_var("SONIUM_MESHLANG_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_MESHLANG_PAR_THRESHOLD"),
            }
            match serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    };

    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([0.0, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(1.0)),
        ],
    );
    let geometry = ir(vec![sphere, project(id(1), ProjectionMethod::IsoSurface)]);
    let bounds = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]];

    // Forced parallel (729 grid nodes >= threshold 1) vs forced serial:
    // ordered collect keeps every sample identical.
    set_env(Some("1"), None);
    let parallel = mesh_projection_iso_surface(&geometry, id(100), bounds, [8, 8, 8], 1.0e-8)
        .expect("forced-parallel iso-surface must sample");
    set_env(None, Some("1"));
    let serial = mesh_projection_iso_surface(&geometry, id(100), bounds, [8, 8, 8], 1.0e-8)
        .expect("forced-serial iso-surface must sample");
    assert_eq!(parallel, serial);

    // Hoisted index matches the per-query rebuild for both entry points.
    set_env(None, None);
    let index = entity_node_index(&geometry);
    let with_index = mesh_projection_iso_surface_with_index(
        &geometry,
        &index,
        id(100),
        bounds,
        [8, 8, 8],
        1.0e-8,
    )
    .expect("indexed iso-surface must sample");
    assert_eq!(with_index, serial);
    let point = evaluate_projection(&geometry, id(100), [3.0, 0.0, 0.0], 1e-6)
        .expect("point projection must evaluate");
    let point_indexed =
        evaluate_projection_with_index(&geometry, &index, id(100), [3.0, 0.0, 0.0], 1e-6)
            .expect("indexed point projection must evaluate");
    assert_eq!(point, point_indexed);

    // SAFETY: restoring the saved gate overrides.
    unsafe {
        match prev_threshold {
            Some(value) => std::env::set_var("SONIUM_MESHLANG_PAR_THRESHOLD", value),
            None => std::env::remove_var("SONIUM_MESHLANG_PAR_THRESHOLD"),
        }
        match prev_serial {
            Some(value) => std::env::set_var("SONIUM_SERIAL", value),
            None => std::env::remove_var("SONIUM_SERIAL"),
        }
    }
}
