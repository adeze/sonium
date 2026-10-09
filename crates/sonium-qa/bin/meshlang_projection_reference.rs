//! Release-facing bounded analytic MeshLang projection consumer reference.

use meshlang_ast::units::Dimension;
use meshlang_ast::{
    BooleanOperation, CanonicalGeometryIr, EntityDimension, EntityId, GeometryNode,
    GeometryOperation, IrArgument, IrArgumentName, IrParameter, IrValue, PrimitiveKind,
    ProjectionMethod, SourceMetadata, SymbolId, TransformOperation,
};
use meshlang_compiler::{
    ProjectionError, ProjectionMeshArtifact, compile_projection_mesh_artifact, compile_to_geo_full,
    compile_to_python_full, evaluate_projection, mesh_projection_iso_surface,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.projection-reference.v1";

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
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
        name: format!("qa::entity_{}", entity.index),
        operation,
        inputs: inputs.to_vec(),
        arguments,
        provenance: Vec::new(),
    }
}

fn project(input: EntityId, method: ProjectionMethod) -> GeometryNode {
    node(
        id(100),
        GeometryOperation::Project { method },
        &[input],
        Vec::new(),
    )
}

fn geometry(nodes: Vec<GeometryNode>) -> CanonicalGeometryIr {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.entities = nodes;
    ir
}

fn run() -> Result<QaResult, String> {
    let mut checks = 0_u64;

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
    let result = evaluate_projection(
        &geometry(vec![
            sphere.clone(),
            project(id(1), ProjectionMethod::SignedDistance),
        ]),
        id(100),
        [3.0, 0.0, 0.0],
        1e-6,
    )
    .map_err(|error| format!("sphere outside: {error}"))?;
    if (result.signed_distance - 1.0).abs() > 1e-12 {
        return Err(format!(
            "sphere outside distance was {}",
            result.signed_distance
        ));
    }
    checks += 1;
    let result = evaluate_projection(
        &geometry(vec![sphere, project(id(1), ProjectionMethod::IsoSurface)]),
        id(100),
        [2.0, 0.0, 0.0],
        1e-6,
    )
    .map_err(|error| format!("sphere iso-surface: {error}"))?;
    if !result.on_iso_surface {
        return Err("sphere iso-surface admission failed".into());
    }
    checks += 1;

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
    let result = evaluate_projection(
        &geometry(vec![boxed, project(id(1), ProjectionMethod::ClosestPoint)]),
        id(100),
        [3.0, 1.0, 2.0],
        1e-5,
    )
    .map_err(|error| format!("box closest point: {error}"))?;
    let closest = result
        .closest_point
        .ok_or_else(|| "box closest point was omitted".to_owned())?;
    if (result.signed_distance - 1.0).abs() > 1e-9 || (closest[0] - 2.0).abs() > 1e-4 {
        return Err(format!("box closest point mismatch: {result:?}"));
    }
    checks += 2;

    let cylinder = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Cylinder,
        },
        &[],
        vec![
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
    let result = evaluate_projection(
        &geometry(vec![
            cylinder,
            scaled,
            project(id(2), ProjectionMethod::SignedDistance),
        ]),
        id(100),
        [4.0, 0.0, 0.0],
        1e-6,
    )
    .map_err(|error| format!("scaled cylinder: {error}"))?;
    if (result.signed_distance - 2.0).abs() > 1e-9 {
        return Err(format!(
            "scaled cylinder distance was {}",
            result.signed_distance
        ));
    }
    checks += 1;

    let left = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(IrArgumentName::Radius, scalar(2.0))],
    );
    let right = node(
        id(2),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![
            argument(IrArgumentName::Center, vector([0.5, 0.0, 0.0])),
            argument(IrArgumentName::Radius, scalar(0.5)),
        ],
    );
    let difference = node(
        id(3),
        GeometryOperation::Boolean {
            kind: BooleanOperation::Difference,
        },
        &[id(1), id(2)],
        Vec::new(),
    );
    let result = evaluate_projection(
        &geometry(vec![
            left,
            right,
            difference,
            project(id(3), ProjectionMethod::SignedDistance),
        ]),
        id(100),
        [0.5, 0.0, 0.0],
        1e-6,
    )
    .map_err(|error| format!("boolean difference: {error}"))?;
    if result.signed_distance < 0.49 {
        return Err(format!(
            "boolean difference distance was {}",
            result.signed_distance
        ));
    }
    checks += 1;

    let mut parameterized = CanonicalGeometryIr::new(SourceMetadata::default());
    parameterized.parameters.push(IrParameter {
        id: SymbolId(7),
        name: "radius".into(),
        value: IrValue::Integer(2),
        dimension: Dimension::LENGTH,
        provenance: Vec::new(),
    });
    parameterized.entities = vec![
        node(
            id(1),
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            &[],
            vec![argument(
                IrArgumentName::Radius,
                IrValue::Parameter(SymbolId(7)),
            )],
        ),
        project(id(1), ProjectionMethod::SignedDistance),
    ];
    let descriptors = meshlang_compiler::projection_backend_descriptors(&parameterized);
    if descriptors.len() != 1
        || descriptors[0]["method"] != "signed_distance"
        || descriptors[0]["native_geometry"] != false
        || descriptors[0]["consumer"] != "ProjectionMeshArtifact"
        || descriptors[0]["consumer_schema_version"]
            != meshlang_compiler::PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION
        || descriptors[0]["consumer_max_bytes"]
            != meshlang_compiler::MAX_PROJECTION_MESH_ARTIFACT_BYTES
    {
        return Err("projection backend descriptor contract".into());
    }
    checks += 1;
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let projected = body |> project_iso_surface(tolerance: 0.01)
        }
    "#;
    let geo = compile_to_geo_full(source, "projection-metadata-geo.mlg")
        .map_err(|error| format!("metadata-only GEO projection: {error}"))?;
    let python = compile_to_python_full(source, "projection-metadata-python.mlg")
        .map_err(|error| format!("metadata-only Python projection: {error}"))?;
    if !geo.code.contains("metadata-only project_iso_surface")
        || !python.code.contains("metadata-only project_iso_surface")
    {
        return Err("metadata-only backend projection marker missing".into());
    }
    let geo_metadata: serde_json::Value = serde_json::from_str(&geo.backend_metadata_json)
        .map_err(|error| format!("GEO metadata JSON: {error}"))?;
    let python_metadata: serde_json::Value = serde_json::from_str(&python.backend_metadata_json)
        .map_err(|error| format!("Python metadata JSON: {error}"))?;
    if geo_metadata["projections"][0]["native_geometry"] != false
        || python_metadata["projections"][0]["native_geometry"] != false
    {
        return Err("metadata-only projection descriptor claims native geometry".into());
    }
    checks += 1;

    let artifact = compile_projection_mesh_artifact(
        source,
        "projection-source.mlg",
        "model.projected",
        [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        [4, 4, 4],
        0.01,
    )
    .map_err(|error| format!("source projection artifact: {error}"))?;
    if artifact.mesh.vertices.is_empty() || artifact.mesh.triangles.is_empty() {
        return Err("source projection artifact produced no mesh".into());
    }
    checks += 1;

    let result = evaluate_projection(&parameterized, id(100), [3.0, 0.0, 0.0], 1e-6)
        .map_err(|error| format!("parameter resolution: {error}"))?;
    if (result.signed_distance - 1.0).abs() > 1e-12 {
        return Err(format!(
            "parameterized distance was {}",
            result.signed_distance
        ));
    }
    checks += 1;

    let dynamic = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(
            IrArgumentName::Radius,
            IrValue::Call {
                function: SymbolId(9),
                arguments: Vec::new(),
            },
        )],
    );
    let error = evaluate_projection(
        &geometry(vec![
            dynamic,
            project(id(1), ProjectionMethod::SignedDistance),
        ]),
        id(100),
        [0.0, 0.0, 0.0],
        1e-6,
    )
    .expect_err("dynamic projection must fail explicitly");
    if !matches!(error, ProjectionError::DynamicArgument { .. }) {
        return Err(format!("unexpected dynamic error: {error}"));
    }
    checks += 1;

    let sphere = node(
        id(1),
        GeometryOperation::Primitive {
            kind: PrimitiveKind::Sphere,
        },
        &[],
        vec![argument(IrArgumentName::Radius, scalar(1.0))],
    );
    let sampled = mesh_projection_iso_surface(
        &geometry(vec![sphere, project(id(1), ProjectionMethod::IsoSurface)]),
        id(100),
        [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
        [8, 8, 8],
        1.0e-8,
    )
    .map_err(|error| format!("iso-surface sampling: {error}"))?;
    if sampled.vertices.is_empty() || sampled.triangles.is_empty() {
        return Err("iso-surface sampling produced no triangles".into());
    }
    if sampled.triangles.iter().any(|triangle| {
        triangle
            .iter()
            .any(|index| *index as usize >= sampled.vertices.len())
    }) {
        return Err("iso-surface triangle references an out-of-range vertex".into());
    }
    checks += 2;

    let artifact = ProjectionMeshArtifact::from_mesh(sampled)
        .map_err(|error| format!("iso-surface artifact validation: {error}"))?;
    let encoded = artifact
        .to_json()
        .map_err(|error| format!("iso-surface artifact serialization: {error}"))?;
    let decoded = ProjectionMeshArtifact::from_json(&encoded)
        .map_err(|error| format!("iso-surface artifact round-trip: {error}"))?;
    if decoded.schema_version != artifact.schema_version
        || decoded.mesh.cells != artifact.mesh.cells
        || decoded.mesh.triangles != artifact.mesh.triangles
        || decoded.mesh.vertices.len() != artifact.mesh.vertices.len()
        || decoded
            .mesh
            .vertices
            .iter()
            .zip(&artifact.mesh.vertices)
            .any(|(decoded_vertex, original_vertex)| {
                decoded_vertex.iter().zip(original_vertex).any(
                    |(decoded_coordinate, original_coordinate)| {
                        (decoded_coordinate - original_coordinate).abs() > 1.0e-15
                    },
                )
            })
    {
        return Err("iso-surface artifact round-trip changed the mesh".into());
    }
    checks += 2;

    Ok(QaResult::passed(CASE_ID)
        .metric("primitive_signed_distance", 2.0)
        .metric("closest_point_and_iso_surface", 3.0)
        .metric("transform_and_cylinder", 1.0)
        .metric("boolean_difference", 1.0)
        .metric("parameter_resolution", 1.0)
        .metric("explicit_dynamic_rejection", 1.0)
        .metric("iso_surface_mesh_checks", 2.0)
        .metric("iso_surface_artifact_round_trip", 2.0)
        .metric("native_metadata_projection", 1.0)
        .metric("checks", checks as f64)
        .input_checksum("analytic/deterministic-fixture:meshlang-projection-v1")
        .reference("bounded canonical primitive SDF and projection consumer")
        .reference_checksum("analytic/deterministic-fixture:meshlang-projection-v1")
        .dimension(
            "sphere, box, cylinder, boolean, transform, parameter, closest-point, iso-surface",
        )
        .formulation("MeshLang canonical GeometryOperation::Project evaluator")
        .tolerance("all signed-distance, closest-point, composition, and rejection checks pass")
        .seed("meshlang-projection-reference:cpu:v1"))
}

fn main() {
    let started = std::time::Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
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
