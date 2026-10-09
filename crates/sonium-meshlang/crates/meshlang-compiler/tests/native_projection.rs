use meshlang_compiler::{
    compile_projection_native_artifacts, projection_mesh_to_native, ProjectionMesh,
    ProjectionMeshArtifact,
};

#[test]
fn native_projection_sources_are_deterministic_and_executable_shaped() {
    let source = r#"
        geometry model {
            let body = Sphere(r: 0.65)
            let projected = body |> project_iso_surface(tolerance: 0.02)
        }
    "#;
    let first = compile_projection_native_artifacts(
        source,
        "native-projection.mlg",
        "model.projected",
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [8, 8, 8],
        0.02,
    )
    .expect("native projection sources");
    let second = compile_projection_native_artifacts(
        source,
        "native-projection.mlg",
        "model.projected",
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [8, 8, 8],
        0.02,
    )
    .expect("repeat native projection sources");

    assert_eq!(first, second);
    assert!(first.vertex_count > 0);
    assert!(first.triangle_count > 0);
    assert!(first.geo_source.contains("Point(1)"));
    assert!(first.geo_source.contains("Line Loop("));
    assert!(first.geo_source.contains("Plane Surface("));
    assert!(first
        .geo_source
        .contains("Physical Surface(\"MeshLangProjection\")"));
    assert!(first.python_source.contains("gmsh.initialize()"));
    assert!(first
        .python_source
        .contains("gmsh.model.geo.addPlaneSurface"));
    assert!(first.python_source.contains("gmsh.model.mesh.generate(2)"));
    assert!(!first.geo_source.contains("metadata-only"));
}

#[test]
fn native_projection_rejects_non_finite_artifacts_before_emission() {
    let artifact = ProjectionMeshArtifact {
        schema_version: meshlang_compiler::PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
        mesh: ProjectionMesh {
            vertices: vec![[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            bounds: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cells: [1, 1, 1],
        },
    };
    let error = projection_mesh_to_native(&artifact).expect_err("non-finite artifact");
    assert!(error.to_string().contains("invalid projection artifact"));
}
