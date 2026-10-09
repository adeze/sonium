use meshlang_compiler::{
    compile_projection_native_volume_artifacts, projection_mesh_to_native_volume, ProjectionMesh,
    ProjectionMeshArtifact, PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
};

fn tetrahedron_artifact() -> ProjectionMeshArtifact {
    ProjectionMeshArtifact {
        schema_version: PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
        mesh: ProjectionMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            bounds: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cells: [1, 1, 1],
        },
    }
}

#[test]
fn native_volume_sources_emit_one_closed_gmsh_volume() {
    let artifacts = projection_mesh_to_native_volume(&tetrahedron_artifact())
        .expect("oriented tetrahedron is a valid closed shell");
    assert_eq!(artifacts.vertex_count, 4);
    assert_eq!(artifacts.triangle_count, 4);
    assert_eq!(artifacts.volume_count, 1);
    assert!(artifacts.geo_source.contains("Surface Loop(1)"));
    assert!(artifacts.geo_source.contains("Volume(1)"));
    assert!(artifacts.geo_source.contains("Mesh 3;"));
    assert!(artifacts
        .python_source
        .contains("gmsh.model.mesh.generate(3)"));
    assert!(artifacts
        .python_source
        .contains("addPhysicalGroup(3, [v1], 2)"));
}

#[test]
fn native_volume_sources_are_deterministic() {
    let artifact = tetrahedron_artifact();
    let first = projection_mesh_to_native_volume(&artifact).expect("volume source");
    let second = projection_mesh_to_native_volume(&artifact).expect("volume source");
    assert_eq!(first, second);
}

#[test]
fn native_volume_rejects_open_shells_before_emission() {
    let mut artifact = tetrahedron_artifact();
    artifact.mesh.triangles.pop();
    let error = projection_mesh_to_native_volume(&artifact).expect_err("open shell must fail");
    assert!(error.to_string().contains("every shell edge exactly twice"));
}

#[test]
fn native_volume_rejects_inconsistent_or_disconnected_shells() {
    let mut inconsistent = tetrahedron_artifact();
    inconsistent.mesh.triangles[0] = [0, 1, 2];
    let error = projection_mesh_to_native_volume(&inconsistent)
        .expect_err("inconsistent orientation must fail");
    assert!(error.to_string().contains("inconsistent orientation"));

    let mut disconnected = tetrahedron_artifact();
    disconnected.mesh.vertices.extend([
        [2.0, 2.0, 2.0],
        [3.0, 2.0, 2.0],
        [2.0, 3.0, 2.0],
        [2.0, 2.0, 3.0],
    ]);
    disconnected
        .mesh
        .triangles
        .extend([[4, 6, 5], [4, 5, 7], [4, 7, 6], [5, 6, 7]]);
    disconnected.mesh.bounds = [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]];
    let error =
        projection_mesh_to_native_volume(&disconnected).expect_err("disconnected shells must fail");
    assert!(error.to_string().contains("one connected closed shell"));
}

#[test]
fn native_volume_source_wrapper_rejects_non_manifold_projection() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let projected = body |> project_iso_surface(tolerance: 0.05)
        }
    "#;
    let error = compile_projection_native_volume_artifacts(
        source,
        "volume.mlg",
        "model.projected",
        [[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]],
        [4, 4, 4],
        0.05,
    )
    .expect_err("volume emission must reject a non-manifold generated projection");
    assert!(error.to_string().contains("every shell edge exactly twice"));
}
