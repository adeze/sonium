//! CPU-only native MeshLang projection volume-source reference.

use meshlang_compiler::{
    PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION, ProjectionMesh, ProjectionMeshArtifact,
    projection_mesh_to_native_volume,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.projection-native-volume-reference.v1";

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

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let artifacts = projection_mesh_to_native_volume(&tetrahedron_artifact())
        .map_err(|error| format!("native projection volume adapter failed: {error}"))?;
    if artifacts.vertex_count != 4
        || artifacts.triangle_count != 4
        || artifacts.volume_count != 1
        || !artifacts.geo_source.contains("Mesh 3;")
        || !artifacts
            .python_source
            .contains("gmsh.model.mesh.generate(3)")
    {
        return Err("native projection volume source contract drifted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("vertices", artifacts.vertex_count as f64)
        .metric("triangles", artifacts.triangle_count as f64)
        .metric("volumes", artifacts.volume_count as f64)
        .metric("geo_bytes", artifacts.geo_source.len() as f64)
        .metric("python_bytes", artifacts.python_source.len() as f64)
        .dimension("3-D closed projection shell to native volume source")
        .formulation("checked connected consistently oriented manifold admission")
        .input_checksum("oriented-tetrahedron-projection-shell:v1")
        .reference_checksum("native-gmsh-volume-source-contract:v1")
        .seed("meshlang-projection-native-volume:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
