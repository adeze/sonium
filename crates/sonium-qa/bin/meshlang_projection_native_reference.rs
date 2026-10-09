//! CPU-only native Gmsh execution reference for MeshLang projection output.

use meshlang_compiler::compile_projection_native_artifacts;
use sonium_qa::QaResult;
use std::process::Command;
use std::time::Instant;

const CASE_ID: &str = "meshlang.projection-native-reference.v1";

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let source = r#"
        geometry model {
            let body = Sphere(r: 0.65)
            let projected = body |> project_iso_surface(tolerance: 0.02)
        }
    "#;
    let artifacts = compile_projection_native_artifacts(
        source,
        "projection-native-reference.mlg",
        "model.projected",
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [8, 8, 8],
        0.02,
    )
    .map_err(|error| format!("native projection compilation: {error}"))?;
    if artifacts.vertex_count == 0 || artifacts.triangle_count == 0 {
        return Err("native projection artifact is empty".into());
    }

    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let geo_path = directory.path().join("projection.geo");
    let msh_path = directory.path().join("projection.msh");
    let python_path = directory.path().join("projection.py");
    std::fs::write(&geo_path, &artifacts.geo_source).map_err(|error| error.to_string())?;
    std::fs::write(&python_path, &artifacts.python_source).map_err(|error| error.to_string())?;

    let python_check = Command::new("python3")
        .args([
            "-c",
            "import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())",
        ])
        .arg(&python_path)
        .output()
        .map_err(|error| format!("python3 unavailable: {error}"))?;
    if !python_check.status.success() {
        return Err(format!(
            "generated Python source failed syntax validation: {}",
            String::from_utf8_lossy(&python_check.stderr)
        ));
    }

    let gmsh_command = std::env::var("GMSH_PATH")
        .or_else(|_| std::env::var("MESHLANG_GMSH"))
        .unwrap_or_else(|_| "gmsh".to_owned());
    let gmsh = Command::new(&gmsh_command)
        .args(["-2", "-format", "msh2", "-v", "0"])
        .arg(&geo_path)
        .args(["-o"])
        .arg(&msh_path)
        .output()
        .map_err(|error| format!("gmsh unavailable ({gmsh_command:?}): {error}"))?;
    if !gmsh.status.success() {
        return Err(format!(
            "generated GEO source failed Gmsh execution: {}",
            String::from_utf8_lossy(&gmsh.stderr)
        ));
    }
    let mesh_bytes = std::fs::read(&msh_path).map_err(|error| error.to_string())?;
    if mesh_bytes.len() < 32
        || !mesh_bytes
            .windows(b"$MeshFormat".len())
            .any(|window| window == b"$MeshFormat")
    {
        return Err("Gmsh did not produce a valid MSH output".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("native_vertices", artifacts.vertex_count as f64)
        .metric("native_triangles", artifacts.triangle_count as f64)
        .metric("gmsh_geo_execution", 1.0)
        .metric("python_syntax_validation", 1.0)
        .metric("msh_output_bytes", mesh_bytes.len() as f64)
        .dimension("canonical projection artifact to native GEO/Python/Gmsh")
        .input_checksum("source:sphere-iso-surface:8x8x8:meshlang-native-projection-v1")
        .reference_checksum("gmsh-native-surface-mesh-execution:v1")
        .formulation(
            "bounded canonical projection sampling with deterministic native source emission",
        )
        .tolerance("GEO executes, Python parses, and Gmsh emits a non-empty MSH2 file")
        .seed("meshlang-projection-native-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
