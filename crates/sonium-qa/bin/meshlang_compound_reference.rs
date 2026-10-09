//! CPU-only canonical mesh-compound backend parity reference.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.compound-reference.v1";

const SOURCE: &str = r#"
    geometry pieces {
        let left = Box(1.0, 1.0, 1.0)
        let right = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
    }
    mesh pieces {
        compound volume(pieces.left, pieces.right)
    }
"#;

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let geo = compile_to_geo_full(SOURCE, "compound-reference.mlg")
        .map_err(|error| format!("GEO compound compilation failed: {error}"))?;
    let python = compile_to_python_full(SOURCE, "compound-reference.mlg")
        .map_err(|error| format!("Python compound compilation failed: {error}"))?;

    let canonical: serde_json::Value = serde_json::from_str(&geo.canonical_ir_json)
        .map_err(|error| format!("canonical IR JSON is invalid: {error}"))?;
    let compound_count = canonical
        .pointer("/ir/mesh")
        .and_then(|mesh| mesh.get("compounds"))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if compound_count != 1 {
        return Err(format!(
            "canonical IR expected one mesh compound, found {compound_count}"
        ));
    }

    let geo_count = geo.code.matches("Compound Volume").count();
    let python_count = python.code.matches("gmsh.model.mesh.setCompound(3").count();
    if geo_count != 1 || python_count != 1 {
        return Err(format!(
            "canonical compound emission count drifted: geo={geo_count}, python={python_count}"
        ));
    }
    if !geo
        .code
        .contains("Compound Volume {pieces__left, pieces__right};")
        || !python
            .code
            .contains("gmsh.model.mesh.setCompound(3, [left, right])")
    {
        return Err("canonical compound dimension/name parity drifted".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("canonical_compounds", compound_count as f64)
        .metric("geo_compounds", geo_count as f64)
        .metric("python_compounds", python_count as f64)
        .metric("geo_bytes", geo.code.len() as f64)
        .metric("python_bytes", python.code.len() as f64)
        .dimension("3-D")
        .formulation("canonical mesh-control authority")
        .input_checksum("source:meshlang-compound-reference:v1")
        .reference_checksum("geo-python-canonical-compound-parity:v1")
        .seed("meshlang-compound-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "{}",
        result
            .to_json_line()
            .expect("QA result serialization must succeed")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
