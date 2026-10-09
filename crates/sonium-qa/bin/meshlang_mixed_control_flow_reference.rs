//! Release-backed MeshLang mixed control-flow and loop-binding reference.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.mixed-control-flow-reference.v1";
const SOURCE: &str = r#"
    param radius = 0.5

    geometry model {
        if false {
            let skipped = Box(radius, 0.1, 0.1)
        } else {
            for i in 0..3 {
                let segment = Box(radius, 0.1, 0.1)
                    |> translate(i * 0.5, 0.0, 0.0)
            }
        }
    }
"#;

fn public_entity_names(canonical_ir_json: &str) -> Result<Vec<String>, String> {
    let ir = serde_json::from_str::<serde_json::Value>(canonical_ir_json)
        .map_err(|error| format!("canonical IR JSON: {error}"))?;
    ir["ir"]["entities"]
        .as_array()
        .ok_or_else(|| "canonical IR entities array missing".to_string())
        .map(|entities| {
            entities
                .iter()
                .filter_map(|entity| entity["name"].as_str())
                .filter(|name| !name.contains(".__ir_"))
                .map(str::to_string)
                .collect()
        })
}

fn run() -> Result<QaResult, String> {
    let geo = compile_to_geo_full(SOURCE, "mixed-control-flow-reference.mlg")
        .map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, "mixed-control-flow-reference.mlg")
        .map_err(|error| error.to_string())?;
    let names = public_entity_names(&geo.canonical_ir_json)?;

    let checks = [
        geo.canonical_ir_hash == python.canonical_ir_hash,
        geo.canonical_ir_json == python.canonical_ir_json,
        geo.provenance_manifest_json == python.provenance_manifest_json,
        geo.sensitivity_manifest_json == python.sensitivity_manifest_json,
        geo.code.matches("Box(").count() == 3,
        python.code.matches("occ.addBox").count() == 3,
        ["model.segment_0", "model.segment_1", "model.segment_2"]
            .iter()
            .all(|name| names.iter().any(|actual| actual == name)),
        !names.iter().any(|name| name == "model.skipped"),
        !geo.code.contains("skipped") && !python.code.contains("skipped"),
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "mixed control-flow parity checks: {checks_passed}/{}; names={names:?}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("geo_box_count", geo.code.matches("Box(").count() as f64)
        .metric(
            "python_box_count",
            python.code.matches("occ.addBox").count() as f64,
        )
        .minimum("checks_passed", checks.len() as f64)
        .minimum("geo_box_count", 3.0)
        .minimum("python_box_count", 3.0)
        .input_checksum("source:mixed-control-flow-reference.mlg:v1")
        .reference("deterministic canonical IR GEO/Python parity for if/else loop bindings")
        .reference_checksum("analytic/deterministic-fixture:meshlang-mixed-control-flow-v1")
        .dimension("one static false branch, one else branch, three compile-time loop bindings")
        .formulation("MeshLang parser/compiler canonical IR GEO/Python lowering")
        .tolerance("nine parity checks and three loop-produced Box bindings")
        .seed("meshlang-mixed-control-flow-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("MeshLang failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
