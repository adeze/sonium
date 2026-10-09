//! Release-facing reference for non-ported multi-statement MeshLang templates.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.nonported-multistatement-template.v1";
const FIXTURE_NAME: &str = "template-multi.mlg";
const SOURCE: &str = r#"
    pub geometry Pair(width: Float) {
        let left = Box(width, 1.0, 1.0)
        let right = Box(width, 0.5, 0.5)
        left | right
    }
    geometry model {
        let pair = Pair(2.0)
    }
"#;

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let geo = compile_to_geo_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;
    let geo_repeat =
        compile_to_geo_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;
    let python_repeat =
        compile_to_python_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;

    let checks = [
        (
            "geo_helper_primitives",
            geo.code.matches("Box(").count() == 2,
        ),
        (
            "geo_boolean_union",
            geo.code.matches("BooleanUnion").count() == 1,
        ),
        (
            "python_helper_primitives",
            python.code.matches("occ.addBox").count() == 2,
        ),
        (
            "python_boolean_union",
            python.code.matches("occ.fuse(").count() == 1,
        ),
        (
            "canonical_left_provenance",
            geo.canonical_ir_json.contains("model.pair.left")
                && python.canonical_ir_json.contains("model.pair.left"),
        ),
        (
            "canonical_right_provenance",
            geo.canonical_ir_json.contains("model.pair.right")
                && python.canonical_ir_json.contains("model.pair.right"),
        ),
        (
            "template_fixture_provenance",
            geo.provenance_manifest_json.contains(FIXTURE_NAME)
                && python.provenance_manifest_json.contains(FIXTURE_NAME),
        ),
        (
            "deterministic_geo_repeat",
            geo.canonical_ir_json == geo_repeat.canonical_ir_json,
        ),
        (
            "deterministic_python_repeat",
            python.canonical_ir_json == python_repeat.canonical_ir_json,
        ),
    ];

    let failed = checks
        .iter()
        .filter_map(|(name, passed)| (!passed).then_some(*name))
        .collect::<Vec<_>>();
    if !failed.is_empty() {
        return Err(format!("failed checks: {}", failed.join(", ")));
    }

    Ok(checks.into_iter().map(|(name, _)| (name, 1.0)).collect())
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("compiler regression fixture template-multi.mlg")
                .reference_checksum("analytic/deterministic-fixture:meshlang-template-v1")
                .dimension("MeshLang geometry template")
                .formulation(
                    "GEO and Gmsh Python backend parity with canonical IR/provenance checks",
                )
                .tolerance(
                    "two Box emissions, one BooleanUnion/occ.fuse, provenance present, repeat canonical IR identical",
                )
                .timing_ms("compile_and_repeat", started.elapsed().as_millis() as u64)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value).minimum(name, 1.0);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("MESHLANG TEMPLATE REFERENCE FAILED");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
