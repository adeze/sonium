//! Release-backed MeshLang multi-statement function-inlining reference.

use meshlang_ast::{BooleanOperation, GeometryOperation, VersionedCanonicalIr};
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.multi-statement-function-reference.v1";
const SOURCE: &str = r#"
fn cutout(base: Solid, r: Float) -> Solid {
    let ball = Sphere(r: r)
    base - ball
}

geometry test {
    let box_geo = Box(1.0, 1.0, 1.0)
    let result = cutout(box_geo, 0.25)
}

export { format: msh4 filename: "multi-statement-function.msh" }
"#;

fn run() -> Result<QaResult, String> {
    let geo = compile_to_geo_full(SOURCE, "multi-statement-function-reference.mlg")
        .map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, "multi-statement-function-reference.mlg")
        .map_err(|error| error.to_string())?;
    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .map_err(|error| format!("canonical IR JSON: {error}"))?
        .ir;
    let canonical_result = ir
        .entities
        .iter()
        .find(|entity| entity.name == "test.result")
        .ok_or_else(|| "canonical IR omitted multi-statement function result".to_string())?;

    let checks = [
        geo.canonical_ir_hash == python.canonical_ir_hash,
        geo.canonical_ir_json == python.canonical_ir_json,
        geo.provenance_manifest_json == python.provenance_manifest_json,
        geo.code.matches("BooleanDifference").count() == 1,
        python.code.matches("occ.cut(").count() == 1,
        matches!(
            &canonical_result.operation,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Difference
            }
        ),
        canonical_result.inputs.len() == 2,
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "multi-statement function backend parity failed: {checks_passed}/{} checks",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("geo_boolean_difference_count", 1.0)
        .metric("python_occ_cut_count", 1.0)
        .metric(
            "canonical_boolean_input_count",
            canonical_result.inputs.len() as f64,
        )
        .minimum("checks_passed", checks.len() as f64)
        .minimum("geo_boolean_difference_count", 1.0)
        .minimum("python_occ_cut_count", 1.0)
        .minimum("canonical_boolean_input_count", 2.0)
        .input_checksum("source:meshlang-multi-statement-function-reference.mlg:v1")
        .reference("deterministic sequential local binding substitution with GEO/Python CSG parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-multi-statement-function-v2")
        .dimension("Solid base, local Sphere binding, final CSG difference")
        .formulation("MeshLang parser/compiler function inlining and backend lowering")
        .tolerance("five parity and emission checks")
        .seed("meshlang-multi-statement-function-reference:cpu:v1"))
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
                QaResult::failed(CASE_ID, &error)
                    .to_json_line()
                    .expect("MeshLang failure result must serialize")
            );
            println!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
