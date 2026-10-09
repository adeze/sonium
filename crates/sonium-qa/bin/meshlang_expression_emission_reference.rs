//! CPU-only release reference for scalar expression emission in both backends.

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.expression-emission-reference.v1";
const SOURCE: &str = r#"
param pick = [1, 2, 3][1]
param grouped = (1 + 2)

geometry test {
    let body = Box(pick, grouped, [1, 2, 3][1])
}
"#;

fn run() -> Result<QaResult, String> {
    let geo = compile_to_geo_full(SOURCE, "expression-emission-reference.mlg")
        .map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, "expression-emission-reference.mlg")
        .map_err(|error| error.to_string())?;

    let checks = [
        geo.code.contains("{1, 2, 3}[1]"),
        geo.code.contains("(1 + 2)"),
        python.code.contains("[1, 2, 3][1]"),
        python.code.contains("(1 + 2)"),
        geo.canonical_ir_hash == python.canonical_ir_hash,
        geo.canonical_ir_json == python.canonical_ir_json,
    ];
    let passed = checks.iter().filter(|check| **check).count();
    if passed != checks.len() {
        return Err(format!(
            "expression emission checks: {passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("checks", passed as f64)
        .metric("geo_array_index", 1.0)
        .metric("python_array_index", 1.0)
        .input_checksum("source:meshlang-expression-emission-reference:v1")
        .reference_checksum("deterministic-backend-expression-parity:v1")
        .formulation("GEO/Python scalar expression emission for parentheses, arrays, and indexing")
        .seed("meshlang-expression-emission-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result must serialize")
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
                    .expect("QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
