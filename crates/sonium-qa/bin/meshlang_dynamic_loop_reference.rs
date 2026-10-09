//! Bounded MeshLang runtime-dynamic loop IR and backend parity reference.

use meshlang_ast::IrValue;
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full, lower_to_canonical_ir};
use meshlang_parser::parse;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.dynamic-loop-reference.v1";

fn run() -> Result<QaResult, String> {
    let source = r#"
        param count = 3
        param width = 2
        geometry model {
            for i in 0..count, j in 0..=1 {
                let body = Box(i + j + 1, 1.0, 1.0)
                for k in 0..width {
                    let nested = Box(0.5, 0.5, 0.5)
                }
            }
        }
        physics {
            group all: volume = model.body
        }
    "#;
    let ast =
        parse(source, "dynamic-loop-reference.mlg").map_err(|error| format!("parse: {error}"))?;
    let ir = lower_to_canonical_ir(&ast).map_err(|error| format!("lower: {error}"))?;
    ir.validate()
        .map_err(|errors| format!("IR validation: {errors:?}"))?;
    let geo = compile_to_geo_full(source, "dynamic-loop-reference.mlg")
        .map_err(|error| format!("GEO compile: {error}"))?;
    let python = compile_to_python_full(source, "dynamic-loop-reference.mlg")
        .map_err(|error| format!("Python compile: {error}"))?;

    let clauses = ir.loops.first().map_or(0, |control| control.clauses.len());
    let checks = [
        ir.loops.len() == 2,
        clauses == 2,
        ir.loops[0].clauses[0].name == "i",
        ir.loops[0].clauses[1].name == "j",
        matches!(ir.loops[0].clauses[0].start, IrValue::Integer(0)),
        matches!(ir.loops[0].clauses[0].end, IrValue::Parameter(_)),
        ir.loops[0].clauses[1].inclusive,
        !ir.loops[0].entities.is_empty(),
        geo.code.contains("For i In {0:(count) - 1}"),
        geo.code.contains("For j In {0:1}"),
        python.code.contains("for i in range(0, count):"),
        python.code.contains("for j in range(0, (1) + 1):"),
        geo.code.contains("__ml_loop_1_0[] = {}"),
        geo.code.contains("__ml_loop_2_0[] = {}"),
        geo.code.contains("__ml_loop_2_0[] += {model__nested}"),
        geo.code.contains("__ml_loop_1_0[] += __ml_loop_2_0[]"),
        python.code.contains("__ml_loop_1_0 = []"),
        python.code.contains("__ml_loop_2_0 = []"),
        python
            .code
            .contains("__ml_loop_2_0.extend(_ml_dimtags(3, nested))"),
        python.code.contains("__ml_loop_1_0.extend(__ml_loop_2_0)"),
        geo.code.contains("Physical Volume(\"all\")"),
        python.code.contains("pg_all"),
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "dynamic-loop checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("loop_count", ir.loops.len() as f64)
        .metric("clause_count", clauses as f64)
        .metric("entity_count", ir.loops[0].entities.len() as f64)
        .minimum("checks_passed", checks.len() as f64)
        .minimum("loop_count", 1.0)
        .minimum("clause_count", 2.0)
        .minimum("entity_count", 1.0)
    .input_checksum("source:dynamic-nested-loop-aggregation:meshlang-v2")
        .reference("canonical loop IR plus GEO/Python runtime-loop emission parity")
    .reference_checksum("analytic/deterministic-fixture:meshlang-dynamic-loop-v2")
    .dimension("nested dynamic two-clause loop with aggregate physical-group consumption")
        .formulation("MeshLang runtime-dynamic loop lowering, validation, materialization, and backend emission")
        .tolerance("twelve deterministic loop-contract checks")
        .seed("meshlang-dynamic-loop-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang dynamic-loop result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang dynamic-loop failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
