//! Release-facing parser/tree-sitter grammar parity reference.

use meshlang_parser::grammar::{OPERATOR_SPELLINGS, PRECEDENCE_ROWS, validate_tree_sitter_grammar};
use meshlang_parser::parse;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.grammar-contract-reference.v1";
const TREE_SITTER_GRAMMAR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sonium-meshlang/tree-sitter-meshlang/grammar.js"
));

fn run() -> Result<QaResult, String> {
    validate_tree_sitter_grammar(TREE_SITTER_GRAMMAR)
        .map_err(|error| format!("tree-sitter grammar contract: {error}"))?;
    let contract_json = meshlang_parser::grammar_contract_json()
        .map_err(|error| format!("grammar contract JSON: {error}"))?;
    let contract: serde_json::Value = serde_json::from_str(&contract_json)
        .map_err(|error| format!("grammar contract JSON parse: {error}"))?;
    if contract["precedence_rows"].as_array().map_or(0, Vec::len) != PRECEDENCE_ROWS.len()
        || contract["operator_spellings"]
            .as_array()
            .map_or(0, Vec::len)
            != OPERATOR_SPELLINGS.len()
    {
        return Err("grammar contract JSON does not match Rust rows".into());
    }

    let expected_rows = [
        "pipe",
        "csg",
        "or",
        "and",
        "comparison",
        "additive",
        "multiplicative",
        "unary",
        "power",
        "postfix",
        "primary",
    ];
    if PRECEDENCE_ROWS
        .iter()
        .map(|(name, _, _)| *name)
        .ne(expected_rows)
    {
        return Err(format!(
            "unexpected Rust precedence rows: {PRECEDENCE_ROWS:?}"
        ));
    }
    if PRECEDENCE_ROWS
        .windows(2)
        .any(|rows| rows[0].1 >= rows[1].1)
        || PRECEDENCE_ROWS.iter().filter(|row| row.2).count() != 1
    {
        return Err("Rust precedence rows are not strictly ordered".into());
    }

    let expression = r#"
        geometry model {
            let left = Box(1.0, 1.0, 1.0)
            let right = Sphere(r: 0.5)
            let merged = left | right & left
            let powered = 2.0 ** 3.0 ** 2.0
        }
    "#;
    parse(expression, "grammar-contract-expression.mlg")
        .map_err(|error| format!("Rust parser expression fixture: {error}"))?;
    parse(
        "geometry model { for i in 0..3 { let body = Box(1.0, 1.0, 1.0) } }",
        "grammar-contract-range.mlg",
    )
    .map_err(|error| format!("Rust parser range fixture: {error}"))?;

    let csg_start = TREE_SITTER_GRAMMAR
        .find("csg_expr:")
        .ok_or_else(|| "tree-sitter csg rule missing after validation".to_owned())?;
    let csg_end = TREE_SITTER_GRAMMAR[csg_start..]
        .find("\n    or_expr:")
        .map(|offset| csg_start + offset)
        .unwrap_or(TREE_SITTER_GRAMMAR.len());
    let csg = &TREE_SITTER_GRAMMAR[csg_start..csg_end];
    if csg.contains("prec.left(4,") {
        return Err("CSG operators are not at one precedence level".into());
    }

    let result = QaResult::passed(CASE_ID)
        .metric("operator_spellings", OPERATOR_SPELLINGS.len() as f64)
        .metric("precedence_rows", PRECEDENCE_ROWS.len() as f64)
        .metric("parser_expression_fixtures", 2.0)
        .metric("tree_sitter_rule_checks", 10.0)
        .metric("versioned_contract_fields", 4.0)
        .input_checksum("source:meshlang-grammar-contract:v1")
        .reference("Rust recursive-descent and checked-in tree-sitter grammar parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-grammar-contract-v1")
        .dimension(
            "longest-match operators, CSG/logical precedence, power associativity, range clauses",
        )
        .formulation("MeshLang parser/editor grammar contract")
        .tolerance("all operator, precedence, parser-fixture, and CSG-level checks pass")
        .seed("meshlang-grammar-contract-reference:cpu:v1");
    Ok(result)
}

fn main() {
    let started = std::time::Instant::now();
    let (result, passed) = match run() {
        Ok(result) => (
            result.elapsed_ms(started.elapsed().as_millis() as u64),
            true,
        ),
        Err(error) => (
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
            false,
        ),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
