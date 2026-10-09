//! Release-facing parser grammar, recovery, and JSON-envelope reference.

use meshlang_ast::{BinaryOp, Declaration, Expr, Statement};
use meshlang_parser::error::DiagnosticCode;
use meshlang_parser::{ParseOptions, lex, parse, parse_json, parse_recovering};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.parser-contract-reference.v1";

const CONSTRUCTOR_SOURCE: &str = r#"
geometry model {
    let shape = Box(1.0, 1.0, 1.0) | Cylinder(radius: 0.25, height: 1.0) |> translate(1.0, 0.0, 0.0)
}
"#;

const LINE_SOURCE: &str = r#"
geometry model {
    let start = Point(0.0, 0.0, 0.0)
    let finish = Point(1.0, 0.0, 0.0)
    let edge = Line(from: start, to: finish)
}
"#;

fn run() -> Result<QaResult, String> {
    let file = parse(CONSTRUCTOR_SOURCE, "parser-contract-reference.mlg")
        .map_err(|error| format!("constructor parse: {error}"))?;
    let line_constructor_parsed = parse(LINE_SOURCE, "line-from-argument.mlg").is_ok();
    let Declaration::Geometry(geometry) = &file.declarations[0] else {
        return Err("expected geometry declaration".into());
    };
    let Statement::Let(binding) = &geometry.body.statements[0] else {
        return Err("expected geometry binding".into());
    };
    let Expr::Pipe(pipe) = &binding.value else {
        return Err("constructor expression did not retain pipe AST".into());
    };
    let Expr::Binary(binary) = pipe.value.as_ref() else {
        return Err("constructor expression did not retain CSG AST".into());
    };
    if binary.op != BinaryOp::CsgUnion {
        return Err(format!("expected CSG union, found {:?}", binary.op));
    }

    let overflow = parse(
        "param count = 999999999999999999999999999999999999",
        "overflow.mlg",
    )
    .expect_err("integer overflow must be rejected");
    if overflow.code != DiagnosticCode::InvalidNumber {
        return Err(format!("overflow diagnostic was {:?}", overflow.code));
    }

    let mut named_mesh_targets = 0usize;
    for (fixture, source) in [
        ("mesh-target-direct.mlg", "mesh model { }"),
        ("mesh-target-for.mlg", "mesh for model { }"),
    ] {
        let file = parse(source, fixture).map_err(|error| format!("{fixture} parse: {error}"))?;
        let Declaration::Mesh(mesh) = &file.declarations[0] else {
            return Err(format!("{fixture} did not produce a mesh declaration"));
        };
        if mesh.target.as_ref().map(|target| target.name.as_str()) != Some("model") {
            return Err(format!("{fixture} lost its mesh target"));
        }
        named_mesh_targets += 1;
    }

    let empty_compound = parse(
        "geometry model { Box(1.0, 1.0, 1.0) }\nmesh model { compound surface() }",
        "empty-compound.mlg",
    )
    .expect_err("an empty mesh compound must be rejected");
    if empty_compound.code != DiagnosticCode::UnexpectedToken {
        return Err(format!(
            "empty compound diagnostic was {:?}",
            empty_compound.code
        ));
    }

    let options = ParseOptions::default();
    let mut corpus_cases = 0usize;
    for fixture in [
        "empty-kv-args.mlg",
        "nested-comment.mlg",
        "recovery.mlg",
        "unterminated-string.mlg",
    ] {
        let source = match fixture {
            "empty-kv-args.mlg" => include_str!(
                "../../sonium-meshlang/crates/meshlang-parser/fuzz/corpus/parse_recovering/empty-kv-args.mlg"
            ),
            "nested-comment.mlg" => include_str!(
                "../../sonium-meshlang/crates/meshlang-parser/fuzz/corpus/parse_recovering/nested-comment.mlg"
            ),
            "recovery.mlg" => include_str!(
                "../../sonium-meshlang/crates/meshlang-parser/fuzz/corpus/parse_recovering/recovery.mlg"
            ),
            "unterminated-string.mlg" => include_str!(
                "../../sonium-meshlang/crates/meshlang-parser/fuzz/corpus/parse_recovering/unterminated-string.mlg"
            ),
            _ => unreachable!(),
        };
        let report = parse_recovering(source, fixture, options);
        if report.diagnostics.len() > options.max_diagnostics {
            return Err(format!("{fixture} exceeded diagnostic bound"));
        }
        corpus_cases += 1;
    }

    let envelope: serde_json::Value = serde_json::from_str(
        &parse_json(CONSTRUCTOR_SOURCE, "parser-contract-reference.mlg", options)
            .map_err(|error| format!("parse JSON: {error}"))?,
    )
    .map_err(|error| format!("parse JSON envelope: {error}"))?;
    let checks = [
        matches!(binary.left.as_ref(), Expr::Constructor(_)),
        matches!(binary.right.as_ref(), Expr::Constructor(_)),
        envelope["parse_schema"].is_number(),
        envelope["grammar_version"].is_string(),
        envelope["diagnostic_schema"].is_number(),
        corpus_cases == 4,
        named_mesh_targets == 2,
        empty_compound.code == DiagnosticCode::UnexpectedToken,
        line_constructor_parsed,
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "parser contract checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    let operator_checks = meshlang_parser::grammar::OPERATOR_SPELLINGS
        .iter()
        .filter(|spelling| {
            lex(spelling)
                .into_iter()
                .filter(|token| token.kind != meshlang_parser::lexer::TokenKind::Eof)
                .count()
                == 1
        })
        .count();

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("operator_spellings_checked", operator_checks as f64)
        .metric("recovery_corpus_cases", corpus_cases as f64)
        .metric("named_mesh_targets", named_mesh_targets as f64)
        .minimum("checks_passed", checks.len() as f64)
        .minimum(
            "operator_spellings_checked",
            meshlang_parser::grammar::OPERATOR_SPELLINGS.len() as f64,
        )
        .minimum("recovery_corpus_cases", 4.0)
        .input_checksum("source:meshlang-parser-contract-reference:v1")
        .reference("normative operator lexing, constructor CSG/pipeline AST, mesh target spellings, empty-compound diagnostics, overflow diagnostics, bounded recovery, and parse JSON envelope")
        .reference_checksum("analytic/deterministic-fixture:meshlang-parser-contract-v1")
        .dimension("MeshLang parser grammar and diagnostic wire contract")
        .formulation("recursive-descent parser with bounded recovering diagnostics")
        .tolerance("exact contract checks; no more than the configured diagnostic bound")
        .seed("meshlang-parser-contract-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang parser result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, &error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("MeshLang parser failure must serialize")
            );
            println!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
