//! CPU-only MeshLang canonical-materialization source-span contract.
//!
//! This exercises the compiler's source-only materialization boundary. It
//! does not emit GEO/Python, invoke Gmsh, or start a simulation.

use meshlang_ast::span::Span;
use meshlang_ast::{AstNode, Expr, LetStatement, SourceFile, Statement};
use meshlang_compiler::materialize_canonical_geometry;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.materialized-span-reference.v1";
const SOURCE: &str = r#"
geometry model {
    /// canonical primitive source
    let body = Box(1.0, 2.0, 3.0)
    /// canonical transform source
    let moved = body |> translate(1.0, 2.0, 3.0)
}
"#;

fn binding<'a>(file: &'a SourceFile, name: &str) -> Option<&'a LetStatement> {
    file.declarations.iter().find_map(|declaration| {
        let meshlang_ast::Declaration::Geometry(geometry) = declaration else {
            return None;
        };
        geometry.body.statements.iter().find_map(|statement| {
            let Statement::Let(binding) = statement else {
                return None;
            };
            (binding.name.name == name).then_some(binding)
        })
    })
}

fn run() -> Result<QaResult, String> {
    let materialized = materialize_canonical_geometry(SOURCE, "materialized-span.mlg")
        .map_err(|error| format!("canonical materialization: {error}"))?;
    let parsed = meshlang_parser::parse(SOURCE, "materialized-span.mlg")
        .map_err(|error| format!("source parse: {error}"))?;
    let source_body = binding(&parsed, "body").ok_or("missing source body binding")?;
    let source_moved = binding(&parsed, "moved").ok_or("missing source moved binding")?;
    let body = binding(&materialized, "body").ok_or("missing materialized body binding")?;
    let moved = binding(&materialized, "moved").ok_or("missing materialized moved binding")?;
    let Expr::Constructor(body_constructor) = &body.value else {
        return Err("body was not canonicalized to a constructor".into());
    };
    let Expr::Pipe(moved_pipe) = &moved.value else {
        return Err("moved was not canonicalized to a pipe".into());
    };

    let checks = [
        body_constructor.span != Span::DUMMY,
        moved_pipe.span != Span::DUMMY,
        body_constructor.span == source_body.value.span(),
        moved_pipe.span == source_moved.value.span(),
        body_constructor.span.file_id == body.span.file_id,
        moved_pipe.span.file_id == moved.span.file_id,
        body_constructor.type_name.span == body_constructor.span,
        moved_pipe.function.span == moved_pipe.span,
        body_constructor
            .args
            .iter()
            .all(|argument| argument.span == body_constructor.span),
        moved_pipe
            .args
            .iter()
            .all(|argument| argument.span == moved_pipe.span),
        body_constructor.args.iter().all(|argument| {
            argument
                .name
                .as_ref()
                .is_none_or(|name| name.span == body_constructor.span)
        }),
        moved_pipe.args.iter().all(|argument| {
            argument
                .name
                .as_ref()
                .is_none_or(|name| name.span == moved_pipe.span)
        }),
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "materialized source-span checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("generated_binding_count", 2.0)
        .minimum("checks_passed", checks.len() as f64)
        .minimum("generated_binding_count", 2.0)
        .input_checksum("source:meshlang-materialized-source-spans:v1")
        .reference("canonical materialization preserves source spans on generated expressions")
        .reference_checksum("analytic/deterministic-fixture:meshlang-materialized-spans-v1")
        .dimension("one geometry block, one primitive, one transform")
        .formulation("canonical IR source-only materialization")
        .tolerance("twelve exact source-span, file, and generated-child parity checks")
        .seed("meshlang-materialized-span-reference:cpu:v1"))
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .elapsed_ms(started.elapsed().as_millis() as u64)
                    .to_json_line()
                    .expect("MeshLang QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .elapsed_ms(started.elapsed().as_millis() as u64)
                    .to_json_line()
                    .expect("MeshLang QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
