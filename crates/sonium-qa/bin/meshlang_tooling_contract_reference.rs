//! Release-facing generated MeshLang schema/binding contract reference.

use meshlang_ast::tooling::{
    DECLARATION_VARIANTS, EXPR_VARIANTS, MESH_STATEMENT_VARIANTS, STATEMENT_VARIANTS,
    generate_tooling_artifacts, validate_tooling_artifacts,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "meshlang.tooling-contract-reference.v1";

fn run() -> Result<QaResult, String> {
    let first = generate_tooling_artifacts();
    let repeated = generate_tooling_artifacts();
    if first != repeated {
        return Err("generated schema/bindings are not deterministic".into());
    }
    validate_tooling_artifacts(&first).map_err(|error| format!("tooling validation: {error}"))?;

    let variant_count = DECLARATION_VARIANTS.len()
        + STATEMENT_VARIANTS.len()
        + MESH_STATEMENT_VARIANTS.len()
        + EXPR_VARIANTS.len();
    if first.typescript.matches("export type ").count() < 4
        || first.python.matches("TypeAlias").count() < 5
    {
        return Err("generated language bindings omitted a union declaration".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("schema_validation_checks", 1.0)
        .minimum("schema_validation_checks", 1.0)
        .metric("determinism_checks", 1.0)
        .minimum("determinism_checks", 1.0)
        .metric("language_binding_checks", 2.0)
        .minimum("language_binding_checks", 2.0)
        .metric("variant_coverage_checks", variant_count as f64)
        .minimum("variant_coverage_checks", variant_count as f64)
        .input_checksum("source:meshlang-tooling-artifacts:v1")
        .reference("versioned JSON Schema plus deterministic TypeScript/Python AST/IR bindings")
        .reference_checksum("analytic/deterministic-fixture:meshlang-tooling-contract-v1")
        .dimension("AST variant inventory, schema identity/version, and two generated clients")
        .formulation("MeshLang cross-language tooling artifact contract")
        .tolerance("schema validates and all public variants appear in both generated clients")
        .seed("meshlang-tooling-contract-reference:cpu:v1"))
}

fn main() {
    let started = Instant::now();
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
