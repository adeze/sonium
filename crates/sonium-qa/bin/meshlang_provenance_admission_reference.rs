//! Bounded MeshLang canonical-IR provenance admission reference.

use meshlang_ast::{IrValidationError, VersionedCanonicalIr};
use meshlang_compiler::compile_to_geo_full;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.provenance-admission-reference.v1";

fn run() -> Result<QaResult, String> {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
        }
    "#;
    let result = compile_to_geo_full(source, "provenance-admission.mlg")
        .map_err(|error| format!("compiler output: {error}"))?;
    let mut document = VersionedCanonicalIr::from_json_slice(result.canonical_ir_json.as_bytes())
        .map_err(|error| format!("canonical IR decode: {error}"))?
        .0;

    let valid = document.ir.validate_provenance().is_ok();
    let entity_id = document
        .ir
        .entities
        .first()
        .map(|entity| entity.id)
        .ok_or_else(|| "compiler emitted no canonical entity".to_owned())?;

    document
        .ir
        .physical_groups
        .push(meshlang_ast::PhysicalGroup {
            id: meshlang_ast::PhysicalGroupId(777),
            name: "admission-group".into(),
            dimension: meshlang_ast::EntityDimension::Volume,
            entities: vec![entity_id],
            material: None,
            provenance: Vec::new(),
        });
    let missing_group = matches!(
        document.ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                IrValidationError::MissingProvenance { context }
                    if context == "physical group 777"
            ))
    );
    let group = document
        .ir
        .physical_groups
        .last_mut()
        .expect("synthetic group exists");
    group.provenance = vec![meshlang_ast::SourceReference {
        source: meshlang_ast::SourceId::named("provenance-admission.mlg"),
        node: meshlang_ast::NodeId(777),
        span: meshlang_ast::span::Span::new(9, 2),
    }];
    let inverted_group = matches!(
        document.ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                IrValidationError::InvalidProvenanceSpan { context }
                    if context == "physical group 777"
            ))
    );
    document.ir.physical_groups.pop();

    let saved = {
        let entity = document
            .ir
            .entities
            .first_mut()
            .expect("entity existence checked above");
        let saved = entity.provenance.clone();
        entity.provenance.clear();
        saved
    };
    let missing = matches!(
        document.ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                IrValidationError::MissingEntityProvenance { entity }
                    if *entity == entity_id
            ))
    );

    {
        let entity = document
            .ir
            .entities
            .first_mut()
            .expect("entity existence checked above");
        entity.provenance = saved;
        let Some(reference) = entity.provenance.first_mut() else {
            return Err("compiler entity had no source reference".into());
        };
        reference.span.start = reference.span.end.saturating_add(1);
    }
    let inverted = matches!(
        document.ir.validate_provenance(),
        Err(errors)
            if errors.iter().any(|error| matches!(
                error,
                IrValidationError::InvalidEntityProvenanceSpan { entity }
                    if *entity == entity_id
            ))
    );

    let checks = [valid, missing, inverted, missing_group, inverted_group];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "provenance admission checks: {checks_passed}/{}",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("entity_count", document.ir.entities.len() as f64)
        .minimum("checks_passed", checks.len() as f64)
        .minimum("entity_count", 1.0)
        .input_checksum("source:single-box:meshlang-provenance-admission-v1")
        .reference("compiler-produced canonical IR plus typed provenance tamper mutations")
        .reference_checksum("analytic/deterministic-fixture:meshlang-provenance-admission-v1")
        .dimension("one canonical volume entity")
        .formulation("canonical entity provenance admission before backend consumption")
        .tolerance("three provenance-admission checks")
        .seed("meshlang-provenance-admission-reference:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("MeshLang QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
