//! Release-facing parity reference for typed ported MeshLang templates.

use std::time::Instant;

use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use serde_json::Value;
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.ported-template-reference.v1";
const FIXTURE_NAME: &str = "ported-template-reference.mlg";

const SOURCE: &str = r#"
    pub geometry Frame(width: Float) -> (body: Solid, inlet: Solid) {
        let body = Box(width, 1.0, 1.0)
        let inlet = Cylinder(r: 0.1, h: 0.5)
    }
    geometry model {
        let frame = Frame(2.0)
        let shifted_inlet = frame.inlet |> translate(1.0, 0.0, 0.0)
    }
    physics {
        group inlet_air: volume = model.frame.inlet
    }
"#;

fn run() -> Result<(usize, usize, usize, usize, u64), String> {
    let started = Instant::now();
    let geo = compile_to_geo_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;

    if geo.canonical_ir_json != python.canonical_ir_json {
        return Err("GEO/Python canonical IR differs for ported template".into());
    }
    let repeated = compile_to_geo_full(SOURCE, FIXTURE_NAME).map_err(|error| error.to_string())?;
    if geo.code != repeated.code || geo.canonical_ir_json != repeated.canonical_ir_json {
        return Err("ported template compilation is not deterministic".into());
    }

    let versioned: Value =
        serde_json::from_str(&geo.canonical_ir_json).map_err(|error| error.to_string())?;
    if !versioned.is_object() {
        return Err("canonical IR versioned envelope is not an object".into());
    }

    let required_ir_names = [
        "model.frame.body",
        "model.frame.inlet",
        "model.shifted_inlet",
    ];
    let missing_ir_names = required_ir_names
        .iter()
        .filter(|name| !geo.canonical_ir_json.contains(**name))
        .count();
    if missing_ir_names != 0 {
        return Err(format!(
            "canonical IR is missing {missing_ir_names} ported names"
        ));
    }

    let geo_entities = geo.code.matches("Box(model__frame__body)").count()
        + geo.code.matches("Cylinder(model__frame__inlet)").count();
    let python_entities = python.code.matches("frame__body = occ.addBox(").count()
        + python
            .code
            .matches("frame__inlet = occ.addCylinder(")
            .count();
    if geo_entities != 2 || python_entities != 2 {
        return Err(format!(
            "ported entity emission counts were GEO={geo_entities}, Python={python_entities}"
        ));
    }
    if !geo.code.contains("inlet_air") || !python.code.contains("inlet_air") {
        return Err("typed physics group was not preserved by both backends".into());
    }

    Ok((
        geo_entities,
        python_entities,
        geo.canonical_ir_json.len(),
        geo.backend_metadata_json.len() + python.backend_metadata_json.len(),
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match run() {
        Ok((geo_entities, python_entities, ir_bytes, metadata_bytes, elapsed_ms)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("geo_entity_emissions", geo_entities as f64)
                .metric("python_entity_emissions", python_entities as f64)
                .metric("canonical_ir_bytes", ir_bytes as f64)
                .metric("backend_metadata_bytes", metadata_bytes as f64)
                .metric("canonical_ir_backend_parity", 1.0)
                .metric("deterministic_repeat_compilation", 1.0)
                .minimum("geo_entity_emissions", 2.0)
                .minimum("python_entity_emissions", 2.0)
                .minimum("canonical_ir_backend_parity", 1.0)
                .minimum("deterministic_repeat_compilation", 1.0)
                .reference(
                    "typed ported-template canonical IR and deterministic GEO/Python emission",
                )
                .reference_checksum("analytic/deterministic-fixture:meshlang-ported-template-v1")
                .input_checksum("source:ported-template-reference.mlg:v1")
                .dimension("typed two-port Solid geometry template with physics-group consumer")
                .formulation(
                    "canonical IR parity, versioned envelope stability, and backend entity naming",
                )
                .tolerance(
                    "exact GEO/Python canonical IR parity and deterministic typed-port emission",
                )
                .seed("meshlang-ported-template-reference:v1")
                .timing_ms("ported_template_reference", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MESHLANG PORTED TEMPLATE REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
