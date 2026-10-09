//! Release-backed MeshLang inline-function control-flow reference.

use meshlang_ast::{BooleanOperation, GeometryOperation, VersionedCanonicalIr};
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use sonium_qa::QaResult;

const CASE_ID: &str = "meshlang.inline-control-flow-reference.v1";
const SOURCE: &str = r#"
fn choose(flag: Bool) -> Solid {
    if flag {
        let selected = Box(1.0, 1.0, 1.0)
        selected
    } else {
        let skipped = Cylinder(r: 0.2, h: 1.0)
        skipped
    }
}

fn tiles() -> Solid {
    for i in 0..2 {
        let tile = Box(1.0, 1.0, 1.0)
        tile
    }
}

geometry test {
    let chosen = choose(true)
    let repeated = tiles()
}
"#;

const NONFINAL_SOURCE: &str = r#"
fn choose() -> Solid {
    if true {
        let selected = Box(1.0, 1.0, 1.0)
    } else {
        let selected = Cylinder(r: 0.2, h: 1.0)
    }
    selected
}

geometry test {
    let chosen = choose()
}
"#;

const DYNAMIC_NONFINAL_SOURCE: &str = r#"
param enabled = true
fn choose(flag: Bool) -> Solid {
    if flag {
        let selected = Box(1.0, 1.0, 1.0)
    } else {
        let selected = Cylinder(r: 0.2, h: 1.0)
    }
    selected
}

geometry test {
    let chosen = choose(enabled)
}
"#;

const NONFINAL_FOR_SOURCE: &str = r#"
fn tiles() -> Solid {
    for i in 0..2 {
        let tile = Box(1.0, 1.0, 1.0)
        let shifted = tile |> translate(i, 0.0, 0.0)
    }
    shifted
}

geometry test {
    let repeated = tiles()
}
"#;

fn run() -> Result<QaResult, String> {
    let geo = compile_to_geo_full(SOURCE, "inline-control-flow-reference.mlg")
        .map_err(|error| error.to_string())?;
    let python = compile_to_python_full(SOURCE, "inline-control-flow-reference.mlg")
        .map_err(|error| error.to_string())?;
    let nonfinal_geo = compile_to_geo_full(
        NONFINAL_SOURCE,
        "inline-control-flow-nonfinal-reference.mlg",
    )
    .map_err(|error| error.to_string())?;
    let nonfinal_python = compile_to_python_full(
        NONFINAL_SOURCE,
        "inline-control-flow-nonfinal-reference.mlg",
    )
    .map_err(|error| error.to_string())?;
    let dynamic_error = compile_to_geo_full(
        DYNAMIC_NONFINAL_SOURCE,
        "inline-control-flow-dynamic-nonfinal-reference.mlg",
    )
    .expect_err("dynamic non-final inline branch must be rejected");
    let nonfinal_for_geo = compile_to_geo_full(
        NONFINAL_FOR_SOURCE,
        "inline-control-flow-nonfinal-for-reference.mlg",
    )
    .map_err(|error| error.to_string())?;
    let nonfinal_for_python = compile_to_python_full(
        NONFINAL_FOR_SOURCE,
        "inline-control-flow-nonfinal-for-reference.mlg",
    )
    .map_err(|error| error.to_string())?;
    let nonfinal_for_ir =
        serde_json::from_str::<VersionedCanonicalIr>(&nonfinal_for_geo.canonical_ir_json)
            .map_err(|error| format!("non-final for canonical IR JSON: {error}"))?
            .ir;
    let nonfinal_for_result = nonfinal_for_ir
        .entities
        .iter()
        .find(|entity| entity.name == "test.repeated")
        .ok_or_else(|| "canonical IR omitted non-final inline loop result".to_string())?;
    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .map_err(|error| format!("canonical IR JSON: {error}"))?
        .ir;
    let repeated = ir
        .entities
        .iter()
        .find(|entity| entity.name == "test.repeated")
        .ok_or_else(|| "canonical IR omitted inline loop result".to_string())?;
    let selected = ir
        .entities
        .iter()
        .find(|entity| entity.name == "test.chosen")
        .ok_or_else(|| "canonical IR omitted inline if result".to_string())?;

    let checks = [
        geo.canonical_ir_hash == python.canonical_ir_hash,
        geo.canonical_ir_json == python.canonical_ir_json,
        geo.provenance_manifest_json == python.provenance_manifest_json,
        geo.code.matches("Box(").count() == 3,
        python.code.matches("occ.addBox(").count() == 3,
        !geo.code.contains("Cylinder("),
        !python.code.contains("occ.addCylinder("),
        matches!(
            &repeated.operation,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Union
            }
        ),
        repeated.inputs.len() == 2,
        !selected.provenance.is_empty(),
        nonfinal_geo.canonical_ir_hash == nonfinal_python.canonical_ir_hash,
        nonfinal_geo.canonical_ir_json == nonfinal_python.canonical_ir_json,
        nonfinal_geo.code.matches("Box(").count() == 1,
        !nonfinal_geo.code.contains("Cylinder("),
        nonfinal_python.code.matches("occ.addBox(").count() == 1,
        !nonfinal_python.code.contains("occ.addCylinder("),
        dynamic_error.to_string().contains("compile-time boolean"),
        nonfinal_for_geo.canonical_ir_hash == nonfinal_for_python.canonical_ir_hash,
        nonfinal_for_geo.canonical_ir_json == nonfinal_for_python.canonical_ir_json,
        nonfinal_for_geo.code.matches("Box(").count() == 2,
        nonfinal_for_python.code.matches("occ.addBox(").count() == 2,
        matches!(
            &nonfinal_for_result.operation,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Union
            }
        ),
        nonfinal_for_result.inputs.len() == 2,
    ];
    let checks_passed = checks.iter().filter(|passed| **passed).count();
    if checks_passed != checks.len() {
        return Err(format!(
            "inline control-flow parity failed: {checks_passed}/{} checks",
            checks.len()
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("checks_passed", checks_passed as f64)
        .metric("geo_box_count", 3.0)
        .metric("python_box_count", 3.0)
        .metric("canonical_union_input_count", repeated.inputs.len() as f64)
        .metric(
            "nonfinal_for_union_input_count",
            nonfinal_for_result.inputs.len() as f64,
        )
        .minimum("checks_passed", checks.len() as f64)
        .minimum("geo_box_count", 3.0)
        .minimum("python_box_count", 3.0)
        .minimum("canonical_union_input_count", 2.0)
        .minimum("nonfinal_for_union_input_count", 2.0)
        .input_checksum("source:meshlang-inline-control-flow-reference.mlg:v1")
        .reference("value-producing if/else and compile-time for canonical IR parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-inline-control-flow-v1")
        .dimension("Solid branch selection and two-input CSG union")
        .formulation("MeshLang canonical inline-function control-flow lowering")
        .tolerance("exact parity, pruning, canonical-shape, and rejection checks")
        .seed("meshlang-inline-control-flow-reference:cpu:v1"))
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
