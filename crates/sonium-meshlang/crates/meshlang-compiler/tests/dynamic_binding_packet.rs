use std::collections::HashMap;

use meshlang_ast::VersionedCanonicalIr;
use meshlang_compiler::{CompileResult, compile_to_geo_full, compile_to_python_full};

fn public_entity_names(result: &CompileResult) -> Vec<String> {
    let ir = serde_json::from_str::<VersionedCanonicalIr>(&result.canonical_ir_json)
        .expect("canonical IR JSON");
    ir.ir
        .entities
        .iter()
        .filter(|entity| !entity.name.contains(".__ir_"))
        .map(|entity| entity.name.clone())
        .collect()
}

pub fn static_control_flow_with_dynamic_multi_binding_is_canonical_and_deterministic() {
    let source = r#"
        param base_width = 1.0

        geometry model {
            if true {
                let a = Box(base_width, 1.0, 1.0)
                let b = Box(base_width, 0.5, 0.5)
                let pair = a | b
                let shifted = pair |> translate(0.0, 0.0, 0.0)
            }
        }
    "#;

    let geo = compile_to_geo_full(source, "dynamic-binding-packet.mlg").expect("GEO compilation");
    let python =
        compile_to_python_full(source, "dynamic-binding-packet.mlg").expect("Python compilation");
    let geo_repeat =
        compile_to_geo_full(source, "dynamic-binding-packet.mlg").expect("repeat GEO compilation");
    let python_repeat = compile_to_python_full(source, "dynamic-binding-packet.mlg")
        .expect("repeat Python compilation");

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(
        geo.provenance_manifest_json,
        python.provenance_manifest_json
    );
    assert_eq!(
        geo.sensitivity_manifest_json,
        python.sensitivity_manifest_json
    );
    assert_eq!(geo.code, geo_repeat.code);
    assert_eq!(geo.backend_metadata_json, geo_repeat.backend_metadata_json);
    assert_eq!(python.code, python_repeat.code);
    assert_eq!(
        python.backend_metadata_json,
        python_repeat.backend_metadata_json
    );

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical IR JSON")
        .ir;
    let names = public_entity_names(&geo);
    for name in ["model.a", "model.b", "model.pair", "model.shifted"] {
        assert!(names.contains(&name.to_string()), "{name}");
    }
    assert_eq!(geo.code.matches("Box(").count(), 2);
    assert_eq!(python.code.matches("occ.addBox").count(), 2);
    assert!(!python.code.contains("if true"));

    let entities = ir
        .entities
        .iter()
        .map(|entity| (entity.name.as_str(), entity))
        .collect::<HashMap<_, _>>();
    let shifted = entities
        .get("model.shifted")
        .expect("shifted canonical entity");
    let input = shifted.inputs.first().expect("shifted input");
    let input_name = ir
        .entities
        .iter()
        .find(|entity| entity.id == *input)
        .expect("shifted input entity")
        .name
        .as_str();
    assert_eq!(input_name, "model.pair");
    assert!(!shifted.provenance.is_empty());
}

pub fn mixed_control_flow_loop_bindings_match_backends() {
    let source = r#"
        param radius = 0.5

        geometry model {
            if false {
                let skipped = Box(radius, 0.1, 0.1)
            } else {
                for i in 0..3 {
                    let segment = Box(radius, 0.1, 0.1)
                        |> translate(i * 0.5, 0.0, 0.0)
                }
            }
        }
    "#;

    let geo =
        compile_to_geo_full(source, "mixed-control-flow-packet.mlg").expect("GEO compilation");
    let python = compile_to_python_full(source, "mixed-control-flow-packet.mlg")
        .expect("Python compilation");

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(
        geo.provenance_manifest_json,
        python.provenance_manifest_json
    );
    assert_eq!(
        geo.sensitivity_manifest_json,
        python.sensitivity_manifest_json
    );
    assert_eq!(geo.code.matches("Box(").count(), 3);
    assert_eq!(python.code.matches("occ.addBox").count(), 3);

    let names = public_entity_names(&geo);
    for suffix in ["segment_0", "segment_1", "segment_2"] {
        assert!(
            names.iter().any(|name| name == &format!("model.{suffix}")),
            "missing loop binding {suffix}: {names:?}"
        );
    }
    assert!(!names.iter().any(|name| name == "model.skipped"));
}

pub fn dynamic_else_if_multi_bindings_have_one_canonical_branch_plan() {
    let source = r#"
        param enabled = true

        geometry model {
            if enabled {
                let primary = Box(1.0, 1.0, 1.0)
                let primary_copy = primary |> translate(1.0, 0.0, 0.0)
            } else if !enabled {
                let fallback = Box(0.5, 0.5, 0.5)
                let fallback_copy = fallback |> translate(-1.0, 0.0, 0.0)
            } else {
                let final_shape = Box(0.25, 0.25, 0.25)
                let final_copy = final_shape |> translate(0.0, 1.0, 0.0)
            }
        }
    "#;
    let geo = compile_to_geo_full(source, "dynamic-else-if-packet.mlg")
        .expect("GEO dynamic else-if compilation");
    let python = compile_to_python_full(source, "dynamic-else-if-packet.mlg")
        .expect("Python dynamic else-if compilation");

    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(
        geo.provenance_manifest_json,
        python.provenance_manifest_json
    );
    assert_eq!(
        geo.sensitivity_manifest_json,
        python.sensitivity_manifest_json
    );
    assert!(geo.code.contains("If (enabled)"), "{}", geo.code);
    assert!(python.code.contains("if enabled:"), "{}", python.code);

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical IR JSON")
        .ir;
    assert_eq!(ir.controls.len(), 1);
    assert_eq!(ir.controls[0].branches.len(), 3);
    assert!(!ir.controls[0].provenance.is_empty());
    assert!(
        ir.controls[0]
            .branches
            .iter()
            .all(|branch| !branch.provenance.is_empty())
    );
    assert!(
        ir.controls[0]
            .branches
            .iter()
            .all(|branch| branch.entities.len() == 2)
    );
    assert!(
        ir.entities
            .iter()
            .all(|entity| !entity.provenance.is_empty())
    );
    assert!(matches!(
        ir.controls[0].branches[0].condition,
        Some(meshlang_ast::IrValue::Parameter(_))
    ));
    assert!(
        ir.entities
            .iter()
            .any(|entity| entity.name == "model.__if_0_0.primary")
    );
    assert!(
        ir.entities
            .iter()
            .any(|entity| entity.name == "model.__if_0_1.fallback")
    );
}

#[test]
fn static_dynamic_binding_test() {
    static_control_flow_with_dynamic_multi_binding_is_canonical_and_deterministic();
}

#[test]
fn mixed_loop_binding_test() {
    mixed_control_flow_loop_bindings_match_backends();
}

#[test]
fn dynamic_else_if_binding_test() {
    dynamic_else_if_multi_bindings_have_one_canonical_branch_plan();
}
