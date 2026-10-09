use meshlang_ast::VersionedCanonicalIr;
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};

pub fn nested_ported_templates_preserve_each_port_through_canonical_ir() {
    let source = r#"
        pub geometry Inner() -> (body: Solid, face: Surface) {
            let body = Box(1.0, 1.0, 1.0)
            let face = Rectangle(1.0, 1.0)
        }

        pub geometry Outer() -> (body: Solid, face: Surface) {
            let inner = Inner()
            let body = inner.body
            let face = inner.face
        }

        geometry model {
            let outer = Outer()
            let body = outer.body
            let face = outer.face
            let combined = face | Rectangle(1.0, 1.0)
        }
    "#;

    let geo = compile_to_geo_full(source, "nested-ported-template.mlg")
        .expect("nested ported template must compile through GEO");
    let python = compile_to_python_full(source, "nested-ported-template.mlg")
        .expect("nested ported template must compile through Python");

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

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("versioned canonical IR")
        .ir;
    for name in [
        "model.outer.body",
        "model.outer.face",
        "model.body",
        "model.face",
        "model.combined",
    ] {
        let entity = ir
            .entities
            .iter()
            .find(|entity| entity.name == name)
            .unwrap_or_else(|| panic!("missing nested port binding {name}"));
        assert!(
            !entity.provenance.is_empty(),
            "missing provenance for {name}"
        );
    }
    assert!(geo.code.contains("Rectangle("));
    assert!(python.code.contains("occ.addRectangle("));
    assert!(!python.code.contains("inner.body"));
    assert!(!python.code.contains("outer.body"));
}

#[test]
fn nested_ported_templates_regression() {
    nested_ported_templates_preserve_each_port_through_canonical_ir();
}

pub fn ported_templates_inside_compile_time_loops_preserve_each_instance_port() {
    let source = r#"
        pub geometry Inner() -> (body: Solid, face: Surface) {
            let body = Box(1.0, 1.0, 1.0)
            let face = Rectangle(1.0, 1.0)
        }
        geometry model {
            for i in 0..3 {
                let inner = Inner()
            }
        }
    "#;
    let geo = compile_to_geo_full(source, "loop-ported-template.mlg")
        .expect("looped ported template must compile through GEO");
    let python = compile_to_python_full(source, "loop-ported-template.mlg")
        .expect("looped ported template must compile through Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("versioned canonical IR")
        .ir;
    for iteration in 0..3 {
        for port in ["body", "face"] {
            let name = format!("model.inner_{iteration}.{port}");
            let entity = ir
                .entities
                .iter()
                .find(|entity| entity.name == name)
                .unwrap_or_else(|| panic!("missing looped port binding {name}"));
            assert!(
                !entity.provenance.is_empty(),
                "missing provenance for {name}"
            );
        }
    }
    assert_eq!(geo.code.matches("Box(").count(), 3);
    assert_eq!(python.code.matches("occ.addBox").count(), 3);
}

pub fn ported_templates_inside_runtime_branches_keep_branch_qualified_ports() {
    let source = r#"
        param enabled = true
        pub geometry Inner() -> (body: Solid, face: Surface) {
            let body = Box(1.0, 1.0, 1.0)
            let face = Rectangle(1.0, 1.0)
        }
        geometry model {
            if enabled {
                let inner = Inner()
            }
        }
    "#;
    let geo = compile_to_geo_full(source, "branch-ported-template.mlg")
        .expect("branched ported template must compile through GEO");
    let python = compile_to_python_full(source, "branch-ported-template.mlg")
        .expect("branched ported template must compile through Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("versioned canonical IR")
        .ir;
    for port in ["body", "face"] {
        let prefix = format!("model.__if_0_0.inner.{port}");
        assert!(
            ir.entities.iter().any(|entity| entity.name == prefix),
            "missing branch-qualified port {prefix}"
        );
    }
    assert_eq!(ir.controls.len(), 1);
}

#[test]
fn ported_templates_inside_compile_time_loops_regression() {
    ported_templates_inside_compile_time_loops_preserve_each_instance_port();
}

#[test]
fn ported_templates_inside_runtime_branches_regression() {
    ported_templates_inside_runtime_branches_keep_branch_qualified_ports();
}
