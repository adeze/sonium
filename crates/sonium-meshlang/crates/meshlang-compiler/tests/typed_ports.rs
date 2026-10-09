use meshlang_compiler::{compile_to_geo_full, compile_to_python_full, DiagLevel, TypeChecker};

#[test]
fn typed_multi_port_surface_survives_projection_and_csg() {
    let source = r#"
pub geometry Frame() -> (body: Solid, face: Surface) {
    let body = Box(1.0, 1.0, 1.0)
    let face = Rectangle(1.0, 1.0)
}
geometry model {
    let frame = Frame()
 let projected = frame.face
    let combined = projected | Rectangle(1.0, 1.0)
}
"#;

    let ast = meshlang_parser::parse(source, "typed-port-projection.mlg")
        .expect("typed multi-port fixture must parse");
    let diagnostics = TypeChecker::new().check(&ast);
    let errors = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.level == DiagLevel::Error)
        .collect::<Vec<_>>();
    assert!(errors.is_empty(), "surface port type was lost: {errors:?}");

    let geo = compile_to_geo_full(source, "typed-port-projection.mlg")
        .expect("surface port must lower through GEO");
    let python = compile_to_python_full(source, "typed-port-projection.mlg")
        .expect("surface port must lower through Python");
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(geo.canonical_ir_json.contains("model.frame.face"));
    assert!(geo.canonical_ir_json.contains("model.projected"));
    assert!(geo.canonical_ir_json.contains("model.combined"));
}

#[test]
fn typed_multi_port_surface_survives_if_branch_with_backend_parity() {
    let source = r#"
pub geometry Frame() -> (body: Solid, face: Surface) {
    let body = Box(1.0, 1.0, 1.0)
    let face = Rectangle(1.0, 1.0)
}
geometry model {
    if true {
        let frame = Frame()
        let face = frame.face
        let combined = face | Rectangle(1.0, 1.0)
    } else {
        let fallback = Box(0.5, 0.5, 0.5)
    }
}
"#;

    let ast = meshlang_parser::parse(source, "typed-port-if-branch.mlg")
        .expect("typed multi-port control-flow fixture must parse");
    let diagnostics = TypeChecker::new().check(&ast);
    let errors = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.level == DiagLevel::Error)
        .collect::<Vec<_>>();
    assert!(
        errors.is_empty(),
        "surface port type was lost inside if branch: {errors:?}"
    );

    let geo = compile_to_geo_full(source, "typed-port-if-branch.mlg")
        .expect("typed port if branch must lower through GEO");
    let python = compile_to_python_full(source, "typed-port-if-branch.mlg")
        .expect("typed port if branch must lower through Python");
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
    assert!(geo.canonical_ir_json.contains("model.face"));
    assert!(geo.canonical_ir_json.contains("model.combined"));
    assert!(!geo.code.contains("if true"));
    assert!(!python.code.contains("if True:"));
}

#[test]
fn typed_multi_port_surface_mismatch_is_rejected_inside_if_branch() {
    let source = r#"
pub geometry Frame() -> (body: Solid, face: Surface) {
    let body = Box(1.0, 1.0, 1.0)
    let face = Rectangle(1.0, 1.0)
}
geometry model {
    if true {
        let frame = Frame()
        let invalid = frame.face | Box(1.0, 1.0, 1.0)
    }
}
"#;

    let ast = meshlang_parser::parse(source, "typed-port-if-mismatch.mlg")
        .expect("typed multi-port mismatch fixture must parse");
    let diagnostics = TypeChecker::new().check(&ast);
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.level == DiagLevel::Error && diagnostic.message.contains("same dimension")
        }),
        "expected branch-local surface/solid mismatch diagnostic, got {diagnostics:?}"
    );
}
