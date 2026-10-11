use meshlang_ast::VersionedCanonicalIr;
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};

pub fn nested_compile_time_control_flow_has_composite_bindings_and_backend_parity() {
    let source = r#"
        geometry model {
            for i in 0..2 {
                for j in 0..2 {
                    if i == j {
                        let cell = Box(1.0, 1.0, 1.0)
                    } else {
                        let cell = Cylinder(r: 0.2, h: 1.0)
                    }
                }
            }
        }
    "#;

    let geo = compile_to_geo_full(source, "nested-control-flow-packet.mlg")
        .expect("nested control flow must compile to GEO");
    let python = compile_to_python_full(source, "nested-control-flow-packet.mlg")
        .expect("nested control flow must compile to Python");

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
    assert!(!geo.code.contains("For "));
    assert!(!python.code.contains("if i == j"));
    assert_eq!(geo.code.matches("Box(").count(), 2);
    assert_eq!(geo.code.matches("Cylinder(").count(), 2);
    assert_eq!(python.code.matches("occ.addBox(").count(), 2);
    assert_eq!(python.code.matches("occ.addCylinder(").count(), 2);

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical IR JSON")
        .ir;
    let names = ir
        .entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect::<Vec<_>>();
    for name in [
        "model.cell_0_0",
        "model.cell_0_1",
        "model.cell_1_0",
        "model.cell_1_1",
    ] {
        assert!(names.contains(&name), "missing canonical binding {name}");
    }
    assert_eq!(
        names
            .iter()
            .filter(|name| name.starts_with("model.cell_"))
            .count(),
        4
    );
    assert!(ir
        .entities
        .iter()
        .filter(|entity| entity.name.starts_with("model.cell_"))
        .all(|entity| !entity.provenance.is_empty()));
}

#[test]
fn nested_compile_time_control_flow_test() {
    nested_compile_time_control_flow_has_composite_bindings_and_backend_parity();
}

#[test]
fn inline_function_if_branches_lower_through_canonical_ir() {
    let source = r#"
        fn choose(flag: Bool) -> Solid {
            if flag {
                let selected = Box(1.0, 1.0, 1.0)
                selected
            } else {
                let skipped = Cylinder(r: 0.2, h: 1.0)
                skipped
            }
        }
        geometry model {
            let result = choose(true)
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-if.mlg")
        .expect("inline function if must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-if.mlg")
        .expect("inline function if must compile to Python");

    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(geo.code.contains("result"));
    assert!(python.code.contains("result"));
    assert_eq!(geo.code.matches("Box(").count(), 1);
    assert!(!geo.code.contains("Cylinder("));
    assert_eq!(python.code.matches("occ.addBox(").count(), 1);
    assert!(!python.code.contains("occ.addCylinder("));
}

#[test]
fn inline_function_compile_time_for_lowers_an_array_result() {
    let source = r#"
        fn tiles() -> Solid {
            for i in 0..2 {
                let tile = Box(1.0, 1.0, 1.0)
                tile
            }
        }
        geometry model {
            let result = tiles()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-for.mlg")
        .expect("inline function for must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-for.mlg")
        .expect("inline function for must compile to Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("Box(").count(), 2);
    assert_eq!(python.code.matches("occ.addBox(").count(), 2);

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical IR JSON")
        .ir;
    let result = ir
        .entities
        .iter()
        .find(|entity| entity.name == "model.result")
        .expect("array result binding");
    assert_eq!(result.inputs.len(), 2);
}

#[test]
fn inline_function_nonfinal_for_aggregates_entity_bindings() {
    let source = r#"
        fn tiles() -> Solid {
            for i in 0..2 {
                let tile = Box(1.0, 1.0, 1.0)
                let shifted = tile |> translate(i, 0.0, 0.0)
            }
            shifted
        }
        geometry model {
            let result = tiles()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nonfinal-for.mlg")
        .expect("non-final inline for must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nonfinal-for.mlg")
        .expect("non-final inline for must compile to Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("Box(").count(), 2);
    assert_eq!(python.code.matches("occ.addBox(").count(), 2);

    let ir = serde_json::from_str::<VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical IR JSON")
        .ir;
    let result = ir
        .entities
        .iter()
        .find(|entity| entity.name == "model.result")
        .expect("aggregated loop result binding");
    assert_eq!(result.inputs.len(), 2);
    assert!(result.provenance.len() >= 2);
}

#[test]
fn inline_function_nested_nonfinal_for_aggregates_all_products() {
    let source = r#"
        fn cells() -> Solid {
            for i in 0..2 {
                for j in 0..2 {
                    let cell = Box(1.0, 1.0, 1.0)
                    let shifted = cell |> translate(i, j, 0.0)
                }
            }
            shifted
        }
        geometry model {
            let result = cells()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nested-nonfinal-for.mlg")
        .expect("nested non-final inline for must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nested-nonfinal-for.mlg")
        .expect("nested non-final inline for must compile to Python");
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("Box(").count(), 4);
    assert_eq!(python.code.matches("occ.addBox(").count(), 4);
}

#[test]
fn inline_function_nonfinal_for_resolves_each_compile_time_branch() {
    let source = r#"
        fn variants() -> Solid {
            for i in 0..2 {
                if i == 0 {
                    let shape = Box(1.0, 1.0, 1.0)
                } else {
                    let shape = Cylinder(r: 0.2, h: 1.0)
                }
            }
            shape
        }
        geometry model {
            let result = variants()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nonfinal-for-branch.mlg")
        .expect("branching non-final inline for must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nonfinal-for-branch.mlg")
        .expect("branching non-final inline for must compile to Python");
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("Box(").count(), 1);
    assert_eq!(geo.code.matches("Cylinder(").count(), 1);
    assert_eq!(python.code.matches("occ.addBox(").count(), 1);
    assert_eq!(python.code.matches("occ.addCylinder(").count(), 1);
}

#[test]
fn inline_function_nonfinal_for_requires_entity_binding() {
    let source = r#"
        fn invalid() -> Solid {
            for i in 0..2 {
                let scalar = i
            }
            scalar
        }
        geometry model {
            let result = invalid()
        }
    "#;
    let error = compile_to_geo_full(source, "inline-function-nonfinal-for-scalar.mlg")
        .expect_err("scalar-only non-final inline for must fail closed");
    assert!(
        error.to_string().contains("must produce an entity binding"),
        "{error}"
    );
}

#[test]
fn inline_function_dynamic_if_is_rejected_at_canonical_boundary() {
    let source = r#"
        param enabled = true
        fn choose(flag: Bool) -> Solid {
            if flag {
                Box(1.0, 1.0, 1.0)
            } else {
                Box(2.0, 2.0, 2.0)
            }
        }
        geometry model {
            let result = choose(enabled)
        }
    "#;
    let error = compile_to_geo_full(source, "inline-function-dynamic-if.mlg")
        .expect_err("dynamic inline function branch must not bypass canonical IR");
    assert!(
        error.to_string().contains("compile-time boolean"),
        "{error}"
    );
}

#[test]
fn inline_function_nonfinal_if_binds_selected_value() {
    let source = r#"
        fn choose() -> Solid {
            if true {
                let selected = Box(1.0, 1.0, 1.0)
            } else {
                let selected = Cylinder(r: 0.2, h: 1.0)
            }
            selected
        }
        geometry model {
            let result = choose()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nonfinal-if.mlg")
        .expect("non-final inline if must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nonfinal-if.mlg")
        .expect("non-final inline if must compile to Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("Box(").count(), 1);
    assert!(!geo.code.contains("Cylinder("));
    assert_eq!(python.code.matches("occ.addBox(").count(), 1);
    assert!(!python.code.contains("occ.addCylinder("));
}

#[test]
fn inline_function_nonfinal_else_if_selects_middle_binding() {
    let source = r#"
        fn choose() -> Solid {
            if false {
                let selected = Box(1.0, 1.0, 1.0)
            } else if 1 < 2 {
                let selected = Cylinder(r: 0.2, h: 1.0)
            } else {
                let selected = Sphere(r: 0.5)
            }
            selected
        }
        geometry model {
            let result = choose()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nonfinal-elseif.mlg")
        .expect("non-final else-if must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nonfinal-elseif.mlg")
        .expect("non-final else-if must compile to Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(!geo.code.contains("Box("));
    assert_eq!(geo.code.matches("Cylinder(").count(), 1);
    assert!(!geo.code.contains("Sphere("));
    assert!(!python.code.contains("occ.addBox("));
    assert_eq!(python.code.matches("occ.addCylinder(").count(), 1);
    assert!(!python.code.contains("occ.addSphere("));
}

#[test]
fn inline_function_nonfinal_dynamic_if_is_rejected() {
    let source = r#"
        param enabled = true
        fn choose(flag: Bool) -> Solid {
            if flag {
                let selected = Box(1.0, 1.0, 1.0)
            } else {
                let selected = Cylinder(r: 0.2, h: 1.0)
            }
            selected
        }
        geometry model {
            let result = choose(enabled)
        }
    "#;

    let error = compile_to_geo_full(source, "inline-function-nonfinal-dynamic-if.mlg")
        .expect_err("dynamic non-final inline if must be rejected");
    assert!(
        error.to_string().contains("compile-time boolean"),
        "{error}"
    );
}

#[test]
fn inline_function_nonfinal_branch_substitutes_bindings_sequentially() {
    let source = r#"
        fn choose() -> Solid {
            if true {
                let selected = Box(1.0, 1.0, 1.0)
                let shifted = selected |> translate(1.0, 2.0, 3.0)
            } else {
                let shifted = Cylinder(r: 0.2, h: 1.0)
            }
            shifted
        }
        geometry model {
            let result = choose()
        }
    "#;

    let geo = compile_to_geo_full(source, "inline-function-nonfinal-sequential.mlg")
        .expect("sequential branch bindings must compile to GEO");
    let python = compile_to_python_full(source, "inline-function-nonfinal-sequential.mlg")
        .expect("sequential branch bindings must compile to Python");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(geo.code.contains("Translate {1.0, 2.0, 3.0}"));
    assert_eq!(geo.code.matches("Box(").count(), 1);
    assert!(!geo.code.contains("Cylinder("));
}
