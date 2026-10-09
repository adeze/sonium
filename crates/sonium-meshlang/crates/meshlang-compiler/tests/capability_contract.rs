use meshlang_ast::{
    CanonicalGeometryIr, Declaration, IrConditionalControl, IrValidationError, SourceMetadata,
    VersionedAst,
};
use meshlang_compiler::{compile_to_geo_full, compile_to_python_full};
use meshlang_parser::parse;

pub fn native_backends_preserve_projected_geometry_as_metadata_only() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let projected = body |> project_iso_surface(tolerance: 0.01)
        }
    "#;

    let geo = compile_to_geo_full(source, "project-capability-geo.mlg")
        .expect("GEO projection metadata must compile");
    assert!(
        geo.code.contains("MeshLang projection")
            && geo.code.contains("metadata-only project_iso_surface")
    );
    let geo_metadata: serde_json::Value =
        serde_json::from_str(&geo.backend_metadata_json).expect("GEO metadata JSON");
    assert_eq!(geo_metadata["backend"], "geo");
    assert_eq!(geo_metadata["projections"][0]["native_geometry"], false);

    let python = compile_to_python_full(source, "project-capability-python.mlg")
        .expect("Python projection metadata must compile");
    assert!(
        python.code.contains("MeshLang projection")
            && python.code.contains("metadata-only project_iso_surface")
    );
    let python_metadata: serde_json::Value =
        serde_json::from_str(&python.backend_metadata_json).expect("Python metadata JSON");
    assert_eq!(python_metadata["backend"], "python");
    assert_eq!(python_metadata["projections"][0]["native_geometry"], false);
}

#[test]
fn native_backends_preserve_projected_geometry_as_metadata_only_test() {
    native_backends_preserve_projected_geometry_as_metadata_only();
}

pub fn typed_ports_parse_and_legacy_ast_roundtrip() {
    let source = r#"
        geometry frame(width: Float = 1.0) -> (body: Solid, inlet: Surface) {
            let body = Box(width, 1.0, 1.0)
            let inlet = Rectangle(1.0, 1.0)
        }
    "#;
    let file = parse(source, "ports.mlg").expect("typed geometry ports must parse");
    let Declaration::Geometry(geometry) = &file.declarations[0] else {
        panic!("expected geometry declaration");
    };
    assert_eq!(geometry.ports.len(), 2);
    assert_eq!(geometry.ports[0].name.name, "body");
    assert_eq!(geometry.ports[1].name.name, "inlet");

    let encoded = VersionedAst::current(file)
        .to_json_vec()
        .expect("ports AST must serialize");
    let (decoded, migration) =
        VersionedAst::from_json_slice(&encoded).expect("ports AST must deserialize");
    assert!(!migration.was_migrated());
    let Declaration::Geometry(decoded_geometry) = &decoded.source_file.declarations[0] else {
        panic!("expected decoded geometry declaration");
    };
    assert_eq!(decoded_geometry.ports.len(), 2);

    let mut legacy: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    legacy["source_file"]["declarations"][0]["Geometry"]
        .as_object_mut()
        .expect("geometry envelope")
        .remove("ports");
    let (decoded_legacy, migration) =
        VersionedAst::from_json_slice(&serde_json::to_vec(&legacy).expect("legacy AST JSON"))
            .expect("AST without ports must remain compatible");
    assert!(!migration.was_migrated());
    let Declaration::Geometry(legacy_geometry) = &decoded_legacy.source_file.declarations[0] else {
        panic!("expected legacy geometry declaration");
    };
    assert!(legacy_geometry.ports.is_empty());
}

#[test]
fn typed_ports_parse_and_legacy_ast_roundtrip_test() {
    typed_ports_parse_and_legacy_ast_roundtrip();
}

pub fn canonical_control_parent_validation() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(IrConditionalControl {
        parent: Some(0),
        parent_loop: None,
        branches: Vec::new(),
        provenance: Vec::new(),
    });
    let errors = ir
        .validate()
        .expect_err("a self-parented canonical control must be rejected");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalParent {
            control: 0,
            parent: 0,
        }
    )));
}

#[test]
fn canonical_control_parent_validation_test() {
    canonical_control_parent_validation();
}

pub fn canonical_control_branch_validation() {
    let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
    ir.controls.push(IrConditionalControl {
        parent: None,
        parent_loop: None,
        branches: vec![
            meshlang_ast::IrConditionalBranch {
                condition: None,
                entities: Vec::new(),
                provenance: Vec::new(),
            },
            meshlang_ast::IrConditionalBranch {
                condition: Some(meshlang_ast::IrValue::Boolean(true)),
                entities: Vec::new(),
                provenance: Vec::new(),
            },
        ],
        provenance: Vec::new(),
    });
    let errors = ir
        .validate()
        .expect_err("unconditional branch must be final");
    assert!(errors.iter().any(|error| matches!(
        error,
        IrValidationError::InvalidConditionalBranchOrder {
            control: 0,
            branch: 0,
        }
    )));
}

pub fn canonical_mixed_control_parent_links() {
    let source = r#"
        param count = 2
        param enabled = true
        geometry model {
            if enabled {
                for i in 0..count {
                    let inside_loop = Box(1.0, 1.0, 1.0)
                }
            }
            for j in 0..count {
                if enabled {
                    let inside_branch = Sphere(r: 0.5)
                }
            }
        }
    "#;
    let geo = compile_to_geo_full(source, "mixed-control-parents.mlg")
        .expect("GEO mixed-control compilation");
    let python = compile_to_python_full(source, "mixed-control-parents.mlg")
        .expect("Python mixed-control compilation");
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .expect("canonical mixed-control IR JSON")
        .ir;
    assert_eq!(ir.controls.len(), 2);
    assert_eq!(ir.loops.len(), 2);
    assert_eq!(ir.controls[0].parent_loop, None);
    assert_eq!(ir.loops[0].parent_control, Some(0));
    assert_eq!(ir.loops[1].parent_control, None);
    assert_eq!(ir.controls[1].parent_loop, Some(1));
}

#[test]
fn canonical_mixed_control_parent_links_test() {
    canonical_mixed_control_parent_links();
}

#[test]
fn canonical_control_branch_validation_test() {
    canonical_control_branch_validation();
}
