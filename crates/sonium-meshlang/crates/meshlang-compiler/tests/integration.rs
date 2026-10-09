//! Integration tests: parse .mlg source → verify AST structure.

use meshlang_ast::*;
use meshlang_parser::parse;

#[test]
fn ported_template_python_uses_flattened_instance_ports() {
    let example = "ported_acoustic_frame.mlg";
    let source = load_example(example);
    let compiled = meshlang_compiler::compile_to_python_full(&source, example)
        .unwrap_or_else(|error| panic!("{example} should compile: {error}"));

    assert!(
        compiled.code.contains("frame__inlet"),
        "Python output did not contain the flattened instance port:\n{}",
        compiled.code
    );
    assert!(
        !compiled.code.contains("frame['inlet']"),
        "Python output retained an unbound source-level instance access:\n{}",
        compiled.code
    );
    assert!(
        !compiled.code.contains("frame.inlet"),
        "Python output retained a dotted source-level instance access:\n{}",
        compiled.code
    );
}

#[test]
fn duplicate_top_level_definition_is_rejected_with_span() {
    let source = "param frequency = 1000.0\nparam frequency = 2000.0\n";
    let error = meshlang_compiler::compile_to_geo(source, "duplicate.mlg").unwrap_err();
    // User-facing text carries file/byte offsets, not the Debug struct.
    let text = error.to_string();
    assert!(text.contains("frequency"), "{text}");
    assert!(!text.contains("Span {"), "{text}");
    match error {
        meshlang_compiler::CompileError::DuplicateDefinition { name, span } => {
            assert_eq!(name, "frequency");
            assert_ne!(span, meshlang_ast::span::Span::DUMMY);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn duplicate_local_definitions_are_rejected_with_spans() {
    let cases = [
        (
            "geometry room {\n let wall = Box(1.0, 1.0, 1.0)\n let wall = Box(2.0, 2.0, 2.0)\n}\n",
            "wall",
        ),
        (
            "fn profile(radius: Float, radius: Float) -> Float { radius }\n",
            "radius",
        ),
        (
            "struct Material { density: Float, density: Float }\n",
            "density",
        ),
        (
            "physics {\n group wall: neumann = boundary\n group wall: impedance = boundary\n}\n",
            "wall",
        ),
    ];

    for (source, expected_name) in cases {
        let error = meshlang_compiler::compile_to_geo(source, "duplicate.mlg").unwrap_err();
        match error {
            meshlang_compiler::CompileError::DuplicateDefinition { name, span } => {
                assert_eq!(name, expected_name);
                assert_ne!(span, meshlang_ast::span::Span::DUMMY);
            }
            other => panic!("unexpected error for {expected_name}: {other}"),
        }
    }
}

#[test]
fn dynamic_geometry_else_if_has_backend_parity_before_emission() {
    let source = r#"
        param enabled = true

        geometry model {
            if false {
                let discarded = Box(1.0, 1.0, 1.0)
            } else if enabled {
                let body = Box(1.0, 1.0, 1.0)
            }
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo_full(source, "dynamic-else-if.mlg")
        .expect("GEO dynamic else-if compilation");
    let python = meshlang_compiler::compile_to_python_full(source, "dynamic-else-if.mlg")
        .expect("Python dynamic else-if compilation");
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
    assert!(geo.code.contains("Box("), "{}", geo.code);
    assert!(python.code.contains("occ.addBox"), "{}", python.code);
}

#[test]
fn test_parse_params() {
    let src = r#"
        const C_AIR = 343.0
        param freq = 1000.0
        param elem_size = C_AIR / freq / 6.0
    "#;
    let ast = parse(src, "test.mlg").unwrap();
    assert_eq!(ast.declarations.len(), 3);

    match &ast.declarations[0] {
        Declaration::Const(c) => assert_eq!(c.name.name, "C_AIR"),
        _ => panic!("expected const"),
    }
    match &ast.declarations[1] {
        Declaration::Param(p) => assert_eq!(p.name.name, "freq"),
        _ => panic!("expected param"),
    }
}

#[test]
fn test_parse_unterminated_string_rejected() {
    let src = r#"const name = "speaker"#;
    let err = parse(src, "test.mlg").unwrap_err();
    assert!(
        err.message.contains("unterminated string"),
        "unterminated string should be rejected, got: {}",
        err.message
    );
}

#[test]
fn test_parse_invalid_compound_unit_rejected() {
    let src = "param speed = 3 m/foo";
    let err = parse(src, "test.mlg").unwrap_err();
    assert!(
        err.message.contains("expected declaration") || err.message.contains("invalid unit"),
        "invalid compound unit should not silently fall back to m, got: {}",
        err.message
    );
}

#[test]
fn test_parse_leading_dot_float() {
    let src = "const gain = .5";
    let ast = parse(src, "test.mlg").unwrap();
    match &ast.declarations[0] {
        Declaration::Const(c) => match &c.value {
            Expr::Literal(LiteralExpr {
                value: Literal::Float(value),
                ..
            }) => assert_eq!(*value, 0.5),
            other => panic!("expected float literal, got {other:?}"),
        },
        other => panic!("expected const declaration, got {other:?}"),
    }
}

#[test]
fn test_parse_geometry_with_csg() {
    let src = r#"
        geometry cabinet {
            let shell = Box(0.3, 0.4, 0.25)
            let port = Cylinder(r: 0.025, h: 0.15)
            let solid = shell - port
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();
    assert_eq!(ast.declarations.len(), 1);

    match &ast.declarations[0] {
        Declaration::Geometry(g) => {
            assert_eq!(g.name.name, "cabinet");
            assert_eq!(g.body.statements.len(), 3);

            // Third statement is CSG difference
            match &g.body.statements[2] {
                Statement::Let(l) => {
                    assert_eq!(l.name.name, "solid");
                    match &l.value {
                        Expr::Binary(b) => {
                            assert_eq!(b.op, BinaryOp::Sub);
                        }
                        _ => panic!("expected binary expr"),
                    }
                }
                _ => panic!("expected let"),
            }
        }
        _ => panic!("expected geometry"),
    }
}

#[test]
fn test_parse_pipe_chain() {
    let src = r#"
        geometry test {
            let cyl = Cylinder(r: 0.1, h: 0.5)
                |> translate(1.0, 0.0, 0.0)
                |> rotate(axis: Z, angle: 90)
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Geometry(g) => {
            match &g.body.statements[0] {
                Statement::Let(l) => {
                    // Should be a nested pipe expression
                    match &l.value {
                        Expr::Pipe(p) => {
                            assert_eq!(p.function.name, "rotate");
                            match p.value.as_ref() {
                                Expr::Pipe(inner) => {
                                    assert_eq!(inner.function.name, "translate");
                                }
                                _ => panic!("expected inner pipe"),
                            }
                        }
                        _ => panic!("expected pipe"),
                    }
                }
                _ => panic!("expected let"),
            }
        }
        _ => panic!("expected geometry"),
    }
}

#[test]
fn test_parse_mesh_block() {
    let src = r#"
        mesh for cabinet {
            default_size: 0.01
            order: 2

            refine near(cabinet.port, distance: 0.03) {
                size: 0.005
                growth_rate: 1.2
            }
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Mesh(m) => {
            assert_eq!(m.target.as_ref().unwrap().name, "cabinet");
            assert_eq!(m.statements.len(), 3); // default_size, order, refine
        }
        _ => panic!("expected mesh"),
    }
}

#[test]
fn test_parse_physics_block() {
    let src = r#"
        physics {
            group wall: neumann = cab.outer.faces
            group inlet: dirichlet = cab.port.faces.top
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Physics(p) => {
            assert_eq!(p.statements.len(), 2);
            match &p.statements[0] {
                PhysicsStatement::Group(g) => {
                    assert_eq!(g.name.name, "wall");
                    assert_eq!(g.bc_type.name, "neumann");
                }
                _ => panic!("expected group"),
            }
        }
        _ => panic!("expected physics"),
    }
}

#[test]
fn test_parse_export_block() {
    let src = r#"
        export {
            format: msh4
            order: 2
            filename: "output.msh"
            binary: true
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Export(e) => {
            assert_eq!(e.settings.len(), 4);
        }
        _ => panic!("expected export"),
    }
}

#[test]
fn test_parse_function() {
    let src = r#"
        pub fn port(radius: Float, depth: Float) -> Solid {
            Cylinder(r: radius, h: depth)
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Function(f) => {
            assert!(f.is_pub);
            assert_eq!(f.name.name, "port");
            assert_eq!(f.params.len(), 2);
        }
        _ => panic!("expected function"),
    }
}

#[test]
fn test_parse_import() {
    let src = r#"
        import "acoustics/port.mlg"
        import "lib.mlg" as mylib
    "#;
    let ast = parse(src, "test.mlg").unwrap();
    assert_eq!(ast.declarations.len(), 2);

    match &ast.declarations[1] {
        Declaration::Import(i) => {
            assert_eq!(i.path, "lib.mlg");
            match &i.kind {
                ImportKind::Aliased(alias) => assert_eq!(alias.name, "mylib"),
                _ => panic!("expected aliased import"),
            }
        }
        _ => panic!("expected import"),
    }
}

#[test]
fn test_parse_nested_csg() {
    let src = r#"
        geometry test {
            let a = Box(1, 1, 1)
            let b = Sphere(r: 0.5)
            let c = Cylinder(r: 0.3, h: 2)
            let result = a - b - c
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    match &ast.declarations[0] {
        Declaration::Geometry(g) => {
            match &g.body.statements[3] {
                Statement::Let(l) => {
                    assert_eq!(l.name.name, "result");
                    // a - b - c  parses as (a - b) - c
                    match &l.value {
                        Expr::Binary(outer) => {
                            assert_eq!(outer.op, BinaryOp::Sub);
                            match outer.left.as_ref() {
                                Expr::Binary(inner) => {
                                    assert_eq!(inner.op, BinaryOp::Sub);
                                }
                                _ => panic!("expected inner binary"),
                            }
                        }
                        _ => panic!("expected binary"),
                    }
                }
                _ => panic!("expected let"),
            }
        }
        _ => panic!("expected geometry"),
    }
}

#[test]
fn test_compile_roundtrip_geo() {
    let src = r#"
        param size = 0.01

        geometry box_test {
            let outer = Box(1.0, 2.0, 3.0)
        }

        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(
        geo.contains("= newv;"),
        "should allocate volume tag at runtime"
    );
    assert!(geo.contains("Box("), "should have Box constructor");
    assert!(geo.contains("Output: test.msh"));
}

#[test]
fn test_compile_roundtrip_python() {
    let src = r#"
        geometry sphere_test {
            let ball = Sphere(r: 0.5)
        }

        export {
            format: msh4
            filename: "sphere.msh"
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(py.contains("import gmsh"));
    assert!(py.contains("occ.addSphere(0, 0, 0, 0.5)"));
    assert!(py.contains("gmsh.write('sphere.msh')"));
}

#[test]
fn test_nested_csg_geo_output() {
    let src = r#"
        geometry test {
            let a = Box(1.0, 1.0, 1.0)
            let b = Sphere(r: 0.5)
            let c = Cylinder(r: 0.3, h: 2.0)
            let result = a - b - c
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    let bool_count = geo.matches("BooleanDifference").count();
    assert_eq!(
        bool_count, 1,
        "chained a - b - c should flatten to 1 BooleanDifference, got {}",
        bool_count
    );
    // The single BoolDiff should list both tools
    let diff_line = geo
        .lines()
        .find(|l| l.contains("BooleanDifference"))
        .unwrap();
    let tool_block = diff_line.split("Delete; }{").nth(1).unwrap();
    assert!(
        tool_block.matches("Volume{").count() == 2 && tool_block.contains("}; Volume{"),
        "each CSG tool must be a semicolon-separated Volume block: {}",
        tool_block
    );
}

#[test]
fn test_surface_csg_uses_surface_dimension_in_both_backends() {
    let src = r#"
        geometry panel {
            let outer = Rectangle(2.0, 2.0)
            let hole = Disk(r: 0.25)
            let result = outer - hole
        }
        export { format: msh4  filename: "panel.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "surface_csg.mlg").unwrap();
    let difference = geo
        .lines()
        .find(|line| line.contains("BooleanDifference"))
        .unwrap();
    assert!(
        difference.contains("Surface{") && !difference.contains("Volume{"),
        "2D CSG must use Surface operands:\n{difference}"
    );

    let python = meshlang_compiler::compile_to_python(src, "surface_csg.mlg").unwrap();
    let cut = python
        .lines()
        .find(|line| line.contains("occ.cut("))
        .unwrap();
    assert!(
        cut.contains("_ml_dimtags(2, outer)") && cut.contains("_ml_dimtags(2, hole)"),
        "Python 2D CSG must use dimension 2:\n{cut}"
    );
}

#[test]
fn test_transfinite_is_emitted_once_from_canonical_ir_with_expression_settings() {
    let src = r#"
param nodes = 3
geometry model {
    let face = Rectangle(0.5, 0.5)
}
mesh model {
    transfinite surface(model.face, arrangement: "Left", corners: [1, 2, 3, nodes + 1])
}
export { format: msh4 filename: "transfinite.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "transfinite.mlg").unwrap();
    assert_eq!(
        geo.matches("Transfinite Surface").count(),
        1,
        "canonical GEO transfinite control must be emitted exactly once:\n{geo}"
    );
    assert!(
        geo.contains("Transfinite Surface {model__face} = {1, 2, 3, nodes + 1} Left;")
            || geo.contains("Transfinite Surface {model__face} = {1, 2, 3, (nodes + 1)} Left;"),
        "canonical GEO transfinite settings must retain arrangement/corners:\n{geo}"
    );

    let python = meshlang_compiler::compile_to_python(src, "transfinite.mlg").unwrap();
    assert_eq!(
        python
            .matches("gmsh.model.mesh.setTransfiniteSurface")
            .count(),
        1,
        "canonical Python transfinite control must be emitted exactly once:\n{python}"
    );
    assert!(
        python.contains(
            "gmsh.model.mesh.setTransfiniteSurface(face, 'Left', [1, 2, 3, (nodes + 1)])"
        ) || python
            .contains("gmsh.model.mesh.setTransfiniteSurface(face, 'Left', [1, 2, 3, nodes + 1])"),
        "canonical Python transfinite settings must retain arrangement/corners:\n{python}"
    );
}

#[test]
fn test_transfinite_curve_and_volume_controls_use_canonical_entities() {
    let src = r#"
geometry model {
    let p0 = Point(0.0, 0.0, 0.0)
    let p1 = Point(1.0, 0.0, 0.0)
    let edge = Line(from: p0, to: p1)
    let body = Box(1.0, 1.0, 1.0)
}
mesh model {
    transfinite curve(model.edge, nodes: 5, type: "Progression", coef: 1.1)
    transfinite volume(model.body, corners: [1, 2, 3, 4, 5, 6, 7, 8])
}
export { format: msh4 filename: "transfinite-curve-volume.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "transfinite-curve-volume.mlg").unwrap();
    assert_eq!(geo.matches("Transfinite Curve").count(), 1);
    assert_eq!(geo.matches("Transfinite Volume").count(), 1);
    assert!(geo.contains("Transfinite Curve {model__edge} = 5 Using Progression 1.1;"));
    assert!(geo.contains("Transfinite Volume {model__body} = {1, 2, 3, 4, 5, 6, 7, 8};"));

    let python = meshlang_compiler::compile_to_python(src, "transfinite-curve-volume.mlg").unwrap();
    assert_eq!(
        python
            .matches("gmsh.model.mesh.setTransfiniteCurve")
            .count(),
        1
    );
    assert_eq!(
        python
            .matches("gmsh.model.mesh.setTransfiniteVolume")
            .count(),
        1
    );
    assert!(python.contains("gmsh.model.mesh.setTransfiniteCurve(edge, 5, 'Progression', 1.1)"));
    assert!(python.contains("gmsh.model.mesh.setTransfiniteVolume(body, [1, 2, 3, 4, 5, 6, 7, 8])"));
}

#[test]
fn test_periodic_is_emitted_once_from_canonical_ir_in_both_backends() {
    let src = r#"
geometry model {
    let source = Rectangle(0.5, 0.5)
    let target = Rectangle(0.5, 0.5) |> translate(1.0, 0.0, 0.0)
}
mesh model {
    periodic surface(model.source, model.target, transform: [1.0, 0.0, 0.0])
}
export { format: msh4 filename: "periodic.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "periodic.mlg").unwrap();
    assert_eq!(
        geo.matches("Periodic Surface").count(),
        1,
        "canonical GEO periodic control must be emitted exactly once:\n{geo}"
    );
    assert!(
        geo.contains("Periodic Surface {model__source} = {model__target} Translate {"),
        "canonical GEO periodic control must use resolved entity names:\n{geo}"
    );

    let python = meshlang_compiler::compile_to_python(src, "periodic.mlg").unwrap();
    assert_eq!(
        python.matches("gmsh.model.mesh.setPeriodic(").count(),
        1,
        "canonical Python periodic control must be emitted exactly once:\n{python}"
    );
    assert!(
        python.contains("gmsh.model.mesh.setPeriodic(2, [source], [target], ["),
        "canonical Python periodic control must use inferred surface dimension:\n{python}"
    );
}

#[test]
fn test_embed_is_emitted_once_from_canonical_ir_in_both_backends() {
    let src = r#"
geometry model {
    let body = Box(1.0, 1.0, 1.0)
    let face = Rectangle(0.5, 0.5)
}
mesh model {
    embed surface(model.face) in volume(model.body)
}
export { format: msh4 filename: "embed.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "embed.mlg").unwrap();
    assert_eq!(
        geo.matches("Surface{").count(),
        1,
        "canonical GEO embed must be emitted exactly once:\n{geo}"
    );
    assert!(
        geo.contains("Surface{model__face} In Volume{model__body};"),
        "canonical GEO embed must use inferred entity dimensions:\n{geo}"
    );

    let python = meshlang_compiler::compile_to_python(src, "embed.mlg").unwrap();
    assert_eq!(
        python.matches("gmsh.model.mesh.embed(").count(),
        1,
        "canonical Python embed must be emitted exactly once:\n{python}"
    );
    assert!(
        python.contains("gmsh.model.mesh.embed(2, [face], 3, body)"),
        "canonical Python embed must use inferred entity dimensions:\n{python}"
    );
}

#[test]
fn test_compound_infers_volume_dimension_in_both_backends() {
    let src = r#"
        geometry pieces {
            let left = Box(1.0, 1.0, 1.0)
            let right = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
        }
        mesh for pieces {
            compound volume(pieces.left, pieces.right)
        }
        export { format: msh4  filename: "compound.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "compound.mlg").unwrap();
    assert!(
        geo.contains("Compound Volume {pieces__left, pieces__right};"),
        "GEO compound must use the inferred volume dimension:\n{geo}"
    );

    assert_eq!(
        geo.matches("Compound Volume").count(),
        1,
        "canonical GEO controls must emit each compound exactly once:\\n{geo}"
    );

    let python = meshlang_compiler::compile_to_python(src, "compound.mlg").unwrap();
    assert!(
        python.contains("gmsh.model.mesh.setCompound(3, [left, right])"),
        "Python compound must use dimension 3 and resolved entity vars:\n{python}"
    );
    assert_eq!(
        python.matches("gmsh.model.mesh.setCompound(3").count(),
        1,
        "canonical Python controls must emit each compound exactly once:\\n{python}"
    );
}

#[test]
fn test_boundary_layer_is_emitted_once_from_canonical_ir_in_both_backends() {
    let src = r#"
geometry model {
    let left = Box(1.0, 1.0, 1.0)
    let right = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
}
mesh model {
    refine near(model.left, distance: 0.1) {
        size: 0.01
    }
    boundary_layer on([model.left, model.right]) {
        first_height: 0.001
        thickness: 0.004
        layers: 4
        growth_rate: 1.25
    }
}
export { format: msh4 filename: "boundary-layer.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "boundary-layer.mlg").unwrap();
    assert_eq!(
        geo.matches("= BoundaryLayer;").count(),
        1,
        "canonical GEO boundary layer must be emitted exactly once:\n{geo}"
    );
    assert!(
        geo.contains("Volume{model__left, model__right}"),
        "canonical GEO boundary layer must use the planned volume names:\n{geo}"
    );
    assert!(geo.contains(".Size = 0.001;"));
    assert!(geo.contains(".Thickness = 0.004;"));
    assert!(geo.contains(".NbLayers = 4;"));
    assert!(geo.contains(".Ratio = 1.25;"));
    let geo_background = geo
        .lines()
        .filter(|line| line.contains("FieldsList"))
        .next_back()
        .unwrap();
    assert!(
        geo_background.contains(','),
        "canonical background field must retain source refine field: {geo_background}"
    );

    let python = meshlang_compiler::compile_to_python(src, "boundary-layer.mlg").unwrap();
    assert_eq!(
        python.matches("field.add('BoundaryLayer')").count(),
        1,
        "canonical Python boundary layer must be emitted exactly once:\n{python}"
    );
    assert!(python.contains("_ml_dimtags(3, [left, right])"));
    assert!(python.contains("'Size', 0.001"));
    assert!(python.contains("'Thickness', 0.004"));
    assert!(python.contains("'NbLayers', 4"));
    assert!(python.contains("'Ratio', 1.25"));
    let python_background = python
        .lines()
        .filter(|line| line.contains("'FieldsList'"))
        .next_back()
        .unwrap();
    assert!(
        python_background.contains(','),
        "canonical background field must retain source refine field: {python_background}"
    );
}

#[test]
fn test_arithmetic_subtraction_let_stays_scalar_in_geo() {
    let src = r#"
        param a = 0.10
        param b = 0.05
        geometry test {
            let x = a - b
            let box1 = Box(x, 0.01, 0.01)
            let box2 = Box(0.05, 0.01, 0.01)
            let model = box1 | box2
        }
    "#;

    let result = meshlang_compiler::compile_to_geo_full(src, "test.mlg").unwrap();
    let errors: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.level == meshlang_compiler::DiagLevel::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected type errors: {:?}", errors);
    assert!(
        result.code.contains("x = (a - b);"),
        "subtraction let should emit as scalar arithmetic:\n{}",
        result.code
    );
    assert!(
        !result.code.contains("BooleanDifference"),
        "scalar subtraction must not emit CSG difference:\n{}",
        result.code
    );
    assert!(
        result.code.contains("BooleanUnion"),
        "box1 | box2 should still emit CSG union"
    );
}

#[test]
fn test_arithmetic_subtraction_let_stays_scalar_in_python() {
    let src = r#"
        param a = 0.10
        param b = 0.05
        geometry test {
            let x = a - b
            let box1 = Box(x, 0.01, 0.01)
            let box2 = Box(0.05, 0.01, 0.01)
            let model = box1 | box2
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("x = (a - b)"),
        "subtraction let should emit as scalar arithmetic:\n{}",
        py
    );
    assert!(
        !py.contains("occ.cut("),
        "scalar subtraction must not emit CSG cut:\n{}",
        py
    );
    assert!(
        py.contains("occ.fuse("),
        "box1 | box2 should still emit fuse"
    );
}

#[test]
fn test_nested_csg_python_no_arithmetic() {
    let src = r#"
        geometry test {
            let a = Box(1.0, 1.0, 1.0)
            let b = Sphere(r: 0.5)
            let result = a - b
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("occ.cut("),
        "Python output should use occ.cut()"
    );
    // Should NOT contain arithmetic subtraction as a Python expression
    assert!(
        !py.contains("(a - b)"),
        "Python output should NOT contain arithmetic (a - b)"
    );
}

#[test]
fn test_triple_csg_python() {
    let src = r#"
        geometry test {
            let a = Box(1.0, 1.0, 1.0)
            let b = Sphere(r: 0.5)
            let c = Cylinder(r: 0.3, h: 2.0)
            let result = a - b - c
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    let cut_count = py.matches("occ.cut(").count();
    assert_eq!(
        cut_count, 1,
        "expected one flattened occ.cut() call, got {}",
        cut_count
    );
}

#[test]
fn test_pipe_transform_geo() {
    let src = r#"
        geometry test {
            let cyl = Cylinder(r: 0.1, h: 0.5)
                |> translate(1.0, 2.0, 3.0)
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Cylinder("));
    assert!(geo.contains("Translate {1.0, 2.0, 3.0}"));
}

#[test]
fn test_python_transforms_use_surface_dimension_for_surface_inputs() {
    let src = r#"
        geometry test {
            let panel = Rectangle(1.0, 1.0)
                |> translate(1.0, 0.0, 0.0)
                |> rotate(axis: Z, angle: 90)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("occ.translate(_ml_dimtags(2, panel), 1.0, 0.0, 0.0)"),
        "translate should use the surface dimension for Rectangle:\n{}",
        py
    );
    assert!(
        py.contains("occ.rotate(_ml_dimtags(2, panel)"),
        "rotate should preserve the surface dimension before extrusion:\n{}",
        py
    );
}

#[test]
fn test_affine_transform_is_emitted_by_native_backends() {
    let src = r#"
        geometry test {
            let panel = Rectangle(1.0, 1.0)
                |> affine(matrix: [1.0, 0.0, 0.0, 0.25,
                                   0.0, 1.0, 0.0, 0.5,
                                   0.0, 0.0, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 1.0])
        }
        export { format: msh4 filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "affine.mlg").unwrap();
    assert!(geo.contains("Affine {1.0, 0.0, 0.0, 0.25"), "{geo}");

    let python = meshlang_compiler::compile_to_python(src, "affine.mlg").unwrap();
    assert!(
        python.contains("occ.affineTransform(_ml_dimtags(2, panel), [1.0, 0.0, 0.0, 0.25"),
        "{python}"
    );
}

#[test]
fn test_affine_transform_rejects_wrong_matrix_shape() {
    let src = r#"
        geometry test {
            let panel = Rectangle(1.0, 1.0) |> affine(matrix: [1.0, 0.0, 0.0])
        }
        export { format: msh4 filename: "test.msh" }
    "#;
    let error = meshlang_compiler::compile_to_geo(src, "bad-affine.mlg").unwrap_err();
    assert!(error.to_string().contains("16 scalars"), "{error}");
}

#[test]
fn test_cyclic_symmetry_annotation_is_preserved_in_backend_artifacts() {
    let src = r#"
        geometry test { let rotor = Box(1.0, 1.0, 0.2) }
        design_region rotor_design {
            kind: design
            region: test.rotor
            annotations: [symmetry(kind: cyclic(axis: [0.0, 0.0, 1.0], sectors: 12))]
        }
        export { format: msh4 filename: "test.msh" }
    "#;
    let geo = meshlang_compiler::compile_to_geo(src, "cyclic-backend.mlg").unwrap();
    assert!(
        geo.contains("MeshLang cyclic symmetry region=rotor_design sectors=12"),
        "{geo}"
    );
    assert!(geo.contains("For __ml_cyclic_sector_0 In {1:11}"), "{geo}");
    assert!(geo.contains("Duplicata"), "{geo}");
    assert!(geo.contains("Rotate"), "{geo}");
    let python = meshlang_compiler::compile_to_python(src, "cyclic-backend.mlg").unwrap();
    assert!(
        python.contains("MeshLang cyclic symmetry region=rotor_design sectors=12"),
        "{python}"
    );
    assert!(
        python.contains("occ.copy(_ml_cyclic_entities_0)"),
        "{python}"
    );
    assert!(
        python.contains("occ.rotate(_ml_cyclic_copies_0"),
        "{python}"
    );
    assert!(
        python.find("occ.copy(_ml_cyclic_entities_0)").unwrap()
            < python.find("gmsh.finalize()").unwrap(),
        "cyclic controls must run before finalization:\n{python}"
    );
}

#[test]
fn test_geo_surface_extrude_reassigns_volume_for_chained_shell() {
    let src = r#"
        geometry test {
            let panel = Rectangle(1.0, 1.0)
                |> extrude(0.0, 0.0, 0.25)
                |> shell(thickness: 0.02)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("[] = Extrude {0.0, 0.0, 0.25} { Surface{"),
        "extrude should capture the returned entities:\n{}",
        geo
    );
    assert!(
        geo.contains("panel = ") && geo.contains("[1];"),
        "extrude should reassign the geometry var to the returned volume:\n{}",
        geo
    );
    assert!(
        geo.contains("ThickSolid("),
        "shell should run after extrusion on the reassigned volume:\n{}",
        geo
    );
}

#[test]
fn test_transform_dimension_mismatches_are_type_errors() {
    let invalid_extrude = r#"
        geometry test {
            let bad = Box(1.0, 1.0, 1.0) |> extrude(0.0, 0.0, 1.0)
        }
        export { format: msh4  filename: "test.msh" }
    "#;
    let err = meshlang_compiler::compile_to_geo(invalid_extrude, "test.mlg").unwrap_err();
    assert!(
        err.to_string()
            .contains("transform 'extrude' expects surface"),
        "solid extrude should fail in type checking, got: {err}"
    );

    let invalid_shell = r#"
        geometry test {
            let bad = Rectangle(1.0, 1.0) |> shell(thickness: 0.02)
        }
        export { format: msh4  filename: "test.msh" }
    "#;
    let err = meshlang_compiler::compile_to_python(invalid_shell, "test.mlg").unwrap_err();
    assert!(
        err.to_string().contains("transform 'shell' expects solid"),
        "surface shell should fail in type checking, got: {err}"
    );
}

#[test]
fn test_physics_topology_geo() {
    let src = r#"
        geometry test {
            let mybox = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
        }
        physics {
            group walls: neumann = test.mybox.faces
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("Boundary{ Volume{"),
        "expected Boundary in .geo"
    );
    assert!(
        geo.contains("Physical Surface(\"walls\")"),
        "expected Physical Surface"
    );
    assert!(
        !geo.contains("pg_walls() = Boundary"),
        "Boundary selector should be emitted directly in the Physical Surface command:\n{geo}"
    );
    assert!(
        geo.contains("Physical Surface(\"walls\") = Boundary{ Volume{"),
        "expected direct Boundary selector in Physical Surface command:\n{geo}"
    );
}

#[test]
fn test_physics_topology_python() {
    let src = r#"
        geometry test {
            let shell = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
        }
        physics {
            group walls: neumann = test.shell.faces
            group top_face: dirichlet = test.shell.faces.top
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("_ml_boundary(_ml_dimtags(3, shell)"),
        "expected live-boundary helper with var name 'shell'"
    );
    assert!(
        py.contains("mlg_find_box_face(_ml_single(shell, 'box face selector'), 2, 'max')"),
        "expected box top face helper"
    );
    assert!(
        py.contains("addPhysicalGroup("),
        "expected addPhysicalGroup"
    );
}

#[test]
fn test_python_cylinder_face_helper_is_not_z_axis_only() {
    let src = r#"
        geometry test {
            let port = Cylinder(r: 0.1, h: 0.5)
                |> rotate(axis: X, angle: 90)
        }
        physics {
            group outlet: radiation = test.port.faces.top
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("mlg_find_cyl_face(_ml_single(port, 'cylinder face selector'), 'top')"),
        "topology query should use the cylinder helper:\n{}",
        py
    );
    assert!(
        !py.contains("coord_idx = 2  # assume Z-aligned cylinder"),
        "cylinder helper must not assume Z-axis alignment:\n{}",
        py
    );
    assert!(
        py.contains("axis_idx = max(range(3)"),
        "cylinder helper should choose the cap axis from geometry extents:\n{}",
        py
    );
}

#[test]
fn test_diagnostics_json_format() {
    let src = r#"
        geometry unused {
            let mybox = Box(1.0, 1.0, 1.0)
        }
    "#;

    let ast = meshlang_parser::parse(src, "test.mlg").unwrap();
    let mut checker = meshlang_compiler::TypeChecker::new();
    let diags = checker.check(&ast);

    let json = meshlang_compiler::diagnostics::to_json(&diags, src, "file:///test.mlg");
    assert!(json.contains("\"source\": \"meshlang\""));
    assert!(json.contains("unused"));
}

#[test]
fn test_mesh_algorithm_options() {
    let src = r#"
        geometry test {
            let mybox = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            algorithm_2d: delaunay
            algorithm_3d: hxt
            smoothing_steps: 10
        }
        export {
            format: msh4
            filename: "test.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Mesh.Algorithm = 5"));
    assert!(geo.contains("Mesh.Algorithm3D = 10"));
    assert!(geo.contains("Mesh.Smoothing = 10"));
}

#[test]
fn unknown_mesh_options_are_typed_backend_errors() {
    let src = r#"
        geometry test {
            let mybox = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            mystery_option: 42
        }
    "#;

    for result in [
        meshlang_compiler::compile_to_geo(src, "unknown-option.mlg"),
        meshlang_compiler::compile_to_python(src, "unknown-option.mlg"),
    ] {
        let error = result.unwrap_err();
        assert!(
            matches!(error, meshlang_compiler::CompileError::Unsupported(_)),
            "unexpected error: {error}"
        );
        assert!(error.to_string().contains("mystery_option"));
    }
}

#[test]
fn disconnected_csg_results_remain_complete_in_both_backends() {
    let source = r#"
        geometry model {
            let left = Box(1.0, 1.0, 1.0)
            let right = Box(1.0, 1.0, 1.0) |> translate(3.0, 0.0, 0.0)
            let domain = left | right
        }
        physics {
            group pieces: volume = model.domain
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(source, "disconnected.mlg").unwrap();
    assert!(geo.contains("model__domain[] = BooleanUnion"));
    assert!(geo.contains("Physical Volume(\"pieces\") = {model__domain[]}"));

    let python = meshlang_compiler::compile_to_python(source, "disconnected.mlg").unwrap();
    assert!(python.contains("domain = _ml_only(3, csg_"));
    assert!(python.contains("pg_pieces = _ml_dimtags(3, domain)"));
    assert!(!python.contains("[0][0][1]"));
}

#[test]
fn shipping_corpus_compiles_through_both_backends_deterministically() {
    fn collect_mlg_files(root: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(root)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to read corpus directory {}: {error}",
                    root.display()
                )
            })
            .map(|entry| {
                entry.unwrap_or_else(|error| panic!("failed to read corpus entry: {error}"))
            })
            .collect();
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                collect_mlg_files(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "mlg") {
                files.push(path);
            }
        }
    }

    let meshlang_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let roots = [
        meshlang_root.join("examples"),
        meshlang_root.join("stdlib"),
        meshlang_root.join("../sonium-sim-pcfan/shapes"),
        meshlang_root.join("../sonium-sim-speaker/examples"),
    ];
    let mut paths = Vec::new();
    for root in roots {
        collect_mlg_files(&root, &mut paths);
    }
    paths.sort();

    // This is the checked-in product corpus manifest. A new shipping .mlg
    // must be added intentionally so it cannot silently skip the backend gate.
    let expected = [
        "../sonium-sim-pcfan/shapes/pcfan.mlg",
        "../sonium-sim-speaker/examples/two_way_speaker.mlg",
        "examples/gmsh_helmholtz_resonator.mlg",
        "examples/gmsh_t16_csg.mlg",
        "examples/gmsh_t19_loft_pipe.mlg",
        "examples/gmsh_t5_cheese.mlg",
        "examples/ported_acoustic_frame.mlg",
        "examples/room_acoustics.mlg",
        "examples/speaker_cabinet.mlg",
        "examples/sphere_scattering.mlg",
        "stdlib/acoustics.mlg",
        "stdlib/shapes.mlg",
    ];
    let discovered: Vec<_> = paths
        .iter()
        .map(|path| {
            path.strip_prefix(&meshlang_root)
                .unwrap_or_else(|error| {
                    panic!("corpus path escaped root: {}: {error}", path.display())
                })
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert_eq!(
        discovered, expected,
        "shipping MeshLang corpus manifest drifted"
    );

    for (path, relative) in paths.iter().zip(discovered.iter()) {
        let source = std::fs::read_to_string(path).unwrap_or_else(|error| {
            panic!(
                "failed to read shipping corpus file {}: {error}",
                path.display()
            )
        });
        let filename = path.to_string_lossy();
        let geo = meshlang_compiler::compile_to_geo_full(&source, &filename)
            .unwrap_or_else(|error| panic!("GEO compilation failed for {relative}: {error}"));
        let python = meshlang_compiler::compile_to_python_full(&source, &filename)
            .unwrap_or_else(|error| panic!("Python compilation failed for {relative}: {error}"));
        let geo_repeat = meshlang_compiler::compile_to_geo_full(&source, &filename)
            .unwrap_or_else(|error| panic!("GEO repeat failed for {relative}: {error}"));
        assert_eq!(
            geo.canonical_ir_hash, python.canonical_ir_hash,
            "backends must consume the same canonical IR for {relative}"
        );
        assert_eq!(
            geo.canonical_ir_json, python.canonical_ir_json,
            "canonical IR drift for {relative}"
        );
        assert_eq!(
            geo.provenance_manifest_json, python.provenance_manifest_json,
            "provenance drift between backends for {relative}"
        );
        assert_eq!(
            geo.sensitivity_manifest_json, python.sensitivity_manifest_json,
            "sensitivity drift between backends for {relative}"
        );
        assert_eq!(
            geo.code, geo_repeat.code,
            "GEO output is not deterministic for {relative}"
        );
        assert_eq!(
            geo.backend_metadata_json, geo_repeat.backend_metadata_json,
            "GEO metadata is not deterministic for {relative}"
        );
        assert!(!geo.code.is_empty(), "GEO output is empty for {relative}");
        assert!(
            !python.code.is_empty(),
            "Python output is empty for {relative}"
        );
    }
}

#[test]
fn standard_library_rounded_box_applies_fillet_in_both_backends() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut source = std::fs::read_to_string(root.join("stdlib/shapes.mlg")).unwrap();
    source.push_str(
        r#"
        geometry model {
            let body = RoundedBox(1.0, 2.0, 3.0, 0.1)
        }
        export { format: msh4  filename: "rounded-box.msh" }
        "#,
    );
    let filename = root.join("rounded-box-test.mlg");
    let filename = filename.to_string_lossy();

    let geo = meshlang_compiler::compile_to_geo_full(&source, &filename).unwrap();
    assert!(
        geo.code.contains("Fillet{"),
        "standard-library RoundedBox must emit a GEO fillet operation:\n{}",
        geo.code
    );

    let python = meshlang_compiler::compile_to_python_full(&source, &filename).unwrap();
    assert!(
        python.code.contains("occ.fillet("),
        "standard-library RoundedBox must emit a Python OCC fillet operation:\n{}",
        python.code
    );
}

#[test]
fn translated_box_refinement_emits_box_bounds_in_both_backends() {
    let source = r#"
        geometry test {
            let body = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            refine in(Box(0.1, 0.2, 0.3) |> translate(1.0, 2.0, 3.0)) {
                size: 0.01
            }
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(source, "translated-box.mlg").unwrap();
    assert!(geo.contains("XMin = (0 + 1.0);"));
    assert!(geo.contains("XMax = (0.1 + 1.0);"));
    assert!(geo.contains("YMin = (0 + 2.0);"));
    assert!(geo.contains("YMax = (0.2 + 2.0);"));
    assert!(geo.contains("ZMin = (0 + 3.0);"));
    assert!(geo.contains("ZMax = (0.3 + 3.0);"));

    let python = meshlang_compiler::compile_to_python(source, "translated-box.mlg").unwrap();
    assert!(python.contains("'XMin', (0 + 1.0)"));
    assert!(python.contains("'XMax', (0.1 + 1.0)"));
    assert!(python.contains("'YMin', (0 + 2.0)"));
    assert!(python.contains("'YMax', (0.2 + 2.0)"));
    assert!(python.contains("'ZMin', (0 + 3.0)"));
    assert!(python.contains("'ZMax', (0.3 + 3.0)"));
}

#[test]
fn dimensional_exponent_overflow_is_reported_as_a_compile_error() {
    let source = r#"
        param high = 1 m^127
        param length = 1 m
        geometry test {
            let value = high * length
        }
    "#;

    let error = meshlang_compiler::compile_to_geo(source, "dimension-overflow.mlg")
        .expect_err("dimensional exponent overflow must not be accepted");
    assert!(
        matches!(error, meshlang_compiler::CompileError::TypeError { .. }),
        "unexpected error: {error:?}"
    );
}

#[test]
fn backend_string_literals_are_escaped() {
    let source = r#"
        param label = "quote: \" slash: \\ line\n"
    "#;

    let geo = meshlang_compiler::compile_to_geo(source, "escaped-string.mlg").unwrap();
    assert!(geo.contains(r#"quote: \""#));
    assert!(geo.contains(r#"slash: \\"#));
    assert!(geo.contains(r#"line\n"#));

    let python = meshlang_compiler::compile_to_python(source, "escaped-string.mlg").unwrap();
    assert!(python.contains("quote: \""));
    assert!(python.contains(r#"slash: \\"#));
    assert!(python.contains(r#"line\n"#));
}

#[test]
fn parameterized_geometry_templates_do_not_emit_unbound_backend_code() {
    let source = r#"
        pub geometry Block(w: Float, h: Float, d: Float) {
            let body = Box(w, h, d)
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo(source, "templates.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python(source, "templates.mlg").unwrap();
    assert!(geo.contains("// geometry template: Block"));
    assert!(python.contains("# geometry template: Block"));
    assert!(!geo.contains("Box("));
    assert!(!python.contains("occ.addBox("));
}

#[test]
fn parameterized_geometry_template_instantiates_with_backend_parity() {
    let source = r#"
        pub geometry Block(w: Float, h: Float, d: Float) {
            let body = Box(w, h, d)
        }
        geometry model {
            let instance = Block(2.0, 3.0, 4.0)
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo(source, "template-instance.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python(source, "template-instance.mlg").unwrap();
    assert!(
        geo.contains("Box(model__instance) = {0, 0, 0, 2.0, 3.0, 4.0}"),
        "{geo}"
    );
    assert!(
        python.contains("instance = occ.addBox(0, 0, 0, 2.0, 3.0, 4.0)"),
        "{python}"
    );
    assert!(geo.contains("// geometry template: Block"));
    assert!(python.contains("# geometry template: Block"));
}

#[test]
fn geometry_template_argument_contract_is_typed_and_total() {
    for (name, invocation) in [
        ("extra", "Block(1.0, 2.0, 3.0, 4.0)"),
        ("mixed", "Block(1.0, h: 2.0, d: 3.0)"),
        ("unknown", "Block(w: 1.0, h: 2.0, depth: 3.0)"),
    ] {
        let source = format!(
            "pub geometry Block(w: Float, h: Float, d: Float) {{ let body = Box(w, h, d) }}\ngeometry model {{ let instance = {invocation} }}"
        );
        let error = meshlang_compiler::compile_to_geo(&source, &format!("template-{name}.mlg"))
            .unwrap_err();
        assert!(
            matches!(error, meshlang_compiler::CompileError::TypeError { .. }),
            "{name}: {error:?}"
        );
    }
}

#[test]
fn non_ported_geometry_template_lowers_multiple_bindings_with_backend_parity() {
    let source = r#"
        pub geometry Pair(width: Float) {
            let left = Box(width, 1.0, 1.0)
            let right = Box(width, 0.5, 0.5)
            left | right
        }
        geometry model {
            let pair = Pair(2.0)
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(source, "template-multi.mlg")
        .expect("multi-statement template must compile to GEO");
    let python = meshlang_compiler::compile_to_python(source, "template-multi.mlg")
        .expect("multi-statement template must compile to Python");
    let ir = meshlang_compiler::preview_canonical_ir_json(source, "template-multi.mlg")
        .expect("multi-statement template must lower to canonical IR");

    assert_eq!(geo.matches("Box(").count(), 2);
    assert!(geo.contains("BooleanUnion"));
    assert_eq!(python.matches("occ.addBox").count(), 2);
    assert!(python.contains("occ.fuse("));
    assert!(ir.contains("model.pair.left"));
    assert!(ir.contains("model.pair.right"));
    assert!(ir.contains("template-multi.mlg"));
}

#[test]
fn geometry_template_named_arguments_and_defaults_are_lowered() {
    let source = r#"
        geometry Block(w: Float, h: Float, d: Float = 1.0) {
            let body = Box(w, h, d)
        }
        geometry model {
            let instance = Block(h: 3.0, w: 2.0)
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo(source, "template-default.mlg").unwrap();
    assert!(
        geo.contains("Box(model__instance) = {0, 0, 0, 2.0, 3.0, 1.0}"),
        "{geo}"
    );
}

#[test]
fn ported_geometry_template_exposes_typed_canonical_entity_ports() {
    let source = r#"
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
    let geo = meshlang_compiler::compile_to_geo_full(source, "ported-template.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python_full(source, "ported-template.mlg").unwrap();
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(geo.code.contains("Box(model__frame__body)"), "{}", geo.code);
    assert!(
        geo.code.contains("Cylinder(model__frame__inlet)"),
        "{}",
        geo.code
    );
    assert!(
        python.code.contains("frame__body = occ.addBox("),
        "{}",
        python.code
    );
    assert!(
        python.code.contains("frame__inlet = occ.addCylinder("),
        "{}",
        python.code
    );

    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    let names = ir
        .entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.frame.body"));
    assert!(names.contains(&"model.frame.inlet"));
    assert!(names.contains(&"model.shifted_inlet"));
    assert_eq!(
        ir.physical_groups[0].entities[0].dimension,
        meshlang_ast::EntityDimension::Volume
    );
}

#[test]
fn ported_geometry_template_rejects_missing_and_mismatched_ports() {
    for (name, declaration) in [
        (
            "missing",
            "geometry Broken() -> (body: Solid) { let other = Box(1.0, 1.0, 1.0) }",
        ),
        (
            "lookalike",
            "geometry Broken() -> (body: Solid) { let body_1 = Box(1.0, 1.0, 1.0) }",
        ),
        (
            "mismatched",
            "geometry Broken() -> (edge: Curve) { let edge = Rectangle(1.0, 1.0) }",
        ),
        (
            "scalar",
            "geometry Broken() -> (count: Int) { let count = Box(1.0, 1.0, 1.0) }",
        ),
        (
            "duplicate",
            "geometry Broken() -> (body: Solid, body: Solid) { let body = Box(1.0, 1.0, 1.0) }",
        ),
        (
            "defaulted",
            "geometry Broken() -> (body: Solid = 1.0) { let body = Box(1.0, 1.0, 1.0) }",
        ),
    ] {
        let source = format!("{declaration}\ngeometry model {{ let instance = Broken() }}");
        let error = meshlang_compiler::compile_to_geo(&source, &format!("ported-{name}.mlg"))
            .expect_err("invalid template ports must fail before backend emission");
        assert!(
            matches!(error, meshlang_compiler::CompileError::TypeError { .. }),
            "{name}: {error:?}"
        );
    }
}

#[test]
fn nested_ported_geometry_templates_preserve_instance_scopes_and_ports() {
    let source = r#"
        pub geometry Inner(width: Float) -> (body: Solid, inlet: Solid) {
            let body = Box(width, 1.0, 1.0)
            let inlet = Cylinder(r: 0.1, h: 0.5)
        }
        pub geometry Outer(width: Float) -> (body: Solid, inlet: Solid) {
            let inner = Inner(width)
            let body = inner.body
            let inlet = inner.inlet
        }
        geometry model {
            let outer = Outer(2.0)
            let shifted = outer.inlet |> translate(1.0, 0.0, 0.0)
        }
    "#;

    let compiled = meshlang_compiler::compile_to_geo_full(source, "nested-ported.mlg")
        .expect("nested ported template must compile");
    let ir =
        serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&compiled.canonical_ir_json)
            .unwrap()
            .ir;
    let names = ir
        .entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.outer.inner.body"));
    assert!(names.contains(&"model.outer.inner.inlet"));
    assert!(names.contains(&"model.outer.body"));
    assert!(names.contains(&"model.outer.inlet"));
    assert!(names.contains(&"model.shifted"));
    assert!(compiled.code.contains("Box(model__outer__inner__body)"));
    assert!(compiled
        .code
        .contains("Cylinder(model__outer__inner__inlet)"));
}

#[test]
fn ported_template_loop_bindings_form_one_canonical_port_union() {
    let source = r#"
        pub geometry Repeated(width: Float) -> (body: Solid) {
            for i in 0..2 {
                let body = Box(width, 1.0, 1.0)
            }
        }
        geometry model {
            let repeated = Repeated(2.0)
            let shifted = repeated.body |> translate(1.0, 0.0, 0.0)
        }
    "#;

    let compiled = meshlang_compiler::compile_to_geo_full(source, "ported-loop.mlg")
        .expect("loop-produced port must lower to a canonical aggregate");
    let python = meshlang_compiler::compile_to_python_full(source, "ported-loop.mlg")
        .expect("loop-produced port must lower through Python too");
    assert_eq!(compiled.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(python.code.matches("occ.fuse(").count(), 1);
    let ir =
        serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&compiled.canonical_ir_json)
            .unwrap()
            .ir;
    let body = ir
        .entities
        .iter()
        .find(|entity| entity.name == "model.repeated.body")
        .expect("declared port must have a stable canonical name");
    assert!(matches!(
        body.operation,
        meshlang_ast::GeometryOperation::Boolean {
            kind: meshlang_ast::BooleanOperation::Union
        }
    ));
    assert_eq!(body.inputs.len(), 2);
    assert!(body.provenance.len() >= 2);
    assert!(ir
        .entities
        .iter()
        .any(|entity| entity.name == "model.shifted"));
    assert_eq!(compiled.code.matches("BooleanUnion").count(), 1);
}

#[test]
fn real_gmsh_mesh_contains_entities_groups_markers_and_topology_when_configured() {
    let Ok(gmsh) = std::env::var("MESHLANG_GMSH") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_GMSH is required for this CI execution lane"
        );
        return;
    };
    let source = r#"
        geometry validation {
            let domain = Box(1.0, 1.0, 1.0)
            let port = Cylinder(r: 0.2, h: 0.8) |> translate(2.0, 0.0, 0.0)
        }
        mesh for validation {
            min_size: 0.1
            max_size: 0.3
            order: 1
        }
        physics {
            group fluid: volume = validation.domain
            group walls: neumann = validation.domain.faces
            group floor: neumann = validation.domain.faces.bottom
            group ceiling: neumann = validation.domain.faces.top
            group port_air: volume = validation.port
            group inlet: velocity = validation.port.faces.bottom
            group outlet: radiation = validation.port.faces.top
        }
        export { format: msh4 order: 1 filename: "validation.msh" }
    "#;
    let compiled = meshlang_compiler::compile_to_geo_full(source, "validation.mlg").unwrap();
    let geo = compiled.code.clone();
    let directory = tempfile::tempdir().unwrap();
    let geo_path = directory.path().join("validation.geo");
    let mesh_path = directory.path().join("validation.msh");
    std::fs::write(&geo_path, geo).unwrap();

    let output = std::process::Command::new(&gmsh)
        .arg("-3")
        .arg(&geo_path)
        .arg("-format")
        .arg("msh4")
        .arg("-o")
        .arg(&mesh_path)
        .arg("-nopopup")
        .arg("-v")
        .arg("2")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Gmsh failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mesh = std::fs::read_to_string(&mesh_path).unwrap();
    if let Ok(artifact_dir) = std::env::var("MESHLANG_ARTIFACT_DIR") {
        let artifact_dir = std::path::Path::new(&artifact_dir);
        std::fs::create_dir_all(artifact_dir).unwrap();
        std::fs::write(
            artifact_dir.join("geo-backend-metadata.json"),
            compiled.backend_metadata_json,
        )
        .unwrap();
        std::fs::copy(&mesh_path, artifact_dir.join("geo-validation.msh")).unwrap();
    }

    fn section<'a>(mesh: &'a str, name: &str) -> Vec<&'a str> {
        let marker = format!("${name}\n");
        let start = mesh
            .find(&marker)
            .unwrap_or_else(|| panic!("missing MSH4 section ${name}"));
        let mut lines = Vec::new();
        for line in mesh[start + marker.len()..].lines() {
            if line.starts_with("$End") {
                break;
            }
            lines.push(line);
        }
        lines
    }

    assert!(mesh.starts_with("$MeshFormat\n4.1 0 8\n"));
    assert!(mesh.contains("$Nodes\n"));
    assert!(mesh.contains("$Elements\n"));

    // MSH4's first Entities line is the authoritative topology cardinality
    // (points, curves, surfaces, volumes), so this catches a backend that
    // merely prints marker strings without producing the expected model.
    let entities = section(&mesh, "Entities");
    let entity_counts: Vec<usize> = entities[0]
        .split_whitespace()
        .map(|value| value.parse().expect("numeric MSH4 entity count"))
        .collect();
    assert_eq!(
        entity_counts.len(),
        4,
        "invalid $Entities header: {entities:?}"
    );
    assert!(entity_counts[..3].iter().all(|count| *count > 0));
    assert_eq!(entity_counts[3], 2, "Box + translated Cylinder volumes");

    let physical_names = section(&mesh, "PhysicalNames");
    let physical_count: usize = physical_names[0]
        .parse()
        .expect("numeric MSH4 physical-name count");
    assert_eq!(
        physical_count, 7,
        "all declared physics groups must survive"
    );
    for group in [
        "fluid", "walls", "floor", "ceiling", "port_air", "inlet", "outlet",
    ] {
        assert!(
            physical_names
                .iter()
                .skip(1)
                .any(|line| line.ends_with(&format!("\"{group}\""))),
            "missing physical group {group}"
        );
    }
}

#[test]
fn configured_gmsh_executes_shipping_geometry_corpus() {
    let Ok(gmsh) = std::env::var("MESHLANG_GMSH") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_GMSH is required for this CI execution lane"
        );
        return;
    };

    let directory = tempfile::tempdir().unwrap();
    for example in [
        "gmsh_t5_cheese.mlg",
        "gmsh_t16_csg.mlg",
        "gmsh_helmholtz_resonator.mlg",
        "ported_acoustic_frame.mlg",
    ] {
        let source = load_example(example);
        let compiled = meshlang_compiler::compile_to_geo_full(&source, example)
            .unwrap_or_else(|error| panic!("{example} should compile: {error}"));
        let stem = example.trim_end_matches(".mlg");
        let geo_path = directory.path().join(format!("{stem}.geo"));
        let mesh_path = directory.path().join(format!("{stem}.msh"));
        std::fs::write(&geo_path, compiled.code).unwrap();

        let output = std::process::Command::new(&gmsh)
            .arg("-3")
            .arg(&geo_path)
            .arg("-format")
            .arg("msh4")
            .arg("-o")
            .arg(&mesh_path)
            .arg("-nopopup")
            .arg("-v")
            .arg("2")
            .output()
            .unwrap_or_else(|error| panic!("failed to launch Gmsh for {example}: {error}"));
        assert!(
            output.status.success(),
            "Gmsh failed for {example}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let mesh = std::fs::read_to_string(&mesh_path)
            .unwrap_or_else(|error| panic!("{example} did not emit MSH4 output: {error}"));
        assert!(mesh.starts_with("$MeshFormat\n4.1 0 8\n"), "{example}");
        assert!(mesh.contains("$Entities\n"), "{example} lacks entities");
        assert!(mesh.contains("$Elements\n"), "{example} lacks elements");
    }
}

#[test]
fn generated_python_mesh_can_be_validated_when_python_gmsh_is_configured() {
    let Ok(python) = std::env::var("MESHLANG_PYTHON") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_PYTHON is required for this CI execution lane"
        );
        return;
    };
    let source = r#"
        geometry validation {
            let domain = Box(1.0, 1.0, 1.0)
        }
        physical_region volume fluid {
            entities: validation.domain
        }
        boundary surface walls {
            entities: validation.domain.faces
            kind: neumann
            value: 0.0
        }
        export { format: msh4 filename: "python-validation.msh" }
    "#;
    let compiled =
        meshlang_compiler::compile_to_python_full(source, "python-validation.mlg").unwrap();
    let script = compiled.code.clone();
    let directory = tempfile::tempdir().unwrap();
    let script_path = directory.path().join("validation.py");
    std::fs::write(&script_path, script).unwrap();
    let output = std::process::Command::new(&python)
        .arg(&script_path)
        .current_dir(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python/Gmsh failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mesh = std::fs::read_to_string(directory.path().join("python-validation.msh")).unwrap();
    if let Ok(artifact_dir) = std::env::var("MESHLANG_ARTIFACT_DIR") {
        let artifact_dir = std::path::Path::new(&artifact_dir);
        std::fs::create_dir_all(artifact_dir).unwrap();
        std::fs::write(
            artifact_dir.join("python-backend-metadata.json"),
            compiled.backend_metadata_json,
        )
        .unwrap();
        std::fs::copy(
            directory.path().join("python-validation.msh"),
            artifact_dir.join("python-validation.msh"),
        )
        .unwrap();
    }
    assert!(mesh.contains("$Entities\n"));
    assert!(mesh.contains("$Nodes\n"));
    assert!(mesh.contains("$Elements\n"));
    assert!(mesh.contains("\"fluid\""));
    assert!(mesh.contains("\"walls\""));
}

#[test]
fn configured_python_gmsh_executes_ported_shipping_template() {
    let Ok(python) = std::env::var("MESHLANG_PYTHON") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_PYTHON is required for this CI execution lane"
        );
        return;
    };

    let example = "ported_acoustic_frame.mlg";
    let source = load_example(example);
    let compiled = meshlang_compiler::compile_to_python_full(&source, example)
        .unwrap_or_else(|error| panic!("{example} should compile: {error}"));
    let directory = tempfile::tempdir().unwrap();
    let script_path = directory.path().join("ported_acoustic_frame.py");
    std::fs::write(&script_path, compiled.code).unwrap();
    let output = std::process::Command::new(&python)
        .arg(&script_path)
        .current_dir(directory.path())
        .output()
        .unwrap_or_else(|error| panic!("failed to launch Python/Gmsh for {example}: {error}"));
    assert!(
        output.status.success(),
        "Python/Gmsh failed for {example}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mesh = std::fs::read_to_string(directory.path().join("ported_acoustic_frame.msh"))
        .unwrap_or_else(|error| panic!("{example} did not emit MSH4 output: {error}"));
    assert!(mesh.contains("$Entities\n"), "{example} lacks entities");
    assert!(mesh.contains("$Nodes\n"), "{example} lacks nodes");
    assert!(mesh.contains("$Elements\n"), "{example} lacks elements");
    for group in ["frame_air", "inlet_air", "outlet_air"] {
        assert!(mesh.contains(&format!("\"{group}\"")), "{example}: {group}");
    }
}

#[test]
fn configured_python_gmsh_executes_native_shipping_corpus() {
    let Ok(python) = std::env::var("MESHLANG_PYTHON") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_PYTHON required CI execution lane"
        );
        return;
    };

    let examples = [
        ("gmsh_t5_cheese.mlg", "t5_cheese.msh"),
        ("gmsh_t16_csg.mlg", "t16_csg.msh"),
        ("gmsh_helmholtz_resonator.mlg", "helmholtz_resonator.msh"),
        ("ported_acoustic_frame.mlg", "ported_acoustic_frame.msh"),
    ];

    for (example, mesh_name) in examples {
        if let Ok(selected) = std::env::var("MESHLANG_PYTHON_SHIPPING_CASE") {
            if selected != example {
                continue;
            }
        }
        let source = load_example(example);
        let compiled = meshlang_compiler::compile_to_python_full(&source, example)
            .unwrap_or_else(|error| panic!("{example} should compile: {error}"));
        let directory = tempfile::tempdir().unwrap();
        let script_path = directory.path().join(example.replace(".mlg", ".py"));
        std::fs::write(&script_path, compiled.code).unwrap();
        let output = std::process::Command::new(&python)
            .arg(&script_path)
            .current_dir(directory.path())
            .output()
            .unwrap_or_else(|error| panic!("failed to launch Python/Gmsh {example}: {error}"));
        assert!(
            output.status.success(),
            "Python/Gmsh failed {example}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let mesh = std::fs::read_to_string(directory.path().join(mesh_name))
            .unwrap_or_else(|error| panic!("{example} did not emit MSH4 output: {error}"));
        assert!(
            mesh.starts_with("$MeshFormat\n4.1 0 8\n"),
            "{example} lacks MSH4 header"
        );
        assert!(mesh.contains("$Entities\n"), "{example} lacks entities");
        assert!(mesh.contains("$Nodes\n"), "{example} lacks nodes");
        assert!(mesh.contains("$Elements\n"), "{example} lacks elements");
    }
}

#[test]
fn configured_python_gmsh_executes_product_geometry_corpus() {
    let Ok(python) = std::env::var("MESHLANG_PYTHON") else {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "MESHLANG_PYTHON required CI execution lane"
        );
        return;
    };

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cases = [
        (
            "pcfan",
            root.join("../sonium-sim-pcfan/shapes/pcfan.mlg"),
            "pcfan_meshlang.msh",
        ),
        (
            "room_acoustics",
            root.join("examples/room_acoustics.mlg"),
            "room.msh",
        ),
        (
            "sphere_scattering",
            root.join("examples/sphere_scattering.mlg"),
            "sphere_bem.msh",
        ),
        (
            "speaker_cabinet",
            root.join("examples/speaker_cabinet.mlg"),
            "cabinet_bem.msh",
        ),
    ];

    for (case_name, source_path, mesh_name) in cases {
        if let Ok(selected) = std::env::var("MESHLANG_PYTHON_PRODUCT_CASE") {
            if selected != case_name {
                continue;
            }
        }
        let source = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("failed to read {case_name}: {error}"));
        let filename = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(case_name);
        let compiled = meshlang_compiler::compile_to_python_full(&source, filename)
            .unwrap_or_else(|error| panic!("{case_name} should compile: {error}"));
        let directory = tempfile::tempdir().unwrap();
        let script_path = directory.path().join(format!("{case_name}.py"));
        std::fs::write(&script_path, compiled.code).unwrap();
        let output = std::process::Command::new(&python)
            .arg(&script_path)
            .current_dir(directory.path())
            .output()
            .unwrap_or_else(|error| panic!("failed to launch Python/Gmsh {case_name}: {error}"));
        assert!(
            output.status.success(),
            "Python/Gmsh failed {case_name}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let mesh = std::fs::read_to_string(directory.path().join(mesh_name))
            .unwrap_or_else(|error| panic!("{case_name} did not emit MSH4 output: {error}"));
        assert!(
            mesh.starts_with("$MeshFormat\n4.1 0 8\n"),
            "{case_name} lacks MSH4 header"
        );
        assert!(mesh.contains("$Entities\n"), "{case_name} lacks entities");
        assert!(mesh.contains("$Nodes\n"), "{case_name} lacks nodes");
        assert!(mesh.contains("$Elements\n"), "{case_name} lacks elements");
    }
}

#[test]
fn test_multiple_exports() {
    let src = r#"
        geometry test {
            let mybox = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
        }
        export {
            format: msh4
            filename: "test.msh"
        }
        export as "visual" {
            format: stl
            filename: "test.stl"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Output: test.msh"));
    assert!(geo.contains("Output: test.stl"));
}

// ── Import / include tests ────────────────────────────────────────────────────

/// Write a `.mlg` file to a temp dir, return (dir, path).
#[cfg(test)]
fn write_mlg(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, content).unwrap();
    p
}

#[test]
fn test_import_whole_inlines_public_decls() {
    let tmp = tempfile::tempdir().unwrap();

    write_mlg(
        tmp.path(),
        "common.mlg",
        r#"
        const C = 343.0
        param f_max = 1000.0
    "#,
    );

    let main_path = write_mlg(
        tmp.path(),
        "main.mlg",
        r#"
        import "common.mlg"

        geometry room {
            let walls = Box(1.0, 2.0, 3.0)
        }
        export {
            format: msh4
            filename: "room.msh"
        }
    "#,
    );

    let source = std::fs::read_to_string(&main_path).unwrap();
    let geo = meshlang_compiler::compile_to_geo(&source, main_path.to_str().unwrap()).unwrap();

    // const and param from common.mlg must appear.
    assert!(
        geo.contains("C = 343"),
        "expected C from common.mlg in geo output"
    );
    assert!(
        geo.contains("f_max = 1000"),
        "expected f_max from common.mlg in geo output"
    );
}

#[test]
fn test_import_selective_includes_only_named() {
    let tmp = tempfile::tempdir().unwrap();

    write_mlg(
        tmp.path(),
        "shapes.mlg",
        r#"
        const A = 1.0
        const B = 2.0
        const C = 3.0
    "#,
    );

    let main_path = write_mlg(
        tmp.path(),
        "main.mlg",
        r#"
        import { A, C } from "shapes.mlg"

        geometry g {
            let b = Box(A, 1.0, C)
        }
        export { format: msh4  filename: "out.msh" }
    "#,
    );

    let source = std::fs::read_to_string(&main_path).unwrap();
    let geo = meshlang_compiler::compile_to_geo(&source, main_path.to_str().unwrap()).unwrap();

    assert!(geo.contains("A = 1"), "A should be imported");
    assert!(geo.contains("C = 3"), "C should be imported");
    assert!(!geo.contains("B = 2"), "B was not requested");
}

#[test]
fn test_import_diamond_deduplication() {
    // main → a, b; a → common; b → common
    // common's declarations must appear exactly once.
    let tmp = tempfile::tempdir().unwrap();

    write_mlg(
        tmp.path(),
        "common.mlg",
        r#"
        const SHARED = 99.0
    "#,
    );

    write_mlg(
        tmp.path(),
        "a.mlg",
        r#"
        import "common.mlg"
        const A_VAL = 1.0
    "#,
    );

    write_mlg(
        tmp.path(),
        "b.mlg",
        r#"
        import "common.mlg"
        const B_VAL = 2.0
    "#,
    );

    let main_path = write_mlg(
        tmp.path(),
        "main.mlg",
        r#"
        import "a.mlg"
        import "b.mlg"

        geometry g {
            let b = Box(SHARED, A_VAL, B_VAL)
        }
        export { format: msh4  filename: "out.msh" }
    "#,
    );

    let source = std::fs::read_to_string(&main_path).unwrap();
    let geo = meshlang_compiler::compile_to_geo(&source, main_path.to_str().unwrap()).unwrap();

    // SHARED must appear exactly once, not twice.
    let count = geo.matches("SHARED = 99").count();
    assert_eq!(count, 1, "SHARED should appear exactly once (got {count})");
    assert!(geo.contains("A_VAL = 1"));
    assert!(geo.contains("B_VAL = 2"));
}

#[test]
fn test_import_mesh_and_physics_not_exported() {
    // mesh / physics / export blocks in imported files should be dropped.
    let tmp = tempfile::tempdir().unwrap();

    write_mlg(
        tmp.path(),
        "lib.mlg",
        r#"
        const SIZE = 0.05

        pub geometry standard_box {
            let shell = Box(1.0, 1.0, 1.0)
        }

        mesh for standard_box {
            default_size: SIZE
        }

        physics {
            group walls: neumann = standard_box.shell.faces
        }
    "#,
    );

    let main_path = write_mlg(
        tmp.path(),
        "main.mlg",
        r#"
        import "lib.mlg"

        export { format: msh4  filename: "out.msh" }
    "#,
    );

    let source = std::fs::read_to_string(&main_path).unwrap();
    let geo = meshlang_compiler::compile_to_geo(&source, main_path.to_str().unwrap()).unwrap();

    assert!(geo.contains("SIZE = 0.05"), "const SIZE should be imported");
    assert!(geo.contains("Box("), "geometry should be imported");
    // The mesh/physics blocks from the library should not bleed in.
    assert!(
        !geo.contains("Mesh.CharacteristicLengthMax"),
        "mesh block must not be re-exported"
    );
    assert!(
        !geo.contains("Physical Surface(\"walls\")"),
        "physics block must not be re-exported"
    );
}

/// End-to-end test: full speaker cabinet (6-plate model) compiles to valid .geo and .py
#[test]
fn test_e2e_speaker_cabinet() {
    let src = std::fs::read_to_string("examples/speaker_cabinet.mlg").unwrap_or_else(|_| {
        // Inline fallback if file not accessible from test cwd
        r#"
                const C_AIR = 343.0
                const ELEM_PER_WAVELENGTH = 6
                param f_max = 2000.0
                param wavelength = C_AIR / f_max
                param elem_size = wavelength / ELEM_PER_WAVELENGTH
                param cab_w = 0.20
                param cab_h = 0.35
                param cab_d = 0.25
                param wall_t = 0.015
                param port_r = 0.025
                param port_d = 0.15
                param driver_r = 0.065
                param port_x = 0.10
                param port_y = 0.07
                param driver_x = 0.10
                param driver_y = 0.25

                geometry cabinet {
                    let back = Box(cab_w, cab_h, wall_t)
                    let left = Box(wall_t, cab_h, cab_d - wall_t)
                        |> translate(0.0, 0.0, wall_t)
                    let right = Box(wall_t, cab_h, cab_d - wall_t)
                        |> translate(cab_w - wall_t, 0.0, wall_t)
                    let bottom = Box(cab_w - 2.0 * wall_t, wall_t, cab_d - wall_t)
                        |> translate(wall_t, 0.0, wall_t)
                    let top = Box(cab_w - 2.0 * wall_t, wall_t, cab_d - wall_t)
                        |> translate(wall_t, cab_h - wall_t, wall_t)
                    let front_solid = Box(cab_w, cab_h, wall_t)
                        |> translate(0.0, 0.0, cab_d - wall_t)
                    let port_cutout = Cylinder(r: port_r, h: wall_t)
                        |> translate(port_x, port_y, cab_d - wall_t)
                    let driver_cutout = Cylinder(r: driver_r, h: wall_t)
                        |> translate(driver_x, driver_y, cab_d - wall_t)
                    let front = front_solid - port_cutout - driver_cutout
                    let port = Cylinder(r: port_r, h: port_d)
                        |> translate(port_x, port_y, cab_d - wall_t - port_d)
                    let shell = back | left | right | bottom | top | front | port
                }

                mesh for cabinet {
                    default_size: elem_size
                    min_size: 0.001
                    order: 2
                    refine near(cabinet.port, distance: 0.03) {
                        size: 0.003
                    }
                }

                physics {
                    group cabinet_wall: neumann = cabinet.shell.faces
                    group port_exit: radiation = cabinet.port.faces.top
                    group driver: velocity = cabinet.front.faces
                }

                export {
                    format: msh4
                    order: 2
                    filename: "cabinet_bem.msh"
                }
            "#
        .to_string()
    });

    // .geo output
    let geo_result = meshlang_compiler::compile_to_geo_full(&src, "speaker_cabinet.mlg");
    assert!(
        geo_result.is_ok(),
        "geo compilation failed: {:?}",
        geo_result.err()
    );
    let geo = geo_result.unwrap();

    // Verify .geo structure
    assert!(geo.code.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(geo.code.contains("Box("), "should have Box primitives");
    assert!(
        geo.code.contains("Cylinder("),
        "should have Cylinder primitives"
    );
    assert!(
        geo.code.contains("BooleanDifference"),
        "front plate needs CSG difference for holes"
    );
    assert!(
        geo.code.contains("BooleanUnion"),
        "plates are assembled via CSG union"
    );
    assert!(geo.code.contains("Mesh.ElementOrder = 2"));
    assert!(geo.code.contains("Field["), "should have refinement fields");
    assert!(geo.code.contains("Physical Surface(\"cabinet_wall\")"));
    assert!(geo.code.contains("Output: cabinet_bem.msh"));

    // Verify tag map has all plates
    assert!(geo.tag_map_json.is_some());
    let map = geo.tag_map_json.unwrap();
    assert!(
        map.contains("cabinet.back"),
        "tag map should have back plate"
    );
    assert!(
        map.contains("cabinet.front"),
        "tag map should have front plate"
    );
    assert!(
        map.contains("cabinet.port"),
        "tag map should have port tube"
    );
    assert!(
        map.contains("cabinet.shell"),
        "tag map should have assembled shell"
    );

    // No type errors
    let errors: Vec<_> = geo
        .diagnostics
        .iter()
        .filter(|d| d.level == meshlang_compiler::DiagLevel::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected type errors: {:?}", errors);

    // .py output
    let py = meshlang_compiler::compile_to_python(&src, "speaker_cabinet.mlg").unwrap();

    // Verify Python structure
    assert!(py.contains("import gmsh"));
    assert!(py.contains("occ.addBox("), "should have Box primitives");
    assert!(
        py.contains("occ.addCylinder("),
        "should have Cylinder primitives"
    );
    assert!(py.contains("occ.cut("), "front plate holes use occ.cut");
    assert!(py.contains("occ.fuse("), "plates assembled with occ.fuse");
    assert!(
        py.contains("occ.translate("),
        "plates are positioned with translate"
    );
    assert!(
        py.contains("addPhysicalGroup("),
        "should have physical groups"
    );
    assert!(py.contains("gmsh.write('cabinet_bem.msh')"));
    assert!(py.contains("gmsh.finalize()"));
}

/// End-to-end: room acoustics with box face resolution
#[test]
fn test_e2e_room_acoustics() {
    let src = r#"
        const C_AIR = 343.0
        param f_max = 1000.0
        param elem_size = C_AIR / f_max / 6.0

        geometry room {
            let walls = Box(5.0, 4.0, 3.0)
        }

        mesh for room {
            default_size: elem_size
            order: 1
        }

        physics {
            group floor: impedance = room.walls.faces.bottom
            group ceiling: impedance = room.walls.faces.top
            group air: volume = room.walls
        }

        export {
            format: msh4
            filename: "room.msh"
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "room.mlg").unwrap();
    assert!(
        py.contains("mlg_find_box_face(_ml_single(walls, 'box face selector'), 2, 'min')"),
        "floor should be Z min"
    );
    assert!(
        py.contains("mlg_find_box_face(_ml_single(walls, 'box face selector'), 2, 'max')"),
        "ceiling should be Z max"
    );
    assert!(
        py.contains("addPhysicalGroup(3,"),
        "air should be volume group (dim 3)"
    );

    let geo = meshlang_compiler::compile_to_geo(src, "room.mlg").unwrap();
    assert!(
        geo.contains("Physical Volume(\"air\")"),
        "air should be Physical Volume in .geo"
    );
}

// ── Bug regression tests ────────────────────────────────────────────────────

#[test]
fn test_geometry_local_bindings_are_isolated_in_both_backends() {
    let src = r#"
        geometry first {
            let shared = Box(1.0, 1.0, 1.0)
        }
        geometry second {
            let shared = Sphere(r: 0.5)
        }
        export { format: msh4  filename: "isolated.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo_full(src, "isolated.mlg").unwrap();
    assert!(geo.code.contains("// geometry: first"));
    assert!(geo.code.contains("// geometry: second"));
    let tag_map: serde_json::Value =
        serde_json::from_str(geo.tag_map_json.as_ref().unwrap()).unwrap();
    let names: Vec<_> = tag_map["entities"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    assert!(names.contains(&"first.shared"));
    assert!(names.contains(&"second.shared"));

    let python = meshlang_compiler::compile_to_python(src, "isolated.mlg").unwrap();
    assert!(python.contains("# geometry: first"));
    assert!(python.contains("# geometry: second"));
    assert!(python.contains("occ.addBox("));
    assert!(python.contains("occ.addSphere("));
}

#[test]
fn test_runtime_tags_are_positive_unique_and_deterministic() {
    let src = r#"
        geometry transformed {
            let first = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
            let second = Box(1.0, 1.0, 1.0) |> translate(2.0, 0.0, 0.0)
        }
        export { format: msh4  filename: "tags.msh" }
    "#;

    let first = meshlang_compiler::compile_to_geo_full(src, "tags.mlg").unwrap();
    let second = meshlang_compiler::compile_to_geo_full(src, "tags.mlg").unwrap();
    assert_eq!(first.tag_map_json, second.tag_map_json);

    let tag_map: serde_json::Value =
        serde_json::from_str(first.tag_map_json.as_ref().unwrap()).unwrap();
    let tags: Vec<_> = tag_map["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["tag"].as_i64().unwrap())
        .collect();
    assert!(
        tags.iter().all(|tag| *tag > 0),
        "sentinel tags leaked: {tags:?}"
    );
    let unique: std::collections::HashSet<_> = tags.iter().collect();
    assert_eq!(
        unique.len(),
        tags.len(),
        "runtime entities collided: {tags:?}"
    );

    let python = meshlang_compiler::compile_to_python(src, "tags.mlg").unwrap();
    assert_eq!(python.matches("occ.translate(").count(), 2);
}

#[test]
fn canonical_artifacts_and_backend_plans_are_shared() {
    let source = r#"
        geometry model {
            let outer = Box(2.0, 2.0, 2.0)
            let inner = Sphere(r: 0.5)
            let result = outer - inner
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo_full(source, "shared-plan.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python_full(source, "shared-plan.mlg").unwrap();
    assert_eq!(geo.canonical_ir_hash, python.canonical_ir_hash);
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert!(!geo.sensitivity_manifest_json.is_empty());
    assert!(geo.provenance_manifest_json.contains("shared-plan.mlg"));

    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    let tag_map: meshlang_compiler::tag_allocator::TagMap =
        serde_json::from_str(geo.tag_map_json.as_deref().unwrap()).unwrap();
    for node in ir
        .entities
        .iter()
        .filter(|node| !node.name.contains(".__ir_"))
    {
        let entry = tag_map
            .entities
            .iter()
            .find(|entry| entry.name == node.name)
            .unwrap();
        assert_eq!(u64::try_from(entry.tag).unwrap(), node.id.index);
        assert_eq!(entry.dim as u8, node.id.dimension.topological_dimension());
        assert!(!entry.provenance.is_empty(), "{}", entry.name);
    }
    let metadata: serde_json::Value = serde_json::from_str(&geo.backend_metadata_json).unwrap();
    assert_eq!(
        metadata["source_map"]["source"]["identity"]["display_name"],
        "shared-plan.mlg"
    );
    assert!(!metadata["source_map"]["entities"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn backend_metadata_records_reproducible_artifact_hash() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 2.0, 3.0)
        }
    "#;
    let first = meshlang_compiler::compile_to_geo_full(source, "repro.mlg").unwrap();
    let second = meshlang_compiler::compile_to_geo_full(source, "repro.mlg").unwrap();
    assert_eq!(first.code, second.code);

    let metadata: serde_json::Value = serde_json::from_str(&first.backend_metadata_json).unwrap();
    assert_eq!(
        metadata["reproducibility"]["artifact_hash"],
        meshlang_ast::StructuralHash::from_bytes(first.code.as_bytes()).to_string()
    );
    assert_eq!(
        metadata["reproducibility"]["hash_kind"],
        "meshlang-structural-fnv1a-256"
    );
}

#[test]
fn material_properties_are_present_in_the_provenance_manifest() {
    let source = r#"
        geometry domain {
            let body = Box(1.0, 1.0, 1.0)
        }
        material_region steel {
            region: domain.body
            material: Steel
            density: 7800.0
        }
    "#;

    let result = meshlang_compiler::compile_to_geo_full(source, "material.mlg").unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&result.provenance_manifest_json).unwrap();
    let regions = manifest["material_regions"].as_array().unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0]["name"], "steel");
    assert_eq!(regions[0]["properties"][0]["name"], "density");
}

#[test]
fn physical_region_material_is_preserved_in_ir_and_backend_metadata() {
    let source = r#"
        geometry domain {
            let body = Box(1.0, 1.0, 1.0)
        }
        physical_region volume body_region {
            entities: domain.body
            material: Steel
        }
    "#;
    let result = meshlang_compiler::compile_to_geo_full(source, "physical-material.mlg").unwrap();
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&result.canonical_ir_json)
        .unwrap()
        .ir;
    assert_eq!(ir.physical_groups.len(), 1);
    assert!(ir.physical_groups[0].material.is_some());
    assert!(result.backend_metadata_json.contains("material_regions"));
    assert!(result.backend_metadata_json.contains("body_region"));
}

#[test]
fn design_variable_perturbation_preserves_entity_and_marker_provenance() {
    let source = |initial: f64| {
        format!(
            r#"
            design width {{
                type: continuous
                initial: {initial}
                lower: 0.5
                upper: 2.0
            }}
            geometry model {{
                let domain = Box(width, 1.0, 1.0)
            }}
            "#
        )
    };
    let first = meshlang_compiler::compile_to_geo_full(&source(1.0), "variant.mlg").unwrap();
    let second = meshlang_compiler::compile_to_geo_full(&source(1.25), "variant.mlg").unwrap();
    let first_ir =
        serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&first.canonical_ir_json)
            .unwrap()
            .ir;
    let second_ir =
        serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&second.canonical_ir_json)
            .unwrap()
            .ir;

    assert_ne!(first.canonical_ir_hash, second.canonical_ir_hash);
    assert_eq!(
        first_ir
            .entities
            .iter()
            .map(|node| (&node.name, node.id))
            .collect::<Vec<_>>(),
        second_ir
            .entities
            .iter()
            .map(|node| (&node.name, node.id))
            .collect::<Vec<_>>()
    );
    assert_eq!(first_ir.sensitivity.entries.len(), 1);
    assert_eq!(
        first_ir.sensitivity.entries[0].entities,
        second_ir.sensitivity.entries[0].entities
    );
    assert_eq!(
        first_ir.sensitivity.entries[0].remesh,
        meshlang_ast::RemeshResponsibility::Global
    );
}

#[test]
fn sensitivity_propagates_through_internal_csg_and_topology_nodes() {
    let source = r#"
        design width {
            type: continuous
            initial: 1.0
            lower: 0.5
            upper: 2.0
        }
        design radius {
            type: continuous
            initial: 0.25
            lower: 0.1
            upper: 0.5
        }
        geometry model {
            let base = Box(width, 1.0, 1.0)
            let tool = Sphere(r: radius)
            let cut = base - tool
            let other = Box(2.0, radius, 1.0)
        }
        physical_region volume cut_group {
            entities: model.cut
        }
        physical_region volume other_group {
            entities: model.other
        }
        physical_region surface cut_faces {
            entities: model.cut.faces
        }
    "#;

    let result = meshlang_compiler::compile_to_geo_full(source, "sensitivity-nested.mlg").unwrap();
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&result.canonical_ir_json)
        .unwrap()
        .ir;

    let names = ir
        .entities
        .iter()
        .map(|node| (node.id, node.name.as_str()))
        .collect::<std::collections::HashMap<_, _>>();
    let groups = ir
        .physical_groups
        .iter()
        .map(|group| (group.name.as_str(), group.id))
        .collect::<std::collections::HashMap<_, _>>();
    let variable_names = ir
        .optimization
        .variables
        .iter()
        .map(|variable| (variable.id, variable.name.as_str()))
        .collect::<std::collections::HashMap<_, _>>();

    let entry_for = |name: &str| {
        let id = variable_names
            .iter()
            .find_map(|(id, variable_name)| (*variable_name == name).then_some(*id))
            .unwrap();
        ir.sensitivity
            .entries
            .iter()
            .find(|entry| entry.variable == id)
            .unwrap()
    };
    let entity_names = |entry: &meshlang_ast::SensitivityEntry| {
        entry
            .entities
            .iter()
            .map(|id| names[id])
            .collect::<std::collections::BTreeSet<_>>()
    };

    let width = entry_for("width");
    let width_entities = entity_names(width);
    assert!(width_entities.contains("model.base"));
    assert!(width_entities.contains("model.cut"));
    assert!(!width_entities.contains("model.tool"));
    assert!(!width_entities.contains("model.other"));
    assert!(width
        .physical_groups
        .contains(groups.get("cut_group").unwrap()));
    assert!(width
        .physical_groups
        .contains(groups.get("cut_faces").unwrap()));
    assert!(!width
        .physical_groups
        .contains(groups.get("other_group").unwrap()));

    let radius = entry_for("radius");
    let radius_entities = entity_names(radius);
    assert!(radius_entities.contains("model.tool"));
    assert!(radius_entities.contains("model.cut"));
    assert!(radius_entities.contains("model.other"));
    assert!(!radius_entities.contains("model.base"));
    assert!(radius
        .physical_groups
        .contains(groups.get("cut_group").unwrap()));
    assert!(radius
        .physical_groups
        .contains(groups.get("cut_faces").unwrap()));
    assert!(radius
        .physical_groups
        .contains(groups.get("other_group").unwrap()));
}

#[test]
fn sensitivity_includes_mesh_and_material_metadata_dependencies() {
    let source = r#"
        design density {
            type: continuous
            initial: 7800.0
            lower: 1000.0
            upper: 9000.0
        }
        geometry model {
            let left = Box(1.0, 1.0, 1.0)
            let right = Box(2.0, 1.0, 1.0)
        }
        material_region right_material {
            region: model.right
            material: Steel
            density: density
        }
    "#;
    let ast = meshlang_parser::parse(source, "metadata-sensitivity.mlg").unwrap();
    let ir = meshlang_compiler::lower_to_canonical_ir(&ast).unwrap();
    let density_id = ir
        .optimization
        .variables
        .iter()
        .find(|variable| variable.name == "density")
        .unwrap()
        .id;
    let entry = ir
        .sensitivity
        .entries
        .iter()
        .find(|entry| entry.variable == density_id)
        .unwrap();
    let entity_names = entry
        .entities
        .iter()
        .map(|id| {
            ir.entities
                .iter()
                .find(|node| node.id == *id)
                .unwrap()
                .name
                .as_str()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        entity_names,
        std::collections::BTreeSet::from(["model.right"])
    );
    assert!(entry.parameters.contains(&density_id));
}

#[test]
fn typed_physical_boundaries_and_interfaces_emit_in_both_backends() {
    let source = r#"
        geometry model {
            let primary = Rectangle(1.0, 1.0)
            let secondary = Rectangle(1.0, 1.0) |> translate(0.0, 0.0, 1.0)
        }
        physical_region surface domain_boundary {
            entities: [model.primary, model.secondary]
        }
        boundary surface inlet {
            entities: model.primary
            kind: dirichlet
            value: 0.0
        }
        interface surface coupling {
            primary: model.primary
            secondary: model.secondary
            kind: conforming
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo_full(source, "metadata.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python_full(source, "metadata.mlg").unwrap();
    assert!(geo.code.contains("Physical Surface(\"domain_boundary\")"));
    assert!(geo.code.contains("Physical Surface(\"inlet\")"));
    assert!(geo.code.contains("Physical Surface(\"coupling_primary\")"));
    assert!(geo
        .code
        .contains("Physical Surface(\"coupling_secondary\")"));
    assert!(python.code.contains("addPhysicalGroup(2"));
    assert!(python.code.contains("name='domain_boundary'"));
    assert!(python.code.contains("name='coupling_primary'"));

    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    assert_eq!(ir.boundaries.len(), 1);
    assert_eq!(ir.interfaces.len(), 1);
    assert_eq!(ir.physical_groups.len(), 2);
}

#[test]
fn typed_isotropic_mesh_metric_emits_restricted_background_field() {
    let source = r#"
        geometry model {
            let panel = Rectangle(1.0, 1.0)
        }
        mesh for model {
            metric model.panel {
                kind: isotropic
                size: 0.025
            }
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo_full(source, "metric.mlg").unwrap();
    let python = meshlang_compiler::compile_to_python_full(source, "metric.mlg").unwrap();
    assert!(geo.code.contains("= MathEval;"));
    assert!(geo.code.contains("= Restrict;"));
    assert!(geo.code.contains(".SurfacesList"));
    assert!(python.code.contains("field.add('MathEval')"));
    assert!(python.code.contains("field.add('Restrict')"));
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    assert_eq!(ir.mesh.metrics.len(), 1);
}

#[test]
fn canonical_ir_retains_all_mesh_control_variants() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let other = Box(1.0, 1.0, 1.0) |> translate(1.0, 0.0, 0.0)
            let face = Rectangle(0.5, 0.5)
        }
        mesh for model {
            default_size: 0.1
            refine near(model.body, distance: 0.2) {
                size: 0.01
            }
            refine on(model.body.faces) {
                size: 0.02
            }
            refine in(Box(0.2, 0.2, 0.2)) {
                size: 0.03
            }
            refine field(size: 0.04) {
                value: 0.05
            }
            transfinite surface(model.body.faces, nodes: 4)
            embed surface(model.face) in volume(model.body)
            compound volume(model.body, model.other)
            metric model.body {
                kind: hessian
                field: 1.0
                minimum_size: 0.01
                maximum_size: 0.1
            }
        }
    "#;

    let ast = meshlang_parser::parse(source, "mesh-controls.mlg").unwrap();
    let ir = meshlang_compiler::lower_to_canonical_ir(&ast).unwrap();
    assert_eq!(ir.mesh.targets, vec!["model"]);
    assert_eq!(ir.mesh.options.len(), 1);
    assert_eq!(ir.mesh.refinements.len(), 4);
    assert_eq!(ir.mesh.transfinite.len(), 1);
    assert_eq!(ir.mesh.embeds.len(), 1);
    assert_eq!(ir.mesh.compounds.len(), 1);
    assert_eq!(ir.mesh.metrics.len(), 1);
    assert!(matches!(
        ir.mesh.metrics[0].tensor,
        meshlang_ast::IrMetricTensor::Hessian { .. }
    ));
}

#[test]
fn canonical_refine_and_metric_controls_emit_once_in_both_backends() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let face = Rectangle(0.5, 0.5)
        }
        mesh model {
            refine near(model.face, distance: 0.2) {
                size: 0.01
                growth_rate: 2.0
            }
            refine on(model.face) {
                size: 0.02
                order: 2
            }
            refine in(Box(0.2, 0.2, 0.2) |> translate(1.0, 2.0, 3.0)) {
                size: 0.03
                size_out: 0.1
            }
            refine field(distance_from: model.face) {
                size: 0.01
                size_max: 0.1
                distance_min: 0.2
                distance_max: 0.5
            }
            metric model.face {
                kind: isotropic
                size: 0.1
            }
            metric model.body {
                kind: anisotropic
                components: [1, 2, 3, 4, 5, 6]
            }
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo_full(source, "canonical-mesh-controls.mlg")
        .expect("canonical GEO mesh controls");
    let python = meshlang_compiler::compile_to_python_full(source, "canonical-mesh-controls.mlg")
        .expect("canonical Python mesh controls");

    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
    assert_eq!(geo.code.matches("= Distance;").count(), 2);
    assert_eq!(geo.code.matches("= Threshold;").count(), 2);
    assert_eq!(geo.code.matches("= Box;").count(), 1);
    assert_eq!(geo.code.matches("= MathEval;").count(), 1);
    assert_eq!(geo.code.matches("= MathEvalAniso;").count(), 1);
    assert_eq!(geo.code.matches("= Restrict;").count(), 2);
    assert_eq!(geo.code.matches("MeshSize").count(), 1);
    assert_eq!(geo.code.matches("Background Field").count(), 1);
    assert!(geo.code.contains(".SurfacesList"));
    assert!(geo.code.contains(".VolumesList"));

    assert_eq!(python.code.matches("field.add('Distance')").count(), 2);
    assert_eq!(python.code.matches("field.add('Threshold')").count(), 2);
    assert_eq!(python.code.matches("field.add('Box')").count(), 1);
    assert_eq!(python.code.matches("field.add('MathEval')").count(), 1);
    assert_eq!(python.code.matches("field.add('MathEvalAniso')").count(), 1);
    assert_eq!(python.code.matches("field.add('Restrict')").count(), 2);
    assert_eq!(python.code.matches("gmsh.model.mesh.setSize").count(), 1);
    assert_eq!(python.code.matches("Mesh.ElementOrder").count(), 1);
    assert_eq!(python.code.matches("setAsBackgroundMesh").count(), 1);
}

#[test]
fn mixed_dimension_mesh_compound_is_rejected_before_backend_emission() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
            let face = Rectangle(1.0, 1.0)
        }
        mesh for model {
            compound volume(model.body, model.face)
        }
    "#;

    let ast = meshlang_parser::parse(source, "mixed-compound.mlg").unwrap();
    let error = meshlang_compiler::lower_to_canonical_ir(&ast)
        .expect_err("mixed-dimension compound must fail at the IR boundary");
    let message = error.to_string();
    assert!(
        message.contains("MeshCompoundDimensionMismatch"),
        "unexpected diagnostic: {message}"
    );
}

#[test]
fn topology_selector_boundaries_lower_to_dimension_correct_ir() {
    let source = r#"
        geometry model {
            let solid = Box(1.0, 1.0, 1.0)
        }
        boundary surface walls {
            entities: model.solid.faces
            kind: neumann
            flux: 0.0
        }
    "#;
    let geo = meshlang_compiler::compile_to_geo_full(source, "boundary-selector.mlg").unwrap();
    let python =
        meshlang_compiler::compile_to_python_full(source, "boundary-selector.mlg").unwrap();
    assert!(geo.code.contains("Boundary{ Volume{"));
    assert!(python.code.contains("_ml_boundary(_ml_dimtags(3,"));
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    let selected = ir.boundaries[0].entities[0];
    assert_eq!(selected.dimension, meshlang_ast::EntityDimension::Surface);
    assert!(matches!(
        ir.entities
            .iter()
            .find(|node| node.id == selected)
            .unwrap()
            .operation,
        meshlang_ast::GeometryOperation::Boundary {
            target: meshlang_ast::EntityDimension::Surface,
            ..
        }
    ));
}

#[test]
fn canonical_projection_pipes_lower_to_dimension_preserving_operations() {
    let source = r#"
        geometry model {
            let solid = Box(1.0, 1.0, 1.0)
            let iso = solid |> project_iso_surface(tolerance: 0.01)
            let signed = solid |> project_signed_distance(tolerance: 0.02)
            let closest = solid |> project_closest_point(tolerance: 0.03)
        }
    "#;

    let ast = meshlang_parser::parse(source, "projection-pipes.mlg").unwrap();
    let ir = meshlang_compiler::lower_to_canonical_ir(&ast).unwrap();

    let expected = [
        (
            "model.iso",
            meshlang_ast::ProjectionMethod::IsoSurface,
            0.01,
        ),
        (
            "model.signed",
            meshlang_ast::ProjectionMethod::SignedDistance,
            0.02,
        ),
        (
            "model.closest",
            meshlang_ast::ProjectionMethod::ClosestPoint,
            0.03,
        ),
    ];
    for (name, method, tolerance) in expected {
        let node = ir
            .entities
            .iter()
            .find(|node| node.name == name)
            .unwrap_or_else(|| panic!("missing canonical projection node {name}"));
        assert_eq!(node.id.dimension, meshlang_ast::EntityDimension::Volume);
        assert!(matches!(
            &node.operation,
            meshlang_ast::GeometryOperation::Project { method: actual } if *actual == method
        ));
        assert!(matches!(
            node.arguments.as_slice(),
            [meshlang_ast::IrArgument {
                name: meshlang_ast::IrArgumentName::Tolerance,
                value: meshlang_ast::IrValue::Scalar { value: actual, .. },
            }] if (*actual - tolerance).abs() < 1.0e-12
        ));
    }
}

#[test]
fn projection_pipe_type_is_preserved_for_csg_validation() {
    let source = r#"
        geometry model {
            let projected = Rectangle(1.0, 1.0) |> project_iso_surface()
            let invalid = projected | Box(1.0, 1.0, 1.0)
        }
    "#;
    let ast = meshlang_parser::parse(source, "projection-types.mlg").unwrap();
    let mut checker = meshlang_compiler::TypeChecker::new();
    let errors = checker
        .check(&ast)
        .into_iter()
        .filter(|diagnostic| diagnostic.level == meshlang_compiler::DiagLevel::Error)
        .collect::<Vec<_>>();

    assert!(
        errors.iter().any(|diagnostic| diagnostic
            .message
            .contains("CSG operands must have the same dimension")),
        "projection should preserve the surface type for CSG validation: {errors:?}"
    );
}

#[test]
fn test_python_backend_fails_on_type_errors_before_codegen() {
    let src = r#"
        geometry bad {
            let p = Polygon(points: [[0, 0], [1, 0], [0, 1]])
            let b = Box(1.0, 1.0, 1.0)
            let m = p - b
        }
    "#;

    let err = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("type error")
            && msg.contains("CSG operands must have the same dimension")
            && msg.contains("surface")
            && msg.contains("solid"),
        "expected same-dimension CSG type error, got: {msg}"
    );
}

#[test]
fn test_geo_full_fails_on_type_errors_before_codegen() {
    let src = r#"
        geometry bad {
            let result = 1 | 2
        }
    "#;

    let err = match meshlang_compiler::compile_to_geo_full(src, "test.mlg") {
        Ok(_) => panic!("expected compile_to_geo_full to reject CSG type errors"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("type error") && msg.contains("operand of CSG operation"),
        "expected CSG type error, got: {msg}"
    );
    assert!(
        !msg.contains("internal compiler error"),
        "type errors must not be hidden by backend errors: {msg}"
    );
}

#[test]
fn test_constructor_arity_error_does_not_panic() {
    let src = r#"
        geometry bad {
            let b = Box(1.0, 2.0)
        }
    "#;

    let err = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("type error") && msg.contains("Box") && msg.contains("3"),
        "expected Box arity type error, got: {msg}"
    );
}

#[test]
fn test_forward_geometry_let_references_are_rejected() {
    let src = r#"
        geometry bad {
            let model = box1 | box2
            let box1 = Box(1.0, 1.0, 1.0)
            let box2 = Box(1.0, 1.0, 1.0)
        }
    "#;

    let err = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("before declaration") && msg.contains("box1"),
        "expected forward reference error, got: {msg}"
    );
}

#[test]
fn test_geo_backend_rejects_face_set_difference() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }

        physics {
            group walls: neumann = test.b.faces - test.b.faces.top
        }
    "#;

    let err = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("face set operations require Python backend"),
        "expected .geo topology set-operation error, got: {msg}"
    );
}

#[test]
fn test_python_backend_emits_face_set_difference() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }

        physics {
            group walls: neumann = test.b.faces - test.b.faces.top
        }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("mlg_face_diff("),
        "face-set difference should emit the Python topology helper"
    );
    assert!(
        py.contains("gmsh.model.addPhysicalGroup"),
        "face-set difference should still create a physical group"
    );
    assert!(
        !py.contains("# group 'walls'"),
        "physics group errors must not be silently emitted as comments"
    );
}

#[test]
fn test_for_loop_uses_local_integer_alias() {
    let src = r#"
        geometry test {
            let count = 1 + 1
            for i in 0..count {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4 filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "local-loop-bound.mlg").unwrap();
    assert_eq!(
        geo.matches("Box(").count(),
        2,
        "local integer binding should drive compile-time loop expansion"
    );
    meshlang_compiler::compile_to_python(src, "local-loop-bound.mlg").unwrap();
}

#[test]
fn test_for_loop_parameter_bound_emits_runtime_loop() {
    let src = r#"
        param count = 2
        geometry test {
            for i in 0..count {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4 filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "dynamic-loop-bound.mlg").unwrap();
    assert!(
        geo.contains("For i In {0:(count) - 1}"),
        "GEO runtime loop header"
    );
    assert!(geo.contains("EndFor"), "GEO runtime loop terminator");
    let python = meshlang_compiler::compile_to_python(src, "dynamic-loop-bound.mlg").unwrap();
    assert!(
        python.contains("for i in range(0, count):"),
        "Python runtime loop header"
    );
}

#[test]
fn test_runtime_loop_aggregates_nested_geometry_and_preserves_binding() {
    let src = r#"
param count = 2
param width = 2
geometry model {
  for i in 0..count {
    let body = Box(1.0, 1.0, 1.0)
    for j in 0..width {
      let nested = Box(0.5, 0.5, 0.5)
    }
  }
}
physics {
  group all: volume = model.body
}
export { format: msh4 filename: "runtime-loop-aggregate.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "runtime-loop-aggregate.mlg").unwrap();
    let runtime_geo: Vec<_> = geo
        .lines()
        .filter(|line| line.contains("For ") || line.contains("loop_") || line.contains("EndFor"))
        .collect();
    assert!(
        geo.contains("__ml_loop_1_0[] = {}"),
        "GEO runtime lines: {runtime_geo:?}"
    );
    assert!(geo.contains("__ml_loop_2_0[] = {}"));
    assert!(geo.contains("__ml_loop_2_0[] += {model__nested}"));
    assert!(geo.contains("__ml_loop_1_0[] += __ml_loop_2_0[]"));
    assert!(
        geo.contains("__ml_loop_1_0[]"),
        "GEO must retain the complete runtime body aggregate"
    );
    assert!(
        geo.contains("Physical Volume(\"all\")") && geo.contains("__ml_loop_1_0[]"),
        "GEO physical groups must consume the aggregate"
    );

    let python = meshlang_compiler::compile_to_python(src, "runtime-loop-aggregate.mlg").unwrap();
    assert!(python.contains("__ml_loop_1_0 = []"));
    assert!(python.contains("__ml_loop_2_0 = []"));
    let runtime_python: Vec<_> = python
        .lines()
        .filter(|line| {
            line.contains("for ") || line.contains("__ml_loop") || line.contains("pg_all")
        })
        .collect();
    assert!(
        python.contains("__ml_loop_2_0.extend(_ml_dimtags(3, nested))"),
        "Python runtime lines: {runtime_python:?}"
    );
    assert!(python.contains("__ml_loop_1_0.extend(__ml_loop_2_0)"));
    assert!(
        python.contains("pg_all") && python.contains("__ml_loop_1_0"),
        "Python must remap a post-loop physical group to the aggregate: {runtime_python:?}"
    );
}

#[test]
fn test_runtime_loop_aggregates_only_the_selected_dynamic_branch() {
    let src = r#"
param count = 2
param use_a = true
geometry model {
  for i in 0..count {
    if use_a {
      let a = Box(1.0, 1.0, 1.0)
    } else {
      let b = Box(0.5, 0.5, 0.5)
    }
  }
}
export { format: msh4 filename: "runtime-loop-branch.msh" }
"#;

    let geo = meshlang_compiler::compile_to_geo(src, "runtime-loop-branch.mlg").unwrap();
    let geo_if = geo.find("If (use_a)").expect("GEO runtime if");
    let geo_else = geo[geo_if..]
        .find("Else")
        .map(|offset| geo_if + offset)
        .unwrap();
    let geo_runtime: Vec<_> = geo
        .lines()
        .filter(|line| line.contains("If ") || line.contains("Else") || line.contains("loop_"))
        .collect();
    assert!(
        geo[geo_if..geo_else].contains("__ml_loop_1_0[] += {model____if_0_0__a}"),
        "GEO runtime lines: {geo_runtime:?}"
    );
    assert!(geo[geo_else..].contains("__ml_loop_1_0[] += {model____if_0_1__b}"));

    let python = meshlang_compiler::compile_to_python(src, "runtime-loop-branch.mlg").unwrap();
    let python_if = python.find("if use_a:").expect("Python runtime if");
    let python_else = python[python_if..]
        .find("else:")
        .map(|offset| python_if + offset)
        .unwrap();
    assert!(
        python[python_if..python_else]
            .contains("__ml_loop_1_0.extend(_ml_dimtags(3, __if_0_0__a))"),
        "Python runtime lines: {:?}",
        python
            .lines()
            .filter(|line| line.contains("if ")
                || line.contains("else:")
                || line.contains("__ml_loop"))
            .collect::<Vec<_>>()
    );
    assert!(
        python[python_else..].contains("__ml_loop_1_0.extend(_ml_dimtags(3, __if_0_1__b))"),
        "Python runtime lines: {:?}",
        python
            .lines()
            .filter(|line| line.contains("if ")
                || line.contains("else:")
                || line.contains("__ml_loop"))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_dynamic_loop_is_preserved_in_canonical_ir() {
    let src = r#"
        param count = 2
        geometry test {
            for i in 0..count, j in 1..=count {
                let b = Box(i + j, 1.0, 1.0)
            }
        }
    "#;
    let ast = parse(src, "dynamic-loop-ir.mlg").expect("dynamic loop parses");
    let ir = meshlang_compiler::lower_to_canonical_ir(&ast).expect("dynamic loop lowers");
    assert_eq!(ir.loops.len(), 1);
    assert_eq!(ir.loops[0].clauses.len(), 2);
    assert_eq!(ir.loops[0].clauses[0].name, "i");
    assert_eq!(ir.loops[0].clauses[1].name, "j");
    assert!(!ir.loops[0].entities.is_empty());
    ir.validate().expect("dynamic loop IR validates");
}

#[test]
fn test_nested_dynamic_loops_preserve_parent_order_and_backend_nesting() {
    let src = r#"
        param outer = 2
        param inner = 3
        geometry test {
            for i in 0..outer {
                for j in 0..inner {
                    let inner_body = Box(i + j + 1, 1.0, 1.0)
                }
            }
        }
    "#;
    let ast = parse(src, "nested-dynamic-loop.mlg").expect("nested loop parses");
    let ir = meshlang_compiler::lower_to_canonical_ir(&ast).expect("nested loop lowers");
    assert_eq!(ir.loops.len(), 2);
    assert_eq!(ir.loops[1].parent, Some(0));
    ir.validate().expect("nested loop IR validates");
    let geo = meshlang_compiler::compile_to_geo(src, "nested-dynamic-loop.mlg").unwrap();
    assert!(geo.matches("For ").count() >= 2);
    let python = meshlang_compiler::compile_to_python(src, "nested-dynamic-loop.mlg").unwrap();
    assert!(python.matches("for ").count() >= 2);
}

#[test]
fn test_for_loop_empty_range_no_output() {
    // for i in 0..0 should produce zero iterations, not one
    let src = r#"
        geometry test {
            for i in 0..0 {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        !geo.contains("Box("),
        "empty range 0..0 should produce no Box"
    );
}

#[test]
fn test_for_loop_single_iteration() {
    let src = r#"
        geometry test {
            for i in 0..1 {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    let box_count = geo.matches("Box(").count();
    assert_eq!(
        box_count, 1,
        "0..1 should produce exactly 1 Box, got {}",
        box_count
    );
}

#[test]
fn test_for_loop_multi_iteration() {
    let src = r#"
        geometry test {
            for i in 0..3 {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    let box_count = geo.matches("Box(").count();
    assert_eq!(
        box_count, 3,
        "0..3 should produce 3 Boxes, got {}",
        box_count
    );
    // Should have BooleanUnion to combine them
    assert!(
        geo.contains("BooleanUnion"),
        "multiple iterations should be unioned"
    );
    assert!(!geo.contains("{-1}"), "sentinel tag leaked into GEO output");
}

#[test]
fn test_mirror_transform_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0) |> mirror(1.0, 0.0, 0.0)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    // Gmsh Symmetry takes 4 args: {a, b, c, d} for plane ax+by+cz+d=0
    assert!(
        geo.contains("Symmetry {1.0, 0.0, 0.0, 0}"),
        "mirror should emit Symmetry with 4 args"
    );
}

#[test]
fn test_mirror_transform_python() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0) |> mirror(1.0, 0.0, 0.0)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    // occ.mirror(dimTags, a, b, c, d) — 4 plane coefficients after dimTags
    assert!(
        py.contains("occ.mirror(_ml_dimtags(3, b), 1.0, 0.0, 0.0, 0)"),
        "mirror should emit 4 plane coefficients"
    );
}

#[test]
fn test_rotation_angles_are_emitted_as_radians() {
    let src = r#"
        geometry test {
            let box_geo = Box(1.0, 2.0, 3.0)
                |> rotate(axis: Z, angle: 90)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("(90 * Pi / 180.0)"),
        ".geo backend should convert MeshLang degree angles to Gmsh radians:\n{}",
        geo
    );

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("(90 * math.pi / 180.0)"),
        "Python backend should convert MeshLang degree angles to Gmsh radians:\n{}",
        py
    );
}

#[test]
fn test_python_extrude_reassigns_result_tag_for_followup_transforms() {
    let src = r#"
        geometry test {
            let panel = Rectangle(1.0, 1.0)
                |> extrude(0.0, 0.0, 0.25)
                |> fillet(r: 0.05)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains(" = occ.extrude(_ml_dimtags(2, panel), 0.0, 0.0, 0.25)"),
        "extrude should capture returned dim-tags:\n{}",
        py
    );
    assert!(
        py.contains("panel = _ml_only(3, extrude_"),
        "extrude should reassign panel to every returned volume before fillet:\n{}",
        py
    );
    assert!(
        py.contains("occ.fillet(_ml_dimtags(3, panel)"),
        "fillet should consume every returned volume tag:\n{}",
        py
    );
}

#[test]
fn test_python_compile_time_elif_chain_folds_from_canonical_values() {
    let src = r#"
        geometry test {
            let use_a = false
            let use_b = false
            let use_c = true

            if use_a {
                let a = Box(1.0, 1.0, 1.0)
            } else if use_b {
                let b = Box(2.0, 1.0, 1.0)
            } else if use_c {
                let c = Box(3.0, 1.0, 1.0)
            } else {
                let d = Box(4.0, 1.0, 1.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        !py.contains("if use_a:") && !py.contains("elif use_b:") && !py.contains("elif use_c:"),
        "canonical compile-time branch selection must not re-introduce source control flow:\n{}",
        py
    );
    assert!(
        py.contains("c = occ.addBox(0, 0, 0, 3.0, 1.0, 1.0)"),
        "selected canonical branch should remain in the Python artifact:\n{}",
        py
    );
    assert!(
        !py.contains("occ.addBox(0, 0, 0, 1.0, 1.0, 1.0")
            && !py.contains("occ.addBox(0, 0, 0, 2.0, 1.0, 1.0")
            && !py.contains("occ.addBox(0, 0, 0, 4.0, 1.0, 1.0"),
        "discarded branches must not be emitted:\n{}",
        py
    );

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        !geo.contains("If(") && geo.contains("Box("),
        "GEO should receive the same canonical branch selection:\n{}",
        geo
    );
}

#[test]
fn test_fn_call_inlining_geo() {
    let src = r#"
        fn make_port(radius: Float, depth: Float) -> Solid {
            Cylinder(r: radius, h: depth)
        }

        geometry cabinet {
            let port = make_port(0.025, 0.15)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("Cylinder("),
        "inlined function should produce Cylinder"
    );
}

#[test]
fn test_fn_call_inlining_python() {
    let src = r#"
        fn make_port(radius: Float, depth: Float) -> Solid {
            Cylinder(r: radius, h: depth)
        }

        geometry cabinet {
            let port = make_port(0.025, 0.15)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("occ.addCylinder("),
        "inlined function should produce addCylinder"
    );
}

#[test]
fn test_multi_statement_function_call_inlines_locals_in_geo_and_python() {
    let src = r#"
        fn cutout(base: Solid, r: Float) -> Solid {
            let ball = Sphere(r: r)
            base - ball
        }

        geometry test {
            let box_geo = Box(1.0, 1.0, 1.0)
            let result = cutout(box_geo, 0.25)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo_full(src, "test.mlg")
        .unwrap_or_else(|error| panic!("GEO multi-statement inline error: {error}"));
    assert!(
        geo.code.contains("BooleanDifference"),
        "GEO should inline the local cutout result"
    );

    let py = meshlang_compiler::compile_to_python_full(src, "test.mlg").unwrap();
    assert!(
        py.code.contains("occ.cut("),
        "Python should inline the local cutout result"
    );
    assert_eq!(geo.canonical_ir_json, py.canonical_ir_json);
    let ir = serde_json::from_str::<meshlang_ast::VersionedCanonicalIr>(&geo.canonical_ir_json)
        .unwrap()
        .ir;
    let result = ir
        .entities
        .iter()
        .find(|entity| entity.name == "test.result")
        .expect("multi-statement function result is canonical IR authority");
    assert!(matches!(
        result.operation,
        meshlang_ast::GeometryOperation::Boolean {
            kind: meshlang_ast::BooleanOperation::Difference
        }
    ));
    assert_eq!(result.inputs.len(), 2);
}

#[test]
fn test_lowercase_call_parsed() {
    let src = r#"
        fn make_port(radius: Float, depth: Float) -> Solid {
            Cylinder(r: radius, h: depth)
        }

        geometry test {
            let p = make_port(0.05, 0.2)
        }
    "#;

    let ast = meshlang_parser::parse(src, "test.mlg").unwrap();
    match &ast.declarations[1] {
        Declaration::Geometry(g) => match &g.body.statements[0] {
            Statement::Let(l) => {
                assert!(
                    matches!(&l.value, Expr::Call(_)),
                    "lowercase fn call should parse as Call"
                );
            }
            _ => panic!("expected let"),
        },
        _ => panic!("expected geometry"),
    }
}

#[test]
fn test_if_true_emits_then_branch() {
    let src = r#"
        geometry test {
            if true {
                let b = Box(1.0, 2.0, 3.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Box("), "if true should emit the then branch");
}

#[test]
fn test_if_false_skips_then_branch() {
    let src = r#"
        geometry test {
            if false {
                let b = Box(1.0, 2.0, 3.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        !geo.contains("Box("),
        "if false should skip the then branch"
    );
}

#[test]
fn test_frequency_aware_sizing_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            target_frequency: 1000
            medium_speed: 343
            elements_per_wavelength: 6
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("Mesh.CharacteristicLengthMax = 343 / 1000 / 6"),
        "should emit frequency-derived sizing"
    );
}

#[test]
fn test_refine_on_emits_meshsize_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            refine on(test.b.faces) {
                size: 0.01
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("MeshSize{"),
        "refine on should emit MeshSize command"
    );
}

// ── Feature: sweep transform ────────────────────────────────────────────────

#[test]
fn test_sweep_transform_geo() {
    let src = r#"
        geometry test {
            let tube = Cylinder(r: 0.05, h: 0.1)
            let ring = tube |> sweep(path: tube)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Wire("), "sweep should emit Wire");
    assert!(
        geo.contains("Using Wire"),
        "sweep should emit Extrude Using Wire"
    );
    assert!(
        geo.contains("= __sweep_2[0]") || geo.contains("= __sweep_1[0]"),
        "sweep result should use the first Gmsh result"
    );
}

#[test]
fn test_sweep_transform_python() {
    let src = r#"
        geometry test {
            let tube = Cylinder(r: 0.05, h: 0.1)
            let ring = tube |> sweep(path: tube)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(py.contains("addWire("), "sweep should emit addWire");
    assert!(py.contains("addPipe("), "sweep should emit addPipe");
}

// ── Feature: struct instantiation ───────────────────────────────────────────

#[test]
fn test_parse_struct_init() {
    let src = r#"
        struct Material {
            density: Float,
            speed_of_sound: Float,
        }

        geometry test {
            let mat = Material { density: 1.225, speed_of_sound: 343.0 }
            let b = Box(1.0, 1.0, 1.0)
        }
    "#;
    let ast = parse(src, "test.mlg").unwrap();

    // Struct decl
    match &ast.declarations[0] {
        Declaration::Struct(s) => {
            assert_eq!(s.name.name, "Material");
            assert_eq!(s.fields.len(), 2);
        }
        _ => panic!("expected struct"),
    }

    // Struct init
    match &ast.declarations[1] {
        Declaration::Geometry(g) => match &g.body.statements[0] {
            Statement::Let(l) => {
                assert!(
                    matches!(&l.value, Expr::StructInit(_)),
                    "expected StructInit"
                );
                if let Expr::StructInit(si) = &l.value {
                    assert_eq!(si.type_name.name, "Material");
                    assert_eq!(si.fields.len(), 2);
                    assert_eq!(si.fields[0].name.name, "density");
                }
            }
            _ => panic!("expected let"),
        },
        _ => panic!("expected geometry"),
    }
}

#[test]
fn test_struct_init_geo() {
    let src = r#"
        struct Material {
            density: Float,
            speed_of_sound: Float,
        }

        geometry test {
            let mat = Material { density: 1.225, speed_of_sound: 343.0 }
            let b = Box(1.0, 1.0, 1.0)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("mat_density = 1.225"),
        "struct init should emit field variables"
    );
    assert!(
        geo.contains("mat_speed_of_sound = 343"),
        "struct init should emit all fields"
    );
}

#[test]
fn test_struct_init_python() {
    let src = r#"
        struct Material {
            density: Float,
            speed_of_sound: Float,
        }

        geometry test {
            let mat = Material { density: 1.225, speed_of_sound: 343.0 }
            let b = Box(1.0, 1.0, 1.0)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("mat = {"),
        "struct init should emit dict in Python"
    );
    assert!(
        py.contains("'density': 1.225"),
        "dict should have density field"
    );
    assert!(
        py.contains("'speed_of_sound': 343"),
        "dict should have speed_of_sound field"
    );
}

#[test]
fn test_struct_type_check() {
    let src = r#"
        struct Material {
            density: Float,
            speed_of_sound: Float,
        }

        geometry test {
            let mat = Material { density: 1.225, speed_of_sound: 343.0 }
            let b = Box(1.0, 1.0, 1.0)
        }
    "#;

    let ast = meshlang_parser::parse(src, "test.mlg").unwrap();
    let mut checker = meshlang_compiler::TypeChecker::new();
    let diags = checker.check(&ast);

    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.level == meshlang_compiler::DiagLevel::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "struct init should not produce type errors: {:?}",
        errors
    );
}

// ── Feature: per-surface mesh order ─────────────────────────────────────────

#[test]
fn test_per_surface_order_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            refine on(test.b.faces) {
                size: 0.01
                order: 2
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("MeshSize{"), "refine on should emit MeshSize");
    assert!(
        geo.contains("Mesh.ElementOrder = 2"),
        "refine on should emit per-surface ElementOrder"
    );
}

#[test]
fn test_per_surface_order_python() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            refine on(test.b.faces) {
                size: 0.01
                order: 2
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("Mesh.ElementOrder', 2"),
        "refine on should emit per-surface ElementOrder"
    );
}

// ── Bug fix: Background Field composition ───────────────────────────────────

#[test]
fn test_refine_near_emits_background_field_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            refine near(test.b, distance: 0.05) {
                size: 0.01
                growth_rate: 1.2
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("Field[") && geo.contains("= Distance;"),
        "should emit Distance field"
    );
    assert!(geo.contains("= Threshold;"), "should emit Threshold field");
    assert!(
        geo.contains("SizeMax"),
        "should emit SizeMax for growth_rate"
    );
    assert!(geo.contains("= Min;"), "should compose fields with Min");
    assert!(
        geo.contains("Background Field ="),
        "should set Background Field"
    );
}

#[test]
fn test_refine_near_emits_background_field_python() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            refine near(test.b, distance: 0.05) {
                size: 0.01
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let py = meshlang_compiler::compile_to_python(src, "test.mlg").unwrap();
    assert!(
        py.contains("field.add('Distance')"),
        "should emit Distance field"
    );
    assert!(
        py.contains("field.add('Threshold')"),
        "should emit Threshold field"
    );
    assert!(py.contains("SizeMax"), "should emit SizeMax default");
    assert!(
        py.contains("field.add('Min')"),
        "should compose fields with Min"
    );
    assert!(
        py.contains("setAsBackgroundMesh("),
        "should set as background mesh"
    );
}

#[test]
fn test_multiple_refine_fields_composed_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
            let s = Sphere(r: 0.3)
        }
        mesh for test {
            default_size: 0.1
            refine near(test.b, distance: 0.05) {
                size: 0.01
            }
            refine near(test.s, distance: 0.02) {
                size: 0.005
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    // Min field should list both threshold field IDs
    assert!(geo.contains("= Min;"), "should have Min field");
    assert!(
        geo.contains("FieldsList"),
        "Min field should have FieldsList"
    );
    // FieldsList should contain a comma (two entries)
    let fields_line = geo.lines().find(|l| l.contains("FieldsList")).unwrap();
    assert!(
        fields_line.contains(','),
        "FieldsList should combine multiple fields: {}",
        fields_line
    );
}

#[test]
fn test_boundary_layer_entity_attachment_geo() {
    let src = r#"
        geometry test {
            let b = Box(1.0, 1.0, 1.0)
        }
        mesh for test {
            default_size: 0.1
            boundary_layer on(test.b.faces) {
                first_height: 0.001
                layers: 5
                growth_rate: 1.2
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("= BoundaryLayer;"),
        "should emit BoundaryLayer field"
    );
    assert!(
        geo.contains("SurfacesList"),
        "BoundaryLayer should have SurfacesList for entity attachment"
    );
    assert!(geo.contains("Size ="), "should set BL Size");
    assert!(geo.contains("NbLayers ="), "should set BL NbLayers");
    assert!(geo.contains("Ratio ="), "should set BL Ratio");
}

// ── Gmsh tutorial translation tests ─────────────────────────────────────────

/// Helper: try to load an example .mlg from the examples/ dir (relative to
/// the sonium-meshlang workspace root, which is two levels up from the
/// compiler crate).
fn load_example(name: &str) -> String {
    // CARGO_MANIFEST_DIR points to the meshlang-compiler crate.
    // Examples live two levels up in sonium-meshlang/examples/.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join("../../examples")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

#[test]
fn test_e2e_gmsh_t5_cheese() {
    let src = load_example("gmsh_t5_cheese.mlg");
    let geo = meshlang_compiler::compile_to_geo(&src, "gmsh_t5_cheese.mlg").unwrap();

    assert!(geo.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(geo.contains("Box("), "should have the main cube");
    assert!(geo.contains("Sphere("), "should have spherical holes");
    assert!(
        geo.contains("BooleanDifference"),
        "holes are punched via CSG difference"
    );
    assert!(
        geo.contains("Field["),
        "should have refinement fields near holes"
    );
    assert!(geo.contains("Physical Volume(\"block_body\")"));
    assert!(geo.contains("Output: t5_cheese.msh"));
}

#[test]
fn test_e2e_gmsh_t16_csg() {
    let src = load_example("gmsh_t16_csg.mlg");
    let geo = meshlang_compiler::compile_to_geo(&src, "gmsh_t16_csg.mlg").unwrap();

    assert!(geo.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(geo.contains("Box("), "should have the two cubes");
    assert!(geo.contains("Sphere("), "should have five spheres");
    assert!(
        geo.contains("BooleanDifference"),
        "corner is cut via difference"
    );
    assert!(
        geo.contains("BooleanFragments"),
        "conformal decomposition via fragment"
    );
    // Fragment tool list should contain all 5 sphere tags (comma-separated)
    let frag_line = geo
        .lines()
        .find(|l| l.contains("BooleanFragments"))
        .unwrap();
    let tool_block = frag_line.split("Delete; }{").nth(1).unwrap();
    let comma_count = tool_block.matches(',').count();
    assert_eq!(
        comma_count, 4,
        "fragment should list 5 volumes (4 commas): {}",
        tool_block
    );
    assert!(geo.contains("Output: t16_csg.msh"));
}

#[test]
fn test_e2e_gmsh_t19_loft() {
    let src = load_example("gmsh_t19_loft_pipe.mlg");
    let geo = meshlang_compiler::compile_to_geo(&src, "gmsh_t19_loft_pipe.mlg").unwrap();

    assert!(geo.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(geo.contains("Disk("), "Circle surfaces become Disk in OCC");
    assert!(
        geo.contains("ThruSections("),
        "Loft compiles to ThruSections"
    );
    assert!(
        geo.contains("Ruled ThruSections("),
        "Loft(..., ruled: true) compiles to `Ruled ThruSections`"
    );
    assert!(geo.contains("Fillet{"), "vase3 should have fillet applied");
    // Helix pipe: 20 points with trig, Spline, sweep
    assert!(geo.contains("Cos("), "helix points should use Cos()");
    assert!(geo.contains("Sin("), "helix points should use Sin()");
    assert!(geo.contains("Pi"), "should reference Pi constant");
    assert!(geo.contains("Spline("), "helix path should be a Spline");
    assert!(
        geo.contains("Using Wire"),
        "sweep should emit Extrude Using Wire"
    );
    assert!(geo.contains("Output: t19_loft.msh"));
}

#[test]
fn test_loft_ruled_emits_ruled_thrusections() {
    let src = r#"
        geometry ruled_demo {
            let s1 = Circle(r: 0.5)
            let s2 = Circle(r: 0.3) |> translate(0.0, 0.0, 1.0)
            let s3 = Circle(r: 0.1) |> translate(0.0, 0.0, 2.0)
            let body = Loft(sections: [s1, s2, s3], ruled: true)
        }
        export {
            format: msh4
            filename: "ruled.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "ruled.mlg").unwrap();
    assert!(
        geo.contains("Ruled ThruSections("),
        ".geo backend should emit `Ruled ThruSections` for ruled: true:\n{}",
        geo
    );

    let py = meshlang_compiler::compile_to_python(src, "ruled.mlg").unwrap();
    assert!(
        py.contains("makeRuled=True"),
        "python backend should pass makeRuled=True to occ.addThruSections:\n{}",
        py
    );
}

#[test]
fn test_loft_default_is_smooth() {
    let src = r#"
        geometry smooth_demo {
            let s1 = Circle(r: 0.5)
            let s2 = Circle(r: 0.3) |> translate(0.0, 0.0, 1.0)
            let body = Loft(sections: [s1, s2])
        }
        export {
            format: msh4
            filename: "smooth.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "smooth.mlg").unwrap();
    assert!(
        !geo.contains("Ruled ThruSections("),
        "default Loft must remain smooth (no `Ruled` prefix):\n{}",
        geo
    );
    assert!(geo.contains("ThruSections("));

    let py = meshlang_compiler::compile_to_python(src, "smooth.mlg").unwrap();
    assert!(
        !py.contains("makeRuled=True"),
        "default python Loft must not pass makeRuled=True:\n{}",
        py
    );
}

#[test]
fn test_loft_polygon_section_uses_curve_loop_tag() {
    // Regression: OCC `ThruSections` requires WIRE/curve-loop tags. Polygon
    // is emitted as `Curve Loop({name}__cl)` + `Plane Surface({name})`. If
    // Loft passes the surface tag, gmsh fails with
    // "Unknown OpenCASCADE wire or curve loop with tag N". The compiler must
    // route Polygon-derived sections to their `__cl` curve-loop tag.
    let src = r#"
        geometry poly_loft {
            let s_root = Polygon(points: [
                [0.0, -0.001, -0.01],
                [0.0,  0.001, -0.01],
                [0.0,  0.001,  0.01],
                [0.0, -0.001,  0.01]
            ])
            let s_tip = Polygon(points: [
                [0.05, -0.001, -0.005],
                [0.05,  0.001, -0.005],
                [0.05,  0.001,  0.005],
                [0.05, -0.001,  0.005]
            ])
            let blade = Loft(sections: [s_root, s_tip], ruled: true)
        }
        export {
            format: msh4
            filename: "poly_loft.msh"
        }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "poly_loft.mlg").unwrap();
    let thru_line = geo
        .lines()
        .find(|l| l.contains("Ruled ThruSections("))
        .unwrap_or_else(|| panic!("missing Ruled ThruSections line:\n{}", geo));
    assert!(
        thru_line.contains("__cl"),
        "Polygon-derived sections must be passed by curve-loop tag (`__cl`):\n{}",
        thru_line
    );

    let py = meshlang_compiler::compile_to_python(src, "poly_loft.mlg").unwrap();
    let thru_py = py
        .lines()
        .find(|l| l.contains("addThruSections"))
        .unwrap_or_else(|| panic!("missing addThruSections call:\n{}", py));
    assert!(
        thru_py.contains("cl_") || thru_py.contains("cl"),
        "Polygon sections in python backend must use curve-loop var:\n{}",
        thru_py
    );
}

#[test]
fn test_loft_ruled_unknown_arg_rejected() {
    let src = r#"
        geometry bad_demo {
            let s1 = Circle(r: 0.5)
            let s2 = Circle(r: 0.3) |> translate(0.0, 0.0, 1.0)
            let body = Loft(sections: [s1, s2], smoothing: true)
        }
        export {
            format: msh4
            filename: "bad.msh"
        }
    "#;

    let result = meshlang_compiler::compile_to_geo(src, "bad.mlg");
    assert!(
        result.is_err(),
        "Loft must reject unknown named arguments: {:?}",
        result
            .as_ref()
            .map(|s| s.lines().take(3).collect::<Vec<_>>())
    );
}

#[test]
fn test_e2e_gmsh_helmholtz_resonator() {
    let src = load_example("gmsh_helmholtz_resonator.mlg");
    let geo = meshlang_compiler::compile_to_geo(&src, "gmsh_helmholtz_resonator.mlg").unwrap();

    assert!(geo.contains("SetFactory(\"OpenCASCADE\")"));
    assert!(geo.contains("Box("), "cavity is a box");
    assert!(geo.contains("Cylinder("), "neck is a cylinder");
    assert!(geo.contains("BooleanUnion"), "cavity+neck joined via union");
    assert!(
        geo.contains("BooleanDifference"),
        "neck hole punched via difference"
    );
    assert!(geo.contains("Mesh.ElementOrder = 2"), "quadratic elements");
    assert!(geo.contains("Physical Surface(\"walls\")"));
    assert!(geo.contains("Output: helmholtz_resonator.msh"));
}

#[test]
fn test_fillet_geo_backend() {
    let src = r#"
        geometry test {
            let cube = Box(1.0, 1.0, 1.0)
                |> fillet(r: 0.1)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Box("), "should have the cube");
    assert!(
        geo.contains("Boundary{ Volume{"),
        "should get boundary surfaces"
    );
    assert!(
        geo.contains("Unique(Abs(Boundary{ Surface{"),
        "should get unique edges from surfaces"
    );
    assert!(geo.contains("Fillet{"), "should emit Fillet command");
    assert!(geo.contains("{0.1}"), "fillet radius should be 0.1");
}

#[test]
fn test_chamfer_geo_backend() {
    let src = r#"
        geometry test {
            let cube = Box(1.0, 1.0, 1.0)
                |> chamfer(d: 0.05)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(geo.contains("Chamfer{"), "should emit Chamfer command");
    assert!(geo.contains("{0.05}"), "chamfer distance should be 0.05");
}

#[test]
fn test_for_loop_constant_integer_overflow_is_rejected() {
    let src = r#"
        geometry test {
            for i in 0..(999999999999 * 999999999999) {
                let b = Box(1.0, 1.0, 1.0)
            }
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let result = std::panic::catch_unwind(|| meshlang_compiler::compile_to_geo(src, "test.mlg"));
    match result {
        Ok(Err(err)) => assert!(
            err.to_string().contains("overflow"),
            "overflow should be reported clearly, got: {err}"
        ),
        Ok(Ok(code)) => panic!("overflowing range should not compile:\n{code}"),
        Err(_) => panic!("overflowing range should return an error, not panic"),
    }
}

#[test]
fn test_shell_geo_backend() {
    let src = r#"
        geometry test {
            let cube = Box(1.0, 1.0, 1.0)
                |> shell(thickness: 0.02)
        }
        export { format: msh4  filename: "test.msh" }
    "#;

    let geo = meshlang_compiler::compile_to_geo(src, "test.mlg").unwrap();
    assert!(
        geo.contains("ThickSolid("),
        "should emit ThickSolid for shell"
    );
    assert!(geo.contains("0.02"), "shell thickness should be 0.02");
}

#[test]
fn native_backends_reject_hessian_metrics_with_typed_errors() {
    let source = r#"
        geometry model {
            let body = Box(1.0, 1.0, 1.0)
        }
        mesh for model {
            metric model.body {
                kind: hessian
                field: 1.0
                minimum_size: 0.01
                maximum_size: 0.1
            }
        }
    "#;

    for (backend, result) in [
        (
            "GEO",
            meshlang_compiler::compile_to_geo_full(source, "hessian.mlg"),
        ),
        (
            "Python",
            meshlang_compiler::compile_to_python_full(source, "hessian.mlg"),
        ),
    ] {
        let error = match result {
            Ok(_) => panic!("{backend} silently compiled a Hessian metric"),
            Err(error) => error,
        };
        assert!(
            matches!(error, meshlang_compiler::CompileError::Unsupported(ref message) if message.contains("Hessian")),
            "{backend} returned the wrong error for Hessian metrics: {error}"
        );
    }
}

#[derive(Debug)]
struct EchoHessianMetricProvider;

impl meshlang_compiler::HessianMetricProvider for EchoHessianMetricProvider {
    fn resolve(
        &self,
        request: meshlang_compiler::HessianMetricRequest<'_>,
    ) -> Result<[meshlang_ast::Expr; 6], String> {
        Ok(std::array::from_fn(|_| request.field.clone()))
    }
}

#[test]
fn configured_hessian_provider_is_shared_by_geo_and_python_backends() {
    let source = r#"
geometry model {
    let body = Box(1.0, 1.0, 1.0)
}
mesh for model {
    metric model.body {
        kind: hessian
        field: 1.0
        minimum_size: 0.01
        maximum_size: 0.1
    }
}
"#;
    let options = meshlang_compiler::CompileOptions {
        hessian_metric_provider: Some(std::sync::Arc::new(EchoHessianMetricProvider)),
        ..meshlang_compiler::CompileOptions::default()
    };

    let geo = meshlang_compiler::compile_to_geo_full_with_options(
        source,
        "hessian-provider.mlg",
        &options,
    )
    .expect("GEO accepts a configured Hessian provider");
    let python = meshlang_compiler::compile_to_python_full_with_options(
        source,
        "hessian-provider.mlg",
        &options,
    )
    .expect("Python accepts a configured Hessian provider");

    assert!(geo.code.contains("MathEvalAniso"));
    assert!(python.code.contains("MathEvalAniso"));
    for component in ["M11", "M22", "M33", "M12", "M13", "M23"] {
        assert!(geo.code.contains(component), "GEO omitted {component}");
        assert!(
            python.code.contains(component),
            "Python omitted {component}"
        );
    }
    assert_eq!(geo.canonical_ir_json, python.canonical_ir_json);
}

#[derive(Debug)]
struct RejectingHessianMetricProvider;

impl meshlang_compiler::HessianMetricProvider for RejectingHessianMetricProvider {
    fn resolve(
        &self,
        _request: meshlang_compiler::HessianMetricRequest<'_>,
    ) -> Result<[meshlang_ast::Expr; 6], String> {
        Err("field samples are unavailable".into())
    }
}

#[test]
fn hessian_provider_failures_are_typed_and_backend_independent() {
    let source = r#"
geometry model {
    let body = Box(1.0, 1.0, 1.0)
}

mesh for model {
    metric model.body {
        kind: hessian
        field: 1.0
        minimum_size: 0.01
        maximum_size: 0.1
    }
}
"#;
    let options = meshlang_compiler::CompileOptions {
        hessian_metric_provider: Some(std::sync::Arc::new(RejectingHessianMetricProvider)),
        ..meshlang_compiler::CompileOptions::default()
    };

    for error in [
        meshlang_compiler::compile_to_geo_full_with_options(
            source,
            "hessian-provider-error.mlg",
            &options,
        )
        .expect_err("GEO must propagate provider failure"),
        meshlang_compiler::compile_to_python_full_with_options(
            source,
            "hessian-provider-error.mlg",
            &options,
        )
        .expect_err("Python must propagate provider failure"),
    ] {
        assert!(matches!(
            error,
            meshlang_compiler::CompileError::TypeError { ref message }
                if message.contains("Hessian metric provider failed")
                    && message.contains("field samples are unavailable")
        ));
    }
}

#[test]
fn test_runtime_loop_aggregates_each_entity_dimension() {
    let source = r#"
param count = 2
geometry model {
    for i in 0..count {
        let point_a = Point(i, 0.0, 0.0)
        let point_b = Point(i + 1.0, 0.0, 0.0)
        let edge = Line(from: point_a, to: point_b)
        let face = Rectangle(1.0, 1.0)
        let body = Box(1.0, 1.0, 1.0)
    }
}
physics {
    group bodies: volume = model.body
}
"#;

    let geo = meshlang_compiler::compile_to_geo(source, "runtime-loop-dimensions.mlg")
        .expect("GEO preserves all runtime entity dimensions");
    for list in [
        "__ml_loop_1_0",
        "__ml_loop_1_1",
        "__ml_loop_1_2",
        "__ml_loop_1_3",
    ] {
        assert!(
            geo.contains(&format!("{list}[] = {{}};")),
            "missing GEO list {list}"
        );
    }
    assert!(geo.contains("Physical Volume(\"bodies\")"));

    let python = meshlang_compiler::compile_to_python(source, "runtime-loop-dimensions.mlg")
        .expect("Python preserves all runtime entity dimensions");
    for list in [
        "__ml_loop_1_0",
        "__ml_loop_1_1",
        "__ml_loop_1_2",
        "__ml_loop_1_3",
    ] {
        assert!(
            python.contains(&format!("{list} = []")),
            "missing Python list {list}"
        );
    }
    assert!(python.contains("pg_bodies"));
}
